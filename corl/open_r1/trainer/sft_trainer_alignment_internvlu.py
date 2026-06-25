# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# InternVL-U T2I flow-matching trainer.
#
# Continuous analog of the Janus AR-CE alignment trainer
# (sft_trainer_alignment.py): a caption is encoded by the (LoRA) LLM into the
# T2I conditioning hidden states h_text; the FROZEN generation_decoder (DiT)
# is asked to denoise a noised VAE latent of the GT image conditioned on
# h_text, and we take the rectified-flow velocity MSE. Gradients flow back
# through the frozen DiT to the LoRA adapters on the LLM.
#
# Why flow-matching and not a one-shot regression: an empirical probe showed
# the frozen DiT cannot reconstruct an image from the LLM hidden states alone
# (image content reaches the DiT only via the VAE-latent image-stream tokens),
# and a one-shot MSE head bypassing the DiT produces blurry latents. The only
# path that yields a usable text-only generator is to keep the DiT in the loop
# and train h_text to drive it -- i.e. the native T2I objective.

import os
import random
import sys
import time
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
import transformers
from packaging import version
from datasets import Dataset, IterableDataset
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.utils import is_peft_available

from trl import ScriptArguments, SFTConfig

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# internvlu (the sibling InternVL-U repo) must be importable. Allow an env
# override; otherwise fall back to the known working-copy path.
try:
    from internvlu import InternVLUPipeline
except ModuleNotFoundError:
    _repo = os.environ.get("INTERNVLU_REPO", "/work/um00109/MLLM/InternVL-U")
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    from internvlu import InternVLUPipeline


# Target image -> square gen-resolution pixels, matching InternVL-U's
# InternVLUFixResGenerationImageProcessor normalization (Normalize([.5],[.5])).
def _build_gen_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_state_mask(input_ids, attention_mask, generation_flags,
                     im_start_token_id, img_start_token_id):
    """Boolean mask [K, B, N] selecting conditioning hidden states from the 2nd
    <|im_start|> (user turn) up to each generated <img> token.

    Copied verbatim from InternVLUPipeline._prepare_hidden_state_mask (pad
    branch) so the trainer is self-contained and matches inference exactly.
    """
    B, N = input_ids.shape
    img_start_positions = (input_ids == img_start_token_id).nonzero()
    gen_img_start_positions = img_start_positions[generation_flags.bool()]  # [K,2]
    state_positions_row = torch.arange(B, device=input_ids.device)[None]    # [1,B]
    state_positions_col = torch.arange(N, device=input_ids.device)[None]    # [1,N]
    state_mask_row = state_positions_row == gen_img_start_positions[:, :1]   # [K,B]
    state_mask_col = state_positions_col <= gen_img_start_positions[:, 1:]   # [K,N]

    im_start_positions = (input_ids == im_start_token_id).nonzero()
    im_state_mask_row = state_positions_row == im_start_positions[:, :1]
    im_start_second_idxs = (im_state_mask_row.cumsum(dim=0) == 2).nonzero(as_tuple=True)[0]
    im_start_second_positions = im_start_positions[im_start_second_idxs]     # [K',2]

    gen_img_start_positions_global = (
        gen_img_start_positions[:, 0] * N + gen_img_start_positions[:, 1]
    )
    bos_positions_global = gen_img_start_positions[:, 0] * N
    im_start_second_positions_global = (
        im_start_second_positions[:, 0] * N + im_start_second_positions[:, 1]
    )
    im_start_second_positions_mask = (
        im_start_second_positions_global[None, :] <= gen_img_start_positions_global[:, None]
    ) & (
        im_start_second_positions_global[None, :] >= bos_positions_global[:, None]
    )
    im_start_second_positions_idxs = im_start_second_positions_mask.int().argmax(dim=1)
    im_start_second_positions_to_gen_img_start = im_start_second_positions[
        im_start_second_positions_idxs
    ]  # [K,2]

    state_mask_col = state_mask_col & (
        im_start_second_positions_to_gen_img_start[:, 1:] <= state_positions_col
    )
    state_mask = (state_mask_row[..., None] & state_mask_col[:, None]).bool()  # [K,B,N]
    state_mask = (state_mask & attention_mask[None]).bool()
    return state_mask


class InternVLUT2IFlowMatch(nn.Module):
    """DDP-safe forward module: caption hidden states -> frozen DiT -> velocity
    MSE. This is the module that gets LoRA-wrapped and optimized; its forward IS
    the train step (mirrors how the Janus model.forward returns the loss inputs).
    """

    def __init__(self, vlm, generation_decoder, vlm_select_layer, flow_shift,
                 logit_mean, logit_std, num_train_timesteps=1000):
        super().__init__()
        self.vlm = vlm
        self.generation_decoder = generation_decoder
        self.vlm_select_layer = list(vlm_select_layer)
        self.flow_shift = float(flow_shift)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.num_train_timesteps = int(num_train_timesteps)
        # Special-token ids are set on the vlm by InternVLUPipeline._init_special_tokens.
        self.im_start_token_id = vlm.im_start_token_id
        self.img_start_token_id = vlm.img_start_token_id
        self.img_context_token_id = vlm.img_context_token_id

    # HF Trainer calls these on the (PEFT-wrapped) model when
    # args.gradient_checkpointing is set; route them to the LLM + frozen DiT.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.vlm.language_model, "gradient_checkpointing_enable"):
            self.vlm.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            # With a frozen base + LoRA, checkpointed layers need their input
            # embeddings to require grad or backward detaches. We call the LLM's
            # input embeddings directly in forward(), so hook it here.
            self.vlm.language_model.enable_input_require_grads()
        # diffusers API: also installs self._gradient_checkpointing_func on the DiT.
        if hasattr(self.generation_decoder.decoder, "enable_gradient_checkpointing"):
            self.generation_decoder.decoder.enable_gradient_checkpointing()

    def gradient_checkpointing_disable(self):
        if hasattr(self.vlm.language_model, "gradient_checkpointing_disable"):
            self.vlm.language_model.gradient_checkpointing_disable()
        if hasattr(self.generation_decoder.decoder, "disable_gradient_checkpointing"):
            self.generation_decoder.decoder.disable_gradient_checkpointing()

    def encode_conditioning(self, input_ids, attention_mask, generation_flags):
        """Caption -> projected/padded DiT conditioning (h_text path), WITH grad."""
        vlm = self.vlm
        # .clone() so the in-place special-token replacement is allowed even when
        # enable_input_require_grads has marked the embedding output (matches
        # InternVLUChatModel.forward).
        emb = vlm.language_model.get_input_embeddings()(input_ids).clone()
        emb = vlm.replace_img_special_tokens(emb, input_ids)
        out = vlm.language_model(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            padding_type="pad",
        )
        hs = out.hidden_states
        B, L = input_ids.shape
        vlm_hidden = torch.cat([hs[i].view(B, L, -1) for i in self.vlm_select_layer], dim=-1)

        state_mask = build_state_mask(
            input_ids, attention_mask, generation_flags,
            self.im_start_token_id, self.img_start_token_id,
        )
        enc_list = [vlm_hidden[s] for s in state_mask]
        selected = (input_ids == self.img_context_token_id)
        imgtok_list = [selected[s] for s in state_mask]
        enc_hidden, enc_attn_mask, enc_imgtok_mask = (
            self.generation_decoder.prepare_forward_input(
                enc_list, encoder_image_token_mask=imgtok_list
            )
        )
        return enc_hidden, enc_attn_mask, enc_imgtok_mask

    def sample_noise(self, z0):
        """Rectified-flow noising with logit-normal timestep + flow_shift."""
        bsz = z0.shape[0]
        u = torch.randn(bsz, device=z0.device) * self.logit_std + self.logit_mean
        sigma = torch.sigmoid(u)
        s = self.flow_shift
        sigma = s * sigma / (1.0 + (s - 1.0) * sigma)
        sig = sigma.view(-1, 1, 1, 1).to(z0.dtype)
        noise = torch.randn_like(z0)
        z_t = (1.0 - sig) * z0 + sig * noise
        target = noise - z0
        timestep = sigma * self.num_train_timesteps
        return z_t, target, timestep, sigma

    def forward(self, input_ids, attention_mask, generation_flags,
                image_grid_thw_gen, z0):
        enc_hidden, enc_attn_mask, enc_imgtok_mask = self.encode_conditioning(
            input_ids, attention_mask, generation_flags
        )
        z_t, target, timestep, sigma = self.sample_noise(z0)

        bsz = z0.shape[0]
        image_fhw_cond = [
            torch.zeros([0, 3], device=z0.device, dtype=torch.long) for _ in range(bsz)
        ]
        grid_gen = image_grid_thw_gen[generation_flags.bool()]
        v_pred = self.generation_decoder.decoder(
            hidden_states=z_t.to(enc_hidden.dtype),
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_attn_mask,
            encoder_image_token_mask=enc_imgtok_mask,
            image_fhw_cond=image_fhw_cond,
            timestep=timestep.to(enc_hidden.dtype),
            image_grid_thw_gen=grid_gen,
            conditional_input=None,
            image_grid_thw_gen_cond=None,
            return_dict=False,
        )[0]

        loss = F.mse_loss(v_pred.float(), target.float())
        return {"loss": loss, "v_pred": v_pred, "target": target, "sigma": sigma}


class SFTInternVLUAlignmentTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: SFTConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset=None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        attn_implementation: str = "sdpa",
        peft_config: Optional["PeftConfig"] = None,
        task_args: ScriptArguments = None,
    ):
        self.task_args = task_args
        if not isinstance(model, str):
            raise ValueError("SFTInternVLUAlignmentTrainer expects a checkpoint path string.")
        model_path = model

        torch_dtype = torch.bfloat16
        if args is not None and args.model_init_kwargs:
            td = args.model_init_kwargs.get("torch_dtype")
            if isinstance(td, str):
                torch_dtype = getattr(torch, td)
            elif isinstance(td, torch.dtype):
                torch_dtype = td

        # ******************* Load the full InternVL-U pipeline *******************
        print(f"[InternVLU] loading pipeline from {model_path}")
        pipe = InternVLUPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.pipe = pipe
        self.processor = pipe.processor
        self.image_pipeline = pipe.image_pipeline   # holds vae + pixels_to_latents
        self.vae = pipe.vae
        vlm = pipe.vlm
        gd = pipe.generation_decoder

        # ******************* Freeze everything except (later) LoRA *******************
        self.init_trainable_parameters(vlm, gd, pipe.vae)

        # Optionally re-enable the DiT's h_text->cond bridge (analogous to Janus
        # gen_aligner); PEFT keeps it via modules_to_save=["decoder_projector"].
        if getattr(task_args, "train_decoder_projector", False):
            for p in gd.decoder_projector.parameters():
                p.requires_grad = True
            print("[InternVLU] decoder_projector is trainable")

        self.gen_image_size = int(getattr(task_args, "gen_image_size", 512))
        self._gen_tf = _build_gen_transform(self.gen_image_size)

        flow_model = InternVLUT2IFlowMatch(
            vlm=vlm,
            generation_decoder=gd,
            vlm_select_layer=gd.config.vlm_select_layer,
            flow_shift=gd.config.flow_shift,
            logit_mean=gd.config.logit_mean,
            logit_std=gd.config.logit_std,
        )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`")
            flow_model = get_peft_model(flow_model, peft_config)
            if getattr(args, "gradient_checkpointing", False) and hasattr(
                flow_model, "enable_input_require_grads"
            ):
                flow_model.enable_input_require_grads()

        # GC (LLM + frozen DiT) is enabled by the Trainer via the wrapper's
        # gradient_checkpointing_enable() when args.gradient_checkpointing is set.

        n_train = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in flow_model.parameters())
        print(f"[InternVLU] trainable params: {n_train:,} / {n_total:,} "
              f"({100.0 * n_train / max(n_total, 1):.3f}%)")

        if processing_class is None:
            processing_class = pipe.processor.tokenizer

        def data_collator(features):
            return features

        self.max_prompt_length = task_args.max_prompt_length
        self.prompt_dropout_prob = float(getattr(task_args, "prompt_dropout_prob", 0.0))
        self.caption_source = getattr(task_args, "caption_source", "original")
        self.caption_column = getattr(task_args, "caption_column", "detailed_caption")
        self.i2t_question = getattr(
            task_args, "i2t_question",
            "What type of medical image is this? Provide enough detail to reconstruct the image faithfully.",
        )
        self.i2t_max_new_tokens = int(getattr(task_args, "i2t_max_new_tokens", 96))
        self._devices_ready = False

        super().__init__(
            model=flow_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self._metrics = defaultdict(list)
        self.model_accepts_loss_kwargs = False

    # ---------------------------------------------------------------- params
    @staticmethod
    def init_trainable_parameters(vlm, generation_decoder, vae):
        for p in vlm.vision_model.parameters():
            p.requires_grad = False
        for p in vlm.mlp1.parameters():
            p.requires_grad = False
        for p in vlm.language_model.parameters():
            p.requires_grad = False
        if hasattr(vlm, "special_token_embedding"):
            for p in vlm.special_token_embedding.parameters():
                p.requires_grad = False
        for p in generation_decoder.parameters():
            p.requires_grad = False
        for p in vae.parameters():
            p.requires_grad = False
        return vlm

    def _ensure_devices(self):
        """Move the frozen side modules (vae, DiT) onto the accelerator device.
        Done lazily because the device is only known after accelerator init."""
        if self._devices_ready:
            return
        dev = self.accelerator.device
        self.vae.to(dev)
        self.image_pipeline.to(dev) if hasattr(self.image_pipeline, "to") else None
        self.generation_decoder_device = dev
        self._devices_ready = True

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # ---------------------------------------------------------------- captions
    @torch.inference_mode()
    def _self_distill_caption(self, image):
        out = self.pipe(
            prompt=self.i2t_question,
            image=image,
            generation_mode="text",
            max_new_tokens=self.i2t_max_new_tokens,
        )
        return self.processor.tokenizer.decode(
            out.generate_output[0], skip_special_tokens=True
        ).strip()

    def _get_caption(self, row, image):
        if self.caption_source == "self_distill":
            return self._self_distill_caption(image)
        cap = row.get(self.caption_column)
        if cap is None:
            raise KeyError(
                f"caption_source='original' but column '{self.caption_column}' "
                f"missing. Row keys: {list(row.keys())}"
            )
        if isinstance(cap, (list, tuple)):
            cap = random.choice(cap)
        return cap

    # ---------------------------------------------------------------- inputs
    def _prepare_inputs(self, inputs):
        self._ensure_devices()
        device = self.accelerator.device

        images = [Image.open(x["image"]).convert("RGB") for x in inputs]
        captions = [self._get_caption(x, img) for x, img in zip(inputs, images)]
        B = len(captions)

        enc = self.processor(
            prompt=captions, image=None, generation_mode="image",
            height=self.gen_image_size, width=self.gen_image_size,
            padding=True, return_tensors="pt",
        )
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        gflags = enc["generation_flags"]
        grid = enc["image_grid_thw_gen"]

        # Processor lays rows out as [none*B, text*B, all*B] (full-cond, drop-text,
        # drop-all). Take the full-cond block; for CFG, randomly swap a sample to
        # its unconditional (drop-all) row so the model learns the uncond path.
        cond_idx = torch.arange(B, device=device)
        src = cond_idx
        if self.model.training and self.prompt_dropout_prob > 0:
            drop = torch.rand(B, device=device) < self.prompt_dropout_prob
            src = torch.where(drop, cond_idx + 2 * B, cond_idx)
            self._metrics["cfg_dropout_frac"].append(float(drop.float().mean()))
        input_ids = input_ids[src]
        attn = attn[src]
        gflags = gflags[src]
        grid = grid[src]

        if self.max_prompt_length is not None and input_ids.shape[1] > self.max_prompt_length:
            input_ids = input_ids[:, -self.max_prompt_length:]
            attn = attn[:, -self.max_prompt_length:]

        # GT image -> frozen VAE latent target (no grad).
        pv = torch.stack([self._gen_tf(img) for img in images]).to(device, self.vae.dtype)
        with torch.no_grad():
            z0 = self.image_pipeline.pixels_to_latents(pv)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "generation_flags": gflags,
            "image_grid_thw_gen": grid,
            "z0": z0,
        }

    # ---------------------------------------------------------------- loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(**inputs)
        loss = out["loss"]
        with torch.no_grad():
            self._metrics["loss_flow"].append(float(loss.item()))
            self._metrics["sigma_mean"].append(float(out["sigma"].mean().item()))
        return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        step_start = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._metrics["train_step_time_s"].append(time.perf_counter() - step_start)
        # Zero NaN/Inf grads so a bad batch can't poison gradient clipping.
        nan_params = []
        for n, p in model.named_parameters():
            if not (p.requires_grad and p.grad is not None):
                continue
            if not torch.isfinite(p.grad).all():
                nan_params.append(n)
                p.grad.detach().zero_()
        if nan_params:
            self._metrics["nan_grad_zeroed"].append(float(len(nan_params)))
            if not getattr(self, "_first_nan_logged", False):
                self._first_nan_logged = True
                print(f"[grad-NaN] step={self.state.global_step} zeroed {len(nan_params)} "
                      f"param grads. First few: {nan_params[:5]}")
        return loss

    def log(self, logs, start_time=None):
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items() if v}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
