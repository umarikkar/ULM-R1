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

# T2I-only AR-CE variant: student is teacher-forced on its OWN AR rollout
# (in inference_mode, no grad), then cross-entropy is taken against GT VQ ids.

import os
import random
import re
import time
from collections import defaultdict
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from packaging import version
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.utils import is_peft_available
from accelerate.utils import set_seed
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as T

from trl.import_utils import is_deepspeed_available
from trl import ScriptArguments, SFTConfig

from transformers.models.llama.modeling_llama import LlamaForCausalLM

if is_deepspeed_available():
    import deepspeed  # noqa: F401

if is_peft_available():
    from peft import PeftConfig, get_peft_model

from janus.models import VLChatProcessor

VQ_TRANSFORM = T.Compose([
    T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

N_IMAGE_TOKENS = 576


def fix_janus_text(out_caption):
    out_caption = out_caption.replace("Ġ", " ")
    out_caption = out_caption.replace("Ċ", "\n")

    # Optional cleanup for extra spacing/newlines
    out_caption = re.sub(r"[ \t]+", " ", out_caption)
    out_caption = re.sub(r"\n\s*\n+", "\n", out_caption)

    return out_caption.strip()


class T2IEvalCallback(TrainerCallback):
    """Runs t2i generation on cached prompts every `freq` steps and saves PNGs."""

    def __init__(self, trainer, freq: int):
        self.trainer = trainer
        self.freq = freq

    def on_step_end(self, args, state, control, **kwargs):
        if self.freq <= 0 or state.global_step == 0:
            return
        if state.global_step % self.freq != 0:
            return
        self.trainer._run_t2i_eval(state.global_step)


class TextToProtoSaveCallback(TrainerCallback):
    """Persist text_to_proto weights alongside each adapter checkpoint.

    The head is attached AFTER PEFT wrapping (not in modules_to_save) to avoid
    a DDP "marked ready twice" error. PEFT's save_pretrained therefore skips
    it. We save it here so checkpoints are complete enough to reload for eval.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if not self.trainer.accelerator.is_main_process:
            return
        unwrapped = self.trainer.accelerator.unwrap_model(self.trainer.model_wrapped)
        inner = unwrapped.base_model.model if hasattr(unwrapped, "base_model") else unwrapped
        if not hasattr(inner, "text_to_proto"):
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        from safetensors.torch import save_file
        sd = {k: v.detach().cpu().contiguous()
              for k, v in inner.text_to_proto.state_dict().items()}
        save_file(sd, str(ckpt_dir / "text_to_proto.safetensors"))
        print(f"[save] text_to_proto -> {ckpt_dir / 'text_to_proto.safetensors'}")


class SFTAlignmentTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            args: SFTConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset=None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[
                Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
            ] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[
                Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
            ] = (None, None),
            attn_implementation: str = "sdpa",
            peft_config: Optional["PeftConfig"] = None,
            task_args: ScriptArguments = None,
    ):
        self.task_args = task_args

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")

        # ******************* Model *******************
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto"
                    or torch_dtype is None):
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a "
                    f"string representing a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Janus" in model_id:
                model = AutoModelForCausalLM.from_pretrained(
                    model, trust_remote_code=True, torch_dtype=torch.bfloat16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )

        self.cond_dropout_prob = float(getattr(task_args, "cond_dropout_prob", 0.0))

        self.use_prototype_conditioning = bool(getattr(task_args, "use_prototype_conditioning", False))
        self.cond_temperature = float(getattr(task_args, "cond_temperature", 0.1))
        self.prototype_centroids_path = getattr(task_args, "prototype_centroids_path", "") or ""
        self._prototype_centroids = None
        self._prototype_K = 0
        if self.use_prototype_conditioning:
            import torch.nn as nn
            d_model = model.language_model.config.hidden_size
            base_dtype = next(model.language_model.parameters()).dtype
            if not os.path.exists(self.prototype_centroids_path):
                raise FileNotFoundError(
                    f"prototype_centroids_path='{self.prototype_centroids_path}' "
                    f"-- run corl/scripts/build_prototype_centroids.sh first"
                )
            cdata = torch.load(self.prototype_centroids_path, map_location="cpu")
            self._prototype_centroids = cdata["centroids"]    # [K, d_feat], unit-norm
            self._prototype_K = int(cdata["K"])
            d_feat = int(cdata.get("d_feat", self._prototype_centroids.shape[-1]))
            proto_emb = nn.Embedding(self._prototype_K + 1, d_model, dtype=base_dtype)
            nn.init.normal_(proto_emb.weight, mean=0.0, std=0.005)
            with torch.no_grad():
                proto_emb.weight[0].zero_()
            model.prototype_emb = proto_emb
            print(f"[proto-cond] attached prototype_emb(K+1={self._prototype_K + 1}, d={d_model}) "
                  f"centroids({self._prototype_K}, {d_feat}) from {self.prototype_centroids_path} "
                  f"cond_dropout_prob={self.cond_dropout_prob} τ={self.cond_temperature}")

        # Text -> prototype head, trained via auxiliary KL against w_image.
        # Used at inference to build cond from the caption alone (no image).
        # Attached AFTER PEFT wrapping (below) to avoid the ModulesToSaveWrapper
        # vs DDP "marked ready twice" error.
        self.use_text_to_proto = bool(getattr(task_args, "use_text_to_proto", False))
        self.text_to_proto_aux_weight = float(getattr(task_args, "text_to_proto_aux_weight", 1.0))
        if self.use_text_to_proto and not self.use_prototype_conditioning:
            raise ValueError("use_text_to_proto=True requires use_prototype_conditioning=True")

        model = self.init_trainable_parameters(model)

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`")
            model = get_peft_model(model, peft_config)
            if getattr(args, "gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        if self.use_text_to_proto:
            import torch.nn as nn
            # Attach to the INNER Janus model (peft_model.base_model.model) so
            # the Janus forward's `self.text_to_proto` lookup finds it. DDP
            # still tracks the params via the full module tree.
            inner = model.base_model.model if hasattr(model, "base_model") else model
            d_model = inner.language_model.config.hidden_size
            base_dtype = next(inner.language_model.parameters()).dtype
            head = nn.Sequential(
                nn.Linear(d_model, d_model, dtype=base_dtype),
                nn.GELU(),
                nn.Linear(d_model, self._prototype_K, dtype=base_dtype),
            )
            with torch.no_grad():
                head[-1].weight.zero_()
                head[-1].bias.zero_()
            for p in head.parameters():
                p.requires_grad = True
            inner.text_to_proto = head
            print(f"[text2proto] attached MLP(d={d_model} -> K={self._prototype_K}), "
                  f"aux_w={self.text_to_proto_aux_weight}")

        warm_start = getattr(task_args, "warm_start_checkpoint", "") or ""
        if warm_start:
            self._load_warm_start_adapter(model, warm_start)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[SFTAlignmentTrainer] trainable params: {n_trainable:,} / {n_total:,} "
              f"({100.0 * n_trainable / max(n_total, 1):.3f}%)")

        # Processing class
        if processing_class is None:
            if "Janus" in model_id:
                processing_class = VLChatProcessor.from_pretrained(model_id)
                processing_class.system_prompt = ""
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )

        def data_collator(features):
            return features

        # ******************* Training arguments *******************

        print(task_args)

        self.max_prompt_length = task_args.max_prompt_length
        self.temperature = task_args.temperature
        self.prompt_dropout_prob = getattr(task_args, "prompt_dropout_prob", 0.0)
        self.eval_image_freq = getattr(task_args, "eval_image_freq", 100)
        self.eval_image_num = getattr(task_args, "eval_image_num", 4)
        self.eval_image_cfg = getattr(task_args, "eval_image_cfg", 5.0)
        self.eval_image_temp = getattr(task_args, "eval_image_temp", 1.0)
        self.eval_image_subdir = getattr(task_args, "eval_image_subdir", "eval_samples")
        self.caption_source = getattr(task_args, "caption_source", "self_distill")
        self.caption_column = getattr(task_args, "caption_column", "Original_Caption")
        self.use_perceptual_loss = getattr(task_args, "use_perceptual_loss", False)
        self.perceptual_weight = getattr(task_args, "perceptual_weight", 0.1)
        self.perceptual_model_id = getattr(
            task_args, "perceptual_model_id",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        self.perceptual_warmup_steps = getattr(task_args, "perceptual_warmup_steps", 500)
        self.perceptual_layers = getattr(task_args, "perceptual_layers", "")
        self._perceptual_model = None
        self._perceptual_blocks = None
        self._perceptual_tap_idx = None
        self._perceptual_buf = {}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self._metrics = defaultdict(list)
        self._eval_prompts_cached = None
        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False

        # Pixel-space LPIPS reconstruction loss (frozen VGG). Lazy-loaded on
        # first compute_loss so the metric lives on the correct device.
        self.use_reconstruction_loss = bool(getattr(task_args, "use_reconstruction_loss", False))
        self.lpips_weight = float(getattr(task_args, "lpips_weight", 1.0))
        self._lpips_metric = None

        if self.eval_image_freq > 0:
            self.add_callback(T2IEvalCallback(self, freq=self.eval_image_freq))
        if self.use_text_to_proto:
            self.add_callback(TextToProtoSaveCallback(self))


    @staticmethod
    def init_trainable_parameters(model):
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.aligner.parameters():
            param.requires_grad = False
        for param in model.gen_vision_model.parameters():
            param.requires_grad = False
        for param in model.gen_embed.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = False

        for param in model.gen_head.parameters():
            param.requires_grad = True
        for param in model.gen_aligner.parameters():
            param.requires_grad = True

        if hasattr(model, "prototype_emb"):
            for param in model.prototype_emb.parameters():
                param.requires_grad = True
        if hasattr(model, "text_to_proto"):
            for param in model.text_to_proto.parameters():
                param.requires_grad = True

        return model

    @staticmethod
    def _load_warm_start_adapter(model, ckpt_dir):
        """Overlay LoRA + modules_to_save weights from a prior Stage-2 ckpt.
        Remaps PEFT's flat keys to the wrapped runtime namespace and uses
        load_state_dict(strict=False) so newly-added modules stay at fresh init.
        """
        import json
        from pathlib import Path
        p = Path(ckpt_dir)
        sf = p / "adapter_model.safetensors"
        bn = p / "adapter_model.bin"
        if sf.exists():
            from safetensors.torch import load_file
            state = load_file(str(sf))
            src = sf
        elif bn.exists():
            state = torch.load(str(bn), map_location="cpu")
            src = bn
        else:
            raise FileNotFoundError(
                f"warm_start_checkpoint='{ckpt_dir}' has no adapter_model.safetensors/.bin"
            )
        # Read the checkpoint's modules_to_save list to know which paths to
        # nest under .modules_to_save.default.
        adapter_cfg_path = p / "adapter_config.json"
        if adapter_cfg_path.exists():
            saved_mts = json.load(open(adapter_cfg_path)).get("modules_to_save") or []
        else:
            saved_mts = []
        LORA_TAGS = ("lora_A.", "lora_B.", "lora_embedding_A.", "lora_embedding_B.",
                     "lora_magnitude_vector.")
        remapped = {}
        for k, v in state.items():
            new_k = k
            for tag in LORA_TAGS:
                if tag in new_k:
                    new_k = new_k.replace(tag, tag + "default.", 1)
                    break
            else:
                for name in saved_mts:
                    pattern = f".{name}."
                    idx = new_k.find(pattern)
                    if idx != -1:
                        after = idx + len(pattern)
                        new_k = new_k[:after] + "modules_to_save.default." + new_k[after:]
                        break
            remapped[new_k] = v
        result = model.load_state_dict(remapped, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        loaded = len(remapped) - len(unexpected)
        print(f"[warm-start] {src.name}: loaded {loaded}/{len(remapped)} keys"
              f" (modules_to_save in ckpt: {saved_mts})")
        if unexpected:
            print(f"  ignored (unexpected in ckpt): {len(unexpected)}  e.g. {unexpected[:3]}")
        trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        fresh_trainable = trainable_keys - set(remapped.keys())
        print(f"  fresh-init trainable keys: {len(fresh_trainable)} / {len(trainable_keys)}")
        if fresh_trainable:
            for ex in sorted(fresh_trainable)[:5]:
                print(f"    {ex}")
        if loaded == 0:
            raise RuntimeError(
                f"warm-start loaded 0 keys from {src} -- check the PEFT key namespace"
            )
        bad_params = []
        for n, p in model.named_parameters():
            if p.requires_grad and not torch.isfinite(p).all():
                bad_params.append((n, p.shape))
        if bad_params:
            print(f"  [warm-start][WARN] {len(bad_params)} trainable params have NaN/Inf!")
            for n, s in bad_params[:5]:
                print(f"    - {n} {tuple(s)}")
        for group in ("gen_head", "gen_aligner", "prototype_emb", "text_to_proto"):
            ps = [p for n, p in model.named_parameters() if group in n and p.requires_grad]
            if ps:
                ns = [p.detach().float().norm().item() for p in ps]
                print(f"  [warm-start] {group}: n_params={len(ps)} "
                      f"norm_mean={sum(ns)/len(ns):.3f} norm_max={max(ns):.3f}")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def load_batch_images(self, inputs):
        loaded_images = []
        for x in inputs:
            image_path = x.get("image")
            if image_path is None:
                raise KeyError("Each sample must contain 'image' or 'image_full_path'.")
            loaded_images.append(Image.open(image_path).convert("RGB"))
        return loaded_images

    @torch.inference_mode()
    def wrap_t2i_prompt(self, captions, device=None):
        prompts = []
        for cap in captions:

            conv = [
                {
                    "role": "<|User|>",
                    "content": f"{cap}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            sft_format = self.processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=conv,
                sft_format=self.processing_class.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.processing_class.image_start_tag
            prompts.append(prompt)

        prompt_inputs = self.processing_class.tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        return prompt_inputs, prompts
    
    @torch.inference_mode()
    def get_image_gen_reps(self, model, device=None, images=None):

        pixel_values = torch.stack([VQ_TRANSFORM(img) for img in images]).to(
            device=device, dtype=torch.bfloat16
        )
        quant, _, info = model.gen_vision_model.encode(pixel_values)

        spatial_shape = quant.shape[2:]                          # (H_vq, W_vq)
        X_image = quant.flatten(2).transpose(1, 2).contiguous() # [B, N, D_vq]
        gt_ids = info[-1].reshape(X_image.shape[0], -1)

        return X_image, gt_ids, spatial_shape

    @torch.inference_mode()
    def get_i2t_t2i_inputs(self, device=None, images=None):

        # task_instruct = "Describe the main content of the image in one sentence."
        # task_instruct = "Describe this medical image in one to two sentences. Include the imaging modality, \
        #                     dominant colours and contrast, spatial arrangement of key structures, and any salient \
        #                     visual features like lesions or abnormalities. Be specific enough to reconstruct the \
        #                     image from the description alone."

        # task_instruct = "Describe this medical image in one to two sentences, focusing exclusively on visually \
        #                         observable features. Include: (1) the imaging modality and orientation if identifiable \
        #                         (e.g. axial CT, H&E histology, MRI or other), (2) the dominant colours, tones, and contrast \
        #                         patterns, (3) the spatial arrangement and shape of visible structures, (4) texture and \
        #                         surface appearance, and (5) any salient or abnormal visual features such as lesions, \
        #                         colour hotspots, or irregular morphology. Avoid diagnostic conclusions — describe only \
        #                         what is directly visible to reconstruct the \
        #                         image from the description alone."

        task_instruct = "Describe this medical image in one to two sentences. Describe only \
                                what is directly visible to reconstruct the \
                                image from the description alone."

        _prompts, _images = [], []

        for img in images:
            _prompts.append(
                [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{task_instruct}",
                        # "images": [example["image"]],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ],
            )
            _images.append([img])

        prepare_inputs = self.processing_class(
            conversations=_prompts, images=_images, force_batchify=True,
        ).to(device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=1,
            pad_token_id=self.processing_class.tokenizer.eos_token_id,
            bos_token_id=self.processing_class.tokenizer.bos_token_id,
            eos_token_id=self.processing_class.tokenizer.eos_token_id,
        )
        gen_captions = self.processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        gen_captions = [fix_janus_text(cap) for cap in gen_captions]

        "this is the t2i input token IDs, which are used to autoregressively generate image tokens. This will be the student."
        t2i_inputs, _ = self.wrap_t2i_prompt(gen_captions, device=device)

        return t2i_inputs
    

    def get_teacher_image_logits(self, model, gt_ids, device=None):
        boi_id = self.processing_class.image_start_id
        boi_ids = torch.full((gt_ids.shape[0], 1), boi_id, device=device, dtype=torch.long)
        boi_mask = torch.ones((gt_ids.shape[0], 1), device=device, dtype=torch.long)

        teacher_out = model(
            t2i_input_ids=boi_ids,
            t2i_attention_mask=boi_mask,
            t2i_discrete_img_ids=gt_ids,
            t2i_logits_to_keep=gt_ids.shape[1],
            task="generation",
        )
        teacher_logits = teacher_out.logits  # [B, N, V_image]
        return teacher_logits


    @torch.inference_mode()
    def ar_rollout_student_ids(self, model, t2i_input_ids, t2i_attention_mask):
        """Autoregressively sample N image tokens from text+BOI using current weights.

        Mirrors generate_image() in evaluate_checkpoints.py (no CFG — the SFT has
        no unconditional branch). Returned ids are used as the *student's own*
        teacher-forcing context for the grad-enabled forward.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        device = t2i_input_ids.device
        B = t2i_input_ids.shape[0]
        N = N_IMAGE_TOKENS

        inputs_embeds = unwrapped.language_model.get_input_embeddings()(t2i_input_ids)
        attn = t2i_attention_mask

        generated_ids = torch.zeros((B, N), dtype=torch.long, device=device)
        outputs = None
        for i in range(N):
            outputs = unwrapped.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None,
            )
            hidden = outputs.last_hidden_state[:, -1, :]
            logits = unwrapped.gen_head(hidden)                                # [B, V_img]
            probs = F.softmax(logits.float() / max(self.temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)   # [B]
            generated_ids[:, i] = next_token

            inputs_embeds = unwrapped.prepare_gen_img_embeds(next_token).unsqueeze(1)
            attn = torch.cat(
                [attn, torch.ones(B, 1, dtype=attn.dtype, device=device)], dim=1
            )

        return generated_ids

    def _cache_eval_prompts(self):
        """Take the first `eval_image_num` training samples and cache prompt
        tokens + image path, so eval renders the same images at every step.
        Honours self.caption_source so eval matches the training condition."""
        device = self.accelerator.device
        k = min(self.eval_image_num, len(self.train_dataset))
        rows = [self.train_dataset[i] for i in range(k)]
        imgs = [Image.open(r["image"]).convert("RGB") for r in rows]
        if self.caption_source == "original":
            # For eval, always take captions[0] when the column is a list — keeps
            # the per-step samples comparable across training steps.
            captions = []
            for r in rows:
                c = r[self.caption_column]
                captions.append(c[0] if isinstance(c, (list, tuple)) else c)
            t2i_inputs, _ = self.wrap_t2i_prompt(captions, device=device)
        else:
            t2i_inputs = self.get_i2t_t2i_inputs(device=device, images=imgs)
        # Reconstruct human-readable captions from the token ids (best-effort).
        # We don't have them as strings, so use the tokenized prompt directly.
        self._eval_prompts_cached = [
            {
                "image_path": rows[i]["image"],
                "input_ids": t2i_inputs["input_ids"][i:i + 1].clone(),
                "attention_mask": t2i_inputs["attention_mask"][i:i + 1].clone(),
            }
            for i in range(k)
        ]

    @torch.inference_mode()
    def _generate_one_image(self, input_ids, attention_mask,
                            img_size=384, patch_size=16, parallel_size=1):
        """Single-image t2i generation with CFG. Mirrors evaluate_checkpoints.py."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        device = input_ids.device
        N = N_IMAGE_TOKENS
        pad_id = self.processing_class.pad_id
        bos_id = self.processing_class.tokenizer.bos_token_id

        # Build cond/uncond batch by duplicating and padding the uncond row.
        L = input_ids.shape[1]
        tokens = torch.zeros((parallel_size * 2, L), dtype=torch.long, device=device)
        attn = torch.zeros((parallel_size * 2, L), dtype=attention_mask.dtype, device=device)
        for i in range(parallel_size * 2):
            tokens[i] = input_ids[0]
            attn[i] = attention_mask[0]
            if i % 2 != 0:
                bos_positions = (tokens[i] == bos_id).nonzero(as_tuple=True)[0]
                if len(bos_positions) > 0:
                    bos_pos = bos_positions[0].item()
                    tokens[i, bos_pos + 1:-1] = pad_id

        inputs_embeds = unwrapped.language_model.get_input_embeddings()(tokens)

        # Prototype conditioning at inference: caption -> w_text -> cond vec.
        # cond row uses w_text @ proto_emb[1:]; uncond row uses proto_emb[0]
        # (the zero "unknown" row). Added on top of every image-token embedding
        # in the loop, matching the training-time t2i_img_pos_bias path.
        proto_bias_2x = None
        if self.use_text_to_proto and hasattr(unwrapped, "text_to_proto") and hasattr(unwrapped, "prototype_emb"):
            cond_mask = attention_mask.to(dtype=inputs_embeds.dtype)              # [1, L]
            cond_in_embeds = unwrapped.language_model.get_input_embeddings()(input_ids)
            text_out = unwrapped.language_model.model(
                inputs_embeds=cond_in_embeds,
                attention_mask=attention_mask,
            )
            text_hidden = text_out.last_hidden_state                               # [1, L, d]
            denom = cond_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (text_hidden * cond_mask.unsqueeze(-1)).sum(dim=1) / denom    # [1, d]
            text_logits = unwrapped.text_to_proto(pooled)                          # [1, K]
            w_text = F.softmax(text_logits.float(), dim=-1)                        # [1, K]
            proto_w = unwrapped.prototype_emb.weight                               # [K+1, d]
            cond_vec = w_text.to(dtype=proto_w.dtype) @ proto_w[1:]                # [1, d]
            uncond_vec = proto_w[0:1].to(dtype=cond_vec.dtype)                     # [1, d]
            d_h = cond_vec.shape[-1]
            proto_bias_2x = torch.zeros(parallel_size * 2, d_h,
                                        device=device, dtype=cond_vec.dtype)
            proto_bias_2x[0::2] = cond_vec.expand(parallel_size, -1)
            proto_bias_2x[1::2] = uncond_vec.expand(parallel_size, -1)

        generated_tokens = torch.zeros((parallel_size, N), dtype=torch.long, device=device)
        outputs = None
        for i in range(N):
            outputs = unwrapped.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None,
            )
            hidden = outputs.last_hidden_state[:, -1, :]
            logits = unwrapped.gen_head(hidden)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + self.eval_image_cfg * (logit_cond - logit_uncond)
            probs = F.softmax(logits.float() / max(self.eval_image_temp, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated_tokens[:, i] = next_token

            both = torch.stack([next_token, next_token], dim=1).view(-1)
            inputs_embeds = unwrapped.prepare_gen_img_embeds(both).unsqueeze(1)
            if proto_bias_2x is not None:
                inputs_embeds = inputs_embeds + proto_bias_2x.to(
                    dtype=inputs_embeds.dtype
                ).unsqueeze(1)
            attn = torch.cat(
                [attn, torch.ones(attn.shape[0], 1, dtype=attn.dtype, device=device)],
                dim=1,
            )

        grid = img_size // patch_size
        dec = unwrapped.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, grid, grid],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(dec[0]), generated_tokens[0].cpu()

    def _run_t2i_eval(self, step: int):
        if self._eval_prompts_cached is None:
            try:
                self._cache_eval_prompts()
            except Exception as e:
                print(f"[T2IEvalCallback] failed to cache eval prompts: {e}")
                return

        out_dir = Path(self.args.output_dir) / self.eval_image_subdir / f"step_{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        was_training = self.model.training
        self.model.eval()
        try:
            for i, p in enumerate(self._eval_prompts_cached):
                try:
                    gen_img, gen_tokens = self._generate_one_image(
                        p["input_ids"], p["attention_mask"],
                    )
                    orig_img = Image.open(p["image_path"]).convert("RGB")
                    # Side-by-side save for quick visual comparison.
                    h = 384
                    orig_resized = orig_img.resize(
                        (int(orig_img.width * h / orig_img.height), h),
                        Image.Resampling.BICUBIC,
                    )
                    canvas = Image.new("RGB", (orig_resized.width + gen_img.width + 10, h), (255, 255, 255))
                    canvas.paste(orig_resized, (0, 0))
                    canvas.paste(gen_img, (orig_resized.width + 10, 0))
                    canvas.save(out_dir / f"sample_{i:02d}.png")
                    # Log basic token diversity to the trainer metrics.
                    n_unique = int(torch.unique(gen_tokens).numel())
                    self._metrics["eval_unique_tokens"].append(n_unique)
                except Exception as e:
                    print(f"[T2IEvalCallback] sample {i} failed: {e}")
            print(f"[T2IEvalCallback] saved {len(self._eval_prompts_cached)} samples to {out_dir}")
        finally:
            if was_training:
                self.model.train()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(model)

        t2i_input_ids = self._apply_prompt_dropout(inputs["t2i_input_ids"])

        # GT image -> VQ token ids (used both as teacher-forcing context AND CE targets).
        with torch.inference_mode():

            images = inputs['images']
            pixel_values = torch.stack([VQ_TRANSFORM(img) for img in images]).to(
            device=device, dtype=torch.bfloat16)

            # t2i_discrete_img_ids = model.module.gen_vision_model.encode(pixel_values)[-1][-1]
            t2i_discrete_img_ids = unwrapped.gen_vision_model.encode(pixel_values)[-1][-1]
            t2i_discrete_img_ids = t2i_discrete_img_ids.reshape(
                            t2i_input_ids.shape[0], -1)

        t2i_discrete_img_ids = t2i_discrete_img_ids.clone()

        t2i_attention_mask = inputs["t2i_attention_mask"]
        fwd_input_ids, fwd_inputs_embeds = t2i_input_ids, None

        # Prototype conditioning: soft-assign image to K centroids, build a
        # mixture-of-prototypes vector, pass via t2i_img_pos_bias.
        t2i_img_pos_bias = None
        if self.use_prototype_conditioning:
            self._ensure_perceptual_model(device)
            with torch.no_grad():
                feat_in = pixel_values.to(dtype=torch.bfloat16)
                feat_in = (feat_in.float() + 1.0) * 0.5
                feat_in = feat_in.clamp(0.0, 1.0)
                feat_in = F.interpolate(feat_in, size=self._perceptual_res, mode="bicubic", align_corners=False)
                feat_in = (feat_in.to(dtype=torch.bfloat16) - self._perceptual_mean) / self._perceptual_std
                img_feat = self._perceptual_model(feat_in)
                img_feat = F.normalize(img_feat.float(), dim=-1)
                centroids = self._prototype_centroids.to(device=device, dtype=torch.float32)
                self._prototype_centroids = centroids
                sims = img_feat @ centroids.t()
                weights = F.softmax(sims / max(self.cond_temperature, 1e-6), dim=-1)
            proto_w = unwrapped.prototype_emb.weight                             # [K+1, d]; row 0 = "unknown"
            w_image = weights

            cond = w_image.to(dtype=proto_w.dtype) @ proto_w[1:]
            if model.training and self.cond_dropout_prob > 0:
                drop = torch.rand(cond.shape[0], device=cond.device) < self.cond_dropout_prob
                if drop.any():
                    zero_row = proto_w[0].to(dtype=cond.dtype).expand_as(cond)
                    cond = torch.where(drop.unsqueeze(-1), zero_row, cond)
                self._metrics["proto_cond_kept"].append(float((~drop).float().mean()))
            else:
                self._metrics["proto_cond_kept"].append(1.0)
            t2i_img_pos_bias = cond                                              # Janus broadcasts to [B, 576, d]

        student_out = model(
            t2i_input_ids=fwd_input_ids,
            t2i_inputs_embeds=fwd_inputs_embeds,
            t2i_attention_mask=t2i_attention_mask,
            t2i_discrete_img_ids=t2i_discrete_img_ids,
            t2i_img_pos_bias=t2i_img_pos_bias,
            t2i_logits_to_keep=t2i_discrete_img_ids.shape[1],
            task="generation",
            t2i_compute_text_to_proto=self.use_text_to_proto,
            t2i_text_pool_mask=t2i_attention_mask if self.use_text_to_proto else None,
        )
        student_logits = student_out.logits

        # text2proto KL aux loss; head was called inside model.forward so DDP
        # sees its params; hidden states are detached inside that call.
        loss_text2proto = None
        if self.use_text_to_proto and getattr(student_out, "text_proto_logits", None) is not None:
            w_text = F.softmax(student_out.text_proto_logits.float(), dim=-1)
            w_img_d = w_image.detach().float().clamp(min=1e-8)
            w_txt_c = w_text.clamp(min=1e-8)
            loss_text2proto = (w_txt_c * (w_txt_c.log() - w_img_d.log())).sum(dim=-1).mean()
            self._metrics["loss_text2proto"].append(float(loss_text2proto.item()))
            with torch.no_grad():
                self._metrics["text2proto_argmax_match"].append(
                    float((w_text.argmax(-1) == w_image.argmax(-1)).float().mean().item())
                )

        B, N, V = student_logits.shape
        if not torch.isfinite(student_logits).all():
            with torch.no_grad():
                finite = torch.isfinite(student_logits)
                bad = (~finite).sum().item()
                total = student_logits.numel()
                if finite.any():
                    fv = student_logits[finite]
                    print(f"[NaN] student_logits non-finite={bad}/{total} "
                          f"finite-min={fv.min().item():.3f} max={fv.max().item():.3f}")
                else:
                    print(f"[NaN] student_logits ENTIRELY non-finite ({bad}/{total})")
                for nm in ("gen_aligner", "gen_head", "prototype_emb", "text_to_proto"):
                    ps = [p for n, p in model.named_parameters() if nm in n and p.requires_grad]
                    bad = sum(int(not torch.isfinite(p).all()) for p in ps)
                    print(f"[NaN] param-group {nm}: {bad}/{len(ps)} non-finite")
                if not hasattr(self, "_nan_dumped"):
                    self._nan_dumped = True
                    dump = {
                        "student_logits": student_logits.detach().cpu(),
                        "t2i_attention_mask": t2i_attention_mask.detach().cpu(),
                    }
                    torch.save(dump, f"{self.args.output_dir}/nan_dump.pt")
                    print(f"[NaN] saved tensors to {self.args.output_dir}/nan_dump.pt")
        loss_ce = F.cross_entropy(
            student_logits.reshape(-1, V).float(), t2i_discrete_img_ids.reshape(-1)
        )
        loss = loss_ce

        if self.use_text_to_proto and loss_text2proto is not None and torch.isfinite(loss_text2proto):
            loss = loss + self.text_to_proto_aux_weight * loss_text2proto.to(dtype=loss.dtype)

        # STE perceptual loss (BiomedCLIP cosine distance).
        if self.use_perceptual_loss:
            # 5% hold then linear ramp. Skipping when ramp==0 avoids 0*NaN poisoning.
            if self.perceptual_warmup_steps > 0:
                step = float(self.state.global_step)
                warmup = float(self.perceptual_warmup_steps)
                hold = warmup * 0.05
                ramp = max(0.0, min(1.0, (step - hold) / max(warmup - hold, 1.0)))
            else:
                ramp = 1.0
        if self.use_perceptual_loss and ramp > 0:
            self._ensure_perceptual_model(device)
            pred_pixels = self._ste_decode_pixels(student_logits, unwrapped)
            with torch.no_grad():
                ok = torch.isfinite(pred_pixels).flatten(1).all(dim=1)
            if not ok.all():
                if ok.any():
                    pred_pixels = pred_pixels[ok]
                    pixel_values_perc = pixel_values[ok]
                else:
                    pred_pixels = None
                self._metrics["perceptual_dropped_samples"].append(int((~ok).sum().item()))
            else:
                pixel_values_perc = pixel_values
            if pred_pixels is None:
                loss_perc = None
            else:
                loss_perc = self._perceptual_distance(pred_pixels, pixel_values_perc.detach())
            # Dynamic balance so effective_w*loss_perc ~= loss_ce at weight=1.0.
            if loss_perc is not None and torch.isfinite(loss_perc):
                balance = (loss_ce.detach() / (loss_perc.detach() + 1e-8)).clamp(max=100.0)
                effective_w = self.perceptual_weight * balance * ramp
                loss = loss_ce + effective_w * loss_perc
                self._metrics["loss_perceptual"].append(loss_perc.item())
                self._metrics["perceptual_weight_eff"].append(float(effective_w))
            else:
                self._metrics["perceptual_skipped"].append(1.0)
            self._metrics["perceptual_ramp"].append(ramp)

        # Pixel-space LPIPS reconstruction loss.
        if self.use_reconstruction_loss:
            self._ensure_lpips(device)
            pred_pixels = self._ste_decode_pixels(student_logits, unwrapped)
            with torch.no_grad():
                ok = torch.isfinite(pred_pixels).flatten(1).all(dim=1)
            if ok.any():
                pred_keep = pred_pixels[ok].clamp(-1.0, 1.0).float()
                gt_keep = pixel_values[ok].clamp(-1.0, 1.0).float()
                lpips_val = self._lpips_metric(pred_keep, gt_keep)
                if torch.isfinite(lpips_val):
                    loss = loss + self.lpips_weight * lpips_val.to(dtype=loss.dtype)
                    self._metrics["loss_lpips"].append(float(lpips_val.item()))
                else:
                    self._metrics["lpips_skipped"].append(1.0)
            else:
                self._metrics["lpips_skipped"].append(1.0)

        self._metrics["loss_ce"].append(loss_ce.item())
        with torch.no_grad():
            pred = student_logits.argmax(dim=-1)
            correct = (pred == t2i_discrete_img_ids).float()                    # [B, N]
            self._metrics["token_acc"].append(correct.mean().item())
            # Per-quartile accuracy: exposes prompt-vs-context reliance.
            # q0 = positions w/ least context (model must use prompt).
            # q3 = positions w/ most context (model can lean on neighbors).
            q_size = N // 4
            for q in range(4):
                s, e = q * q_size, (q + 1) * q_size if q < 3 else N
                self._metrics[f"token_acc_q{q}"].append(correct[:, s:e].mean().item())

        return loss

    # ---------- Pixel-space LPIPS (VGG) ----------

    def _ensure_lpips(self, device):
        if self._lpips_metric is not None:
            return
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)
        m = m.to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        self._lpips_metric = m

    # ---------- Perceptual loss (STE decode + BiomedCLIP feature distance) ----------

    def _ensure_perceptual_model(self, device):
        """Lazy-load BiomedCLIP visual encoder, register multi-scale hooks."""
        if self._perceptual_model is not None:
            return
        import open_clip
        model, _ = open_clip.create_model_from_pretrained(self.perceptual_model_id)
        model = model.visual
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device=device, dtype=torch.bfloat16)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
        self._perceptual_model = model
        self._perceptual_mean = mean
        self._perceptual_std = std
        try:
            self._perceptual_res = int(model.image_size if isinstance(model.image_size, int) else model.image_size[0])
        except Exception:
            self._perceptual_res = 224

        blocks = self._find_transformer_blocks(model)
        if blocks is None:
            self._perceptual_blocks = None
            self._perceptual_tap_idx = []
            return
        n = len(blocks)
        if self.perceptual_layers.strip():
            tap = sorted({int(x) - 1 for x in self.perceptual_layers.split(",") if x.strip()})
            tap = [i for i in tap if 0 <= i < n]
        else:
            tap = sorted({max(0, (k * n) // 4 - 1) for k in range(1, 5)})
        self._perceptual_blocks = blocks
        self._perceptual_tap_idx = tap

        def _make_hook(idx):
            def _hook(_module, _inp, out):
                self._perceptual_buf[idx] = out[0] if isinstance(out, tuple) else out
            return _hook

        for i in tap:
            blocks[i].register_forward_hook(_make_hook(i))

    @staticmethod
    def _find_transformer_blocks(visual):
        """Return the ModuleList of transformer blocks across open_clip/timm towers."""
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            return visual.transformer.resblocks
        if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
            return visual.trunk.blocks
        if hasattr(visual, "blocks"):
            return visual.blocks
        return None

    def _ste_decode_pixels(self, student_logits, unwrapped):
        """STE image decode. Returns pixels in [-1, 1], shape [B, 3, 384, 384]."""
        quantizer = unwrapped.gen_vision_model.quantize
        codebook = quantizer.embedding.weight
        if getattr(quantizer, "l2_norm", False):
            codebook = F.normalize(codebook, p=2, dim=-1)
        codebook = codebook.to(dtype=student_logits.dtype)

        probs = F.softmax(student_logits, dim=-1)
        soft_emb = probs @ codebook
        hard_ids = student_logits.argmax(dim=-1)
        hard_emb = codebook[hard_ids]
        # STE: forward = hard_emb, backward = ∂L/∂soft_emb.
        ste_emb = soft_emb + (hard_emb - soft_emb).detach()

        B, N, D = ste_emb.shape
        grid = int(N ** 0.5)
        assert grid * grid == N, f"image-token count {N} is not a square grid"
        z = ste_emb.transpose(1, 2).reshape(B, D, grid, grid).contiguous()

        decoder_dtype = next(unwrapped.gen_vision_model.decoder.parameters()).dtype
        z = z.to(dtype=decoder_dtype)
        return unwrapped.gen_vision_model.decode(z)

    def _perceptual_distance(self, pred_pixels, gt_pixels):
        """Multi-layer cosine distance over BiomedCLIP tapped-block features."""
        def _prep(x):
            x = (x.float() + 1.0) * 0.5
            x = x.clamp(0.0, 1.0)
            x = F.interpolate(x, size=self._perceptual_res, mode="bicubic", align_corners=False)
            x = (x.to(dtype=torch.bfloat16) - self._perceptual_mean) / self._perceptual_std
            return x

        def _token_cos_dist(a, b):
            an = F.normalize(a.float(), dim=-1)
            bn = F.normalize(b.float(), dim=-1)
            return (1.0 - (an * bn).sum(dim=-1)).mean()

        with torch.no_grad():
            gt_final = self._perceptual_model(_prep(gt_pixels))
            gt_feats = {i: self._perceptual_buf[i].detach() for i in self._perceptual_tap_idx}

        pred_final = self._perceptual_model(_prep(pred_pixels))
        pred_feats = {i: self._perceptual_buf[i] for i in self._perceptual_tap_idx}

        if not self._perceptual_tap_idx:
            pred_n = F.normalize(pred_final.float(), dim=-1)
            gt_n = F.normalize(gt_final.float(), dim=-1)
            return (1.0 - (pred_n * gt_n).sum(dim=-1)).mean()

        dists = []
        for i in self._perceptual_tap_idx:
            d = _token_cos_dist(pred_feats[i], gt_feats[i])
            dists.append(d)
            self._metrics[f"loss_perc_l{i + 1}"].append(d.item())

        return torch.stack(dists).mean()

    def _apply_prompt_dropout(self, t2i_input_ids):
        """For ~p fraction of samples, replace prompt body with pad_id (keep BOS + BOI)."""
        if not self.model.training or self.prompt_dropout_prob <= 0:
            return t2i_input_ids
        B = t2i_input_ids.shape[0]
        drop_mask = torch.rand(B, device=t2i_input_ids.device) < self.prompt_dropout_prob
        if not drop_mask.any():
            return t2i_input_ids

        pad_id = self.processing_class.pad_id
        bos_id = self.processing_class.tokenizer.bos_token_id
        out = t2i_input_ids.clone()
        for b in range(B):
            if not drop_mask[b]:
                continue
            bos_positions = (out[b] == bos_id).nonzero(as_tuple=True)[0]
            if len(bos_positions) == 0:
                continue
            bos_pos = bos_positions[0].item()
            # Keep BOS at bos_pos and BOI at -1; pad everything strictly between.
            out[b, bos_pos + 1:-1] = pad_id
        return out

    def training_step(self, model, inputs, num_items_in_batch=None):
        step_start = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._metrics["train_step_time_s"].append(time.perf_counter() - step_start)
        # Zero NaN grads so a bad batch doesn't poison total_norm in clipping.
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
                print(f"[grad-NaN] step={self.state.global_step} loss={float(loss):.4f} "
                      f"zeroed {len(nan_params)} param grads. First few: {nan_params[:5]}")
        return loss

    def _prepare_inputs(self, inputs):
        device = self.accelerator.device
        loaded_images = self.load_batch_images(inputs)

        if self.caption_source == "original":
            captions = []
            for x in inputs:
                cap = x.get(self.caption_column)
                if cap is None:
                    raise KeyError(
                        f"caption_source='original' but column "
                        f"'{self.caption_column}' is missing from the batch row. "
                        f"Available keys: {list(x.keys())}"
                    )
                # If the column holds K cached captions per image (e.g. from
                # build_caption_cache.py), pick one at random each step.
                if isinstance(cap, (list, tuple)):
                    cap = random.choice(cap)
                captions.append(cap)
            t2i_inputs, _ = self.wrap_t2i_prompt(captions, device=device)
        else:
            t2i_inputs = self.get_i2t_t2i_inputs(device=device, images=loaded_images)

        t2i_input_ids = t2i_inputs["input_ids"]
        t2i_attention_mask = t2i_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            t2i_input_ids = t2i_input_ids[:, -self.max_prompt_length:]
            t2i_attention_mask = t2i_attention_mask[:, -self.max_prompt_length:]

        batch = {
            "t2i_input_ids": t2i_input_ids,
            "t2i_attention_mask": t2i_attention_mask,
            # "t2i_discrete_img_ids": gt_image_ids,
            "images": loaded_images,  # placeholder for potential future use
            # Per-sample ids used by the lagged w_text cache (text2proto path).
            # None when a row has no id field.
            "sample_ids": [x.get("id") for x in inputs],
        }
        return batch

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()
