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

        model = self.init_trainable_parameters(model)

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`")
            model = get_peft_model(model, peft_config)

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
        self.prompt_dropout_prob = getattr(task_args, "prompt_dropout_prob", 0.1)
        self.eval_image_freq = getattr(task_args, "eval_image_freq", 100)
        self.eval_image_num = getattr(task_args, "eval_image_num", 4)
        self.eval_image_cfg = getattr(task_args, "eval_image_cfg", 5.0)
        self.eval_image_temp = getattr(task_args, "eval_image_temp", 1.0)
        self.eval_image_subdir = getattr(task_args, "eval_image_subdir", "eval_samples")

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

        if self.eval_image_freq > 0:
            self.add_callback(T2IEvalCallback(self, freq=self.eval_image_freq))


    @staticmethod
    def init_trainable_parameters(model):
        # fix und
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.aligner.parameters():
            param.requires_grad = False

        # fix gen vision (VQ-VAE codebook is already sufficient for target domain)
        for param in model.gen_vision_model.parameters():
            param.requires_grad = False
        # fix gen_embed (let gen_aligner absorb any re-mapping of the embedding table)
        for param in model.gen_embed.parameters():
            param.requires_grad = False

        # fix LLM backbone — only the two T2I adapters are trainable
        for param in model.language_model.parameters():
            param.requires_grad = False

        # trainable: gen_head, gen_aligner (full fine-tune)
        for param in model.gen_head.parameters():
            param.requires_grad = True
        for param in model.gen_aligner.parameters():
            param.requires_grad = True

        return model

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

        task_instruct = "Describe the main content of the image in one sentence."
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
        """Take the first `eval_image_num` training samples, run i2t once, cache
        (caption, image_path) so eval renders the same images at every step."""
        device = self.accelerator.device
        k = min(self.eval_image_num, len(self.train_dataset))
        rows = [self.train_dataset[i] for i in range(k)]
        imgs = [Image.open(r["image"]).convert("RGB") for r in rows]
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

        # GT image -> VQ token ids (used both as teacher-forcing context AND CE targets).
        with torch.inference_mode():
            _, gt_ids, _ = self.get_image_gen_reps(
                unwrapped, device=device, images=inputs["images"]
            )

        # Out of inference_mode so the tensor can participate in a grad forward.
        gt_ids = gt_ids.clone()

        # Classifier-free-guidance training: with prob p, replace inner prompt tokens
        # with pad_id so the model also learns the unconditional distribution.
        t2i_input_ids = self._apply_prompt_dropout(inputs["t2i_input_ids"])

        # Standard AR teacher forcing: prompt + GT image tokens as context.
        # modeling_vlm.py forward (task="generation") returns N logits taken from
        # hidden_states[:, -N-1:-1, :] — i.e., logits[i] predicts gt_ids[i] given
        # (prompt, gt_ids[<i]). The position shift is handled inside the model.
        student_logits = model(
            t2i_input_ids=t2i_input_ids,
            t2i_attention_mask=inputs["t2i_attention_mask"],
            t2i_discrete_img_ids=gt_ids,
            t2i_logits_to_keep=gt_ids.shape[1],
            task="generation",
        ).logits                                                                # [B, N, V_img]

        B, N, V = student_logits.shape
        loss = F.cross_entropy(
            student_logits.reshape(-1, V).float(), gt_ids.reshape(-1)
        )

        self._metrics["loss_ce"].append(loss.item())
        with torch.no_grad():
            pred = student_logits.argmax(dim=-1)
            correct = (pred == gt_ids).float()                                  # [B, N]
            self._metrics["token_acc"].append(correct.mean().item())
            # Per-quartile accuracy: exposes prompt-vs-context reliance.
            # q0 = positions w/ least context (model must use prompt).
            # q3 = positions w/ most context (model can lean on neighbors).
            q_size = N // 4
            for q in range(4):
                s, e = q * q_size, (q + 1) * q_size if q < 3 else N
                self._metrics[f"token_acc_q{q}"].append(correct[:, s:e].mean().item())

        return loss

    def _apply_prompt_dropout(self, t2i_input_ids):
        """For ~p fraction of samples, replace prompt body with pad_id (keep BOS + BOI).

        Mirrors the unconditional branch used by CFG in evaluate_checkpoints.py:104-105.
        """
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
        return loss

    def _prepare_inputs(self, inputs):
        device = self.accelerator.device
        loaded_images = self.load_batch_images(inputs)

        # vq_pixel_values = torch.stack([VQ_TRANSFORM(img) for img in loaded_images]).to(
        #     device=device, dtype=torch.bfloat16
        # )

        t2i_inputs = self.get_i2t_t2i_inputs(device=device, images=loaded_images)
        # gt_image_ids = self.get_gt_image_ids(device=device, images=loaded_images)

        t2i_input_ids = t2i_inputs["input_ids"]
        t2i_attention_mask = t2i_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            t2i_input_ids = t2i_input_ids[:, -self.max_prompt_length:]
            t2i_attention_mask = t2i_attention_mask[:, -self.max_prompt_length:]

        return {
            "t2i_input_ids": t2i_input_ids,
            "t2i_attention_mask": t2i_attention_mask,
            # "t2i_discrete_img_ids": gt_image_ids,
            "images": loaded_images,  # placeholder for potential future use
        }

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()
