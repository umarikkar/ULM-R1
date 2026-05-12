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
import torchvision.transforms as T

from trl.import_utils import is_deepspeed_available
from trl import ScriptArguments, SFTConfig

if is_deepspeed_available():
    import deepspeed  # noqa: F401

if is_peft_available():
    from peft import PeftConfig, get_peft_model

from janus.models import VLChatProcessor

VQ_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

N_IMAGE_TOKENS = 196


def fix_janus_text(out_caption):
    out_caption = out_caption.replace("Ġ", " ")
    out_caption = out_caption.replace("Ċ", "\n")

    # Optional cleanup for extra spacing/newlines
    out_caption = re.sub(r"[ \t]+", " ", out_caption)
    out_caption = re.sub(r"\n\s*\n+", "\n", out_caption)

    return out_caption.strip()


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
        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False


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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(model)

        # GT image -> VQ token ids (supervision targets).
        with torch.inference_mode():
            _, gt_ids, _ = self.get_image_gen_reps(
                unwrapped, device=device, images=inputs["images"]
            )

            # AR rollout under current weights -> student's own image-token context.
            student_ids = self.ar_rollout_student_ids(
                model, inputs["t2i_input_ids"], inputs["t2i_attention_mask"]
            )

        # Need a writable copy: tensors from inference_mode can't be used in a
        # grad-enabled forward without cloning out of inference state.
        student_ids = student_ids.clone()
        gt_ids = gt_ids.clone()

        # Student forward (grad on): text prompt + student's own AR ids as
        # teacher-forcing context, read logits at the N image positions.
        student_logits = model(
            t2i_input_ids=inputs["t2i_input_ids"],
            t2i_attention_mask=inputs["t2i_attention_mask"],
            t2i_discrete_img_ids=student_ids,
            t2i_logits_to_keep=student_ids.shape[1],
            task="generation",
        ).logits                                                                # [B, N, V_img]

        B, N, V = student_logits.shape
        loss = F.cross_entropy(
            student_logits.reshape(-1, V).float(), gt_ids.reshape(-1)
        )

        self._metrics["loss_ce"].append(loss.item())
        with torch.no_grad():
            pred = student_logits.argmax(dim=-1)
            self._metrics["token_acc"].append((pred == gt_ids).float().mean().item())
            self._metrics["rollout_gt_match"].append(
                (student_ids == gt_ids).float().mean().item()
            )

        return loss

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
