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

# T2I-only variant of grpo_trainer_unified.py — all mm2t (VL QA) computation removed.
# The model forward is invoked with task="generation" instead of task="unify".

import time
from collections import defaultdict
from typing import Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from packaging import version
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import gather, is_peft_model, set_seed
from PIL import Image

from trl.import_utils import is_deepspeed_available
from trl import ScriptArguments
from trl.trainer.callbacks import SyncRefModelCallback
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import selective_log_softmax

if is_deepspeed_available():
    import deepspeed  # noqa: F401

if is_peft_available():
    from peft import PeftConfig, get_peft_model

from janus.models import VLChatProcessor
from corl.open_r1.rewards.r_t2i import T2ICycleConsistencyReward


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1)
    return torch.sqrt(variance)


class JanusProUnifiedGRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
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
            args = GRPOConfig(f"{model_name}-GRPO")

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

        # ******************* Reward functions (t2i only) *******************
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 't2i_CycleConsistency' in reward_func:
                reward_funcs[i] = T2ICycleConsistencyReward(self.task_args)
            elif isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )

        unsupported = [
            rf for rf in reward_funcs
            if "t2i_CycleConsistency" not in rf.__name__ and "t2i_match" not in rf.__name__
        ]
        if unsupported:
            raise ValueError(
                "This trainer (noMM variant) only supports t2i_CycleConsistency and t2i_match "
                f"rewards. Got unsupported rewards: {[rf.__name__ for rf in unsupported]}. "
                "Use the full JanusProUnifiedGRPOTrainer from grpo_trainer_unified.py instead."
            )

        self.t2i_reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number "
                    f"of reward functions ({len(reward_funcs)})"
                )
            self.t2i_reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.t2i_reward_weights = torch.ones(len(self.t2i_reward_funcs), dtype=torch.float32)

        def data_collator(features):
            return features

        # ******************* Training arguments *******************
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature

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

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            if "Janus" in model_id:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs)
        elif is_peft_model(model):
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        self._metrics = defaultdict(list)

        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.tokenizer.eos_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
        )
        self.t2i_generation_kwargs = {
            "cfg_weight": 5,
            "parallel_size": self.num_generations,
            "temperature": 1,  # HACK
            "image_token_num_per_image": 576,
            "img_size": 384,
            "patch_size": 16,
            "pad_id": processing_class.pad_id,
            "seed": self.args.seed,
        }

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(
                ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.t2i_reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.t2i_reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )
            elif isinstance(reward_func, T2ICycleConsistencyReward):
                reward_func.load_external_model(self.accelerator.device)

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

    def _get_per_token_logps(
            self, model,
            t2i_inputs_ids, t2i_attention_mask,
            t2i_discrete_img_ids, t2i_logits_to_keep,
    ):
        t2i_discrete_img_ids = t2i_discrete_img_ids.to(t2i_inputs_ids.dtype)

        outputs = model(
            t2i_input_ids=t2i_inputs_ids,
            t2i_attention_mask=t2i_attention_mask,
            t2i_discrete_img_ids=t2i_discrete_img_ids,
            t2i_logits_to_keep=t2i_logits_to_keep,
            task="generation",
        )
        t2i_logits = outputs.logits / self.temperature
        return selective_log_softmax(t2i_logits, t2i_discrete_img_ids)

    def load_batch_images(self, inputs):
        loaded_images = []
        for x in inputs:
            if "image" in x and x["image"] is not None:
                loaded_images.append(x["image"].convert("RGB"))
                continue

            image_path = x.get("image_full_path")
            if image_path is None:
                raise KeyError("Each sample must contain 'image' or 'image_full_path'.")
            loaded_images.append(Image.open(image_path).convert("RGB"))
        return loaded_images

    def wrap_t2i_prompt(self, inputs, device=None):
        prompts = []
        for xp in inputs:
            sft_format = self.processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=xp["prompt"],
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

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device

        loaded_images = self.load_batch_images(inputs)

        # Prepare t2i prompt inputs
        t2i_prompt_inputs, t2i_prompts = self.wrap_t2i_prompt(inputs, device)
        t2i_prompt_ids = t2i_prompt_inputs["input_ids"]
        t2i_prompt_mask = t2i_prompt_inputs["attention_mask"]
        t2i_prompts = [
            pp.strip('<|User|>:').strip('<|Assistant|>:<begin_of_image>').strip()
            for pp in t2i_prompts
        ]
        if self.max_prompt_length is not None:
            t2i_prompt_ids = t2i_prompt_ids[:, -self.max_prompt_length:]
            t2i_prompt_mask = t2i_prompt_mask[:, -self.max_prompt_length:]

        # === Generate completion ===
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            t2i_completion_ids, t2i_completions = unwrapped_model.t2i_generate_parallel(
                input_ids=t2i_prompt_ids, attention_mask=t2i_prompt_mask,
                **self.t2i_generation_kwargs
            )

        t2i_prompt_ids = t2i_prompt_ids.repeat_interleave(self.num_generations, dim=0)
        t2i_prompt_mask = t2i_prompt_mask.repeat_interleave(self.num_generations, dim=0)
        t2i_completion_mask = torch.ones_like(t2i_completion_ids).int()
        t2i_logits_to_keep = t2i_completion_ids.size(1)

        # Old per-token logps (only needed when num_iterations > 1)
        with torch.inference_mode():
            if self.num_iterations > 1:
                t2i_old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    t2i_inputs_ids=t2i_prompt_ids,
                    t2i_attention_mask=t2i_prompt_mask,
                    t2i_discrete_img_ids=t2i_completion_ids,
                    t2i_logits_to_keep=t2i_logits_to_keep,
                )
            else:
                t2i_old_per_token_logps = None

        # === Compute rewards ===
        t2i_prompts = [pp for pp in t2i_prompts for _ in range(self.num_generations)]
        batch_images = [img for img in loaded_images for _ in range(self.num_generations)]

        t2i_rewards_per_func = torch.zeros(
            len(t2i_prompts), len(self.t2i_reward_funcs), device=device)
        for i, reward_func in enumerate(self.t2i_reward_funcs):
            if "t2i_CycleConsistency" in reward_func.__name__:
                reward_kwargs = {"image": batch_images}
                output_reward_func = reward_func(
                    completions=t2i_completions, prompts=t2i_prompts,
                    mmgpt=self.model, processing_class=self.processing_class,
                    **reward_kwargs
                )
            elif "t2i_match" in reward_func.__name__:
                output_reward_func = reward_func(
                    completions=t2i_completions, prompts=t2i_prompts,
                    mmgpt=self.model, processing_class=self.processing_class,
                )
            else:
                raise ValueError(f"Unknown reward function: {reward_func}")

            output_reward_func = [
                reward if reward is not None else torch.nan for reward in output_reward_func
            ]
            t2i_rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device)

        t2i_rewards_per_func = gather(t2i_rewards_per_func)
        t2i_rewards = (t2i_rewards_per_func * self.t2i_reward_weights.to(
            device).unsqueeze(0)).nansum(dim=1)

        # === Grouped advantages ===
        t2i_mean_grouped_rewards = t2i_rewards.view(-1, self.num_generations).mean(dim=1)
        t2i_std_grouped_rewards = t2i_rewards.view(-1, self.num_generations).std(dim=1)
        t2i_mean_grouped_rewards = t2i_mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        t2i_std_grouped_rewards = t2i_std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)

        t2i_advantages = (t2i_rewards - t2i_mean_grouped_rewards) / (
                t2i_std_grouped_rewards + 1e-4)
        process_slice = slice(
            self.accelerator.process_index * len(t2i_prompts),
            (self.accelerator.process_index + 1) * len(t2i_prompts),
        )
        t2i_advantages = t2i_advantages[process_slice]

        self._metrics["reward_t2i"].append(t2i_mean_grouped_rewards.mean().item())

        for i, rf_name in enumerate([rf.__name__ for rf in self.t2i_reward_funcs]):
            mean_rewards = torch.nanmean(t2i_rewards_per_func[:, i]).item()
            self._metrics[f"rewards/{rf_name}/mean"].append(mean_rewards)

        return {
            "t2i_inputs_ids": t2i_prompt_ids,
            "t2i_attention_mask": t2i_prompt_mask,
            "t2i_discrete_img_ids": t2i_completion_ids,
            "t2i_completion_mask": t2i_completion_mask,
            "t2i_old_per_token_logps": t2i_old_per_token_logps,
            "t2i_advantages": t2i_advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        t2i_logits_to_keep = inputs["t2i_completion_mask"].size(1)

        t2i_per_token_logps = self._get_per_token_logps(
            model,
            t2i_inputs_ids=inputs["t2i_inputs_ids"],
            t2i_attention_mask=inputs["t2i_attention_mask"],
            t2i_discrete_img_ids=inputs["t2i_discrete_img_ids"],
            t2i_logits_to_keep=t2i_logits_to_keep,
        )

        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    t2i_ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        t2i_inputs_ids=inputs["t2i_inputs_ids"],
                        t2i_attention_mask=inputs["t2i_attention_mask"],
                        t2i_discrete_img_ids=inputs["t2i_discrete_img_ids"],
                        t2i_logits_to_keep=t2i_logits_to_keep,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        t2i_ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            t2i_inputs_ids=inputs["t2i_inputs_ids"],
                            t2i_attention_mask=inputs["t2i_attention_mask"],
                            t2i_discrete_img_ids=inputs["t2i_discrete_img_ids"],
                            t2i_logits_to_keep=t2i_logits_to_keep,
                        )
            t2i_per_token_kl = (
                    torch.exp(t2i_ref_per_token_logps - t2i_per_token_logps) - (
                        t2i_ref_per_token_logps - t2i_per_token_logps) - 1
            )

        t2i_completion_mask = inputs["t2i_completion_mask"]

        if inputs["t2i_old_per_token_logps"] is None:
            t2i_old_per_token_logps = t2i_per_token_logps.detach()
        else:
            t2i_old_per_token_logps = inputs["t2i_old_per_token_logps"]

        t2i_advantages = inputs["t2i_advantages"]
        t2i_per_token_loss = t2i_advantages.unsqueeze(1) * (
                t2i_per_token_logps - t2i_old_per_token_logps
        ).exp()

        if self.beta != 0.0:
            t2i_per_token_loss = -(t2i_per_token_loss - self.beta * t2i_per_token_kl)
            t2i_mean_kl = ((t2i_per_token_kl * t2i_completion_mask).sum(
                dim=1) / t2i_completion_mask.sum(dim=1)).mean()
            self._metrics["kl_t2i"].append(
                self.accelerator.gather_for_metrics(t2i_mean_kl).mean().item()
            )
        else:
            t2i_per_token_loss = -t2i_per_token_loss

        loss_t2i = (t2i_per_token_loss * t2i_completion_mask).sum() / t2i_completion_mask.sum().clamp(min=1.0)
        return loss_t2i

    def training_step(self, model, inputs, num_items_in_batch=None):
        step_start = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._metrics["train_step_time_s"].append(time.perf_counter() - step_start)
        return loss

    def _prepare_inputs(self, inputs):
        return self._generate_and_score_completions(inputs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()
