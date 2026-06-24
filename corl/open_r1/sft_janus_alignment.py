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

from dataclasses import dataclass, field
import json
import os
import time
from typing import Optional
from datasets import load_dataset, Image as HFImage
from PIL import Image

from transformers import TrainerCallback
from trl import (
    GRPOConfig, ModelConfig, SFTConfig, ScriptArguments,
    TrlParser, get_peft_config
)
from corl.open_r1.trainer.sft_trainer_alignment import SFTAlignmentTrainer

class ParameterInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"Fixed: {name}, Shape: {param.shape}, Parameters: {param.numel():,}")


@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the SFT training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'",
            "nargs": "+",
        },
    )
    task_format: Optional[str] = field(
        default="t2i",
        metadata={
            "help": "Possible values: 't2i' (text to image), 'mm2t' (mm to text), "
                    "'joint': , 'unify'"
        },
    )
    mm2t_format: Optional[str] = field(
        default='qa',
        metadata={
            "help": "Possible values: 'qa', 'od' (object detection), 'oc' (object classification)"
        },
    )

    alignment_losses: list[str] = field(
        default_factory=lambda: ["masking"],
        metadata={
            "help": "List of alignment losses to apply. Possible values: 'masking', 'hidden'",
            "nargs": "+",
        },
    )

    caption_cs_metrics: list[str] = field(
        default_factory=lambda: ["jaccard", "bertscore"],
        metadata={
            "help": "List of caption consistency metrics. "
                    "Possible values: 'jaccard', 'bertscore', 'SPICE'",
            "nargs": "+",
        },
    )
    using_simcse: bool = field(
        default=False,
        metadata={"help": "."},
    )
    using_image_cs: bool = field(
        default=True,
        metadata={"help": "."},
    )
    image_cs_metrics: list[str] = field(
        default_factory=lambda: ["mse"],
        metadata={
            "help": "List of image consistency metrics. "
                    "Possible values: 'lpips', 'mse', ''",
            "nargs": "+",
        },
    )
    using_external_caption_model: bool = field(
        default=False,
        metadata={"help": "."},
    )
    model_ckpt_dir: str = field(
        default="./checkpoint/"
    )

    dataset_cache_dir: str = field(
        default=os.environ.get("HF_DATASETS_CACHE", None),
    )
    data_dir: str = field(
        default="",
        metadata={"help": "Base directory for image_path column in parquet datasets."},
    )
    attribute_sidecar: str = field(
        default="",
        metadata={
            "help": (
                "Path to the per-image attribute sidecar JSON produced by "
                "build_attribute_sidecar.py (id -> {modality, pose, is_grid}). "
                "When set, rows are joined by id and is_grid=='multi' rows are "
                "filtered out. Empty disables both join and filter."
            )
        },
    )
    exclude_ids_json: str = field(
        default="",
        metadata={
            "help": (
                "Path to a JSON file (either a list of id strings or a list of "
                "dicts each with an 'id' key, e.g. corl/eval/test_split.json) "
                "whose ids are dropped from the training set so train/test stay "
                "disjoint. Empty disables."
            )
        },
    )
    lazy_image_loading: bool = field(
        default=False,
        metadata={"help": "If true, keep image paths only and decode images lazily in the trainer."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional limit on samples per split for faster debugging runs."},
    )

    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Max token length for prompt inputs."},
    )
    max_completion_length: int = field(
        default=512,
        metadata={"help": "Max token length for completion outputs."},
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling during training."},
    )

    use_reconstruction_loss: bool = field(
        default=False,
        metadata={"help": "If True, add a pixel-space LPIPS reconstruction loss on top of the latent MSE."},
    )
    lpips_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the LPIPS term: total_loss = loss_align + lpips_weight * loss_lpips."},
    )

    prompt_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of replacing prompt body with pad_id during training (CFG training)."},
    )
    eval_image_freq: int = field(
        default=100,
        metadata={"help": "Run t2i inference and save images every N optimizer steps. 0 disables."},
    )
    eval_image_num: int = field(
        default=4,
        metadata={"help": "Number of fixed eval prompts to render at each eval step."},
    )
    eval_image_cfg: float = field(
        default=5.0,
        metadata={"help": "CFG weight used during eval-time generation."},
    )
    eval_image_temp: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature used during eval-time generation."},
    )
    eval_image_subdir: str = field(
        default="eval_samples",
        metadata={"help": "Subdirectory under output_dir to save eval-time generations."},
    )

    # ---- STE perceptual loss ----
    use_perceptual_loss: bool = field(
        default=False,
        metadata={"help": "Add a BiomedCLIP-feature perceptual loss on STE-decoded pixels."},
    )
    perceptual_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Target ratio of perceptual contribution to CE contribution post-warmup. "
                "1.0 -> perceptual term is auto-scaled each step so that "
                "(effective_lambda * loss_perceptual) == loss_ce.detach(). "
                "0.5 -> perceptual contributes half as much as CE. "
                "2.0 -> twice as much."
            )
        },
    )
    perceptual_model_id: str = field(
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        metadata={"help": "open_clip create_model_from_pretrained() id for the frozen image encoder."},
    )
    perceptual_layers: str = field(
        default="",
        metadata={
            "help": (
                "Comma-separated 1-indexed transformer layers to tap for the perceptual "
                "loss (e.g. '3,6,9,12'). Intermediate layers capture low-level structure, "
                "which matters for structurally-similar medical images. Empty -> auto-pick "
                "4 evenly-spaced layers (quarters), always including the final layer."
            )
        },
    )
    perceptual_warmup_steps: int = field(
        default=500,
        metadata={
            "help": (
                "Linearly ramp the perceptual loss weight from 0 -> perceptual_weight "
                "over this many optimizer steps. 0 disables warmup (full weight from step 1)."
            )
        },
    )

    cond_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": (
                "Bernoulli CFG dropout prob. Replaces the prototype conditioning vector "
                "with the 'unknown' (index 0) zero row, so the model learns the "
                "unconditional path for classifier-free guidance at sample time."
            )
        },
    )

    # ---- Unsupervised prototype conditioning ----
    use_prototype_conditioning: bool = field(
        default=False,
        metadata={
            "help": (
                "Soft-assign each training image to K BiomedCLIP-feature prototypes (built "
                "offline via build_prototype_centroids.py) and ADD the weighted prototype "
                "embedding to every image-position embedding before the LM. No prepended "
                "tokens, no RoPE shift, no label dependency."
            )
        },
    )
    prototype_centroids_path: str = field(
        default="data/prototype_centroids.pt",
        metadata={"help": "Path to centroids .pt produced by build_prototype_centroids.py."},
    )
    cond_temperature: float = field(
        default=0.1,
        metadata={
            "help": "Temperature for softmax over cosine sims to centroids. Smaller = sharper "
                    "(more like hard assignment); larger = softer mixing of prototypes."
        },
    )
    # ---- Text -> prototype head (bridges train/inference for prototype cond) ----
    use_text_to_proto: bool = field(
        default=False,
        metadata={
            "help": (
                "Attach an MLP head that maps pooled caption hidden states to a "
                "K-dim distribution over prototypes (w_text). Trained with an "
                "auxiliary KL(w_text || w_image.detach()) loss. At inference, "
                "w_text replaces w_image (no image needed) to build the prototype "
                "conditioning vector."
            )
        },
    )
    text_to_proto_aux_weight: float = field(
        default=1.0,
        metadata={"help": "Weight on the auxiliary KL(w_text || w_image.detach()) loss."},
    )
    warm_start_checkpoint: str = field(
        default="",
        metadata={
            "help": (
                "Path to a prior Stage-2 checkpoint dir (containing "
                "adapter_model.safetensors). When set, LoRA + gen_head + "
                "gen_aligner weights are loaded via load_state_dict(strict=False) "
                "AFTER PEFT wrapping; new modules (e.g. prototype_emb, text_to_proto) "
                "stay at their fresh init. Optimizer/step counter are NOT restored."
            )
        },
    )

    caption_source: str = field(
        default="self_distill",
        metadata={
            "help": (
                "Where the T2I prompt caption comes from each step. "
                "'self_distill': run i2t on the image and use the model's own caption (current default). "
                "'original': read 'Original_Caption' from the dataset row (real PubMed caption)."
            ),
            "choices": ["self_distill", "original"],
        },
    )
    caption_column: str = field(
        default="Original_Caption",
        metadata={"help": "Column name to read when caption_source='original'."},
    )



def _build_stage2_peft_config(model_args, script_args=None):
    """Build a LoraConfig from TRL's ModelConfig, overriding two things:

    - task_type=None so PEFT uses the generic PeftModel wrapper.
      PeftModelForCausalLM (TRL's default) would assume a stock causal-LM
      forward signature that MultiModalityCausalLM does not have.
    - modules_to_save defaults to ["gen_head","gen_aligner"] when unset,
      so Stage 1's gen-side fine-tune stays alive on top of LoRA.
    """
    if not getattr(model_args, "use_peft", False):
        return None
    from peft import LoraConfig

    target_modules = model_args.lora_target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    modules_to_save = list(model_args.lora_modules_to_save or ["gen_head", "gen_aligner"])
    if script_args is not None and getattr(script_args, "use_prototype_conditioning", False):
        if "prototype_emb" not in modules_to_save:
            modules_to_save.append("prototype_emb")
    # NOTE: text_to_proto is intentionally NOT added to modules_to_save. PEFT's
    # ModulesToSaveWrapper triggers DDP "marked ready twice" because the head is
    # called via `unwrapped.text_to_proto(...)` outside the DDP-wrapped forward.
    # We attach it as a plain submodule AFTER PEFT wrapping instead.

    return LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save),
        bias="none",
        task_type=None,
    )


def main(script_args, training_args, model_args, max_samples=None):
    preprocess_start = time.perf_counter()

    if 'PubMedVision' in script_args.dataset_name:
        dataset = load_dataset("json", data_files=os.path.join(script_args.data_dir, script_args.dataset_name))
        img_key = "image"
    else:
        dataset = load_dataset("parquet", data_files=script_args.dataset_name)
        img_key = "image_path"

    # Optionally limit dataset size for debugging
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))

    # Resolve image paths and filter out missing images
    data_dir = script_args.data_dir

    def resolve_image_path(example, img_key="image"):

        if isinstance(example[img_key], list):
            img_path = os.path.join(data_dir, example[img_key][0]) if data_dir else example[img_key][0]
        elif isinstance(example[img_key], str):
            img_path = os.path.join(data_dir, "images", example[img_key]) if data_dir else os.path.join("images", example[img_key])
        else:
            raise ValueError(f"Unexpected type for image path: {type(example[img_key])}")
        example["image"] = img_path
        return example
    
    def add_dummy_prompt(example):
        example["prompt"] = "Describe the main content of the image in one sentence."
        return example

    dataset = dataset.map(resolve_image_path, fn_kwargs={"img_key": img_key})
    dataset = dataset.map(add_dummy_prompt)
    # Filter out rows where the image file is missing
    dataset = dataset.filter(lambda x: os.path.exists(x["image"]))

    # Attribute sidecar: only used as a grid filter (drop is_grid=='multi' rows).
    # The labeled modality/pose conditioning path has been removed.
    if script_args.attribute_sidecar:
        with open(script_args.attribute_sidecar) as _f:
            _side = {r["id"]: r for r in json.load(_f)}
        print(f"[sidecar] loaded {len(_side)} rows from {script_args.attribute_sidecar}")

        def _attach_grid(example):
            s = _side.get(example.get("id"), {}) if "id" in example else {}
            example["is_grid"] = s.get("is_grid")
            return example

        for split in dataset:
            before = len(dataset[split])
            dataset[split] = dataset[split].map(_attach_grid)
            dataset[split] = dataset[split].filter(lambda x: x.get("is_grid") != "multi")
            after = len(dataset[split])
            print(f"[sidecar] split={split} {before} -> {after} after dropping is_grid=='multi' "
                  f"({100*(before-after)/max(before,1):.1f}% removed)")

    # Drop held-out test IDs so train/test stay disjoint.
    if script_args.exclude_ids_json:
        with open(script_args.exclude_ids_json) as _f:
            _raw = json.load(_f)
        if _raw and isinstance(_raw[0], dict):
            _excl = {r["id"] for r in _raw if "id" in r}
        else:
            _excl = set(_raw)
        print(f"[exclude] loaded {len(_excl)} ids from {script_args.exclude_ids_json}")
        for split in dataset:
            before = len(dataset[split])
            dataset[split] = dataset[split].filter(lambda x: x.get("id") not in _excl)
            after = len(dataset[split])
            print(f"[exclude] split={split} {before} -> {after} "
                  f"({before-after} test ids removed)")

    print(f"[timing] dataset preprocessing took {time.perf_counter() - preprocess_start:.2f}s")

    trainer_cls = SFTAlignmentTrainer

    print("using: ", trainer_cls)

    # Stage-2 LoRA config (None when --use_peft is not passed, i.e. Stage 1).
    peft_config = _build_stage2_peft_config(model_args, script_args=script_args)

    # Initialize the GRPO trainer
    train_start = time.perf_counter()
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[
            script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        # callbacks=[ParameterInfoCallback()],
        task_args=script_args,
    )

    trainer.train()
    print(f"[timing] trainer.train() took {time.perf_counter() - train_start:.2f}s")

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))

    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args, max_samples=script_args.max_samples)
