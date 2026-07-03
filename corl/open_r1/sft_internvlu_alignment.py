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

# InternVL-U T2I flow-matching training entry point. Mirrors
# sft_janus_alignment.py but trains the InternVL-U backbone: LoRA on the LLM,
# frozen DiT + VAE, rectified-flow velocity MSE on the GT image's VAE latent.

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from trl import ModelConfig, SFTConfig, ScriptArguments, TrlParser

from corl.open_r1.trainer.sft_trainer_alignment_internvlu import (
    SFTInternVLUAlignmentTrainer,
)


@dataclass
class SFTScriptArguments(ScriptArguments):
    """Script arguments for the InternVL-U T2I training script."""

    data_dir: str = field(
        default="",
        metadata={"help": "Base directory joined with the dataset's image_path column."},
    )
    dataset_cache_dir: str = field(
        default=os.environ.get("HF_DATASETS_CACHE", None),
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on samples per split for debugging runs."},
    )

    gen_image_size: int = field(
        default=512,
        metadata={"help": "Square generation/target resolution (pixels)."},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Max token length for the (left-truncated) T2I prompt."},
    )
    prompt_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Per-sample prob of using the unconditional prompt (CFG training)."},
    )

    caption_source: str = field(
        default="original",
        metadata={
            "help": "'original' reads caption_column from the row; 'self_distill' "
                    "captions the image with the model itself each step.",
            "choices": ["original", "self_distill"],
        },
    )
    caption_column: str = field(
        default="detailed_caption",
        metadata={"help": "Column read when caption_source='original'."},
    )
    i2t_question: str = field(
        default="What type of medical image is this? Provide enough detail to reconstruct the image faithfully.",
        metadata={"help": "Prompt used to self-distill a caption when caption_source='self_distill'."},
    )
    i2t_max_new_tokens: int = field(
        default=96,
        metadata={"help": "Max new tokens for self-distilled captions."},
    )

    train_decoder_projector: bool = field(
        default=False,
        metadata={
            "help": "Also train the DiT's decoder_projector (the h_text->DiT bridge, "
                    "analogous to Janus's gen_aligner). Kept in modules_to_save."
        },
    )

    image_column: str = field(
        default="image_path",
        metadata={"help": "Dataset column holding the relative image path."},
    )


def _build_stage2_peft_config(model_args, script_args=None):
    """LoRA on the LLM. task_type=None because the optimized module is a custom
    flow-matching wrapper, not a stock causal LM."""
    if not getattr(model_args, "use_peft", False):
        return None
    from peft import LoraConfig

    target_modules = model_args.lora_target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    modules_to_save = list(model_args.lora_modules_to_save or [])
    if script_args is not None and getattr(script_args, "train_decoder_projector", False):
        if "decoder_projector" not in modules_to_save:
            modules_to_save.append("decoder_projector")

    return LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save) or None,
        bias="none",
        task_type=None,
    )


def main(script_args, training_args, model_args, max_samples=None):
    preprocess_start = time.perf_counter()

    data_dir = script_args.data_dir

    if 'PubMedVision' in script_args.dataset_name:
        dataset = load_dataset("json", data_files=os.path.join(data_dir, script_args.dataset_name))
        img_key = "image"
    else:
        dataset = load_dataset("parquet", data_files=script_args.dataset_name)
        img_key = script_args.image_column

    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))

    def resolve_image_path(example):
        rel = example[img_key]
        if isinstance(rel, (list, tuple)):
            rel = rel[0]
        example["image"] = os.path.join(data_dir, rel) if data_dir else rel
        return example

    dataset = dataset.map(resolve_image_path)
    dataset = dataset.filter(lambda x: os.path.exists(x["image"]))
    print(f"[timing] dataset preprocessing took {time.perf_counter() - preprocess_start:.2f}s")

    peft_config = _build_stage2_peft_config(model_args, script_args=script_args)

    train_start = time.perf_counter()
    trainer = SFTInternVLUAlignmentTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no" else None
        ),
        peft_config=peft_config,
        task_args=script_args,
    )

    trainer.train()
    print(f"[timing] trainer.train() took {time.perf_counter() - train_start:.2f}s")

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, max_samples=script_args.max_samples)
