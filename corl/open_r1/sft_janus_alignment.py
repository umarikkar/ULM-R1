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


def main(script_args, training_args, model_args, max_samples=None):
    preprocess_start = time.perf_counter()

    dataset = load_dataset("json", data_files=os.path.join(script_args.data_dir, script_args.dataset_name))

    # Optionally limit dataset size for debugging
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))

    # Resolve image paths and filter out missing images
    data_dir = script_args.data_dir

    def resolve_image_path(example):
        img_path = os.path.join(data_dir, example["image"][0]) if data_dir else example["image"][0]
        example["image"] = img_path
        return example
    
    def add_dummy_prompt(example):
        example["prompt"] = "Describe the main content of the image in one sentence."
        return example

    dataset = dataset.map(resolve_image_path)
    dataset = dataset.map(add_dummy_prompt)
    # Filter out rows where the image file is missing
    dataset = dataset.filter(lambda x: os.path.exists(x["image"]))

    print(f"[timing] dataset preprocessing took {time.perf_counter() - preprocess_start:.2f}s")

    trainer_cls = SFTAlignmentTrainer

    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    train_start = time.perf_counter()
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[
            script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
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
