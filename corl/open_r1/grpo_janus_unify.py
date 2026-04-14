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
from typing import Optional
from datasets import load_dataset, Image as HFImage
from PIL import Image

from transformers import TrainerCallback
from trl import (
    GRPOConfig, ModelConfig, ScriptArguments,
    TrlParser, get_peft_config
)
from corl.open_r1.trainer.grpo_trainer_unified import JanusProUnifiedGRPOTrainer
from corl.open_r1.rewards import reward_funcs_registry


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
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

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
    unify_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to delete unused modules in janus."},
    )
    unify_reward: bool = field(
        default=True,
        metadata={"help": "Whether to delete unused modules in janus."},
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
    blip_model_ckpt: str = field(
        default="./checkpoint/blip-image-captioning-base"
    )
    dataset_cache_dir: str = field(
        default=os.environ.get("HF_DATASETS_CACHE", None),
    )
    image_base_dir: str = field(
        default="",
        metadata={"help": "Base directory for image_path column in parquet datasets."},
    )


def main(script_args, training_args, model_args, max_samples=None):
    if script_args.dataset_name.endswith(".parquet") or os.path.isdir(script_args.dataset_name):
        data_files = (
            os.path.join(script_args.dataset_name, "*.parquet")
            if os.path.isdir(script_args.dataset_name)
            else script_args.dataset_name
        )
        dataset = load_dataset("parquet", data_files=data_files, cache_dir=script_args.dataset_cache_dir)
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            cache_dir=script_args.dataset_cache_dir,
        )

    # Drop QA-dependent rewards if dataset has no qa_type column
    train_cols = dataset[script_args.dataset_train_split].column_names
    requested_rewards = list(script_args.reward_funcs)
    if "qa_type" not in train_cols and "qa_accuracy" in requested_rewards:
        print("[reward] no 'qa_type' column in dataset — dropping 'qa_accuracy' reward")
        requested_rewards = [r for r in requested_rewards if r != "qa_accuracy"]
    reward_funcs = [reward_funcs_registry[func] for func in requested_rewards]

    # Optionally limit dataset size for debugging
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))

    # Resolve image paths and filter out missing images
    image_base_dir = script_args.image_base_dir

    if "image_path" in dataset[script_args.dataset_train_split].column_names:
        def resolve_image_path(example):
            img_path = os.path.join(image_base_dir, example["image_path"]) if image_base_dir else example["image_path"]
            example["image_full_path"] = img_path
            return example

        dataset = dataset.map(resolve_image_path)
        # Filter out rows where the image file is missing
        dataset = dataset.filter(lambda x: os.path.exists(x["image_full_path"]))

        def load_image(example):
            example["image"] = Image.open(example["image_full_path"]).convert("RGB")
            return example

        dataset = dataset.map(load_image)
    elif "image" in dataset[script_args.dataset_train_split].column_names:
        dataset = dataset.cast_column("image", HFImage())

        def ensure_rgb(example):
            example["image"] = example["image"].convert("RGB")
            return example

        dataset = dataset.map(ensure_rgb)

    # Format into conversation
    def make_conversation_t2i(example):
        return {
            "prompt": [
                {
                    "role": "<|User|>",
                    "content": f"{example['detailed_caption'].strip()}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        }

    def make_conversation_mm2t(example):
        if script_args.mm2t_format == 'qa':
            question = example["qa_problem"]
        elif script_args.mm2t_format == 'oc':
            question = example["cls_problem"]
        elif script_args.mm2t_format == 'od':
            question = example["od_problem"]
        else:
            question = example["problem"]

        return {
            "qa_prompt": [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        }

    def make_conversation_joint(example):
        return {
            "prompt": [
                {
                    "role": "<|User|>",
                    "content": f"{example['detailed_caption'].strip()}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
            "qa_prompt": [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{example['caption']}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
            "qa_solution": example["caption"],
        }

    if script_args.task_format == "t2i":
        dataset = dataset.map(make_conversation_t2i)
    elif script_args.task_format == "mm2t":
        dataset = dataset.map(make_conversation_mm2t)
        # dataset = dataset.remove_columns(["type"])
    else:
        dataset = dataset.map(make_conversation_joint)

    trainer_cls = JanusProUnifiedGRPOTrainer

    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[
            script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        # callbacks=[ParameterInfoCallback()],
        task_args=script_args,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))

    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)
