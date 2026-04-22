"""
Precompute and cache the mapped GRPO dataset with eagerly loaded images.

This script resolves image paths, filters missing files, decodes images to RGB,
builds the conversation fields used by the trainer, and saves the processed
dataset to disk so training can reuse it without remapping every run.

Example:
  python corl/scripts/precompute_mapped_dataset.py \
    --dataset /projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama.parquet \
    --image-base-dir /projects/u6gd/datasets/PubMedVision/images \
    --output-dir /projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama_mapped
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from datasets import DatasetDict, Image as HFImage, load_dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and save mapped dataset with images loaded.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Parquet file, parquet directory, or dataset name to load.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset config name when loading from the Hub.",
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default="",
        help="Base directory used to resolve image_path values.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the processed dataset will be written with save_to_disk().",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging or quick precompute runs.",
    )
    parser.add_argument(
        "--dataset-train-split",
        type=str,
        default="train",
        help="Split name used to inspect the schema (default: train).",
    )
    parser.add_argument(
        "--dataset-test-split",
        type=str,
        default="test",
        help="Split name to keep when present.",
    )
    parser.add_argument(
        "--task-format",
        type=str,
        default="unify",
        choices=["t2i", "mm2t", "joint", "unify"],
        help="Conversation format to materialize into the dataset.",
    )
    parser.add_argument(
        "--mm2t-format",
        type=str,
        default="qa",
        choices=["qa", "oc", "od", "problem"],
        help="Which question column to use for mm2t-style prompts.",
    )
    return parser.parse_args()


def _load_dataset(dataset_path: str, dataset_config: Optional[str]):
    if dataset_path.endswith(".parquet") or os.path.isdir(dataset_path):
        data_files = (
            os.path.join(dataset_path, "*.parquet")
            if os.path.isdir(dataset_path)
            else dataset_path
        )
        return load_dataset("parquet", data_files=data_files)

    return load_dataset(dataset_path, name=dataset_config)


def _resolve_image_path(example: dict, image_base_dir: str) -> dict:
    img_path = example["image_path"]
    if image_base_dir:
        img_path = os.path.join(image_base_dir, img_path)
    example["image_full_path"] = img_path
    return example


def _load_image(example: dict) -> dict:
    example["image"] = Image.open(example["image_full_path"]).convert("RGB")
    return example


def _make_conversation_t2i(example: dict) -> dict:
    return {
        "prompt": [
            {
                "role": "<|User|>",
                "content": f"{example['detailed_caption'].strip()}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ],
    }


def _make_conversation_mm2t(example: dict, mm2t_format: str) -> dict:
    if mm2t_format == "qa":
        question = example["qa_problem"]
    elif mm2t_format == "oc":
        question = example["cls_problem"]
    elif mm2t_format == "od":
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


def _make_conversation_joint(example: dict) -> dict:
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


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(args.dataset, args.dataset_config)

    if isinstance(dataset, DatasetDict) and args.dataset_train_split in dataset:
        train_cols = dataset[args.dataset_train_split].column_names
    else:
        train_cols = next(iter(dataset.values())).column_names if isinstance(dataset, DatasetDict) else dataset.column_names

    if args.max_samples is not None:
        if isinstance(dataset, DatasetDict):
            for split in dataset:
                if len(dataset[split]) > args.max_samples:
                    dataset[split] = dataset[split].select(range(args.max_samples))
        else:
            if len(dataset) > args.max_samples:
                dataset = dataset.select(range(args.max_samples))

    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
    else:
        splits = ["train"]
        dataset = DatasetDict({"train": dataset})

    for split in splits:
        if "image_path" in dataset[split].column_names:
            dataset[split] = dataset[split].map(lambda ex: _resolve_image_path(ex, args.image_base_dir))
            dataset[split] = dataset[split].filter(lambda ex: os.path.exists(ex["image_full_path"]))
            dataset[split] = dataset[split].map(_load_image)
        elif "image" in dataset[split].column_names:
            dataset[split] = dataset[split].cast_column("image", HFImage())
            dataset[split] = dataset[split].map(lambda ex: {**ex, "image": ex["image"].convert("RGB")})

    # Keep the same conversation fields expected by training.
    if args.task_format == "t2i":
        mapper = _make_conversation_t2i
    elif args.task_format == "mm2t":
        mapper = lambda ex: _make_conversation_mm2t(ex, args.mm2t_format)
    else:
        mapper = _make_conversation_joint

    for split in splits:
        dataset[split] = dataset[split].map(mapper)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Saved mapped dataset to {args.output_dir}")


if __name__ == "__main__":
    main()