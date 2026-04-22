"""
Precompute and cache the mapped GRPO dataset.

This script resolves image paths, filters missing files, optionally decodes
images to RGB, builds the conversation fields used by the trainer, and saves
the processed dataset to disk so training can reuse it without remapping every
run.

Example:
  python corl/scripts/precompute_mapped_dataset.py \
    --dataset /projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama.parquet \
    --image-base-dir /projects/u6gd/datasets/PubMedVision/images \
    --output-dir /projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama_mapped

Shard one chunk at a time (good for parallel workers or resumable runs):
    python corl/scripts/precompute_mapped_dataset.py \
        --num-shards 8 \
        --shard-index 0 \
        --output-dir data/t2i_midlevel_llama_mapped
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from datasets import DatasetDict, Image as HFImage, load_dataset
from PIL import Image


DEFAULT_IMAGE_BASE_DIR = "/projects/u6gd/datasets/PubMedVision/images"
RETINA03_IMAGE_BASE_DIR = "/work/um00109/MLLM/datasets/PubMedVision/images"
RETINA03_HOSTNAME = "cvssp-retina03"
HOSTNAME = os.uname().nodename.split(".")[0]
DEFAULT_RESOLVED_IMAGE_BASE_DIR = (
    RETINA03_IMAGE_BASE_DIR if HOSTNAME == RETINA03_HOSTNAME else DEFAULT_IMAGE_BASE_DIR
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and save mapped dataset with images loaded.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/t2i_midlevel_llama.parquet",
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
        default=DEFAULT_RESOLVED_IMAGE_BASE_DIR,
        help="Base directory used to resolve image_path values.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/t2i_midlevel_llama_mapped",
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
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split each dataset split into N shards and process only one shard per run.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to process when --num-shards > 1.",
    )
    parser.add_argument(
        "--save-max-shard-size",
        type=str,
        default="1GB",
        help="Max on-disk shard size passed to save_to_disk (e.g. 500MB, 1GB).",
    )
    parser.add_argument(
        "--lazy-image-loading",
        action="store_true",
        default=True,
        help="Keep image paths only (do not decode PIL images) for faster/lower-memory precompute.",
    )
    parser.add_argument(
        "--eager-image-loading",
        action="store_true",
        help="Decode images to RGB and store them in the saved dataset (slower, higher CPU/memory).",
    )
    parser.add_argument(
        "--process-all-shards",
        action="store_true",
        help="Process shard indices [0, num_shards) sequentially, mapping and saving each shard independently.",
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


def _process_single_shard(args: argparse.Namespace, shard_index: int) -> str:
    dataset = _load_dataset(args.dataset, args.dataset_config)

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

    if args.num_shards > 1:
        for split in splits:
            dataset[split] = dataset[split].shard(
                num_shards=args.num_shards,
                index=shard_index,
                contiguous=True,
            )

    for split in splits:
        if "image_path" in dataset[split].column_names:
            dataset[split] = dataset[split].map(lambda ex: _resolve_image_path(ex, args.image_base_dir))
            dataset[split] = dataset[split].filter(lambda ex: os.path.exists(ex["image_full_path"]))
            if args.eager_image_loading:
                dataset[split] = dataset[split].map(_load_image)
        elif "image" in dataset[split].column_names:
            dataset[split] = dataset[split].cast_column("image", HFImage())
            if args.eager_image_loading:
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

    output_dir = args.output_dir
    if args.num_shards > 1:
        output_dir = os.path.join(
            args.output_dir,
            f"shard_{shard_index:04d}_of_{args.num_shards:04d}",
        )

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir, max_shard_size=args.save_max_shard_size)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.eager_image_loading:
        args.lazy_image_loading = False
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    if args.process_all_shards:
        for shard_index in range(args.num_shards):
            print(f"[shard] processing {shard_index + 1}/{args.num_shards}")
            output_dir = _process_single_shard(args, shard_index)
            print(f"Saved mapped dataset to {output_dir}")
    else:
        output_dir = _process_single_shard(args, args.shard_index)
        print(f"Saved mapped dataset to {output_dir}")


if __name__ == "__main__":
    main()