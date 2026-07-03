"""Granularity-leveled caption cache builder.

For every image in PubMedVision_Original_Caption.json this writes, per image,
    {id, image, cached_captions_l1_meta, cached_captions_l1,
     cached_captions_l2, cached_captions_l3}

The granularity levels (coarse -> fine):
    l1_meta : clean templated label from the row's modality/body_part fields.
    l1      : a few words (modality + main anatomy), image-conditioned.
    l2      : one sentence (modality + anatomy + most salient finding).
    l3      : two-three sentences, full visible detail.

All three model levels are generated *from the image* with level-specific
prompts. An earlier design derived L1/L2 by text-only summarization of L3, but
Janus-Pro-1B degenerates into repetition loops on ~half of inputs when
summarizing text; image-conditioned generation stays on-distribution and is
robust. l1_meta is a zero-noise label-style alternative to the prose l1, kept
alongside so the cleaner variant can be chosen at train time.

Greedy decoding, one caption per level, stored as plain strings. Each column is
drop-in compatible with the trainer's caption_column reader; set
caption_column="cached_captions_l1" (l1_meta / l2 / l3) to train on a level.

Launch (multi-GPU):
    bash corl/scripts/build_caption_cache_levels.sh

Merge after run:
    python corl/scripts/build_caption_cache_levels.py --merge \
        --out_dir /path/to/cache_dir \
        --merged_out /path/to/PubMedVision_CachedCaptions_Levels.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForCausalLM

# Project root on path so `from janus...` works.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from janus.models import VLChatProcessor


LEVEL_KEYS = [
    "cached_captions_l1_meta",
    "cached_captions_l1",
    "cached_captions_l2",
    "cached_captions_l3",
]

# Image-conditioned prompts, coarse -> fine. The (column, instruct, max_new_tokens)
# tuples are generated in order, one greedy pass each.
IMG_LEVELS = [
    (
        "cached_captions_l1",
        "Describe this image in a single sentence that names the type of scan "
        "and the part of the body or object shown. Do not describe any findings "
        "or abnormalities.",
        48,
    ),
    (
        "cached_captions_l2",
        "Summarize this image in one sentence, covering the scan type, the body "
        "region or anatomy, and the main finding. Describe only what is "
        "directly visible.",
        64,
    ),
    (
        "cached_captions_l3",
        "Describe this medical image in two to three sentences: the imaging "
        "modality, the anatomy, all visible findings, their locations, and "
        "notable visual attributes. Describe only what is directly visible to "
        "reconstruct the image from the description alone.",
        160,
    ),
]


def _fix_janus_text(s: str) -> str:
    return (
        s.replace("Ġ", " ")
         .replace("Ċ", "\n")
         .strip()
    )


def _first_sentence(s: str) -> str:
    """Keep L1 to a single clean sentence: drop list scaffolding, trailing text."""
    s = _fix_janus_text(s)
    # First non-empty line, stripping any leading enumeration ("1.", "-", "*").
    line = ""
    for cand in s.splitlines():
        cand = cand.strip().lstrip("0123456789.)-* ").strip()
        if cand:
            line = cand
            break
    # Cut at the end of the first sentence (keep the terminator).
    for i, ch in enumerate(line):
        if ch in ".!?":
            return line[: i + 1].strip()
    return line.strip()


def _meta_l1(row) -> str:
    """Templated label-style L1 from the dataset's modality/body_part fields."""
    mod = (row.get("modality") or "").strip()
    bp = (row.get("body_part") or "").strip()
    parts = [p for p in (mod, bp) if p and p.lower() != "others"]
    return ", ".join(parts) if parts else "Medical image"


def _build_image_inputs(processor, images, instruct, device):
    convs = [
        [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{instruct}"},
            {"role": "<|Assistant|>", "content": ""},
        ]
        for _ in images
    ]
    prepared = processor(
        conversations=convs,
        images=[[img] for img in images],
        force_batchify=True,
    ).to(device)
    return prepared


def main_worker(args):
    # ---- DDP setup ----
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    # ---- Load dataset ----
    with open(args.data_json, "r") as f:
        rows = json.load(f)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    # Shard rows by rank (contiguous slices for cache friendliness).
    n = len(rows)
    per_rank = (n + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, n)
    my_rows = rows[start:end]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_rank{rank:02d}.jsonl"

    # ---- Resume support: a row is done only if ALL levels are present ----
    done_ids = set()
    if shard_path.exists():
        with open(shard_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if all(rec.get(k) for k in LEVEL_KEYS):
                        done_ids.add(rec["id"])
                except Exception:
                    pass
    print(f"[rank {rank}] {len(my_rows)} rows to process "
          f"({len(done_ids)} already done in {shard_path.name})", flush=True)

    # ---- Load model + processor ----
    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    eos_id = processor.tokenizer.eos_token_id
    bos_id = processor.tokenizer.bos_token_id

    gen_kw = dict(
        do_sample=False,
        pad_token_id=eos_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )

    # ---- Iterate in batches ----
    todo = [r for r in my_rows if r["id"] not in done_ids]
    t0 = time.perf_counter()
    n_done_session = 0
    with open(shard_path, "a") as out_f:
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start: batch_start + args.batch_size]
            # Resolve image paths and skip-and-log missing files.
            imgs, kept = [], []
            for r in batch:
                p = os.path.join(args.data_dir, r["image"][0])
                if not os.path.exists(p):
                    continue
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    kept.append(r)
                except Exception as e:
                    print(f"[rank {rank}] skip {p}: {e}", flush=True)
            if not kept:
                continue

            # One image-conditioned greedy caption per granularity level.
            per_level = {}  # column -> list[str] aligned with kept
            with torch.inference_mode():
                for col, instruct, max_new in IMG_LEVELS:
                    prepared = _build_image_inputs(processor, imgs, instruct, device)
                    inputs_embeds = model.prepare_inputs_embeds(**prepared)
                    out_ids = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepared.attention_mask,
                        max_new_tokens=max_new, **gen_kw,
                    )
                    decoded = processor.tokenizer.batch_decode(
                        out_ids, skip_special_tokens=True)
                    if col == "cached_captions_l1":
                        per_level[col] = [_first_sentence(d) for d in decoded]
                    else:
                        per_level[col] = [_fix_janus_text(d) for d in decoded]

            for i, row in enumerate(kept):
                rec = {"id": row["id"], "image": row["image"],
                       "cached_captions_l1_meta": _meta_l1(row)}
                for col, _, _ in IMG_LEVELS:
                    rec[col] = per_level[col][i]
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done_session += len(kept)

            if (batch_start // args.batch_size) % 5 == 0:
                elapsed = time.perf_counter() - t0
                rate = n_done_session / max(elapsed, 1e-3)
                remaining = (len(todo) - n_done_session) / max(rate, 1e-3)
                print(
                    f"[rank {rank}] {n_done_session}/{len(todo)} "
                    f"({rate:.2f} img/s, ETA {remaining / 3600:.2f}h)",
                    flush=True,
                )

    print(f"[rank {rank}] finished, wrote {shard_path}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main_merge(args):
    """Combine per-rank JSONL shards into a single JSON for the trainer."""
    shards = sorted(Path(args.out_dir).glob("shard_rank*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {args.out_dir}")
    all_rows = []
    seen = set()
    for s in shards:
        with open(s, "r") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] in seen:
                    continue
                if not all(rec.get(k) for k in LEVEL_KEYS):
                    continue  # skip partially-written rows
                seen.add(rec["id"])
                all_rows.append(rec)
    out = Path(args.merged_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_rows, f, ensure_ascii=False)
    print(f"Merged {len(all_rows)} rows from {len(shards)} shards -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--data_json", required=False,
                    help="Path to PubMedVision_Original_Caption.json")
    ap.add_argument("--data_dir", required=False,
                    help="Base dir for relative image paths in the JSON")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write per-rank shards into")
    ap.add_argument("--merged_out", default=None,
                    help="Output path for the merged JSON (used with --merge)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Optional cap for debugging")
    ap.add_argument("--merge", action="store_true",
                    help="Run the merge step instead of generation")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.merge:
        if args.merged_out is None:
            raise SystemExit("--merge requires --merged_out")
        main_merge(args)
    else:
        if not args.data_json or not args.data_dir:
            raise SystemExit("--data_json and --data_dir are required for generation")
        main_worker(args)
