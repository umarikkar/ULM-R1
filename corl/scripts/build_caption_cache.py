"""One-time caption cache builder.

For every image in PubMedVision_Original_Caption.json, runs Janus-Pro i2t with
do_sample=True and num_return_sequences=K, then writes (id, image, captions[K])
to a per-rank JSONL file. After all ranks finish, run with --merge to produce
the final consolidated JSON that the trainer reads.

Launch (8 GPUs):
    bash corl/scripts/build_caption_cache.sh

Merge after run:
    python corl/scripts/build_caption_cache.py --merge \
        --out_dir /path/to/cache_dir \
        --merged_out /path/to/PubMedVision_CachedCaptions.json
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


TASK_INSTRUCT = (
    "Describe this medical image in one to two sentences. Describe only "
    "what is directly visible to reconstruct the image from the description alone."
)


def _fix_janus_text(s: str) -> str:
    return (
        s.replace("Ġ", " ")
         .replace("Ċ", "\n")
         .strip()
    )


def _build_inputs(processor, images, device):
    convs = [
        [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{TASK_INSTRUCT}"},
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

    # ---- Resume support: skip rows already in the shard ----
    done_ids = set()
    if shard_path.exists():
        with open(shard_path, "r") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
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

            with torch.inference_mode():
                prepared = _build_inputs(processor, imgs, device)
                inputs_embeds = model.prepare_inputs_embeds(**prepared)
                out_ids = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepared.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.k,
                    pad_token_id=eos_id,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                )
            # out_ids shape: [B*K, T]. Decode and group.
            decoded = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for i, row in enumerate(kept):
                caps = [_fix_janus_text(decoded[i * args.k + k]) for k in range(args.k)]
                rec = {
                    "id": row["id"],
                    "image": row["image"],
                    "cached_captions": caps,
                }
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
    ap.add_argument("--k", type=int, default=4,
                    help="Captions per image (num_return_sequences)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
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
