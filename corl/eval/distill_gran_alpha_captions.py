"""Distill random-granularity captions with the frozen gran-LoRA (no grad).

For each training image we sample a random LoRA scale alpha and caption the image
with gran_lora_v2 under the fixed neutral prompt. The captions are written as a
new column in a drop-in copy of the dataset JSON, so the existing T2I trainer can
consume them via `--caption_source original --caption_column <col>`.

Why a two-pass cache (and not on-the-fly in the T2I loop): the T2I run is **1
epoch**, so every image is seen once -> "random alpha per iteration" == "one
random alpha per image", which is exactly what this produces. Caching it makes
the captions inspectable, restartable, and shardable across GPUs, and keeps the
captioner (i2t, gran-LoRA on language_model) out of the T2I trainer's graph.

alpha is drawn per generation-batch (rows are shuffled first, so each image still
gets an effectively-random granularity); batching needs one LoRA scale per
forward. Usable band for gran_lora_v2 is [0.3, 0.9] (a 3-step staircase:
~[0.3,0.45]=l1, ~[0.45,0.75]=l2, ~[0.75,0.9]=l3). Use --alphas_discrete to draw
from fixed anchors instead of the continuous band.

Shardable:  CUDA_VISIBLE_DEVICES=k python ... --num_shards N --shard_id k
then merge the shard files (see distill_gran_alpha_captions.sh).

    python corl/eval/distill_gran_alpha_captions.py \
        --adapter_dir results/GranLoRA/gran_lora_v2 \
        --data_json  $DATA_DIR/PubMedVision_CachedCaptions_Levels.json \
        --data_dir   $DATA_DIR \
        --exclude_ids_json corl/eval/test_split.json \
        --out_json   $DATA_DIR/PubMedVision_RandAlpha.json \
        --alpha_lo 0.3 --alpha_hi 0.9 --gen_batch 8
"""
import argparse
import json
import os
import random
import sys

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from janus.models import VLChatProcessor
from corl.open_r1.janus_tokenizer_fix import load_fast_tokenizer

NEUTRAL_PROMPT = "Describe this medical image."
CAP_COL = "cached_captions_randalpha"
A_COL = "randalpha"


def _fix(s):
    return s.replace("Ġ", " ").replace("Ċ", "\n").strip()


def set_scale(model, base, alpha):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                m.scaling[adp] = base[(name, adp)] * alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--exclude_ids_json", default="")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--alpha_lo", type=float, default=0.3)
    ap.add_argument("--alpha_hi", type=float, default=0.9)
    ap.add_argument("--alphas_discrete", default="",
                    help="If set (e.g. '0.3,0.6,0.9'), draw alpha from this set "
                         "instead of continuous U(alpha_lo, alpha_hi).")
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--do_sample", action="store_true",
                    help="temperature-sample captions (adds caption diversity); "
                         "default greedy (deterministic given alpha).")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    rng = random.Random(args.seed + 1000 * args.shard_id)
    discrete = [float(x) for x in args.alphas_discrete.split(",")] if args.alphas_discrete else None

    rows = json.load(open(args.data_json))
    if args.exclude_ids_json:
        raw = json.load(open(args.exclude_ids_json))
        excl = {r["id"] if isinstance(r, dict) else r for r in raw}
        rows = [r for r in rows if r.get("id") not in excl]
    if args.max_samples:
        rows = rows[: args.max_samples]
    # deterministic global shuffle (so per-batch alpha == effectively per-image)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    # this shard's slice of the shuffled order
    mine = order[args.shard_id::args.num_shards]
    print(f"[distill] shard {args.shard_id}/{args.num_shards}: {len(mine)} of {len(rows)} rows "
          f"| alpha={'{'+args.alphas_discrete+'}' if discrete else f'U({args.alpha_lo},{args.alpha_hi})'} "
          f"| {'sampled' if args.do_sample else 'greedy'}", flush=True)

    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    processor.tokenizer = load_fast_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter_dir)
    model = model.to(device).eval()
    base = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                base[(name, adp)] = float(m.scaling[adp])
    tok = processor.tokenizer
    eos = tok.eos_token_id

    def gen(imgs, alpha):
        set_scale(model, base, alpha)
        convs = [[{"role": "<|User|>", "content": f"<image_placeholder>\n{NEUTRAL_PROMPT}"},
                  {"role": "<|Assistant|>", "content": ""}] for _ in imgs]
        prep = processor(conversations=convs, images=[[im] for im in imgs],
                         force_batchify=True).to(device)
        with torch.inference_mode():
            emb = model.prepare_inputs_embeds(
                input_ids=prep.input_ids, pixel_values=prep.pixel_values,
                images_seq_mask=prep.images_seq_mask, images_emb_mask=prep.images_emb_mask)
            out = model.language_model.generate(
                inputs_embeds=emb, attention_mask=prep.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample, temperature=args.temperature if args.do_sample else None,
                pad_token_id=eos, bos_token_id=tok.bos_token_id, eos_token_id=eos)
        return [_fix(d) for d in tok.batch_decode(out, skip_special_tokens=True)]

    out_rows, done, n = [], 0, len(mine)
    for i in range(0, n, args.gen_batch):
        idxs = mine[i:i + args.gen_batch]
        batch = [rows[j] for j in idxs]
        paths, keep = [], []
        for r, j in zip(batch, idxs):
            rel = r["image"][0] if isinstance(r["image"], (list, tuple)) else r["image"]
            p = os.path.join(args.data_dir, rel)
            if os.path.exists(p):
                paths.append(p); keep.append(r)
        if not keep:
            continue
        alpha = rng.choice(discrete) if discrete else rng.uniform(args.alpha_lo, args.alpha_hi)
        imgs = [Image.open(p).convert("RGB") for p in paths]
        caps = gen(imgs, alpha)
        for r, c in zip(keep, caps):
            nr = dict(r)
            nr[CAP_COL] = c
            nr[A_COL] = round(alpha, 4)
            out_rows.append(nr)
        done += len(keep)
        if done % (args.gen_batch * 25) < args.gen_batch:
            print(f"[distill] {done}/{n} (last alpha={alpha:.3f}) e.g. {caps[0][:80]!r}", flush=True)

    out = args.out_json
    if args.num_shards > 1:
        out = args.out_json.replace(".json", f".shard{args.shard_id}.json")
    json.dump(out_rows, open(out, "w"), ensure_ascii=False)
    print(f"[distill] wrote {len(out_rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
