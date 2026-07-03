"""Generate T2I outputs for every row in a test split JSON from an InternVL-U checkpoint.

Produces the same manifest format as generate.py so compute_metrics.py works unchanged.

Usage:
    python corl/eval/generate_internvlu.py \\
        --base_model /path/to/InternVL-U \\
        --adapter_dir results/internvlu_t2i_cached_captions_v2/checkpoint-15000 \\
        --test_split corl/eval/test_split_small.json \\
        --data_dir /projects/u6gd/datasets/PubMedVision \\
        --caption_field cached_captions \\
        --out_dir results/eval/internvlu_v2_cached

    # For original captions model:
        --adapter_dir results/internvlu_t2i_original_captions_v2/checkpoint-15000 \\
        --caption_field Original_Caption \\
        --out_dir results/eval/internvlu_v2_original
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

INTERNVLU_REPO = os.environ.get("INTERNVLU_REPO", "/projects/u6gd/umar/codes/InternVL-U")
if INTERNVLU_REPO not in sys.path:
    sys.path.insert(0, INTERNVLU_REPO)

from internvlu import InternVLUPipeline
from peft import PeftModel
from corl.open_r1.trainer.sft_trainer_alignment_internvlu import InternVLUT2IFlowMatch


def load_pipeline(base_model: str, adapter_dir: str | None, device, dtype=torch.bfloat16):
    print(f"[gen] loading base pipeline from {base_model} ...")
    pipe = InternVLUPipeline.from_pretrained(base_model, torch_dtype=dtype)
    for attr in vars(pipe).values():
        if hasattr(attr, "to") and callable(attr.to):
            try:
                attr.to(device)
            except Exception:
                pass

    if adapter_dir:
        gd = pipe.generation_decoder
        flow_model = InternVLUT2IFlowMatch(
            vlm=pipe.vlm,
            generation_decoder=gd,
            vlm_select_layer=gd.config.vlm_select_layer,
            flow_shift=gd.config.flow_shift,
            logit_mean=gd.config.logit_mean,
            logit_std=gd.config.logit_std,
        )
        print(f"[gen] loading LoRA adapter from {adapter_dir} ...")
        flow_model = PeftModel.from_pretrained(flow_model, adapter_dir)
        flow_model.merge_and_unload()
        print("[gen] LoRA merged.")

    return pipe


@torch.inference_mode()
def generate_image(pipe, caption: str, height=512, width=512,
                   num_inference_steps=20, all_cfg_scale=4.5, part_cfg_scale=2.0):
    out = pipe(
        caption,
        generation_mode="image",
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        all_cfg_scale=all_cfg_scale,
        part_cfg_scale=part_cfg_scale,
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default="")
    ap.add_argument("--test_split", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--caption_field", default="Original_Caption",
                    help="Row field to use as prompt. Use 'cached_captions' for cached model.")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--all_cfg_scale", type=float, default=4.5)
    ap.add_argument("--part_cfg_scale", type=float, default=2.0)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", "0")))
    ap.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    with open(args.test_split) as f:
        rows = json.load(f)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print(f"[gen] shard {args.shard}/{args.num_shards}: {len(rows)} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.base_model, args.adapter_dir or None, device)

    manifest_path = out_dir / f"manifest_shard{args.shard}.json"
    manifest = []
    t0 = time.perf_counter()

    for i, r in enumerate(rows):
        out_png = out_dir / "images" / f"{r['id']}.png"
        if out_png.exists():
            manifest.append({**r, "gen_path": str(out_png)})
            continue

        cap = r.get(args.caption_field) or r.get("Original_Caption") or ""
        if isinstance(cap, list):
            cap = cap[0]

        try:
            img = generate_image(pipe, cap,
                                 height=args.height, width=args.width,
                                 num_inference_steps=args.num_inference_steps,
                                 all_cfg_scale=args.all_cfg_scale,
                                 part_cfg_scale=args.part_cfg_scale)
            img.save(out_png)
            manifest.append({**r, "gen_path": str(out_png)})
        except Exception as e:
            print(f"[gen] {r['id']} failed: {e}")
            continue

        if (i + 1) % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"[gen] {i+1}/{len(rows)} done ({(i+1)/dt:.2f} img/s)")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"[gen] wrote {len(manifest)} rows -> {manifest_path}")


if __name__ == "__main__":
    main()
