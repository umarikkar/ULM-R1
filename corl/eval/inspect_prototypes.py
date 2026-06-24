"""Sanity-check the BiomedCLIP prototype clusters.

For a sample of PubMedVision images:
  1. Encode each with frozen BiomedCLIP visual.
  2. Hard-assign to its closest centroid.
  3. Per prototype: report modality + body_part distribution and save a grid
     of 4 representative images.

Usage:
    python corl/eval/inspect_prototypes.py \\
        --centroids data/prototype_centroids.pt \\
        --rows corl/eval/test_split_small.json \\
        --data_dir /work/um00109/MLLM/datasets/PubMedVision \\
        --out_dir results/eval_small/proto_inspect
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)


def load_biomedclip(device):
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.visual.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    try:
        res = int(model.image_size if isinstance(model.image_size, int) else model.image_size[0])
    except Exception:
        res = 224
    return model, res


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def preprocess(img: Image.Image, res: int, device):
    img = img.convert("RGB").resize((res, res), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    arr = arr.unsqueeze(0).to(device)
    return (arr - CLIP_MEAN.to(device)) / CLIP_STD.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", default="data/prototype_centroids.pt")
    ap.add_argument("--rows", default="corl/eval/test_split_small.json")
    ap.add_argument("--data_dir", default="/work/um00109/MLLM/datasets/PubMedVision")
    ap.add_argument("--out_dir", default="results/eval_small/proto_inspect")
    ap.add_argument("--n_per_proto", type=int, default=4,
                    help="Number of sample images shown per prototype in the grid.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.rows))
    print(f"[proto-inspect] {len(rows)} rows from {args.rows}")

    cdata = torch.load(args.centroids, map_location="cpu")
    centroids = cdata["centroids"].float()           # [K, d], L2-normalised
    K = int(cdata["K"])
    print(f"[proto-inspect] {K} centroids, d={centroids.shape[-1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, res = load_biomedclip(device)
    centroids_dev = centroids.to(device)

    # Assign each row to nearest prototype.
    assigns = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ip = os.path.join(args.data_dir, r["image"])
            if not os.path.exists(ip):
                continue
            try:
                x = preprocess(Image.open(ip), res, device).to(dtype=torch.bfloat16)
                model_bf = model.to(dtype=torch.bfloat16)
                feat = model_bf(x).float()
                feat = F.normalize(feat, dim=-1)
                sims = feat @ centroids_dev.t()       # [1, K]
                top1 = int(sims.argmax(dim=-1).item())
                top_sim = float(sims.max().item())
                assigns.append({"id": r["id"], "image": r["image"],
                                "modality": r.get("modality"),
                                "body_part": r.get("body_part"),
                                "proto": top1, "sim": top_sim})
            except Exception as e:
                print(f"[proto-inspect]   {r['id']}: {e}")
            if (i + 1) % 100 == 0:
                print(f"[proto-inspect]   {i+1}/{len(rows)} assigned")

    print(f"[proto-inspect] assigned {len(assigns)} rows")
    json.dump(assigns, open(out_dir / "assignments.json", "w"))

    # Aggregate.
    by_proto = defaultdict(list)
    for a in assigns:
        by_proto[a["proto"]].append(a)

    print("\n=== prototype population + dominant modality / body_part ===")
    print(f"{'proto':>5s} {'n':>5s} {'top modality (frac)':35s} {'top body_part (frac)':35s}")
    rows_summary = []
    for k in range(K):
        mr = by_proto.get(k, [])
        if not mr:
            print(f"{k:>5d} {0:>5d} (empty)")
            continue
        mods = Counter(r["modality"] for r in mr)
        bps = Counter(r["body_part"] for r in mr)
        m_top, m_n = mods.most_common(1)[0]
        b_top, b_n = bps.most_common(1)[0]
        m_str = f"{m_top} ({m_n}/{len(mr)} = {m_n/len(mr):.0%})"
        b_str = f"{b_top} ({b_n}/{len(mr)} = {b_n/len(mr):.0%})"
        print(f"{k:>5d} {len(mr):>5d} {m_str:35s} {b_str:35s}")
        rows_summary.append({"proto": k, "n": len(mr),
                             "top_modality": m_top, "top_modality_frac": m_n/len(mr),
                             "top_body_part": b_top, "top_body_part_frac": b_n/len(mr),
                             "modality_dist": dict(mods),
                             "body_part_dist": dict(bps)})
    json.dump(rows_summary, open(out_dir / "proto_summary.json", "w"), indent=2)

    # Build a grid: 1 row per prototype, n_per_proto cols, by descending sim.
    thumb_h = 192
    cols = args.n_per_proto
    canvas_w = cols * thumb_h
    canvas_h = K * thumb_h
    grid = Image.new("RGB", (canvas_w + 220, canvas_h), (255, 255, 255))
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default()
    except Exception:
        draw = None

    for k in range(K):
        mr = sorted(by_proto.get(k, []), key=lambda r: -r["sim"])[:cols]
        for c, r in enumerate(mr):
            ip = os.path.join(args.data_dir, r["image"])
            try:
                img = Image.open(ip).convert("RGB")
                w, h = img.size
                img = img.resize((int(w * thumb_h / h), thumb_h), Image.Resampling.BICUBIC)
                img = img.crop((0, 0, min(thumb_h, img.width), thumb_h))
            except Exception:
                continue
            grid.paste(img, (220 + c * thumb_h, k * thumb_h))
        if draw is not None:
            mods = Counter(r["modality"] for r in by_proto.get(k, []))
            top = ", ".join(f"{m}:{c}" for m, c in mods.most_common(3))
            draw.text((5, k * thumb_h + 5), f"P{k}  n={len(by_proto.get(k, []))}",
                      fill="black", font=font)
            draw.text((5, k * thumb_h + 25), top[:30], fill="black", font=font)
    grid.save(out_dir / "proto_grid.png")
    print(f"\n[proto-inspect] wrote {out_dir/'proto_grid.png'}")
    print(f"[proto-inspect] wrote {out_dir/'proto_summary.json'}")


if __name__ == "__main__":
    main()