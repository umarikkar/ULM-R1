"""Visualise which images fall under each centroid, for every prototype variant.

Encodes the 600-image test split through frozen BiomedCLIP *once* (cached to
``feat_cache.pt``), then for each ``data/prototype_variants/*.pt`` hard-assigns
every image to its nearest centroid (cosine) and writes:

  <out_dir>/<variant>/proto_grid.png   - 1 row per prototype, top-N images by sim
  <out_dir>/<variant>/proto_summary.json
  <out_dir>/<variant>/assignments.json

Usage:
    python corl/eval/inspect_prototype_variants.py \
        --variants_dir data/prototype_variants \
        --rows corl/eval/test_split_small.json \
        --data_dir /work/um00109/MLLM/datasets/PubMedVision \
        --out_dir results/eval_small/proto_variants
"""

import argparse
import glob
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

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


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


def preprocess(img: Image.Image, res: int):
    img = img.convert("RGB").resize((res, res), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return (arr.unsqueeze(0) - CLIP_MEAN) / CLIP_STD


def encode_rows(rows, data_dir, device, batch_size=64):
    """Return (feats [N, d] L2-normalised float, kept_rows). Skips missing imgs."""
    model, res = load_biomedclip(device)
    model = model.to(dtype=torch.bfloat16)
    feats, kept = [], []
    batch, batch_rows = [], []

    def flush():
        if not batch:
            return
        x = torch.cat(batch).to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            f = model(x).float()
        f = F.normalize(f, dim=-1).cpu()
        feats.append(f)
        kept.extend(batch_rows)
        batch.clear(); batch_rows.clear()

    for i, r in enumerate(rows):
        ip = os.path.join(data_dir, r["image"])
        if not os.path.exists(ip):
            continue
        try:
            batch.append(preprocess(Image.open(ip), res))
            batch_rows.append(r)
        except Exception as e:
            print(f"[encode]   skip {r['id']}: {e}")
            continue
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 100 == 0:
            print(f"[encode]   {i+1}/{len(rows)}")
    flush()
    return torch.cat(feats, dim=0), kept


def build_variant(name, centroids, K, feats, kept, out_dir, n_per_proto):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sims = feats @ centroids.t()                  # [N, K]
    top1 = sims.argmax(dim=-1)
    top_sim = sims.max(dim=-1).values

    by_proto = defaultdict(list)
    assigns = []
    for j, r in enumerate(kept):
        k = int(top1[j])
        a = {"id": r["id"], "image": r["image"], "modality": r.get("modality"),
             "body_part": r.get("body_part"), "proto": k, "sim": float(top_sim[j])}
        assigns.append(a)
        by_proto[k].append(a)
    json.dump(assigns, open(out_dir / "assignments.json", "w"))

    summary = []
    print(f"\n=== {name}: prototype population (top modality / body_part) ===")
    print(f"{'proto':>5s} {'n':>5s} {'top modality':28s} {'top body_part':28s}")
    for k in range(K):
        mr = by_proto.get(k, [])
        if not mr:
            print(f"{k:>5d} {0:>5d} (empty)")
            summary.append({"proto": k, "n": 0})
            continue
        mods = Counter(r["modality"] for r in mr)
        bps = Counter(r["body_part"] for r in mr)
        m_top, m_n = mods.most_common(1)[0]
        b_top, b_n = bps.most_common(1)[0]
        print(f"{k:>5d} {len(mr):>5d} "
              f"{f'{m_top} ({m_n}/{len(mr)})':28s} {f'{b_top} ({b_n}/{len(mr)})':28s}")
        summary.append({"proto": k, "n": len(mr),
                        "top_modality": m_top, "top_modality_frac": m_n / len(mr),
                        "top_body_part": b_top, "top_body_part_frac": b_n / len(mr),
                        "modality_dist": dict(mods), "body_part_dist": dict(bps)})
    json.dump(summary, open(out_dir / "proto_summary.json", "w"), indent=2)

    # Grid: 1 row per prototype, n_per_proto cols, by descending sim.
    thumb = 192
    label_w = 230
    grid = Image.new("RGB", (label_w + n_per_proto * thumb, K * thumb), (255, 255, 255))
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default()
    except Exception:
        draw = None
    for k in range(K):
        mr = sorted(by_proto.get(k, []), key=lambda r: -r["sim"])[:n_per_proto]
        for c, r in enumerate(mr):
            try:
                img = Image.open(os.path.join(args_data_dir, r["image"])).convert("RGB")
                w, h = img.size
                img = img.resize((int(w * thumb / h), thumb), Image.Resampling.BICUBIC)
                img = img.crop((0, 0, min(thumb, img.width), thumb))
            except Exception:
                continue
            grid.paste(img, (label_w + c * thumb, k * thumb))
        if draw is not None:
            mods = Counter(r["modality"] for r in by_proto.get(k, []))
            top = ", ".join(f"{m}:{c}" for m, c in mods.most_common(2))
            draw.text((5, k * thumb + 5), f"P{k}  n={len(by_proto.get(k, []))}",
                      fill="black", font=font)
            draw.text((5, k * thumb + 25), top[:34], fill="black", font=font)
    grid.save(out_dir / "proto_grid.png")
    print(f"[{name}] wrote {out_dir / 'proto_grid.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants_dir", default="data/prototype_variants")
    ap.add_argument("--rows", default="corl/eval/test_split_small.json")
    ap.add_argument("--data_dir", default="/work/um00109/MLLM/datasets/PubMedVision")
    ap.add_argument("--out_dir", default="results/eval_small/proto_variants")
    ap.add_argument("--n_per_proto", type=int, default=6)
    args = ap.parse_args()

    global args_data_dir
    args_data_dir = args.data_dir

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(args.rows))
    print(f"[variants] {len(rows)} rows from {args.rows}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = out_root / "feat_cache.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        feats, kept = d["feats"], d["kept"]
        print(f"[variants] loaded cached feats for {len(kept)} images from {cache}")
    else:
        feats, kept = encode_rows(rows, args.data_dir, device)
        torch.save({"feats": feats, "kept": kept}, cache)
        print(f"[variants] encoded + cached {len(kept)} images -> {cache}")

    variants = sorted(glob.glob(os.path.join(args.variants_dir, "*.pt")))
    print(f"[variants] {len(variants)} centroid files found")
    for vf in variants:
        name = Path(vf).stem
        cdata = torch.load(vf, map_location="cpu", weights_only=False)
        centroids = F.normalize(cdata["centroids"].float(), dim=-1)
        K = int(cdata["K"])
        build_variant(name, centroids, K, feats, kept, out_root / name, args.n_per_proto)

    print(f"\n[variants] done -> {out_root}")


if __name__ == "__main__":
    main()
