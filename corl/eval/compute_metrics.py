"""Stage 2 of the eval pipeline: compute metrics from a generation manifest.

For each generated image:
  - Encode (gen, gt) with BiomedCLIP visual.
  - Encode caption with BiomedCLIP text.
  - Accumulate per modality.

Reported:
  - FID-BiomedCLIP per modality + macro-averaged
  - CLIPScore-BiomedCLIP: mean cos(gen_img, text), per modality + macro
  - CLIPScore-BiomedCLIP (GT, upper bound): mean cos(gt_img, text), per modality
  - I2I cos: mean cos(gen_img, gt_img), per modality

Usage:
    python corl/eval/compute_metrics.py \\
        --manifest_glob "results/eval/exp5/manifest_shard*.json" \\
        --data_dir /work/.../PubMedVision \\
        --out results/eval/exp5/metrics.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
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
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_biomedclip(device):
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_ID)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    try:
        res = int(model.visual.image_size if isinstance(model.visual.image_size, int)
                  else model.visual.image_size[0])
    except Exception:
        res = 224
    return model, tokenizer, res


def preprocess_image(img: Image.Image, res: int, device):
    img = img.convert("RGB").resize((res, res), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    arr = arr.unsqueeze(0).to(device)
    arr = (arr - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
    return arr


@torch.inference_mode()
def encode_images(model, paths, res, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch = torch.cat(
            [preprocess_image(Image.open(p), res, device) for p in batch_paths], dim=0,
        )
        f = model.encode_image(batch)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


@torch.inference_mode()
def encode_text(model, tokenizer, texts, device, batch_size=64):
    # open_clip's HFTokenizer.__call__ uses `batch_encode_plus`, which was
    # dropped in transformers 5.x. Call the underlying HF tokenizer directly.
    inner = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    ctx_len = int(getattr(tokenizer, "context_length", 256))
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            tok = tokenizer(batch).to(device)
        except AttributeError:
            enc = inner(batch, return_tensors="pt", max_length=ctx_len,
                        padding="max_length", truncation=True)
            tok = enc.input_ids.to(device)
        f = model.encode_text(tok)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


def compute_fid(feats_a: torch.Tensor, feats_b: torch.Tensor) -> float:
    """Frechet distance between two Gaussian-fit feature sets."""
    from scipy.linalg import sqrtm
    a = feats_a.numpy().astype(np.float64)
    b = feats_b.numpy().astype(np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = sqrtm(cov_a @ cov_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_glob", required=True,
                    help='Glob over per-shard manifests, e.g. ".../manifest_shard*.json"')
    ap.add_argument("--data_dir", required=True,
                    help="Base dir for GT image paths in the manifest.")
    ap.add_argument("--out", default="")
    ap.add_argument("--caption_field", default="Original_Caption")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # Merge per-shard manifests.
    paths = sorted(glob.glob(args.manifest_glob))
    if not paths:
        raise SystemExit(f"no manifests matched {args.manifest_glob}")
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.load(f))
    print(f"[metrics] loaded {len(rows)} rows from {len(paths)} shard(s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, res = load_biomedclip(device)
    print(f"[metrics] BiomedCLIP loaded ({res}x{res})")

    # Bucket rows by modality.
    by_mod = defaultdict(list)
    for r in rows:
        by_mod[r.get("modality")].append(r)

    results = {"per_modality": {}, "macro": {}, "n_total": len(rows)}
    macro = defaultdict(list)

    for mod, mrows in by_mod.items():
        gen_paths = [r["gen_path"] for r in mrows]
        gt_paths = [os.path.join(args.data_dir, r["image"]) for r in mrows]
        caps = [r.get(args.caption_field) or r.get("Original_Caption") or "" for r in mrows]
        caps = [c[0] if isinstance(c, list) else c for c in caps]

        print(f"[metrics] {mod}: encoding {len(mrows)} samples")
        feats_gen = encode_images(model, gen_paths, res, device, args.batch_size)
        feats_gt = encode_images(model, gt_paths, res, device, args.batch_size)
        feats_txt = encode_text(model, tokenizer, caps, device, args.batch_size)

        gen_n = F.normalize(feats_gen, dim=-1)
        gt_n = F.normalize(feats_gt, dim=-1)
        txt_n = F.normalize(feats_txt, dim=-1)

        clipscore_gen = (gen_n * txt_n).sum(dim=-1).mean().item()
        clipscore_gt = (gt_n * txt_n).sum(dim=-1).mean().item()
        i2i_cos = (gen_n * gt_n).sum(dim=-1).mean().item()
        fid = compute_fid(feats_gen, feats_gt)

        results["per_modality"][mod] = {
            "n": len(mrows),
            "fid_biomedclip": fid,
            "clipscore_gen": clipscore_gen,
            "clipscore_gt_upper_bound": clipscore_gt,
            "i2i_cosine": i2i_cos,
        }
        for k, v in (("fid_biomedclip", fid), ("clipscore_gen", clipscore_gen),
                     ("clipscore_gt_upper_bound", clipscore_gt), ("i2i_cosine", i2i_cos)):
            macro[k].append(v)
        print(f"[metrics]   FID={fid:.2f}  CLIPScore-gen={clipscore_gen:.4f}  "
              f"CLIPScore-gt={clipscore_gt:.4f}  i2i={i2i_cos:.4f}")

    results["macro"] = {k: float(np.mean(v)) for k, v in macro.items()}

    print("\n[metrics] === macro-average ===")
    for k, v in results["macro"].items():
        print(f"  {k:30s}  {v:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[metrics] wrote {args.out}")


if __name__ == "__main__":
    main()
