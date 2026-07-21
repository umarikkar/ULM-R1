"""Standard Inception-v3 FID (Heusel et al. 2017) for a generation manifest.

Uses the canonical pytorch-fid InceptionV3 (TF-ported weights), 2048-d pool3
features. Computes a single POOLED FID per run (gen set vs GT set over all
images) -- pooled, not per-modality-macro, because 2048-d covariance needs
N > 2048 to be full-rank (per-modality N~833 would be rank-deficient).

Writes metrics_inception.json = {"fid_inception": float, "n": int}.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm


def frechet(feats_a, feats_b):
    a = feats_a.astype(np.float64)
    b = feats_b.astype(np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = sqrtm(cov_a @ cov_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean))


def inception_feats(paths, model, device, bs=50):
    feats = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                im = Image.open(p).convert("RGB").resize((299, 299), Image.BILINEAR)
            except Exception:
                continue
            batch.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            f = model(x)[0].squeeze(-1).squeeze(-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_glob", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=50)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.manifest_glob))
    rows = []
    for p in paths:
        rows.extend(json.load(open(p)))
    # Keep rows whose gen + GT both exist.
    gen, gt = [], []
    for r in rows:
        gp = r["gen_path"]
        tp = os.path.join(args.data_dir, r["image"])
        if os.path.exists(gp) and os.path.exists(tp):
            gen.append(gp); gt.append(tp)
    print(f"[fid-incep] {len(gen)} paired images from {len(paths)} shards")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from pytorch_fid.inception import InceptionV3
    bidx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([bidx]).eval().to(device)

    fg = inception_feats(gen, model, device, args.batch_size)
    ft = inception_feats(gt, model, device, args.batch_size)
    fid = frechet(fg, ft)
    print(f"[fid-incep] FID_inception = {fid:.3f}  (n_gen={len(fg)}, n_gt={len(ft)})")

    json.dump({"fid_inception": fid, "n": int(min(len(fg), len(ft)))},
              open(args.out, "w"), indent=2)
    print(f"[fid-incep] wrote {args.out}")


if __name__ == "__main__":
    main()