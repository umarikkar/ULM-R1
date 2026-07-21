"""Standard Inception-v3 FID for all train x eval cells, efficiently.

Computes the GT Inception statistics ONCE over the test-split real images, then
each cell's generated set is scored against that single fixed reference (as FID
is meant to be used). Single process, single GPU -> no self-contention and half
the image reads of the naive per-cell version.

Writes results/eval_levels/<tag>/metrics_inception.json = {"fid_inception", "n"}.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def feats_for(paths, model, device, bs):
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                im = Image.open(p).convert("RGB").resize((299, 299), Image.BILINEAR)
            except Exception:
                continue
            batch.append(torch.from_numpy(np.asarray(im).copy())
                         .permute(2, 0, 1).float() / 255.0)
        if not batch:
            continue
        with torch.no_grad():
            f = model(torch.stack(batch).to(device))[0].squeeze(-1).squeeze(-1)
        out.append(f.cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"    {i+len(batch)}/{len(paths)}", flush=True)
    return np.concatenate(out, 0).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_levels")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="corl/eval/test_split_levels.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_per_cell", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    tags = [f"{t}__{e}" for t in ("l1", "l2", "l3") for e in ("l1", "l2", "l3")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(device)

    # ---- GT reference stats (once) ----
    split = json.load(open(args.split))
    gt_paths = [os.path.join(args.data_dir, r["image"]) for r in split]
    gt_paths = [p for p in gt_paths if os.path.exists(p)]
    if args.max_per_cell:
        gt_paths = gt_paths[: max(args.max_per_cell, 4998)]
    print(f"[fid] GT reference: {len(gt_paths)} images", flush=True)
    mu_gt, cov_gt = stats(feats_for(gt_paths, model, device, args.batch_size))

    # ---- each cell's gen set vs GT ----
    for tag in tags:
        mans = sorted(glob.glob(f"{args.root}/{tag}/manifest_shard*.json"))
        if not mans:
            print(f"[fid] {tag}: no manifest, skip"); continue
        rows = []
        for m in mans:
            rows.extend(json.load(open(m)))
        gen = [r["gen_path"] for r in rows if os.path.exists(r["gen_path"])]
        if args.max_per_cell:
            gen = gen[: args.max_per_cell]
        print(f"[fid] {tag}: {len(gen)} gen images", flush=True)
        mu_g, cov_g = stats(feats_for(gen, model, device, args.batch_size))
        fid = frechet(mu_g, cov_g, mu_gt, cov_gt)
        out = f"{args.root}/{tag}/metrics_inception.json"
        json.dump({"fid_inception": fid, "n": len(gen)}, open(out, "w"), indent=2)
        print(f"[fid] {tag}: FID_inception = {fid:.2f}  -> {out}", flush=True)

    print("ALL INCEPTION FID DONE")


if __name__ == "__main__":
    main()