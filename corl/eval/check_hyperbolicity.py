"""Decide whether a tree could form if we embedded the BiomedCLIP features in
hyperbolic space. Two diagnostics:

  (A) Gromov delta-hyperbolicity (relative) of the feature metric. delta_rel
      near 0 => tree-like (hyperbolic helps a lot); near 0.5 => flat / no tree.
  (B) Raw (un-normalized) feature-norm spread. Hyperbolic hierarchy lives on the
      RADIAL axis; if norms are ~constant there is no abstraction axis to use.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def delta_hyperbolicity(D, base=0):
    """Fournier-Ismail-Vigneron batch delta via max-min matrix product."""
    n = D.shape[0]
    row = D[base][None, :]
    col = D[base][:, None]
    G = 0.5 * (row + col - D)              # Gromov products w.r.t. base
    M = np.full_like(G, -np.inf)
    for k in range(n):                      # (G o G)[i,j] = max_k min(G[i,k], G[k,j])
        M = np.maximum(M, np.minimum(G[:, k][:, None], G[k, :][None, :]))
    delta = float((M - G).max())
    diam = float(D.max())
    return delta, 2 * delta / (diam + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_cache", default="results/eval_small/proto_variants/feat_cache.pt")
    ap.add_argument("--shard_dir", default="data/prototype_shards")
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()

    d = torch.load(args.feat_cache, map_location="cpu")
    feats = F.normalize(d["feats"].float(), dim=-1).numpy()
    kept = d["kept"]
    modality = [r.get("modality") for r in kept]
    n = min(args.sample, feats.shape[0])
    X = feats[:n]

    print(f"[hyp] {n} normalized features, dim {X.shape[1]}\n")

    # --- (A) delta-hyperbolicity under two metrics -------------------------
    for name, D in [
        ("euclidean (on sphere)", np.sqrt(np.maximum(0, 2 - 2 * (X @ X.T)))),
        ("angular/geodesic",      np.arccos(np.clip(X @ X.T, -1, 1))),
    ]:
        delta, drel = delta_hyperbolicity(D)
        print(f"[A] delta-hyperbolicity ({name:22s}): delta={delta:.4f}  "
              f"delta_rel={drel:.3f}")
    print("    ref: tree~0.0 | strongly hyperbolic <0.25 | flat/high-dim ~0.4-0.5\n")

    # random-Gaussian baseline of same shape, for calibration
    rng = np.random.default_rng(0)
    G = rng.standard_normal((n, X.shape[1])); G /= np.linalg.norm(G, axis=1, keepdims=True)
    Dg = np.sqrt(np.maximum(0, 2 - 2 * (G @ G.T)))
    _, drel_g = delta_hyperbolicity(Dg)
    print(f"[A] baseline random unit-Gaussian delta_rel = {drel_g:.3f} "
          f"(this is what 'no structure' looks like)\n")

    # --- (B) raw feature-norm spread (the radial axis) ---------------------
    shards = sorted(glob.glob(os.path.join(args.shard_dir, "feats_rank*.pt")))
    if shards:
        raw = torch.load(shards[0], map_location="cpu")["feats"].float().numpy()
        norms = np.linalg.norm(raw, axis=1)
        print(f"[B] raw feature norms over {len(norms)} train imgs (shard 0):")
        print(f"    mean={norms.mean():.3f}  std={norms.std():.3f}  "
              f"cv={norms.std()/norms.mean():.3f}  min={norms.min():.3f}  "
              f"max={norms.max():.3f}  max/min={norms.max()/norms.min():.2f}")
        print("    cv near 0 => no radial/abstraction axis => naive map gives no tree.\n")
    else:
        print("[B] no raw shards found; skipping norm test\n")

    # --- (C) does body_part nest within modality? (label-side sanity) ------
    by_mod = defaultdict(set)
    for r in kept:
        by_mod[r.get("modality")].add(r.get("body_part"))
    print("[C] body-parts per modality (is there a real 2-level taxonomy?):")
    for m, bps in sorted(by_mod.items(), key=lambda kv: -len(kv[1])):
        print(f"    {m:28s} {len(bps):2d} body-parts")


if __name__ == "__main__":
    main()
