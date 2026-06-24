"""Score every prototype variant to decide which clustering is best.

Uses the cached 600-image BiomedCLIP features (results/.../feat_cache.pt) and the
modality / body_part labels. For each variant we recompute the hard assignment to
its centroids and report:

  Balance   : n_empty, size CV (std/mean, lower=more balanced), Gini.
  Geometry  : silhouette (cosine, higher=better), Davies-Bouldin (lower=better).
  Semantics : weighted purity + NMI / V-measure vs modality and body_part.

K differs across variants, so we lean on K-robust metrics (silhouette, NMI,
V-measure, ARI) for the final ranking rather than raw homogeneity.
"""

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    normalized_mutual_info_score, v_measure_score, adjusted_rand_score,
)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n


def weighted_purity(labels, gt):
    """Sum over clusters of max-class-count, divided by N (size-weighted)."""
    total, n = 0, len(gt)
    for c in set(labels):
        members = [gt[i] for i in range(n) if labels[i] == c]
        if members:
            total += Counter(members).most_common(1)[0][1]
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants_dir", default="data/prototype_variants")
    ap.add_argument("--feat_cache", default="results/eval_small/proto_variants/feat_cache.pt")
    ap.add_argument("--out", default="results/eval_small/proto_variants/method_scores.json")
    args = ap.parse_args()

    d = torch.load(args.feat_cache, map_location="cpu")
    feats = F.normalize(d["feats"].float(), dim=-1)        # [N, 512] unit sphere
    kept = d["kept"]
    X = feats.numpy()
    modality = [r.get("modality") for r in kept]
    body_part = [r.get("body_part") for r in kept]
    N = len(kept)
    print(f"[score] {N} images, {len(set(modality))} modalities, {len(set(body_part))} body parts\n")

    rows = []
    for vf in sorted(glob.glob(os.path.join(args.variants_dir, "*.pt"))):
        name = Path(vf).stem
        cdata = torch.load(vf, map_location="cpu", weights_only=False)
        centroids = F.normalize(cdata["centroids"].float(), dim=-1)
        K = int(cdata["K"])
        labels = (feats @ centroids.t()).argmax(dim=-1).numpy()

        counts = np.bincount(labels, minlength=K)
        nonempty = counts[counts > 0]
        n_empty = int((counts == 0).sum())
        size_cv = float(nonempty.std() / nonempty.mean())

        # geometry — silhouette needs >=2 populated clusters
        uniq = len(set(labels))
        sil = float(silhouette_score(X, labels, metric="cosine")) if uniq > 1 else float("nan")
        db = float(davies_bouldin_score(X, labels)) if uniq > 1 else float("nan")

        rows.append({
            "variant": name, "method": cdata.get("method"), "K": K,
            "n_empty": n_empty,
            "size_min": int(nonempty.min()), "size_max": int(nonempty.max()),
            "size_cv": round(size_cv, 3), "gini": round(float(gini(counts)), 3),
            "silhouette": round(sil, 4), "davies_bouldin": round(db, 3),
            "purity_modality": round(weighted_purity(labels, modality), 3),
            "purity_bodypart": round(weighted_purity(labels, body_part), 3),
            "nmi_modality": round(float(normalized_mutual_info_score(modality, labels)), 3),
            "vmeasure_modality": round(float(v_measure_score(modality, labels)), 3),
            "ari_modality": round(float(adjusted_rand_score(modality, labels)), 3),
            "nmi_bodypart": round(float(normalized_mutual_info_score(body_part, labels)), 3),
        })

    # composite rank: z-score the K-robust signals (higher=better), average.
    def z(vals, higher_better=True):
        a = np.array(vals, dtype=float)
        s = a.std() if a.std() > 1e-9 else 1.0
        zz = (a - a.mean()) / s
        return zz if higher_better else -zz

    comp = (
        z([r["silhouette"] for r in rows], True)
        + z([r["davies_bouldin"] for r in rows], False)
        + z([r["nmi_modality"] for r in rows], True)
        + z([r["vmeasure_modality"] for r in rows], True)
        + z([r["size_cv"] for r in rows], False)
        + z([r["nmi_bodypart"] for r in rows], True)
    ) / 6.0
    for r, c in zip(rows, comp):
        r["composite"] = round(float(c), 3)
    rows.sort(key=lambda r: -r["composite"])

    hdr = ["variant", "K", "n_empty", "size_cv", "gini", "silhouette",
           "davies_bouldin", "purity_modality", "nmi_modality",
           "vmeasure_modality", "ari_modality", "nmi_bodypart", "composite"]
    w = {h: max(len(h), 10) for h in hdr}
    print("  ".join(h.rjust(w[h]) for h in hdr))
    for r in rows:
        print("  ".join(str(r[h]).rjust(w[h]) for h in hdr))

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\n[score] wrote {args.out}")
    print(f"[score] best by composite: {rows[0]['variant']}")


if __name__ == "__main__":
    main()
