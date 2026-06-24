"""Try multiple unsupervised clustering schemes on the cached BiomedCLIP
feature shards (data/prototype_shards/feats_rank*.pt). Each method writes a
centroid file with a descriptive name so downstream code can swap them in.

Methods covered:
  - vanilla   : sklearn KMeans (same as the existing default; baseline)
  - sinkhorn  : Sinkhorn-balanced K-means (SwAV-style; equal cluster sizes)
  - spherical : explicit unit-sphere K-means (cosine geometry)
  - gmm       : Gaussian Mixture with diagonal covariance, soft assignment
  - sweep     : just vanilla K-means at several K values
  - hier      : two-stage K=6 -> K=4 per modality cluster (24 total)

Usage:
    python corl/scripts/cluster_variants.py \\
        --shard_dir data/prototype_shards \\
        --out_dir   data/prototype_variants

Then inspect each .pt against test_split_small via inspect_prototypes.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_feats(shard_dir):
    shards = sorted(Path(shard_dir).glob("feats_rank*.pt"))
    if not shards:
        sys.exit(f"no feats_rank*.pt under {shard_dir}")
    feats, ids = [], []
    for s in shards:
        d = torch.load(s, map_location="cpu")
        feats.append(d["feats"])
        ids.extend(d.get("ids", []))
    X = torch.cat(feats, dim=0).numpy().astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)   # unit sphere
    print(f"[load] {X.shape[0]} features, dim {X.shape[1]}")
    return X, ids


def save_centroids(out_path, centroids, K, X_dim, n, counts=None, extra=None):
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    payload = {
        "centroids": torch.from_numpy(centroids.astype(np.float32)),
        "K": int(K),
        "d_feat": int(X_dim),
        "n_samples": int(n),
    }
    if counts is not None:
        payload["cluster_counts"] = torch.from_numpy(counts.astype(np.int64))
    if extra:
        payload.update(extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[save] {out_path}  K={K}  sizes={counts.tolist() if counts is not None else '-'}")


# --- 1. Vanilla K-means -------------------------------------------------
def run_vanilla(X, K, out_path):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    counts = np.bincount(km.labels_, minlength=K)
    save_centroids(out_path, km.cluster_centers_, K, X.shape[1], X.shape[0], counts,
                   extra={"method": "vanilla_kmeans"})


# --- 2. Sinkhorn-balanced K-means (SwAV style) --------------------------
def sinkhorn_assign(scores, n_iters=3, eps=0.05):
    Q = np.exp(scores / eps).astype(np.float64)
    Q /= Q.sum()
    N, K = Q.shape
    for _ in range(n_iters):
        # marginal on K  -> 1/K
        Q /= Q.sum(axis=0, keepdims=True); Q /= K
        # marginal on N  -> 1/N
        Q /= Q.sum(axis=1, keepdims=True); Q /= N
    return (Q * N).astype(np.float32)   # soft assignment, col-sums ~1


def run_sinkhorn(X, K, out_path, n_outer=30, eps=0.05):
    # init with vanilla K-means centroids for a warm start
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(20000, X.shape[0]), replace=False)
    centroids = KMeans(n_clusters=K, n_init=3, random_state=0).fit(X[idx]).cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    for it in range(n_outer):
        scores = X @ centroids.T                            # cosine, [N, K]
        Q = sinkhorn_assign(scores, n_iters=3, eps=eps)     # [N, K], col sums ~1
        # centroid update: weighted mean of features
        new = (Q.T @ X) / (Q.sum(axis=0, keepdims=True).T + 1e-8)
        new = new / (np.linalg.norm(new, axis=1, keepdims=True) + 1e-8)
        delta = float(np.linalg.norm(new - centroids))
        centroids = new
        if it % 5 == 0 or it == n_outer - 1:
            print(f"  [sinkhorn] iter {it} delta={delta:.4f}")
        if delta < 1e-4:
            break
    # hard counts for reporting
    counts = np.bincount(scores.argmax(axis=1), minlength=K)
    save_centroids(out_path, centroids, K, X.shape[1], X.shape[0], counts,
                   extra={"method": "sinkhorn_balanced_kmeans", "eps": eps})


# --- 3. Spherical K-means -----------------------------------------------
def run_spherical(X, K, out_path, n_iters=100):
    rng = np.random.default_rng(0)
    # init: random sample of features (already unit-norm)
    centroids = X[rng.choice(X.shape[0], size=K, replace=False)].copy()
    for it in range(n_iters):
        sims = X @ centroids.T                              # cosine
        labels = sims.argmax(axis=1)
        new = np.zeros_like(centroids)
        for k in range(K):
            mask = labels == k
            if mask.any():
                m = X[mask].mean(axis=0)
                m /= (np.linalg.norm(m) + 1e-8)
                new[k] = m
            else:
                new[k] = X[rng.choice(X.shape[0])]
        delta = float(np.linalg.norm(new - centroids))
        centroids = new
        if delta < 1e-5:
            break
    sims = X @ centroids.T
    counts = np.bincount(sims.argmax(axis=1), minlength=K)
    save_centroids(out_path, centroids, K, X.shape[1], X.shape[0], counts,
                   extra={"method": "spherical_kmeans"})


# --- 4. GMM with diagonal covariance ------------------------------------
def run_gmm(X, K, out_path):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=K, covariance_type="diag",
                         max_iter=200, random_state=0, init_params="kmeans").fit(X)
    # GMM means aren't unit norm; we re-normalise to use the same downstream
    # cosine-based assignment as K-means centroids.
    centroids = gm.means_.astype(np.float32)
    labels = gm.predict(X)
    counts = np.bincount(labels, minlength=K)
    save_centroids(out_path, centroids, K, X.shape[1], X.shape[0], counts,
                   extra={"method": "gmm_diag"})


# --- 5. Hierarchical: K=6 then K=4 within each --------------------------
def run_hierarchical(X, out_path, K_outer=6, K_inner=4):
    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=K_outer, n_init=10, random_state=0).fit(X)
    sub_centroids = []
    for k in range(K_outer):
        Xk = X[km1.labels_ == k]
        if Xk.shape[0] < K_inner:
            # not enough samples for inner clustering — just keep the outer centroid
            sub_centroids.append(km1.cluster_centers_[k:k+1])
            continue
        km2 = KMeans(n_clusters=K_inner, n_init=5, random_state=0).fit(Xk)
        sub_centroids.append(km2.cluster_centers_)
    centroids = np.concatenate(sub_centroids, axis=0).astype(np.float32)
    K_eff = centroids.shape[0]
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sims = X @ centroids.T
    counts = np.bincount(sims.argmax(axis=1), minlength=K_eff)
    save_centroids(out_path, centroids, K_eff, X.shape[1], X.shape[0], counts,
                   extra={"method": "hierarchical", "K_outer": K_outer, "K_inner": K_inner})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", default="data/prototype_shards")
    ap.add_argument("--out_dir", default="data/prototype_variants")
    args = ap.parse_args()

    X, _ = load_feats(args.shard_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. vanilla
    print("\n=== vanilla K-means K=16 ===")
    run_vanilla(X, 16, out / "v01_vanilla_k16.pt")

    # 2. sinkhorn balanced
    print("\n=== sinkhorn-balanced K=16 ===")
    run_sinkhorn(X, 16, out / "v02_sinkhorn_k16.pt")

    # 3. spherical (explicit cosine)
    print("\n=== spherical K=16 ===")
    run_spherical(X, 16, out / "v03_spherical_k16.pt")

    # 4. K sweep (vanilla baseline at multiple K)
    for K in (8, 12, 24, 32):
        print(f"\n=== K-sweep vanilla K={K} ===")
        run_vanilla(X, K, out / f"v04_sweep_k{K:02d}.pt")

    # 5. hierarchical (K_outer=6 x K_inner=4 = 24)
    print("\n=== hierarchical 6x4 = 24 ===")
    run_hierarchical(X, out / "v05_hier_6x4.pt", K_outer=6, K_inner=4)

    # GMM (diag) — bonus
    print("\n=== GMM diagonal K=16 ===")
    run_gmm(X, 16, out / "v06_gmm_diag_k16.pt")

    print("\n[done] all variants written to", out)


if __name__ == "__main__":
    main()
