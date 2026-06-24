"""Build BiomedCLIP-feature prototype centroids for unsupervised conditioning.

We pass a sample of training images through frozen BiomedCLIP (same model the
perceptual loss uses), then KMeans into K prototypes. The centroids are saved
as a small .pt file; the trainer loads them at init and soft-assigns each
training image to the K prototypes via cosine similarity.

Phases (run in order via the .sh launcher):
  features : DDP forward all (or a sample of) training images through BiomedCLIP,
             write per-rank feature shards. Resumable.
  cluster  : single-process merge of shards + KMeans into K centroids. Writes
             ``data/prototype_centroids.pt``.

Caption-first / VLM labels are NOT used here -- the whole point is that the
prior is discovered from the unlabeled image distribution itself.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)


def _load_biomedclip(model_id, device):
    """Mirror of sft_trainer_alignment._ensure_perceptual_model setup."""
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(model_id)
    model = model.visual
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device=device, dtype=torch.bfloat16)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    try:
        res = int(model.image_size if isinstance(model.image_size, int) else model.image_size[0])
    except Exception:
        res = 224
    return model, mean, std, res


def run_features(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    with open(args.captions, "r") as f:
        rows = json.load(f)
    if args.max_samples and args.max_samples < len(rows):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rows))[: args.max_samples]
        rows = [rows[int(i)] for i in idx]
        if rank == 0:
            print(f"[features] subsampled to {len(rows)} rows (seed=0)")

    n = len(rows)
    per_rank = (n + world_size - 1) // world_size
    my_rows = rows[rank * per_rank: min((rank + 1) * per_rank, n)]
    print(f"[rank {rank}] processing {len(my_rows)} rows", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"feats_rank{rank:02d}.pt"
    done_ids = set()
    if shard_path.exists() and not args.overwrite:
        prev = torch.load(shard_path, map_location="cpu")
        done_ids = set(prev.get("ids", []))
        print(f"[rank {rank}] resumed: {len(done_ids)} ids already done", flush=True)

    todo = [r for r in my_rows if r.get("id") not in done_ids]
    if not todo:
        print(f"[rank {rank}] nothing to do", flush=True)
        if world_size > 1:
            dist.barrier(); dist.destroy_process_group()
        return

    model, mean, std, res = _load_biomedclip(args.model_id, device)
    transform = T.Compose([
        T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ])

    def _rel(img):
        return img[0] if isinstance(img, (list, tuple)) else img

    feats, ids = [], []
    if done_ids:
        prev = torch.load(shard_path, map_location="cpu")
        feats.append(prev["feats"])
        ids.extend(prev.get("ids", []))

    batch_imgs, batch_ids = [], []
    t0 = time.perf_counter()
    n_done_session = 0

    def _flush():
        nonlocal batch_imgs, batch_ids, n_done_session
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device=device, dtype=torch.bfloat16)
        x = (x - mean) / std
        with torch.inference_mode():
            f = model(x)
        feats.append(f.float().cpu())
        ids.extend(batch_ids)
        n_done_session += len(batch_ids)
        batch_imgs.clear(); batch_ids.clear()

    for i, r in enumerate(todo):
        p = os.path.join(args.data_dir, _rel(r["image"]))
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[rank {rank}] skip {p}: {e}", flush=True)
            continue
        batch_imgs.append(transform(im))
        batch_ids.append(r["id"])
        if len(batch_imgs) >= args.batch_size:
            _flush()
            if (i // args.batch_size) % 10 == 0:
                rate = n_done_session / max(time.perf_counter() - t0, 1e-3)
                eta_h = (len(todo) - n_done_session) / max(rate, 1e-3) / 3600
                print(f"[rank {rank}] {n_done_session}/{len(todo)} "
                      f"({rate:.1f} img/s, ETA {eta_h:.2f}h)", flush=True)
                torch.save({"feats": torch.cat(feats, dim=0), "ids": ids}, shard_path)
    _flush()
    torch.save({"feats": torch.cat(feats, dim=0), "ids": ids}, shard_path)
    print(f"[rank {rank}] done -> {shard_path} ({len(ids)} feats)", flush=True)
    if world_size > 1:
        dist.barrier(); dist.destroy_process_group()


def run_cluster(args):
    from sklearn.cluster import KMeans
    shards = sorted(Path(args.out_dir).glob("feats_rank*.pt"))
    if not shards:
        sys.exit(f"no feats_rank*.pt under {args.out_dir} -- run --phase features first")
    feats, ids = [], []
    for s in shards:
        d = torch.load(s, map_location="cpu")
        feats.append(d["feats"])
        ids.extend(d.get("ids", []))
    X = torch.cat(feats, dim=0).numpy()
    print(f"[cluster] {X.shape[0]} feats in dim {X.shape[1]}; running KMeans(K={args.K}) ...")
    # L2-normalize so KMeans on the unit sphere ~= cosine clustering.
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=args.K, n_init=10, random_state=0, verbose=1)
    km.fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    counts = np.bincount(km.labels_, minlength=args.K)
    print(f"[cluster] cluster sizes: min={counts.min()} max={counts.max()} "
          f"mean={counts.mean():.0f} median={np.median(counts):.0f}")
    for k in range(args.K):
        print(f"  prototype {k:>2d}: n={counts[k]:>7d} ({100*counts[k]/X.shape[0]:.1f}%)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "centroids": torch.from_numpy(centroids),
        "K": args.K,
        "d_feat": int(X.shape[1]),
        "model_id": args.model_id,
        "n_samples": int(X.shape[0]),
        "cluster_counts": torch.from_numpy(counts.astype(np.int64)),
    }, args.out)
    print(f"[cluster] wrote centroids -> {args.out}")


def main():
    host = os.uname().nodename.split(".")[0]
    default_dd = {
        "cvssp-retina03": "/work/um00109/MLLM/datasets/PubMedVision",
        "ulws072": "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision",
    }.get(host, "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision")

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["features", "cluster"])
    ap.add_argument("--captions", default=os.path.join(default_dd, "PubMedVision_CachedCaptions_K4.json"))
    ap.add_argument("--data_dir", default=default_dd)
    ap.add_argument("--out_dir", default="data/prototype_shards")
    ap.add_argument("--out", default="data/prototype_centroids.pt")
    ap.add_argument("--model_id", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=50000,
                    help="cap feature-phase rows (0=all). KMeans converges well below 100k.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.phase == "features":
        run_features(args)
    else:
        run_cluster(args)


if __name__ == "__main__":
    main()
