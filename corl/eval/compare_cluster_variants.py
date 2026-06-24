"""Compare multiple centroid files on a single sample.

For each variant in --variants_dir, runs argmax assignment of the test images
and reports:
  - effective K (#clusters with >= 10 samples)
  - min/median/max cluster size
  - macro modality purity (mean of top-1 modality fraction per cluster)
  - share of samples in mixed (<60% single modality) clusters
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

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_biomedclip(device):
    import open_clip
    m, _ = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    m = m.visual.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    try:
        res = int(m.image_size if isinstance(m.image_size, int) else m.image_size[0])
    except Exception:
        res = 224
    return m, res


def prep(img: Image.Image, res, device):
    img = img.convert("RGB").resize((res, res), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    arr = arr.unsqueeze(0).to(device)
    return (arr - CLIP_MEAN.to(device)) / CLIP_STD.to(device)


@torch.inference_mode()
def encode_all(rows, data_dir, device, dtype=torch.float32):
    m, res = load_biomedclip(device)
    if dtype == torch.bfloat16:
        m = m.to(dtype=torch.bfloat16)
    feats = []
    keep_rows = []
    for i, r in enumerate(rows):
        p = os.path.join(data_dir, r["image"])
        if not os.path.exists(p):
            continue
        try:
            x = prep(Image.open(p), res, device).to(dtype=dtype)
            f = m(x).float()
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
            keep_rows.append(r)
        except Exception as e:
            print(f"  {r['id']}: {e}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  encoded {i+1}/{len(rows)}", file=sys.stderr)
    return torch.cat(feats, dim=0), keep_rows


def summarise(rows, labels, K):
    by_p = defaultdict(list)
    for r, k in zip(rows, labels.tolist()):
        by_p[k].append(r)
    sizes = np.array([len(by_p.get(k, [])) for k in range(K)])
    effK = int((sizes >= 10).sum())
    purities = []
    pop_mixed = 0
    for k in range(K):
        mr = by_p.get(k, [])
        if not mr: continue
        mods = Counter(r["modality"] for r in mr)
        top_frac = mods.most_common(1)[0][1] / len(mr)
        purities.append(top_frac)
        if top_frac < 0.6:
            pop_mixed += len(mr)
    macro_purity = float(np.mean(purities)) if purities else 0.0
    pop_mixed_frac = pop_mixed / len(rows)
    return {
        "K_nominal": K,
        "K_effective_ge10": effK,
        "min_size": int(sizes.min()),
        "median_size": int(np.median(sizes)),
        "max_size": int(sizes.max()),
        "macro_modality_purity": macro_purity,
        "pop_in_mixed_clusters_frac": pop_mixed_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants_dir", default="data/prototype_variants")
    ap.add_argument("--rows", default="corl/eval/test_split_small.json")
    ap.add_argument("--data_dir", default="/work/um00109/MLLM/datasets/PubMedVision")
    ap.add_argument("--out", default="results/eval_small/cluster_variants_summary.json")
    ap.add_argument("--feat_cache", default="results/eval_small/test_split_small_biomedclip_feats.pt")
    args = ap.parse_args()

    rows = json.load(open(args.rows))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if Path(args.feat_cache).exists():
        print(f"[cache] loading cached features from {args.feat_cache}")
        c = torch.load(args.feat_cache, map_location="cpu")
        feats, rows_kept = c["feats"], c["rows"]
    else:
        print(f"[encode] BiomedCLIP on {len(rows)} test images ({device})")
        feats, rows_kept = encode_all(rows, args.data_dir, device,
                                      dtype=(torch.bfloat16 if device == "cuda" else torch.float32))
        Path(args.feat_cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": feats, "rows": rows_kept}, args.feat_cache)
        print(f"[cache] wrote {args.feat_cache}")
    feats = feats.float()
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)

    summaries = []
    for v in sorted(Path(args.variants_dir).glob("*.pt")):
        d = torch.load(v, map_location="cpu")
        centroids = d["centroids"].float()
        centroids = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-8)
        K = int(d["K"])
        labels = (feats @ centroids.t()).argmax(dim=-1)
        s = summarise(rows_kept, labels, K)
        s["variant"] = v.name
        s["method"] = d.get("method", "?")
        summaries.append(s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summaries, open(args.out, "w"), indent=2)

    print(f"\n{'variant':36s} {'K':>3s} {'effK':>4s} {'sz min/med/max':>16s} {'purity':>7s} {'mixed%':>7s}")
    print("-" * 80)
    for s in summaries:
        sz = f"{s['min_size']}/{s['median_size']}/{s['max_size']}"
        print(f"{s['variant']:36s} {s['K_nominal']:>3d} {s['K_effective_ge10']:>4d} "
              f"{sz:>16s} {s['macro_modality_purity']:>7.3f} {100*s['pop_in_mixed_clusters_frac']:>6.1f}%")


if __name__ == "__main__":
    main()