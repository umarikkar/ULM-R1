"""Compare T2I reconstruction quality across runs at a common eval step.

Each eval PNG is side-by-side [original | gap | generated]. We split it, then
score generated-vs-original with LPIPS (lower=better), SSIM (higher=better) and
PSNR (higher=better). Produces a per-run table and a visual montage
(rows = samples, cols = [GT, run_1, run_2, ...]).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

GEN_SIZE = 384
GAP = 10
RESULTS_ROOT = "results/JanusPro-1B-T2I-Stage2-LoRA"

# label -> run directory name
RUNS = {
    "no_perc (pw0)": "20260527_170518_original_cached_captions_pw0",
    "final-only (pw0.5)": "20260527_170518_original_cached_captions_pw0.5",
    "multilayer (pw0.25, l3-6-9)": "20260528_112510_original_cached_captions_pw0.25_pl3-6-9",
}


def split_sample(path: Path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    assert h == GEN_SIZE, f"unexpected height {h} in {path}"
    orig_w = w - GAP - GEN_SIZE
    orig = img.crop((0, 0, orig_w, h)).resize((GEN_SIZE, GEN_SIZE), Image.Resampling.BICUBIC)
    gen = img.crop((orig_w + GAP, 0, w, h))
    return orig, gen


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def common_sample_indices(step_dirs):
    """Sample indices present in every run at this step."""
    sets = []
    for d in step_dirs:
        idx = {int(p.stem.split("_")[1]) for p in d.glob("sample_*.png")}
        sets.append(idx)
    return sorted(set.intersection(*sets)) if sets else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=25000, help="eval step common to all runs")
    ap.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="cpu", help="cpu to avoid disturbing training GPUs")
    ap.add_argument("--out", default=f"{RESULTS_ROOT}/compare_step{{step}}.png")
    args = ap.parse_args()

    dev = args.device
    step_tag = f"step_{args.step:06d}"
    step_dirs = {lbl: Path(RESULTS_ROOT) / rd / "eval_samples" / step_tag for lbl, rd in RUNS.items()}
    for lbl, d in step_dirs.items():
        if not d.is_dir():
            raise SystemExit(f"missing {d} for run '{lbl}'")

    idxs = common_sample_indices(list(step_dirs.values()))
    if not idxs:
        raise SystemExit("no sample indices common to all runs at this step")
    print(f"Comparing step {args.step} over {len(idxs)} common samples: {idxs}\n")

    lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.net, normalize=False).to(dev).eval()
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(dev)   # data in [-1,1] -> range 2
    psnr = PeakSignalNoiseRatio(data_range=2.0).to(dev)

    # gen tensors cached for the montage; metrics accumulated per run.
    summary = {}
    gens_for_montage = {}
    gt_for_montage = None
    with torch.no_grad():
        for lbl, d in step_dirs.items():
            l_vals, s_vals, p_vals = [], [], []
            gens_for_montage[lbl] = {}
            gt_cache = {}
            for i in idxs:
                orig, gen = split_sample(d / f"sample_{i:02d}.png")
                a = to_tensor(orig).to(dev)   # GT
                b = to_tensor(gen).to(dev)    # generated
                l_vals.append(lpips(b, a).item())
                s_vals.append(ssim(b, a).item())
                p_vals.append(psnr(b, a).item())
                gens_for_montage[lbl][i] = gen
                gt_cache[i] = orig
            if gt_for_montage is None:
                gt_for_montage = gt_cache
            summary[lbl] = {
                "LPIPS": np.array(l_vals),
                "SSIM": np.array(s_vals),
                "PSNR": np.array(p_vals),
            }

    # ---- table ----
    print(f"{'run':<30} {'LPIPS↓':>16} {'SSIM↑':>16} {'PSNR↑(dB)':>16}")
    print("-" * 80)
    for lbl, m in summary.items():
        print(f"{lbl:<30} "
              f"{m['LPIPS'].mean():>7.4f}±{m['LPIPS'].std():<7.4f} "
              f"{m['SSIM'].mean():>7.4f}±{m['SSIM'].std():<7.4f} "
              f"{m['PSNR'].mean():>7.3f}±{m['PSNR'].std():<7.3f}")
    print()

    # ---- montage: rows = samples, cols = [GT, run1, run2, run3] ----
    labels = ["GT"] + list(RUNS.keys())
    ncol, nrow = len(labels), len(idxs)
    pad, header = 6, 22
    cell = GEN_SIZE
    W = ncol * cell + (ncol + 1) * pad
    H = header + nrow * cell + (nrow + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for c, lbl in enumerate(labels):
        x = pad + c * (cell + pad)
        draw.text((x + 4, 4), lbl[:46], fill=(0, 0, 0))
    for r, i in enumerate(idxs):
        y = header + pad + r * (cell + pad)
        canvas.paste(gt_for_montage[i], (pad, y))
        for c, lbl in enumerate(RUNS.keys(), start=1):
            x = pad + c * (cell + pad)
            canvas.paste(gens_for_montage[lbl][i], (x, y))
    out = args.out.format(step=args.step)
    canvas.save(out)
    print(f"wrote montage -> {out}")


if __name__ == "__main__":
    main()
