"""Compute LPIPS between original (left) and regenerated (right) eval samples."""
import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


GEN_SIZE = 384
GAP = 10


def split_sample(path: Path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    assert h == GEN_SIZE, f"unexpected height {h} in {path}"
    orig_w = w - GAP - GEN_SIZE
    orig = img.crop((0, 0, orig_w, h))
    gen = img.crop((orig_w + GAP, 0, w, h))
    orig = orig.resize((GEN_SIZE, GEN_SIZE), Image.Resampling.BICUBIC)
    return orig, gen


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def step_key(p: Path) -> int:
    m = re.search(r"step_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="results/JanusPro-1B-T2I-Stage2-LoRA/20260522_101529_self_distill_caption/eval_samples",
    )
    ap.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.root)
    step_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=step_key)
    if not step_dirs:
        raise SystemExit(f"no step_* dirs under {root}")

    lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.net, normalize=False).to(args.device)
    lpips.eval()

    print(f"{'step':>8} {'n':>4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    rows = []
    with torch.no_grad():
        for d in step_dirs:
            samples = sorted(d.glob("sample_*.png"))
            if not samples:
                continue
            vals = []
            for s in samples:
                orig, gen = split_sample(s)
                a = to_tensor(orig).to(args.device)
                b = to_tensor(gen).to(args.device)
                v = lpips(a, b).item()
                vals.append(v)
            vals = np.array(vals)
            step = step_key(d)
            rows.append((step, vals))
            print(f"{step:>8d} {len(vals):>4d} {vals.mean():>8.4f} {vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

    out_csv = root / "lpips_per_step.csv"
    with out_csv.open("w") as f:
        n_samples = max(len(v) for _, v in rows)
        header = ["step", "mean", "std"] + [f"s{i:02d}" for i in range(n_samples)]
        f.write(",".join(header) + "\n")
        for step, vals in rows:
            cells = [str(step), f"{vals.mean():.6f}", f"{vals.std():.6f}"]
            cells += [f"{v:.6f}" for v in vals]
            f.write(",".join(cells) + "\n")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
