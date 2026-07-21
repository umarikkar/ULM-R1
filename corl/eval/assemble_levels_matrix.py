"""Assemble train-level x eval-level matrices, one per metric.

Rows = adapter trained on level (l1/l2/l3); cols = eval prompt level (l1/l2/l3).
Metrics: FID-BiomedCLIP (down), CLIPScore-gen (up), modality-accuracy (up),
plausibility (up). Writes MATRICES.md and (if matplotlib present) heatmap PNGs.
"""
import argparse
import json
from pathlib import Path

LEVELS = ["l1", "l2", "l3"]
METRICS = [
    ("FID-Inception (down)", "fidincep", "metrics", "fid_inception", 2, "down"),
    ("FID-BiomedCLIP (down)", "fid", "metrics", "fid_biomedclip", 2, "down"),
    ("CLIPScore-gen (up)", "clip", "metrics", "clipscore_gen", 4, "up"),
    ("Modality-accuracy (up)", "modacc", "judge", "macro_modality_acc", 3, "up"),
    ("Plausibility (up)", "plaus", "judge", "macro_plausibility", 2, "up"),
]


def load(root, judge_summary):
    root = Path(root)
    metrics, judge = {}, {}
    for tr in LEVELS:
        for ev in LEVELS:
            tag = f"{tr}__{ev}"
            mp = root / tag / "metrics.json"
            if mp.exists():
                metrics[tag] = json.load(open(mp)).get("macro", {})
            ip = root / tag / "metrics_inception.json"
            if ip.exists():
                metrics.setdefault(tag, {})["fid_inception"] = \
                    json.load(open(ip)).get("fid_inception")
    if judge_summary and Path(judge_summary).exists():
        for s in json.load(open(judge_summary)):
            judge[s["name"]] = s
    return metrics, judge


def value(metrics, judge, tag, src, key):
    d = (metrics if src == "metrics" else judge).get(tag, {})
    return d.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_levels")
    ap.add_argument("--judge", default="results/eval_levels/_judge/summary.json")
    ap.add_argument("--out", default="results/eval_levels/MATRICES.md")
    args = ap.parse_args()
    metrics, judge = load(args.root, args.judge)

    lines = ["# Train-level x eval-level matrices",
             "",
             "Rows = adapter trained on level; columns = eval prompt level. "
             "Diagonal = matched; first column = coarse (l1) prompt.", ""]
    grids = {}
    for title, short, src, key, prec, _ in METRICS:
        lines += [f"## {title}", "",
                  "| train ↓ \\ eval → | l1 | l2 | l3 |", "|---|---|---|---|"]
        g = [[None] * 3 for _ in range(3)]
        for i, tr in enumerate(LEVELS):
            cells = []
            for j, ev in enumerate(LEVELS):
                v = value(metrics, judge, f"{tr}__{ev}", src, key)
                g[i][j] = v
                cells.append(f"{v:.{prec}f}" if isinstance(v, (int, float)) else "—")
            lines.append(f"| **{tr}** | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
        grids[short] = (title, g)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"[matrix] wrote {out}")

    # Optional heatmaps.
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axlist = axes.ravel()
        for ax in axlist[len(METRICS):]:
            ax.axis("off")
        for ax, (title, short, *_rest) in zip(axlist, METRICS):
            _, g = grids[short]
            arr = np.array([[v if isinstance(v, (int, float)) else np.nan
                             for v in row] for row in g], dtype=float)
            direction = _rest[-1]
            cmap = "RdYlGn" if direction == "up" else "RdYlGn_r"
            im = ax.imshow(arr, cmap=cmap)
            ax.set_xticks(range(3)); ax.set_xticklabels([f"eval {l}" for l in LEVELS])
            ax.set_yticks(range(3)); ax.set_yticklabels([f"train {l}" for l in LEVELS])
            ax.set_title(title, fontsize=10)
            for i in range(3):
                for j in range(3):
                    t = "—" if np.isnan(arr[i, j]) else f"{arr[i, j]:.2f}"
                    ax.text(j, i, t, ha="center", va="center", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Janus caption-level: train x eval matrices", fontsize=12)
        fig.tight_layout()
        png = out.parent / "matrices.png"
        fig.savefig(png, dpi=130, bbox_inches="tight")
        print(f"[matrix] wrote {png}")
    except Exception as e:
        print(f"[matrix] heatmap skipped: {e}")


if __name__ == "__main__":
    main()
