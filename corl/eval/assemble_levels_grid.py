"""Assemble the Janus caption-level eval grid into summary tables.

Reads per-run metrics.json (FID/CLIPScore-BiomedCLIP from compute_metrics.py) and
the local-judge summary.json (per-modality accuracy + plausibility), then writes
SUMMARY.md with one table per prompt condition:
    matched         : l1<-l1, l2<-l2, l3<-l3     (end-to-end systems)
    fixed-Original  : all adapters <- Original_Caption   (training-effect isolation)
    fixed-l1        : all adapters <- l1 caption          (modality-fidelity probe)
"""
import argparse
import json
from pathlib import Path

# tag -> (adapter_level, prompt_label)
TAGS = {
    "l1__l1": ("l1", "l1"), "l2__l2": ("l2", "l2"), "l3__l3": ("l3", "l3"),
    "l1__orig": ("l1", "orig"), "l2__orig": ("l2", "orig"), "l3__orig": ("l3", "orig"),
    "l2__l1": ("l2", "l1"), "l3__l1": ("l3", "l1"),
}
CONDITIONS = {
    "Matched (adapter <- its own level)": ["l1__l1", "l2__l2", "l3__l3"],
    "Fixed Original_Caption": ["l1__orig", "l2__orig", "l3__orig"],
    "Fixed l1 (coarse) prompt": ["l1__l1", "l2__l1", "l3__l1"],
}


def load(root, judge_summary):
    root = Path(root)
    metrics = {}
    for tag in TAGS:
        mp = root / tag / "metrics.json"
        if mp.exists():
            metrics[tag] = json.load(open(mp)).get("macro", {})
    judge = {}
    if judge_summary and Path(judge_summary).exists():
        for s in json.load(open(judge_summary)):
            judge[s["name"]] = s
    return metrics, judge


def cell(metrics, judge, tag):
    m = metrics.get(tag, {})
    j = judge.get(tag, {})
    fid = m.get("fid_biomedclip")
    cs = m.get("clipscore_gen")
    acc = j.get("macro_modality_acc")
    pl = j.get("macro_plausibility")
    f = lambda v, p: (f"{v:.{p}f}" if isinstance(v, (int, float)) else "—")
    return f(fid, 2), f(cs, 4), f(acc, 3), f(pl, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_levels")
    ap.add_argument("--judge", default="results/eval_levels/_judge/summary.json")
    ap.add_argument("--out", default="results/eval_levels/SUMMARY.md")
    args = ap.parse_args()

    metrics, judge = load(args.root, args.judge)
    lines = ["# Janus caption-level eval grid",
             "",
             "FID-BiomedCLIP (↓), CLIPScore-BiomedCLIP-gen (↑), "
             "Qwen modality-accuracy (↑), plausibility (↑). Macro-averaged over "
             "6 modalities.", ""]
    for cond, tags in CONDITIONS.items():
        lines += [f"## {cond}", "",
                  "| adapter | FID ↓ | CLIPScore ↑ | Mod-acc ↑ | Plaus ↑ |",
                  "|---|---|---|---|---|"]
        for tag in tags:
            adp = TAGS[tag][0]
            fid, cs, acc, pl = cell(metrics, judge, tag)
            lines.append(f"| {adp} | {fid} | {cs} | {acc} | {pl} |")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    # Also dump a machine-readable merge.
    merged = {tag: {"adapter": TAGS[tag][0], "prompt": TAGS[tag][1],
                    "macro": metrics.get(tag, {}),
                    "judge": {k: judge.get(tag, {}).get(k)
                              for k in ("macro_modality_acc", "macro_plausibility")}}
              for tag in TAGS}
    (out.parent / "summary_grid.json").write_text(json.dumps(merged, indent=2))
    print("\n".join(lines))
    print(f"\n[assemble] wrote {out} and summary_grid.json")


if __name__ == "__main__":
    main()
