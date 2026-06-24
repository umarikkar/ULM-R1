"""Build a stratified PubMedVision test split for T2I evaluation.

Drops the same is_grid=='multi' rows used at training time, then stratifies
across a fixed set of medical-relevant modalities. The output JSON has one
row per image: {id, image, modality, body_part, Original_Caption}.

Usage:
    python corl/eval/build_test_split.py \
        --data_dir /path/to/PubMedVision \
        --sidecar  data/attribute_sidecar.json \
        --out      corl/eval/test_split.json
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# 6 medical-relevant modalities. Heterogeneous buckets like "Others" and
# "Digital Photography" are dropped to keep per-modality metrics meaningful.
TARGET_MODALITIES = [
    "Computed Tomography",
    "Magnetic Resonance Imaging",
    "Microscopy Images",       # histopathology
    "Ultrasound",
    "Endoscopy",
    "Fundus Photography",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str,
                    default="/vol/research/fmodel_medical/people/umar/datasets/PubMedVision")
    ap.add_argument("--caption_json", type=str, default="PubMedVision_Original_Caption.json",
                    help="Filename inside data_dir.")
    ap.add_argument("--sidecar", type=str, default="data/attribute_sidecar.json",
                    help="Path to attribute sidecar (used only for is_grid filter).")
    ap.add_argument("--out", type=str, default="corl/eval/test_split.json")
    ap.add_argument("--per_modality", type=int, default=833,
                    help="Samples per modality (6 mods * 833 ~= 5000).")
    ap.add_argument("--min_per_modality", type=int, default=200,
                    help="Fail if any modality has fewer than this many candidates.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    cap_path = os.path.join(args.data_dir, args.caption_json)
    with open(cap_path) as f:
        rows = json.load(f)
    print(f"[split] loaded {len(rows):,} rows from {cap_path}")

    with open(args.sidecar) as f:
        side = {r["id"]: r for r in json.load(f)}
    print(f"[split] loaded {len(side):,} sidecar rows from {args.sidecar}")

    rows = [r for r in rows if side.get(r["id"], {}).get("is_grid") != "multi"]
    print(f"[split] {len(rows):,} after grid filter")

    by_mod = defaultdict(list)
    for r in rows:
        by_mod[r.get("modality")].append(r)

    selected = []
    chosen_ids = set()
    for mod in TARGET_MODALITIES:
        cands = by_mod.get(mod, [])
        if len(cands) < args.min_per_modality:
            raise RuntimeError(
                f"modality '{mod}' has {len(cands)} candidates, below min={args.min_per_modality}"
            )
        k = min(args.per_modality, len(cands))
        # Stratify within modality by body_part when possible.
        by_bp = defaultdict(list)
        for r in cands:
            by_bp[r.get("body_part") or "unknown"].append(r)
        # Round-robin draw across body_part buckets so rare anatomies survive.
        bps = list(by_bp.keys())
        for v in by_bp.values():
            random.shuffle(v)
        picked = []
        bp_idx = 0
        while len(picked) < k and any(by_bp.values()):
            bp = bps[bp_idx % len(bps)]
            if by_bp[bp]:
                picked.append(by_bp[bp].pop())
            bp_idx += 1
        for r in picked:
            chosen_ids.add(r["id"])
            # Pick first image when there are multiple panels (we already dropped
            # is_grid=='multi'; remaining multi-image rows are non-grid panels).
            img = r["image"][0] if isinstance(r["image"], list) else r["image"]
            selected.append({
                "id": r["id"],
                "image": img,
                "modality": r.get("modality"),
                "body_part": r.get("body_part"),
                "Original_Caption": r.get("Original_Caption"),
            })
        print(f"[split]   {mod}: picked {len(picked)} / {len(cands)} "
              f"across {len(bps)} body_part buckets")

    random.shuffle(selected)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"[split] wrote {len(selected):,} rows -> {out_path}")

    # Brief summary so the user can sanity-check.
    mod_counts = Counter(r["modality"] for r in selected)
    bp_counts = Counter(r["body_part"] for r in selected)
    print("[split] modality counts:")
    for m, c in mod_counts.most_common():
        print(f"           {c:>5}  {m}")
    print(f"[split] body_part buckets: {len(bp_counts)}")
    print(f"[split] unique ids: {len(chosen_ids)}")


if __name__ == "__main__":
    main()