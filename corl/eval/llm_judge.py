"""LLM judge for T2I outputs — image-only evaluation.

For each generated image, asks a vision LLM (Claude) two questions WITHOUT
showing the caption:
  1. What is the imaging modality? (classification)
  2. How plausible is this as a real medical image? (1-5)

The judge has no relationship to either the training caption distribution or
the eval caption distribution. So scores are independent of caption-OOD
effects and reflect pure image quality / domain correctness.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python corl/eval/llm_judge.py \\
        --method_dirs results/eval_small/vanilla results/eval_small/exp1_pubmed_captions ... \\
        --out_dir   results/eval_small/judge \\
        --model    claude-opus-4-7 \\
        --concurrency 8
"""

import argparse
import base64
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# PubMedVision modality strings -> short labels we use in the judge prompt.
MOD_MAP = {
    "Computed Tomography": "CT",
    "Magnetic Resonance Imaging": "MRI",
    "Microscopy Images": "Microscopy",
    "Ultrasound": "Ultrasound",
    "Endoscopy": "Endoscopy",
    "Fundus Photography": "Fundus",
}
SHORT_LABELS = sorted(set(MOD_MAP.values()))
LABELS_FOR_PROMPT = ", ".join(SHORT_LABELS)

PROMPT = (
    "You are a medical imaging expert. Look at this image and answer two questions.\n\n"
    "1. What is the imaging modality? Choose ONE from: " + LABELS_FOR_PROMPT + ", "
    "or 'Other' if none of these apply.\n\n"
    "2. How plausible is this as a real medical image? Rate on a 1-5 scale:\n"
    "   1 = Not medical at all (looks like an object, person, or natural scene)\n"
    "   2 = Has some medical visual elements but mostly implausible\n"
    "   3 = Roughly medical-looking but with clear artifacts or wrong content\n"
    "   4 = Plausible medical image with minor issues\n"
    "   5 = Looks like a real medical image\n\n"
    "Respond with a SINGLE LINE of JSON, nothing else, like:\n"
    '{"modality": "CT", "plausibility": 4}'
)


def encode_image(path: str) -> tuple[str, str]:
    """Return (media_type, base64_data) for the image file."""
    with open(path, "rb") as f:
        b = f.read()
    mt = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return mt, base64.standard_b64encode(b).decode("utf-8")


def call_judge(client, model: str, image_path: str, max_retries: int = 3) -> dict:
    """Single judge call. Returns parsed dict or {'error': ...}."""
    media_type, b64 = encode_image(image_path)
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": media_type, "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            text = resp.content[0].text.strip()
            # Try to extract JSON line.
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"no JSON in response: {text!r}")
            parsed = json.loads(text[start:end + 1])
            return {
                "modality_pred": parsed.get("modality"),
                "plausibility": int(parsed.get("plausibility", 0)),
                "raw": text,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(1.5 * (attempt + 1))


def judge_method(method_dir: str, out_path: str, model: str, concurrency: int,
                 limit: int | None = None, per_modality: int | None = None):
    """Run judge over all images in a method dir; save per-row scores."""
    import anthropic
    client = anthropic.Anthropic()

    rows = []
    for p in sorted(glob.glob(os.path.join(method_dir, "manifest_shard*.json"))):
        with open(p) as f:
            rows.extend(json.load(f))
    if per_modality is not None:
        by_mod = {}
        for r in rows:
            by_mod.setdefault(r.get("modality"), []).append(r)
        balanced = []
        for m in sorted(by_mod):
            balanced.extend(sorted(by_mod[m], key=lambda r: r["id"])[:per_modality])
        rows = balanced
    if limit:
        rows = rows[:limit]
    print(f"[judge] {method_dir}: {len(rows)} rows")

    # Resume: load existing scores and skip done rows.
    scores = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            scores = {r["id"]: r for r in json.load(f)}
        print(f"[judge]   resuming with {len(scores)} cached scores")

    todo = [r for r in rows if r["id"] not in scores]
    t0 = time.perf_counter()
    n_done = 0
    write_every = 25

    def _work(row):
        res = call_judge(client, model, row["gen_path"])
        return {
            "id": row["id"],
            "modality_gt": MOD_MAP.get(row.get("modality"), row.get("modality")),
            **res,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_work, r): r for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                row = futs[fut]
                rec = {"id": row["id"], "error": str(e)}
            scores[rec["id"]] = rec
            n_done += 1
            if n_done % write_every == 0:
                with open(out_path, "w") as f:
                    json.dump(list(scores.values()), f)
                dt = time.perf_counter() - t0
                rate = n_done / dt
                eta = (len(todo) - n_done) / max(rate, 1e-6) / 60
                print(f"[judge]   {n_done}/{len(todo)} done ({rate:.1f}/s, ETA {eta:.1f} min)")

    with open(out_path, "w") as f:
        json.dump(list(scores.values()), f)
    print(f"[judge]   wrote {len(scores)} scores -> {out_path}")


def aggregate(score_path: str, name: str) -> dict:
    """Per-modality accuracy + mean plausibility from a judge_scores.json."""
    with open(score_path) as f:
        rows = json.load(f)
    rows = [r for r in rows if "error" not in r]
    by_mod = {}
    for r in rows:
        m = r["modality_gt"]
        by_mod.setdefault(m, []).append(r)
    out = {"name": name, "n": len(rows), "per_modality": {}}
    acc_macro = []
    pl_macro = []
    for m, mr in sorted(by_mod.items()):
        acc = sum(1 for r in mr if r.get("modality_pred") == m) / len(mr)
        pl = sum(r.get("plausibility", 0) for r in mr) / len(mr)
        out["per_modality"][m] = {"n": len(mr), "modality_acc": acc, "plausibility": pl}
        acc_macro.append(acc)
        pl_macro.append(pl)
    out["macro_modality_acc"] = sum(acc_macro) / max(len(acc_macro), 1)
    out["macro_plausibility"] = sum(pl_macro) / max(len(pl_macro), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap per method (for smoke testing).")
    ap.add_argument("--per_modality", type=int, default=None,
                    help="Stratified: take N samples per modality from each method.")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    summary = []
    for md in args.method_dirs:
        name = Path(md).name
        out_path = os.path.join(args.out_dir, f"{name}_scores.json")
        if not args.aggregate_only:
            judge_method(md, out_path, args.model, args.concurrency,
                         args.limit, args.per_modality)
        summary.append(aggregate(out_path, name))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LLM Judge summary ===")
    print(f"{'method':32s} {'mod_acc':>8s} {'plaus':>7s}  per-modality acc / plaus")
    for s in summary:
        per = "  ".join(f"{m}:{v['modality_acc']:.2f}/{v['plausibility']:.2f}"
                       for m, v in s["per_modality"].items())
        print(f"{s['name']:32s} {s['macro_modality_acc']:>8.3f} "
              f"{s['macro_plausibility']:>7.3f}  {per}")


if __name__ == "__main__":
    main()
