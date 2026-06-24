"""Local VLM judge for T2I outputs — image-only, no API.

Loads Qwen2.5-VL-7B-Instruct (or any HF VLM that follows the qwen2.5-vl chat
template) and runs the same two-question protocol as `llm_judge.py`:
  1. Modality classification (no caption shown).
  2. Plausibility 1-5.

Per-modality stratified sampling via --per_modality N.

Usage:
    python corl/eval/llm_judge_local.py \\
        --method_dirs results/eval_small_cached_cap/exp7_cached_no_aux_1ep \\
                      results/eval_small_cached_cap/exp8_spherical_proto_joint \\
                      results/eval_small_cached_cap/exp6_cached_proto_joint \\
        --out_dir results/eval_small_cached_cap/judge_qwen25vl \\
        --per_modality 8
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

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


def load_qwen(model_id: str, device: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    return model, processor


@torch.inference_mode()
def judge_one(model, processor, image_path: str, device: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    try:
        start, end = decoded.find("{"), decoded.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON: {decoded!r}")
        parsed = json.loads(decoded[start:end + 1])
        return {
            "modality_pred": parsed.get("modality"),
            "plausibility": int(parsed.get("plausibility", 0)),
            "raw": decoded,
        }
    except Exception as e:
        return {"error": str(e), "raw": decoded}


def gather_rows(method_dir: str, per_modality: int | None, limit: int | None):
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
    return rows


def judge_method(model, processor, method_dir: str, out_path: str, device: str,
                 per_modality: int | None, limit: int | None):
    rows = gather_rows(method_dir, per_modality, limit)
    print(f"[judge] {method_dir}: {len(rows)} rows")

    scores = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            scores = {r["id"]: r for r in json.load(f)}
        print(f"[judge]   resuming with {len(scores)} cached scores")

    todo = [r for r in rows if r["id"] not in scores]
    t0 = time.perf_counter()
    for i, r in enumerate(todo):
        res = judge_one(model, processor, r["gen_path"], device)
        scores[r["id"]] = {
            "id": r["id"],
            "modality_gt": MOD_MAP.get(r.get("modality"), r.get("modality")),
            **res,
        }
        if (i + 1) % 10 == 0:
            with open(out_path, "w") as f:
                json.dump(list(scores.values()), f)
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            eta = (len(todo) - i - 1) / max(rate, 1e-6) / 60
            print(f"[judge]   {i+1}/{len(todo)} ({rate:.2f}/s, ETA {eta:.1f} min)")
    with open(out_path, "w") as f:
        json.dump(list(scores.values()), f)
    print(f"[judge]   wrote {len(scores)} -> {out_path}")


def aggregate(score_path: str, name: str) -> dict:
    with open(score_path) as f:
        rows = json.load(f)
    rows = [r for r in rows if "error" not in r]
    by_mod = {}
    for r in rows:
        m = r["modality_gt"]
        by_mod.setdefault(m, []).append(r)
    out = {"name": name, "n": len(rows), "per_modality": {}}
    acc_macro, pl_macro = [], []
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
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--per_modality", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = (None, None)
    if not args.aggregate_only:
        print(f"[judge] loading {args.model_id} on {device}")
        model, processor = load_qwen(args.model_id, device)

    summary = []
    for md in args.method_dirs:
        name = Path(md).name
        out_path = os.path.join(args.out_dir, f"{name}_scores.json")
        if not args.aggregate_only:
            judge_method(model, processor, md, out_path, device,
                         args.per_modality, args.limit)
        summary.append(aggregate(out_path, name))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Qwen2.5-VL Judge summary ===")
    print(f"{'method':40s} {'mod_acc':>8s} {'plaus':>7s}  per-modality acc / plaus")
    for s in summary:
        per = "  ".join(f"{m}:{v['modality_acc']:.2f}/{v['plausibility']:.2f}"
                       for m, v in s["per_modality"].items())
        print(f"{s['name']:40s} {s['macro_modality_acc']:>8.3f} "
              f"{s['macro_plausibility']:>7.3f}  {per}")


if __name__ == "__main__":
    main()
