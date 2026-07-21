"""Did the gran-LoRA SFT reduce caption variability / absorb content?

Hypothesis: supervising the LoRA toward *specific* greedy-decoded l1/l2/l3
captions made it memorise templated content, not just the granularity axis. If
so, at a fixed alpha the LoRA's captions across DIFFERENT images collapse toward
a template (low diversity), possibly more so than the (already templated) cached
targets it was trained on.

This scores inter-image caption diversity on HELD-OUT test-split images for:
  - BASE      (alpha=0, LoRA off)         -> the LLM's own captions
  - LoRA@0.3 / 0.6 / 0.9                  -> the granularity knob (l1/l2/l3)
  - TARGET l1 / l2 / l3                   -> the cached SFT ground-truth captions
and prints, per group, diversity metrics + example captions for eyeballing.

Compare LoRA@alpha_k vs TARGET l_k (matched granularity => length-controlled):
lower distinct-n / higher pairwise-similarity / more shared prefixes for the LoRA
than for its target == the SFT collapsed content variability into the LoRA.

    python corl/eval/diagnose_content_leak.py --adapter_dir results/GranLoRA/gran_lora_v2 \
        --data_json corl/eval/test_split_levels.json --data_dir .../PubMedVision \
        --per_modality 12 --out results/GranLoRA/content_leak_v2.json
"""
import argparse
import itertools
import json
import os
import sys

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from janus.models import VLChatProcessor
from corl.open_r1.janus_tokenizer_fix import load_fast_tokenizer

NEUTRAL_PROMPT = "Describe this medical image."
LEVEL_ALPHA = {"l1": 0.3, "l2": 0.6, "l3": 0.9}


def set_scale(model, base, alpha):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                m.scaling[adp] = base[(name, adp)] * alpha


def _toks(s):
    return s.lower().split()


def diversity(caps):
    """Inter-caption diversity metrics for a list of captions (one per image)."""
    toks = [_toks(c) for c in caps]
    uni = [t for t in toks]
    all_uni = list(itertools.chain.from_iterable(uni))
    all_bi = list(itertools.chain.from_iterable(zip(t, t[1:]) for t in uni))
    all_tri = list(itertools.chain.from_iterable(zip(t, t[1:], t[2:]) for t in uni))
    d1 = len(set(all_uni)) / max(len(all_uni), 1)
    d2 = len(set(all_bi)) / max(len(all_bi), 1)
    d3 = len(set(all_tri)) / max(len(all_tri), 1)
    # mean pairwise Jaccard over token sets (higher => more similar => less diverse)
    sets = [set(t) for t in toks]
    pairs = list(itertools.combinations(range(len(sets)), 2))
    if pairs:
        jac = sum(len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
                  for i, j in pairs) / len(pairs)
    else:
        jac = 0.0
    # shared 5-word prefix rate (direct templating signal)
    from collections import Counter
    pref = Counter(" ".join(t[:5]) for t in toks if len(t) >= 5)
    dup = sum(v for v in pref.values() if v > 1) / max(len(toks), 1)
    return {
        "n": len(caps),
        "mean_words": sum(len(t) for t in toks) / max(len(toks), 1),
        "vocab": len(set(all_uni)),
        "distinct1": round(d1, 4),
        "distinct2": round(d2, 4),
        "distinct3": round(d3, 4),
        "mean_pairwise_jaccard": round(jac, 4),
        "shared_prefix5_rate": round(dup, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--per_modality", type=int, default=12)
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda"

    rows = json.load(open(args.data_json))
    seen, sample = {}, []
    for r in rows:
        m = r.get("modality", "?")
        if seen.get(m, 0) < args.per_modality:
            p = os.path.join(args.data_dir, r["image"][0] if isinstance(r["image"], list) else r["image"])
            if os.path.exists(p):
                sample.append(r | {"_path": p})
                seen[m] = seen.get(m, 0) + 1
    print(f"[data] {len(sample)} images across {len(seen)} modalities", flush=True)

    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    processor.tokenizer = load_fast_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter_dir)
    model = model.to(device).eval()
    base = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                base[(name, adp)] = float(m.scaling[adp])
    tok = processor.tokenizer
    eos = tok.eos_token_id

    def caption(imgs, alpha):
        set_scale(model, base, alpha)
        out_caps = []
        for i in range(0, len(imgs), args.gen_batch):
            chunk = imgs[i:i + args.gen_batch]
            convs = [[{"role": "<|User|>", "content": f"<image_placeholder>\n{NEUTRAL_PROMPT}"},
                      {"role": "<|Assistant|>", "content": ""}] for _ in chunk]
            prep = processor(conversations=convs, images=[[im] for im in chunk],
                             force_batchify=True).to(device)
            with torch.inference_mode():
                emb = model.prepare_inputs_embeds(
                    input_ids=prep.input_ids, pixel_values=prep.pixel_values,
                    images_seq_mask=prep.images_seq_mask, images_emb_mask=prep.images_emb_mask)
                gen = model.language_model.generate(
                    inputs_embeds=emb, attention_mask=prep.attention_mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=eos, bos_token_id=tok.bos_token_id, eos_token_id=eos)
            out_caps += [d.strip() for d in tok.batch_decode(gen, skip_special_tokens=True)]
        return out_caps

    imgs = [Image.open(s["_path"]).convert("RGB") for s in sample]
    groups = {}
    groups["BASE(a=0)"] = caption(imgs, 0.0)
    for lv, a in LEVEL_ALPHA.items():
        groups[f"LoRA@{a}({lv})"] = caption(imgs, a)
    for lv in ("l1", "l2", "l3"):
        groups[f"TARGET_{lv}"] = [s[f"cached_captions_{lv}"] for s in sample]

    report = {"per_group": {}, "compare": {}}
    hdr = f"{'group':>16} {'n':>4} {'words':>6} {'vocab':>6} {'dist1':>6} {'dist2':>6} {'dist3':>6} {'pair_jac':>8} {'pref5':>6}"
    print("\n" + hdr)
    for g, caps in groups.items():
        d = diversity(caps)
        report["per_group"][g] = d
        print(f"{g:>16} {d['n']:>4} {d['mean_words']:>6.1f} {d['vocab']:>6} "
              f"{d['distinct1']:>6.3f} {d['distinct2']:>6.3f} {d['distinct3']:>6.3f} "
              f"{d['mean_pairwise_jaccard']:>8.3f} {d['shared_prefix5_rate']:>6.3f}")

    print("\n[compare LoRA vs its SFT TARGET at matched granularity]"
          "  (LoRA lower distinct / higher jac,pref => collapsed content)")
    for lv, a in LEVEL_ALPHA.items():
        L = report["per_group"][f"LoRA@{a}({lv})"]
        T = report["per_group"][f"TARGET_{lv}"]
        cmp = {k: round(L[k] - T[k], 4) for k in ("distinct2", "distinct3",
               "mean_pairwise_jaccard", "shared_prefix5_rate", "mean_words")}
        report["compare"][lv] = {"LoRA": L, "TARGET": T, "LoRA_minus_TARGET": cmp}
        print(f"  {lv}: dist2 {L['distinct2']:.3f} vs {T['distinct2']:.3f} (Δ{cmp['distinct2']:+.3f}) | "
              f"pair_jac {L['mean_pairwise_jaccard']:.3f} vs {T['mean_pairwise_jaccard']:.3f} "
              f"(Δ{cmp['mean_pairwise_jaccard']:+.3f}) | pref5 {L['shared_prefix5_rate']:.3f} vs "
              f"{T['shared_prefix5_rate']:.3f} (Δ{cmp['shared_prefix5_rate']:+.3f})")

    # eyeball: same alpha across different images (are they templated?)
    print("\n--- LoRA@0.6 (l2) captions across 6 different images ---")
    ex = groups["LoRA@0.6(l2)"]
    for i in range(0, len(ex), max(1, len(ex) // 6))[:6]:
        print(f"  [{sample[i]['modality']}] {ex[i][:150]}")
    report["examples_lora_l2"] = [{"modality": sample[i]["modality"], "caption": ex[i]}
                                  for i in range(min(len(ex), 12))]

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
