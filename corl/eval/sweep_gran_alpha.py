"""Validate the gran-LoRA granularity knob: sweep alpha and measure granularity.

Loads Janus + a trained gran-LoRA, then for a fixed modality-balanced sample
captions each image (greedy, fixed neutral prompt) at several LoRA scales alpha.
alpha=0 = LoRA off = base default caption. Reports mean caption length per alpha
(the grounding check: it should be MONOTONE in alpha) plus example captions.

    python corl/eval/sweep_gran_alpha.py --adapter_dir results/GranLoRA/gran_lora_v1 \
        --data_json .../PubMedVision_CachedCaptions_Levels.json --data_dir .../PubMedVision
"""
import argparse
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


def _fix(s):
    return s.replace("Ġ", " ").replace("Ċ", "\n").strip()


def set_scale(model, base, alpha):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                m.scaling[adp] = base[(name, adp)] * alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--per_modality", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    device = "cuda"

    # modality-balanced sample
    rows = json.load(open(args.data_json))
    seen, sample = {}, []
    for r in rows:
        m = r.get("modality", "?")
        if seen.get(m, 0) < args.per_modality:
            p = os.path.join(args.data_dir, r["image"][0] if isinstance(r["image"], list) else r["image"])
            if os.path.exists(p):
                sample.append({"id": r["id"], "path": p, "modality": m})
                seen[m] = seen.get(m, 0) + 1

    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    processor.tokenizer = load_fast_tokenizer(args.model)  # spaces-preserving encode
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter_dir)
    model = model.to(device).eval()
    base = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                base[(name, adp)] = float(m.scaling[adp])

    eos = processor.tokenizer.eos_token_id
    imgs = [Image.open(s["path"]).convert("RGB") for s in sample]

    def caption_all(alpha):
        set_scale(model, base, alpha)
        convs = [[{"role": "<|User|>", "content": f"<image_placeholder>\n{NEUTRAL_PROMPT}"},
                  {"role": "<|Assistant|>", "content": ""}] for _ in imgs]
        prep = processor(conversations=convs, images=[[i] for i in imgs],
                         force_batchify=True).to(device)
        with torch.inference_mode():
            emb = model.prepare_inputs_embeds(
                input_ids=prep.input_ids, pixel_values=prep.pixel_values,
                images_seq_mask=prep.images_seq_mask, images_emb_mask=prep.images_emb_mask)
            out = model.language_model.generate(
                inputs_embeds=emb, attention_mask=prep.attention_mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=eos, bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=eos)
        return [_fix(d) for d in processor.tokenizer.batch_decode(out, skip_special_tokens=True)]

    report = {"alphas": alphas, "per_alpha": {}}
    print(f"{'alpha':>6} {'mean_words':>10} {'mean_chars':>10}")
    for a in alphas:
        caps = caption_all(a)
        wl = [len(c.split()) for c in caps]
        cl = [len(c) for c in caps]
        mean_w = sum(wl) / len(wl)
        report["per_alpha"][a] = {"mean_words": mean_w,
                                  "mean_chars": sum(cl) / len(cl),
                                  "examples": [{"modality": sample[i]["modality"], "caption": caps[i]}
                                               for i in range(min(3, len(caps)))]}
        print(f"{a:>6.2f} {mean_w:>10.1f} {sum(cl)/len(cl):>10.1f}")

    # monotonicity check on mean_words. alpha=0 is the LoRA-*off* base default
    # caption, which is outside the knob's learned range (l1..l3 map to alpha>0),
    # so also report the verdict over the trained alphas (>0) — that is the real
    # grounding check.
    def _verdict(al):
        v = [report["per_alpha"][a]["mean_words"] for a in al]
        up = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
        dn = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
        return "monotone-increasing" if up else "monotone-decreasing" if dn else "NON-MONOTONE"

    report["monotonic"] = _verdict(alphas)
    trained = [a for a in alphas if a > 0]
    report["monotonic_trained"] = _verdict(trained) if len(trained) > 1 else "n/a"
    print(f"\n[verdict] mean_words vs alpha (all):        {report['monotonic']}")
    print(f"[verdict] mean_words vs alpha (trained, >0): {report['monotonic_trained']}")
    print("\n--- example captions per alpha (modality: caption) ---")
    for a in alphas:
        ex = report["per_alpha"][a]["examples"][0]
        print(f"  a={a:.1f} [{ex['modality']}]: {ex['caption'][:140]}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
