"""Pick the FIXED anchor prompt for the gran-LoRA knob (the alpha=0 / base caption).

The prompt is shared across all alpha (alpha carries granularity, not the prompt) and
is the KL anchor for RL. alpha only *compresses* from the base, so alpha=0 is the
quality+verbosity ceiling -> pick the prompt that maximises a good, FAITHFUL verbose
caption from the BASE Janus model (no LoRA), once, before RL.

Two phases (so we never hold both models):
  1. generate: base Janus captions each sampled image under every candidate prompt.
  2. judge   : Qwen2.5-VL scores each (image, caption) for faithfulness + hallucination;
               plus cheap non-judge stats (token length, repetition, modality mention).

Output: a ranked table. Higher faithfulness, fewer hallucinations, and a base length
already near the verbose target (l3~137 tok) are all good (less RL work at alpha=0).

    python corl/eval/select_anchor_prompt.py \
        --data_json corl/eval/test_split_levels.json --data_dir .../PubMedVision \
        --per_modality 8 --out_dir results/GranLoRA/anchor_prompt
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from corl.open_r1 import gran_reward as R

# --------------------------------------------------------------------------- #
# candidate anchor prompts  (name -> (system_prompt, user_prompt))
# Vary persona / task-framing / anti-hallucination independently so we can see
# which component actually helps a small (1B) model.
# --------------------------------------------------------------------------- #
EXPERT = "You are an expert in medical image analysis."
FRAMING = (" Describe the imaging modality, the anatomy shown, and all clinically "
           "salient findings.")
ANTIHALL = (" Describe only what is visibly present in the image; do not speculate "
            "or state diagnoses that are not directly supported by what you see.")

CANDIDATES = {
    # current baseline: no system prompt, plain instruction
    "neutral":            ("", "Describe this medical image."),
    # persona only
    "persona":            (EXPERT, "Describe this medical image."),
    # persona + explicit task framing
    "persona_framing":    (EXPERT + FRAMING, "Describe this medical image."),
    # persona + anti-hallucination constraint
    "persona_antihall":   (EXPERT + ANTIHALL, "Describe this medical image."),
    # full: persona + framing + anti-hallucination
    "full":               (EXPERT + FRAMING + ANTIHALL, "Describe this medical image."),
    # framing folded into the USER turn instead of a system prompt
    "user_framing":       ("", "You are an expert radiologist. Describe the modality, "
                               "anatomy, and all visible findings in this medical image, "
                               "describing only what is visibly present."),
}

MOD_SYNONYMS = {
    "Computed Tomography": ["ct", "computed tomography", "cat scan"],
    "Magnetic Resonance Imaging": ["mri", "magnetic resonance"],
    "Microscopy Images": ["microscop", "histolog", "patholog", "h&e", "stain"],
    "Ultrasound": ["ultrasound", "sonograph", "doppler", "echocardiograph"],
    "Endoscopy": ["endoscop", "colonoscop", "laparoscop", "gastroscop"],
    "Fundus Photography": ["fundus", "retina", "retinal", "optic disc"],
}

FAITH_PROMPT = (
    "You are an expert in medical image analysis. Below is a caption someone wrote for "
    "the medical image shown.\n\nCAPTION: \"{cap}\"\n\n"
    "Assess the caption against the image:\n"
    "1. faithfulness (1-5): how accurately it describes what is actually visible "
    "(5 = fully accurate; 1 = mostly wrong or hallucinated).\n"
    "2. hallucinations: the number of specific claims in the caption that are NOT "
    "supported by the image.\n\n"
    "Respond with a SINGLE LINE of JSON, nothing else, like:\n"
    '{{"faithfulness": 4, "hallucinations": 1}}'
)


def _fix(s):
    return s.replace("Ġ", " ").replace("Ċ", "\n").strip()


def sample_rows(data_json, data_dir, per_modality):
    rows = json.load(open(data_json))
    by_mod, out = {}, []
    for r in rows:
        m = r.get("modality", "?")
        img_rel = r["image"][0] if isinstance(r["image"], (list, tuple)) else r["image"]
        p = os.path.join(data_dir, img_rel)
        if by_mod.get(m, 0) < per_modality and os.path.exists(p):
            out.append({"id": r["id"], "path": p, "modality": m})
            by_mod[m] = by_mod.get(m, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# phase 1: generate base captions under each candidate prompt
# --------------------------------------------------------------------------- #
def generate(args, sample):
    from transformers import AutoModelForCausalLM
    from janus.models import VLChatProcessor
    from corl.open_r1.janus_tokenizer_fix import load_fast_tokenizer

    processor = VLChatProcessor.from_pretrained(args.model)
    processor.tokenizer = load_fast_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    eos = processor.tokenizer.eos_token_id
    imgs = [Image.open(s["path"]).convert("RGB") for s in sample]

    @torch.inference_mode()
    def caption_batch(system_prompt, user_prompt, imgs_chunk):
        processor.system_prompt = system_prompt
        convs = [[{"role": "<|User|>", "content": f"<image_placeholder>\n{user_prompt}"},
                  {"role": "<|Assistant|>", "content": ""}] for _ in imgs_chunk]
        prep = processor(conversations=convs, images=[[i] for i in imgs_chunk],
                         force_batchify=True).to("cuda")
        emb = model.prepare_inputs_embeds(
            input_ids=prep.input_ids, pixel_values=prep.pixel_values,
            images_seq_mask=prep.images_seq_mask, images_emb_mask=prep.images_emb_mask)
        out = model.language_model.generate(
            inputs_embeds=emb, attention_mask=prep.attention_mask,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=eos, bos_token_id=processor.tokenizer.bos_token_id, eos_token_id=eos)
        caps = [_fix(d) for d in processor.tokenizer.batch_decode(out, skip_special_tokens=True)]
        # count generated tokens directly (trim at first eos), the same unit RL uses
        lens = []
        for row in out:
            ids = row.tolist()
            lens.append(ids.index(eos) if eos in ids else len(ids))
        return caps, lens

    def caption_all(system_prompt, user_prompt):
        # mini-batch so large samples don't OOM (all imgs in one generate() blows up).
        caps, lens = [], []
        for i in range(0, len(imgs), args.gen_batch_size):
            c, l = caption_batch(system_prompt, user_prompt, imgs[i:i + args.gen_batch_size])
            caps += c; lens += l
        return caps, lens

    result = {}
    for name, (sysp, userp) in CANDIDATES.items():
        caps, lens = caption_all(sysp, userp)
        result[name] = [{"id": sample[i]["id"], "modality": sample[i]["modality"],
                         "path": sample[i]["path"], "caption": caps[i], "len_tok": lens[i]}
                        for i in range(len(sample))]
        print(f"[gen] {name:16s} mean_len={sum(lens)/len(lens):6.1f} tok  "
              f"ex: {caps[0][:90]}", flush=True)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# phase 2: Qwen2.5-VL faithfulness judge + cheap stats -> ranked table
# --------------------------------------------------------------------------- #
def judge(args, gen):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    jp = AutoProcessor.from_pretrained(args.judge_model, trust_remote_code=True)
    jm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.judge_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    @torch.inference_mode()
    def judge_one(path, cap):
        img = Image.open(path).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": FAITH_PROMPT.format(cap=cap)}]}]
        text = jp.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = jp(text=[text], images=[img], return_tensors="pt", padding=True).to("cuda")
        out = jm.generate(**inp, max_new_tokens=48, do_sample=False)
        dec = jp.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        try:
            s, e = dec.find("{"), dec.rfind("}")
            p = json.loads(dec[s:e + 1])
            return float(p.get("faithfulness", 0)), float(p.get("hallucinations", 0))
        except Exception:
            return None, None

    summary = []
    for name, items in gen.items():
        faiths, halls, lens, reps, modhit = [], [], [], [], []
        for it in items:
            f, h = judge_one(it["path"], it["caption"])
            if f is not None:
                faiths.append(f); halls.append(h)
            lens.append(it["len_tok"])
            reps.append(R.rep_frac(it["caption"]))
            syn = MOD_SYNONYMS.get(it["modality"], [])
            modhit.append(1.0 if any(s in it["caption"].lower() for s in syn) else 0.0)
        n = max(len(faiths), 1)
        summary.append({
            "name": name, "n_judged": len(faiths),
            "faithfulness": sum(faiths) / n,
            "hallucinations": sum(halls) / n,
            "modality_mention_acc": sum(modhit) / len(modhit),
            "mean_len_tok": sum(lens) / len(lens),
            "rep_frac": sum(reps) / len(reps),
        })
        json.dump(items, open(os.path.join(args.out_dir, f"caps_{name}.json"), "w"),
                  indent=2, ensure_ascii=False)

    # rank: faithfulness first, then fewer hallucinations
    summary.sort(key=lambda s: (-s["faithfulness"], s["hallucinations"]))
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("\n=== anchor-prompt ranking (best first) ===")
    print(f"{'prompt':18s} {'faith':>6s} {'halluc':>7s} {'mod_acc':>8s} "
          f"{'len':>6s} {'rep':>6s}")
    for s in summary:
        print(f"{s['name']:18s} {s['faithfulness']:>6.2f} {s['hallucinations']:>7.2f} "
              f"{s['modality_mention_acc']:>8.2f} {s['mean_len_tok']:>6.0f} {s['rep_frac']:>6.3f}")
    print(f"\n[note] verbose target ~{R.L_VERBOSE_DEFAULT:.0f} tok; a base length near it "
          f"means alpha=0 already sits at the ceiling.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data_json", default="corl/eval/test_split_levels.json")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--per_modality", type=int, default=8)
    ap.add_argument("--gen_batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out_dir", default="results/GranLoRA/anchor_prompt")
    ap.add_argument("--gen_only", action="store_true")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sample = sample_rows(args.data_json, args.data_dir, args.per_modality)
    print(f"[data] {len(sample)} images across "
          f"{len(set(s['modality'] for s in sample))} modalities", flush=True)

    gen_path = os.path.join(args.out_dir, "generations.json")
    gen = generate(args, sample)
    json.dump(gen, open(gen_path, "w"), indent=2, ensure_ascii=False)
    print(f"[gen] wrote {gen_path}", flush=True)
    if args.gen_only:
        return
    judge(args, gen)


if __name__ == "__main__":
    main()
