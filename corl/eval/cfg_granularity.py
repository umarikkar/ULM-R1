"""V1 CFG granularity control: guide with the gran-LoRA delta at inference.

At each decode step we run TWO forwards of the same model:
  - LoRA at scale alpha  -> logits_cond   (the granularity direction)
  - LoRA off (scale 0)   -> logits_base   (the base LLM = content anchor)
and combine, classifier-free-guidance style, per row:
  logits = logits_base + w * (logits_cond - logits_base)
w=0 -> base, w=1 -> plain LoRA@alpha, w>1 -> granularity extrapolated beyond the
LoRA, 0<w<1 -> attenuated. Both branches keep their own KV cache; the SAME
(combined) token is fed to both each step. We batch the w-sweep together (one row
per w) for a fixed image, so a single image = one batched decode.

Idea being tested: is w a clean granularity knob whose CONTENT stays image-specific
(anchored by the base branch), vs the plain LoRA which slightly over-templates?

    python corl/eval/cfg_granularity.py --adapter_dir results/GranLoRA/gran_lora_v2 \
        --data_json corl/eval/test_split_levels.json --data_dir .../PubMedVision \
        --alpha 0.9 --ws 0,0.5,1,1.5,2,3 --per_modality 6 --out results/GranLoRA/cfg_v2.json
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
from corl.eval.diagnose_content_leak import diversity

NEUTRAL_PROMPT = "Describe this medical image."


def set_scale(model, base, alpha):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                m.scaling[adp] = base[(name, adp)] * alpha


@torch.inference_mode()
def cfg_decode(model, base_scaling, embeds0, attn0, ws, alpha, max_new, eos):
    """Batched CFG greedy decode; row b uses guidance weight ws[b]. Returns token lists."""
    lm = model.language_model
    B = embeds0.shape[0]
    device = embeds0.device
    wt = torch.tensor(ws, device=device, dtype=embeds0.dtype).view(B, 1)

    def fwd(alpha_, **kw):
        set_scale(model, base_scaling, alpha_)
        return lm(use_cache=True, **kw)

    oc = fwd(alpha, inputs_embeds=embeds0, attention_mask=attn0)
    ob = fwd(0.0, inputs_embeds=embeds0, attention_mask=attn0)
    cache_c, cache_b = oc.past_key_values, ob.past_key_values
    lc, lb = oc.logits[:, -1, :], ob.logits[:, -1, :]

    seqs = [[] for _ in range(B)]
    done = [False] * B
    attn = attn0
    for _ in range(max_new):
        comb = lb + wt * (lc - lb)
        nxt = comb.argmax(-1)                       # [B]
        for b in range(B):
            if not done[b]:
                if nxt[b].item() == eos:
                    done[b] = True
                else:
                    seqs[b].append(nxt[b].item())
        if all(done):
            break
        attn = torch.cat([attn, torch.ones(B, 1, device=device, dtype=attn.dtype)], 1)
        inp = nxt.view(B, 1)
        oc = fwd(alpha, input_ids=inp, attention_mask=attn, past_key_values=cache_c)
        ob = fwd(0.0, input_ids=inp, attention_mask=attn, past_key_values=cache_b)
        cache_c, cache_b = oc.past_key_values, ob.past_key_values
        lc, lb = oc.logits[:, -1, :], ob.logits[:, -1, :]
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.9, help="LoRA scale = granularity direction")
    ap.add_argument("--ws", default="0,0.5,1,1.5,2,3", help="CFG guidance weights to sweep")
    ap.add_argument("--per_modality", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=140)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ws = [float(x) for x in args.ws.split(",")]
    device = "cuda"

    rows = json.load(open(args.data_json))
    seen, sample = {}, []
    for r in rows:
        m = r.get("modality", "?")
        if seen.get(m, 0) < args.per_modality:
            p = os.path.join(args.data_dir, r["image"][0] if isinstance(r["image"], list) else r["image"])
            if os.path.exists(p):
                sample.append({"modality": m, "path": p})
                seen[m] = seen.get(m, 0) + 1
    print(f"[data] {len(sample)} images | alpha={args.alpha} | ws={ws}", flush=True)

    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    processor.tokenizer = load_fast_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter_dir)
    model = model.to(device).eval()
    base_scaling = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                base_scaling[(name, adp)] = float(m.scaling[adp])
    tok = processor.tokenizer
    eos = tok.eos_token_id

    caps_by_w = {w: [] for w in ws}
    for s in sample:
        img = Image.open(s["path"]).convert("RGB")
        conv = [[{"role": "<|User|>", "content": f"<image_placeholder>\n{NEUTRAL_PROMPT}"},
                 {"role": "<|Assistant|>", "content": ""}]]
        prep = processor(conversations=conv[0], images=[img], force_batchify=True).to(device)
        emb = model.prepare_inputs_embeds(
            input_ids=prep.input_ids, pixel_values=prep.pixel_values,
            images_seq_mask=prep.images_seq_mask, images_emb_mask=prep.images_emb_mask)
        B = len(ws)
        emb = emb.repeat(B, 1, 1)
        attn = prep.attention_mask.repeat(B, 1)
        seqs = cfg_decode(model, base_scaling, emb, attn, ws, args.alpha, args.max_new_tokens, eos)
        for w, sq in zip(ws, seqs):
            caps_by_w[w].append(tok.decode(sq, skip_special_tokens=True).strip())

    report = {"alpha": args.alpha, "ws": ws, "per_w": {}}
    print(f"\n{'w':>5} {'words':>6} {'vocab':>6} {'dist2':>6} {'dist3':>6} {'pair_jac':>8} {'pref5':>6}")
    for w in ws:
        d = diversity(caps_by_w[w])
        report["per_w"][w] = {"metrics": d,
                              "examples": [{"modality": sample[i]["modality"], "caption": caps_by_w[w][i]}
                                           for i in range(min(len(sample), 12))]}
        print(f"{w:>5.1f} {d['mean_words']:>6.1f} {d['vocab']:>6} {d['distinct2']:>6.3f} "
              f"{d['distinct3']:>6.3f} {d['mean_pairwise_jaccard']:>8.3f} {d['shared_prefix5_rate']:>6.3f}")

    print("\n--- one image across w (granularity should scale with w) ---")
    for w in ws:
        print(f"  w={w:>4}: {caps_by_w[w][0][:150]}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
