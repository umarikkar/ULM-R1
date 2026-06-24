"""Baseline demo: caption a PubMedVision image with Janus, then T2I from that caption.

Loads vanilla deepseek-ai/Janus-Pro-1B (no fine-tuning), runs i2t on one image,
then runs t2i on the produced caption, and saves a side-by-side PNG.

Usage:
    python corl/scripts/demo_janus_baseline.py \
        --image_path /vol/.../PubMedVision/images/pmc_1_0.jpg \
        --out_dir results/janus_baseline
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def fix_janus_text(text: str) -> str:
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

torch.backends.cudnn.enabled = False

MODEL_ID = "deepseek-ai/Janus-Pro-1B"
N_IMAGE_TOKENS = 576
IMG_SIZE = 384
PATCH_SIZE = 16


@torch.inference_mode()
def caption_image(model, processor, image: Image.Image, device, max_new_tokens=256):
    instruction = (
        "Describe this image in one sentence"
    )
    conv = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{instruction}"},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepare_inputs = processor(
        conversations=[conv], images=[[image]], force_batchify=True,
    ).to(device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fix_janus_text(text)


@torch.inference_mode()
def generate_image(model, processor, caption: str, device,
                   cfg_scale=5.0, temperature=1.0, parallel_size=1):
    conv = [
        {"role": "<|User|>", "content": caption},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prompt = (
        processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=processor.sft_format, system_prompt="",
        )
        + processor.image_start_tag
    )
    enc = processor.tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    pad_id = processor.pad_id
    bos_id = processor.tokenizer.bos_token_id
    L = input_ids.shape[1]

    # Build cond/uncond CFG batch (interleaved).
    tokens = torch.zeros((parallel_size * 2, L), dtype=torch.long, device=device)
    attn = torch.zeros((parallel_size * 2, L), dtype=attention_mask.dtype, device=device)
    for i in range(parallel_size * 2):
        tokens[i] = input_ids[0]
        attn[i] = attention_mask[0]
        if i % 2 != 0:
            bos_positions = (tokens[i] == bos_id).nonzero(as_tuple=True)[0]
            if len(bos_positions) > 0:
                bos_pos = bos_positions[0].item()
                tokens[i, bos_pos + 1:-1] = pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, N_IMAGE_TOKENS), dtype=torch.long, device=device)
    outputs = None
    for i in range(N_IMAGE_TOKENS):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = model.gen_head(hidden)
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
        probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        generated_tokens[:, i] = next_token

        both = torch.stack([next_token, next_token], dim=1).view(-1)
        inputs_embeds = model.prepare_gen_img_embeds(both).unsqueeze(1)
        attn = torch.cat(
            [attn, torch.ones(attn.shape[0], 1, dtype=attn.dtype, device=device)], dim=1,
        )

    grid = IMG_SIZE // PATCH_SIZE
    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, grid, grid],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(dec[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/janus_baseline")
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo] loading {MODEL_ID} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    processor = VLChatProcessor.from_pretrained(MODEL_ID)
    processor.system_prompt = ""

    image = Image.open(args.image_path).convert("RGB")
    print(f"[demo] input image: {args.image_path} ({image.size})")

    caption = caption_image(model, processor, image, device)
    print(f"[demo] caption:\n  {caption}")

    gen_img = generate_image(
        model, processor, caption, device,
        cfg_scale=args.cfg_scale, temperature=args.temperature,
    )

    # Side-by-side: original | generated. Resize original to gen's height.
    h = gen_img.height
    orig_resized = image.resize(
        (int(image.width * h / image.height), h), Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (orig_resized.width + gen_img.width + 10, h), (255, 255, 255))
    canvas.paste(orig_resized, (0, 0))
    canvas.paste(gen_img, (orig_resized.width + 10, 0))
    out_path = out_dir / (Path(args.image_path).stem + "_baseline.png")
    canvas.save(out_path)
    cap_path = out_dir / (Path(args.image_path).stem + "_caption.txt")
    cap_path.write_text(caption + "\n")
    print(f"[demo] saved {out_path}")
    print(f"[demo] saved {cap_path}")


if __name__ == "__main__":
    main()