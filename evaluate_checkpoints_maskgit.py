"""
MaskGIT-style inference for Janus checkpoints trained with sft_trainer_alignment.py.

Matches training: prompt ending in image_start_tag, N=576 image slots filled
with the learned `mask_token_embed`, single parallel forward with
task="generation", iteratively commit highest-confidence positions over K steps.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor


DEFAULT_CHECKPOINTS = [
    "./results/results/JanusPro-1B-CoRL-AlignmentSFT_v2/AlignmentSFT/checkpoint-8600",
    "./results/results/JanusPro-1B-CoRL-AlignmentSFT_v3/AlignmentSFT/checkpoint-9000",
]

PARQUET_PATH = "/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/data/t2i_midlevel_llama.parquet"
IMAGE_ROOT = "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision/images/"

N_IMAGE_TOKENS = 576           # 24x24 grid for 384px @ patch 16
GRID_H = GRID_W = 24
IMG_SIZE = 384
PATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="./checkpoint_evals_maskgit")
    parser.add_argument("--steps", type=int, default=12, help="Number of MaskGIT decoding steps.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--gumbel-temp", type=float, default=4.5,
        help="Confidence-noise temperature (annealed linearly to 0 over steps).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    return device_arg


def resolve_dtype(dtype_arg: str, device: str) -> Optional[torch.dtype]:
    if dtype_arg == "auto":
        return torch.bfloat16 if device == "cuda" else None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_arg]


def build_t2i_prompt(vl_chat_processor: VLChatProcessor, user_text: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": user_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def maskgit_generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    steps: int = 12,
    temperature: float = 1.0,
    gumbel_temp: float = 4.5,
) -> np.ndarray:
    """Iterative parallel decode matching the masked-alignment SFT objective."""
    device = next(mmgpt.parameters()).device
    param_dtype = next(mmgpt.parameters()).dtype

    # Tokenize prompt (ends with image_start_tag, same as training wrap_t2i_prompt).
    input_ids = torch.LongTensor(
        vl_chat_processor.tokenizer.encode(prompt)
    ).unsqueeze(0).to(device)                                  # [1, L]
    L = input_ids.shape[1]

    text_embeds = mmgpt.language_model.get_input_embeddings()(input_ids)   # [1, L, D]

    B = 1
    N = N_IMAGE_TOKENS

    # Initial state: every image slot is the learned mask token, nothing committed.
    committed_ids = torch.full((B, N), -1, dtype=torch.long, device=device)
    mask_token = mmgpt.mask_token_embed.to(param_dtype).expand(B, N, -1)   # [B, N, D]

    text_attn = torch.ones(B, L, device=device, dtype=torch.long)
    img_attn = torch.ones(B, N, device=device, dtype=torch.long)
    full_attn_mask = torch.cat([text_attn, img_attn], dim=1)

    for step in range(steps):
        # Build current image embeds: committed slots use codebook embeds, others use mask.
        is_committed = (committed_ids >= 0)                                # [B, N]
        safe_ids = committed_ids.clamp(min=0)
        committed_embeds = mmgpt.prepare_gen_img_embeds(safe_ids)          # [B, N, D]
        img_embeds = torch.where(is_committed.unsqueeze(-1), committed_embeds, mask_token)

        inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)        # [B, L+N, D]

        logits = mmgpt(
            t2i_inputs_embeds=inputs_embeds,
            t2i_attention_mask=full_attn_mask,
            t2i_logits_to_keep=N,
            task="generation",
        ).logits                                                            # [B, N, V_image]

        probs = F.softmax(logits.float() / max(temperature, 1e-6), dim=-1)  # [B, N, V]
        # Sample a candidate id per masked position.
        flat_probs = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat_probs, num_samples=1).reshape(B, N)  # [B, N]
        sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)    # [B, N]

        # Keep already-committed ids; replace mask slots with newly sampled.
        new_ids = torch.where(is_committed, committed_ids, sampled)

        # Cosine schedule: fraction of slots that should *remain masked* after this step.
        ratio = (step + 1) / steps
        mask_frac = math.cos(0.5 * math.pi * ratio)                         # 1 → 0
        n_mask_next = int(math.floor(mask_frac * N))
        n_mask_next = max(n_mask_next, 0)

        if n_mask_next == 0 or step == steps - 1:
            committed_ids = new_ids
            break

        # Confidence with annealed Gumbel noise; already-committed slots get +inf so we never remask them.
        temp_anneal = gumbel_temp * (1.0 - ratio)
        u = torch.rand_like(sampled_conf).clamp_(min=1e-6, max=1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        confidence = sampled_conf.log() + temp_anneal * gumbel
        confidence = torch.where(
            is_committed,
            torch.full_like(confidence, float("inf")),
            confidence,
        )

        # Mask the n_mask_next lowest-confidence slots; commit the rest.
        _, mask_idx = torch.topk(confidence, k=n_mask_next, dim=-1, largest=False)
        next_committed = torch.ones_like(new_ids, dtype=torch.bool)
        next_committed.scatter_(1, mask_idx, False)

        committed_ids = torch.where(next_committed, new_ids, torch.full_like(new_ids, -1))

    if (committed_ids < 0).any():
        # Final safety net — shouldn't trigger because the last step commits everything.
        committed_ids = torch.where(
            committed_ids < 0, torch.zeros_like(committed_ids), committed_ids
        )

    dec = mmgpt.gen_vision_model.decode_code(
        committed_ids.to(torch.int),
        shape=[B, 8, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec                                                              # [B, H, W, 3]


# ---- imaging helpers (mirrors evaluate_checkpoints.py) ----

def checkpoint_to_dirname(ckpt_path: str) -> str:
    path = Path(ckpt_path).resolve()
    if path.parent.name:
        return f"{path.parent.name}__{path.name}"
    return path.name


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    target_w = max(1, int(round(img.width * (target_h / img.height))))
    return img.resize((target_w, target_h), Image.Resampling.BICUBIC)


def wrap_text_for_width(draw, text, font, max_width):
    words = text.replace("\n", " \n ").split(" ")
    lines, current = [], ""
    for word in words:
        if word == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        proposal = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), proposal, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not current:
            current = proposal
        else:
            lines.append(current.rstrip())
            current = word
    if current:
        lines.append(current.rstrip())
    return lines


def save_comparison_image(out_path: Path, orig_img: Image.Image, regen_img: Image.Image, caption: str):
    pad, gap = 20, 20
    target_h = max(orig_img.height, regen_img.height)
    orig = resize_to_height(orig_img, target_h)
    regen = resize_to_height(regen_img, target_h)
    font = ImageFont.load_default()

    canvas_w = orig.width + regen.width + pad * 2 + gap
    temp = Image.new("RGB", (canvas_w, 10), color=(255, 255, 255))
    draw_temp = ImageDraw.Draw(temp)

    title = "Original | Regenerated (MaskGIT)"
    tb = draw_temp.textbbox((0, 0), title, font=font)
    title_h = (tb[3] - tb[1]) + 10

    wrapped = wrap_text_for_width(draw_temp, "Caption: " + caption, font, canvas_w - 2 * pad)
    lh = (draw_temp.textbbox((0, 0), "Ag", font=font)[3]
          - draw_temp.textbbox((0, 0), "Ag", font=font)[1]) + 6
    caption_h = max(lh, lh * len(wrapped))

    canvas_h = pad + title_h + target_h + gap + caption_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(0, 0, 0), font=font)
    y_img = pad + title_h
    canvas.paste(orig, (pad, y_img))
    canvas.paste(regen, (pad + orig.width + gap, y_img))
    y_cap = y_img + target_h + gap
    for i, line in enumerate(wrapped):
        draw.text((pad, y_cap + i * lh), line, fill=(0, 0, 0), font=font)
    canvas.save(out_path)


def load_checkpoint(ckpt_path: str, device: str, torch_dtype: Optional[torch.dtype]):
    path = Path(ckpt_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
    print(f"\n[Checkpoint] {ckpt_path}")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(ckpt_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch_dtype,
    )
    vl_gpt = vl_gpt.to(device).eval()
    print(f"- Device: {next(vl_gpt.parameters()).device}")
    print(f"- Dtype:  {next(vl_gpt.parameters()).dtype}")
    return vl_chat_processor, vl_gpt


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)

    print(f"Using device: {device}")
    print(f"Using dtype:  {torch_dtype}")
    print(f"MaskGIT steps: {args.steps}, temperature: {args.temperature}, gumbel_temp: {args.gumbel_temp}")

    table = pq.read_table(PARQUET_PATH)
    stride = max(1, table.num_rows // max(args.num_samples, 1))
    samples = []
    for i in range(args.num_samples):
        row = table.slice(i * stride, 1).to_pydict()
        img_path = os.path.join(IMAGE_ROOT, row["image_path"][0])
        if not os.path.exists(img_path):
            print(f"  [skip] {img_path} not found")
            continue
        samples.append({
            "image_path": img_path,
            "caption": row["caption"][0],
            "image_name": row["image_path"][0],
        })
    print(f"Loaded {len(samples)} samples (stride {stride})")

    for ckpt in args.checkpoints:
        vl_chat_processor, vl_gpt = load_checkpoint(ckpt, device=device, torch_dtype=torch_dtype)
        # Decode in fp32 for numerical safety (matches evaluate_checkpoints.py).
        vl_gpt.gen_vision_model = vl_gpt.gen_vision_model.to(torch.float32)

        ckpt_out_dir = Path(args.out_dir) / checkpoint_to_dirname(ckpt)
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to: {ckpt_out_dir}")

        for idx, s in enumerate(samples):
            orig_img = Image.open(s["image_path"]).convert("RGB")
            caption = s["caption"]
            t2i_prompt = build_t2i_prompt(vl_chat_processor, caption)
            gen_imgs = maskgit_generate(
                vl_gpt, vl_chat_processor, t2i_prompt,
                steps=args.steps,
                temperature=args.temperature,
                gumbel_temp=args.gumbel_temp,
            )
            gen_img_pil = Image.fromarray(gen_imgs[0])

            image_stem = Path(s["image_name"]).stem
            save_path = ckpt_out_dir / f"{idx:05d}_{image_stem}.png"
            save_comparison_image(save_path, orig_img, gen_img_pil, caption)
            print(f"[{idx + 1}/{len(samples)}] saved {save_path}")


if __name__ == "__main__":
    main()
