"""
Load and validate Janus checkpoint directories as runnable models.

Example:
    python evaluate_checkpoints.py \
      --checkpoints \
      /projects/u6gd/umar/codes/ULM-R1/JanusPro-1B-CoRL-noMM/CycleOnly-G4-bs16-genHead-genAligner/checkpoint-200 \
      /projects/u6gd/umar/codes/ULM-R1/JanusPro-1B-CoRL-Uniified/RFT22k-CycleMatchAccFormat-UniReward-G4-beta004-bs16/checkpoint-800
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

import pyarrow.parquet as pq
import os
from PIL import Image, ImageDraw, ImageFont

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np


DEFAULT_CHECKPOINTS = [
    "/projects/u6gd/umar/codes/ULM-R1/JanusPro-1B-CoRL-noMM/"
    "CycleOnly-G4-bs16-genHead-genAligner/checkpoint-600",
    # "/projects/u6gd/umar/codes/ULM-R1/JanusPro-1B-CoRL-Uniified/"
    # "RFT22k-CycleMatchAccFormat-UniReward-G4-beta004-bs16/checkpoint-800",
]

PARQUET_PATH = "/projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama.parquet"
IMAGE_ROOT = "/projects/u6gd/datasets/PubMedVision/images/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one or more Janus checkpoints as Hugging Face models."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint directory paths to load.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to place the model on.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype used for model weights.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sampled rows to evaluate from the parquet table.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/projects/u6gd/umar/codes/ULM-R1/checkpoint_evals",
        help="Root output directory. A subfolder is created per checkpoint.",
    )
    return parser.parse_args()


@torch.inference_mode()
def generate_image(
    mmgpt,
    vl_chat_processor,
    prompt: str,
    parallel_size: int = 1,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    device = next(mmgpt.parameters()).device

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros(
        (parallel_size * 2, len(input_ids)), dtype=torch.int, device=device
    )
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int, device=device
    )

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec  # (parallel_size, H, W, 3)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    return device_arg


def resolve_dtype(dtype_arg: str, device: str) -> Optional[torch.dtype]:
    if dtype_arg == "auto":
        if device == "cuda":
            return torch.bfloat16
        return None

    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    return None


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_t2i_prompt(vl_chat_processor, user_text: str) -> str:
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


def wrap_text_for_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int):
    words = text.replace("\n", " \n ").split(" ")
    lines = []
    current = ""

    for word in words:
        if word == "\n":
            lines.append(current.rstrip())
            current = ""
            continue

        proposal = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), proposal, font=font)
        proposal_width = bbox[2] - bbox[0]

        if proposal_width <= max_width or not current:
            current = proposal
        else:
            lines.append(current.rstrip())
            current = word

    if current:
        lines.append(current.rstrip())
    return lines


def save_comparison_image(
    out_path: Path,
    orig_img: Image.Image,
    regen_img: Image.Image,
    caption: str,
):
    pad = 20
    gap = 20

    target_h = max(orig_img.height, regen_img.height)
    orig = resize_to_height(orig_img, target_h)
    regen = resize_to_height(regen_img, target_h)

    font = ImageFont.load_default()

    canvas_w = orig.width + regen.width + pad * 2 + gap
    temp_canvas = Image.new("RGB", (canvas_w, 10), color=(255, 255, 255))
    draw_temp = ImageDraw.Draw(temp_canvas)

    title = "Original | Regenerated"
    title_bbox = draw_temp.textbbox((0, 0), title, font=font)
    title_h = (title_bbox[3] - title_bbox[1]) + 10

    caption_prefix = "Caption: "
    caption_text = caption_prefix + caption
    wrapped = wrap_text_for_width(draw_temp, caption_text, font, canvas_w - 2 * pad)

    line_h = (draw_temp.textbbox((0, 0), "Ag", font=font)[3] - draw_temp.textbbox((0, 0), "Ag", font=font)[1]) + 6
    caption_h = max(line_h, line_h * len(wrapped))

    canvas_h = pad + title_h + target_h + gap + caption_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, pad), title, fill=(0, 0, 0), font=font)

    y_img = pad + title_h
    canvas.paste(orig, (pad, y_img))
    canvas.paste(regen, (pad + orig.width + gap, y_img))

    y_caption = y_img + target_h + gap
    for i, line in enumerate(wrapped):
        draw.text((pad, y_caption + i * line_h), line, fill=(0, 0, 0), font=font)

    canvas.save(out_path)


def load_checkpoint(ckpt_path: str, device: str, torch_dtype: Optional[torch.dtype]) -> None:
    path = Path(ckpt_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    print(f"\n[Checkpoint] {ckpt_path}")
    print("- Loading VLChatProcessor...")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(ckpt_path)

    print("- Loading AutoModelForCausalLM...")
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    vl_gpt = vl_gpt.to(device).eval()

    n_params = count_params(vl_gpt)
    model_dtype = next(vl_gpt.parameters()).dtype
    model_device = next(vl_gpt.parameters()).device

    print("- Loaded successfully")
    print(f"- Device: {model_device}")
    print(f"- Dtype:  {model_dtype}")
    print(f"- Params: {n_params:,}")
    print(f"- Tokenizer vocab size: {len(vl_chat_processor.tokenizer)}")

    return vl_chat_processor, vl_gpt


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)

    print(f"Using device: {device}")
    print(f"Using dtype:  {torch_dtype}")

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
        vl_gpt.gen_vision_model = vl_gpt.gen_vision_model.to(torch.float32)

        ckpt_out_dir = Path(args.out_dir) / checkpoint_to_dirname(ckpt)
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to: {ckpt_out_dir}")

        for idx, s in enumerate(samples):
            orig_img = Image.open(s["image_path"]).convert("RGB")
            caption = s["caption"]
            t2i_prompt = build_t2i_prompt(vl_chat_processor, caption)
            gen_imgs = generate_image(
                vl_gpt, vl_chat_processor, t2i_prompt,
                parallel_size=1,
                temperature=1,
                cfg_weight=5,
            )
            gen_img_pil = Image.fromarray(gen_imgs[0])

            image_stem = Path(s["image_name"]).stem
            save_path = ckpt_out_dir / f"{idx:05d}_{image_stem}.png"
            save_comparison_image(
                out_path=save_path,
                orig_img=orig_img,
                regen_img=gen_img_pil,
                caption=caption,
            )

            print(f"[{idx + 1}/{len(samples)}] saved {save_path}")

            



if __name__ == "__main__":
    main()
