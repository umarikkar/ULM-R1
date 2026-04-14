"""
Sanity-check Janus-Pro-1B's out-of-the-box T2I behavior on medical prompts.

Usage (inside the corl env):
    python test_janus_medical.py
    python test_janus_medical.py --model-path deepseek-ai/Janus-Pro-1B \
                                 --out-dir janus_medical_samples \
                                 --parallel-size 4 --cfg-weight 5

The script runs a fixed set of medical prompts at three levels of specificity:
  1. bare modality          ("A chest X-ray.")
  2. mid-level              ("A contrast-enhanced axial CT of the chest.")
  3. detailed / findings    ("A T2-weighted sagittal MRI of the lumbar spine
                              showing disc herniation.")
and saves a 2x2 (or NxN) grid per prompt plus a combined contact sheet.

Model + generate() API are the standard DeepSeek Janus flow:
    VLChatProcessor + MultiModalityCausalLM, CFG-guided autoregressive
    image-token decoding, then gen_vision_model.decode_code.
"""

import argparse
import math
import os
import time

import numpy as np
import torch

# cuDNN in this conda env fails to initialize on the VQ decoder's conv2d
# (CUDNN_STATUS_NOT_INITIALIZED). Disable cuDNN so the convs fall back to
# native CUDA kernels. Must be set before any conv runs.
torch.backends.cudnn.enabled = False
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM

from janus.models import VLChatProcessor


# ---------------------------------------------------------------------------
# Prompts — grouped by specificity. Tweak freely.
# ---------------------------------------------------------------------------
PROMPTS = [
    # --- Chest X-ray --------------------------------------------------------
    ("xray_L1", "A chest X-ray."),
    ("xray_L2", "A frontal chest X-ray of an adult patient."),
    ("xray_L3", "A frontal chest X-ray of a patient with right lower lobe "
                "pneumonia and a small pleural effusion."),

    # --- CT -----------------------------------------------------------------
    ("ct_L1", "A CT scan of the abdomen."),
    ("ct_L2", "A contrast-enhanced axial CT of the chest."),
    ("ct_L3", "A contrast-enhanced axial CT of the chest showing "
              "a pulmonary nodule in the right upper lobe."),

    # --- MRI ----------------------------------------------------------------
    ("mri_L1", "An MRI of the brain."),
    ("mri_L2", "An axial FLAIR MRI of the brain."),
    ("mri_L3", "An axial diffusion-weighted MRI of the brain showing "
               "an acute middle cerebral artery infarct."),

    # --- Ultrasound ---------------------------------------------------------
    ("us_L1", "An ultrasound image."),
    ("us_L2", "An abdominal ultrasound of the liver."),
    ("us_L3", "An abdominal ultrasound of the liver showing a hyperechoic "
              "hepatic lesion consistent with a hemangioma."),

    # --- Histology ----------------------------------------------------------
    ("histo_L1", "A histology slide."),
    ("histo_L2", "An H&E stained histology slide of liver tissue."),
    ("histo_L3", "An H&E stained histology slide of liver tissue showing "
                 "cirrhosis with bridging fibrosis."),

    # --- Fundus photography -------------------------------------------------
    ("fundus_L1", "A fundus photograph."),
    ("fundus_L2", "A colour fundus photograph of the retina."),
    ("fundus_L3", "A colour fundus photograph of the retina showing "
                  "diabetic retinopathy with microaneurysms and hard exudates."),
]


# ---------------------------------------------------------------------------
# DeepSeek's reference T2I generate() function, unchanged except for dtype
# and the hard-coded .cuda() being replaced with model.device.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate(
    mmgpt,
    vl_chat_processor,
    prompt: str,
    parallel_size: int = 4,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,  # 24x24 tokens
    img_size: int = 384,
    patch_size: int = 16,
):
    device = next(mmgpt.parameters()).device

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int, device=device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:  # unconditional branch for CFG
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

        # classifier-free guidance
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
    return dec  # (parallel_size, H, W, 3) uint8


def build_conversation_prompt(vl_chat_processor, user_text: str) -> str:
    conversation = [
        {"role": "<|User|>",      "content": user_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


def make_grid(images: np.ndarray) -> Image.Image:
    """images: (N, H, W, 3) uint8 -> PIL grid."""
    n, h, w, _ = images.shape
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    grid = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = images[i]
    return Image.fromarray(grid)


def caption_image(img: Image.Image, text: str, bar_h: int = 36) -> Image.Image:
    """Add a white bar with the prompt text above the image (for contact sheet)."""
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), "white")
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()
    # crude truncation so it fits
    max_chars = max(1, w // 8)
    label = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    draw.text((6, 8), label, fill="black", font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--out-dir",    type=str,
                        default="/work/um00109/MLLM/datasets/VL-Health/janus_medical_samples")
    parser.add_argument("--parallel-size", type=int, default=4,
                        help="samples per prompt")
    parser.add_argument("--cfg-weight",    type=float, default=5.0)
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--seed",          type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    grids_dir = os.path.join(args.out_dir, "grids")
    singles_dir = os.path.join(args.out_dir, "singles")
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(singles_dir, exist_ok=True)

    print(f"Loading VLChatProcessor from {args.model_path}")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)

    print(f"Loading MultiModalityCausalLM from {args.model_path}")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    # Keep the VQ decoder in fp32 — with cuDNN disabled the native fp32 conv
    # path is reliable and the cost is negligible for 24x24 latents.
    vl_gpt.gen_vision_model = vl_gpt.gen_vision_model.to(torch.float32)
    print(f"Model ready on {next(vl_gpt.parameters()).device}")

    contact_tiles = []
    total_t0 = time.time()

    for name, user_text in PROMPTS:
        print(f"\n[{name}] {user_text}")
        prompt = build_conversation_prompt(vl_chat_processor, user_text)

        t0 = time.time()
        imgs = generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            parallel_size=args.parallel_size,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
        )
        dt = time.time() - t0
        print(f"  generated {args.parallel_size} samples in {dt:.1f}s")

        # save individual samples
        for i in range(args.parallel_size):
            Image.fromarray(imgs[i]).save(
                os.path.join(singles_dir, f"{name}_{i}.png")
            )

        # per-prompt grid
        grid = make_grid(imgs)
        grid.save(os.path.join(grids_dir, f"{name}.png"))

        # shrink first sample for contact sheet
        tile = Image.fromarray(imgs[0]).resize((256, 256), Image.LANCZOS)
        contact_tiles.append(caption_image(tile, user_text))

    # Build contact sheet: N rows x 1 col of captioned tiles
    if contact_tiles:
        tile_w, tile_h = contact_tiles[0].size
        cols = 3
        rows = int(math.ceil(len(contact_tiles) / cols))
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        for idx, t in enumerate(contact_tiles):
            r, c = divmod(idx, cols)
            sheet.paste(t, (c * tile_w, r * tile_h))
        sheet_path = os.path.join(args.out_dir, "contact_sheet.png")
        sheet.save(sheet_path)
        print(f"\nContact sheet: {sheet_path}")

    print(f"\nAll done in {time.time() - total_t0:.0f}s")
    print(f"Grids:   {grids_dir}")
    print(f"Singles: {singles_dir}")


if __name__ == "__main__":
    main()
