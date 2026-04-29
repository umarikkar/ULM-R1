"""
Run an image -> caption pipeline with Janus-Pro-1B and report VQ/caption outputs.

For each sampled example:
1) Generate caption from original image (I2T)
2) Save the VQ reconstruction and caption metrics
3) Save a side-by-side panel (original + VQ reconstruction + text/metrics)
"""

import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor


# cuDNN in this conda env fails to initialize on conv2d
# (CUDNN_STATUS_NOT_INITIALIZED). Disable cuDNN so convs fall back to native kernels.
torch.backends.cudnn.enabled = False

IMAGE_ROOT = "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision/images"
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "janus_medical_i2t_vq_caption_samples"
)
IMG_SIZE = 384
PATCH_SIZE = 16
QUESTION = (
    "<image_placeholder>\n"
    "Describe the main content of the image."
)

VQ_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def clean_generated_text(text: str) -> str:
    """Fix GPT-BPE artifacts and normalize whitespace."""
    cleaned = text.replace("\u0120", " ").replace("Ġ", " ")
    return " ".join(cleaned.split())


def sanitize_name(value) -> str:
    s = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int):
    lines = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def pil_to_unit_tensor(image: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
    resized = image.resize(target_size, Image.BICUBIC)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.inference_mode()
def generate_caption(
    vl_gpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    pil_image: Image.Image,
    max_new_tokens: int,
) -> str:
    conversation = [
        {
            "role": "<|User|>",
            "content": QUESTION,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return clean_generated_text(answer)


@torch.inference_mode()
def encode_decode_image(gen_vision_model, pil_image: Image.Image):
    """Encode a real image into discrete latent tokens, then decode it back."""
    x = VQ_TRANSFORM(pil_image).unsqueeze(0)
    x = x.to(next(gen_vision_model.parameters()).device)

    quant, _, info = gen_vision_model.encode(x)

    codes = info[2].reshape(1, -1)

    dec = gen_vision_model.decode_code(
        codes.to(dtype=torch.int),
        shape=[1, 8, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE],
    )
    dec_np = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec_np = np.clip((dec_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(dec_np[0]), quant


def compute_pixel_metrics(orig_pil: Image.Image, recon_pil: Image.Image):
    orig = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)).astype(np.float64)
    recon = np.array(recon_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)).astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    l1 = np.mean(np.abs(orig - recon))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return {"mse": mse, "l1": l1, "psnr": psnr}


def save_experiment_panel(
    out_path: str,
    image_id,
    question_text: str,
    generated_caption: str,
    gt_caption: str,
    orig_image: Image.Image,
    vq_recon_image: Image.Image,
    vq_metrics: dict,
):
    margin = 20
    spacing = 14
    line_spacing = 6
    image_gap = 20

    panel_width = max(orig_image.width * 2 + image_gap + (margin * 2), 1100)
    header_h = max(orig_image.height, vq_recon_image.height)

    font = ImageFont.load_default()
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (panel_width, 8), "white"))
    text_width = panel_width - (margin * 2)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]

    metric_line = f"VQ recon: PSNR={vq_metrics['psnr']:.2f} dB, MSE={vq_metrics['mse']:.2f}, L1={vq_metrics['l1']:.2f}"
    q_lines = wrap_text(tmp_draw, f"Question: {question_text}", font, text_width)
    c_lines = wrap_text(tmp_draw, f"Generated caption: {generated_caption}", font, text_width)
    gt_lines = wrap_text(tmp_draw, f"Dataset caption: {gt_caption}", font, text_width)
    m_lines = wrap_text(tmp_draw, metric_line, font, text_width)

    total_text_lines = 1 + len(q_lines) + 1 + len(c_lines) + 1 + len(gt_lines) + 1 + len(m_lines)
    text_height = total_text_lines * (line_height + line_spacing)
    canvas_height = margin + header_h + spacing + text_height + margin

    canvas = Image.new("RGB", (panel_width, canvas_height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    left_x = margin
    mid_x = margin + orig_image.width + image_gap
    top_y = margin

    canvas.paste(orig_image, (left_x, top_y))
    canvas.paste(vq_recon_image, (mid_x, top_y))

    draw.text((left_x, top_y - 14), "Original", fill=(20, 20, 20), font=font)
    draw.text((mid_x, top_y - 14), "VQ encode -> decode", fill=(20, 20, 20), font=font)

    y = margin + header_h + spacing
    draw.text((margin, y), f"ID: {image_id}", fill=(30, 30, 30), font=font)
    y += line_height + line_spacing

    for line in q_lines:
        draw.text((margin, y), line, fill=(10, 10, 10), font=font)
        y += line_height + line_spacing

    y += line_spacing
    for line in c_lines:
        draw.text((margin, y), line, fill=(10, 10, 10), font=font)
        y += line_height + line_spacing

    y += line_spacing
    for line in gt_lines:
        draw.text((margin, y), line, fill=(10, 10, 10), font=font)
        y += line_height + line_spacing

    y += line_spacing
    for line in m_lines:
        draw.text((margin, y), line, fill=(10, 10, 10), font=font)
        y += line_height + line_spacing

    canvas.save(out_path)


def load_query_images(parquet_path: str, n: int):
    """Return list of (image_id, PIL.Image, gt_caption)."""
    table = pq.read_table(parquet_path)
    stride = max(1, table.num_rows // max(n, 1))

    out = []
    for i in range(n):
        row = table.slice(i * stride, 1).to_pydict()
        image_name = row["image_path"][0]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = Image.open(image_path).convert("RGB")
        gt_text = row["detailed_caption"][0]
        out.append((row["global_index"][0], image, gt_text))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--parquet", type=str, default="data/t2i_midlevel_llama.parquet")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--image-token-num-per-image", type=int, default=576)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    orig_dir = os.path.join(args.out_dir, "original")
    vq_dir = os.path.join(args.out_dir, "vq_recon")
    best_dir = os.path.join(args.out_dir, "best_text_recon")
    panel_dir = os.path.join(args.out_dir, "panels")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(vq_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)

    print(f"Loading VLChatProcessor from {args.model_path}")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)

    print(f"Loading MultiModalityCausalLM from {args.model_path}")
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    # Keep VQ decoder in fp32 for robust conv path with cuDNN disabled.
    vl_gpt.gen_vision_model = vl_gpt.gen_vision_model.to(torch.float32)
    print(f"Model ready on {next(vl_gpt.parameters()).device}")

    print(f"Loading {args.num_images} query images from {args.parquet}")
    samples = load_query_images(args.parquet, args.num_images)

    question_for_display = QUESTION.replace("<image_placeholder>\n", "")
    log_path = os.path.join(args.out_dir, "vq_caption_results.txt")
    total_t0 = time.time()

    vq_psnr_vals, vq_mse_vals = [], []

    with open(log_path, "w") as f:
        for idx, (img_id, orig_img, gt_caption) in enumerate(samples):
            safe_id = sanitize_name(img_id)
            orig_path = os.path.join(orig_dir, f"{idx:02d}_{safe_id}.png")
            vq_path = os.path.join(vq_dir, f"{idx:02d}_{safe_id}.png")
            panel_path = os.path.join(panel_dir, f"{idx:02d}_{safe_id}_panel.png")

            orig_img.save(orig_path)

            t0 = time.time()
            vq_recon_img, _ = encode_decode_image(vl_gpt.gen_vision_model, orig_img)
            vq_recon_img.save(vq_path)

            caption = generate_caption(
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                pil_image=orig_img,
                max_new_tokens=args.max_new_tokens,
            )

            vq_metrics = compute_pixel_metrics(orig_img, vq_recon_img)
            vq_psnr_vals.append(vq_metrics["psnr"])
            vq_mse_vals.append(vq_metrics["mse"])

            save_experiment_panel(
                out_path=panel_path,
                image_id=img_id,
                question_text=question_for_display,
                generated_caption=caption,
                gt_caption=str(gt_caption),
                orig_image=orig_img,
                vq_recon_image=vq_recon_img,
                vq_metrics=vq_metrics,
            )

            dt = time.time() - t0
            header = (
                f"[{idx:02d}] {img_id} ({orig_img.size[0]}x{orig_img.size[1]}) "
                f"{dt:.1f}s | VQ_PSNR={vq_metrics['psnr']:.2f}dB"
            )
            gt_preview = str(gt_caption).replace("\n", " ")[:160]

            print(header)
            print(f"Saved panel: {panel_path}")

            f.write(header + "\n")
            f.write(f"  Q: {question_for_display}\n")
            f.write(f"  Generated caption: {caption}\n")
            f.write(f"  (dataset caption preview: {gt_preview})\n")
            f.write(f"  VQ reconstruction: {vq_path}\n")
            f.write(f"  VQ metrics: {vq_metrics}\n")
            f.write(f"  Panel: {panel_path}\n\n")

        if vq_mse_vals:
            mean_vq_psnr = float(np.mean(vq_psnr_vals))
            mean_vq_mse = float(np.mean(vq_mse_vals))
            summary = (
                f"Mean VQ metrics over {len(vq_mse_vals)} samples: "
                f"PSNR={mean_vq_psnr:.2f} dB, MSE={mean_vq_mse:.2f}"
            )
            print("\n" + summary)
            f.write(summary + "\n")

    print(f"\nAll done in {time.time() - total_t0:.0f}s")
    print(f"Original images: {orig_dir}")
    print(f"VQ reconstructions: {vq_dir}")
    print(f"Panels: {panel_dir}")
    print(f"Report: {log_path}")


if __name__ == "__main__":
    main()
