"""
Run an image -> caption -> image cycle with Janus-Pro-1B and report cycle loss.

For each sampled example:
1) Generate caption from original image (I2T)
2) Generate image from that caption (T2I)
3) Measure cycle consistency loss between original and regenerated image
4) Save a side-by-side panel (original + regenerated + text/metrics)
"""

import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor


# cuDNN in this conda env fails to initialize on conv2d
# (CUDNN_STATUS_NOT_INITIALIZED). Disable cuDNN so convs fall back to native kernels.
torch.backends.cudnn.enabled = False

IMAGE_ROOT = "/work/um00109/MLLM/datasets/PubMedVision/images"
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "janus_medical_i2t_t2i_cycle_samples"
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


def caption_to_t2i_prompt_text(caption: str) -> str:
    """Make sure T2I prompt text is pure text with no image markers."""
    cleaned = caption.replace("<image_placeholder>", " ")
    cleaned = cleaned.replace("<|User|>", " ").replace("<|Assistant|>", " ")
    return " ".join(cleaned.split())


def build_t2i_prompt(vl_chat_processor: VLChatProcessor, caption: str) -> str:
    prompt_text = caption_to_t2i_prompt_text(caption)
    conversation = [
        # Text-only prompt for T2I stage (no image attached here).
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft + vl_chat_processor.image_start_tag


def build_candidate_texts(generated_caption: str, gt_caption: str, question_text: str):
    candidates = [
        ("generated_caption", generated_caption),
        ("ground_truth_caption", gt_caption),
        ("generic_medical", "A medical image."),
        ("question_prompt", question_text),
        ("generic_xray", "A chest X-ray."),
        ("generic_ct", "A CT scan of the abdomen."),
        ("generic_mri", "An MRI of the brain."),
        ("generic_ultrasound", "An ultrasound image."),
        ("generic_histology", "A histology slide."),
    ]

    seen = set()
    deduped = []
    for name, text in candidates:
        cleaned = caption_to_t2i_prompt_text(text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append((name, cleaned))
    return deduped


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
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    """User-provided Janus-Pro T2I generation loop (CFG + VQ decode)."""
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

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
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
            dim=1,
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_imgs = [Image.fromarray(d) for d in dec]
    return generated_tokens, visual_imgs


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


def feature_match_score(reference_features: torch.Tensor, candidate_features: torch.Tensor) -> float:
    """Compare VQ latent feature maps using symmetric max cosine similarity."""
    reference = reference_features.reshape(reference_features.shape[0], reference_features.shape[1], -1)
    candidate = candidate_features.reshape(candidate_features.shape[0], candidate_features.shape[1], -1)

    reference = reference.permute(0, 2, 1).reshape(-1, reference_features.shape[1])
    candidate = candidate.permute(0, 2, 1).reshape(-1, candidate_features.shape[1])

    reference = F.normalize(reference, p=2, dim=-1)
    candidate = F.normalize(candidate, p=2, dim=-1)
    sim_matrix = reference @ candidate.T
    max_ref = sim_matrix.max(dim=1).values
    max_cand = sim_matrix.max(dim=0).values
    return 0.5 * (max_ref.mean() + max_cand.mean()).item()


def compute_pixel_metrics(orig_pil: Image.Image, recon_pil: Image.Image):
    orig = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)).astype(np.float64)
    recon = np.array(recon_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)).astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    l1 = np.mean(np.abs(orig - recon))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return {"mse": mse, "l1": l1, "psnr": psnr}


def summarize_top_candidates(scored_candidates, top_k: int = 3):
    ranked = sorted(scored_candidates, key=lambda item: item["score"], reverse=True)
    lines = []
    for item in ranked[:top_k]:
        lines.append(f"{item['name']}={item['score']:.4f}: {item['text']}")
    return lines


def save_experiment_panel(
    out_path: str,
    image_id,
    question_text: str,
    generated_caption: str,
    gt_caption: str,
    token_rows,
    orig_image: Image.Image,
    vq_recon_image: Image.Image,
    best_text: str,
    best_image: Image.Image,
    best_score: float,
    vq_metrics: dict,
    best_metrics: dict,
):
    margin = 20
    spacing = 14
    line_spacing = 6
    image_gap = 20

    panel_width = max(orig_image.width * 3 + image_gap * 2 + (margin * 2), 1400)
    header_h = max(orig_image.height, vq_recon_image.height, best_image.height)

    font = ImageFont.load_default()
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (panel_width, 8), "white"))
    text_width = panel_width - (margin * 2)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]

    metric_line = f"VQ recon: PSNR={vq_metrics['psnr']:.2f} dB, MSE={vq_metrics['mse']:.2f}, L1={vq_metrics['l1']:.2f}"
    best_line = f"Recovered prompt score: {best_score:.4f} | Best metrics: PSNR={best_metrics['psnr']:.2f} dB, MSE={best_metrics['mse']:.2f}, L1={best_metrics['l1']:.2f}"

    q_lines = wrap_text(tmp_draw, f"Question: {question_text}", font, text_width)
    c_lines = wrap_text(tmp_draw, f"Generated caption: {generated_caption}", font, text_width)
    gt_lines = wrap_text(tmp_draw, f"Dataset caption: {gt_caption}", font, text_width)
    m_lines = wrap_text(tmp_draw, metric_line, font, text_width)
    b_lines = wrap_text(tmp_draw, f"Recovered prompt: {best_text}", font, text_width)
    t_lines = []
    for name, text, score in token_rows:
        t_lines.extend(wrap_text(tmp_draw, f"{name}: {score:.4f} | {text}", font, text_width))

    total_text_lines = 1 + len(q_lines) + 1 + len(c_lines) + 1 + len(gt_lines) + 1 + len(m_lines) + 1 + len(b_lines) + 1 + len(t_lines)
    text_height = total_text_lines * (line_height + line_spacing)
    canvas_height = margin + header_h + spacing + text_height + margin

    canvas = Image.new("RGB", (panel_width, canvas_height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    left_x = margin
    mid_x = margin + orig_image.width + image_gap
    right_x = panel_width - margin - best_image.width
    top_y = margin

    canvas.paste(orig_image, (left_x, top_y))
    canvas.paste(vq_recon_image, (mid_x, top_y))
    canvas.paste(best_image, (right_x, top_y))

    draw.text((left_x, top_y - 14), "Original", fill=(20, 20, 20), font=font)
    draw.text((mid_x, top_y - 14), "VQ encode -> decode", fill=(20, 20, 20), font=font)
    draw.text((right_x, top_y - 14), "Best text -> decode", fill=(20, 20, 20), font=font)

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

    y += line_spacing
    for line in b_lines:
        draw.text((margin, y), line, fill=(10, 10, 10), font=font)
        y += line_height + line_spacing

    y += line_spacing
    for line in t_lines:
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
    log_path = os.path.join(args.out_dir, "cycle_results.txt")
    total_t0 = time.time()

    vq_psnr_vals, vq_mse_vals, best_score_vals = [], [], []

    with open(log_path, "w") as f:
        for idx, (img_id, orig_img, gt_caption) in enumerate(samples):
            safe_id = sanitize_name(img_id)
            orig_path = os.path.join(orig_dir, f"{idx:02d}_{safe_id}.png")
            vq_path = os.path.join(vq_dir, f"{idx:02d}_{safe_id}.png")
            best_path = os.path.join(best_dir, f"{idx:02d}_{safe_id}.png")
            panel_path = os.path.join(panel_dir, f"{idx:02d}_{safe_id}_panel.png")

            orig_img.save(orig_path)

            t0 = time.time()
            vq_recon_img, orig_features = encode_decode_image(vl_gpt.gen_vision_model, orig_img)
            vq_recon_img.save(vq_path)

            caption = generate_caption(
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                pil_image=orig_img,
                max_new_tokens=args.max_new_tokens,
            )

            candidate_texts = build_candidate_texts(caption, gt_caption, QUESTION.replace("<image_placeholder>\n", ""))
            scored_candidates = []
            for name, text in candidate_texts:
                prompt = build_t2i_prompt(vl_chat_processor, text)
                generated_tokens, generated_images = generate(
                    mmgpt=vl_gpt,
                    vl_chat_processor=vl_chat_processor,
                    prompt=prompt,
                    temperature=args.temperature,
                    parallel_size=1,
                    cfg_weight=args.cfg_weight,
                    image_token_num_per_image=args.image_token_num_per_image,
                    img_size=args.img_size,
                    patch_size=args.patch_size,
                )

                _, candidate_features = encode_decode_image(vl_gpt.gen_vision_model, generated_images[0])
                match_score = feature_match_score(orig_features, candidate_features)
                scored_candidates.append({
                    "name": name,
                    "text": text,
                    "score": match_score,
                    "tokens": generated_tokens,
                    "image": generated_images[0],
                })

            best = max(scored_candidates, key=lambda item: item["score"])
            best_img = best["image"]
            best_img.save(best_path)

            vq_metrics = compute_pixel_metrics(orig_img, vq_recon_img)
            best_metrics = compute_pixel_metrics(orig_img, best_img)
            vq_psnr_vals.append(vq_metrics["psnr"])
            vq_mse_vals.append(vq_metrics["mse"])
            best_score_vals.append(best["score"])

            token_rows = [(item["name"], item["text"], item["score"]) for item in scored_candidates]
            save_experiment_panel(
                out_path=panel_path,
                image_id=img_id,
                question_text=question_for_display,
                generated_caption=caption,
                gt_caption=str(gt_caption),
                token_rows=token_rows,
                orig_image=orig_img,
                vq_recon_image=vq_recon_img,
                best_text=best["text"],
                best_image=best_img,
                best_score=best["score"],
                vq_metrics=vq_metrics,
                best_metrics=best_metrics,
            )

            dt = time.time() - t0
            header = (
                f"[{idx:02d}] {img_id} ({orig_img.size[0]}x{orig_img.size[1]}) "
                f"{dt:.1f}s | VQ_PSNR={vq_metrics['psnr']:.2f}dB best_token_match={best['score']:.4f}"
            )
            gt_preview = str(gt_caption).replace("\n", " ")[:160]

            print(header)
            print(f"Recovered prompt: {best['name']} -> {best['score']:.4f}")
            for line in summarize_top_candidates(scored_candidates, top_k=3):
                print(f"  {line}")
            print(f"Saved panel: {panel_path}")

            f.write(header + "\n")
            f.write(f"  Q: {question_for_display}\n")
            f.write(f"  Generated caption: {caption}\n")
            f.write(f"  (dataset caption preview: {gt_preview})\n")
            f.write(f"  VQ reconstruction: {vq_path}\n")
            f.write(f"  Best text reconstruction: {best_path}\n")
            f.write(f"  Recovered prompt: {best['text']}\n")
            f.write(f"  Recovered prompt source: {best['name']}\n")
            f.write(f"  Recovered prompt score: {best['score']:.6f}\n")
            for line in summarize_top_candidates(scored_candidates, top_k=3):
                f.write(f"  Top candidate: {line}\n")
            f.write(f"  VQ metrics: {vq_metrics}\n")
            f.write(f"  Best metrics: {best_metrics}\n")
            f.write(f"  Panel: {panel_path}\n\n")

        if vq_mse_vals:
            mean_vq_psnr = float(np.mean(vq_psnr_vals))
            mean_vq_mse = float(np.mean(vq_mse_vals))
            mean_best_score = float(np.mean(best_score_vals))
            summary = (
                f"Mean VQ metrics over {len(vq_mse_vals)} samples: "
                f"PSNR={mean_vq_psnr:.2f} dB, MSE={mean_vq_mse:.2f}; "
                f"Mean best-token-match={mean_best_score:.4f}"
            )
            print("\n" + summary)
            f.write(summary + "\n")

    print(f"\nAll done in {time.time() - total_t0:.0f}s")
    print(f"Original images: {orig_dir}")
    print(f"VQ reconstructions: {vq_dir}")
    print(f"Best text reconstructions: {best_dir}")
    print(f"Panels: {panel_dir}")
    print(f"Report: {log_path}")


if __name__ == "__main__":
    main()
