"""
Cycle-consistency evaluation for Janus-Pro on medical images.

For N samples from t2i_midlevel.parquet:
  1. Load the original image and its mid-level caption.
  2. Text → Image:  generate an image from the caption  (t2i).
  3. Image → Text:  caption the *generated* image       (i2t).
  4. Compare:
       - original caption  vs. regenerated caption  (text similarity)
       - original image    vs. generated image      (visual similarity)
  5. Save a side-by-side visualisation per sample and a summary contact sheet.

Usage:
    python test_cycle_consistency.py
    python test_cycle_consistency.py --num-samples 16 --parallel-size 1
"""

import argparse
import math
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch

torch.backends.cudnn.enabled = False

import re

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from janus.models import VLChatProcessor

# ── Paths ────────────────────────────────────────────────────────────────
PARQUET_PATH = "/work/um00109/MLLM/datasets/VL-Health/t2i_midlevel.parquet"
IMAGE_ROOT   = "/work/um00109/MLLM/datasets/PubMedVision/images"

# ── I2T question (same as test_janus_medical_i2t.py) ─────────────────────
I2T_QUESTION = (
    "<image_placeholder>\n"
    "Describe the main content of the medical image in one sentence. Be specific about the modality, body region, and any notable findings such that it can be regernated using the description. Keep it concise. Do not include any information that is not directly observable in the image."
)


# ── T2I generate (from test_janus_medical.py) ────────────────────────────
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


# ── I2T caption (from test_janus_medical_i2t.py) ─────────────────────────
@torch.inference_mode()
def caption_image_i2t(vl_gpt, vl_chat_processor, pil_image: Image.Image,
                      max_new_tokens: int = 96) -> str:
    conversation = [
        {"role": "<|User|>", "content": I2T_QUESTION},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
    ).to(vl_gpt.device)
    prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(
        next(vl_gpt.parameters()).dtype
    )
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
    answer = vl_chat_processor.tokenizer.decode(
        outputs[0].cpu().tolist(), skip_special_tokens=True
    )
    answer = answer.replace("\u0120", " ").replace("Ġ", " ")
    return answer.strip()


# ── LLM judge for text similarity ─────────────────────────────────────────
JUDGE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

JUDGE_PROMPT = """\
You are an expert medical imaging judge. You will be given two short captions describing a medical image. Rate how semantically similar they are on a scale of 1-5:

1 = Completely different (different modality, different body region)
2 = Slightly related (same broad domain but different modality or region)
3 = Moderately similar (same modality but different details)
4 = Very similar (same modality and region, minor wording differences)
5 = Essentially identical (same meaning, possibly different phrasing)

Caption A: "{caption_a}"
Caption B: "{caption_b}"

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def load_judge(device="cuda"):
    """Load the Phi-3 judge model. Returns a text-generation pipeline."""
    print(f"Loading LLM judge: {JUDGE_MODEL}")
    judge_pipe = pipeline(
        "text-generation",
        model=JUDGE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        max_new_tokens=80,
    )
    return judge_pipe


def llm_judge_score(judge_pipe, caption_a: str, caption_b: str) -> dict:
    """Ask the LLM judge to score similarity. Returns {"score": int, "reason": str}."""
    prompt = JUDGE_PROMPT.format(caption_a=caption_a, caption_b=caption_b)
    messages = [{"role": "user", "content": prompt}]
    out = judge_pipe(messages, do_sample=False, max_new_tokens=80)
    raw = out[0]["generated_text"]
    # Extract assistant reply (last message)
    if isinstance(raw, list):
        raw = raw[-1]["content"]
    # Parse score from JSON-like output
    score_match = re.search(r'"score"\s*:\s*(\d)', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    score = int(score_match.group(1)) if score_match else 0
    reason = reason_match.group(1) if reason_match else raw.strip()
    return {"score": score, "reason": reason}


# ── Visualisation helpers ─────────────────────────────────────────────────
def _get_font(size=13):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def make_comparison(
    orig_img: Image.Image,
    gen_img: Image.Image,
    orig_caption: str,
    regen_caption: str,
    judge_score: int,
    judge_reason: str,
    cell_size: int = 384,
) -> Image.Image:
    """Side-by-side: original image | generated image, with captions below."""
    font = _get_font(13)
    bar_h = 60
    w = cell_size * 2
    h = cell_size + bar_h * 2

    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    orig_resized = orig_img.resize((cell_size, cell_size), Image.LANCZOS)
    gen_resized = gen_img.resize((cell_size, cell_size), Image.LANCZOS)

    canvas.paste(orig_resized, (0, bar_h))
    canvas.paste(gen_resized, (cell_size, bar_h))

    draw.text((6, 4), "Original Image", fill="black", font=font)
    draw.text((cell_size + 6, 4), "Generated Image (t2i)", fill="black", font=font)

    max_chars = max(1, (w - 12) // 8)
    cap_text = f"Caption: {orig_caption}"
    if len(cap_text) > max_chars:
        cap_text = cap_text[: max_chars - 1] + "..."
    draw.text((6, 22), cap_text, fill="blue", font=font)

    y_bottom = cell_size + bar_h + 4
    regen_text = f"Re-caption (i2t): {regen_caption}"
    if len(regen_text) > max_chars:
        regen_text = regen_text[: max_chars - 1] + "..."
    draw.text((6, y_bottom), regen_text, fill="darkred", font=font)

    reason_text = f"LLM Judge: {judge_score}/5 — {judge_reason}"
    if len(reason_text) > max_chars:
        reason_text = reason_text[: max_chars - 1] + "..."
    draw.text((6, y_bottom + 18), reason_text, fill="black", font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--parquet", type=str, default=PARQUET_PATH)
    parser.add_argument("--image-root", type=str, default=IMAGE_ROOT)
    parser.add_argument(
        "--out-dir", type=str,
        default="/work/um00109/MLLM/datasets/VL-Health/cycle_consistency_samples_v2",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--parallel-size", type=int, default=1,
                        help="t2i samples per prompt (best-of-N; uses first)")
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load parquet & sample ────────────────────────────────────────────
    table = pq.read_table(args.parquet)
    stride = max(1, table.num_rows // max(args.num_samples, 1))
    samples = []
    for i in range(args.num_samples):
        row = table.slice(i * stride, 1).to_pydict()
        img_path = os.path.join(args.image_root, row["image_path"][0])
        if not os.path.exists(img_path):
            print(f"  [skip] {img_path} not found")
            continue
        samples.append({
            "image_path": img_path,
            "caption": row["detailed_caption"][0],
            "image_name": row["image_path"][0],
        })
    print(f"Loaded {len(samples)} samples (stride {stride})")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading VLChatProcessor from {args.model_path}")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)

    print(f"Loading MultiModalityCausalLM from {args.model_path}")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    vl_gpt.gen_vision_model = vl_gpt.gen_vision_model.to(torch.float32)
    print(f"Model ready on {next(vl_gpt.parameters()).device}")

    # ── Load LLM judge ─────────────────────────────────────────────────
    judge_pipe = load_judge(device="cuda")

    # ── Run cycle ────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, "cycle_log.txt")
    comparisons = []
    judge_scores = []
    total_t0 = time.time()

    with open(log_path, "w") as log:
        for idx, s in enumerate(samples):
            orig_img = Image.open(s["image_path"]).convert("RGB")
            caption = s["caption"]

            # step 0: first caption original image (for reference)
            t0 = time.time()
            regen_caption = caption_image_i2t(
                vl_gpt, vl_chat_processor, orig_img,
                max_new_tokens=args.max_new_tokens,
            )
            i2t_dt = time.time() - t0


            # Step 1: text → image from generated prompt
            t2i_prompt = build_t2i_prompt(vl_chat_processor, regen_caption)
            t0 = time.time()
            gen_imgs = generate_image(
                vl_gpt, vl_chat_processor, t2i_prompt,
                parallel_size=args.parallel_size,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight,
            )
            t2i_dt = time.time() - t0
            gen_img_pil = Image.fromarray(gen_imgs[0])

            # # Step 2: generated image → text
            # t0 = time.time()
            # regen_caption = caption_image_i2t(
            #     vl_gpt, vl_chat_processor, gen_img_pil,
            #     max_new_tokens=args.max_new_tokens,
            # )
            # i2t_dt = time.time() - t0

            # Step 3: LLM judge similarity
            verdict = llm_judge_score(judge_pipe, caption, regen_caption)
            score = verdict["score"]
            reason = verdict["reason"]
            judge_scores.append(score)

            # Log
            header = (
                f"[{idx:02d}] {s['image_name']}  "
                f"t2i={t2i_dt:.1f}s  i2t={i2t_dt:.1f}s  judge={score}/5"
            )
            print(f"\n{header}")
            print(f"  Original caption : {caption}")
            print(f"  Re-caption (i2t) : {regen_caption}")
            print(f"  Judge reason     : {reason}")

            log.write(header + "\n")
            log.write(f"  Original caption : {caption}\n")
            log.write(f"  Re-caption (i2t) : {regen_caption}\n")
            log.write(f"  Judge score      : {score}/5\n")
            log.write(f"  Judge reason     : {reason}\n\n")

            # # Save individual images
            # orig_img.save(os.path.join(args.out_dir, f"{idx:02d}_orig.png"))
            # gen_img_pil.save(os.path.join(args.out_dir, f"{idx:02d}_gen.png"))

            # Side-by-side comparison
            comp = make_comparison(
                orig_img, gen_img_pil, caption, regen_caption, score, reason,
            )
            comp.save(os.path.join(args.out_dir, f"{idx:02d}_compare.png"))
            comparisons.append(comp)

        # Summary
        mean_score = np.mean(judge_scores) if judge_scores else 0.0
        summary = (
            f"\nMean LLM Judge Score: {mean_score:.2f}/5  "
            f"({len(judge_scores)} samples)"
        )
        print(summary)
        log.write(summary + "\n")

    # ── Contact sheet ────────────────────────────────────────────────────
    if comparisons:
        tile_w, tile_h = comparisons[0].size
        rows = len(comparisons)
        sheet = Image.new("RGB", (tile_w, rows * tile_h), "white")
        for i, comp in enumerate(comparisons):
            sheet.paste(comp, (0, i * tile_h))
        sheet_path = os.path.join(args.out_dir, "contact_sheet.png")
        sheet.save(sheet_path)
        print(f"Contact sheet: {sheet_path}")

    print(f"\nAll done in {time.time() - total_t0:.0f}s")
    print(f"Results: {args.out_dir}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()
