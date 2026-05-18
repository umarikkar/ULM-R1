"""
Load and validate Janus checkpoint directories as runnable models.

Example:
    python evaluate_checkpoints.py \
      --checkpoints \
      /vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/JanusPro-1B-CoRL-noMM/CycleOnly-G4-bs16-genHead-genAligner/checkpoint-200 \
      /vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/JanusPro-1B-CoRL-Uniified/RFT22k-CycleMatchAccFormat-UniReward-G4-beta004-bs16/checkpoint-800
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
    "results/results/JanusPro-1B-AlignmentSFT/AR_Loss/checkpoint-6400",
    # "results/results/JanusPro-1B-AlignmentSFT/AR_Loss_highLR/checkpoint-5400",
    # "/projects/u6gd/umar/codes/ULM-R1/JanusPro-1B-CoRL-Uniified/"
    # "RFT22k-CycleMatchAccFormat-UniReward-G4-beta004-bs16/checkpoint-800",
]



# PARQUET_PATH = "/projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama.parquet"
# IMAGE_ROOT = "/projects/u6gd/datasets/PubMedVision/images/"


PARQUET_PATH = "/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/data/t2i_midlevel_llama.parquet"
IMAGE_ROOT = "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision/images/"




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
        default="./checkpoint_evals",
        help="Root output directory. A subfolder is created per checkpoint.",
    )
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG weight for t2i.")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature for t2i.")
    parser.add_argument(
        "--run-tag", type=str, default=None,
        help="Appended to per-checkpoint output dir name (default: cfg{X}_t{Y}).",
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
    return dec, generated_tokens[0].detach().cpu().to(torch.long)  # imgs, tokens


@torch.inference_mode()
def encode_image_to_tokens(
    mmgpt,
    pil_image: Image.Image,
    img_size: int = 384,
    patch_size: int = 16,
):
    device = next(mmgpt.gen_vision_model.parameters()).device
    dtype = next(mmgpt.gen_vision_model.parameters()).dtype

    img = pil_image.convert("RGB").resize((img_size, img_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    _, _, info = mmgpt.gen_vision_model.encode(x)
    indices = info[2].view(-1).to(torch.long).cpu()
    return indices  # (img_size/patch_size)**2 tokens


@torch.inference_mode()
def decode_tokens_to_image(
    mmgpt,
    tokens: torch.Tensor,
    img_size: int = 384,
    patch_size: int = 16,
) -> Image.Image:
    device = next(mmgpt.gen_vision_model.parameters()).device
    grid = img_size // patch_size
    codes = tokens.view(1, -1).to(device=device, dtype=torch.int)
    dec = mmgpt.gen_vision_model.decode_code(
        codes, shape=[1, 8, grid, grid],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(dec[0])


def token_stats(tokens: torch.Tensor, codebook_size: int = 16384) -> dict:
    t = tokens.to(torch.long)
    n = int(t.numel())
    counts = torch.bincount(t, minlength=codebook_size).to(torch.float64)
    nz = counts[counts > 0]
    probs = nz / nz.sum()
    entropy = float(-(probs * probs.log2()).sum())
    sorted_counts, _ = torch.sort(counts, descending=True)
    top1_frac = float(sorted_counts[0] / n)
    top5_frac = float(sorted_counts[:5].sum() / n)
    top10_frac = float(sorted_counts[:10].sum() / n)
    unique = int((counts > 0).sum())
    coverage = unique / codebook_size
    perplexity = float(torch.exp(torch.tensor(entropy) * np.log(2)))
    eff_usage = perplexity / codebook_size
    # consecutive repetition rate
    if n > 1:
        repeat_rate = float((t[1:] == t[:-1]).float().mean())
    else:
        repeat_rate = 0.0
    return {
        "n": n,
        "unique": unique,
        "coverage": coverage,
        "entropy_bits": entropy,
        "perplexity": perplexity,
        "eff_usage": eff_usage,
        "top1_frac": top1_frac,
        "top5_frac": top5_frac,
        "top10_frac": top10_frac,
        "repeat_rate": repeat_rate,
        "counts": counts,
    }


def format_stats(name: str, s: dict) -> str:
    return (
        f"  [{name}] unique={s['unique']}/{s['n']} cov={s['coverage']*100:.2f}% "
        f"H={s['entropy_bits']:.2f}b ppl={s['perplexity']:.1f} "
        f"top1={s['top1_frac']*100:.1f}% top5={s['top5_frac']*100:.1f}% "
        f"top10={s['top10_frac']*100:.1f}% repeat={s['repeat_rate']*100:.1f}%"
    )


def spatial_periodicity(tokens: torch.Tensor, grid: int = 24) -> dict:
    """Detect tiling: how often whole rows / columns repeat in the 24x24 grid."""
    if tokens.numel() != grid * grid:
        return {"row_dup_rate": 0.0, "col_dup_rate": 0.0, "row_period": 0, "col_period": 0}
    g = tokens.view(grid, grid).to(torch.long)
    # fraction of rows identical to the row above
    row_dup = float((g[1:] == g[:-1]).all(dim=1).float().mean())
    col_dup = float((g[:, 1:] == g[:, :-1]).all(dim=0).float().mean())
    # smallest period p where row[i] == row[i+p] for most i
    def best_period(mat):
        best_p, best_score = 0, 0.0
        for p in range(1, grid):
            score = float((mat[p:] == mat[:-p]).all(dim=1).float().mean())
            if score > best_score:
                best_score, best_p = score, p
            if score > 0.9:
                return p, score
        return best_p, best_score
    row_p, row_s = best_period(g)
    col_p, col_s = best_period(g.t().contiguous())
    return {
        "row_dup_rate": row_dup,
        "col_dup_rate": col_dup,
        "row_period": row_p, "row_period_score": row_s,
        "col_period": col_p, "col_period_score": col_s,
    }


def format_spatial(name: str, sp: dict) -> str:
    return (
        f"  [{name} spatial] row_dup={sp['row_dup_rate']*100:.1f}% "
        f"col_dup={sp['col_dup_rate']*100:.1f}% "
        f"row_period={sp['row_period']}(score={sp['row_period_score']:.2f}) "
        f"col_period={sp['col_period']}(score={sp['col_period_score']:.2f})"
    )


def collapse_verdict(s: dict) -> str:
    flags = []
    if s["top1_frac"] > 0.20:
        flags.append(f"top1>{s['top1_frac']*100:.0f}%")
    if s["top10_frac"] > 0.50:
        flags.append(f"top10>{s['top10_frac']*100:.0f}%")
    if s["unique"] < 0.3 * s["n"]:
        flags.append(f"low-unique({s['unique']}/{s['n']})")
    if s["repeat_rate"] > 0.10:
        flags.append(f"repeats={s['repeat_rate']*100:.0f}%")
    return "COLLAPSED: " + ", ".join(flags) if flags else "ok"


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
    recon_img: Optional[Image.Image] = None,
    title_override: Optional[str] = None,
):
    pad = 20
    gap = 20

    panels = [("Original", orig_img)]
    if recon_img is not None:
        panels.append(("VQ Recon (encode→decode)", recon_img))
    panels.append(("T2I Generated", regen_img))

    target_h = max(p[1].height for p in panels)
    sized = [(name, resize_to_height(img, target_h)) for name, img in panels]

    font = ImageFont.load_default()

    canvas_w = sum(img.width for _, img in sized) + pad * 2 + gap * (len(sized) - 1)
    temp_canvas = Image.new("RGB", (canvas_w, 10), color=(255, 255, 255))
    draw_temp = ImageDraw.Draw(temp_canvas)

    title = title_override or " | ".join(name for name, _ in sized)
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
    x = pad
    for _, img in sized:
        canvas.paste(img, (x, y_img))
        x += img.width + gap

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

        run_tag = args.run_tag or f"cfg{args.cfg:g}_t{args.temp:g}"
        ckpt_out_dir = Path(args.out_dir) / f"{checkpoint_to_dirname(ckpt)}__{run_tag}"
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to: {ckpt_out_dir}")

        agg_enc_counts, agg_t2i_counts = [], []
        agg_t2i_top1, agg_t2i_unique = [], []
        agg_t2i_rowdup, agg_t2i_coldup = [], []

        for idx, s in enumerate(samples):
            orig_img = Image.open(s["image_path"]).convert("RGB")
            caption = s["caption"]
            t2i_prompt = build_t2i_prompt(vl_chat_processor, caption)
            gen_imgs, t2i_tokens = generate_image(
                vl_gpt, vl_chat_processor, t2i_prompt,
                parallel_size=1,
                temperature=args.temp,
                cfg_weight=args.cfg,
            )
            gen_img_pil = Image.fromarray(gen_imgs[0])

            enc_tokens = encode_image_to_tokens(vl_gpt, orig_img)
            recon_img = decode_tokens_to_image(vl_gpt, enc_tokens)
            codebook_size = int(vl_gpt.gen_vision_model.quantize.embedding.weight.shape[0])
            s_enc = token_stats(enc_tokens, codebook_size=codebook_size)
            s_t2i = token_stats(t2i_tokens, codebook_size=codebook_size)
            sp_enc = spatial_periodicity(enc_tokens)
            sp_t2i = spatial_periodicity(t2i_tokens)
            print(f"  Token stats for sample {idx} (codebook={codebook_size}):")
            print(format_stats("img-encode", s_enc))
            print(format_stats("t2i-gen   ", s_t2i))
            print(format_spatial("img-encode", sp_enc))
            print(format_spatial("t2i-gen   ", sp_t2i))
            print(f"  collapse verdict (t2i): {collapse_verdict(s_t2i)}")
            agg_enc_counts.append(s_enc["counts"])
            agg_t2i_counts.append(s_t2i["counts"])
            agg_t2i_top1.append(s_t2i["top1_frac"])
            agg_t2i_unique.append(s_t2i["unique"])
            agg_t2i_rowdup.append(sp_t2i["row_dup_rate"])
            agg_t2i_coldup.append(sp_t2i["col_dup_rate"])

            image_stem = Path(s["image_name"]).stem
            save_path = ckpt_out_dir / f"{idx:05d}_{image_stem}.png"
            save_comparison_image(
                out_path=save_path,
                orig_img=orig_img,
                regen_img=gen_img_pil,
                caption=caption,
                recon_img=recon_img,
                title_override=f"Original | VQ Recon | T2I (cfg={args.cfg}, T={args.temp})",
            )

            print(f"[{idx + 1}/{len(samples)}] saved {save_path}")

        if agg_t2i_counts:
            stacked_enc = torch.stack(agg_enc_counts).sum(dim=0)
            stacked_t2i = torch.stack(agg_t2i_counts).sum(dim=0)
            enc_unique_total = int((stacked_enc > 0).sum())
            t2i_unique_total = int((stacked_t2i > 0).sum())
            cb = stacked_enc.numel()
            print(f"\n=== Checkpoint summary ({len(samples)} samples) ===")
            print(f"  codebook size: {cb}")
            print(f"  unique tokens used across all samples — img-encode: {enc_unique_total}/{cb} ({enc_unique_total/cb*100:.2f}%)")
            print(f"  unique tokens used across all samples — t2i-gen   : {t2i_unique_total}/{cb} ({t2i_unique_total/cb*100:.2f}%)")
            print(f"  mean t2i top1 frac : {np.mean(agg_t2i_top1)*100:.2f}%  (>20% suggests collapse)")
            print(f"  mean t2i unique/sample : {np.mean(agg_t2i_unique):.1f}/576")
            print(f"  mean t2i row-dup rate  : {np.mean(agg_t2i_rowdup)*100:.2f}%  (high => horizontal tiling)")
            print(f"  mean t2i col-dup rate  : {np.mean(agg_t2i_coldup)*100:.2f}%  (high => vertical tiling)")

            



if __name__ == "__main__":
    main()
