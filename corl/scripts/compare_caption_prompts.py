"""Quick A/B/C/D test of the candidate i2t prompts.

Runs Janus-Pro-1B i2t on N sample images with each of the prompt variants
in `PROMPTS` and prints the outputs side-by-side. Single-GPU, no DDP.

Usage:
    python corl/scripts/compare_caption_prompts.py --n_images 5 --k 2
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, set_seed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from janus.models import VLChatProcessor


PROMPTS = {
    "P1_short_generic": (
        "Describe the main content of the image in one sentence."
    ),
    "P2_medium_med": (
        "Describe this medical image in one to two sentences. Include the "
        "imaging modality, dominant colours and contrast, spatial arrangement "
        "of key structures, and any salient visual features like lesions or "
        "abnormalities. Be specific enough to reconstruct the image from the "
        "description alone."
    ),
    "P3_long_structured": (
        "Describe this medical image in one to two sentences, focusing "
        "exclusively on visually observable features. Include: (1) the imaging "
        "modality and orientation if identifiable (e.g. axial CT, H&E "
        "histology, MRI or other), (2) the dominant colours, tones, and "
        "contrast patterns, (3) the spatial arrangement and shape of visible "
        "structures, (4) texture and surface appearance, and (5) any salient "
        "or abnormal visual features such as lesions, colour hotspots, or "
        "irregular morphology. Avoid diagnostic conclusions — describe only "
        "what is directly visible to reconstruct the image from the "
        "description alone."
    ),
    "P4_short_med_active": (
        "Describe this medical image in one to two sentences. Describe only "
        "what is directly visible to reconstruct the image from the "
        "description alone."
    ),
}


def _fix(s):
    return s.replace("Ġ", " ").replace("Ċ", "\n").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument(
        "--data_json",
        default=None,
        help="Path to PubMedVision_Original_Caption.json. If unset, auto-pick by hostname.",
    )
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--k", type=int, default=2,
                    help="Number of samples per (image, prompt) to show sampling variance.")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_md", default="./results/caption_prompt_comparison.md",
                    help="Markdown file with results.")
    args = ap.parse_args()

    # Auto-pick paths if not provided.
    if args.data_dir is None or args.data_json is None:
        host = os.uname().nodename.split(".")[0]
        default_data_dir = {
            "cvssp-retina03": "/work/um00109/MLLM/datasets/PubMedVision",
            "ulws072": "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision",
        }.get(host, "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision")
        args.data_dir = args.data_dir or default_data_dir
        args.data_json = args.data_json or os.path.join(
            args.data_dir, "PubMedVision_Original_Caption.json"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + processor.
    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    eos_id = processor.tokenizer.eos_token_id
    bos_id = processor.tokenizer.bos_token_id

    # Pick N images.
    with open(args.data_json, "r") as f:
        rows = json.load(f)
    # Spread the picks so we hit different modalities, not the first N which are
    # all from the same article.
    stride = max(1, len(rows) // args.n_images)
    picks = []
    for i in range(args.n_images):
        idx = (i * stride) % len(rows)
        r = rows[idx]
        p = os.path.join(args.data_dir, r["image"][0])
        if not os.path.exists(p):
            continue
        picks.append((r, p))
    print(f"Picked {len(picks)} images.")

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    md = ["# i2t prompt comparison\n",
          f"model: `{args.model}` | k={args.k} | temp={args.temperature} | "
          f"top_p={args.top_p} | max_new_tokens={args.max_new_tokens} | seed={args.seed}\n"]

    for img_idx, (row, path) in enumerate(picks):
        img = Image.open(path).convert("RGB")
        gt_cap = row.get("Original_Caption", "")
        header = (
            f"\n## Image {img_idx + 1}: `{row['image'][0]}`  "
            f"(modality: {row.get('modality')}, body: {row.get('body_part')})\n"
            f"**Original_Caption (PubMed):** {gt_cap}\n"
        )
        print(header)
        md.append(header)

        for prompt_name, prompt_text in PROMPTS.items():
            md.append(f"\n### {prompt_name}\n")
            md.append(f"> _{prompt_text}_\n")
            print(f"\n-- {prompt_name} --")
            for k_i in range(args.k):
                # Re-seed per (image, prompt, sample) for repeatability without
                # losing variance across samples.
                set_seed(args.seed + 1000 * img_idx + 100 * list(PROMPTS).index(prompt_name) + k_i)

                conv = [
                    {"role": "<|User|>",
                     "content": f"<image_placeholder>\n{prompt_text}"},
                    {"role": "<|Assistant|>", "content": ""},
                ]
                prepared = processor(
                    conversations=[conv], images=[[img]], force_batchify=True,
                ).to(device)
                with torch.inference_mode():
                    inputs_embeds = model.prepare_inputs_embeds(**prepared)
                    out_ids = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepared.attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_return_sequences=1,
                        pad_token_id=eos_id,
                        bos_token_id=bos_id,
                        eos_token_id=eos_id,
                    )
                text = _fix(processor.tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True
                )[0])
                line = f"  [{k_i}] {text}"
                print(line)
                md.append(f"- **sample {k_i}:** {text}\n")

    with open(args.out_md, "w") as f:
        f.writelines(md)
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
