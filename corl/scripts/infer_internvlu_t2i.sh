#!/bin/bash
set -euo pipefail

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu

export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1
export HF_HUB_OFFLINE=1

cd /projects/u6gd/umar/codes/InternVL-U

python - <<'PYEOF'
import os, sys, json, random, textwrap
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ["PYTHONPATH"])

from internvlu import InternVLUPipeline
from peft import PeftModel
from corl.open_r1.trainer.sft_trainer_alignment_internvlu import InternVLUT2IFlowMatch

BASE_CKPT     = "/projects/u6gd/umar/codes/InternVL-U/InternVL-U"
CACHED_CKPT   = "/projects/u6gd/umar/codes/ULM-R1/results/internvlu_t2i_cached_captions_v2/checkpoint-12000"
ORIGINAL_CKPT = "/projects/u6gd/umar/codes/ULM-R1/results/internvlu_t2i_original_captions_v2/checkpoint-11000"
EVAL_JSON     = "/projects/u6gd/umar/codes/ULM-R1/corl/eval/test_split_small.json"
IMG_ROOT      = "/projects/u6gd/datasets/PubMedVision"
OUT_DIR       = "/projects/u6gd/umar/codes/ULM-R1/results/internvlu_samples_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Pick 1 sample per modality, sorted for reproducibility
with open(EVAL_JSON) as f:
    eval_data = json.load(f)

MODALITIES = [
    "Computed Tomography",
    "Magnetic Resonance Imaging",
    "Ultrasound",
    "Fundus Photography",
    "Microscopy Images",
    "Endoscopy",
]
samples = {}
for row in eval_data:
    mod = row["modality"]
    if mod in MODALITIES and mod not in samples:
        samples[mod] = row
samples = [samples[m] for m in MODALITIES]

orig_prompts   = [s["Original_Caption"] for s in samples]
cached_prompts = [s["cached_captions"][0] for s in samples]
modalities     = [s["modality"] for s in samples]
img_paths      = [os.path.join(IMG_ROOT, s["image"]) for s in samples]

print(f"Eval samples ({len(samples)}):")
for i, s in enumerate(samples):
    print(f"  [{s['modality']}] orig={s['Original_Caption'][:60]}...")
    print(f"              cached={s['cached_captions'][0][:60]}...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16


def load_pipeline_with_lora(adapter_dir):
    print(f"  Loading base pipeline from {BASE_CKPT} ...")
    pipe = InternVLUPipeline.from_pretrained(BASE_CKPT, torch_dtype=dtype)
    for attr in vars(pipe).values():
        if hasattr(attr, 'to') and callable(attr.to):
            try: attr.to(device)
            except Exception: pass
    gd = pipe.generation_decoder
    flow_model = InternVLUT2IFlowMatch(
        vlm=pipe.vlm,
        generation_decoder=gd,
        vlm_select_layer=gd.config.vlm_select_layer,
        flow_shift=gd.config.flow_shift,
        logit_mean=gd.config.logit_mean,
        logit_std=gd.config.logit_std,
    )
    print(f"  Loading LoRA adapter from {adapter_dir} ...")
    flow_model = PeftModel.from_pretrained(flow_model, adapter_dir)
    flow_model = flow_model.merge_and_unload()
    print("  LoRA merged.")
    return pipe


def generate(pipe, prompts, tag):
    images = []
    for i, prompt in enumerate(prompts):
        print(f"  [{tag}] {i+1}/{len(prompts)}: {prompt[:70]}...")
        out = pipe(
            prompt,
            generation_mode="image",
            height=512, width=512,
            num_inference_steps=20,
            all_cfg_scale=4.5,
            part_cfg_scale=2.0,
        )
        images.append(out.images[0])
    return images


# ── Run 1: Cached Captions model — prompted with cached captions ──────────────
print("\n=== Cached Captions model (step 12000) — using cached captions as prompts ===")
pipe_cached = load_pipeline_with_lora(CACHED_CKPT)
imgs_cached = generate(pipe_cached, cached_prompts, "cached")
del pipe_cached
torch.cuda.empty_cache()

# ── Run 2: Original Captions model — prompted with original captions ──────────
print("\n=== Original Captions model (step 11000) — using original captions as prompts ===")
pipe_orig = load_pipeline_with_lora(ORIGINAL_CKPT)
imgs_orig = generate(pipe_orig, orig_prompts, "original")
del pipe_orig
torch.cuda.empty_cache()

# ── Plot: GT | Cached model | Original model ──────────────────────────────────
from PIL import Image

N = len(samples)
fig, axes = plt.subplots(N, 3, figsize=(15, 4.5 * N))
fig.suptitle('InternVL-U T2I — PubMedVQA Eval Samples\n(GT image | Cached model + cached caption | Original model + original caption)',
             fontsize=12, fontweight='bold')

col_titles = ["GT Image", "Cached K4 model\n(cached caption prompt)", "Original model\n(original caption prompt)"]
for j, title in enumerate(col_titles):
    axes[0][j].set_title(title, fontsize=10, fontweight='bold')

for i, s in enumerate(samples):
    gt = Image.open(img_paths[i]).convert("RGB")
    row_label = f"[{modalities[i]}]\n" + "\n".join(textwrap.wrap(orig_prompts[i][:80], 28))

    for j, img in enumerate([gt, imgs_cached[i], imgs_orig[i]]):
        ax = axes[i][j]
        ax.imshow(img)
        ax.axis("off")
    axes[i][0].set_ylabel(row_label, fontsize=7, rotation=0, labelpad=170, va='center')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "t2i_eval_grid.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nSaved grid: {out_path}")

# Save individual images
for i, s in enumerate(samples):
    slug = s["modality"].lower().replace(" ", "_")
    imgs_cached[i].save(os.path.join(OUT_DIR, f"cached_{i+1}_{slug}.png"))
    imgs_orig[i].save(os.path.join(OUT_DIR, f"original_{i+1}_{slug}.png"))
    Image.open(img_paths[i]).convert("RGB").save(os.path.join(OUT_DIR, f"gt_{i+1}_{slug}.png"))
print(f"Individual images saved to {OUT_DIR}/")
PYEOF
