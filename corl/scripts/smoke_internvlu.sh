#!/bin/bash
set -euo pipefail

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu

echo "=== GPU SANITY CHECK ==="
python -c "import platform,torch; print(platform.machine(),'|',torch.__version__,'|',torch.cuda.is_available())"

export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1
export HF_HUB_OFFLINE=1
cd /projects/u6gd/umar/codes/InternVL-U

echo "=== SMOKE TEST ==="
python - <<'PYEOF'
import os, sys, random
sys.path.insert(0, os.environ["PYTHONPATH"])
from trl import SFTConfig, ModelConfig
from corl.open_r1.sft_internvlu_alignment import SFTScriptArguments, main

CKPT    = "/projects/u6gd/umar/codes/InternVL-U/InternVL-U"
DATASET = "PubMedVision_CachedCaptions_K4.json"
DATADIR = "/projects/u6gd/datasets/PubMedVision"

main(
    SFTScriptArguments(
        dataset_name=DATASET, dataset_train_split="train", data_dir=DATADIR,
        image_column="image", caption_source="original",
        caption_column="cached_captions", gen_image_size=512,
        prompt_dropout_prob=0.1, max_prompt_length=1024, max_samples=4),
    SFTConfig(
        output_dir="/projects/u6gd/umar/codes/ULM-R1/results/smoke_internvlu",
        per_device_train_batch_size=2, max_steps=2,
        learning_rate=1e-4, logging_steps=1, save_strategy="no", eval_strategy="no",
        bf16=True, gradient_checkpointing=True, remove_unused_columns=False,
        dataloader_num_workers=0, report_to=[]),
    ModelConfig(model_name_or_path=CKPT, use_peft=True, lora_r=8, lora_alpha=16, lora_dropout=0.05),
    max_samples=4,
)
print("SMOKE OK")
PYEOF
