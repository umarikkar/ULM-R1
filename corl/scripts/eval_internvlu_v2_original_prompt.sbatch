#!/bin/bash
#SBATCH --job-name=eval_v2_orig_prompt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err

set -euo pipefail
mkdir -p /projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh

BASE=/projects/u6gd/umar/codes/InternVL-U/InternVL-U
DATADIR=/projects/u6gd/datasets/PubMedVision
TEST_SPLIT=/projects/u6gd/umar/codes/ULM-R1/corl/eval/test_split_small.json
ROOT=/projects/u6gd/umar/codes/ULM-R1

export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=$ROOT
export HF_HUB_OFFLINE=1

hostname
nvidia-smi --list-gpus

# ── Stage 1: Generate — Cached model with Original_Caption prompts ────────────
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu
echo "[eval] Generating cached model with original caption prompts..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $ROOT/results/internvlu_t2i_cached_captions_v2/checkpoint-15000 \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out_dir $ROOT/results/eval/internvlu_v2_cached_original_prompt

# ── Stage 2: Judge both dirs on original captions ─────────────────────────────
echo "[eval] Running Qwen judge on both models with original caption prompts..."
python $ROOT/corl/eval/llm_judge_local.py \
    --method_dirs \
        $ROOT/results/eval/internvlu_v2_cached_original_prompt \
        $ROOT/results/eval/internvlu_v2_original \
    --out_dir $ROOT/results/eval/judge_internvlu_v2_original_prompt \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --per_modality 100

echo "[eval] Done."
