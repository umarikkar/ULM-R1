#!/bin/bash
#SBATCH --job-name=judge_internvlu_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err

set -euo pipefail
mkdir -p /projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu

export HF_HUB_OFFLINE=1
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1

ROOT=/projects/u6gd/umar/codes/ULM-R1
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

hostname
nvidia-smi --list-gpus

python $ROOT/corl/eval/llm_judge_local.py \
    --method_dirs \
        $ROOT/results/eval/internvlu_v2_cached \
        $ROOT/results/eval/internvlu_v2_original_cached_prompt \
    --out_dir $ROOT/results/eval/judge_internvlu_v2 \
    --model_id $MODEL_ID \
    --per_modality 100

echo "[judge] Done."
