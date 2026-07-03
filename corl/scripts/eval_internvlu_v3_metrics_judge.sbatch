#!/bin/bash
#SBATCH --job-name=eval_v3_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err

set -euo pipefail
mkdir -p /projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh

DATADIR=/projects/u6gd/datasets/PubMedVision
ROOT=/projects/u6gd/umar/codes/ULM-R1

export PYTHONPATH=$ROOT

hostname
nvidia-smi --list-gpus

# ── Stage 1: Remaining metrics (corlv2 env) ───────────────────────────────────
conda activate /projects/u6gd/umar/.conda/envs/corlv2
export HF_HUB_OFFLINE=0
export OPENBLAS_NUM_THREADS=8

echo "[eval] Metrics — cached model + original captions..."
python $ROOT/corl/eval/compute_metrics.py \
    --manifest_glob "$ROOT/results/eval/internvlu_v3_cached_original_prompt/manifest_shard*.json" \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out $ROOT/results/eval/internvlu_v3_cached_original_prompt/metrics.json

echo "[eval] Metrics — original model + cached captions..."
python $ROOT/corl/eval/compute_metrics.py \
    --manifest_glob "$ROOT/results/eval/internvlu_v3_original_cached_prompt/manifest_shard*.json" \
    --data_dir $DATADIR \
    --caption_field cached_captions \
    --out $ROOT/results/eval/internvlu_v3_original_cached_prompt/metrics.json

echo "[eval] Metrics — original model + original captions..."
python $ROOT/corl/eval/compute_metrics.py \
    --manifest_glob "$ROOT/results/eval/internvlu_v3_original/manifest_shard*.json" \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out $ROOT/results/eval/internvlu_v3_original/metrics.json

# ── Stage 2: Qwen judge (internvlu env, GPU) ──────────────────────────────────
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu
export HF_HUB_OFFLINE=1

echo "[eval] Qwen judge — all 4 v3 dirs..."
python $ROOT/corl/eval/llm_judge_local.py \
    --method_dirs \
        $ROOT/results/eval/internvlu_v3_cached \
        $ROOT/results/eval/internvlu_v3_cached_original_prompt \
        $ROOT/results/eval/internvlu_v3_original_cached_prompt \
        $ROOT/results/eval/internvlu_v3_original \
    --out_dir $ROOT/results/eval/judge_internvlu_v3 \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --per_modality 100

# ── Stage 3: Summary ──────────────────────────────────────────────────────────
echo ""
echo "=== Quantitative Results (v3 checkpoint-35000) ==="
python3 -c "
import json
runs = [
    ('cached_model+cached_prompt',    '$ROOT/results/eval/internvlu_v3_cached/metrics.json'),
    ('cached_model+original_prompt',  '$ROOT/results/eval/internvlu_v3_cached_original_prompt/metrics.json'),
    ('original_model+cached_prompt',  '$ROOT/results/eval/internvlu_v3_original_cached_prompt/metrics.json'),
    ('original_model+original_prompt','$ROOT/results/eval/internvlu_v3_original/metrics.json'),
]
print(f'{\"run\":<35s}  {\"FID\":>8s}  {\"CLIPScore\":>9s}  {\"I2I\":>7s}')
for name, path in runs:
    d = json.load(open(path))
    m = d['macro']
    print(f'{name:<35s}  {m[\"fid_biomedclip\"]:>8.1f}  {m[\"clipscore_gen\"]:>9.4f}  {m[\"i2i_cosine\"]:>7.4f}')
"

echo "[eval] Done."
