#!/bin/bash
#SBATCH --job-name=eval_internvlu_v2
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

# ── Stage 1a: Generate — Cached Captions model (checkpoint-15000) ─────────────
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu
echo "[eval] Generating cached captions v2..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $ROOT/results/internvlu_t2i_cached_captions_v2/checkpoint-15000 \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field cached_captions \
    --out_dir $ROOT/results/eval/internvlu_v2_cached

# ── Stage 1b: Generate — Original Captions model (checkpoint-15000) ───────────
echo "[eval] Generating original captions v2..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $ROOT/results/internvlu_t2i_original_captions_v2/checkpoint-15000 \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out_dir $ROOT/results/eval/internvlu_v2_original

# ── Stage 2: Compute metrics (needs open_clip — switch to corl env) ───────────
conda activate /projects/u6gd/umar/.conda/envs/corlv2
export HF_HUB_OFFLINE=0   # BiomedCLIP may need hub access if not cached

echo "[eval] Computing metrics — cached captions v2..."
python $ROOT/corl/eval/compute_metrics.py \
    --manifest_glob "$ROOT/results/eval/internvlu_v2_cached/manifest_shard*.json" \
    --data_dir $DATADIR \
    --caption_field cached_captions \
    --out $ROOT/results/eval/internvlu_v2_cached/metrics.json

echo "[eval] Computing metrics — original captions v2..."
python $ROOT/corl/eval/compute_metrics.py \
    --manifest_glob "$ROOT/results/eval/internvlu_v2_original/manifest_shard*.json" \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out $ROOT/results/eval/internvlu_v2_original/metrics.json

echo "[eval] Done. Results:"
python3 -c "
import json, sys
for name, path in [
    ('cached_v2',  '$ROOT/results/eval/internvlu_v2_cached/metrics.json'),
    ('original_v2','$ROOT/results/eval/internvlu_v2_original/metrics.json'),
]:
    d = json.load(open(path))
    m = d['macro']
    print(f'{name}:  FID={m[\"fid_biomedclip\"]:.2f}  CLIPScore={m[\"clipscore_gen\"]:.4f}  I2I={m[\"i2i_cosine\"]:.4f}')
"
