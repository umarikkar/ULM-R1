#!/bin/bash
#SBATCH --job-name=eval_internvlu_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
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

CACHED_CKPT=$ROOT/results/internvlu_t2i_cached_captions_v3/checkpoint-35000
ORIGINAL_CKPT=$ROOT/results/internvlu_t2i_original_captions_v3/checkpoint-35000

export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=$ROOT
export HF_HUB_OFFLINE=1

hostname
nvidia-smi --list-gpus

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Generate (internvlu env, GPU)
# ═══════════════════════════════════════════════════════════════════════════════
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu

echo "[eval] 1/4 — cached model + cached captions..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $CACHED_CKPT \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field cached_captions \
    --out_dir $ROOT/results/eval/internvlu_v3_cached

echo "[eval] 2/4 — cached model + original captions..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $CACHED_CKPT \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out_dir $ROOT/results/eval/internvlu_v3_cached_original_prompt

echo "[eval] 3/4 — original model + cached captions..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $ORIGINAL_CKPT \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field cached_captions \
    --out_dir $ROOT/results/eval/internvlu_v3_original_cached_prompt

echo "[eval] 4/4 — original model + original captions..."
python $ROOT/corl/eval/generate_internvlu.py \
    --base_model $BASE \
    --adapter_dir $ORIGINAL_CKPT \
    --test_split $TEST_SPLIT \
    --data_dir $DATADIR \
    --caption_field Original_Caption \
    --out_dir $ROOT/results/eval/internvlu_v3_original

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Metrics (corlv2 env, open_clip/BiomedCLIP)
# ═══════════════════════════════════════════════════════════════════════════════
conda activate /projects/u6gd/umar/.conda/envs/corlv2
export HF_HUB_OFFLINE=0
export OPENBLAS_NUM_THREADS=8

echo "[eval] Metrics — cached model + cached captions..."
python $ROOT/corl/eval/compute_metrics.py \
    --manifest_glob "$ROOT/results/eval/internvlu_v3_cached/manifest_shard*.json" \
    --data_dir $DATADIR \
    --caption_field cached_captions \
    --out $ROOT/results/eval/internvlu_v3_cached/metrics.json

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

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Qwen VLM judge (internvlu env, GPU)
# ═══════════════════════════════════════════════════════════════════════════════
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu
export HF_HUB_OFFLINE=1

echo "[eval] Qwen judge — all 4 dirs..."
python $ROOT/corl/eval/llm_judge_local.py \
    --method_dirs \
        $ROOT/results/eval/internvlu_v3_cached \
        $ROOT/results/eval/internvlu_v3_cached_original_prompt \
        $ROOT/results/eval/internvlu_v3_original_cached_prompt \
        $ROOT/results/eval/internvlu_v3_original \
    --out_dir $ROOT/results/eval/judge_internvlu_v3 \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --per_modality 100

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Print summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[eval] === Quantitative Results (v3 checkpoint-35000) ==="
python3 -c "
import json
runs = [
    ('cached_model+cached_prompt',   '$ROOT/results/eval/internvlu_v3_cached/metrics.json'),
    ('cached_model+original_prompt', '$ROOT/results/eval/internvlu_v3_cached_original_prompt/metrics.json'),
    ('original_model+cached_prompt', '$ROOT/results/eval/internvlu_v3_original_cached_prompt/metrics.json'),
    ('original_model+original_prompt','$ROOT/results/eval/internvlu_v3_original/metrics.json'),
]
print(f'{\"run\":<35s}  {\"FID\":>8s}  {\"CLIPScore\":>9s}  {\"I2I\":>7s}')
for name, path in runs:
    d = json.load(open(path))
    m = d['macro']
    print(f'{name:<35s}  {m[\"fid_biomedclip\"]:>8.1f}  {m[\"clipscore_gen\"]:>9.4f}  {m[\"i2i_cosine\"]:>7.4f}')
"

echo "[eval] Done."
