#!/bin/bash
# Janus-Pro T2I alignment SFT on a chosen caption-granularity level.
#
# Levels come from build_caption_cache_levels.py (PubMedVision_CachedCaptions_Levels.json):
#   l1_meta : templated modality/body_part label
#   l1      : one sentence (modality + region)
#   l2      : one sentence + main finding
#   l3      : detailed paragraph
#
# Usage:
#   LEVEL=l1 bash corl/scripts/sft_janus_levels.sh      # local / interactive
#   sbatch corl/scripts/sft_janus_levels.sbatch         # Isambard (SLURM wrapper)
#
# Everything is env-overridable (LEVEL, NPROC, PER_DEVICE_BS, MAX_STEPS, LR, ...).
set -euo pipefail

LEVEL=${LEVEL:-l1}
case "$LEVEL" in
    l1_meta|l1|l2|l3) ;;
    *) echo "LEVEL must be one of: l1_meta l1 l2 l3   (got '$LEVEL')"; exit 1 ;;
esac

# ---- Host-aware paths / conda / sensible per-host GPU + batch defaults ----
HOSTNAME_SHORT=$(hostname -s)
case "$HOSTNAME_SHORT" in
    cvssp-retina03)
        source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
        REPO=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
        DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision
        NPROC_DEFAULT=8;  BS_DEFAULT=8 ;;      # 8x RTX 3090 24GB
    ulws072)
        source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
        REPO=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
        DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision
        NPROC_DEFAULT=4;  BS_DEFAULT=8 ;;
    *)  # Isambard (u6gd project) -- NVIDIA GH200, 96GB HBM3 per GPU
        source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
        [ -f /projects/u6gd/umar/env.sh ] && source /projects/u6gd/umar/env.sh
        REPO=/projects/u6gd/umar/codes/ULM-R1
        DATA_DIR=/projects/u6gd/datasets/PubMedVision
        NPROC_DEFAULT=4;  BS_DEFAULT=16 ;;     # 96GB lets us go well above 24GB cards
esac
conda activate "${CONDA_ENV:-corl}"

# ---- Config (override any via env) ----
CKPT_PATH=${CKPT_PATH:-deepseek-ai/Janus-Pro-1B}
DATASET_NAME=${DATASET_NAME:-PubMedVision_CachedCaptions_Levels.json}
NPROC=${NPROC:-$NPROC_DEFAULT}
PER_DEVICE_BS=${PER_DEVICE_BS:-$BS_DEFAULT}
GRAD_ACC=${GRAD_ACC:-1}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-1e-4}
SAVE_STEPS=${SAVE_STEPS:-1000}
REPORT_TO=${REPORT_TO:-none}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

SAVE_DIR=${SAVE_DIR:-$REPO/results/JanusPro-1B-Levels}
SAVE_PATH=${SAVE_PATH:-$SAVE_DIR/level_${LEVEL}}
mkdir -p "$SAVE_PATH"
cp "$0" "$SAVE_PATH/run.sh"

echo "[sft_janus_levels] host=$HOSTNAME_SHORT LEVEL=$LEVEL NPROC=$NPROC bs=$PER_DEVICE_BS -> $SAVE_PATH"

torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-12345}" \
    corl/open_r1/sft_janus_alignment.py \
    --model_name_or_path "${CKPT_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --data_dir "${DATA_DIR}" \
    --caption_level "${LEVEL}" \
    --exclude_ids_json corl/eval/test_split.json \
    --task_format t2i \
    --lazy_image_loading true \
    --max_prompt_length 1024 \
    --max_completion_length 576 \
    --prompt_dropout_prob 0.1 \
    --use_perceptual_loss true \
    --perceptual_weight 0.5 \
    --perceptual_layers "3,6,9" \
    --perceptual_warmup_steps 2000 \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --use_peft true --lora_r 32 --lora_alpha 64 --lora_dropout 0.0 \
    --save_steps "${SAVE_STEPS}" --save_total_limit 2 --save_only_model true \
    --remove_unused_columns false \
    --logging_steps 5 \
    --output_dir "${SAVE_PATH}" \
    --report_to "${REPORT_TO}"
