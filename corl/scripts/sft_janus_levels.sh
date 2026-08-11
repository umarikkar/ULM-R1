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
# NB: no `set -u` -- conda's compiler-env (de)activation scripts reference unbound
# vars (CONDA_BACKUP_CXX) and would abort under nounset. All vars are :- guarded.
set -eo pipefail

# Caption selection: either a single fixed LEVEL, or (if RANDOM_LEVELS is set)
# per-step random draw over a comma-separated set of levels.
LEVEL=${LEVEL:-l1}
RANDOM_LEVELS=${RANDOM_LEVELS:-}
EPOCH_SCHEDULE=${EPOCH_SCHEDULE:-}      # curriculum: per-epoch levels e.g. "l2,l3"
CAPTION_COLUMN=${CAPTION_COLUMN:-}      # arbitrary cached column, e.g. distilled random-alpha captions
if [ -n "${CAPTION_COLUMN}" ]; then
    CAP_ARG="--caption_source original --caption_column ${CAPTION_COLUMN}"
    TAG="col_${CAPTION_COLUMN}"
elif [ -n "${RANDOM_LEVELS}" ]; then
    CAP_ARG="--caption_random_levels ${RANDOM_LEVELS}"
    TAG="random_${RANDOM_LEVELS//,/-}"
elif [ -n "${EPOCH_SCHEDULE}" ]; then
    CAP_ARG="--caption_epoch_schedule ${EPOCH_SCHEDULE}"
    TAG="curriculum_${EPOCH_SCHEDULE//,/-}"
else
    case "$LEVEL" in
        l1_meta|l1|l2|l3) ;;
        *) echo "LEVEL must be one of: l1_meta l1 l2 l3   (got '$LEVEL')"; exit 1 ;;
    esac
    CAP_ARG="--caption_level ${LEVEL}"
    TAG="level_${LEVEL}"
fi

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
MAX_SAMPLES=${MAX_SAMPLES:-}      # set to a small int for a smoke test
LR=${LR:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SAVE_STEPS=${SAVE_STEPS:-1000}
# save_only_model=true saves just the adapter (small) but is NOT resumable
# (no optimizer/scheduler). Set false to make checkpoints fully resumable.
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}
REPORT_TO=${REPORT_TO:-none}
# Warm-start: dir with adapter_model.safetensors (e.g. a prior 1-epoch level
# checkpoint). Loads LoRA+gen_head+gen_aligner; optimizer/scheduler start fresh.
WARM_START_CKPT=${WARM_START_CKPT:-}
# Resume: a checkpoint-NNNN dir to CONTINUE (restores optimizer/scheduler/step).
# Use to finish an interrupted run; keep NPROC/PER_DEVICE_BS/GRAD_ACC identical
# to the original so the global-step math matches.
RESUME_FROM=${RESUME_FROM:-}

# --- Losses / eval (lean baseline: everything extra OFF by default) ---
# Perceptual loss adds a BiomedCLIP forward every step (memory + compute); off
# isolates the caption-granularity effect. Set USE_PERCEPTUAL=true to enable.
USE_PERCEPTUAL=${USE_PERCEPTUAL:-false}
# In-training eval-image generation AR-decodes full images and spikes VRAM
# (OOMs on 24GB cards shared with others). Default effectively off -- evaluate
# checkpoints afterwards with corl/eval/generate.py. On a big card set e.g.
# EVAL_IMAGE_FREQ=500 for in-training previews.
EVAL_IMAGE_FREQ=${EVAL_IMAGE_FREQ:-100000000}
EVAL_IMAGE_NUM=${EVAL_IMAGE_NUM:-4}

EXTRA_ARGS=""
if [ -n "${MAX_SAMPLES}" ]; then EXTRA_ARGS="${EXTRA_ARGS} --max_samples ${MAX_SAMPLES}"; fi
if [ -n "${WARM_START_CKPT}" ]; then EXTRA_ARGS="${EXTRA_ARGS} --warm_start_checkpoint ${WARM_START_CKPT}"; fi
if [ -n "${RESUME_FROM}" ]; then EXTRA_ARGS="${EXTRA_ARGS} --resume_from_checkpoint ${RESUME_FROM} --ignore_data_skip true"; fi

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

SAVE_DIR=${SAVE_DIR:-$REPO/results/JanusPro-1B-Levels}
SAVE_PATH=${SAVE_PATH:-$SAVE_DIR/${TAG}}
mkdir -p "$SAVE_PATH"
cp "$0" "$SAVE_PATH/run.sh"

echo "[sft_janus_levels] host=$HOSTNAME_SHORT caption=${TAG} NPROC=$NPROC bs=$PER_DEVICE_BS -> $SAVE_PATH"

torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-12345}" \
    corl/open_r1/sft_janus_alignment.py \
    --model_name_or_path "${CKPT_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --data_dir "${DATA_DIR}" \
    ${CAP_ARG} \
    --exclude_ids_json corl/eval/test_split.json \
    --task_format t2i \
    --lazy_image_loading true \
    --max_prompt_length 1024 \
    --max_completion_length 576 \
    --prompt_dropout_prob 0.1 \
    --use_perceptual_loss ${USE_PERCEPTUAL} \
    --perceptual_weight 0.5 \
    --perceptual_layers "3,6,9" \
    --perceptual_warmup_steps 2000 \
    --eval_image_freq ${EVAL_IMAGE_FREQ} \
    --eval_image_num ${EVAL_IMAGE_NUM} \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type cosine \
    --bf16 --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --use_peft true --lora_r 32 --lora_alpha 64 --lora_dropout 0.0 \
    --save_steps "${SAVE_STEPS}" --save_total_limit 2 --save_only_model "${SAVE_ONLY_MODEL}" \
    --remove_unused_columns false \
    --logging_steps 5 \
    --output_dir "${SAVE_PATH}" \
    --report_to "${REPORT_TO}" \
    ${EXTRA_ARGS}
