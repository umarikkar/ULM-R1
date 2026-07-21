#!/bin/bash
# Stage-1 RL refinement of the gran-LoRA granularity knob (GRPO, length-target reward).
# Fresh rank-8 LoRA on the Janus captioner. Per step: sample alpha~U(0,1), sample K
# captions at scale=alpha, reward = -|log len - log_target(alpha)| - w_rep*rep_frac,
# GRPO advantage (mean baseline), PG + beta*KL-to-base (KL uses the model at scale 0).
# No text targets; faithfulness held by the KL anchor. See HANDOFF_GRANULARITY_RL.md.
set -eo pipefail

HOSTNAME_SHORT=$(hostname -s)
case "$HOSTNAME_SHORT" in
    cvssp-retina03)
        source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
        REPO=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
        DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision
        NPROC_DEFAULT=8 ;;
    ulws072)
        source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
        REPO=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
        DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision
        NPROC_DEFAULT=4 ;;
    *)  # Isambard (GH200 96GB)
        source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
        [ -f /projects/u6gd/umar/env.sh ] && source /projects/u6gd/umar/env.sh
        REPO=/projects/u6gd/umar/codes/ULM-R1
        DATA_DIR=/projects/u6gd/datasets/PubMedVision
        NPROC_DEFAULT=4 ;;
esac
conda activate "${CONDA_ENV:-corl}"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-deepseek-ai/Janus-Pro-1B}
DATASET=${DATASET:-PubMedVision_CachedCaptions_Levels.json}
NPROC=${NPROC:-$NPROC_DEFAULT}
# rollout: K samples per image, grad-accum over IMAGES_PER_STEP images per GPU.
GROUP_SIZE=${GROUP_SIZE:-4}
IMAGES_PER_STEP=${IMAGES_PER_STEP:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-200}
TEMPERATURE=${TEMPERATURE:-0.7}   # 1.0 degenerates the 1B base into gibberish
BETA_KL=${BETA_KL:-0.04}
W_REP=${W_REP:-1.0}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAVE_STEPS=${SAVE_STEPS:-500}
LORA_R=${LORA_R:-8}
OUT_DIR=${OUT_DIR:-$REPO/results/GranLoRA/gran_lora_rl_v1}

EXTRA=""
[ -n "${MAX_SAMPLES}" ] && EXTRA="${EXTRA} --max_samples ${MAX_SAMPLES}"
[ -n "${GRAD_CKPT}" ] && EXTRA="${EXTRA} --gradient_checkpointing"

mkdir -p "$OUT_DIR"; cp "$0" "$OUT_DIR/run.sh"
echo "[rl_gran_lora] host=$HOSTNAME_SHORT NPROC=$NPROC K=$GROUP_SIZE beta=$BETA_KL -> $OUT_DIR"

torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-12349}" \
    corl/open_r1/rl_gran_lora.py \
    --model "${MODEL}" \
    --data_json "${DATA_DIR}/${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --exclude_ids_json corl/eval/test_split.json \
    --lora_r ${LORA_R} \
    --group_size ${GROUP_SIZE} \
    --images_per_step ${IMAGES_PER_STEP} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --beta_kl ${BETA_KL} \
    --w_rep ${W_REP} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --out_dir "${OUT_DIR}" \
    ${EXTRA}
