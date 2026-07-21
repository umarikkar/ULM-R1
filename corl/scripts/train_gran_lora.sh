#!/bin/bash
# Stage-0 grounding of the gran-LoRA granularity knob (supervised, scale-consistency).
# Trains one LoRA on the Janus captioner so LoRA scale alpha controls granularity
# under a fixed neutral prompt. alpha=0 = base default caption.
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
# Long seqs (576 image tokens + caption) make full-vocab logits big; reduce
# fragmentation so batch 4 fits on 24GB cards.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-deepseek-ai/Janus-Pro-1B}
DATASET=${DATASET:-PubMedVision_CachedCaptions_Levels.json}
LEVELS=${LEVELS:-l1,l2,l3}
ALPHA_MAP=${ALPHA_MAP:-l1:1.0,l2:0.6,l3:0.3}   # higher alpha => coarser (base is detailed)
NPROC=${NPROC:-$NPROC_DEFAULT}
BATCH=${BATCH:-4}   # bs=8 OOMs on 24GB (long seq + full-vocab logits)
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAVE_STEPS=${SAVE_STEPS:-1000}
OUT_DIR=${OUT_DIR:-$REPO/results/GranLoRA/gran_lora_v1}

EXTRA=""
[ -n "${MAX_SAMPLES}" ] && EXTRA="${EXTRA} --max_samples ${MAX_SAMPLES}"

mkdir -p "$OUT_DIR"; cp "$0" "$OUT_DIR/run.sh"
echo "[train_gran_lora] host=$HOSTNAME_SHORT NPROC=$NPROC bs=$BATCH alpha_map=$ALPHA_MAP -> $OUT_DIR"

torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-12348}" \
    corl/open_r1/train_gran_lora.py \
    --model "${MODEL}" \
    --data_json "${DATA_DIR}/${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --levels "${LEVELS}" \
    --alpha_map "${ALPHA_MAP}" \
    --exclude_ids_json corl/eval/test_split.json \
    --batch_size ${BATCH} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --out_dir "${OUT_DIR}" \
    ${EXTRA}
