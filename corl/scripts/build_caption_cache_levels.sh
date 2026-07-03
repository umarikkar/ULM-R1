#!/bin/bash
# Build a per-image granularity-leveled caption cache (l1/l2/l3) for SFT.
#
# Two phases:
#   1. Distributed generation -> per-rank shards in $OUT_DIR
#   2. Merge shards into the final JSON the trainer reads
#
# Resumable: rerunning the same OUT_DIR skips ids whose 3 levels are all present.
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

HOSTNAME_SHORT=$(hostname -s)
case "$HOSTNAME_SHORT" in
    cvssp-retina03) DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision ;;
    ulws072)        DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision ;;
    *)              DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

MODEL=${MODEL:-deepseek-ai/Janus-Pro-1B}
DATA_JSON=${DATA_JSON:-${DATA_DIR}/PubMedVision_Original_Caption.json}
OUT_DIR=${OUT_DIR:-${DATA_DIR}/cached_captions_levels}
MERGED_OUT=${MERGED_OUT:-${DATA_DIR}/PubMedVision_CachedCaptions_Levels.json}

BATCH_SIZE=${BATCH_SIZE:-64}
NPROC=${NPROC:-8}
MAX_SAMPLES=${MAX_SAMPLES:-}  # set to a small int for a smoke test

mkdir -p "${OUT_DIR}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

EXTRA_ARGS=""
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_ARGS="--max_samples ${MAX_SAMPLES}"
fi

# Phase 1: distributed generation (l1/l2/l3 per image, greedy, k=1)
torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12347 \
    corl/scripts/build_caption_cache_levels.py \
    --model "${MODEL}" \
    --data_json "${DATA_JSON}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    ${EXTRA_ARGS}

# Phase 2: merge shards into a single JSON
python corl/scripts/build_caption_cache_levels.py \
    --merge \
    --out_dir "${OUT_DIR}" \
    --merged_out "${MERGED_OUT}"

echo ""
echo "Done. To train on a granularity level, set:"
echo "  caption_source=original"
echo "  caption_column=cached_captions_l1   # or l2 / l3"
echo "  --dataset_name $(basename ${MERGED_OUT})"
