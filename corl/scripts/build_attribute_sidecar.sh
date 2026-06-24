#!/bin/bash
# Build the per-image attribute sidecar {modality, pose, is_grid}.
#
# Phase 1 (text) : regex over cached captions          -> ${TEXT_OUT}  (CPU, fast)
# Phase 2 (grid) : projection-profile layout detection -> rewrites ${TEXT_OUT} (CPU mp)
# Phase 3 (vlm)  : DDP Janus labeling of modality/pose -> per-rank shards in ${OUT_DIR}
# Phase 4 (merge): overlay shards onto text rows       -> ${MERGED_OUT}
#
# Resumable: rerunning the same OUT_DIR skips ids already present.
set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

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
CAPTIONS=${CAPTIONS:-${DATA_DIR}/PubMedVision_CachedCaptions_K4.json}
TEXT_OUT=${TEXT_OUT:-data/attribute_sidecar.text.jsonl}
OUT_DIR=${OUT_DIR:-data/attribute_sidecar_shards}
MERGED_OUT=${MERGED_OUT:-data/attribute_sidecar.json}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-40}
NPROC=${NPROC:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}
NUM_WORKERS=${NUM_WORKERS:-0}   # grid procs (0 = os.cpu_count())
LIMIT=${LIMIT:-0}          # per-rank cap for a trial run (0 = all)
PHASE=${PHASE:-all}        # text | grid | vlm | merge | all

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

run_text() {
    python corl/scripts/build_attribute_sidecar.py --phase text \
        --captions "${CAPTIONS}" --data_dir "${DATA_DIR}" --text_out "${TEXT_OUT}"
}
run_grid() {
    python corl/scripts/build_attribute_sidecar.py --phase grid \
        --data_dir "${DATA_DIR}" --text_out "${TEXT_OUT}" --num_workers ${NUM_WORKERS}
}
run_vlm() {
    torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
        --master_addr=127.0.0.1 --master_port=12348 \
        corl/scripts/build_attribute_sidecar.py --phase vlm \
        --model "${MODEL}" --data_dir "${DATA_DIR}" \
        --text_out "${TEXT_OUT}" --out_dir "${OUT_DIR}" \
        --batch_size ${BATCH_SIZE} --max_new_tokens ${MAX_NEW_TOKENS} --limit ${LIMIT}
}
run_merge() {
    python corl/scripts/build_attribute_sidecar.py --phase merge \
        --text_out "${TEXT_OUT}" --out_dir "${OUT_DIR}" --merged_out "${MERGED_OUT}"
}

case "${PHASE}" in
    text)  run_text ;;
    grid)  run_grid ;;
    vlm)   run_vlm ;;
    merge) run_merge ;;
    all)   run_text; run_grid; run_vlm; run_merge ;;
    *) echo "unknown PHASE=${PHASE}"; exit 1 ;;
esac
