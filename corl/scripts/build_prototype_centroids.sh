#!/bin/bash
# Build BiomedCLIP-feature prototype centroids for unsupervised conditioning.
#
# Phase 1 (features): DDP forward of a sample of training images -> per-rank shards
# Phase 2 (cluster) : single-process KMeans on merged feats -> centroids .pt
#
# Resumable: rerunning the same OUT_DIR skips ids already present in shards.
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

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

CAPTIONS=${CAPTIONS:-${DATA_DIR}/PubMedVision_CachedCaptions_K4.json}
OUT_DIR=${OUT_DIR:-data/prototype_shards}
OUT=${OUT:-data/prototype_centroids.pt}
K=${K:-16}
MAX_SAMPLES=${MAX_SAMPLES:-50000}
BATCH_SIZE=${BATCH_SIZE:-128}
NPROC=${NPROC:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}
PHASE=${PHASE:-all}    # features | cluster | all

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

run_features() {
    torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
        --master_addr=127.0.0.1 --master_port=12349 \
        corl/scripts/build_prototype_centroids.py --phase features \
        --captions "${CAPTIONS}" --data_dir "${DATA_DIR}" \
        --out_dir "${OUT_DIR}" --K ${K} \
        --max_samples ${MAX_SAMPLES} --batch_size ${BATCH_SIZE}
}
run_cluster() {
    python corl/scripts/build_prototype_centroids.py --phase cluster \
        --out_dir "${OUT_DIR}" --out "${OUT}" --K ${K}
}

case "${PHASE}" in
    features) run_features ;;
    cluster)  run_cluster ;;
    all)      run_features; run_cluster ;;
    *) echo "unknown PHASE=${PHASE}"; exit 1 ;;
esac
