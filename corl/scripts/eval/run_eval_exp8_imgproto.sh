#!/bin/bash
# Image-side (oracle) prototype eval for exp8. Encodes the GT image with
# BiomedCLIP, soft-assigns to spherical-K16 centroids, uses that as the
# prototype cond. Upper bound on what the prototype path could give if the
# text->proto head were perfect.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")
cd "${PROJECT_ROOT}"

NAME=${NAME:-exp8_spherical_proto_joint}
EVAL_GPUS=${EVAL_GPUS:-"4,5,6,7"}
CENTROIDS=${CENTROIDS:-data/prototype_variants/v03_spherical_k16.pt}
IFS=',' read -ra GPU_LIST <<< "${EVAL_GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}

BASE_MODEL=deepseek-ai/Janus-Pro-1B
TEST_SPLIT=corl/eval/test_split_small.json
DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision
CKPT=results/JanusPro-1B-T2I-Stage2-LoRA/${NAME}/checkpoint-12122
OUT_ROOT=results/eval_small_cached_cap_imgproto
LOG_ROOT=logs/eval_small_cached_cap_imgproto
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

out_dir=${OUT_ROOT}/${NAME}
mkdir -p "${out_dir}"

echo "[eval-img] === ${NAME} (proto_source=image, centroids=${CENTROIDS}, gpus=${EVAL_GPUS}) ==="
pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=${GPU_LIST[$s]}
    log="${LOG_ROOT}/${NAME}_shard${s}.log"
    CUDA_VISIBLE_DEVICES=${gpu} SHARD=${s} NUM_SHARDS=${NUM_SHARDS} \
    /vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
        corl/eval/generate.py \
        --base_model "${BASE_MODEL}" \
        --adapter_dir "${CKPT}" \
        --test_split "${TEST_SPLIT}" \
        --data_dir "${DATA_DIR}" \
        --out_dir "${out_dir}" \
        --caption_field cached_captions \
        --proto_source image \
        --prototype_centroids_path "${CENTROIDS}" \
        --cond_temperature 0.1 \
        > "${log}" 2>&1 &
    pids+=($!)
    echo "[eval-img]   launched shard ${s} on GPU ${gpu} (pid ${pids[-1]})"
done
for pid in "${pids[@]}"; do wait "${pid}"; done
echo "[eval-img] generation done for ${NAME}; computing metrics"

CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} \
/vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
    corl/eval/compute_metrics.py \
    --manifest_glob "${out_dir}/manifest_shard*.json" \
    --data_dir "${DATA_DIR}" \
    --caption_field cached_captions \
    --out "${out_dir}/metrics.json" \
    > "${LOG_ROOT}/${NAME}_metrics.log" 2>&1
echo "[eval-img] metrics -> ${out_dir}/metrics.json"
echo "[eval-img] DONE for ${NAME}"
