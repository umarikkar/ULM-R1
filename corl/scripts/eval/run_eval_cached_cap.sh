#!/bin/bash
# Re-evaluate exp2/3/4 conditioning on the cached self-distill captions
# (instead of Original_Caption) to test whether the gap to exp1 is a
# caption-distribution OOD artifact.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")
cd "${PROJECT_ROOT}"

EVAL_GPUS=${EVAL_GPUS:-"4,5,6,7"}
IFS=',' read -ra GPU_LIST <<< "${EVAL_GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}

BASE_MODEL=deepseek-ai/Janus-Pro-1B
TEST_SPLIT=corl/eval/test_split_small.json
DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision
CKPT_ROOT=results/JanusPro-1B-T2I-Stage2-LoRA
OUT_ROOT=results/eval_small_cached_cap
LOG_ROOT=logs/eval_small_cached_cap
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

EXPERIMENTS=(
    "exp2_cached_captions:${CKPT_ROOT}/exp2_cached_captions/checkpoint-24244"
    "exp3_cached_lpips:${CKPT_ROOT}/exp3_cached_lpips/checkpoint-24244"
    "exp4_cached_biomedclip:${CKPT_ROOT}/exp4_cached_biomedclip/checkpoint-24244"
)

for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%:*}"
    adapter="${entry#*:}"
    out_dir="${OUT_ROOT}/${name}"
    mkdir -p "${out_dir}"
    echo "[eval-cap] === ${name} (caption_field=cached_captions) ==="

    pids=()
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        gpu=${GPU_LIST[$s]}
        log="${LOG_ROOT}/${name}_shard${s}.log"
        CUDA_VISIBLE_DEVICES=${gpu} SHARD=${s} NUM_SHARDS=${NUM_SHARDS} \
        /vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
            corl/eval/generate.py \
            --base_model "${BASE_MODEL}" \
            --adapter_dir "${adapter}" \
            --test_split "${TEST_SPLIT}" \
            --data_dir "${DATA_DIR}" \
            --out_dir "${out_dir}" \
            --caption_field cached_captions \
            > "${log}" 2>&1 &
        pids+=($!)
        echo "[eval-cap]   launched shard ${s} on GPU ${gpu} (pid ${pids[-1]}) -> ${log}"
    done
    for pid in "${pids[@]}"; do wait "${pid}"; done
    echo "[eval-cap] generation done for ${name}; computing metrics"

    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} \
    /vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
        corl/eval/compute_metrics.py \
        --manifest_glob "${out_dir}/manifest_shard*.json" \
        --data_dir "${DATA_DIR}" \
        --caption_field cached_captions \
        --out "${out_dir}/metrics.json" \
        > "${LOG_ROOT}/${name}_metrics.log" 2>&1
    echo "[eval-cap] metrics -> ${out_dir}/metrics.json"
done

echo "[eval-cap] ALL DONE"
