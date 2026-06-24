#!/bin/bash
# Sequentially evaluate vanilla Janus + 4 fine-tuned experiments on the test
# split, sharded 4-ways across the EVAL_GPUS list. exp5 evaluation is launched
# separately because it's retraining; this script intentionally skips it.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")
cd "${PROJECT_ROOT}"

EVAL_GPUS=${EVAL_GPUS:-"4,5,6,7"}
IFS=',' read -ra GPU_LIST <<< "${EVAL_GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}

BASE_MODEL=${BASE_MODEL:-deepseek-ai/Janus-Pro-1B}
TEST_SPLIT=${TEST_SPLIT:-corl/eval/test_split.json}
DATA_DIR=${DATA_DIR:-/work/um00109/MLLM/datasets/PubMedVision}
CKPT_ROOT=${CKPT_ROOT:-results/JanusPro-1B-T2I-Stage2-LoRA}
OUT_ROOT=${OUT_ROOT:-results/eval}
LOG_ROOT=${LOG_ROOT:-logs/eval}
mkdir -p "${LOG_ROOT}"

# name : adapter_dir (empty = vanilla)
EXPERIMENTS=(
    "vanilla::"
    "exp1_pubmed_captions:${CKPT_ROOT}/exp1_pubmed_captions/checkpoint-24244"
    "exp2_cached_captions:${CKPT_ROOT}/exp2_cached_captions/checkpoint-24244"
    "exp3_cached_lpips:${CKPT_ROOT}/exp3_cached_lpips/checkpoint-24244"
    "exp4_cached_biomedclip:${CKPT_ROOT}/exp4_cached_biomedclip/checkpoint-24244"
)

for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%:*}"
    adapter="${entry#*:}"
    adapter="${adapter#:}"
    out_dir="${OUT_ROOT}/${name}"
    mkdir -p "${out_dir}"
    echo "[eval-all] === ${name} (adapter=${adapter:-vanilla}) ==="

    pids=()
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        gpu=${GPU_LIST[$s]}
        log="${LOG_ROOT}/${name}_shard${s}.log"
        adapter_flag=""
        if [ -n "${adapter}" ]; then adapter_flag="--adapter_dir ${adapter}"; fi
        CUDA_VISIBLE_DEVICES=${gpu} SHARD=${s} NUM_SHARDS=${NUM_SHARDS} \
        /vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
            corl/eval/generate.py \
            --base_model "${BASE_MODEL}" \
            ${adapter_flag} \
            --test_split "${TEST_SPLIT}" \
            --data_dir "${DATA_DIR}" \
            --out_dir "${out_dir}" \
            > "${log}" 2>&1 &
        pids+=($!)
        echo "[eval-all]   launched shard ${s} on GPU ${gpu} (pid ${pids[-1]}) -> ${log}"
    done
    for pid in "${pids[@]}"; do wait "${pid}"; done
    echo "[eval-all] generation done for ${name}; computing metrics"

    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} \
    /vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
        corl/eval/compute_metrics.py \
        --manifest_glob "${out_dir}/manifest_shard*.json" \
        --data_dir "${DATA_DIR}" \
        --out "${out_dir}/metrics.json" \
        > "${LOG_ROOT}/${name}_metrics.log" 2>&1
    echo "[eval-all] metrics -> ${out_dir}/metrics.json"
done

echo "[eval-all] ALL DONE"
