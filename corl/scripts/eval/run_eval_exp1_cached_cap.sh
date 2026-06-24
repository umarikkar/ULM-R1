#!/bin/bash
# Cross-distribution check: evaluate exp1 (trained on GPT captions) under the
# self-distilled caption distribution. Completes the 2x2 caption-OOD design:
#   - exp1 / GPT (results/eval_small/exp1_pubmed_captions)
#   - exp1 / cached  <-- this script
#   - exp2 / GPT (results/eval_small/exp2_cached_captions)
#   - exp2 / cached  (run_eval_cached_cap.sh)

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

name=exp1_pubmed_captions
adapter=${CKPT_ROOT}/exp1_pubmed_captions/checkpoint-24244
out_dir=${OUT_ROOT}/${name}
mkdir -p "${out_dir}"

echo "[eval-cap-x] === ${name} (caption_field=cached_captions) ==="
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
    echo "[eval-cap-x]   launched shard ${s} on GPU ${gpu} (pid ${pids[-1]})"
done
for pid in "${pids[@]}"; do wait "${pid}"; done
echo "[eval-cap-x] generation done for ${name}; computing metrics"

CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} \
/vol/research/fmodel_medical/people/umar/miniconda3/envs/corl/bin/python \
    corl/eval/compute_metrics.py \
    --manifest_glob "${out_dir}/manifest_shard*.json" \
    --data_dir "${DATA_DIR}" \
    --caption_field cached_captions \
    --out "${out_dir}/metrics.json" \
    > "${LOG_ROOT}/${name}_metrics.log" 2>&1
echo "[eval-cap-x] metrics -> ${out_dir}/metrics.json"
echo "[eval-cap-x] DONE"
