#!/bin/bash
# Full-grid generation + local metrics for the Janus caption-level adapters.
# 8 unique (adapter, caption_field) runs; each sharded across 8 GPUs.
# Grid = {matched, fixed-Original, fixed-l1} x {l1,l2,l3 adapters}, deduped
# ((l1 adapter, l1 prompt) is shared between matched and fixed-l1).
set -eo pipefail

source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
conda activate corl
cd /vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE=deepseek-ai/Janus-Pro-1B
DATA=${DATA_DIR:-/work/um00109/MLLM/datasets/PubMedVision}
SPLIT=${SPLIT:-corl/eval/test_split_levels.json}
ROOT=${ROOT:-results/eval_levels}
NUM_SHARDS=${NUM_SHARDS:-8}

# "adapterLevel  captionField  outTag"
combos=(
  "l1 cached_captions_l1 l1__l1"
  "l2 cached_captions_l2 l2__l2"
  "l3 cached_captions_l3 l3__l3"
  "l1 Original_Caption   l1__orig"
  "l2 Original_Caption   l2__orig"
  "l3 Original_Caption   l3__orig"
  "l2 cached_captions_l1 l2__l1"
  "l3 cached_captions_l1 l3__l1"
)

for c in "${combos[@]}"; do
  set -- $c; ADP=$1; CF=$2; TAG=$3
  OUT="$ROOT/$TAG"
  echo "========== GEN $TAG (adapter=level_$ADP field=$CF) $(date) =========="
  pids=()
  for S in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$S SHARD=$S NUM_SHARDS=$NUM_SHARDS \
      python corl/eval/generate.py \
        --base_model "$BASE" \
        --adapter_dir "results/JanusPro-1B-Levels/level_$ADP" \
        --test_split "$SPLIT" --data_dir "$DATA" \
        --out_dir "$OUT" --caption_field "$CF" &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "========== METRICS $TAG $(date) =========="
  python corl/eval/compute_metrics.py \
    --manifest_glob "$OUT/manifest_shard*.json" \
    --data_dir "$DATA" --caption_field "$CF" \
    --out "$OUT/metrics.json" || echo "[warn] metrics failed for $TAG"
  echo "========== DONE $TAG $(date) =========="
done
echo "ALL GEN+METRICS DONE $(date)"
