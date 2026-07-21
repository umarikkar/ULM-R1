#!/bin/bash
# Generate the 4 off-diagonal train x eval level combos missing from the grid,
# to complete the full 3x3 train-level x eval-level matrix.
#   l1->l2, l1->l3, l2->l3, l3->l2   (adapter -> eval prompt level)
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
  "l1 cached_captions_l2 l1__l2"
  "l1 cached_captions_l3 l1__l3"
  "l2 cached_captions_l3 l2__l3"
  "l3 cached_captions_l2 l3__l2"
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
echo "ALL CROSS DONE $(date)"
