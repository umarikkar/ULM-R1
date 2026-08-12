#!/bin/bash
# Stage 1 of the random-alpha distillation pipeline: caption every training image
# with the frozen gran-LoRA at a random alpha (no grad), sharded across GPUs, then
# merge into one dataset JSON with a `cached_captions_randalpha` column that the
# T2I trainer consumes (see distill_gran_alpha_captions.py). Stage 2 = train:
#   CAPTION_COLUMN=cached_captions_randalpha DATASET_NAME=<out> NUM_EPOCHS=1 \
#       bash corl/scripts/sft_janus_levels.sh
set -eo pipefail

HOSTNAME_SHORT=$(hostname -s)
case "$HOSTNAME_SHORT" in
    cvssp-retina03)
        source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
        REPO=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
        DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision ;;
    ulws072)
        source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
        REPO=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1
        DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision ;;
    *)  source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
        [ -f /projects/u6gd/umar/env.sh ] && source /projects/u6gd/umar/env.sh
        REPO=/projects/u6gd/umar/codes/ULM-R1
        DATA_DIR=/projects/u6gd/datasets/PubMedVision ;;
esac
conda activate "${CONDA_ENV:-corl}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ADAPTER=${ADAPTER:-results/GranLoRA/gran_lora_v2}
SRC=${SRC:-PubMedVision_CachedCaptions_Levels.json}
OUT_NAME=${OUT_NAME:-PubMedVision_RandAlpha.json}
OUT="$DATA_DIR/$OUT_NAME"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}           # GPU ids to shard across
EXCLUDE=${EXCLUDE:-corl/eval/test_split.json}
MAX_SAMPLES=${MAX_SAMPLES:-}            # int for a subset / smoke test
ALPHA_LO=${ALPHA_LO:-0.3}; ALPHA_HI=${ALPHA_HI:-0.9}
ALPHAS_DISCRETE=${ALPHAS_DISCRETE:-}   # e.g. "0.3,0.6,0.9" to draw discrete anchors
GEN_BATCH=${GEN_BATCH:-8}
DO_SAMPLE=${DO_SAMPLE:-}               # set to 1 for temperature sampling (caption variety)

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NSHARDS=${#GPU_ARR[@]}
EXTRA=""
[ -n "$MAX_SAMPLES" ] && EXTRA="$EXTRA --max_samples $MAX_SAMPLES"
[ -n "$EXCLUDE" ] && EXTRA="$EXTRA --exclude_ids_json $EXCLUDE"
[ -n "$ALPHAS_DISCRETE" ] && EXTRA="$EXTRA --alphas_discrete $ALPHAS_DISCRETE"
[ -n "$DO_SAMPLE" ] && EXTRA="$EXTRA --do_sample"

echo "[distill] host=$HOSTNAME_SHORT shards=$NSHARDS gpus=$GPUS -> $OUT  (alpha $ALPHA_LO..$ALPHA_HI ${ALPHAS_DISCRETE:+discrete=$ALPHAS_DISCRETE})"
mkdir -p logs
pids=()
for i in "${!GPU_ARR[@]}"; do
    g="${GPU_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="$g" python corl/eval/distill_gran_alpha_captions.py \
        --adapter_dir "$ADAPTER" \
        --data_json "$DATA_DIR/$SRC" --data_dir "$DATA_DIR" \
        --out_json "$OUT" \
        --alpha_lo "$ALPHA_LO" --alpha_hi "$ALPHA_HI" \
        --gen_batch "$GEN_BATCH" \
        --num_shards "$NSHARDS" --shard_id "$i" --seed 0 \
        $EXTRA > "logs/distill_shard${i}.log" 2>&1 &
    pids+=($!)
done
echo "[distill] launched ${#pids[@]} shards; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 1 ] && { echo "[distill] a shard FAILED (see logs/distill_shard*.log)"; exit 1; }

# merge shards (single-shard writes OUT directly; nothing to merge)
if [ "$NSHARDS" -gt 1 ]; then
python - "$OUT" "$NSHARDS" <<'PY'
import json, os, sys
out, ns = sys.argv[1], int(sys.argv[2])
rows = []
for i in range(ns):
    f = out.replace(".json", f".shard{i}.json")
    rows += json.load(open(f))
json.dump(rows, open(out, "w"), ensure_ascii=False)
for i in range(ns):
    os.remove(out.replace(".json", f".shard{i}.json"))
print(f"[distill] merged {ns} shards -> {out}  ({len(rows)} rows)")
PY
else
    echo "[distill] single shard -> $OUT"
fi
echo "[distill] DONE -> $OUT"
echo "Stage 2:  CAPTION_COLUMN=cached_captions_randalpha DATASET_NAME=$OUT_NAME NUM_EPOCHS=1 bash corl/scripts/sft_janus_levels.sh"
