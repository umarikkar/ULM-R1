#!/bin/bash
# Random-granularity arm: 3 epochs of Janus-Pro T2I alignment where EACH sample
# independently draws one of {l1,l2,l3} per step (trainer-side random over
# cached_captions_<lvl>). This is the "mix all granularities" comparison point
# against the fixed-level baselines and the l1->l2->l3 curriculum.
#
# Schedule is matched to the per-level baselines so the ONLY difference across
# arms is the caption policy:
#   Phase 1 (epoch 1)    : from base Janus, cosine LR 1e-4, warmup 0.03.
#   Phase 2 (epochs 2-3) : warm-start from phase-1, cosine LR 5e-5, warmup 0.03.
#   (baseline lX = same schedule, always level X; curriculum = e1 l1 / e2 l2 /
#    e3 l3; random = every step samples a level.)
#
# Usage (run when the GPUs are free):
#   bash corl/scripts/sft_janus_levels_random.sh
# Overrides:  RANDOM_LEVELS="l1,l2,l3" PER_DEVICE_BS=4 bash ...
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO=$(realpath "${SCRIPT_DIR}/../..")

export RANDOM_LEVELS=${RANDOM_LEVELS:-"l1,l2,l3"}
export NPROC=${NPROC:-8}
export PER_DEVICE_BS=${PER_DEVICE_BS:-4}

P1_DIR=${P1_DIR:-$REPO/results/JanusPro-1B-Random/phase1}          # epoch 1
P3_DIR=${P3_DIR:-$REPO/results/JanusPro-1B-Random-3ep/random}      # epochs 2-3 (final)

# ---------- Phase 1: epoch 1 from base Janus ----------
echo "========== RANDOM PHASE 1 (epoch 1, LR 1e-4) $(date) =========="
NUM_EPOCHS=1 LR=1e-4 WARMUP_RATIO=0.03 \
SAVE_PATH="$P1_DIR" \
bash "$REPO/corl/scripts/sft_janus_levels.sh"

# ---------- Phase 2: epochs 2-3, warm-started from phase 1 ----------
if [ ! -f "$P1_DIR/adapter_model.safetensors" ]; then
    echo "phase-1 adapter missing at $P1_DIR -- aborting"; exit 1
fi
echo "========== RANDOM PHASE 2 (epochs 2-3, LR 5e-5, warm-start) $(date) =========="
NUM_EPOCHS=2 LR=5e-5 WARMUP_RATIO=0.03 \
WARM_START_CKPT="$P1_DIR" \
SAVE_PATH="$P3_DIR" \
bash "$REPO/corl/scripts/sft_janus_levels.sh"

echo "RANDOM-ARM COMPLETE -> $P3_DIR  $(date)"
