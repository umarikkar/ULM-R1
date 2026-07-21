#!/bin/bash
# Continue each per-level baseline for 2 MORE epochs (total = 3 epochs), warm-
# started from its already-trained 1-epoch adapter. Produces the per-level
# 3-epoch baselines to compare against the future l1 -> l2 -> l3 curriculum.
#
# warm_start loads LoRA+gen_head+gen_aligner but does NOT restore the optimizer
# or scheduler, so these 2 epochs run with a FRESH optimizer + FRESH schedule
# that we control (see scheduler note below).
#
# ---- Scheduler (the important bit) ----
# Epoch 1 already ran a 1-epoch cosine 1e-4 -> ~0, so its optimizer/scheduler
# state is "spent". For the 2-epoch continuation we use:
#     cosine, peak LR 5e-5, warmup_ratio 0.03, decayed to 0 over the 2 epochs.
# Rationale: (a) warm-restarting to ~half the original peak refines rather than
# re-learns the already-adapted LoRA; (b) 5e-5 is roughly the LR a single
# 3-epoch cosine would pass through after epoch 1, so it approximates the tail
# of an "ideal" 3-epoch schedule; (c) the short warmup smooths the jump up from
# epoch-1's terminal LR~=0.
# IMPORTANT: reuse this SAME schedule (cosine / 5e-5 / warmup 0.03 / 2 epochs)
# for the curriculum's epochs 2-3, so baseline-vs-curriculum stays apples-to-
# apples (only the caption level per epoch differs).
#
# Usage (run when the GPUs are free):
#   bash corl/scripts/sft_janus_levels_continue.sh
# Overrides:  LR=7.5e-5 LEVELS="l1 l2 l3" PER_DEVICE_BS=4 bash ...
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO=$(realpath "${SCRIPT_DIR}/../..")

LEVELS=${LEVELS:-"l1 l2 l3"}
SRC_DIR=${SRC_DIR:-$REPO/results/JanusPro-1B-Levels}       # 1-epoch adapters (in)
DST_DIR=${DST_DIR:-$REPO/results/JanusPro-1B-Levels-3ep}   # 3-epoch adapters (out)

# Continuation hyperparameters (exported so the inner launcher picks them up).
export NPROC=${NPROC:-8}
export PER_DEVICE_BS=${PER_DEVICE_BS:-4}
export NUM_EPOCHS=${NUM_EPOCHS:-2}        # 2 MORE epochs, on top of the 1 already done
export LR=${LR:-5e-5}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
# perceptual/eval-image already default OFF in the launcher.

for L in $LEVELS; do
    SRC="$SRC_DIR/level_$L"
    if [ ! -f "$SRC/adapter_model.safetensors" ]; then
        echo "!! missing 1-epoch adapter for $L at $SRC -- skipping"
        continue
    fi
    echo "========== CONTINUE LEVEL $L  (warm-start: $SRC)  $(date) =========="
    LEVEL="$L" \
    WARM_START_CKPT="$SRC" \
    SAVE_PATH="$DST_DIR/level_$L" \
    bash "$REPO/corl/scripts/sft_janus_levels.sh" || { echo "LEVEL $L FAILED $(date)"; break; }
    echo "========== DONE LEVEL $L  $(date) =========="
done
echo "CONTINUE-CHAIN COMPLETE $(date)"
