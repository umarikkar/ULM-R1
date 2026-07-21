#!/bin/bash
# Granularity CURRICULUM (option A): coarse -> fine, one level per epoch.
#   epoch 1 = l1  (REUSE the existing 1-epoch results/JanusPro-1B-Levels/level_l1)
#   epoch 2 = l2  }  a 2-epoch continuation warm-started from that l1 adapter,
#   epoch 3 = l3  }  with --caption_epoch_schedule l2,l3.
#
# Schedule is IDENTICAL to the per-level 3-epoch baselines (warm-start, cosine
# LR 5e-5, warmup 0.03, effective batch 32) so baseline-vs-curriculum differs
# ONLY in the caption ordering across epochs.
#
# Memory: PER_DEVICE_BS=8 x 4 GPUs = effective batch 32 (same as baselines) and
# peaks ~22GB/24GB on L3 (epoch 3, the longest captions) -> fills the card so a
# co-tenant can't squeeze in and OOM us. GRAD_ACC=1 keeps effective batch at 32.
#
# Usage:  CUDA_VISIBLE_DEVICES=4,5,6,7 bash corl/scripts/sft_janus_curriculum.sh
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO=$(realpath "${SCRIPT_DIR}/../..")

export EPOCH_SCHEDULE=${EPOCH_SCHEDULE:-l2,l3}   # epochs 2 & 3 (epoch 1 = warm-start l1)
export NPROC=${NPROC:-4}
export PER_DEVICE_BS=${PER_DEVICE_BS:-8}          # 8*4=32 eff batch (== baselines); ~22GB/24GB on L3
export GRAD_ACC=${GRAD_ACC:-1}
export NUM_EPOCHS=${NUM_EPOCHS:-2}               # the 2-epoch continuation (l2 then l3)
export LR=${LR:-5e-5}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
export SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-false} # resumable if a co-tenant still OOMs us
export WARM_START_CKPT=${WARM_START_CKPT:-$REPO/results/JanusPro-1B-Levels/level_l1}
export SAVE_PATH=${SAVE_PATH:-$REPO/results/JanusPro-1B-Curriculum/l1_l2_l3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MASTER_PORT=${MASTER_PORT:-29610}

bash "$REPO/corl/scripts/sft_janus_levels.sh"
