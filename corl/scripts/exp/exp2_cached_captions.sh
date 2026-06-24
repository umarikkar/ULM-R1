#!/bin/bash
# Exp 2: SFT with self-distilled (cached) captions, no perceptual loss.
# 4 GPUs, 2 epochs. Self-distill baseline.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
export master_port=${master_port:-12352}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"exp2_cached_captions"}

export num_train_epochs=2
export caption_source=original
export caption_column=cached_captions

export use_perceptual_loss=false
export use_reconstruction_loss=false
export use_prototype_conditioning=false
export use_text_to_proto=false

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"