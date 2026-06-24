#!/bin/bash
# Exp 3: SFT with cached captions + pixel-space LPIPS (VGG) reconstruction loss.
# 4 GPUs, 2 epochs.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export master_port=${master_port:-12353}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"exp3_cached_lpips"}

export num_train_epochs=2
export caption_source=original
export caption_column=cached_captions

export use_perceptual_loss=false
export use_reconstruction_loss=true
export lpips_weight=${lpips_weight:-1.0}
export use_prototype_conditioning=false
export use_text_to_proto=false

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"