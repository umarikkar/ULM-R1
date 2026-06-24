#!/bin/bash
# Exp 4: SFT with cached captions + BiomedCLIP perceptual loss, layers 3,6,8,12.
# 4 GPUs, 2 epochs.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
export master_port=${master_port:-12354}
export EXP_NAME=${EXP_NAME:-"exp4_cached_biomedclip"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${EXP_NAME}"}

export num_train_epochs=2
export caption_source=original
export caption_column=cached_captions

export use_perceptual_loss=true
export perceptual_layers="3,6,9,12"
export perceptual_weight=${perceptual_weight:-0.5}
export use_reconstruction_loss=false
export use_prototype_conditioning=false
export use_text_to_proto=false

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"