#!/bin/bash
# Exp 1: SFT with PubMedVision Original (GPT) captions.
# 4 GPUs, 2 epochs. Baseline using the dataset's own captions.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export master_port=${master_port:-12351}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"exp1_pubmed_captions"}

export num_train_epochs=2
export DATA_PATH=PubMedVision_Original_Caption.json
export caption_source=original
export caption_column=Original_Caption

# Losses: T2I CE only (no perceptual, no recon, no proto, no text2proto).
export use_perceptual_loss=false
export use_reconstruction_loss=false
export use_prototype_conditioning=false
export use_text_to_proto=false

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"