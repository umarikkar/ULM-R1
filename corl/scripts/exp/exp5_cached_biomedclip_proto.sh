#!/bin/bash
# Exp 5: SFT with cached captions + BiomedCLIP perceptual (3,6,8,12) +
# unsupervised prototype conditioning + text->prototype head (the proposed
# full method). 4 GPUs, 2 epochs.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
export master_port=${master_port:-12355}
export EXP_NAME=${EXP_NAME:-"exp5_cached_biomedclip_proto"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${EXP_NAME}"}

export num_train_epochs=2
export caption_source=original
export caption_column=cached_captions

export use_perceptual_loss=true
export perceptual_layers="3,6,8,12"
export perceptual_weight=${perceptual_weight:-0.5}
export use_reconstruction_loss=false

# Prototype conditioning + text->prototype head.
export use_prototype_conditioning=true
export use_text_to_proto=true
export prototype_centroids_path=${prototype_centroids_path:-data/prototype_centroids.pt}
export cond_temperature=${cond_temperature:-0.1}
export cond_dropout_prob=${cond_dropout_prob:-0.1}
export text_to_proto_aux_weight=${text_to_proto_aux_weight:-1.0}

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"
