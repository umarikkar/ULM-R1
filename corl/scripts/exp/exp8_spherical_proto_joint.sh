#!/bin/bash
# Exp 8: exp6 recipe (cached captions, joint-LM prototype + text2proto) but
# swapping vanilla K-means centroids for SPHERICAL K-means (v03_spherical_k16).
# 8 GPUs, 1 epoch, effective batch matched to exp6 (32) via grad_accum=2.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export master_port=${master_port:-12358}
export EXP_NAME=${EXP_NAME:-"exp8_spherical_proto_joint"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${EXP_NAME}"}

export num_train_epochs=${num_train_epochs:-1}
export caption_source=original
export caption_column=cached_captions

export use_perceptual_loss=false
export use_reconstruction_loss=false

export use_prototype_conditioning=true
export use_text_to_proto=true
export prototype_centroids_path=${prototype_centroids_path:-data/prototype_variants/v03_spherical_k16.pt}
export cond_temperature=${cond_temperature:-0.1}
export cond_dropout_prob=${cond_dropout_prob:-0.1}
export text_to_proto_aux_weight=${text_to_proto_aux_weight:-1.0}

# Match exp6 effective batch (32) on 8 GPUs: 2 * 2 * 8 = 32.
export gradient_accumulation_steps=${gradient_accumulation_steps:-2}

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"
