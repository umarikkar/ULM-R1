#!/bin/bash
# Exp 6: exp2 setup (cached captions, no LPIPS, no BMC perceptual) + prototype
# conditioning + text->prototype head, BUT with the KL gradients flowing back
# through the LM (LoRA). Detach was removed in modeling_vlm.py so the text head
# AND the LM share the prototype-alignment signal.
#
# Motivation: exp5 prototype path failed because hidden states were detached,
# so the head's MLP had to learn from frozen LLaMA-pooled-text features that
# never naturally clustered into BiomedCLIP image clusters. Now both the head
# and the LM (via LoRA) can adapt to make the mapping work.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export master_port=${master_port:-12356}
export EXP_NAME=${EXP_NAME:-"exp6_cached_proto_joint"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${EXP_NAME}"}

export num_train_epochs=${num_train_epochs:-1}
export caption_source=original
export caption_column=cached_captions

# NO auxiliary pixel/perceptual losses -- skips the slow STE-decode +
# BiomedCLIP forward path. Training should be ~exp2 speed.
export use_perceptual_loss=false
export use_reconstruction_loss=false

# Prototype conditioning + text->prototype head with joint LM training.
export use_prototype_conditioning=true
export use_text_to_proto=true
export prototype_centroids_path=${prototype_centroids_path:-data/prototype_centroids.pt}
export cond_temperature=${cond_temperature:-0.1}
export cond_dropout_prob=${cond_dropout_prob:-0.1}
export text_to_proto_aux_weight=${text_to_proto_aux_weight:-1.0}

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"