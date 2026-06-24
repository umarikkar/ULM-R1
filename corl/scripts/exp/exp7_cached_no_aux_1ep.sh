#!/bin/bash
# Exp 7: exp2 setup (cached captions, no aux losses, no prototype path)
# but trained for 1 epoch. Fair-comparison baseline for exp6 (also no aux
# losses, but WITH prototype + text2proto). Same data, same caption dist,
# same step count -- the only difference is the prototype machinery.

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
export master_port=${master_port:-12357}
export EXP_NAME=${EXP_NAME:-"exp7_cached_no_aux_1ep"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${EXP_NAME}"}

export num_train_epochs=1
export caption_source=original
export caption_column=cached_captions

export use_perceptual_loss=false
export use_reconstruction_loss=false
export use_prototype_conditioning=false
export use_text_to_proto=false

bash "${PROJECT_ROOT}/corl/scripts/corl_sft_stage2.sh"