#!/bin/bash
#SBATCH --job-name=gran_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err
#
# Stage-1 GRPO refinement of the gran-LoRA granularity knob, on Isambard (GH200 96GB).
# Delegates to corl/scripts/rl_gran_lora.sh, which handles conda, host paths and torchrun.
#
# The 96GB card lets the rollout go well past the 24GB defaults (K=4, 2 images/step).
# Defaults below are the GH200 settings; override at submit time, e.g.:
#   sbatch --export=ALL,MAX_STEPS=1000 corl/scripts/rl_gran_lora.sbatch
#   sbatch --export=ALL,GROUP_SIZE=8,IMAGES_PER_STEP=4,BETA_KL=0.02 corl/scripts/rl_gran_lora.sbatch
#
# ALWAYS cap MAX_STEPS: a full epoch is ~636k images and RL rollouts are slow.
set -eo pipefail

export NPROC=${NPROC:-4}                      # must match --gpus above
export GROUP_SIZE=${GROUP_SIZE:-8}            # K completions per image
export IMAGES_PER_STEP=${IMAGES_PER_STEP:-4}  # grad-accum images per GPU
export MAX_STEPS=${MAX_STEPS:-1000}
export SAVE_STEPS=${SAVE_STEPS:-250}

REPO=/projects/u6gd/umar/codes/ULM-R1
mkdir -p "$REPO/corl/scripts/logs"

hostname
nvidia-smi --list-gpus

bash "$REPO/corl/scripts/rl_gran_lora.sh"
