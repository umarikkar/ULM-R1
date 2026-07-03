#!/bin/bash
#SBATCH --job-name=janus_levels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err
#
# Janus-Pro T2I alignment SFT on a caption-granularity level, on Isambard.
# Nodes here are NVIDIA GH200 (96GB HBM3 per GPU), so the .sh Isambard branch
# defaults to PER_DEVICE_BS=16 (well above what a 24GB card allows).
#
# Choose the level (default l1). Either edit LEVEL below or submit with:
#   sbatch --export=ALL,LEVEL=l2 corl/scripts/sft_janus_levels.sbatch
# Push the batch higher on the 96GB card with:
#   sbatch --export=ALL,LEVEL=l1,PER_DEVICE_BS=32 corl/scripts/sft_janus_levels.sbatch
set -euo pipefail

export LEVEL=${LEVEL:-l1}
export NPROC=${NPROC:-4}          # must match --gpus above

REPO=/projects/u6gd/umar/codes/ULM-R1
mkdir -p "$REPO/corl/scripts/logs"

hostname
nvidia-smi --list-gpus

# The .sh handles conda activation, host paths and the torchrun launch.
bash "$REPO/corl/scripts/sft_janus_levels.sh"
