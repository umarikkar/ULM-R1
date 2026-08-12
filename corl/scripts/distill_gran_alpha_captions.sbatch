#!/bin/bash
#SBATCH --job-name=gran_distill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err
#
# Stage 1 (Isambard GH200): distill random-alpha captions with frozen gran_lora_v2,
# 4-way shard over the node's GPUs, merge -> a dataset column for T2I SFT.
# Delegates to distill_gran_alpha_captions.sh (conda, host paths, sharding).
#
# Debug (small): sbatch --export=ALL,MAX_SAMPLES=200,OUT_NAME=RandAlpha_debug.json corl/scripts/distill_gran_alpha_captions.sbatch
# Full run:      sbatch corl/scripts/distill_gran_alpha_captions.sbatch
set -eo pipefail

export GPUS=${GPUS:-0,1,2,3}          # must match --gpus above (GH200 nodes have 4/node)

REPO=/projects/u6gd/umar/codes/ULM-R1
mkdir -p "$REPO/corl/scripts/logs"
cd "$REPO"

hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "$REPO/corl/scripts/distill_gran_alpha_captions.sh"
