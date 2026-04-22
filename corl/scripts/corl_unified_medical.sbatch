#!/bin/bash
#SBATCH --job-name=corl_unified_medical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err

set -euo pipefail

mkdir -p /projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs
cd /projects/u6gd/umar/codes/ULM-R1

hostname
nvidia-smi --list-gpus

bash corl/scripts/corl_unified_medical.sh
