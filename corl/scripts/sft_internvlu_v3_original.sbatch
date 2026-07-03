#!/bin/bash
#SBATCH --job-name=internvlu_v3_original
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.out
#SBATCH --error=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs/%x-%j.err

set -euo pipefail
mkdir -p /projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu

export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1
export HF_HUB_OFFLINE=1

hostname
nvidia-smi --list-gpus

cd /projects/u6gd/umar/codes/InternVL-U

CKPT=/projects/u6gd/umar/codes/InternVL-U/InternVL-U
DATADIR=/projects/u6gd/datasets/PubMedVision

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12345 \
    -m corl.open_r1.sft_internvlu_alignment \
    --model_name_or_path $CKPT \
    --dataset_name PubMedVision_Original_Caption.json \
    --data_dir $DATADIR \
    --caption_source original \
    --caption_column Original_Caption \
    --image_column image \
    --gen_image_size 512 \
    --use_peft --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --bf16 \
    --learning_rate 2e-4 \
    --max_steps 50000 --logging_steps 10 --save_steps 5000 \
    --output_dir /projects/u6gd/umar/codes/ULM-R1/results/internvlu_t2i_original_captions_v3 \
    --remove_unused_columns False \
    --report_to none
