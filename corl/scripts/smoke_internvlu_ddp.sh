#!/bin/bash
set -euo pipefail

source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu

export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1
export HF_HUB_OFFLINE=1

echo "=== DDP SMOKE TEST (2 GPUs, use_reentrant=False) ==="
cd /projects/u6gd/umar/codes/InternVL-U

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12345 \
    -m corl.open_r1.sft_internvlu_alignment \
    --model_name_or_path /projects/u6gd/umar/codes/InternVL-U/InternVL-U \
    --dataset_name PubMedVision_CachedCaptions_K4.json \
    --data_dir /projects/u6gd/datasets/PubMedVision \
    --caption_source original \
    --caption_column cached_captions \
    --image_column image \
    --gen_image_size 512 \
    --use_peft --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --bf16 \
    --learning_rate 1e-4 \
    --max_steps 3 --logging_steps 1 \
    --save_strategy no --eval_strategy no \
    --output_dir /projects/u6gd/umar/codes/ULM-R1/results/smoke_ddp \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --max_samples 8 \
    --report_to none

echo "DDP SMOKE OK"
