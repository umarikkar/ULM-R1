#!/bin/bash

# source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
# source /projects/u6gd/umar/env.sh

source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh

# cd /projects/u6gd/umar/codes/ULM-R1

cd /vol/research/fmodel_medical/people/umar/MLMM/ULM-R1


CUDA_VISIBLE_DEVICES=0

conda activate corl

export WANDB_API_KEY="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"

wandb login

export PYTHONPATH="$(pwd):${PYTHONPATH}"

# *****************  ***************** #
CKPT_PATH=deepseek-ai/Janus-Pro-1B
# DATA_DIR=/projects/u6gd/datasets/PubMedVision
DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision
DATASET_NAME=PubMedVision_Original_Caption.json

SAVE_DIR=./results/JanusPro-1B-CoRL-AlignmentSFT
SAVE_PATH=${SAVE_DIR}/AlignmentSFT
mkdir -p $SAVE_PATH
cp $0 $SAVE_PATH/run.sh

learning_rate=4e-6
num_train_epochs=1

max_prompt_length=1024
max_completion_length=512

per_device_train_batch_size=1
gradient_accumulation_steps=4
max_samples=50000
lazy_image_loading=True

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    corl/open_r1/sft_janus_alignment.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATASET_NAME} \
    --data_dir ${DATA_DIR} \
    --lazy_image_loading ${lazy_image_loading} \
    --max_samples ${max_samples} \
    --max_prompt_length ${max_prompt_length} \
    --max_completion_length ${max_completion_length} \
    --report_to wandb \
    --logging_steps 1 \
    --remove_unused_columns false \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${learning_rate} \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --save_steps 200 \
    --save_total_limit 1 \
    --save_only_model true
