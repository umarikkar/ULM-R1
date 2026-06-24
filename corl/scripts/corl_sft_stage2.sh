#!/bin/bash
# Stage 2: LoRA on the LLaMA backbone + full-FT of gen_head/gen_aligner.
# Target: ~1 hour on 8x RTX 3090 (24GB) for a first medical T2I pass.
# Self-distilled prompts (image -> caption -> image), CFG dropout p=0.1.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# ******************* Paths *******************
# Auto-pick data dir based on hostname (mirrors debug_corl_sft.py).
HOSTNAME_SHORT=$(hostname -s)
case "$HOSTNAME_SHORT" in
    cvssp-retina03) DATA_DIR=/work/um00109/MLLM/datasets/PubMedVision ;;
    ulws072)        DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision ;;
    *)              DATA_DIR=/vol/research/fmodel_medical/people/umar/datasets/PubMedVision ;;
esac

CKPT_PATH=deepseek-ai/Janus-Pro-1B
# Optional warm start: path to a prior Stage-2 checkpoint dir containing
# adapter_model.safetensors. Empty -> fresh LoRA from CKPT_PATH (base Janus).
# CKPT_PATH is always the BASE model id; WARM_START_CKPT is the adapter.
WARM_START_CKPT=${WARM_START_CKPT:-}
DATA_PATH=${DATA_PATH:-PubMedVision_CachedCaptions_K4.json}
# DATA_PATH=/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/data/t2i_midlevel_llama.parquet

# Caption source: "self_distill" (model captions its own image each step) or
# "original" (use the real PubMed Original_Caption from the JSON row).
caption_source=${caption_source:-original}
caption_column=${caption_column:-cached_captions}
perceptual_weight=${perceptual_weight:-0.5}
perceptual_layers=${perceptual_layers:-"3,6,9,12"}
# Data filter via sidecar: drops is_grid=='multi' rows so training matches the
# labeled-attribute run's data distribution (for a fair A/B vs prototypes).
# Set attribute_sidecar="" to keep grids in. Only used as a grid filter --
# labeled modality/pose conditioning has been removed from the codebase.
attribute_sidecar=${attribute_sidecar:-data/attribute_sidecar.json}
# Held-out eval ids dropped from the training set so train/test stay disjoint.
exclude_ids_json=${exclude_ids_json:-corl/eval/test_split.json}
cond_dropout_prob=${cond_dropout_prob:-0.1}

# Unsupervised prototype conditioning (the new default path). Soft-assigns each
# training image to K BiomedCLIP-feature prototypes and ADDS the weighted
# prototype embedding to every image-position embedding.
use_prototype_conditioning=${use_prototype_conditioning:-true}
prototype_centroids_path=${prototype_centroids_path:-data/prototype_centroids.pt}
cond_temperature=${cond_temperature:-0.1}

# Text -> prototype head: lets us drop the image at inference time and still
# build a prototype cond from the caption alone. Trained via auxiliary KL
# against the image-side soft assignment. Single-forward design: training cond
# uses w_image; KL aligns w_text -> w_image so inference cond ~= training cond.
use_text_to_proto=${use_text_to_proto:-true}
text_to_proto_aux_weight=${text_to_proto_aux_weight:-1.0}

# Loss switches.
use_perceptual_loss=${use_perceptual_loss:-true}
use_reconstruction_loss=${use_reconstruction_loss:-false}
lpips_weight=${lpips_weight:-1.0}

# Optional wandb run name override.
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

SAVE_DIR=./results/JanusPro-1B-T2I-Stage2-LoRA
_attr_tag=""
if [ -n "${attribute_sidecar}" ]; then _attr_tag="_nogrid"; fi
_proto_tag=""
if [ "${use_prototype_conditioning}" = "true" ]; then _proto_tag="_proto-t${cond_temperature}-d${cond_dropout_prob}"; fi
_t2p_tag=""
if [ "${use_text_to_proto}" = "true" ]; then _t2p_tag="_t2p-w${text_to_proto_aux_weight}"; fi
_ws_tag=""
if [ -n "${WARM_START_CKPT}" ]; then _ws_tag="_ws"; fi
# If EXP_NAME is set, use it as the run dir verbatim (clean, deterministic
# location for the experiment script wrappers). Otherwise fall back to the
# timestamped, config-tagged name for ad-hoc runs.
if [ -n "${EXP_NAME}" ]; then
    RUN_TAG=${EXP_NAME}
else
    RUN_TAG=$(date +%Y%m%d_%H%M%S)_${caption_source}_${caption_column}_pw${perceptual_weight}_pl${perceptual_layers//,/-}${_attr_tag}${_proto_tag}${_t2p_tag}${_ws_tag}
fi
SAVE_PATH=${SAVE_DIR}/${RUN_TAG}
mkdir -p "$SAVE_PATH"
cp "$0" "$SAVE_PATH/run.sh"

# ******************* Training budget *******************
# 8 GPUs * bs=1 * grad_accum=4  =>  effective batch 32.
# Measured ~17s per optimizer step on 8x3090 (dominated by the per-step
# i2t self-distill rollout, 256 AR tokens). 200 steps ~= 1 hour.
per_device_train_batch_size=${per_device_train_batch_size:-2}
gradient_accumulation_steps=${gradient_accumulation_steps:-4}
num_train_epochs=${num_train_epochs:-1}
max_steps=${max_steps:--1}

# ******************* Hyperparameters *******************
learning_rate=1e-4         # LoRA likes higher LR than full FT
max_prompt_length=1024
max_completion_length=576  # N_IMAGE_TOKENS

# LoRA (TRL ModelConfig flags)
use_peft=true
lora_r=32
lora_alpha=64
lora_dropout=0.0

# CFG training
prompt_dropout_prob=0.1

# Perceptual loss
perceptual_warmup_steps=2000
# Transformer layers (1-indexed) tapped for the multi-scale perceptual loss are
# set with caption defaults above (perceptual_layers); the shallow taps capture
# low-level structure (medical images are structurally similar), the deep taps
# capture semantics.

# Eval-time generation
eval_image_freq=500
eval_image_num=8
eval_image_cfg=5.0
eval_image_temp=1.0

# ******************* Launch *******************
# The script imports `corl.open_r1.trainer....`, so the project root must be
# on PYTHONPATH (torchrun only adds the entrypoint's directory).
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

# export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-12345}

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
    corl/open_r1/sft_janus_alignment.py \
    --model_name_or_path "${CKPT_PATH}" \
    --dataset_name "${DATA_PATH}" \
    --data_dir "${DATA_DIR}" \
    --attribute_sidecar "${attribute_sidecar}" \
    --exclude_ids_json "${exclude_ids_json}" \
    --cond_dropout_prob ${cond_dropout_prob} \
    --use_prototype_conditioning ${use_prototype_conditioning} \
    --prototype_centroids_path "${prototype_centroids_path}" \
    --cond_temperature ${cond_temperature} \
    --use_text_to_proto ${use_text_to_proto} \
    --text_to_proto_aux_weight ${text_to_proto_aux_weight} \
    --warm_start_checkpoint "${WARM_START_CKPT}" \
    --output_dir "${SAVE_PATH}" \
    --task_format t2i \
    --lazy_image_loading true \
    --report_to wandb \
    --logging_steps 5 \
    --max_prompt_length ${max_prompt_length} \
    --max_completion_length ${max_completion_length} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --max_steps ${max_steps} \
    --learning_rate ${learning_rate} \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --save_steps 1000 \
    --save_total_limit 2 \
    --save_only_model true \
    --remove_unused_columns false \
    --use_peft ${use_peft} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --prompt_dropout_prob ${prompt_dropout_prob} \
    --caption_source ${caption_source} \
    --caption_column ${caption_column} \
    --eval_image_freq ${eval_image_freq} \
    --eval_image_num ${eval_image_num} \
    --eval_image_cfg ${eval_image_cfg} \
    --eval_image_temp ${eval_image_temp} \
    --use_perceptual_loss ${use_perceptual_loss} \
    --perceptual_weight ${perceptual_weight} \
    --perceptual_warmup_steps ${perceptual_warmup_steps} \
    --perceptual_layers "${perceptual_layers}" \
    --use_reconstruction_loss ${use_reconstruction_loss} \
    --lpips_weight ${lpips_weight} \
    ${WANDB_RUN_NAME:+--run_name "${WANDB_RUN_NAME}"} \
