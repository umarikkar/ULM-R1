"""
Debug script for corl_unified.sh — run directly with: python corl/scripts/debug_corl_unified.py
Equivalent to the torchrun command but single-process for easier debugging.
"""

# import debugpy

# debugpy.configure({"justMyCode": False})

import os
import sys

# Set environment before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Single GPU for debugging
os.environ["CUDA_HOME"] = "/vol/research/fmodel_medical/people/umar/miniconda3/envs/corl"



import torch
torch.backends.cudnn.enabled = False  # cuDNN 9.1.9 fails to initialize on this system
os.environ["WANDB_DISABLED"] = "true"

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, ModelConfig, ScriptArguments, get_peft_config

from corl.open_r1.grpo_janus_unify import GRPOScriptArguments, main


if __name__ == "__main__":
    # ---- Paths (edit these) ---- #
    CKPT_PATH = "deepseek-ai/Janus-Pro-1B"
    DATA_PATH = "/work/um00109/MLLM/datasets/VL-Health/t2i_midlevel_llama.parquet"  # Can be a single parquet file or a directory containing multiple parquet files
    SAVE_DIR = "./JanusPro-1B-CoRL-Uniified"
    SAVE_PATH = f"{SAVE_DIR}/RFT22k-CycleMatchAccFormat-UniReward-G4-beta004-bs16"

    os.makedirs(SAVE_PATH, exist_ok=True)

    # ---- Script arguments ---- #
    script_args = GRPOScriptArguments(
        dataset_name=DATA_PATH,
        image_base_dir="/work/um00109/MLLM/datasets/PubMedVision/images",
        reward_funcs=["t2i_bid_cycle_reward", "t2i_ti_sim", "qa_accuracy", "format"],
        task_format="unify",
        unify_advantage=False,
        unify_reward=True,
    )

    # ---- Training arguments ---- #
    training_args = GRPOConfig(
        output_dir=SAVE_PATH,
        report_to="none",
        logging_steps=1,
        beta=0.0,
        max_prompt_length=1024,
        max_completion_length=384,
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=4e-6,
        bf16=True,
        gradient_checkpointing=False,
        save_steps=200,
        save_total_limit=1,
        save_only_model=True,
    )

    # ---- Model arguments ---- #
    model_args = ModelConfig(
        model_name_or_path=CKPT_PATH,
        torch_dtype="bfloat16",
    )

    main(script_args, training_args, model_args, max_samples=1000)
