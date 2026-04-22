"""
Debug script for corl_unified.sh — run directly with: python corl/scripts/debug_corl_unified.py
Equivalent to the torchrun command but single-process for easier debugging.
"""

# import debugpy

# debugpy.configure({"justMyCode": False})

import os
import sys
import threading
from datetime import datetime



import torch
torch.backends.cudnn.enabled = False  # cuDNN 9.1.9 fails to initialize on this system
os.environ["WANDB_DISABLED"] = "true"

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, ModelConfig, ScriptArguments, get_peft_config

from corl.open_r1.grpo_janus_unify import GRPOScriptArguments, main


def _get_rss_mb() -> float:
    """Return current process RSS (MB) using /proc to avoid extra dependencies."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: VmRSS:\t  123456 kB
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _format_cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return (
        f"gpu{device}: alloc={allocated:.2f}GB "
        f"reserved={reserved:.2f}GB peak_alloc={peak:.2f}GB total={total:.2f}GB"
    )


def _start_memory_monitor(interval_sec: int = 20):
    stop_event = threading.Event()

    def _loop():
        while not stop_event.is_set():
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rss_mb = _get_rss_mb()
            rss_text = f"rss={rss_mb:.1f}MB" if rss_mb >= 0 else "rss=unknown"
            print(f"[mem {ts}] {rss_text} {_format_cuda_mem()}", flush=True)
            stop_event.wait(interval_sec)

    thread = threading.Thread(target=_loop, name="memory-monitor", daemon=True)
    thread.start()
    return stop_event


if __name__ == "__main__":
    # ---- Paths (edit these) ---- #
    CKPT_PATH = "deepseek-ai/Janus-Pro-1B"
    MODEL_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoint")
    DATA_PATH = "/projects/u6gd/umar/codes/ULM-R1/data/t2i_midlevel_llama.parquet"  # Can be a single parquet file or a directory containing multiple parquet files
    SAVE_DIR = "./JanusPro-1B-CoRL-Uniified"
    SAVE_PATH = f"{SAVE_DIR}/RFT22k-CycleMatchAccFormat-UniReward-G4-beta004-bs16"

    os.makedirs(SAVE_PATH, exist_ok=True)

    # ---- Script arguments ---- #
    script_args = GRPOScriptArguments(
        dataset_name=DATA_PATH,
        model_ckpt_dir=MODEL_CKPT_DIR,
        image_base_dir="/projects/u6gd/datasets/PubMedVision/images",
        lazy_image_loading=True,
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
        max_completion_length=512,
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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

    # Print memory every N seconds while debugging training.
    monitor_interval = int(os.environ.get("DEBUG_MEM_INTERVAL", "20"))
    mem_stop_event = _start_memory_monitor(interval_sec=monitor_interval)
    try:
        main(script_args, training_args, model_args, max_samples=1000)
    finally:
        mem_stop_event.set()
