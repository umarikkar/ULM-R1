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

DEFAULT_DATA_DIR = "/projects/u6gd/datasets/PubMedVision"


DATA_DIR_DICT = {
    "cvssp-retina03": "/work/um00109/MLLM/datasets/PubMedVision",
    "ulws072": "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision",
}



HOSTNAME = os.uname().nodename.split(".")[0]
DATA_DIR = (DATA_DIR_DICT.get(HOSTNAME) if HOSTNAME in DATA_DIR_DICT.keys() else DEFAULT_DATA_DIR)

# DATA_DIR = 'data/t2i_midlevel_llama.parquet'


# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, ModelConfig, ScriptArguments, get_peft_config

from corl.open_r1.sft_janus_alignment import SFTScriptArguments, main


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


    DATA_PATH = "./PubMedVision_CachedCaptions_K4.json" 
    # DATA_PATH = "/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/data/t2i_midlevel_llama.parquet"

    SAVE_DIR = "./results/DEBUGGING"
    SAVE_PATH = f"{SAVE_DIR}/AlignmentSFT_Stage2_LoRA"

    os.makedirs(SAVE_PATH, exist_ok=True)

    # ---- Script arguments ---- #
    script_args = SFTScriptArguments(
        dataset_name=DATA_PATH,
        model_ckpt_dir=MODEL_CKPT_DIR,
        data_dir=DATA_DIR,
        lazy_image_loading=True,
        max_prompt_length=1024,
        max_completion_length=512,
        alignment_losses=["masking", "hidden"],
        use_reconstruction_loss=False,  # set True to add pixel-space LPIPS on top of latent MSE
        lpips_weight=1.0,
        prompt_dropout_prob=0.1,
        eval_image_freq=5,
        eval_image_num=2,
        # "self_distill": model captions its own image each step (current default).
        # "original": use the real PubMed Original_Caption from the JSON row.
        caption_source="original",
        # caption_column="caption",
        caption_column="cached_captions",
        use_perceptual_loss=True,
        perceptual_weight=0.5,
        perceptual_warmup_steps=10,
        perceptual_layers="3,6,9",
        # Keep the sidecar so the is_grid=='multi' filter still applies (same
        # training data as the labeled-attribute run). Conditioning *labels*
        # from the sidecar are NOT used -- only the grid filter.
        attribute_sidecar="data/attribute_sidecar.json",
        exclude_ids_json="corl/eval/test_split.json",
        cond_dropout_prob=0.1,
        # Unsupervised prototype conditioning -- BiomedCLIP-feature K-means
        # prototypes, soft-assigned per image, added to image-position embeds.
        # Requires data/prototype_centroids.pt produced by
        # build_prototype_centroids.sh.
        use_prototype_conditioning=True,
        prototype_centroids_path="data/prototype_centroids.pt",
        cond_temperature=0.1,
        # Text -> prototype head: bridges training (image features) and
        # inference (caption only). Anneal cond from w_image -> w_text.detach().
        use_text_to_proto=True,
        text_to_proto_aux_weight=1.0,
    )

    # ---- Training arguments ---- #
    training_args = SFTConfig(
        output_dir=SAVE_PATH,
        report_to="none",
        logging_steps=1,
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        max_steps=12,
        num_train_epochs=1,
        learning_rate=4e-5,
        bf16=True,
        gradient_checkpointing=False,
        save_steps=200,
        save_total_limit=1,
        save_only_model=True,
    )

    # ---- Model arguments (Stage 2: LoRA on the LLaMA backbone) ---- #
    model_args = ModelConfig(
        model_name_or_path=CKPT_PATH,
        torch_dtype="bfloat16",
        use_peft=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        # target_modules/modules_to_save are left None so the trainer fills in
        # the LLaMA defaults + ["gen_head","gen_aligner"].
    )


    main(script_args, training_args, model_args, max_samples=5000)

    # # Print memory every N seconds while debugging training.
    # monitor_interval = int(os.environ.get("DEBUG_MEM_INTERVAL", "20"))
    # mem_stop_event = _start_memory_monitor(interval_sec=monitor_interval)
    # try:
    #     main(script_args, training_args, model_args, max_samples=1000)
    # finally:
    #     mem_stop_event.set()
