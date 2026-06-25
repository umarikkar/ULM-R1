# InternVL-U T2I Training on Isambard — START HERE (for Claude)

You are a fresh Claude instance on **Isambard-AI** (`u6gd.aip2.isambard`, ARM64 / GH200).
This doc is a complete handoff: read it, then set up the environment, verify it, and be
ready to launch InternVL-U text-to-image (T2I) training. Everything below was worked out
and **verified on an x86 box already**; your job is to reproduce it on this ARM machine.

---

## 0. TL;DR — what to do, in order

1. **Create the conda env** from `requirements-internvlu.txt` (ARM/CUDA wheels — see §3).
2. **Verify the env** with the GPU sanity check **inside a SLURM allocation** (login node has no GPU — see §4).
3. **Locate the datasets** (parquet + images) — they may not be on Isambard yet (see §5).
4. **Smoke-test the trainer** (1–2 steps) before any real run (see §6).
5. **Launch training** (see §7).

Do **not** re-derive the training design — the hard decisions are settled in §1. Don't try to
"fix" the loss into a one-shot regression; that was tested and rejected (§1).

---

## 1. Background & settled decisions (do NOT re-litigate)

**Goal:** add InternVL-U as a second generation backbone (alongside an existing Janus-Pro
backbone in this repo) and teach **text-only T2I** in a medical domain.

**How it's trained — rectified-flow velocity MSE through the FROZEN DiT:**
- caption → **LoRA** LLM hidden states `h_text` → frozen `generation_decoder` (DiT) denoises a
  noised VAE latent of the GT image → **MSE on the flow velocity** (`target = noise − z0`).
- Trainable: **LoRA on `vlm.language_model`** only (optionally `decoder_projector`).
  Frozen: `vision_model`, `mlp1`, the whole DiT, the VAE.
- This is the continuous analog of the repo's Janus AR-CE alignment trainer.

**Why not the simpler alternatives (already tested, rejected):**
- A probe showed the frozen DiT **cannot** reconstruct an image from the LLM hidden states
  alone — image content reaches the DiT only via the **VAE-latent image-stream tokens** of its
  joint attention, not via the cross-attn conditioning. So "match `h_text` to `h_img`" transfers
  nothing.
- A one-shot MLP/attention head regressing straight to the VAE latent bypasses the DiT →
  blurry. So **keep the DiT in the loop** (flow-matching). This is the only path that yields a
  usable text-only generator.

---

## 2. The code (already written & smoke-verified on x86)

Two files in **this repo** (`ULM-R1`):
- `corl/open_r1/trainer/sft_trainer_alignment_internvlu.py` — `SFTInternVLUAlignmentTrainer`
  + `InternVLUT2IFlowMatch` (a DDP-safe nn.Module whose `forward` IS the train step and returns
  the flow loss). Gradient-checkpointing, CFG dropout, NaN-grad guard, metrics logging all wired.
- `corl/open_r1/sft_internvlu_alignment.py` — entry point, mirrors `sft_janus_alignment.py`.

These already contain the non-obvious fixes (don't undo them): the LLM forward replicated
**without** `@torch.no_grad`, `.clone()` before the special-token replace, `enable_input_require_grads`
on the LLM, and diffusers `enable_gradient_checkpointing()` on the DiT.

---

## 3. Machine & paths on Isambard

| Thing | Path / value |
|---|---|
| Arch | `aarch64` (ARM), GH200 GPUs |
| conda base | `/projects/u6gd/umar/miniconda3` |
| Target env prefix | `/projects/u6gd/umar/miniconda3/envs/internvlu` |
| InternVL-U **code** | `/projects/u6gd/umar/codes/InternVL-U` (git `main` @ `018a59c`) → set `INTERNVLU_REPO` to this |
| InternVL-U **checkpoint** | `/projects/u6gd/umar/codes/InternVL-U/InternVL-U` (8.2 G, verified complete: vlm 4.4G / generation_decoder 3.6G / vae 243M / `model_index.json` / 3 safetensors) |
| This repo (`ULM-R1`) | clone/pull it under `/projects/u6gd/umar/codes/ULM-R1` (adjust if different); needs to be on `PYTHONPATH` for `corl`) |
| `requirements-internvlu.txt` | in this repo root |

---

## 4. Step 1 — create & verify the environment

The source env is x86_64 and was **not** copied (binaries can't run on ARM). Recreate from spec.

```bash
# On the LOGIN node (it has internet: GitHub + HuggingFace reachable):
conda create -p /projects/u6gd/umar/miniconda3/envs/internvlu python=3.10 -y
conda activate /projects/u6gd/umar/miniconda3/envs/internvlu
pip install -r /projects/u6gd/umar/codes/ULM-R1/requirements-internvlu.txt
```

**Critical ARM/CUDA gotcha:** the spec pins `torch==2.6.0+cu124` from the PyTorch cu124 index.
On aarch64 this must resolve to the **GH200 (sbsa) CUDA** wheel, not CPU-only. If pip can't find
the aarch64 `+cu124` wheel, fall back to **cu126**:
```bash
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126
# then re-run `pip install -r requirements-internvlu.txt` with the two torch lines removed
```
`flash_attn` is intentionally omitted (InternVL-U falls back to SDPA — fine).

**GPU sanity check — must run inside a SLURM GPU allocation** (the login node has NO GPU, so
`torch.cuda.is_available()` will be `False` there and that's expected). Discover the scheduler
first (`sinfo`, `sacctmgr show user $USER`, project account is likely `u6gd`), then e.g.:
```bash
srun --account=u6gd --gpus=1 --time=00:20:00 --pty bash   # adjust partition/flags to this cluster
python -c "import platform,torch;print(platform.machine(),'|',torch.__version__,'|',torch.cuda.is_available())"
# expect:  aarch64 | 2.6.0+cu124 | True
```

---

## 5. Step 2 — datasets (CHECK FIRST; may be missing)

Training reads a parquet of captions + an image root. On the source box these were:
- parquet: `.../datasets/VL-Health/t2i_midlevel_llama.parquet` (cols `image_path`, `detailed_caption`)
- images: `.../datasets/PubMedVision/images/`

**On Isambard these are probably not present yet.** Search `/projects/u6gd` for them; if absent,
they must be transferred from the source box (`/work/um00109/MLLM/datasets/...`) or re-obtained.
Do not start training until both the parquet and its image root exist locally. Note the
trainer's `--data_dir` is the image root that gets joined with each row's `image_path`.

---

## 6. Step 3 — smoke-test the trainer (do this before any real run)

Inside a GPU allocation, with the env active. Set env vars so `internvlu` and `corl` import:
```bash
export INTERNVLU_REPO=/projects/u6gd/umar/codes/InternVL-U
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1
export HF_HUB_OFFLINE=1   # checkpoint is local; avoid network at train time
cd /projects/u6gd/umar/codes/InternVL-U        # so the package + checkpoint resolve
```
Then run a tiny in-process smoke (2 steps, batch 2). Write this to a temp file and run it
(it mirrors the x86 smoke that passed — loss ~0.09, finite grads):
```python
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from trl import SFTConfig, ModelConfig
from corl.open_r1.sft_internvlu_alignment import SFTScriptArguments, main

CKPT    = "/projects/u6gd/umar/codes/InternVL-U/InternVL-U"
PARQUET = "<path-to>/t2i_midlevel_llama.parquet"     # from §5
IMGROOT = "<path-to>/PubMedVision/images"            # from §5

main(
    SFTScriptArguments(
        dataset_name=PARQUET, dataset_train_split="train", data_dir=IMGROOT,
        image_column="image_path", caption_source="original",
        caption_column="detailed_caption", gen_image_size=512,
        prompt_dropout_prob=0.1, max_prompt_length=1024, max_samples=4),
    SFTConfig(
        output_dir="./smoke_out", per_device_train_batch_size=2, max_steps=2,
        learning_rate=1e-4, logging_steps=1, save_strategy="no", eval_strategy="no",
        bf16=True, gradient_checkpointing=True, remove_unused_columns=False,
        dataloader_num_workers=0, report_to=[]),
    ModelConfig(model_name_or_path=CKPT, use_peft=True, lora_r=8, lora_alpha=16, lora_dropout=0.05),
    max_samples=4,
)
print("SMOKE OK")
```
Expect: pipeline loads, `trainable params ~8.7M / 4.17B (0.21%)`, two loss lines, `SMOKE OK`.
GH200 has 96 GB, so batch size and resolution can go well above the 3090 settings used on x86.

---

## 7. Step 4 — launch a real run (SLURM batch)

```bash
python -m corl.open_r1.sft_internvlu_alignment \
  --model_name_or_path /projects/u6gd/umar/codes/InternVL-U/InternVL-U \
  --dataset_name <path>/t2i_midlevel_llama.parquet \
  --data_dir <path>/PubMedVision/images \
  --use_peft --lora_r 16 --lora_alpha 32 \
  --per_device_train_batch_size 4 --gradient_checkpointing --bf16 \
  --learning_rate 1e-4 --max_steps 5000 --logging_steps 10 --save_steps 500 \
  --output_dir ./out_internvlu_t2i --remove_unused_columns False
```
Wrap this in an `sbatch` script with the right `--account`/partition/`--gpus`. Keep
`INTERNVLU_REPO`, `PYTHONPATH`, `HF_HUB_OFFLINE=1` exported in the job. Tune batch size up for GH200.

---

## 8. Verified architecture facts (so you trust the shapes)

Measured from the real checkpoint:
- LLM hidden = **2048**; `generation_decoder.config.vlm_select_layer = [-1, -2]` → conditioning
  concatenates 2 layers → width **4096**; padded to **768** tokens.
- VAE latent for 512×512 = **`[B, 16, 64, 64]`** (z_dim 16, spatial /8); DiT `image_grid_thw_gen=[1,64,64]`.
- The checkpoint's processor is **dynamic-res** (`fix_resolution=False`); the trainer therefore
  builds the target latent by resizing the GT image to 512² + `Normalize([.5],[.5])` →
  `image_pipeline.pixels_to_latents(...)` (do NOT feed dynamic-res flattened patches to that).
- DiT train-forward needs `image_fhw_cond=[zeros(0,3)]*B` (never None) and `conditional_input=None`
  for pure T2I. (All handled inside the trainer.)

---

## 9. Known limitations / TODO (mention to the user; don't silently change)

- **Square 512 only:** targets are resized to 512² (aspect ratio dropped). Native/any-res is a follow-up.
- **No eval-image callback:** no mid-training T2I sample generation yet (needs LoRA-aware pipeline
  wiring). Add if visual monitoring is wanted.
- **Single-process verified:** smoke-tested single-GPU. The forward module is built to be DDP-safe,
  but multi-GPU DDP hasn't been run end-to-end — verify on first multi-GPU launch.
- **CFG:** prompt dropout (`--prompt_dropout_prob`) trains the unconditional path so inference
  dual-CFG works.

---

## 10. First things to tell the user when you start

Confirm: (a) env created & GPU check passed (`aarch64 | … | True`); (b) datasets located or
flagged missing; (c) smoke test result. Then ask whether to launch a real run and with what
budget (steps, batch size, LoRA rank).
```
