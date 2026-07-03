# `debug_corl_sft.py` — what actually happens (end-to-end)

Single-process (no `torchrun`) debug entrypoint for the Janus alignment SFT.
Run with: `python corl/scripts/debug_corl_sft.py`. Equivalent to the
`corl_unified.sh` torchrun command but one process for breakpoint debugging.

Files in the call chain:
- [debug_corl_sft.py](../scripts/debug_corl_sft.py) — hard-codes args, calls `main`.
- [sft_janus_alignment.py](../open_r1/sft_janus_alignment.py) — `main`: dataset prep + builds trainer.
- [sft_trainer_alignment.py](../open_r1/trainer/sft_trainer_alignment.py) — `SFTAlignmentTrainer`: the real work.

---

## 1. `debug_corl_sft.py` (the launcher)

- Disables cuDNN (`torch.backends.cudnn.enabled = False`) — cuDNN 9.1.9 fails to init on this box.
- Disables W&B (`WANDB_DISABLED`, `report_to="none"`).
- Resolves `DATA_DIR` by hostname (falls back to `/projects/u6gd/datasets/PubMedVision`).
- Model: `deepseek-ai/Janus-Pro-1B`; `model_ckpt_dir = <repo>/checkpoint`.
- Data: `./PubMedVision_CachedCaptions_K4.json` (so the `'PubMedVision' in dataset_name` branch fires → JSON loader).
- Output: `./results/DEBUGGING/AlignmentSFT_Stage2_LoRA`.
- Builds three arg objects and calls `main(script_args, training_args, model_args, max_samples=5000)`.

Notable config it pins on (Stage-2 LoRA T2I run):
- `alignment_losses=["masking","hidden"]` (this field is **not** actually read by the trainer — see Gotchas).
- `caption_source="original"`, `caption_column="cached_captions"` → captions come from the JSON row's `cached_captions` list (one picked at random per step), **not** self-distilled.
- `use_perceptual_loss=True`, `perceptual_weight=0.5`, `perceptual_warmup_steps=10`, `perceptual_layers="3,6,9"` (BiomedCLIP feature loss on STE-decoded pixels).
- `use_prototype_conditioning=True` + `prototype_centroids_path="data/prototype_centroids.pt"`, `cond_temperature=0.1`, `cond_dropout_prob=0.1`.
- `use_text_to_proto=True`, `text_to_proto_aux_weight=1.0` (caption→prototype head w/ KL aux).
- `attribute_sidecar="data/attribute_sidecar.json"` — used **only** as a grid filter (drop `is_grid=='multi'`); its modality/pose labels are NOT used.
- `exclude_ids_json="corl/eval/test_split.json"` — drops held-out test ids.
- LoRA `r=32, alpha=64`, bf16, `max_steps=12`, batch 1, lr 4e-5, no grad checkpointing.

## 2. `main()` in `sft_janus_alignment.py` (dataset + trainer build)

1. Loads JSON dataset, sets `img_key="image"`.
2. `max_samples=5000` truncation per split.
3. `resolve_image_path` → builds absolute `example["image"]` path; `add_dummy_prompt` adds a placeholder `prompt`; filters rows whose image file is missing.
4. **Sidecar grid filter**: joins by `id`, drops `is_grid=='multi'` rows (prints before/after counts).
5. **Exclude ids**: drops test-split ids so train/test stay disjoint.
6. Builds Stage-2 PEFT config via `_build_stage2_peft_config`:
   - `task_type=None` (generic `PeftModel`, since `MultiModalityCausalLM` isn't a stock causal LM).
   - `target_modules` = LLaMA attn+MLP projections.
   - `modules_to_save = ["gen_head","gen_aligner"]` (+`"prototype_emb"` when prototype cond is on). **`text_to_proto` is deliberately NOT here** (avoids DDP "marked ready twice").
7. Instantiates `SFTAlignmentTrainer(...)`, calls `trainer.train()`, then `trainer.save_model(output_dir)`.

## 3. `SFTAlignmentTrainer` (the trainer)

### `__init__`
- Loads Janus via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, torch_dtype=bf16)`.
- If `use_prototype_conditioning`: loads centroids `.pt`, attaches `model.prototype_emb = nn.Embedding(K+1, d_model)` (row 0 = zero "unknown" row for CFG dropout).
- `init_trainable_parameters(model)`: **freezes** vision_model, aligner, gen_vision_model, gen_embed, language_model; **trains** `gen_head`, `gen_aligner` (+`prototype_emb`, +`text_to_proto` if present).
- Wraps in PEFT (`get_peft_model`) → LoRA on the LLaMA backbone.
- If `use_text_to_proto`: attaches an MLP head `d_model → GELU → K` onto the **inner** Janus model (`model.base_model.model`) *after* PEFT wrapping, last layer zero-init.
- Optional `warm_start_checkpoint`: overlays prior LoRA + modules_to_save weights via `load_state_dict(strict=False)` (optimizer/step NOT restored).
- Processing class = Janus `VLChatProcessor` with empty system prompt.
- `data_collator` is identity (`return features`) → batches are lists of raw dict rows; image decoding happens later.
- Registers `T2IEvalCallback` (every `eval_image_freq` steps) and `TextToProtoSaveCallback`.

### Per-step data prep — `_prepare_inputs(inputs)`
- `load_batch_images`: opens each row's `image` as RGB.
- `caption_source="original"` → reads `caption_column` (`cached_captions`), picks one at random per row, wraps with `wrap_t2i_prompt` (User/Assistant SFT template + `image_start_tag`, left-padded). (The `"self_distill"` branch would instead run i2t generation via `get_i2t_t2i_inputs` to caption the image with the model itself.)
- Truncates prompt to `max_prompt_length` (last 1024 tokens).
- Returns `{t2i_input_ids, t2i_attention_mask, images, sample_ids}`.

### The loss — `compute_loss(model, inputs, ...)`
1. `_apply_prompt_dropout`: with prob `prompt_dropout_prob`, replaces prompt body with `pad_id` (keeps BOS + final BOI) → CFG-style unconditional training.
2. Under `inference_mode`: VQ-encode the GT image → `t2i_discrete_img_ids` `[B, 576]`. These are **both** the teacher-forcing context AND the CE targets.
3. **Prototype conditioning** (if on): run frozen BiomedCLIP on the GT image → features → cosine-sim to centroids → softmax(τ) → `w_image` → `cond = w_image @ prototype_emb[1:]`. With `cond_dropout_prob`, some rows swapped to the zero "unknown" row. Passed to the model as `t2i_img_pos_bias` (Janus broadcasts to `[B, 576, d]`, added to image-position embeds).
4. `student_out = model(t2i_input_ids=..., t2i_discrete_img_ids=..., t2i_img_pos_bias=..., task="generation", t2i_compute_text_to_proto=..., t2i_text_pool_mask=...)` → `student_logits` `[B, 576, V_img]`.
5. **Primary loss** = `cross_entropy(student_logits, t2i_discrete_img_ids)` (`loss_ce`).
6. **text2proto KL aux** (if on): `KL(w_text || w_image.detach())`, added with `text_to_proto_aux_weight`. The head is called *inside* `model.forward` so DDP tracks its params; pooled hidden states are detached there.
7. **Perceptual loss** (if on): warmup ramp (5% hold then linear over `perceptual_warmup_steps`); STE-decode `student_logits` → pixels (`_ste_decode_pixels`: softmax·codebook for grad, hard argmax-emb for forward, straight-through); BiomedCLIP multi-layer cosine distance vs GT; dynamically scaled so `effective_w·loss_perc ≈ loss_ce` at weight 1.0.
8. **LPIPS reconstruction** (`use_reconstruction_loss`, OFF here): pixel-space VGG LPIPS on STE pixels.
9. Logs `loss_ce`, `token_acc`, and per-quartile `token_acc_q0..q3` (q0 = least context → must rely on prompt; q3 = most context).
10. Returns scalar `loss`.

### `training_step`
- Calls `super().training_step`, then **zeroes any non-finite grads** so one bad batch doesn't poison gradient-norm clipping (logs `nan_grad_zeroed`, prints once).

### Eval images — `T2IEvalCallback` → `_run_t2i_eval`
- Every `eval_image_freq` steps: caches the first `eval_image_num` train prompts once, then `_generate_one_image` does **CFG** AR generation (cond/uncond pair, `eval_image_cfg` guidance), decodes via `gen_vision_model.decode_code`, and saves an original|generated side-by-side PNG under `output_dir/eval_samples/step_XXXXXX/`. With text2proto, the inference cond vector is built from the caption alone (`w_text`), no image needed.

---

## Gotchas / things to remember

- **Stale top-of-file docstring.** It claims "student is teacher-forced on its OWN AR rollout (inference_mode), then CE against GT VQ ids." That is **NOT** the current path: `ar_rollout_student_ids()` is defined but **never called**. `compute_loss` teacher-forces on the **GT VQ ids** and takes CE against those same GT ids. (Standard teacher-forcing, not self-rollout.)
- **`alignment_losses=["masking","hidden"]` is dead config** for this trainer — `compute_loss` never reads it. The actual losses are CE (+ optional text2proto KL, perceptual, LPIPS).
- **`add_dummy_prompt` / `prompt` column is unused** at train time; the real prompt comes from `cached_captions` (or self-distill).
- **`attribute_sidecar` labels are ignored** — only its `is_grid` field is used to drop multi-panel figures.
- `gradient_checkpointing=False` and `use_cache` interplay: model loads with `use_cache=None` here.
- Trainable surface: LoRA (LLaMA) + `gen_head` + `gen_aligner` + `prototype_emb` + `text_to_proto`. Everything else (vision, VQ codec, LM base) is frozen.
- `text_to_proto` weights are saved separately by `TextToProtoSaveCallback` (`text_to_proto.safetensors` per checkpoint) because PEFT's `save_pretrained` skips it.
