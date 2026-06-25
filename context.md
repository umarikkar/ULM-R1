# InternVL-U Alignment Training — Session Context

## Primary Goal

Add InternVL-U (sibling repo `../InternVL-U`) as a second backbone alongside the existing Janus-Pro backbone.

### Specific Tasks
- Apply LoRA to the InternVL-U LLM component
- Implement an alignment loss: image through LLM → latent feature set, aligned with text-to-image latent feature set
- Mirror existing `corl/open_r1/sft_janus_alignment.py` + `corl/open_r1/trainer/sft_trainer_alignment.py`
- **Chosen approach: Option B — VAE-latent regression**

---

## Architecture Comparison

### Janus-Pro (existing)
- Autoregressive discrete VQ token generation
- LLM predicts VQ token IDs; CE loss against GT VQ token IDs from `gen_vision_model.encode(image)`
- LoRA on LM; `gen_head` and `gen_aligner` are fully trainable

### InternVL-U (new)
- Continuous latent diffusion; LLM acts as a **conditioning encoder**, not AR generator
- Separate DiT (`InternVLUGenerationDecoder`) does image generation
- VAE is continuous (not VQ) — no codebook, no discrete token IDs

---

## Key Technical Details

### State Mask
- Selects VLM hidden states from 2nd `<|im_start|>` (user turn start) to `<img_start>` token
- This gives T_k tokens = user prompt token count (≈ 10–80 depending on caption length)

### Padding to 512
- Variable-length `[T_k, D]` tensors padded with learned `encoder_padding_token` (nn.Parameter) to fixed `[512, D]`
- DiT uses attention_mask to ignore padded positions
- `decoder_projector` MLP: `[B, 512, llm_hidden_size]` → `[B, 512, 2304]` for DiT

### Dual CFG
- DiT runs 3 copies per step (full-cond, part-cond, uncond)
- `encoder_hidden_states` must have length divisible by 3

### num_image_token = 256
- `(448//14)^2 * (0.5)^2 = 256`
- `<IMG_CONTEXT>` tokens representing an image in I2T understanding path
- h_img from I2T = `[B, 256, llm_hidden_size]` — **incompatible** with h_text = `[B, T_k, llm_hidden_size]`

### VAE Latent Shape
- `[B, z_dim, H//32, W//32]` e.g. `[B, z_dim, 16, 16]` for 512×512
- `vae_scale_factor=32`, `z_dim = vae.config.z_dim`

### LoRA Target Modules (same as Janus)
- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `task_type=None` (generic PeftModel wrapper, not PeftModelForCausalLM)

### Special Tokens
- `<img>` = img_start_token, `</img>` = img_end_token
- `<IMG_CONTEXT>` = img_context_token, `<img_uncond>` = img_uncond_token
- `<|im_start|>` / `<|im_end|>` = chat template tokens

---

## Key Files

### Existing (Janus) — Reference
- `corl/open_r1/sft_janus_alignment.py` — training script
  - `_build_stage2_peft_config`: LoraConfig with `task_type=None`, same target modules, `modules_to_save=["gen_head","gen_aligner"]`
  - `SFTScriptArguments`: all training hyperparams
- `corl/open_r1/trainer/sft_trainer_alignment.py` — `SFTAlignmentTrainer(Trainer)`
  - `init_trainable_parameters`: freezes `vision_model`, `aligner`, `gen_vision_model`, `gen_embed`, `language_model`; keeps `gen_head`, `gen_aligner` trainable
  - `get_image_gen_reps`: `model.gen_vision_model.encode(pixel_values)` → `[B, N, D_vq]` + `gt_ids [B, N]`
  - `compute_loss`: CE loss between student logits and `t2i_discrete_img_ids`

### InternVL-U Source (in `../InternVL-U/`)
- `internvlu/vlm/modeling_internvlu_chat.py`
  - `InternVLUChatModel`: submodules `vision_model`, `language_model` (Qwen3/LLaMA), `mlp1` (vision→LLM projector), `special_token_embedding`
  - `generate_hidden_states`: returns hidden states with `output_hidden_states=True`
- `internvlu/pipeline_internvlu.py`
  - `_prepare_hidden_state_mask`: builds boolean mask selecting tokens from 2nd `<|im_start|>` to `<img_start>`
  - `_prepare_diffusion_inputs`: extracts VLM hidden states via state_mask → list of `[T_k, llm_hidden_size]`
- `internvlu/diffusion/modeling_internvlu_generation_decoder.py`
  - `encoder_padding_token = nn.Parameter(torch.zeros(input_hidden_size))`
  - `decoder_projector`: MLP `input_hidden_size → output_hidden_size*3 → output_hidden_size`
  - `_prepare_forward_input_default`: pads list of tensors to `[B, 512, D]`, builds attention_mask
- `internvlu/diffusion/pipeline_internvlu_generation_decoder.py`
  - `vae_scale_factor=32`, `latent_channels = vae.config.z_dim`
  - `pixels_to_latents` / `latents_to_pixels`: VAE encode/decode with normalization
- `internvlu/diffusion/configuration_internvlu_generation_decoder.py`
  - `input_hidden_size=1536`, `output_hidden_size=2304`, `max_sequence_length=512`
  - `gen_image_height=512`, `gen_image_width=512`, `vae_downsample_factor=32`
- `internvlu/vlm/constants.py`
  - `IMG_CONTEXT_TOKEN`, `IMG_START_TOKEN`, `IMG_END_TOKEN`, `IMG_UNCOND_TOKEN`

### To Create (in this repo)
- `corl/open_r1/sft_internvlu_alignment.py` — training script mirroring `sft_janus_alignment.py`
- `corl/open_r1/trainer/sft_trainer_alignment_internvlu.py` — trainer mirroring `sft_trainer_alignment.py`

---

## Option B: VAE-Latent Regression Design

### Why Token-by-Token Alignment Fails
- h_img (I2T path): `[B, 256, llm_hidden_size]` — 256 image patch tokens through understanding path
- h_text (T2I path): `[B, T_k, llm_hidden_size]` — variable prompt tokens through generation conditioning path
- Different sequence lengths with no token correspondence → cannot align token-by-token

### Option B Approach
1. Extract T2I conditioning hidden states via state_mask → `[B, T_k, D]`
2. Pool (mean over T_k) → `[B, llm_hidden_size]`
3. Apply trainable `z_proj = nn.Linear(llm_hidden_size, z_dim * (H//32) * (W//32))`
4. MSE loss against `vae.encode(image).detach()` (frozen VAE latent)

### Implementation Design Decisions
- **LoRA**: on `vlm.language_model` with same target modules as Janus
- **Freeze**: `vlm.vision_model`, `vlm.mlp1`, `generation_decoder`, `vae`
- **New trainable head**: `z_proj` — analogous to Janus's `gen_head`
- **`modules_to_save = ["z_proj"]`** in LoRA config
- **Model loading**: `AutoModel.from_pretrained(..., trust_remote_code=True)` for VLM
- **Processing**: InternVL-U's tokenizer/processor (not VLChatProcessor)
- **`init_trainable_parameters`**: freeze `vlm.vision_model`, `vlm.mlp1`, `vlm.language_model` base; keep `z_proj` trainable; LoRA handles LM
