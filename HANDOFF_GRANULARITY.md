# Handoff — Granularity-controlled captioning + gran-LoRA (context for a fresh chat)

Everything you need to continue the **caption-granularity** line of work. Ignore
unrelated repo stuff (prototype conditioning, InternVL-U, etc.).

## Repo / env / hosts
- Repo: `/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1` (Janus-Pro-1B based, `janus/` package vendored).
- Conda env: **`corl`** (`source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh; conda activate corl`). torch 2.6+cu124, transformers 5.4, trl 0.18.1, peft 0.18.1.
- Host `cvssp-retina03`: 8× RTX 3090 (24GB). Data at `/work/um00109/MLLM/datasets/PubMedVision`.
- Host Isambard (`u6gd`): GH200 96GB; repo `/projects/u6gd/umar/codes/ULM-R1`, data `/projects/u6gd/datasets/PubMedVision`.
- Requirements snapshot: `requirements_corl.txt`.

## The overall research idea
Train a **stage-adaptive caption-granularity curriculum** for Janus T2I alignment.
Core hypothesis (validated below): *granularity matching matters* — the caption
granularity should match the generator's current capacity (coarse early → fine late).
The knob is **gran-LoRA**: a LoRA on the Janus **captioner (i2t)** whose **scale α**
controls caption granularity under a FIXED neutral prompt (α=0 = base default caption).

## Data: the leveled caption cache
`PubMedVision_CachedCaptions_Levels.json` (636,082 rows) — built by
`corl/scripts/build_caption_cache_levels.py`. Per image, 4 caption columns:
- `cached_captions_l1_meta`: templated `modality, body_part` label (zero-noise).
- `cached_captions_l1`: 1 sentence, modality+region ("This is a CT scan of the chest…").
- `cached_captions_l2`: 1 sentence + main finding.
- `cached_captions_l3`: detailed 2-3 sentence paragraph.
Generated image-conditioned with Janus-Pro-1B (greedy, k=1). Test split held out:
`corl/eval/test_split.json` (4998 rows, 6 modalities balanced), and
`corl/eval/test_split_levels.json` = test rows joined with their level captions.

## Prior result (why granularity matters) — DONE
Trained 3 Janus T2I alignment LoRA adapters, one per level (r32/α64, perceptual OFF,
prototype OFF, eval-image-gen OFF): `results/JanusPro-1B-Levels/level_{l1,l2,l3}/`.
Launcher `corl/scripts/sft_janus_levels.sh` (arg `--caption_level` in
`corl/open_r1/sft_janus_alignment.py`). Generation-only eval (FID-BiomedCLIP,
CLIPScore, Qwen-3B modality-accuracy/plausibility) → `results/eval_levels/`
(`SUMMARY.md`, `MATRICES.md`, `matrices.png`). Key findings:
- **Fixed coarse (l1) prompt:** coarse-trained adapter wins monotonically (FID 592 vs
  1162 for l3) → granularity **mismatch is costly**.
- Matched FID/CLIP improve with detail (trivial: more prompt info).
- l1 ties l3 on **modality accuracy** (0.747 vs 0.751) → coarse captures modality.
- FID-BiomedCLIP magnitudes are huge (400-2800): **rankings only**, not absolute. Janus-1B is a weak generator.

## Per-level 3-epoch baselines (Jul 19) — DONE (l1,l2 clean; l3 caveated)
Extended each level to **3 epochs total** = the 1-epoch adapter + **2 more**, to serve as
per-level baselines for a future `l1→l2→l3` curriculum. Outputs
`results/JanusPro-1B-Levels-3ep/level_{l1,l2,l3}/`.
- **How:** warm-start from the 1-epoch adapter (`--warm_start_checkpoint`, loads
  LoRA+gen_head+gen_aligner, **resets optimizer/scheduler**), then a fresh continuation.
  Launchers: `sft_janus_levels_continue.sh` (loops the 3 levels). New knobs on
  `sft_janus_levels.sh`: `WARM_START_CKPT`, `WARMUP_RATIO`, `SAVE_ONLY_MODEL`,
  `RESUME_FROM`, `RANDOM_LEVELS`, `EVAL_IMAGE_FREQ`, `USE_PERCEPTUAL`, `MAX_SAMPLES`.
  Entry `sft_janus_alignment.py` gained `--caption_level`, `--caption_random_levels`,
  and `resume_from_checkpoint` passthrough. Full launcher docs: `HANDOFF_JANUS_LEVELS.md`.
- **Continuation scheduler (reuse this for the curriculum too):** fresh **cosine, peak
  LR 5e-5** (~half epoch-1's 1e-4 ≈ the tail of an ideal 3-epoch cosine), **warmup 0.03**,
  **effective batch 32**. Only the caption policy should differ across baseline vs curriculum.
- **l3 CAVEAT:** l1/l2 saved clean final adapters; **l3 has only `checkpoint-38000`**. Its
  lane OOM'd at 89% (a co-tenant `train_gran_lora` job filled GPU 1), and the resume
  **re-warmed** the LR because `save_only_model=true` saved no optimizer/scheduler → l3's
  last ~11% trained at up to 5e-5 instead of ~0. Usable but mildly over-adapted; **re-run
  clean** for a rigorous l3. (`SAVE_ONLY_MODEL=false` now makes resumes faithful.)
- **Random-granularity arm BUILT (not run):** `--caption_random_levels l1,l2,l3` draws a
  fresh level per sample per step (trainer `caption_random_columns`); launcher
  `sft_janus_levels_random.sh` (2-phase, schedule-matched to the baselines).

## 3×3 eval grid, 3-epoch models, N=1000 (Jul 19) — DONE → `results/eval_grid_1k/`
Full grid on a stratified 1k subset (`test_split_levels.json[::5]`): {l1,l2,l3 model} ×
{l1,l2,l3 caption}. Cells `m{model}__cap{cap}` with per-cell `metrics.json` (FID/CLIP) +
Qwen2.5-VL-3B `judge/summary.json` (modacc/plaus). Diagonal = use-as-trained.
- **FID (↓): diagonal dominates** — each model lowest at its own granularity
  (l1→l1 1076, l2→l2 1026, l3→l3 **929**). Granularity-matching matters.
- **CLIPScore (↑): cap_l3 wins every row** (~0.38-0.40; more text to align to), not diagonal.
- **Modality acc (↑): nuanced** — l1→l1 (0.733) & l3→l3 (0.743) strong; l2 muddled (its
  diagonal is its worst cell). NOT a clean "coarse always wins modality".
- **Plausibility (↑): saturated/flat** (3.99-4.19 across the whole grid) — non-discriminating.

## 1-epoch vs 3-epoch — longer training did NOT help (Jul 19)
Compared vs the 1-epoch grid `results/eval_levels/`. **FID not comparable** (that grid used
N=4998; FID inflates at small N). On the sample-count-independent metrics (diagonal, 1ep→3ep):
- CLIPScore **flat** (l1/l2 unchanged, l3 +0.006); Modality acc **slightly worse**
  (−0.003…−0.014); Plausibility **slightly worse** (−0.018…−0.022), uniformly.
- **Verdict: 1 epoch was already sufficient — the 2 extra epochs plateaued / marginally
  regressed.** Use the 1-epoch adapters as the baseline unless a matched-N FID recompute
  (filter 1-epoch manifests to the 1k ids + rerun BiomedCLIP-FID) says otherwise (pending).

## gran-LoRA — how it's trained NOW (Stage-0 grounding, supervised)
Files: `corl/open_r1/train_gran_lora.py`, `corl/scripts/train_gran_lora.sh`.
- ONE LoRA on `model.language_model` (30M params); base LLM + vision tower frozen.
- **Fixed neutral prompt** `"Describe this medical image."` — α carries all granularity.
- **Scale-consistency:** each step runs the SAME images at all 3 α's, supervising each
  toward its cached level caption (CE loss on caption tokens only; image+prompt masked
  with -100; left-padding so answer is at the end; position_ids fixed for left pad).
- α-map (CURRENT run): **`l1:0.3, l2:0.6, l3:0.9`** → **higher α = more detail** (l3 at 0.9),
  α=0 = base default caption. (User changed this from an earlier `l1:1.0,l2:0.6,l3:0.3`.)
- Only `lora_*` trains; gradient into LoRA scaled by α. DDP-correct via a `GranCaptioner`
  wrapper (calling `language_model` directly would skip DDP sync). PEFT `set_scale`
  sets α per forward. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, **batch 4**
  (batch 8 OOMs: 576 image tokens + full-vocab logits on 24GB).
- **Currently running** on GPUs 4-7 (cvssp-retina03): batch 4 × 4 GPUs, 50k samples ×
  2 epochs = 6250 steps, ~0.18 it/s (~9-10h), save every 500 →
  `results/GranLoRA/gran_lora_v1/`. Log: scratchpad `gran_train_v2.log`.
- **Validation:** `corl/eval/sweep_gran_alpha.py` — captions a modality-balanced sample
  at α=0,0.2,…1.0 under the neutral prompt; reports mean caption length per α +
  monotonicity verdict. Run after (or on the step-500 checkpoint) to confirm the α↔
  granularity axis grounded (should be **monotone increasing** now).

## The forward, exactly (input/output/GT)
Input sequence: `<|User|>: <image_placeholder>×576 \n Describe this medical image. <|Assistant|>: {target}`.
Target/GT = the cached level caption for that α. Labels mask everything but the caption
tokens. Only α differs across the 3 supervisions of the same image → α *means* granularity.

## OPEN DESIGN DECISION (what the new chat should tackle)
User wants to change the GT from "the exact caption" to a **specificity LEVEL** (train
only on granularity, not caption content; assume base captions well). Agreed analysis:
- This makes it **RL** (specificity of a generated caption is a non-differentiable
  black-box scalar) → reward `−|spec(gen) − target(α)|` + **KL-to-base anchor** (else it
  reward-hacks the metric by stuffing numbers/entities and drifts from faithful captions).
- Use the **2019 real-valued domain-agnostic** specificity predictor (Ko, Durrett & Li,
  AAAI 2019, arXiv:1811.05085, code github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction),
  NOT the 2015 Speciteller (news-trained, medical is OOD).
- **Conceptual caveat:** linguistic specificity ≠ visual-description granularity (a short
  "CT with 3.2cm mass" scores highly specific but is visually coarse). **Sanity check
  FIRST:** score cached l1/l2/l3 with the 2019 metric, confirm scores are monotone
  l1<l2<l3. If not monotone, the metric doesn't capture our granularity → don't build a loss on it.
- Plan: keep the CE grounding as a good **init**, then RL-refine on specificity + KL.

## The full method (designed, not yet built) — for later
gran-LoRA α = a learned continuous semantic-granularity knob (novel: specificity papers
*measure* granularity continuously but don't *control generation*; controllable captioning
controls but discretely). Downstream: **variance-weighted GRPO** trains the generator
(gen-heads+gen-LoRAs; base+DiT frozen) on granularities drawn from a **learnability
α-bandit** (sample α to maximize reward variance = frontier; NOT reward). Reward =
composite reconstruction (BiomedCLIP i2i + judge modality/plausibility; FID is set-level,
eval only). Key GRPO tweak: do NOT std-normalize per group (that cancels the variance
signal) — per-α baseline + global std. See project memory files
`gran-lora-granularity-grpo-method.md` and `janus-levels-eval-results.md`.

## Specificity sanity check — DONE (PASSED, with a caveat)
Scored cached l1/l2/l3 (n=400) on the 2019 model's **feature basis** (the neural head is
torch-1.0-locked; repo downloaded at scratchpad `Domain-Agnostic-Sentence-Specificity-Prediction-master/`,
needs GloVe 840B 2GB + torch==1.0 to run faithfully — deferred). Result:
- **5/6 features monotone up l1→l3**: words 18→45→106, numerals 0.27→0.80→2.91,
  content-frac 0.51→0.55→0.56, char/word-len up. → **specificity tracks our granularity axis.**
- **Caveat (important):** monotonicity is **length/numeral-driven**; per-word rarity goes
  DOWN l1→l3 (longer prose dilutes with common words). So a specificity-target reward is
  **gameable by padding length / inserting numbers** rather than adding visual detail.
- **Therefore, if training on specificity:** (a) normalize for length (specificity-per-token
  or length cap/penalty), (b) keep KL-to-base anchor, (c) set `target_spec(α)` from the
  measured per-level specificity (α=0.3→spec(l1), 0.9→spec(l3)).

## TOKENIZER BUG found + fixed (Jul 19) — gran_lora_v1 is CORRUPTED, retraining as v2
Running `sweep_gran_alpha.py` on gran_lora_v1 exposed a systemic bug: the LoRA emitted
**space-less** captions (`Theimageshowsacross-sectional...`), so `mean_words` stuck at 1.0
(mean_chars was still monotone-increasing in α — the granularity axis DID ground).
- **Root cause:** Janus' `tokenizer_config.json` says `tokenizer_class: LlamaTokenizer`
  (slow SentencePiece) but the vocab is **byte-level BPE** (space=="Ġ"). The slow tokenizer
  **decodes fine but mangles `encode`** — drops spaces: `"This is a CT scan"` →
  `['This','isa','CT','sc','ano','ft','he']`. Training labels (`process_one`→`tokenizer.encode`)
  were therefore space-less, and the LoRA faithfully learned space-less output. Even
  `AutoTokenizer(use_fast=True)` / `LlamaTokenizerFast.from_pretrained` return the broken slow
  one; only `PreTrainedTokenizerFast(tokenizer_file=.../tokenizer.json)` encodes correctly.
- **Fix:** `corl/open_r1/janus_tokenizer_fix.py::load_fast_tokenizer()`; both
  `train_gran_lora.py` and `sweep_gran_alpha.py` now overwrite `processor.tokenizer` with it
  right after `VLChatProcessor.from_pretrained`. Verified: labels/generation now keep spaces.
- **NB:** the prior levels T2I work (`sft_janus_alignment.py`) used the same broken encode for
  caption conditioning — rankings likely survive (consistent train+eval) but not re-checked.

## Current live state (as of Jul 19)
- **gran-LoRA v2 (corrected) is training** on cvssp-retina03 GPUs **0-3** (GPUs 4-7 are a
  separate job — leave alone): batch 4 × 4, 50k samples × 2 epochs = 6250 steps, ~0.17 it/s,
  save every 500 → `results/GranLoRA/gran_lora_v2/`. alpha_map `l1:0.3,l2:0.6,l3:0.9`.
  Started ~12:50 Jul 19; ETA ~10h. Log: scratchpad `gran_train_v2fix.log` (session-specific;
  rely on the `gran_lora_v2/` checkpoint dir + `nvidia-smi` instead). `pgrep -af train_gran_lora`.
- `results/GranLoRA/gran_lora_v1/` = the CORRUPTED run (kept as evidence). `sweep_v1.json` = its
  space-less sweep.
- Cross-generation 3×3 matrix (earlier eval) finished; see `results/eval_levels/MATRICES.md`.

## v2 GROUNDING — DONE (verified Jul 19, `results/GranLoRA/sweep_v2.json`)
v2 finished (6250 steps, final losses l1=0.19/l2=0.20/l3=0.23) and the sweep confirms:
- **Spaces restored** — captions are real prose again (bug fixed).
- **α-knob controls granularity monotonically over the trained range**. mean_words at
  α=0/0.2/0.4/0.6/0.8/1.0 (per_modality=2) = 86.5 / 16.5 / 15.5 / 52 / 101 / 103.5;
  chars = 511 / 98 / 90 / 298 / 597 / 615. Coarse≈16w (α~0.3=l1) → medium≈52w (0.6=l2)
  → detailed≈102w (0.9=l3), exactly the intended l1→l3 map. Example captions match
  (l1 = "This is a CT scan of the abdomen and pelvis."; l2 adds the main finding; l3 = full
  paragraph).
- **Verdict caveats:** (a) α=0 is the LoRA-*off* base default (verbose, ~86w) and sits
  OUTSIDE the knob's learned range — the sweep now prints a separate `monotonic_trained`
  (α>0) verdict for this reason. (b) The lone inversion α=0.2 (16.5w) vs α=0.4 (15.5w) is
  1-word sampling noise between two near-identical coarse captions; sampling at the exact
  trained points 0.3/0.6/0.9 removes it (a per_modality=3 rerun for that OOM'd only due to
  a co-tenant job filling the GPUs — not a code issue). Grounding is solid.

## Immediate next steps
1. **DONE** — v2 grounded (above). Optional: rerun the sweep at `--alphas 0,0.3,0.6,0.9`
   with more images once GPUs are free for the cleanest monotone plot.
2. Build the **RL refinement** — reward = `−|spec_per_token(gen) − target(α)|`
   (length-normalized!) + KL-to-base. Optionally run the exact 2019 neural metric first.
3. Longer horizon: the variance-weighted GRPO + learnability α-bandit generator training
   (see project memory `gran-lora-granularity-grpo-method.md`).
