# Handoff — gran-LoRA RL refinement (research doc)

Scope: **only the RL stage** that refines the gran-LoRA granularity knob on a
*specificity-target* reward. A **first RL loop is now built** (GRPO on a validated
*length*-target reward — `corl/open_r1/rl_gran_lora.py`, launch recipe in **§6a**,
runs on **Isambard**); the rest of the doc is still the research agenda for the
*reward metric* — the open question is whether a **specificity/visual-grounding**
reward can replace the length proxy. For the supervised grounding that precedes
this, and the broader project, see `HANDOFF_GRANULARITY.md`.

---

## 0. TL;DR of where we are
- **Stage-0 (supervised grounding) is DONE and verified.** One LoRA ("gran-LoRA")
  on the Janus-Pro-1B captioner; its **scale α** controls caption granularity under
  a fixed neutral prompt. Grounded init = `results/GranLoRA/gran_lora_v2/`.
- **Stage-1 (this doc) = RL refinement.** Idea: stop supervising the *exact* caption
  text and instead reward the generated caption for hitting a **target specificity
  level** set by α — so α becomes a clean continuous granularity knob, decoupled from
  memorising specific caption content. Requires RL because "specificity of a generated
  caption" is a non-differentiable black-box scalar.
- **RL algorithm: decided and built.** GRPO (mean baseline, no per-group std-norm by
  default), KL-to-base via a scale-0 forward, fresh rank-8 LoRA, α~U(0,1) per step.
  `corl/open_r1/rl_gran_lora.py` + `corl/scripts/rl_gran_lora.{sh,sbatch}`.
  **Built but not yet run** — first job goes to **Isambard** (§6a), *not* retina03.
- **Still open: the reward metric.** The built loop uses a **length**-target proxy
  (`gran_reward.py`). Whether a real specificity measure tracks *visual* granularity
  rather than just length is the unresolved research question (§3) — it is the natural
  drop-in upgrade for the reward once §3c passes.

---

## 1. The granularity levels (l1/l2/l3) and the α knob — core reference

**Leveled caption cache:** `PubMedVision_CachedCaptions_Levels.json` (636,082 rows),
data dir `/work/um00109/MLLM/datasets/PubMedVision`. Per image, 4 caption columns,
all generated image-conditioned by Janus-Pro-1B (greedy):

| level | column | what it is | typical length (words) |
|-------|--------|------------|------------------------|
| l1_meta | `cached_captions_l1_meta` | templated `modality, body_part` label (zero noise) | ~5 |
| **l1** | `cached_captions_l1` | 1 sentence: modality + region ("This is a CT scan of the chest…") | ~18 |
| **l2** | `cached_captions_l2` | 1 sentence + main finding | ~45 |
| **l3** | `cached_captions_l3` | detailed 2–3 sentence paragraph | ~106 |

**The α-map (trained into gran_lora_v2):** `l1:0.3, l2:0.6, l3:0.9` →
**higher α = more detail**. α=0 = LoRA off = base model's default caption (verbose,
~86 words; NOT part of the knob's learned range — treat α∈[0.3,0.9] as the operating band).

**Grounding proof** (`sweep_gran_alpha.py` on gran_lora_v2, `results/GranLoRA/sweep_v2.json`):
mean_words at α=0/0.2/0.4/0.6/0.8/1.0 = 86.5 / 16.5 / 15.5 / 52 / 101 / 103.5. Monotone
over the trained band; example captions match the level definitions (l1 coarse modality+region,
l2 adds finding, l3 full paragraph). **So α already controls granularity** — RL is about
making that control *robust and content-decoupled*, and extending it to a continuous target
rather than 3 discrete anchors.

**Fixed neutral prompt** (all α share it): `"Describe this medical image."` — α carries
all the granularity signal.

---

## 2. The RL problem statement

Sample a caption `c ~ π_α(· | image, neutral_prompt)` from gran-LoRA at scale α. We want
its **specificity** to match a **target** set by α: `target_spec(α)`, anchored so
`target_spec(0.3)=spec(l1)`, `target_spec(0.6)=spec(l2)`, `target_spec(0.9)=spec(l3)`
(interpolate between). Reward (proposed):

```
r(c, α) = − | spec_norm(c) − target_spec(α) |   +   β · (KL-to-base anchor, per token)
```

- **Why the target, not "maximize detail":** we want a *knob*, not just "be verbose".
  The reward is a distance-to-target (band-pass), zero at the desired level.
- **Why the KL-to-base anchor (critical):** without it the policy reward-hacks the metric
  — stuffing numbers / entities / padding to move the scalar — and drifts from *faithful*
  captions. KL to the frozen base captioner (or to the grounded gran_lora_v2 init) keeps
  captions on-manifold. Tuning β is a core experiment.
- **Init = gran_lora_v2** (already granularity-aware) → RL is *refinement*, short horizon.

---

## 3. Metric research — the central open question

**We must NOT build a loss on a metric that doesn't track *visual* granularity.**

### 3a. What we already know (specificity sanity check — DONE, PASSED w/ caveat)
Scored cached l1/l2/l3 (n=400) on the 2019 predictor's **feature basis** (not the neural
head — see 3b). Findings:
- **5/6 features monotone up l1→l3**: words 18→45→106, numerals 0.27→0.80→2.91,
  content-word-fraction 0.51→0.55→0.56, char/word-length up. → specificity *does* track
  our granularity axis at the corpus level.
- **CAVEAT (the whole ballgame):** the monotonicity is **length/numeral-driven**. Per-word
  *rarity* actually goes **DOWN** l1→l3 (longer prose dilutes with common words). So a raw
  specificity-target reward is **gameable by padding length / inserting numbers** rather
  than adding genuine visual detail.
- **Implications for the reward:** (a) **length-normalize** (specificity-per-token, or a
  hard length cap / penalty), (b) keep the **KL anchor**, (c) set `target_spec(α)` from the
  measured per-level specificity.

### 3b. Candidate metrics to evaluate (research task)
1. **2019 domain-agnostic specificity predictor** (Ko, Durrett & Li, AAAI 2019,
   arXiv:1811.05085; code `github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction`).
   Real-valued [0,1], semi-supervised, designed to transfer across domains (better for
   medical than the 2015 Speciteller, which is news-trained → OOD here).
   - **Blocker:** the neural head is **torch==1.0-locked**; needs **GloVe 840B (2 GB)**.
     Repo already downloaded at scratchpad
     `Domain-Agnostic-Sentence-Specificity-Prediction-master/` (scratchpad is session-scoped
     — may need re-download). **Research task:** either stand up a torch==1.0 conda env to
     run the pretrained head faithfully, OR reimplement/port the head to modern torch, OR
     retrain a small head on its features. Decide which.
2. **2015 Speciteller** — simpler, but news-trained; likely OOD for radiology. Baseline/sanity only.
3. **Length-normalized lexical proxies** — content-word count per token, type-token ratio,
   numeral/entity density, mean word rarity (IDF from the corpus). Cheap, CPU-only, fully
   controllable. **Must validate they separate l1/l2/l3** before use.
4. **Visual-grounding metrics (the "right" axis, arguably):** the caveat is that *linguistic*
   specificity ≠ *visual-description* granularity (a short "CT with 3.2 cm mass" is
   linguistically specific but visually coarse). Consider rewarding **visual** granularity
   directly instead of/alongside specificity:
   - BiomedCLIP / CLIP **image–text alignment** (does more detail = more image-grounded?).
   - A **VQA / entailment judge** (Qwen2.5-VL-3B already used elsewhere in repo) counting
     *correct, image-supported* findings — penalises hallucinated padding, which pure
     specificity does not.
   - **Research question:** is our target "linguistic specificity" or "count of faithful
     visual details"? These diverge; pick deliberately. A composite (specificity for the
     *level* + a faithfulness/grounding term to stop hacking) is likely.

### 3c. Metric deliverable before any RL
A `spec(text) -> float` scorer + a short report proving it is **monotone l1<l2<l3** AND
**not trivially gameable by length alone** (e.g. show it doesn't jump when you pad an l1
caption with filler). If no metric passes both, the specificity-reward plan is wrong and we
should reward visual grounding instead.

---

## 4. RL algorithm research — how to actually train it

Design axes to research and decide:

- **Algorithm.** Options, roughly increasing complexity:
  - **REINFORCE / RLOO / best-of-N with a baseline** — simplest; sample K captions per
    (image, α), reward each, subtract a baseline, policy-gradient. Often enough for
    short-horizon LLM reward tuning; no value net.
  - **GRPO** (already used elsewhere in this repo via `trl`) — group-relative; sample a
    group per prompt, normalise within group. **Caveat for us:** if a future stage wants a
    *variance* signal across α (see the generator-training design in the main handoff), do
    **not** std-normalise per group (it cancels variance) — use per-α baseline + global std.
    For *this* stage (per-α distance-to-target) standard GRPO is fine.
  - **PPO** — most control, most machinery (value head, clipping). Probably overkill for a
    30M-param LoRA refinement.
  - **Research task:** pick the lightest algorithm that trains stably. Start with
    REINFORCE-with-baseline or GRPO.
- **KL anchor.** To base captioner or to gran_lora_v2 init? Per-token KL penalty vs a
  KL-in-reward (trl style). Tune **β**; sweep to find the point where captions stay faithful
  but α still moves granularity.
- **Sampling at a given α.** gran-LoRA scale is set per forward by scaling every LoRA layer
  (`set_scale`/`set_gran_scale` in `train_gran_lora.py` / `sweep_gran_alpha.py`). RL rollout
  = set α, sample (do_sample=True, temperature) K captions, score, update *only when LoRA is
  at that α* (gradient already scales by α). **Which α's per step?** Either the 3 anchors, or
  sample α∈[0.3,0.9] continuously (tests interpolation / continuous knob claim).
- **On/off-policy & length control.** Length cap in the sampler + length penalty in reward
  (from 3a). Watch for reward-hacking (degenerate repetition, number stuffing) — log sample
  captions every N steps.
- **What trains / stays frozen.** Only `lora_*` on `model.language_model` (30M). Base LLM,
  vision tower, gen heads all frozen. Same wrapper/DDP setup as Stage-0.
- **Compute — run this on Isambard (GH200 96 GB), not cvssp-retina03.** retina03 is
  contended by other users; treat it as the fallback/debug box only. The GH200's 96 GB also
  removes the memory pressure that shaped the 24 GB defaults: RL adds K samples + a reference
  forward for KL on top of 576 image tokens + full-vocab logits, which is exactly what a
  24 GB card struggles with. See **§6a** for the launch recipe.
  Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on either host.

### The novelty framing (worth preserving)
gran-LoRA α = a **learned continuous semantic-granularity knob**. Specificity papers
*measure* granularity continuously but don't *control generation*; controllable-captioning
work controls but *discretely*. A continuous, RL-refined knob is the contribution.

---

## 5. Reading list
- **Ko, Durrett, Li 2019** — Domain-Agnostic Sentence Specificity Prediction, AAAI
  (arXiv:1811.05085). The candidate metric.
- **Louis & Nenkova 2011 / Speciteller 2015** — original sentence specificity (news). Baseline.
- **RLHF / reward-target LLM tuning:** InstructGPT (Ouyang 2022) for KL-anchored PG;
  **RLOO** (Ahmadian 2024) and **GRPO** (DeepSeekMath, Shao 2024) for lightweight, value-free
  PG — likely our starting algorithms.
- **Controllable captioning** (length/detail control, e.g. length-controllable / abstraction
  control) — for the "controls discretely" contrast and for length-normalisation ideas.
- **Faithfulness/hallucination in medical captioning / CLIPScore & BiomedCLIP** — for the
  visual-grounding reward alternative (§3b.4) and the anti-hacking term.
- In-repo prior art: `results/eval_levels/`, `results/eval_grid_1k/` (Qwen judge for
  modality-accuracy/plausibility, BiomedCLIP-FID, CLIPScore) — reusable reward components.

---

## 6. Repo / env / concrete hooks
- **Hosts.** Default training host is **Isambard** (`u6gd`, GH200 96 GB): repo
  `/projects/u6gd/umar/codes/ULM-R1`, data `/projects/u6gd/datasets/PubMedVision`, conda base
  `/projects/u6gd/umar/miniconda3`. CVSSP `cvssp-retina03` (8×3090 24 GB, data
  `/work/um00109/MLLM/datasets/PubMedVision`) is **fallback/debug only — it is usually
  contended.** Both are handled automatically by the launcher's `hostname -s` switch.
- Env **`corl`** on both (`source <conda-base>/etc/profile.d/conda.sh; conda activate corl`).
  torch 2.6+cu124, transformers 5.4, **trl 0.18.1**, peft 0.18.1.
- **Grounded init:** `results/GranLoRA/gran_lora_v2/` (adapter on `model.language_model`).
- **How to sample at α (reuse):** `corl/eval/sweep_gran_alpha.py` — loads Janus +
  gran-LoRA, `set_scale(model, base, α)`, `prepare_inputs_embeds`, `language_model.generate`.
  Copy this rollout path for the RL sampler.
- **Trainer to fork for the loop:** `corl/open_r1/train_gran_lora.py` (`GranCaptioner`
  wrapper, `set_gran_scale`, left-pad label logic) — swap the CE loss for the PG update.
- **TOKENIZER GOTCHA (do not reintroduce):** Janus' declared slow `LlamaTokenizer` **mangles
  `encode`** (drops spaces on this byte-level-BPE vocab), which silently corrupted the first
  grounding run. Any text you tokenize as a reward input / prompt / target must go through
  `corl/open_r1/janus_tokenizer_fix.py::load_fast_tokenizer` (already wired into the trainer
  and sweep). Decode is fine; **encode is the trap.**
- **Data / test split:** cache above; held-out `corl/eval/test_split.json` (4998 rows) /
  `corl/eval/test_split_levels.json` (rows + level captions) — use for reward validation and
  eval, exclude from RL training (`--exclude_ids_json`).

---

---

## 6a. Running the RL job on Isambard

The Stage-1 loop is built: `corl/open_r1/rl_gran_lora.py` (GRPO, length-target reward,
fresh rank-8 LoRA, KL-to-base via a scale-0 forward). Launchers:

| File | Role |
|------|------|
| `corl/scripts/rl_gran_lora.sh` | Host-aware launcher (Isambard / retina03 / ulws072): conda, paths, `torchrun`. |
| `corl/scripts/rl_gran_lora.sbatch` | Isambard SLURM wrapper; delegates to the `.sh`. |

### Prerequisites (check before submitting)
1. **Repo synced:** `git pull` in `/projects/u6gd/umar/codes/ULM-R1` so `rl_gran_lora.py`,
   `gran_reward.py`, `janus_tokenizer_fix.py` and both launchers are present.
2. **Caption cache present** at `/projects/u6gd/datasets/PubMedVision/PubMedVision_CachedCaptions_Levels.json`
   (already rsynced for the levels runs — see `HANDOFF_JANUS_LEVELS.md` §Prerequisites; re-rsync
   from CVSSP if missing) plus its `images/` root.
3. **Janus-Pro-1B weights cached locally.** Default `--model deepseek-ai/Janus-Pro-1B` pulls from
   the HF hub; compute nodes may have no internet. Warm the cache on the **login** node first
   (or point `MODEL=` at a local dir), then `export HF_HUB_OFFLINE=1` in the job.
4. **No Stage-0 adapter needed** — this RL stage trains a *fresh* rank-8 LoRA, so
   `gran_lora_v2/` does **not** have to be copied to Isambard. (You only need it there if you
   run the §7.5 comparison eval on the same host.)

### Submit
```bash
cd /projects/u6gd/umar/codes/ULM-R1
sbatch corl/scripts/rl_gran_lora.sbatch                                  # GH200 defaults
sbatch --export=ALL,MAX_STEPS=32,MAX_SAMPLES=512 corl/scripts/rl_gran_lora.sbatch    # smoke
sbatch --export=ALL,BETA_KL=0.02,MAX_STEPS=1000 corl/scripts/rl_gran_lora.sbatch     # β-sweep point
```
Logs: `corl/scripts/logs/<job-name>-<jobid>.out|.err`. Output adapter + a copy of the exact
launch script: `results/GranLoRA/gran_lora_rl_v1/` (override with `OUT_DIR=`).

### GH200 defaults vs the 24 GB defaults

| Var | Isambard (`.sbatch`) | retina03 (`.sh`) | Notes |
|-----|----------------------|------------------|-------|
| `NPROC` | `4` | `8` | Must match `#SBATCH --gpus`. |
| `GROUP_SIZE` (K) | `8` | `4` | Completions per image; bigger K = lower-variance GRPO advantage. |
| `IMAGES_PER_STEP` | `4` | `2` | Grad-accum images per GPU. |
| `MAX_STEPS` | `1000` | `-1` (1 epoch) | See note below — this is the **LR horizon**, not just a cap. |
| `SAVE_STEPS` | `250` | `500` | |
| `BETA_KL` | `0.04` | `0.04` | The β to sweep (§2). |
| `TEMPERATURE` | `0.7` | `0.7` | 1.0 degenerates the 1B base into gibberish — don't raise it. |
| `MAX_NEW_TOKENS` | `200` | `200` | Also the de-facto length cap for the reward. |

Effective rollouts/step = `NPROC × IMAGES_PER_STEP × GROUP_SIZE` (= 128 at GH200 defaults).
Start with the smoke submit above and read the logged sample captions before burning a full run —
that is the reward-hacking tripwire (degenerate repetition, number stuffing).

**`MAX_STEPS` sets the LR schedule, so choose it deliberately.** `rl_gran_lora.py:282` uses it as
`total_steps` for the warmup+cosine schedule. Consequences:
- With `-1` the cosine is sized for a **full epoch ≈ 39,400 steps** (631k images ÷ 4 GPUs ÷ 4 per
  step). A job that hits the wall clock before then trains at near-peak LR throughout and
  **never anneals** — and there is no resume path, so that run is largely wasted.
- Set `MAX_STEPS` to what actually fits in `--time`. The `1000` default is an **unvalidated
  starting guess** (~2.5% of an epoch, 16k images, ~128k rollouts at K=8) — plausible for GRPO
  refinement of a rank-8 LoRA on a dense per-sample reward, but **not derived from a measured
  step time or a convergence curve.** Re-derive it from the smoke run and revise this line.
- `MAX_SAMPLES` independently caps epoch length: steps/epoch = `MAX_SAMPLES ÷ (NPROC × IMAGES_PER_STEP)`.
  Keep the two consistent or the loop exits early with a mis-sized cosine (the smoke submit above
  is tuned so 512 samples = exactly 32 steps).

**Wall-clock caveat:** each step generates `IMAGES_PER_STEP × GROUP_SIZE` captions of up to 200
tokens *plus* a reference forward, so steps are far slower than Stage-0 SFT steps. The 12 h
`--time` in the `.sbatch` is a guess — time the smoke run and adjust before the real submit.

---

## 7. Concrete next steps (in order)

**Track A — get the built length-target loop running (do this first; it is unblocked).**
1. **Prereq check on Isambard** (§6a): repo pulled, caption cache + images present, Janus-Pro-1B
   warmed into the HF cache from the login node.
2. **Smoke submit:** `sbatch --export=ALL,MAX_STEPS=200,MAX_SAMPLES=512 corl/scripts/rl_gran_lora.sbatch`.
   Read the logged sample captions — check reward spread is non-degenerate and watch for
   repetition / number stuffing. Time it, then fix the real run's `--time`.
3. **Real run + β-sweep** (`BETA_KL` ≈ 0.01 / 0.04 / 0.1): find where captions stay faithful but
   α still moves granularity.
4. **Eval:** rerun `corl/eval/sweep_gran_alpha.py` on the saved adapter (does α move length
   monotonically, more robustly than supervised?) + faithfulness check (Qwen judge / BiomedCLIP)
   vs the `gran_lora_v2` supervised baseline.

**Track B — upgrade the reward from length to specificity (the research question, §3).**
5. **Metric (blocking for Track B).** Stand up a `spec(text)->float` scorer (decide: port the 2019
   neural head to modern torch, run it in a torch==1.0 env, or a validated length-normalized
   proxy). Prove monotone l1<l2<l3 **and** not length-gameable (pad-an-l1 test). Save the
   per-level `spec(l1/l2/l3)` values → these define `target_spec(α)`.
6. **Decide the reward:** pure specificity-distance vs specificity + visual-grounding/faithfulness
   term. Fix length normalization + the KL anchor target (base vs v2).
7. **Swap it into the loop:** replace the length target in `corl/open_r1/gran_reward.py` — the RL
   loop itself (rollout, GRPO, KL) needs no change — and rerun Track A steps 2–4 to compare.