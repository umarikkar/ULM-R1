# Proposal: Variance-driven caption-granularity curriculum for T2I GRPO

Status: **design only — not implemented.** Pre-implementation literature review pending.
Owner: Umar. Date: 2026-06-26.

Related code: [grpo_trainer_unified.py](../open_r1/trainer/grpo_trainer_unified.py),
[rewards/__init__.py](../open_r1/rewards/__init__.py),
[sft_trainer_alignment.py](../open_r1/trainer/sft_trainer_alignment.py)
(`get_i2t_t2i_inputs` = the on-the-fly captioner pattern).
See also [debug_corl_sft_flow.md](debug_corl_sft_flow.md).

---

## 1. Motivation / observation

- Detailed GPT captions (`PubMedVision_Original_Caption.json`) train **slightly worse**
  than the simpler self-distilled captions (`PubMedVision_CachedCaptions_K4.json`).
- Two confounded reasons: detailed captions are (a) harder/more specific, and
  (b) **out-of-distribution** vs. the model's own phrasing. The cached captions
  are self-distilled → in-distribution → easier to fit.
- Hypothesis: a **coarse → fine** curriculum (general medical images from simple
  captions first, then add detail) should beat a fixed caption set.

## 2. Core idea

Borrow CGPO's *learnability* signal (group reward **variance** = the learnable
frontier) but make the **adaptive axis the granularity of the generated caption**,
not which prompt is sampled. The captioner is **steered by its instruction prompt**
(kept in `inference_mode` — it is a *controller*, not the RL policy). The GRPO
policy gradient still flows only through the T2I image-token generation.

Intended trajectory (emergent, not hand-scheduled):
1. **Early**: simple captions sit at the learnable frontier (high reward variance)
   → controller prompts the captioner for **simple** captions.
2. **Mid**: once simple captions are mastered, their variance collapses; the
   frontier slides to **medium** granularity → controller asks for more detail.
3. **Late**: frontier moves to fully detailed captions.

## 3. What already exists (extend, don't rebuild)

`grpo_trainer_unified.py` already provides the entire CGPO base:
- **Group rollouts**: `t2i_generate_parallel(..., parallel_size=num_generations)` → G images/prompt.
- **Reward models**: registry has `t2i_match_reward` (image–text sim), `t2i_clip_reward`,
  `t2i_pixel_mse_reward`, `T2ICycleConsistencyReward`.
- **The variance signal**: `t2i_std_grouped_rewards = t2i_rewards.view(-1, G).std(dim=1)`
  with `advantages = (r − mean)/(std+ε)` (~L633). This per-group std **is** the
  controller input.

## 4. New components (3)

1. **On-the-fly granularity-parameterized captioner.** Insert an i2t generation
   step (the `get_i2t_t2i_inputs` pattern) *before* `wrap_t2i_prompt`, parameterized
   by level `g → (instruction, max_new_tokens)`. Stays `inference_mode`.
2. **Granularity controller (bandit over g).** EMA of `t2i_std_grouped_rewards` per
   level → `V_g`. Each step pick `g` by sampling ∝ `V_g` (softmax / UCB). As low-`g`
   variance collapses, mass slides upward. Cold start needs optimistic init or
   ε-greedy/UCB exploration.
3. **Image-hardness sampler (literal CGPO).** Per-image EMA variance → sample which
   images to train on. *Defer to v2* — debug one adaptive signal at a time.

## 5. Suggested v1 (de-risked)

- Granularity axis **only** (image sampler deferred).
- 3–4 discrete levels (instruction detail + `max_new_tokens` ramp).
- Reward = `T2ICycleConsistencyReward` (image→recaption→compare; granularity-robust),
  optionally + `t2i_match_reward`.
- Controller = softmax over per-level EMA std, optimistic init + probability floor.
- **Log `V_g` and mean reward per level every step** so the frontier movement is visible.

## 6. Possible drawbacks / risks

- **Cross-granularity variance is not directly comparable (biggest risk).** Detailed
  captions are intrinsically harder to satisfy → may show structurally higher `V_g`
  *regardless of mastery*. The controller could jump to "detailed" prematurely, or
  never leave it. Mitigation: define "mastered" as **mean reward high AND variance
  falling**, or normalize each `V_g` to its own running baseline, rather than
  comparing levels in absolute terms.
- **Reward hacking / illusory advantage.** Scalar reward maximization (CLIP/match)
  is known to be hackable (cf. Pref-GRPO). Cycle-consistency partly guards against
  this but can be gamed by degenerate-but-recaptionable images.
- **Non-stationary captioner.** Captioner = the same Janus model; as GRPO updates the
  shared backbone/LoRA, the caption distribution drifts even at a fixed instruction.
  The "granularity level" is therefore only a soft control, not a fixed target.
- **Granularity control is fuzzy.** Prompt + token budget only *loosely* set detail;
  no guarantee level-2 captions are strictly more detailed than level-1.
- **Variance estimate is noisy at batch size 1 / small G.** `V_g` needs enough
  revisits per level; EMA smoothing is essential, and the bandit must not starve
  rarely-picked levels.
- **Compute.** Per step = i2t caption gen (≈256 tok) + G T2I rollouts (G×576 tok) +
  G reward passes. Heavy; already the GRPO cost, controller is cheap on top.
- **Emergent ≠ controllable.** The curriculum is only as good as the variance signal;
  if §6.1 bias dominates, the "coarse→fine" story won't materialize and it degrades
  to plain GRPO at a stuck granularity.
- **Confound not resolved.** If the cached-caption advantage is mostly
  *in-distribution-ness* rather than *simplicity*, a granularity curriculum may not
  capture the real gain. Worth an ablation isolating the two.

## 7. Open decisions (pick before coding)

- Mastery/move-on signal: raw `V_g` vs. (mean↑ & var↓) vs. per-level normalized var.
- Discrete levels vs. continuous token-budget dial.
- Reward: cycle-consistency only vs. + match vs. pairwise-preference (Pref-GRPO style).
- Whether to ever train the captioner on granularity (RL on the i2t side) or keep it
  a frozen prompt-steered controller (current plan: the latter).

---

## 8. Reading list (review before implementing)

Verify arXiv ids before formal citation; 2026 ids are very recent.

### A. RL / learning over the *conditioning prompt* (most on-target)
- **Promptist — "Optimizing Prompts for Text-to-Image Generation"**, Hao et al.,
  arXiv:2212.09611. SFT then RL to rewrite prompts to maximize a reward. The
  canonical "learn the conditioning" precedent.
- **PromptEnhancer**, arXiv:2509.04545. CoT prompt rewriter trained via RL with a
  fine-grained reward model (AlignEvaluator). Closest recent analog to a learned
  caption controller.
- **RePrompt: Reasoning-Augmented Reprompting via RL**, arXiv:2505.17540.
- **Self-Rewarding LVLMs for Optimizing Prompts in T2I**, arXiv:2505.16763.
- **Input-Side Inference-Time Scaling for T2I**, arXiv:2510.12041.

### B. Difficulty-adaptive / bandit curriculum in RL (the variance-frontier core)
- **CGPO — Curriculum Group Policy Optimization** (the paper you shared),
  arXiv:2605.17807. Reward-variance = learnability; adaptive prompt sampling.
- **DAPO** (open-source RL system), *dynamic sampling* drops zero-variance groups
  (all-pass/all-fail) — the exact reward-variance criterion. (verify id, ~2503.14476.)
- **Graves et al. 2017 — Automated Curriculum Learning for Neural Networks**,
  arXiv:1704.03003. Multi-armed bandit choosing task/difficulty by *learning
  progress*. Classical analog of the granularity bandit — read this.
- **Actor-Curator: Co-adaptive Curriculum via Policy-Improvement Bandits**,
  arXiv:2602.20532. No difficulty labels / manual structuring.
- **Manifold Bandits: Bayesian Curriculum over LLM Latent Geometry**,
  arXiv:2606.19750. Notes sampling decisions *steer* how learning evolves
  (endogenous non-stationarity) — relevant caution for our feedback loop.
- **SPARD: Self-Paced Curriculum for RL Alignment via Reward Dynamics + Data Utility**,
  arXiv:2604.07837.
- **Adaptive Curriculum Learning for RLHF** (MAB, clusters as arms),
  OpenReview 8HvWBamUkS.

### C. Curriculum / learnability foundations
- **Bengio et al. 2009 — Curriculum Learning**, ICML. Foundational.
- **Kumar et al. 2010 — Self-Paced Learning for Latent Variable Models**, NeurIPS.
- **Mindermann et al. 2022 — RHO-Loss: Prioritized Training on Points that are
  Learnable, Worth Learning, and Not Yet Learnt**, arXiv:2206.07137. Selects
  *learnable* (not too easy/hard) points — SFT analog of the variance criterion.

### D. Caption detail / recaptioning for T2I (explains the simple-vs-detailed result)
- **DALL·E 3 — "Improving Image Generation with Better Captions"**, Betker et al.,
  2023 (OpenAI). Highly descriptive synthetic captions improve T2I; bears directly
  on the detailed-vs-simple tradeoff and recaptioning.
- **PixArt-α**, Chen et al., arXiv:2310.00426. Dense pseudo-captions for T2I training.

### E. Background — RL fine-tuning of the image generator (not our axis, for context)
- **DDPO**, Black et al., arXiv:2305.13301; **DPOK**, Fan et al., arXiv:2305.16381;
  **ImageReward / ReFL**, Xu et al., arXiv:2304.05977.
- **Flow-GRPO** (OpenReview oCBKGw5HNf), **DanceGRPO**, **T2I-R1** (NeurIPS 2025),
  **Pref-GRPO** (arXiv:2508.20751, reward-hacking caution).
