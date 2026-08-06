# gran-LoRA Training Paradigm

Reference: [rl_gran_lora.py](open_r1/rl_gran_lora.py), [gran_reward.py](open_r1/gran_reward.py), [HANDOFF_GRANULARITY_RL.md](../HANDOFF_GRANULARITY_RL.md).

## 1. Goal

Learn a **single scalar knob** $\alpha \in [0,1]$ that controls Janus-Pro's caption
granularity, with **no text supervision**:

- $\alpha = 1 \Rightarrow$ concise caption (~23 tokens)
- $\alpha = 0 \Rightarrow$ verbose caption (~137 tokens)
- monotonic interpolation between the two, faithfulness held by a KL anchor.

Only a **fresh rank-8 LoRA** on the language model trains; everything else frozen.
The knob is implemented as the LoRA's scaling coefficient — $\alpha=0$ literally
disables the adapter, so the model *is* the pretrained base.

## 2. Architecture — where $\alpha$ lives

```
                pretrained Janus-Pro-1B (frozen)
                   ┌────────────────────────┐
input  ─────────▶  │  base_layer (W_base)   │ ───┐
                   └────────────────────────┘    │
                                                 ▼
                                             (+) ─────▶ output
                                                 ▲
                   ┌────────────────────────┐    │
input  ─────────▶  │ LoRA:  B · A · x  · s  │ ───┘
                   └────────────────────────┘
                            s = α · (lora_α/r)   ← the KNOB
```

Injected on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` of
`janus.language_model` (only). Trainable params: **7,569,408** at rank 8.

**Per-sample scaling.** A batch of $B$ samples can carry $B$ different $\alpha$s
in one forward pass. Each `LoraLayer.forward` reads a stashed tensor
$\boldsymbol{\alpha} \in \mathbb{R}^B$:

$$
\text{out}_b \;=\; W_{\text{base}}\,x_b \;+\; \alpha_b \cdot \frac{\text{lora\_alpha}}{r} \cdot B\,A\,x_b
$$

This is what makes image-parallel batching possible ([rl_gran_lora.py:98-153](open_r1/rl_gran_lora.py#L98-L153)).

## 3. Reward — length target + repetition penalty

Length target interpolates **geometrically** in $\alpha$ (the measured levels
$\ell_1 \approx 23$, $\ell_3 \approx 137$ are near-geometric):

$$
\log \ell^\star(\alpha) \;=\; (1-\alpha)\,\log \ell_{\text{verbose}} \;+\; \alpha\,\log \ell_{\text{concise}}
$$

Per-completion reward is length-fit **in log-space** minus a repetition penalty:

$$
r(c;\alpha) \;=\; -\bigl|\log|c| - \log \ell^\star(\alpha)\bigr| \;-\; w_{\text{rep}} \cdot \bigl(1 - \text{distinct-3}(c)\bigr)
$$

- $|c|$ = completion token count.
- distinct-3 = fraction of unique 3-grams (word-level, tokenizer-free).
- $r_{\text{len}}$ is scale-free and symmetric; $r_{\text{rep}} \in [-1, 0]$ never rewards.
- **Faithfulness is not in the reward** — it is held by the KL anchor (§5).

## 4. Rollout — sample $K$ completions per image at $\alpha$

For each of the $B$ images in a step:

1. Sample $\alpha_b \sim \mathcal{U}(0,1)$.
2. Set per-sample LoRA scale to $\alpha_b$.
3. Sample $K$ captions from the policy (nucleus, $T=0.7$, $p=0.9$):
   $$ c_{b,1}, \ldots, c_{b,K} \sim \pi_\theta(\cdot \mid \text{img}_b, \alpha_b) $$

All $B \cdot K$ completions are produced in **one batched `generate()`**
([rl_gran_lora.py:126-153](open_r1/rl_gran_lora.py#L126-L153)).

## 5. GRPO advantage + KL-to-base loss

**Advantage** (within-image, mean-only baseline):

$$
A_{b,k} \;=\; r_{b,k} - \frac{1}{K}\sum_{j=1}^{K} r_{b,j}
$$

**Reference logprobs** are computed by the **same model** with per-sample scale
set to $0$ (LoRA off $\Rightarrow$ pretrained base) — no separate frozen copy needed:

$$
\log \pi_{\text{base}}(c_{b,k} \mid \text{img}_b) \;=\; \log \pi_\theta(c \mid \text{img}, \alpha{=}0)
$$

**Per-token loss** (policy gradient + $k_3$ KL estimator, per Schulman):

$$
\mathcal{L}_{b,k} \;=\; \frac{1}{|c_{b,k}|} \sum_t \Bigl[ -A_{b,k}\,\log\pi_\theta(c_{b,k,t}) \;+\; \beta \cdot \bigl( e^{\Delta_t} - \Delta_t - 1 \bigr) \Bigr]
$$

with $\Delta_t = \log\pi_{\text{base}}(c_{b,k,t}) - \log\pi_\theta(c_{b,k,t})$. The $k_3$
KL is nonnegative and unbiased when averaged over $\pi_\theta$-samples.

**Total loss** (averaged over the $N$ non-empty completions in the batch):

$$
\mathcal{L} \;=\; \frac{1}{N} \sum_{b,k} \mathcal{L}_{b,k}
$$

**Grad-parallel over B·K** — all reference logprobs and all policy logprobs are
each computed in **one batched forward** of shape $[N, P + L_{\max}, H]$,
producing a single autograd graph → single `.backward()`
([rl_gran_lora.py:283-317](open_r1/rl_gran_lora.py#L283-L317)).

## 6. Step loop

```
for step:
    ── data ─────────────────────────────────────────────────────────
    load B images from DataLoader
    sample α_b ~ U(0,1)  for b = 1..B

    ── rollout (inference, no grad) ─────────────────────────────────
    set per-sample α = α_b     (broadcast to B·K)
    generate B·K completions in ONE batched call
    reward r_{b,k}, advantage A_{b,k} = r_{b,k} - mean_k(r_{b,·})

    ── reference logprobs (inference, no grad) ──────────────────────
    set per-sample α = 0        (LoRA off, model = base)
    log π_base(c_{b,k})  for all N non-empty  ← ONE batched forward

    ── policy logprobs (grad on) ────────────────────────────────────
    set per-sample α = α_b
    [gradient checkpointing ON for this forward only]
    log π_θ(c_{b,k})     for all N non-empty  ← ONE batched forward

    ── loss + backward ──────────────────────────────────────────────
    L = mean_{b,k} [ -A_{b,k} log π_θ  +  β · k3_KL(log π_base, log π_θ) ]
    L.backward()

    ── DDP allreduce + step ─────────────────────────────────────────
    all-reduce LoRA grads across ranks (manual, we don't wrap in DDP)
    clip → AdamW → cosine LR schedule (with warmup)
```

## 7. Batched-forward shapes

Per step, per GPU (with `IPS=B=4`, `K=8`, `max_new_tokens=200`, prompt length $P$):

| Stage | Batch dim | Tensor | Shape |
|---|---|---|---|
| Prompt embeds | B | `emb, attn` | `[4, P, H]`, `[4, P]` |
| Rollout `generate()` | B·K = 32 | `inputs_embeds` (expanded) | `[32, P, H]` |
| Rollout output | B·K = 32 | `out` | `[32, ≤200]` |
| Ref/Pol forward | N (≤ B·K) | `[emb; tok_emb]` | `[N, P+L_max, H]` |
| Ref/Pol logits | N | `.logits` | `[N, P+L_max, V]` |

Per-sample LoRA scale is stashed as $[N]$ (or $[B{\cdot}K]$ for rollout) so
each row runs at its own $\alpha$ in the same forward.

## 8. Key hyperparameters (Isambard GH200 defaults)

| Symbol | Flag | Default | Meaning |
|---|---|---|---|
| $B$ | `IMAGES_PER_STEP` | 4 | images per optimizer step, per GPU |
| $K$ | `GROUP_SIZE` | 8 | completions per image (GRPO group) |
| — | `MAX_NEW_TOKENS` | 200 | rollout cap |
| $T$ | `TEMPERATURE` | 0.7 | 1.0 degenerates 1B base into gibberish |
| $p$ | `TOP_P` | 0.9 | nucleus truncation |
| $\beta$ | `BETA_KL` | 0.04 | KL-to-base weight |
| $w_{\text{rep}}$ | `W_REP` | 1.0 | repetition-penalty weight |
| $r$ | `LORA_R` | 8 | LoRA rank (knob is ~1-D) |
| lr | `LR` | 1e-5 | AdamW, wd=0 |
| $\ell_{\text{verbose}}$ | `L_VERBOSE` | 137 | anchor for $\alpha=0$ |
| $\ell_{\text{concise}}$ | `L_CONCISE` | 23 | anchor for $\alpha=1$ |

Effective per-step consumption on 4 GPUs: **$4 \times B = 16$ images**,
**$4 \times B \times K = 128$ rollouts**.
Full epoch on PubMedVision (~631k images) $\approx$ **39,443 steps**.

## 9. Validation (post-training)

Run [corl/eval/sweep_gran_alpha.py](eval/sweep_gran_alpha.py) on the saved adapter.
Expected: caption length varies smoothly and monotonically with $\alpha$; sampled
captions stay faithful (spot-check vs. the base model at $\alpha=0$).
