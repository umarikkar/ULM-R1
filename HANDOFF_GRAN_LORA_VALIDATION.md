# Handoff — gran-LoRA α-sweep validation

**Purpose:** validate that a trained gran-LoRA checkpoint has a monotonic
$\alpha \to $ caption-length relationship, and determine the usable α range.
This unblocks the next design (variance-weighted curriculum T2I; see below).

## What to check first

1. **Find any existing gran-LoRA checkpoint on this host.**
   ```bash
   find /vol/research/fmodel_medical/people/umar -maxdepth 6 -name "adapter_model.safetensors" 2>/dev/null | grep -i gran
   find ~ -maxdepth 6 -name "adapter_model.safetensors" 2>/dev/null | grep -i gran
   ```
   Expected locations (per [rl_gran_lora.sh:19](corl/scripts/rl_gran_lora.sh#L19)):
   `/vol/research/fmodel_medical/people/umar/MLMM/ULM-R1/results/GranLoRA/...`

2. **Also check the trainer's default OUT_DIR:**
   `$REPO/results/GranLoRA/gran_lora_rl_v1` on this host.

If nothing exists, stop here — nothing to sweep.

## How α was actually trained (important context)

- Training samples α *continuously* from $\mathcal{U}(0,1)$ per image per step
  ([rl_gran_lora.py:308](corl/open_r1/rl_gran_lora.py#L308)):
  ```python
  alpha = torch.rand(1).item()
  ```
  There is **no discrete grid** (no {0.3, 0.6, 0.9} or similar). The whole open
  interval $(0, 1)$ is training support.

- Length anchors ([gran_reward.py:28-29](corl/open_r1/gran_reward.py#L28-L29)):
  - `L_VERBOSE_DEFAULT = 137.0` → target at $\alpha = 0^+$
  - `L_CONCISE_DEFAULT = 23.0`  → target at $\alpha = 1$
  - Log-linear interpolation ([gran_reward.py:32-34](corl/open_r1/gran_reward.py#L32-L34)):
    $\log \ell^\star(\alpha) = (1 - \alpha) \log 137 + \alpha \log 23$.

- **α = 0 is a special discrete case**, not part of the trained knob range:
  at α = 0 the LoRA is off ($\text{scaling} = 0$), so the model behaves as the
  pretrained base. Used as the KL anchor during training.

- **Usable α range for downstream use: $(0, 1]$.**

## Running the sweep

Sweep script: [corl/eval/sweep_gran_alpha.py](corl/eval/sweep_gran_alpha.py).
Modality-balanced sample, greedy decode, mean-words + monotonicity verdict.

```bash
cd $REPO   # your ULM-R1 root on retina
python corl/eval/sweep_gran_alpha.py \
    --adapter_dir results/GranLoRA/<your_checkpoint>/ \
    --data_json $DATA_DIR/PubMedVision_CachedCaptions_Levels.json \
    --data_dir  $DATA_DIR \
    --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --per_modality 3 \
    --out results/GranLoRA/<your_checkpoint>/alpha_sweep.json
```

Fine-grained α grid because we don't yet know where the knob is smooth vs.
saturating.

## What to look for

Sweep prints one row per α:

```
 alpha mean_words mean_chars
  0.00       ...        ...   <- base captioner (LoRA off)
  0.10       ...        ...
  ...
  1.00       ...        ...
```

Then two verdicts:

- **`monotonic (all)`** — mean_words vs α across all queried αs. Expected:
  `monotone-decreasing` (higher α → shorter caption). But α=0 sits outside the
  trained range, so the `all` verdict may violate monotonicity even for a
  correctly trained knob.
- **`monotonic_trained (>0)`** — verdict over $\alpha > 0$ only. This is the
  real grounding check. Expected: `monotone-decreasing`.

**Passing criterion for downstream (variance-weighted T2I curriculum):**
- `monotonic_trained` is `monotone-decreasing`.
- `mean_words(α=1) ≲ 30` (near `L_CONCISE ≈ 23`).
- `mean_words(α≈0.1) ≳ 100` (near `L_VERBOSE ≈ 137`).
- Ideally 4–5 well-separated length buckets across the range (needed for the
  multi-level curriculum).

Spot-check the printed example captions per α:
- α close to 1 → 1-sentence concise caption, still faithful.
- α close to 0.1 → multi-sentence detailed caption, still faithful (no
  hallucinated findings).

## What to report back

Paste (or push into `results/GranLoRA/<ckpt>/alpha_sweep.json` + the printed
summary):

1. The table (α, mean_words, mean_chars).
2. Both monotonicity verdicts.
3. A few example captions per α (script prints one per α by default).
4. Which αs give distinct length buckets — this defines the discrete grid the
   curriculum pipeline will use.

## Why this matters — next-stage design (context)

We want to train a T2I model with GRPO under a **variance-weighted curriculum**:
at each step, produce prompts at multiple granularities $\{g_1, \ldots, g_M\}$,
run K-sample rollouts per level, and weight each level's GRPO loss by the
within-group reward std $\sigma_g$. This gives a self-adjusting difficulty
schedule with zero delay.

The rewriter that produces the M variants is initially just **gran-LoRA at M
discrete α values** — no new training needed for the rewriter side. The sweep
above tells us which αs to pick for the grid.

If gran-LoRA fails validation, plan B is a prompt-engineered LLM rewriter
(no training, ships immediately, weaker knob but usable).

## Session status (as of handoff)

- Original gran-LoRA training on Isambard: **stalled**. Two 12h submissions
  OOMed after a batched-forward refactor. Fixes applied (`logits_to_keep`,
  `GRAD_CKPT=1`), not yet re-tested.
- Isambard checkpoint at `results/GranLoRA/gran_lora_rl_v1/`: only 3 steps of
  training (smoke test), not usable for validation.
- Retina host may have a longer run — that's what this handoff is asking
  you to check.
