# Handoff — Janus caption-granularity alignment SFT (l1 / l2 / l3)

Train Janus-Pro T2I alignment on captions of increasing granularity, one run per
level, to test whether simpler captions give a better-aligned generation baseline.

## What each level is

Built once by `corl/scripts/build_caption_cache_levels.py` into a single file with
four caption columns per image:

| Column                     | Granularity | Example |
|----------------------------|-------------|---------|
| `cached_captions_l1_meta`  | label       | `Computed Tomography, Abdomen` |
| `cached_captions_l1`       | 1 sentence, modality + region | `This is a CT scan of the chest, showing the heart and the pericardial fat pad.` |
| `cached_captions_l2`       | 1 sentence + main finding | `The image shows a cross-sectional view of the chest, highlighting the aorta … the pericardial fat pad is indicated…` |
| `cached_captions_l3`       | detailed paragraph | 2–3 sentences, all visible findings/locations/attributes |

`l1_meta` is a zero-noise templated label (from the dataset's `modality`/`body_part`);
`l1/l2/l3` are Janus-Pro-1B image-conditioned, greedy, k=1.

## Files

| File | Role |
|------|------|
| `corl/open_r1/sft_janus_alignment.py` | Training entry. New arg **`--caption_level {l1_meta,l1,l2,l3}`** → forces `caption_source=original` and reads `cached_captions_<level>`. |
| `corl/scripts/sft_janus_levels.sh` | Host-aware launcher (CVSSP / Isambard). Selects the level via `LEVEL=…`. |
| `corl/scripts/sft_janus_levels.sbatch` | Isambard SLURM wrapper; delegates to the `.sh`. |
| `PubMedVision_CachedCaptions_Levels.json` | The 636,082-row caption cache (all four columns). |

## Prerequisites (once)

1. **Get the cache onto the training host.** It was generated on CVSSP at
   `/work/um00109/MLLM/datasets/PubMedVision/PubMedVision_CachedCaptions_Levels.json`.
   Copy it to Isambard's data dir:
   ```bash
   # from CVSSP
   rsync -avP /work/um00109/MLLM/datasets/PubMedVision/PubMedVision_CachedCaptions_Levels.json \
       <isambard>:/projects/u6gd/datasets/PubMedVision/
   ```
2. **Sync the repo** on Isambard (`git pull` in `/projects/u6gd/umar/codes/ULM-R1`) so the
   new `--caption_level` arg and scripts are present.
3. Env is `corl` (already used for the cache build). The `.sh` activates it.

## Run — one submission per level (Isambard)

The launcher is parameterized; you don't need three separate files — just three submits:

```bash
sbatch --job-name=janus_l1 --export=ALL,LEVEL=l1 corl/scripts/sft_janus_levels.sbatch
sbatch --job-name=janus_l2 --export=ALL,LEVEL=l2 corl/scripts/sft_janus_levels.sbatch
sbatch --job-name=janus_l3 --export=ALL,LEVEL=l3 corl/scripts/sft_janus_levels.sbatch
```

(Optionally add the templated-label baseline: `LEVEL=l1_meta`.)

Each writes to its own dir: `results/JanusPro-1B-Levels/level_<LEVEL>/`.

### Local / interactive (CVSSP) alternative
```bash
LEVEL=l2 bash corl/scripts/sft_janus_levels.sh
```

## Knobs (env overrides, same on `.sh` and via `--export=ALL,VAR=…`)

| Var | Default (Isambard) | Notes |
|-----|--------------------|-------|
| `LEVEL` | `l1` | `l1_meta` / `l1` / `l2` / `l3` |
| `PER_DEVICE_BS` | `16` | GH200 has 96GB — safe to push to `32`. (24GB cards default to `8`.) |
| `NPROC` | `4` | Must match `#SBATCH --gpus`. |
| `MAX_STEPS` | `-1` (1 epoch) | Set an int to cap steps. |
| `LR` | `1e-4` | |
| `GRAD_ACC` | `1` | Effective batch = `NPROC × PER_DEVICE_BS × GRAD_ACC`. |
| `REPORT_TO` | `none` | `wandb` to log. |
| `DATASET_NAME` | `PubMedVision_CachedCaptions_Levels.json` | |
| `USE_PERCEPTUAL` | `false` | BiomedCLIP perceptual loss; adds a forward/step. |
| `EVAL_IMAGE_FREQ` | `100000000` (off) | In-training image previews OOM on shared 24GB cards; keep off, eval post-hoc. On GH200 set e.g. `500`. |
| `MAX_SAMPLES` | *(unset)* | Small int → smoke test. |

**Controls held constant across levels** (so the only variable is caption granularity):
LoRA r32/α64 on the LLM backbone, **perceptual loss off**, prototype & text-to-proto
conditioning **off**, in-training eval-image generation **off**, CFG prompt-dropout 0.1,
test split excluded (`corl/eval/test_split.json`) for disjoint eval. (Flip `USE_PERCEPTUAL=true`
if you want the perceptual term — just keep it consistent across all three levels.)

## Outputs, monitor, resume

- Checkpoints/adapters: `results/JanusPro-1B-Levels/level_<LEVEL>/` (a copy of the exact
  launch script is saved there as `run.sh`).
- Logs (Isambard): `corl/scripts/logs/<job-name>-<jobid>.out|.err`.
- Resume: relaunch the same job pointing `--output_dir`/`SAVE_PATH` at the existing dir
  (HF Trainer resumes from the latest checkpoint).

## Continue to 3-epoch per-level baselines (2 more epochs)

Each level trained **1 epoch** already (`results/JanusPro-1B-Levels/level_{l1,l2,l3}/`).
To get the per-level 3-epoch baselines (for comparison against a future
`l1 -> l2 -> l3` curriculum), warm-start from those adapters and run **2 more**:

```bash
bash corl/scripts/sft_janus_levels_continue.sh          # l1,l2,l3 sequentially
```

- Warm-starts each level from its own 1-epoch adapter (`--warm_start_checkpoint`),
  which loads LoRA+gen_head+gen_aligner but **resets the optimizer/scheduler**.
- Writes to a **new** dir `results/JanusPro-1B-Levels-3ep/level_<L>/` (the 1-epoch
  adapters are preserved untouched — they're also the curriculum's stage-1 start).

**Scheduler (why these settings):** epoch 1 ran a 1-epoch cosine `1e-4 -> ~0`, so its
schedule is spent. The continuation uses a **fresh cosine, peak `LR=5e-5`, `warmup 0.03`,
decayed to 0 over the 2 epochs**. That (a) refines rather than re-learns the adapted
LoRA, (b) ≈ the LR the tail of an ideal single 3-epoch cosine would pass through, and
(c) the short warmup smooths the jump from epoch-1's terminal LR≈0. **Reuse the identical
schedule for the curriculum's epochs 2-3** so baseline-vs-curriculum only differs in the
caption level per epoch. (Alternative peak `7.5e-5` if you'd rather match a 3-epoch
cosine tail more closely — just keep it the same across baseline and curriculum.)

New launcher knobs used here: `WARM_START_CKPT` (adapter dir), `WARMUP_RATIO`, `LR`,
`NUM_EPOCHS`, `SAVE_PATH`.

Isambard equivalent (separate chained jobs, one epoch-budget each):
```bash
j1=$(sbatch --parsable --export=ALL,LEVEL=l1,WARM_START_CKPT=$SRC/level_l1,NUM_EPOCHS=2,LR=5e-5,SAVE_DIR=$DST corl/scripts/sft_janus_levels.sbatch)
# ...l2 with --dependency=afterok:$j1, l3 with afterok:$j2
```

## Random-granularity arm (3 epochs, level sampled per step)

A third comparison point: instead of a fixed level or a curriculum, **each sample
independently draws one of {l1,l2,l3} every step**. Enabled trainer-side via the new
`--caption_random_levels l1,l2,l3` (added to `sft_janus_alignment.py` +
`sft_trainer_alignment.py`; takes precedence over `--caption_level`).

```bash
bash corl/scripts/sft_janus_levels_random.sh
```

**Schedule-matched to the baselines** (only the caption policy differs): it runs the
same two phases — phase 1 (epoch 1) from base Janus @`1e-4`, phase 2 (epochs 2-3)
warm-started @`5e-5`, both cosine/warmup 0.03 — but with a random level per step
throughout. Outputs the final adapter to `results/JanusPro-1B-Random-3ep/random/`.

### The three arms (all 3 epochs, identical schedule; only caption policy differs)
| Arm | epoch 1 | epoch 2 | epoch 3 | Launcher |
|-----|---------|---------|---------|----------|
| Fixed-level baseline `lX` | lX | lX | lX | `sft_janus_levels.sh` (e1, done) + `sft_janus_levels_continue.sh` (e2-3) |
| Curriculum | l1 | l2 | l3 | *(to scaffold)* |
| Random | rand | rand | rand | `sft_janus_levels_random.sh` |

Eval in random mode uses a **fixed** level (first of the list) for the periodic
in-training previews so they stay comparable step-to-step.

## Next step — evaluation (generation-only)

Compare the three checkpoints with the existing generation eval:
`corl/eval/generate.py` (loads/merges the LoRA adapter) → `corl/eval/compute_metrics.py`
(FID-BiomedCLIP, CLIPScore-BiomedCLIP, per-modality) and `corl/eval/llm_judge.py`
(per-modality accuracy + plausibility). Hypothesis: high-level captions (l1 / l1_meta)
win per-modality accuracy; the ladder shows the granularity trade-off.
