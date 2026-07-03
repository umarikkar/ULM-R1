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

**Controls held constant across levels** (so the only variable is caption granularity):
LoRA r32/α64 on the LLM backbone, perceptual loss on (w=0.5, layers 3/6/9), prototype &
text-to-proto conditioning **off**, CFG prompt-dropout 0.1, test split excluded
(`corl/eval/test_split.json`) for disjoint eval.

## Outputs, monitor, resume

- Checkpoints/adapters: `results/JanusPro-1B-Levels/level_<LEVEL>/` (a copy of the exact
  launch script is saved there as `run.sh`).
- Logs (Isambard): `corl/scripts/logs/<job-name>-<jobid>.out|.err`.
- Resume: relaunch the same job pointing `--output_dir`/`SAVE_PATH` at the existing dir
  (HF Trainer resumes from the latest checkpoint).

## Next step — evaluation (generation-only)

Compare the three checkpoints with the existing generation eval:
`corl/eval/generate.py` (loads/merges the LoRA adapter) → `corl/eval/compute_metrics.py`
(FID-BiomedCLIP, CLIPScore-BiomedCLIP, per-modality) and `corl/eval/llm_judge.py`
(per-modality accuracy + plausibility). Hypothesis: high-level captions (l1 / l1_meta)
win per-modality accuracy; the ladder shows the granularity trade-off.
