# Eval data & "eval indexes" — handoff note

How the held-out T2I evaluation set is defined, where it lives, and how to get the
sample identities. **There is no separate index file — the eval set is identified
by the `id` field.**

## Files

| Path | Rows | Purpose |
|---|---|---|
| `corl/eval/test_split.json` | 4998 | Held-out T2I test set (full) |
| `corl/eval/test_split_small.json` | 600 | Smaller subset for quick eval |
| `corl/eval/build_test_split.py` | — | Script that generates the split |
| `corl/eval/generate.py` | — | Generates T2I outputs for every row in a split |

Each row:
```json
{
  "id": "Original_Caption_235407",
  "image": "images/pmc_235407_0.jpg",
  "modality": "Ultrasound",
  "body_part": "Abdomen",
  "Original_Caption": "..."
}
```

## How the split is built

`corl/eval/build_test_split.py` builds it from `PubMedVision_Original_Caption.json`:

- Drops `is_grid=='multi'` rows (same filter used at training time, via
  `data/attribute_sidecar.json`).
- **Stratified** across 6 modalities: Computed Tomography, Magnetic Resonance
  Imaging, Microscopy Images, Ultrasound, Endoscopy, Fundus Photography.
- `per_modality=833` (≈5000 total), round-robin within `body_part` so rare
  anatomies survive, `seed=0`.

Rebuild command:
```bash
python corl/eval/build_test_split.py \
    --data_dir /path/to/PubMedVision \
    --sidecar  data/attribute_sidecar.json \
    --out      corl/eval/test_split.json
```

## "Eval indexes" — it's id-based, not index-based

The eval set's identity is the **`id`** field. The same ids serve two roles:

1. **Train/test disjointness.** Training excludes these ids via
   `exclude_ids_json="corl/eval/test_split.json"`. The trainer
   (`corl/open_r1/sft_janus_alignment.py`) reads the list of dicts, pulls
   `r["id"]`, and drops those rows from the train set. So **eval ids ==
   the ids removed from training.**

2. **Evaluation.** `corl/eval/generate.py` loads `test_split.json` and iterates
   **every** row. It does not select a subset by index.

The **only** place a positional index is used is **sharding** for parallel
generation (`generate.py`):
```python
with open(args.test_split) as f:
    rows = json.load(f)
# shard across processes by position; coverage stays per-modality
rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
```
That splits the generation workload across `--num_shards` processes — it is not
sample selection.

## How to get the eval ids / indexes

```python
import json

rows = json.load(open("corl/eval/test_split.json"))

ids = [r["id"] for r in rows]      # eval-set identity == ids excluded from training
n   = len(rows)                    # 4998

# positional index i -> rows[i]
# generate.py sharding: process `shard` of `num_shards` handles rows where
#   i % num_shards == shard
```

To filter another file to the eval (or train) set by id:
```python
eval_ids = {r["id"] for r in json.load(open("corl/eval/test_split.json"))}
train_rows = [r for r in all_rows if r["id"] not in eval_ids]
eval_rows  = [r for r in all_rows if r["id"] in eval_ids]
```

## TL;DR

- Eval set = the **4998 ids** in `corl/eval/test_split.json` (or 600 in
  `test_split_small.json`).
- Those exact ids are **excluded from training** via `exclude_ids_json`.
- There is **no index list**; "indexing" only matters for sharding the
  generation workload in `generate.py` (`i % num_shards == shard`).
