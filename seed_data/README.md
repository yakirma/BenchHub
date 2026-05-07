# Batch dataset seeding

Use `scripts/seed_datasets.py` to import a curated list of HF datasets
in one shot — each entry runs through the same auto-import + auto-LB
flow the UI exposes, but without any clicking.

## Configs

One JSON file per domain. Each entry is one HF repo to import:

```json
{
  "hf_repo_id": "stanfordnlp/sst2",
  "dataset_name": "sst2",
  "sample_cap": 500,
  "revision": null,
  "auto_create_lb": true,
  "lb_name": "SST-2 sentiment"
}
```

The starter configs in this directory are **suggestions**, not a
verified manifest. HF repos move, get gated, or have their schema
change — verify each repo loads with `_hf_fetch_features` before
running a 10-shot batch.

## Running

```bash
# Dry-run: print the inferred mapping for every entry, don't import.
python scripts/seed_datasets.py seed_data/llm.json \
    --owner-email you@example.com --dry-run

# For real. Use --skip-existing to make the script re-runnable.
python scripts/seed_datasets.py seed_data/depth.json \
    --owner-email you@example.com --skip-existing
```

The owner's saved HF token (set via `/settings/hf_token`) is used for
gated repos. Set `HF_TOKEN` in the environment as a fallback.

## What you get per entry

1. A `Dataset` row imported from HF, samples capped at `sample_cap`.
2. ClassLabel sidecars (`<col>_class` text + per-sample tags).
3. A `Leaderboard` (when `auto_create_lb=true`) with auto-proposed
   metrics + visualizations attached. The proposer's defaults are
   *baselines* — replace with task-correct metrics (IoU for seg,
   PSNR for denoising, F1 for QA) before treating the LB as a real
   benchmark.

## Failure modes the script handles

- HF gating / 401 → logs and skips, batch continues.
- `_hf_fetch_features` empty (repo moved or schema invisible) → skip.
- `_import_hf_auto` raises → catches, logs, batch continues.
- LB-name collision → skipped, dataset still imports.
