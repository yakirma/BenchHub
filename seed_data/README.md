# Batch dataset seeding + baseline runners

Two pieces in this directory:

1. **`scripts/seed_datasets.py` + `*.json` configs** — bulk-import HF
   datasets and create their auto-LBs. Below.
2. **`seed_baselines.py` + `baselines_<domain>.py` stubs** — run a list
   of HF model checkpoints against an LB's GT and POST submissions.
   See "Baseline runners" further down.

## Datasets

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

---

# Baseline runners

`seed_baselines.py` is the boilerplate for "run N HF model
checkpoints against an LB's GT and POST one submission per model."
It does:

1. Download + extract the GT zip.
2. Discover sample names from the GT folder layout.
3. For each model spec: load the model, run `predictor_fn` on every
   sample, collect predictions into a `<field>/<sample>.txt` tree
   (bare-name folders — `metric_*` is reserved for user-precomputed
   metric values, not raw predictions).
4. ZIP and POST to `/api/leaderboard/<id>/submission/upload` with
   `source_colab_url` so the LB page back-links to the notebook that
   produced each submission.

## Domain stubs

| File                            | Task                              | Models |
| ------------------------------- | --------------------------------- | ------:|
| `baselines_depth.py`            | Monocular depth estimation        |     10 |
| `baselines_segmentation.py`     | Semantic segmentation             |     10 |
| `baselines_language.py`         | Text classification (sentiment / NLI / topic) | 10 |
| `baselines_denoising.py`        | Image restoration / denoising     |     10 |

Each defines `MODELS`, `predictor_fn(spec, model, processor, inputs)`
and `load_inputs(gt_root, sample_name)`. The stubs use the simplest
sensible scalar prediction (mean depth, dominant class, mean intensity)
so the upload mechanics work end-to-end. Replace `predictor_fn` with
the real metric your LB needs once the pipe is verified.

## Running

You'll want a Colab GPU runtime (or local CUDA). Quick start:

```python
# In a Colab cell, with this directory uploaded as /content/seed_data:
%cd /content/seed_data
!pip -q install transformers torch pillow numpy requests

import os
os.environ['BENCHHUB_API_TOKEN'] = '<your token>'

from seed_baselines import seed_baselines
from baselines_depth import MODELS, predictor_fn, load_inputs

seed_baselines(
    leaderboard_id=42,
    api_token=os.environ['BENCHHUB_API_TOKEN'],
    gt_zip_url='https://benchhub.fly.dev/dataset/17/download',
    models=MODELS,
    predictor_fn=predictor_fn,
    load_inputs=load_inputs,
    source_colab_url='https://colab.research.google.com/...',
)
```

Or as a CLI:

```bash
python seed_baselines.py --domain depth \
    --leaderboard-id 42 \
    --api-token "$BENCHHUB_API_TOKEN" \
    --gt-zip-url https://benchhub.fly.dev/dataset/17/download
```

## Caveats

- **GPU required** for any of the bigger checkpoints. Colab T4 is fine
  for 8 of the 10 depth models; SDXL upscalers in the denoising stub
  need at least an A100.
- **Repo IDs rotate.** Any of these can disappear or get renamed —
  `_hf_loader` will surface the load failure and the batch continues
  to the next model.
- **Predictor stubs are deliberately naive.** A real depth LB wants
  per-pixel error vs the GT depth map, not the mean of the predicted
  depth. The mean-of-pixels predictor exists so the upload pipe is
  verifiable end-to-end without paying full evaluation cost.
