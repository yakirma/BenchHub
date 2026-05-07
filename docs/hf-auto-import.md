# HF auto-import design

This is a living design note. It exists because the user asked: *"think about
a way we can automate HF dataset import (also in the cost of changes in our
system to support it)"*. The goal is to turn `huggingface://user/repo` URLs
into BenchHub datasets without the operator having to pre-format anything.

## Why the current importer is limited

`scripts/seed_nyu_v2_curated.py` already proved we can pull from HF — it
streams the first WebDataset shard of `sayakpaul/nyu_depth_v2`, decodes the
HDF5 files, and writes BenchHub's `image_rgb/` + `raw_depth/` folder layout
on the fly. That works because we wrote glue code specifically for that
schema. **In production we accept exactly one shape**: a HF repo whose
top-level files already match BenchHub's `metric_*/`, `hist_*/`, `raw_*/`,
`image_*/` folder convention. Almost no real HF datasets do.

## The actual HF dataset shapes we'd encounter

| Shape | Frequency | Examples |
|---|---|---|
| Parquet rows in `data/*.parquet` (HF Datasets default) | **Most common** | most modern datasets uploaded via `datasets.push_to_hub` |
| Arrow shards | Common | older HF Datasets |
| WebDataset shards (`.tar` containing per-sample bundles) | Common in vision/audio | `sayakpaul/nyu_depth_v2`, LAION |
| Loose files in folders matching their own naming convention | Niche | research uploads |
| Loader script (`my_dataset.py`) | **Deprecated by HF as of mid-2025** | older repos still ship them but `datasets` lib won't load |

A generic "import any HF dataset" path needs at minimum a parquet→folder
adapter. WebDataset is a separate adapter. Loader scripts are dead.

## What "auto-import" could realistically mean

Three levels of ambition, in increasing effort:

### Level 1 — Schema-aware adapters for a fixed list of formats

Pick three ingest paths and write a converter for each:

1. **BenchHub-shaped repos** (current path; no change).
2. **Parquet with image+label columns** (most common HF shape). Use
   `datasets.load_dataset(streaming=True)`, iterate, write each row into a
   folder layout like `image_<column>/<idx>.png` and `metric_<column>/<idx>.txt`.
   Operator picks at import time which column goes into which BenchHub field
   type via a small mapping UI: "column `image` → `image_rgb`, column `depth`
   → `raw_depth`, column `label` → `metric_label`".
3. **WebDataset tar shards** (NYU v2 path generalized). Stream the first
   shard, walk by file extension within each tar member, default-map common
   names (`*.jpg` → `image_rgb`, `*.h5` → unpack via h5py, etc.).

**Cost:**
- ~400–600 lines of converter code.
- New UI on the HF import form: column-mapping picker (after the user
  pastes a repo ID, we fetch the schema preview and ask them to map).
- `datasets` library becomes a production dep (not just a seed-script dep).
  Adds ~200 MB to the Docker image; we'd want to vendor only the slim
  loader subset if possible.
- Quota math gets fuzzier — we don't know dataset size until we start
  streaming. Either time-cap the import (e.g. 2 minutes, then stop) or
  sample-cap (e.g. first 1000 rows, configurable).

**System changes required:**
- `Dataset` gets a `source_kind` column (`zip` / `hf-bench` / `hf-parquet` /
  `hf-webdataset`) so we know how to refresh later.
- `Dataset.source_metadata` JSON column to remember the column mapping for
  re-import.
- Background-job pipeline. The current `process_dataset_zip` runs synchronously
  inside the request; HF imports of any size need to move to Celery so the
  HTTP response doesn't time out.

### Level 2 — Inferred-schema imports

Like Level 1, but instead of asking the user to map columns, we *guess*
based on column names + dtypes. Heuristics:

| If the column… | Map to |
|---|---|
| is dtype `Image()` and named `image`/`rgb`/`color` | `image_rgb` |
| is dtype `Image()` and named `depth`/`depth_map` | `raw_depth` (convert to NPZ) |
| is a numeric scalar named `score`/`metric`/`label` | `metric_<name>` |
| is `Sequence(int)` of length 256/512/1024 | `hist_<name>` |
| anything else | skip (with a warning surfaced to the operator) |

**Cost:**
- All of Level 1, plus ~200 lines of inference rules.
- Lots of edge cases — datasets get named weirdly, "depth" sometimes means
  millimeters and sometimes a normalized 0–1 scale, etc. Inference will be
  wrong sometimes; the operator needs an "override" path that falls back to
  Level 1's manual mapping.
- We almost certainly accumulate a mapping registry per HF org over time
  ("oh, NYU-style depth always wants this transform").

**System changes required:** same as Level 1, plus a `dataset_import_review/`
queue UI where imports flagged as "ambiguous" wait for operator approval
before going public.

### Level 3 — Background pull-replicate-rebuild

Treat HF imports as ongoing mirrors. Subscribe to the HF repo's commit
stream (HF has webhooks); when upstream pushes new commits, requeue the
import. Add an `auto_refresh: bool` flag.

**Cost:** Levels 1+2 plus a webhook receiver, conflict resolution
("upstream's column rename collides with our existing field"), and a UI for
"this dataset's last sync was X days ago".

**Probably not worth doing until the platform has paying users.** A
manual "Refresh from HF" button hits 90% of the value at 10% of the cost.

## Recommendation

**Build Level 1 next, with parquet support only**, scoped tightly:
- One new converter (`hf_parquet_to_benchhub.py`) that handles the
  `Image()` + scalar-label case (covers the bulk of vision benchmark
  datasets).
- Move dataset ingestion to a Celery task so long imports don't block.
- Add `Dataset.source_kind` + `source_metadata` columns now, even if only
  Level 1 uses them, so future levels don't need a migration.
- Operator-supplied column mapping at import time (no inference). Saves
  the ~200 lines of fuzzy heuristics for a future round.

Skip WebDataset for now — it's a separate code path and the BenchHub-shape
repos can already cover that case manually (the NYU v2 seed script is a
template).

This unblocks roughly half the HF Datasets catalog as drag-and-drop
imports without committing to schema inference complexity.
