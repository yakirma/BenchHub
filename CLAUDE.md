# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BenchHub (a.k.a. dTOF Benchmarking) is a Flask + Celery + SQLite web app for benchmarking pipeline submissions (originally dTOF SPAD histograms) against ground-truth datasets. Users upload ZIPs, define Python metrics/visualizations dynamically, and view leaderboards/comparisons. The app runs on `http://localhost:6060`.

## Running the app

```bash
# 1. Redis (broker + result backend, default port 6379)
redis-server

# 2. Celery worker (in repo root)
celery -A app.celery worker --loglevel=info

# 3. Flask app
python app.py
```

**Tests live under `tests/`** (60+ files) and run with `pytest tests/`. Shared fixtures are in `tests/conftest.py`:
- Per-session `app` fixture wires Flask + Celery into TEST mode (`task_always_eager=True`) so submission/eval flows run inline.
- Per-test `db_session` drops + recreates all tables, so tests are independent.
- `auth_client` is a `client` with `session['user_id']` already set to a fresh `logged_in_user`.
- `make_zip(name, layout, root_folder=...)` builds a fake submission/dataset ZIP for upload-path tests.

`BENCHHUB_DATA_DIR` is redirected to a per-session tempdir so tests never touch `~/.dtofbenchmarking`.

The standalone `test_chain.py` / `test_celery_chain.py` / `test_chain_app.py` files at the repo root are ad-hoc Celery experiments — NOT part of the pytest suite. Run pytest with `pytest tests/` (not bare `pytest`) so it doesn't try to collect them.

No lint or build commands are wired up.

Dependencies: `pip install -r requirements.txt` (Flask, Flask-SQLAlchemy, celery, redis, numpy, scipy, matplotlib, Pillow, h5py, soundfile, …). `pytest` isn't pinned in requirements.txt — install separately for the test suite.

## Data and config locations

- **DB + uploads live OUTSIDE the repo**, at `~/.dtofbenchmarking/` (`database.db` and `uploads/`). The empty `database.db` and `uploads/` in the repo are vestigial — do not assume they are the live ones.
- `local_config.py` (gitignored-style, may not exist on a fresh clone) provides `GIT_REPO_PATH` for the *external* git repo from which submission/dataset author info is extracted via `git log`. It is imported optionally; if missing, `GIT_REPO_PATH = None`.
- Global UI settings (column widths, theme) are stored as JSON in `~/.dtofbenchmarking/settings.json` via the `GlobalSettings` singleton.

## Architecture

### One-file Flask app
`app.py` is ~6600 lines and holds essentially everything: SQLAlchemy models, all Flask routes, ZIP processing, DB migrations, custom-field detection, and visualization rendering. When extending, prefer editing `app.py` over creating new modules — the existing code does not have a layered structure to slot into. Two small helpers live outside:
- `metric_engine.py` — `evaluate_dynamic_metric` (exec's user-supplied Python code), `get_metric_context` (assembles the kwargs for a metric call), and `sort_metrics_by_dependency` (Kahn's-algorithm topo sort so metric B can consume metric A's output).
- `tasks.py` — Celery tasks. **Important circular import shape**: `tasks.py` imports from `app` (`celery, db, Submission, ...`), and `app.py` lazily imports `tasks` inside route handlers (search for `tasks.process_submission.delay`). Don't move task definitions into `app.py` or rearrange imports without understanding this.
- `metric_routes.py` — orphaned legacy snippets (uses `@app.route` without importing `app`). Not actually wired into the running app; the equivalents live in `app.py`. Treat as dead code unless explicitly resurrecting.

### Domain model (`app.py` ~270–510)
- **Project** → has many **Leaderboard**s. `Project` is just a namespace; URLs are prefixed with `/<project_name>/...` and resolved via `@app.before_request load_project_context` (cookie-fallback to `active_project_id`).
- **Dataset** is **global** (not project-scoped, despite older code comments). Linked to leaderboards via the `leaderboard_datasets` association table — a leaderboard can have multiple datasets. The legacy `Leaderboard.dataset_id` column is deprecated but still populated for back-compat.
- **Sample** belongs to a Dataset. `HistogramData`, `SignalShape`, `ConfigData` are legacy per-sample tables; new data flows through `CustomField` instead. The `Sample.histogram_data` / `Sample.signal_shape` Python @properties shadow the SQLAlchemy relationships and transparently fall back to `CustomField` rows — be aware when querying.
- **CustomField** is the unified, dynamically-typed bag for arbitrary per-sample (Dataset) and per-(submission, sample) data. `field_type` ∈ `{image, scalar, metric, histogram, depth, json, text}`. Per-sample metric *outputs* are also written here as `name=f"lm_{leaderboard_metric_id}"`, which is what enables `reaggregate_submission_metrics` to recompute pooling without re-running user code.
- **GlobalMetric** / **GlobalVisualization** store user-uploaded Python source. **LeaderboardMetric** / **LeaderboardVisualization** are link tables that bind a global definition to a leaderboard with `arg_mappings` (JSON dict mapping function arg → context key), `target_name` (display alias used as the dependency name for chaining), `pooling_type` (`mean|median|percentile|min|max`), `sort_direction`, and `tag_filter`.
- **MetricResult** stores the final aggregated scalar per (submission, leaderboard_metric).

### Submission processing pipeline
1. Upload (`upload_submission` route) → `process_submission_zip` extracts ZIP into `uploads/submissions/<id>/`, runs `detect_custom_fields` to populate `CustomField` rows.
2. `tasks.process_submission.delay(sub.id)` enqueues async work.
3. `_process_submission_impl` in `tasks.py`:
   - Builds a per-sample context dict via `get_metric_context` (GT custom fields + submission custom fields + on-the-fly entropy from histogram folders).
   - Topo-sorts `LeaderboardMetric`s (`sort_metrics_by_dependency`) so dependencies run first; their outputs are merged back into each sample's context.
   - For per-sample metrics: writes each value to a `CustomField` row (so re-aggregation can skip re-execution), then pools via `pooling_type`.
   - For aggregated metrics: passes the full list of values for non-aggregated dependencies and the scalar for aggregated dependencies.
   - Pre-caches aggregated visualizations.
   - Updates `Submission.processing_status` granularly (`Pending` → `Processing: Metric N/M (name)` → `Generating Visualizations` → `Processed` / `Error: ...`).
4. Batch recalculation uses `process_submissions_batch_sequential` which runs submissions one-at-a-time on purpose — concurrency was rolled back (see commit `8a77b48`), so don't re-introduce a `group()`/`chord()` here without checking why.

### Folder convention for ZIPs (dynamic field detection)
`detect_custom_fields` scans every folder in a dataset/submission ZIP and infers types from prefix + file extensions:
| Prefix          | Type      | Files                                  |
| --------------- | --------- | -------------------------------------- |
| `metric_`       | metric    | `<sample>.txt` containing a float      |
| `hist_` / `raw_histogram` / `hist` | histogram | `<sample>.npz` (`bins`, `counts`) |
| `raw_`          | depth     | `<sample>_<W>x<H>.npz`                 |
| (anything else) | image / scalar / json / text | by file extension          |

`git_info.json` (or `git.info`) at the ZIP root is parsed for commit metadata; if `author` is absent, `get_author_from_git_commit` shells out to `git -C $GIT_REPO_PATH log origin/<branch>` to recover it.

### Frontend
Server-rendered Jinja templates in `templates/` (no framework, vanilla JS). The big screens are `leaderboard.html`, `comparison.html`, `dataset_view.html`, `edit_leaderboard.html`. Static assets are minimal (`static/css/`, `static/js/`).

### DLP-safe code path
Some networks block `.py` uploads. The metric editor encodes user code as `BASE64:<...>` client-side; `handle_dlp_safe_code` (in `app.py`) detects the prefix and decodes server-side. `scripts/obfuscator.html` and `scripts/obfuscator_gui.py` are standalone helper tools for the same pipeline. Preserve this pathway when touching metric upload/edit endpoints.

### DB migrations
There is no Alembic. `check_and_migrate_db()` (called from `if __name__ == '__main__':`) runs raw SQLite `PRAGMA table_info` checks and `ALTER TABLE ... ADD COLUMN` against `~/.dtofbenchmarking/database.db` on every startup. When adding a new column to a model, also add a migration block here or existing installations will break. SQLite is opened with `journal_mode=WAL` and a 120s `busy_timeout`.

## Things to be careful with
- The `Sample` class redefines `histogram_data` and `signal_shape` as @properties *after* declaring them as relationships; the Python descriptor wins at attribute access time. Don't "clean this up" without verifying every read site.
- `Attachment.kind` is a Python `@property` that returns `'bh'` or `'hf'` from `dataset_id IS NULL`. **It is NOT a DB column**, so filtering a query with `.filter(Attachment.kind == 'hf')` silently matches zero rows. Filter on the underlying columns (`Attachment.hf_repo_id.isnot(None)` for HF, `Attachment.dataset_id.isnot(None)` for BH) when writing SQL; use `att.kind` only in Python code that's already iterating rows.
- `secret_key = 'supersecretkey'` is hardcoded; fine for local dev, do not assume any auth/CSRF protections.
- `evaluate_dynamic_metric` calls `exec()` on user-supplied Python — by design. Treat this app as trusted-local-network only.
- `app.py` uses `@app.url_value_preprocessor` + `@app.url_defaults` + a monkey-patch of `werkzeug.routing.Map.is_endpoint_expecting` to inject `project_name` into every URL automatically. New routes that take a `<project_name>` path component will get the value injected on `url_for(...)` without you passing it.

## Frontend conventions (theme, layout)
- **Theme is light-only by design.** `<html data-bs-theme="light">` is hardcoded in `base.html`. The CSS override block sets identical values for `[data-bs-theme="light"]` and `[data-bs-theme="dark"]`, but Bootstrap's own navbar CSS vars (`--bs-navbar-color`, `--bs-tertiary-bg-rgb`) aren't covered, so any dark-mode rendering leaks white-on-white. `global_settings.theme_mode` still defaults to `'dark'` in SQLite but the template no longer reads it. Don't reintroduce a real dark mode without overriding *every* `--bs-navbar-*` and `-rgb` variant.
- **Navbar text is pinned manually** (`.navbar .nav-link { color: #281950 }` etc.) as belt-and-suspenders.
- **`stretched-link` inside a sticky sidebar needs `position: relative` on the parent.** Without it the first card's anchor covers the entire scroll container and intercepts every later click. Bit us in `/explore` category tree.
- **Mobile pattern for long lists** (metrics, visualizations): render a `<select>` with `d-md-none`, hide the sidebar with `d-none d-md-block`. Keeps the detail pane on-screen without a Bootstrap collapse dance.

## HF dataset attachment patterns
- **`_HF_SPLIT_PREFERENCE = ['test', 'validation', 'val', 'dev', 'train']`** in `app.py`. PWC bulk imports default `Attachment.hf_split='train'`, but that's a hint, not a hard preference. `_resolve_hf_split_and_load(att, load_fn)` walks the preference order, probes row 0 to verify mapped GT columns aren't all null, falls back to first loadable split if none have full GT, and **persists the resolved split back via `_persist_resolved_split`** so the LB-detail badge tells the truth.
- **`_infer_mapping(features)` defaults string→text, Audio→audio, everything-else→json.** Used to skip-and-leave-empty for any column it didn't recognise, silently dropping most QA / relation-extraction / structured GT. If you change this, double-check `_persist_hf_eval_snapshots` and `_virtual_sample_from_hf_row` still know how to persist the new kind.
- **`Value:unknown` (HF's flattening of nested types) → json**, not skip. DocRED's `sents`/`vertexSet`/`labels` look like this.
- **`_pwc_task_to_category` strips domain prefixes** (Medical, Aerial, Satellite, Few-Shot, Self-Supervised, …) before classification, so "Medical Image Segmentation" → "Vision/Image Segmentation". New prefixes go in `_DOMAIN_PREFIXES` — order them shortest-first so "medical image" doesn't get half-eaten.
- **`populate_lb_samples` has a 5-min `soft_time_limit`.** PWC's `suggest_hf_repo` fallback sometimes lands on a monolithic HDF5 repo (e.g. `btherien/imagenet-64x64x3` is 100GB+ behind `load_dataset`), and without the timeout one task takes down the whole worker. **The Fly machine hosts Flask + Celery + Redis on one box** — don't bulk-enqueue dozens of populate tasks; the site becomes unresponsive. Use the per-LB "Populate samples" button instead, or rate-limit any bulk operation.

## Field-type taxonomy (CustomField.field_type)

| field_type   | Storage | Comparison cell | Notes |
|--------------|---------|-----------------|-------|
| `scalar`     | `value_float` | `gt_scalar_value`, smart_num-formatted (integer → no `.0000`) | Togglable as a column (used to live only in `per_source_stats`). |
| `text`       | `value_text` | `gt_text_value`, scrollable card | Default for any string column. |
| `metric`     | `value_float` | Goes through `per_sample_metrics` chart panel | NOT togglable as a normal column. |
| `image`/`depth`/`mask` | marker row + bench_cache | `<img>` → `serve_custom_field_image` → `serve_gt_viz` | Bytes don't live on the volume. |
| `audio`      | marker row + bench_cache (waveform PNG) | Same `<img>` path; route sniffs PNG magic | HF Audio decode needs `soundfile`. |
| `json`       | `value_text` (serialised JSON) | JSON scroll box | Dicts / Sequence-of-dict / bboxes / Translation features. `get_metric_context` json-decodes back into Python. |
| `topk_list`  | `value_text` (JSON array) | Falls back to text render | Ranked-list predictions for Hits@N / MRR. Deserialised by `get_metric_context` into a Python list. |
| `histogram`  | `value_blob` | Sparkline / chart | Fixed-length int sequences. |

## Comparison view (`/comparison/<lb_id>`) gotchas
- **`samples_only=1` must thread through every navigation link** (pagination, View Options form, filters). Without it the page collapses back into full submission-comparison mode the moment the URL drops the param.
- **`samples_only_mode` filters out only submission-needing panels** (`per_sample_metrics`, `per_source_stats`, `pred_histogram`, `viz_*`) — NOT scalar/text/json/audio columns.
- **`leaderboard.comparison_display_columns` CSV is legacy.** The renderer uses `available_display_options - hidden_comparison_display_columns`. The form persists only `hidden_*`. Don't rely on the CSV for visibility decisions.
- **The template gates header+cell on `all_field_types.get(col_key) != 'metric'`** (used to be `not in ['scalar', 'metric']`). If you add a new field type, make sure it isn't accidentally excluded.

## "Explorable" status
- `_compute_explorable_lb_ids(lb_ids)` returns the LB IDs whose GT is actually cached: BH dataset Sample rows OR LB-scoped CustomField rows (sample_id+submission_id both NULL — the HF-stub marker rows). Drives the green/yellow pill on `/explore`, `/home`, `/landing`, and the "Explore samples" button label on the LB detail page.
- **An LB with `canonical_for_repo IS NOT NULL` and zero GT CFs is effectively broken** — surface the owner-only "Populate samples" button instead of silently rendering an empty Explore page.
- **`/datasets` lists two sections**: the regular `Dataset` rows (BH ZIP uploads) and a "Cached HuggingFace datasets" section built from distinct `Attachment.hf_repo_id` rows whose owning LB is in `_compute_explorable_lb_ids`. Each HF row links to the first LB's Explore-samples view. Filter is intentional: a non-explorable HF row would link to an empty page.

## Metric authoring
- **LLM-authored metrics from `_llm_generate_metric_code` are not safe to ship verbatim** for non-trivial cases (rank-based, span-overlap, BLEU-family). They tend to mix scalar-vs-list logic awkwardly and quietly return the wrong number. The rank-based LBs (Link Prediction) needed manual rewrites with `_rank_of_gt(gt, pred_list)` helpers; spot-check any new ones before relying on the column.
- **Metrics are stored as Python source on `GlobalMetric.python_code`** and exec'd inside `evaluate_dynamic_metric` with `numpy as np` available. To replace one cleanly, query the row by name and overwrite `python_code` — no migration needed.

## Migration patterns
- **Every model-level column add needs a corresponding `ALTER TABLE ... ADD COLUMN` block in `check_and_migrate_db()`** (no Alembic). Recent additions: `Leaderboard.category` (two-level "Area/Task" taxonomy).
- **Idempotent data backfill belongs in the same `check_and_migrate_db()` block** after the ALTER, gated on "any existing rows still NULL". The PWC-category backfill at `--- 3b. ---` is the template: probe optional resources (`pwc_client._index_path()`), best-effort match, swallow exceptions so fresh installs aren't blocked.

## Tests
The pytest suite under `tests/` already covers most of the regression-prone surface — PWC import, HF features fallback, attachment iteration, metric context, comparison routes, smart_num, etc. **Add a test next to the closest existing one when you fix a bug** rather than writing it after the fact:

| Touched code | Test file to extend |
|--------------|---------------------|
| `_pwc_task_to_category`, `_PWC_AREA_RULES`, `_DOMAIN_PREFIXES` | `tests/test_pwc_category.py` |
| `_resolve_hf_split_and_load`, `_HF_SPLIT_PREFERENCE`, `_persist_resolved_split` | `tests/test_hf_split_resolver.py` |
| `_infer_mapping` (the `Value:unknown` → json fall-through, Audio kind, Sequence-of-* fallback) | `tests/test_pwc_import.py` or `tests/test_hf_features_fallback.py` |
| `_compute_explorable_lb_ids` | `tests/test_explorable.py` |
| `_VirtualSample` / `_VirtualCustomField` json/topk_list/audio dispatch | `tests/test_attachment_iter.py` |
| `get_metric_context` deserialization of text/json/topk_list | `tests/test_metric_context_arrays.py` |
| Samples-only mode in `comparison_view` (incl. pagination + form param threading) | `tests/test_routes_comparison.py` |

Run with `pytest tests/` (not bare `pytest`, to avoid the ad-hoc root-level `test_chain*.py` files).

## Working notes for future sessions
- **Write decisions down as they happen.** If a fix involves a non-obvious gotcha (CSS leak, framework default, schema quirk, ordering trap), append a bullet under the appropriate section in this file *during the change*, not after. Section anchors: "Things to be careful with", "Frontend conventions", "HF dataset attachment patterns", "Comparison view gotchas", "Metric authoring", "Migration patterns".
- **Treat CLAUDE.md as the durable memory.** Commit messages document one change; this file documents what to know to make the NEXT change.
