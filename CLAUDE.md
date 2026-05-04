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

There are no test, lint, or build commands wired up. The repo's `test_*.py` files at the root (`test_chain.py`, `test_celery_chain.py`, `test_chain_app.py`) are ad-hoc Celery chain experiments, not a real test suite.

Dependencies: `pip install -r requirements.txt` (Flask, Flask-SQLAlchemy, celery, redis, numpy, scipy). Matplotlib and PIL are also imported by `app.py` but missing from `requirements.txt`; install separately if needed.

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
- `secret_key = 'supersecretkey'` is hardcoded; fine for local dev, do not assume any auth/CSRF protections.
- `evaluate_dynamic_metric` calls `exec()` on user-supplied Python — by design. Treat this app as trusted-local-network only.
- `app.py` uses `@app.url_value_preprocessor` + `@app.url_defaults` + a monkey-patch of `werkzeug.routing.Map.is_endpoint_expecting` to inject `project_name` into every URL automatically. New routes that take a `<project_name>` path component will get the value injected on `url_for(...)` without you passing it.
