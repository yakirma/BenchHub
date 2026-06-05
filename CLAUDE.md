# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Typed-contract architecture (Phases A–D shipped)

The legacy folder-name ZIP path is gone; the strict typed contract is the spine of the system end-to-end. Phases A through D are live in prod — read this section before touching code or writing docs that reference deleted concepts.

**The pipeline now:**

1. **Admin uploads a typed dataset** via `POST /admin/import_typed_dataset` (server-side path) or the `scripts/import_typed_dataset.py` CLI. Format: a directory with `manifest.json` + `<field>/<sample>.<ext>` per field. The importer materialises Dataset + Sample + CustomField rows and copies file-backed kinds under `uploads/datasets/<id>/`. Inline kinds (Scalar, Label) decode into `value_float` / `value_text`.
2. **A leaderboard declares its contract** via `Leaderboard.required_pred_fields_json` (list of `{name, kind, params, role}` entries; `role ∈ {input, gt, pred}`).
3. **Submitters use `benchhub-client`** — `bh.Client(token)` → `client.submission(lb_id).predict(sample, **kwargs).submit()`. The client validates each `DataType` instance at stage time, packs them into a ZIP matching the server's on-disk format, and POSTs to `/api/submit/<lb_id>`.
4. **Server validates** the submission manifest against the LB contract (every required pred name present, kinds match), writes Submission + CustomField pred rows, enqueues `tasks.process_submission`.
5. **Metric engine** builds the per-sample context with both primitive (`gt_depth_pred`) AND typed (`__typed__gt_depth_pred`) entries. Metrics that declared `GlobalMetric.input_kinds` (non-empty JSON array) receive `bh.Depth` instances; legacy metrics keep the primitive. Five typed reference metrics seeded: `accuracy`, `rmse_depth`, `mae_depth`, `iou_mask`, `exact_match_text`.

**Key files:**
- `benchhub/types.py` — 9 `DataType` subclasses (`Image`, `Mask`, `Depth`, `Audio`, `Text`, `BBoxes`, `Label`, `Scalar`, `Json`). Source of truth.
- `benchhub/manifest.py` — manifest spec + `import_typed_dataset` + `import_typed_submission` + `check_submission_matches_contract`.
- `benchhub/client.py` — `Client`, `SubmissionBuilder`, `FlaskTestClientTransport` (the in-process transport tests use). `Client.iter_samples(lb_id, *, force_download=False)` pulls all file-backed inputs as ONE bulk ZIP from `/api/leaderboard/<id>/inputs.zip` (server route `api_leaderboard_inputs_archive`, `ZIP_STORED`), extracts to `~/.cache/benchhub/<host>/lb_<id>/` (override root via `$BENCHHUB_CACHE_DIR`), and yields decoded `bh.<Kind>` instances. Cache is keyed on the `cache_token` from `/samples` (busts when a materialisation is rebuilt) + the sorted `<field>/<sample>` entry list; `force_download=True` re-fetches. Masks pack the raw `.classid.png` so they decode to `bh.Mask`, not the palette `bh.Image`. Falls back to per-sample `fetch_bytes` if the archive route 404s (older server). Tests must isolate the cache — `conftest` points `$BENCHHUB_CACHE_DIR` at the session tmp and wipes it per test (LB ids repeat across tests).
- `scripts/import_typed_dataset.py` — admin CLI.
- `scripts/seed_reference_metrics.py` — idempotent typed-metric seed.
- `metric_engine.py:_typed_for_cf` / `_stash_typed` / `_metric_wants_typed` — the typed-instance plumbing.

**End-to-end verification:** `tests/test_phase_b_end_to_end.py` exercises the whole loop (typed import → client → typed submit → typed-instance metric eval → asserted MetricResult). 924 passing tests, 0 failures.

## Hybrid storage (Stages A–C shipped)

The catalog now defaults to a lightweight preview tier; full-resolution bytes only land on disk for leaderboards that bind a subset and materialise.

**Two-tier storage:**
- **Preview** (always present): `uploads/datasets/<id>/<field>/<sample>.<ext>` — downscaled+JPG-encoded image/mask/depth (max 512px, q85), waveform PNG for audio, inline content for text/json/scalar/label. Marked `Dataset.preview_only=True`. ~30–50 KB per visual sample. The dataset_view samples table renders directly from here; users can't tell visually it's not full-resolution.
- **Materialised** (per-LB): `uploads/lb_materializations/<lb_id>/<field>/<sample>.<ext>` — full-resolution bytes for the subset the LB chose. Counts against the LB owner's quota.

**Per-LB sample selection:**
- The `/create_lb_for_dataset` page (and `/create_lb_chooser`) carries a wizard: `sample_cap`, `sampling` (head / random / stratified), `stratify_field`, `sampling_seed`.
- Random is default; stratified is auto-default when the dataset has a `label`-kind field.
- POST to `/create_leaderboard` writes a `LeaderboardMaterialization` row + `.delay()`s `tasks.materialize_leaderboard`.
- The Celery task runs `benchhub.lb_materialize.materialize_for_lb` — picks samples via `pick_samples()`, re-runs `materialize_hf_to_typed_dir` at full resolution into a temp dir, copies the chosen subset into `uploads/lb_materializations/<lb_id>/`, sets status `ready`. Failures stay in `status='failed'` with `error_message` and surface a Retry button on the LB page (`/leaderboard/<id>/materialize/retry`).

**Path resolution at scoring time:**
- `extract_viz_arg_value(sample, submission, field_key, *, leaderboard_id=None)` for file-backed `gt_<field>` lookups consults `benchhub.lb_materialize.materialized_or_preview_path()`. Materialised file wins when present; preview fallback otherwise. Inline kinds (scalar/label/text/json) unaffected.
- `execute_visualization` route passes `leaderboard_id=lv.leaderboard_id`, so the COCO overlay viz on an LB renders against full-resolution images even though the dataset row itself is preview-only.

**Quotas (split-bucket, Phase 13):**
- Every user has two byte caps on `User`:
  - `quota_public_max_bytes` — **50 GB** default (dropped from 100 GB at public launch; `check_and_migrate_db` backfills old-default rows). Charged whenever a row whose own `visibility == 'public'` is created or grown. Covers public Datasets + LB materialisations owned by the user.
  - `quota_private_max_bytes` — **10 GB** default. Charged for `visibility in {'private', 'unlisted', NULL}`. Working space for unpublished content.
- `check_quota(user, *, kind='dataset_create', incoming_bytes, visibility=...)` reads the bucket implied by the row being written; the visibility kwarg is **required** for new code (default `'private'` fails safe on the smaller bucket).
- `storage_used_bytes(user, *, visibility=...)` partitions per bucket; pass `None` for the legacy total.
- Helpers live next to each other in `app.py` (~1763–1944): `_visibility_bucket`, `storage_used_bytes`, `quota_cap_for`, `check_quota`.
- **Publish flip pre-flight**: `set_dataset_visibility` and `set_leaderboard_visibility` reject a private→public flip when the user's public bucket can't absorb the moving bytes. Surfaces a flash and 302s back to the settings page. Admins bypass.
- Submission ZIPs are not charged to either bucket — the LB owner already paid for the materialised inputs the submitter is responding to.
- The legacy `quota_max_storage_bytes` column stays around for back-compat with old admin tools but `check_quota` no longer reads it.
- Admins still bypass entirely via `is_admin()`.
- Quota gate is pre-flight on uploads (refusal) and pre-flight on publish-flip (refusal); the post-flight write of `Dataset.storage_bytes` is the authoritative number used by future gauges.

**Key files:**
- `benchhub/preview.py` — `image_preview`, `depth_preview` (turbo colormap), `mask_preview` (deterministic palette), `audio_preview` (waveform PNG), single dispatch via `render_preview(kind, payload)`.
- `benchhub/manifest.py:import_typed_dataset(..., preview_only=True)` — routes vis modalities through the preview helpers, writes `.jpg`/`.png` instead of canonical extensions, sets `Dataset.preview_only = True`.
- `benchhub/lb_materialize.py` — `pick_samples()` (head/random/stratified) + `materialize_for_lb()` (re-fetches full bytes for the subset) + `materialized_or_preview_path()` (the resolver).
- `tasks.py:materialize_leaderboard` — Celery wrapper around `materialize_for_lb` so big imports don't block the request handler.

**Migration notes:**
- `Dataset.preview_only` column added in `check_and_migrate_db`.
- `leaderboard_materialization` table created in `check_and_migrate_db`.
- A one-shot at `/tmp/migrate_to_preview.py` converted all 25 pre-existing full-storage datasets in place (18 GB → 1.8 GB on disk, no failed renders, all 26 datasets are now preview-only). Refuses to migrate any dataset whose bound LBs have non-zero submissions — that case needs Stage C materialisation first.
- `tasks.run_hf_import` (admin /import_from_hf path) currently does NOT set `preview_only=True` — the bulk LLM loop does. Consider unifying.

## ⚠️ Pre-existing deletions (Phase A delete pile)

Big chunks of legacy machinery were already removed. Read this before touching code or writing docs that reference deleted concepts.

**Deleted (Phase A delete pile, commits `6707189`, `97f4b6c`, `66ffcc6`)**:
- HuggingFace import: every `import_from_hf*`, `admin_pwc_*`, `admin_lb_sota*`, `populate_lb_samples_route`, `hf_token_*` route; `_VirtualSample` / `_VirtualCustomField`; `_infer_mapping`; `_resolve_hf_split_and_load` / `_HF_SPLIT_PREFERENCE`; `_persist_hf_eval_snapshots`; `_create_lb_from_pwc_benchmark` and PWC task helpers; `_HF_MASK_TOKENS`; `pwc_client.py` entirely.
- SOTA / Colab notebook generation: `_static_colab_notebook`, `_personalize_notebook_for_user`, `_ensure_user_colab_gist`, `_ensure_colab_gist`, `_push_one_off_gist`, `_llm_generate_metric_code`, `_llm_generate_visualization_code`, `_llm_propose_text_evaluation_suite`, `_llm_infer_mapping`, `_llm_colab_notebook`, `_llm_sota_colab_notebook`, `leaderboard_colab_*`.
- `canonicality` concept: `admin_promote_leaderboard` route + UI form. Column stays in DB for back-compat but no code reads it. `canonical_for_repo` column also dead (HF-only metadata).
- Folder-prefix ZIP ingest: `detect_custom_fields`, `_classify_image_path`, `_folder_name_prefix_kind`, `_FIELD_TYPE_PREFIXES`, `process_dataset_zip`, `process_submission_zip`, `upload_dataset` route, `upload_submission` route. Upload UI replaced with a "paused" placeholder pointing at `/supported_types`.
- Legacy per-sample tables: `HistogramData`, `SignalShape`, `ConfigData` model classes. The SQLite tables themselves stay for existing DBs. `Sample.histogram_data` / `.signal_shape` / `.config_data` accesses now resolve to `None`.
- Tests: 26 entire test files removed (HF, PWC, colab, sota, llm proposer, smart-num-classlabel, auto-lb-metrics, lb-preview-extras, detect-custom-fields, process-dataset-zip, process-submission-zip, prune-incomplete-datasets, routes-dataset, remote-submissions, quotas-curated, canonicality, account-delete-hf, attachment-iter, get-metric-context, sota-picker, submission-colab-link, text-gt, create-lb-chooser, hf-* full suite).
- `/explore` is now a back-compat 302 → `/leaderboards`. The `Explore samples` button on LB pages is gone; the catalog is at `/leaderboards` only.

**Live state**: 800+ passing tests, zero xfailed, zero TODOs in source. Site serves cleanly at `runbenchhub.com`.

**`benchhub/` package** (single source of truth for the typed contract):
- `benchhub.types` defines 10 `DataType` subclasses: `Image`, `Mask`, `Depth`, `Audio`, `Text`, `BBoxes`, `Label`, `LabelList` (top-K with required `k`), `Scalar`, `Json`. Each with `encode()`/`decode()`/`validate()`/`visualize()` + the `DTYPES` registry. Imported by `app.py`, `metric_engine.py`, and the `benchhub-client` package surface (`Client`, `SubmissionBuilder`, `BHDatasetCreator`).
- `/supported_types` page is driven from `DTYPES` at request time so it can't drift from code.

**Metric authoring convention** (auto-derives from signatures):
- `def accuracy(gt: bh.Label, pred: bh.Label | bh.LabelList)` — annotations are parsed at save time into `GlobalMetric.input_kinds` (PEP 604 unions allowed, `|`-joined). Arg names like `gt` / `pred` / `input` map to `input_roles` via heuristic.
- Runtime `isinstance` asserts live in the metric body; the engine injects `bh` / `benchhub` aliases into the exec scope.

## Project overview

BenchHub is a Flask + Celery + SQLite web app for benchmarking model predictions against curated datasets. The original folder-name ZIP-ingest path is gone; the new typed contract (defined in `benchhub.types`) is the spine of the system. The app runs on `http://localhost:6060`.

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

## Deployment

Production is **self-hosted** on a home Ubuntu 24.04 box at
`runbenchhub.com`. gunicorn + celery + redis run directly under systemd
(no Docker), nginx + certbot in front, Cloudflare DNS in DNS-only mode,
`ddclient` for DDNS. The operational runbook — code-push flow, `.env`
keys, log tailing, rollback, breakages — is **`docs/SELFHOST_RUNBOOK.md`**;
read it before suggesting deploy/operations commands.

**Claude Code runs on the box itself**, not on a laptop. There are two
checkouts on disk and you need to keep them straight:

| Path | Role |
|---|---|
| `~/Git/BenchHub` (current working dir) | **Dev checkout** — edits + commits land here. Hot edits do NOT touch the live app. |
| `~/benchhub` | **Production checkout** — what gunicorn/celery actually serve. Updated only via `git pull`. |

The runbook's "ssh -p 2222 ymatri@runbenchhub.com" step is skippable —
you're already on the box. The deploy reduces to: commit + push from
`~/Git/BenchHub`, then `cd ~/benchhub && git pull && sudo systemctl
reload benchhub-web` from anywhere. **Never edit `~/benchhub` directly**;
it's the equivalent of editing prod on a server, and the next `git pull`
will clobber it (or worse, conflict).

**Fly.io is dead.** The artifacts (`fly.toml`, `Dockerfile`,
`.dockerignore`, `start.sh`, `entrypoint.sh`, `DEPLOY.md`,
`runner/fly.toml`) were moved to `archive/fly/` so a future Fly redeploy
can rebuild from there — they are NOT used by anything live. Don't
suggest `fly deploy`, `fly logs`, `fly secrets set`, or anything
Fly-specific; use the systemd / `git pull` flow from the runbook.
`runner/Dockerfile`, `runner/harness.py`, `runner/server.py` stayed in
place — local sandbox tests (`tests/test_sandbox_*`) still reference
them. Quick reference:

```bash
# From the dev checkout (~/Git/BenchHub):
git push origin main

# Then from anywhere on the box — we're already in:
cd ~/benchhub && git pull
sudo systemctl reload benchhub-web        # graceful HUP, no dropped requests
sudo systemctl restart benchhub-celery    # celery has no SIGHUP code-reload

# .env change or new schema column → full restart (HUP doesn't re-read env or rerun migrations)
sudo systemctl restart benchhub-web benchhub-celery

# Tail logs
journalctl -u benchhub-web -f
journalctl -u benchhub-celery -f
```

`BENCHHUB_AUTO_MIGRATE=1` is set in `.env`, so `check_and_migrate_db()`
runs on every process boot — that's what makes a model-column ALTER apply
on `restart`. Secrets live only in `~/benchhub/.env` on the box; there's
no `fly secrets list` to recover them from.

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

### Folder convention for ZIPs (`<type>_<field_name>`)
The canonical naming for any dataset/submission folder is `<type>_<field_name>`. Recognised type prefixes live in `_FIELD_TYPE_PREFIXES`:

| Type | Folder example | File(s) |
|------|----------------|---------|
| `image` | `image_rgb` | `<sample>.png` / `.jpg` / `.jpeg` / `.bmp` / `.tiff` |
| `mask` | `mask_annotation` | `<sample>.png` (single-channel class IDs or low-color RGB) |
| `depth` | `depth_gt` | `<sample>.npz` (key `depth`, HxW float) |
| `audio` | `audio_clip` | `<sample>.wav` / `.mp3` / `.flac` |
| `scalar` | `scalar_score` | `<sample>.txt` (one float) |
| `text` | `text_caption` | `<sample>.txt` |
| `json` | `json_bbox` | `<sample>.json` |
| `histogram` | `histogram_dtof` | `<sample>.npz` (`bins`, `counts`) |
| `metric` | `metric_iou` | `<sample>.txt` (pre-computed) |

`_folder_name_prefix_kind(folder_name)` returns the canonical kind when the prefix matches, and that decision is **authoritative** — the content-peek heuristics (`_classify_image_path`, `_classify_npz`) only run for folders without a recognised prefix (back-compat with legacy datasets). The `tags` folder is the one hardcoded exception (always text).

The same convention is mirrored on the HF-import side: `_infer_mapping` emits `target_field=<kind>_<col>` for every kind so the field name on the LB matches what a BH-uploaded dataset would use. If the HF column name already starts with that kind prefix, it's not double-prefixed.

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

## Palette (warm-cream after Cabinet retheme)
- Body bg: `#fcf9f4` (warm off-white), tertiary/card bg `#f5efe2` (warm cream).
- Body / heading text: `#3a2614` and `#2a1f10` (warm dark brown).
- Border / divider: `#e8dfc8` (tan).
- Primary accent **kept violet** `#7c3aed` (brand identity). Don't swap the primary without an explicit ask — many components (badges, focus rings, hover states) lean on it.
- Background radial gradients are amber `rgba(217,119,6,…)` + peach `rgba(244,114,89,…)`; if you re-tint the page bg, keep the gradient stops in the same hue family or it'll clash.

## Frontend conventions (theme, layout)
- **Theme is light-only by design.** `<html data-bs-theme="light">` is hardcoded in `base.html`. The CSS override block sets identical values for `[data-bs-theme="light"]` and `[data-bs-theme="dark"]`, but Bootstrap's own navbar CSS vars (`--bs-navbar-color`, `--bs-tertiary-bg-rgb`) aren't covered, so any dark-mode rendering leaks white-on-white. `global_settings.theme_mode` still defaults to `'dark'` in SQLite but the template no longer reads it. Don't reintroduce a real dark mode without overriding *every* `--bs-navbar-*` and `-rgb` variant.
- **Navbar text is pinned manually** (`.navbar .nav-link { color: #281950 }` etc.) as belt-and-suspenders.
- **`stretched-link` inside a sticky sidebar needs `position: relative` on the parent.** Without it the first card's anchor covers the entire scroll container and intercepts every later click. Bit us in the `/leaderboards` category tree (then called `/explore`).
- **Mobile pattern for long lists** (metrics, visualizations): render a `<select>` with `d-md-none`, hide the sidebar with `d-none d-md-block`. Keeps the detail pane on-screen without a Bootstrap collapse dance.

## HF dataset attachment patterns
- **`_HF_SPLIT_PREFERENCE = ['test', 'validation', 'val', 'dev', 'train']`** in `app.py`. PWC bulk imports default `Attachment.hf_split='train'`, but that's a hint, not a hard preference. `_resolve_hf_split_and_load(att, load_fn)` walks the preference order, probes row 0 to verify mapped GT columns aren't all null, falls back to first loadable split if none have full GT, and **persists the resolved split back via `_persist_resolved_split`** so the LB-detail badge tells the truth.
- **`_infer_mapping(features)` defaults string→text, Audio→audio, everything-else→json.** Used to skip-and-leave-empty for any column it didn't recognise, silently dropping most QA / relation-extraction / structured GT. If you change this, double-check `_persist_hf_eval_snapshots` and `_virtual_sample_from_hf_row` still know how to persist the new kind.
- **`Value:unknown` (HF's flattening of nested types) → json**, not skip. DocRED's `sents`/`vertexSet`/`labels` look like this.
- **`_pwc_task_to_category` strips domain prefixes** (Medical, Aerial, Satellite, Few-Shot, Self-Supervised, …) before classification, so "Medical Image Segmentation" → "Vision/Image Segmentation". New prefixes go in `_DOMAIN_PREFIXES` — order them shortest-first so "medical image" doesn't get half-eaten.
- **`populate_lb_samples` has a 5-min `soft_time_limit`.** PWC's `suggest_hf_repo` fallback sometimes lands on a monolithic HDF5 repo (e.g. `btherien/imagenet-64x64x3` is 100GB+ behind `load_dataset`), and without the timeout one task takes down the whole worker. **The Fly machine hosts Flask + Celery + Redis on one box** — don't bulk-enqueue dozens of populate tasks; the site becomes unresponsive. Use the per-LB "Populate samples" button instead, or rate-limit any bulk operation.

## Input vs GT roles on dataset columns
Every HF-attachment mapping entry now carries an optional `role` field: `input` (conditioning given to the submitter at inference time, NOT predicted) or `gt` (held server-side, target of prediction). Default = `gt` when missing (back-compat).
- `_pwc_task_input_kinds(task_name)` returns the set of `target_kind`s that should be flagged `input` for a given PWC task. New entries land via `_create_lb_from_pwc_benchmark` at import time.
- `_lb_submission_pred_fields` filters out pred fields whose GT column is flagged `input` — so e.g. `label_pred` no longer appears in the Image-Generation submission contract.
- Owner-editable on `/edit_leaderboard/<id>` → Prediction-fields tab → "Dataset field roles" panel. Frozen on LBs with verified submissions.
- The arg_mappings on LeaderboardMetric rows must reference a GT-role column or the metric won't have a valid pred field. The `.tag_input_gt.py` one-shot script (kept for reference) walks every PWC LB, flips roles per task, and rewrites arg_mappings to point at the highest-priority GT field.

## Metric / Visualization input-kind declarations
`GlobalMetric.input_kinds` and `GlobalVisualization.input_kinds` are nullable JSON arrays of accepted `target_kind` strings in argument order. NULL = unconstrained (legacy / undeclared). The metric detail pane on `/metrics?selected=<id>` surfaces a small "Accepts: <kind>×<kind>" row; backfilled for 18 curated metrics in `.backfill_input_kinds.py`. The LB→metric binding UI doesn't yet *enforce* the kinds — that's the next step in a follow-up. Add new patterns to `KIND_HINTS` in `.backfill_input_kinds.py` when a new metric ships.

## User-registered data types (`DataTypeDef`) + the decode hook
- A user can register a new `kind` BenchHub doesn't ship (NIfTI, point clouds, EEG, …) from the dedicated page **`/datatypes/new`** (route `datatype_new`; the `/supported_types` "Add a data type" button links here — there is no inline form anymore) or the client (`client.create_datatype(...)`). The `/datatypes/create` POST (`create_datatype_web`) redirects back to `/datatypes/new` on error, `/supported_types` on success. `/datatypes` 302s to `/supported_types`.
- `DataTypeDef` columns: `name` (globally unique, joins the same namespace as built-in `DTYPES`), `file_ext` (NULL ⇒ inline text), `viz_mime`, `visualize_code` (`def visualize(blob, params) -> PIL.Image`), **`decode_code`** (optional `def decode(blob, params) -> object`), `owner_user_id`, `visibility`. Storage is **bytes-verbatim** (encode = identity). Both `visualize` and `decode` run **only in the sandbox**.
- **The decode hook is the deserialize side of the contract.** When `decode_code` is set, a metric that consumes the registered kind receives the decoded object instead of raw bytes (mirrors built-in `bh.Depth.array`); absent ⇒ the metric gets the raw bytes. Wiring:
  - `metric_engine.RegisteredBlob` is the in-context carrier (`kind`, `blob`, `params`, `decode_code`). `get_metric_context` emits it for any GT/input CustomField whose `data_type` is **not** in `DTYPES` (via `_registered_blob_for_cf`, lazy `from app import DataTypeDef`).
  - Sandbox: `_jsonify_kwarg(RegisteredBlob)` → `{"__dtype__","decode","params","b64"}`; `runner/harness.py:_decode_arg` runs `decode` **inside the metric's own container** (no extra spawn) or returns the raw bytes.
  - In-process (non-sandbox) path: `evaluate_dynamic_metric` resolves `RegisteredBlob` via `_resolve_registered_blob` right before calling the metric.
- **Import admission**: `benchhub.manifest` (the standalone package) can't see `DataTypeDef`, so `validate_manifest` / `load_manifest` / `expected_file_path` / `import_typed_dataset` all take an optional **`extra_kinds={name: file_ext}`** map; a kind in `extra_kinds` is accepted and stored bytes-verbatim (no `DTYPES` class, no preview render). Server callers pass `app._registered_extra_kinds(owner_user_id)` (public + owner's own). The four call sites: `_ingest_typed_dataset_zip`, `admin_import_typed_dataset` (request scope, use `g`), and `tasks.run_hf_import` / `tasks.run_file_tree_import` (Celery — **lazy-import** `_registered_extra_kinds` inside the task to avoid the app↔tasks circular import; do NOT add it to the top-level `from app import (...)`).
- **Registered-kind predictions (bytes-in).** Both GT/input AND pred fields now support registered kinds. The submitter serializes their model output themselves and the client packs it **verbatim** — there is deliberately **no `encode` hook** (the producer owns serialization, exactly as a dataset author produces the GT file; `decode` is the only hook because only the *server* must deserialize to score). Wiring:
  - Client: `benchhub.RawPrediction(kind, data, *, file_ext=None, params=None)` (`.from_file(...)` for a path). `SubmissionBuilder.predict(sample, field=RawPrediction(...))` accepts it alongside `DataType`s; `build_manifest`/`build_zip` derive kind from `.kind` and pack the bytes under the field's ext. The ext comes from the LB **contract** (`/api/leaderboard/<id>/contract` entries are enriched with `file_ext` via `_kind_file_ext`) — so `set_contract()`/`fetch_contract()` (or an explicit `file_ext=`) is required for a registered pred, else `build_zip` raises.
  - Server: `validate_submission_manifest` / `import_typed_submission` take `extra_kinds` (same shape as the dataset importer); the submit route passes `_registered_extra_kinds(lb.owner_user_id)`. Registered pred bytes store verbatim; `get_metric_context` emits a `RegisteredBlob` for the pred CustomField (the `cf.data_type not in DTYPES` branch in the `sub` loop), decoded for the metric just like GT.
  - `check_submission_matches_contract` is kind-string only (no `DTYPES` gate), so it needed no change. `_enforce_shape_constraint` skips registered kinds (no spatial shape).
- A registered kind used by a public LB can't be deleted or made private (`_datatype_used_by_public_lb` guard).

## Editing the LB pred-field schema
- Owner/admin can edit each LB's prediction-field schema on `/edit_leaderboard/<id>` → "Prediction fields" tab. Each row: name (`<x>_pred`), kind (`image`/`mask`/`depth`/`audio`/`scalar`/`text`/`json`/`histogram`), description, remove. Add-row button for extras.
- Frozen once **verified** submissions exist (mirrored PWC submissions don't count) — changing kinds afterwards would silently re-interpret existing prediction files through the wrong decoder. Delete the verified subs to unlock.
- Writes the list to `Leaderboard.required_pred_fields_json`; `_lb_submission_pred_fields` already merges that as an authoritative override of metric-derived entries.
- `_create_lb_from_pwc_benchmark` picks the gt_field for arg_mappings via `_pwc_task_pred_kind_priority(task_name)` — task-aware ordering so image-generation tasks land on the image kind, segmentation on mask, depth on depth, etc. Falls back to the default `(scalar > depth > image > mask > text)` order. Add new patterns there when a future bulk import lands a task type that's not covered.

## Mask vs image disambiguation
Both upload paths now route segmentation masks to `target_kind='mask'` (rendered with the deterministic-hue palette + paired with IoU-family metric defaults) rather than 'image':
- **HF datasets** (`_infer_mapping`): an `Image`-typed column whose name contains any of `mask`, `segmentation`, `segment_map`, `seg_map`, `annotation`, `panoptic`, `label_map`, `semseg` → mask. Tokens live in `_HF_MASK_TOKENS`; check via `_col_name_looks_like_mask(col)`.
- **BH ZIP uploads** (`detect_custom_fields`): folder-name token check short-circuits to mask. Otherwise `_classify_image_path(path)` peeks the first file and inspects PIL `mode` + unique-value/color count (downsampled to 256×256 for speed): mode `P` → mask; mode `L`/`I` with ≤32 unique values → mask; mode `RGB`/`RGBA` with ≤32 unique colors → mask; else image.

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
- `_compute_explorable_lb_ids(lb_ids)` returns the LB IDs whose GT is actually cached: BH dataset Sample rows OR LB-scoped CustomField rows (sample_id+submission_id both NULL — the HF-stub marker rows). Drives the green/yellow pill on `/leaderboards`, `/home`, `/landing`, and the "Explore samples" button label on the LB detail page. (`/explore` is a back-compat redirect to `/leaderboards` since commit `21b5222`; all in-app links use `url_for('leaderboards', ...)`.)
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

## User-owned content visibility
- `GlobalMetric` and `GlobalVisualization` rows created by a non-admin user default to `visibility='private'`. Admins (BENCHHUB_ADMIN_EMAILS / `is_admin` flag) default their new rows to `public`. Owners can flip via the detail-pane select on `/metrics?selected=<id>` and `/visualizations?selected=<id>` (routes: `set_global_metric_visibility`, `set_global_visualization_visibility`).
- **Name uniqueness is two-tier**: `GlobalMetric.name` and `GlobalVisualization.name` are NOT globally unique anymore. Two users can each have a private `my_iou` metric. Two SQLite indexes (added in `check_and_migrate_db`) carry the new contract:
  - `uq_<table>_name_public` — partial unique on `name` WHERE `visibility='public'` (cross-user uniqueness only for public).
  - `uq_<table>_name_per_owner` — composite unique on `(owner_user_id, name)` so a single user can't have two metrics named the same.
- **Promote-to-public collision UX**: `set_global_metric_visibility` (and the viz variant) detects a public-name collision before flipping the row and redirects to `resolve_name_collision.html`, which proposes a `<name>_<N>` suggestion via `_suggest_unique_public_name()` and lets the user edit. The `/visibility/confirm` route is the second hop — it re-checks (and re-suggests if the user tried another taken name) before committing.
- `Leaderboard.canonicality` is **dropped** as a concept. The column stays in the DB for back-compat (no destructive column drop on SQLite) but no code reads it. Visibility (public/private/unlisted) is the only catalog-membership flag. The legacy `admin_promote_leaderboard` route still exists as a back-compat alias that just flips `visibility`; it's owner-OR-admin now, not admin-only.
- `Leaderboard.canonical_for_repo` is **informational metadata** ("this LB tracks the X HF repo"). Multiple LBs may share a repo. Admin-only `admin_set_canonical_for_repo` route lets you adjust the label without affecting visibility.

## FeatureRequest
- New `FeatureRequest` table backs `/feature_requests` (user-facing form + list of own submissions) and `/admin/feature_requests` (admin triage with status + note). Used for new-data-type asks now that we're NOT shipping a user-pluggable field-type system this round.
- Statuses: `open` (default), `planned`, `in_progress`, `resolved`, `declined`. Admin can attach an `admin_note` visible to the requester.

## OAuth
- GitHub + Google are wired through Authlib (`oauth.github`, `oauth.google`). Configure with `GITHUB_CLIENT_ID/SECRET` and `GOOGLE_CLIENT_ID/SECRET` (env vars in dev, `fly secrets set` in prod). Google's authorized redirect URI on the Cloud Console must be `<site>/oauth/callback/google`.
- Apple sign-in is NOT wired up. It needs a signing key + team ID + key ID from the Apple Developer portal, which isn't something the repo can generate. When you set that up, follow the same pattern: register an `oauth.apple` Authlib client with a JWT-generated `client_secret` and add `/login/apple` + `/oauth/callback/apple` routes mirroring the Google ones.

## Depth visualization
- Depth-kind GT thumbs cache as **8-bit grayscale PNG** (normalized 0..255 of the source range). Don't burn a colormap at cache time.
- `/api/gt_viz/<lb_id>/<col>/<sample_name>?cmap=<name>` recolors the gray PNG at view time. Supported names: `turbo`, `jet`, `viridis`, `magma`, `inferno`, `plasma`, `gray`, `normal`. Unknown names fall back to turbo. `normal` computes a Sobel-based tangent-space surface normal map; it's qualitative, not metric (no depth-unit calibration).
- The comparison view's depth column header has a colormap select that rewrites every `.depth-img[data-col-key]` src in that column on change. `serve_custom_field_image` forwards `?cmap=` through its redirect to `serve_gt_viz` since Flask doesn't carry query args across `redirect()` automatically.

## Working notes for future sessions
- **Write decisions down as they happen.** If a fix involves a non-obvious gotcha (CSS leak, framework default, schema quirk, ordering trap), append a bullet under the appropriate section in this file *during the change*, not after. Section anchors: "Things to be careful with", "Frontend conventions", "HF dataset attachment patterns", "Comparison view gotchas", "Metric authoring", "Migration patterns".
- **Treat CLAUDE.md as the durable memory.** Commit messages document one change; this file documents what to know to make the NEXT change.
