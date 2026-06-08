# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Typed-contract architecture (Phases A–D shipped)

The legacy folder-name ZIP path is gone; the strict typed contract is the spine of the system end-to-end. Phases A through D are live in prod — read this section before touching code or writing docs that reference deleted concepts.

**The pipeline now:**

1. **Admin uploads a typed dataset** via `POST /admin/import_typed_dataset` or `scripts/import_typed_dataset.py`. Format: a dir with `manifest.json` + `<field>/<sample>.<ext>`. Materialises Dataset + Sample + CustomField rows, copies file-backed kinds under `uploads/datasets/<id>/`; inline kinds (Scalar, Label) decode into `value_float`/`value_text`.
2. **LB declares its contract** via `Leaderboard.required_pred_fields_json` (`{name, kind, params, role}`; `role ∈ {input,gt,pred}`).
3. **Submitters use `benchhub-client`** — `bh.Client(token)` → `client.submission(lb_id).predict(sample, **kwargs).submit()`. Client validates each `DataType` at stage time, packs a ZIP matching the server's on-disk format, POSTs to `/api/submit/<lb_id>`.
4. **Server validates** the manifest against the contract (every pred name present, kinds match), writes Submission + CustomField pred rows, enqueues `tasks.process_submission`.
5. **Metric engine** builds per-sample context with both primitive (`gt_depth_pred`) AND typed (`__typed__gt_depth_pred`) entries. Metrics declaring `GlobalMetric.input_kinds` (non-empty JSON) get `bh.Depth` instances; legacy metrics keep the primitive. Five typed reference metrics seeded: `accuracy`, `rmse_depth`, `mae_depth`, `iou_mask`, `exact_match_text`.

**Key files:**
- `benchhub/types.py` — 9 `DataType` subclasses (`Image`, `Mask`, `Depth`, `Audio`, `Text`, `BBoxes`, `Label`, `Scalar`, `Json`). Source of truth.
- `benchhub/manifest.py` — manifest spec + `import_typed_dataset` + `import_typed_submission` + `check_submission_matches_contract`.
- `benchhub/client.py` — `Client`, `SubmissionBuilder`, `FlaskTestClientTransport` (in-process transport tests use). `Client.iter_samples(lb_id, *, force_download=False)` pulls all file-backed inputs as ONE bulk ZIP from `/api/leaderboard/<id>/inputs.zip` (route `api_leaderboard_inputs_archive`, `ZIP_STORED`), extracts to `~/.cache/benchhub/<host>/lb_<id>/` (override via `$BENCHHUB_CACHE_DIR`), yields decoded `bh.<Kind>`. Cache keyed on `cache_token` from `/samples` (busts on re-materialise) + sorted `<field>/<sample>` list; `force_download=True` re-fetches. Masks pack raw `.classid.png` → decode to `bh.Mask`, not palette `bh.Image`. Falls back to per-sample `fetch_bytes` if the archive route 404s (older server). Tests isolate the cache — `conftest` points `$BENCHHUB_CACHE_DIR` at session tmp, wipes per test (LB ids repeat).
- `scripts/import_typed_dataset.py` — admin CLI.
- `scripts/seed_reference_metrics.py` — idempotent typed-metric seed.
- `metric_engine.py:_typed_for_cf` / `_stash_typed` / `_metric_wants_typed` — the typed-instance plumbing.

**End-to-end verification:** `tests/test_phase_b_end_to_end.py` exercises the whole loop (typed import → client → typed submit → typed-instance metric eval → asserted MetricResult). 924 passing tests, 0 failures.

## Hybrid storage (Stages A–C shipped)

The catalog now defaults to a lightweight preview tier; full-resolution bytes only land on disk for leaderboards that bind a subset and materialise.

**Two-tier storage:**
- **Preview** (always present): `uploads/datasets/<id>/<field>/<sample>.<ext>` — downscaled+JPG image/mask/depth (max 512px, q85), waveform PNG for audio, inline text/json/scalar/label. `Dataset.preview_only=True`, ~30–50 KB/sample. dataset_view renders from here (visually indistinguishable from full-res).
- **Materialised** (per-LB): `uploads/lb_materializations/<lb_id>/<field>/<sample>.<ext>` — full-res bytes for the LB's chosen subset; counts against the LB owner's quota.

**Per-LB sample selection:** `/create_lb_for_dataset` (+ `/create_lb_chooser`) wizard: `sample_cap`, `sampling` (head/random/stratified), `stratify_field`, `sampling_seed`. Random default; stratified auto-default when the dataset has a `label` field. POST `/create_leaderboard` writes a `LeaderboardMaterialization` + `.delay()`s `tasks.materialize_leaderboard` → `benchhub.lb_materialize.materialize_for_lb` (picks via `pick_samples()`, re-runs `materialize_hf_to_typed_dir` at full res into temp, copies subset to `uploads/lb_materializations/<lb_id>/`, status `ready`). Failures → `status='failed'` + `error_message` + Retry button (`/leaderboard/<id>/materialize/retry`).

**Path resolution at scoring:** for file-backed `gt_<field>`, `extract_viz_arg_value(sample, submission, field_key, *, leaderboard_id=None)` consults `materialized_or_preview_path()` — materialised wins, preview fallback (inline scalar/label/text/json unaffected). `execute_visualization` passes `leaderboard_id=lv.leaderboard_id` so LB viz renders full-res even on a preview-only dataset row.

**Quotas (split-bucket, Phase 13):** two byte caps on `User`:
- `quota_public_max_bytes` — **50 GB** default (was 100 GB; `check_and_migrate_db` backfills old rows). Charged when a `visibility=='public'` row is created/grown (public Datasets + LB materialisations).
- `quota_private_max_bytes` — **10 GB** default. Charged for `visibility in {'private','unlisted',NULL}`.
- `check_quota(user, *, kind='dataset_create', incoming_bytes, visibility=...)` reads the bucket implied by the row; `visibility` is **required** for new code (default `'private'` fails safe on the smaller bucket). `storage_used_bytes(user, *, visibility=...)` partitions per bucket (`None` = legacy total). Helpers cluster in `app.py` (~1763–1944): `_visibility_bucket`, `storage_used_bytes`, `quota_cap_for`, `check_quota`.
- **Publish-flip pre-flight**: `set_dataset_visibility`/`set_leaderboard_visibility` reject a private→public flip when the public bucket can't absorb the moving bytes (flash + 302 back). Admins bypass everything via `is_admin()`.
- Submission ZIPs aren't charged (LB owner already paid for the materialised inputs). Legacy `quota_max_storage_bytes` column stays for old admin tools but `check_quota` ignores it. Gate is pre-flight (refusal) on uploads + publish-flip; post-flight `Dataset.storage_bytes` is the authoritative gauge number.

**Key files:**
- `benchhub/preview.py` — `image_preview`, `depth_preview` (turbo colormap), `mask_preview` (deterministic palette), `audio_preview` (waveform PNG), single dispatch via `render_preview(kind, payload)`.
- `benchhub/manifest.py:import_typed_dataset(..., preview_only=True)` — routes vis modalities through the preview helpers, writes `.jpg`/`.png` instead of canonical extensions, sets `Dataset.preview_only = True`.
- `benchhub/lb_materialize.py` — `pick_samples()` (head/random/stratified) + `materialize_for_lb()` (re-fetches full bytes for the subset) + `materialized_or_preview_path()` (the resolver).
- `tasks.py:materialize_leaderboard` — Celery wrapper around `materialize_for_lb` so big imports don't block the request handler.

**Migration notes:** `Dataset.preview_only` column + `leaderboard_materialization` table added in `check_and_migrate_db`. A one-shot `/tmp/migrate_to_preview.py` converted the 25 pre-existing full-storage datasets in place (18 GB → 1.8 GB; refuses any dataset whose bound LBs have submissions — that needs Stage C materialisation first). `tasks.run_hf_import` does NOT set `preview_only=True` (the bulk LLM loop does) — consider unifying.

## ⚠️ Pre-existing deletions (Phase A delete pile)

Big chunks of legacy machinery were already removed. Read this before touching code or writing docs that reference deleted concepts.

**Deleted (Phase A delete pile, commits `6707189`, `97f4b6c`, `66ffcc6`)** — don't reintroduce:
- Old HuggingFace/PWC import stack (`import_from_hf*`, `admin_pwc_*`, `admin_lb_sota*`, `_VirtualSample`/`_VirtualCustomField`, `_infer_mapping`, `_resolve_hf_split_and_load`/`_HF_SPLIT_PREFERENCE`, `_persist_hf_eval_snapshots`, `_create_lb_from_pwc_benchmark` + PWC helpers, `_HF_MASK_TOKENS`, `pwc_client.py`).
- SOTA/Colab notebook generation (`*_colab_*`, `_llm_generate_*`, `_llm_propose_*`, `_llm_infer_mapping`, `leaderboard_colab_*`).
- `canonicality` concept: `admin_promote_leaderboard` route + UI form. Column stays in DB but no code reads it; `canonical_for_repo` likewise dead (HF-only metadata).
- Folder-prefix ZIP ingest (`detect_custom_fields`, `_classify_image_path`, `_folder_name_prefix_kind`, `_FIELD_TYPE_PREFIXES`, `process_*_zip`, `upload_dataset`/`upload_submission` routes). Upload UI → "paused" placeholder pointing at `/supported_types`.
- Legacy per-sample model classes `HistogramData`/`SignalShape`/`ConfigData` (SQLite tables stay; `Sample.histogram_data`/`.signal_shape`/`.config_data` resolve to `None`). 26 test files removed.
- `/explore` → back-compat 302 to `/leaderboards`; the catalog lives only there.

⚠️ **This list is stale**: some named symbols (`_infer_mapping`, `_VirtualSample`, `_HF_SPLIT_PREFERENCE`, `_pwc_task_to_category`, `populate_lb_samples`, `process_submission_zip`, `detect_custom_fields`, `_llm_generate_metric_code`) reappear in current `app.py`/`metric_engine.py` — a later import system (agent-mode + file-tree, see `scripts/import_hf_agent.py`, `benchhub/file_tree_import.py`) was built on the cleared ground. Verify against code before trusting any "deleted"/"live" claim below.

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

**Tests** under `tests/` (60+ files), run `pytest tests/`. Fixtures in `tests/conftest.py`: per-session `app` (Flask+Celery in TEST mode, `task_always_eager=True`); per-test `db_session` (drop+recreate all tables → independent); `auth_client` (client with `session['user_id']` = fresh `logged_in_user`); `make_zip(name, layout, root_folder=...)`. `BENCHHUB_DATA_DIR` → per-session tempdir so tests never touch `~/.dtofbenchmarking`. Root-level `test_chain*.py` are ad-hoc Celery experiments, NOT in the suite — use `pytest tests/`, not bare `pytest`.

No lint/build wired up. Deps: `pip install -r requirements.txt` (Flask, Flask-SQLAlchemy, celery, redis, numpy, scipy, matplotlib, Pillow, h5py, soundfile, …); `pytest` isn't pinned — install separately.

## Deployment

**Self-hosted** on a home Ubuntu 24.04 box at `runbenchhub.com`: gunicorn + celery + redis under systemd (no Docker), nginx + certbot, Cloudflare DNS-only, `ddclient` for DDNS. Full runbook (push flow, `.env` keys, logs, rollback): **`docs/SELFHOST_RUNBOOK.md`** — read before suggesting ops commands. Claude Code runs **on the box** (the runbook's `ssh` step is skippable). Two checkouts:

| Path | Role |
|---|---|
| `~/Git/BenchHub` (cwd) | **Dev** — edits + commits land here; does NOT touch the live app. |
| `~/benchhub` | **Prod** — what gunicorn/celery serve. Updated only via `git pull`. Never edit directly (next `git pull` clobbers it). |

Deploy = commit + push from `~/Git/BenchHub`, then from anywhere on the box:

```bash
cd ~/benchhub && git pull
sudo systemctl restart benchhub-web        # ⚠️ unit has NO ExecReload — `reload` errors ("Job type reload is not applicable"); use restart
sudo systemctl restart benchhub-celery     # celery has no SIGHUP code-reload either
# .env change / new schema column → restart both (HUP wouldn't re-read env or migrate anyway)
sudo systemctl restart benchhub-web benchhub-celery
# logs: journalctl -u benchhub-web -f   /   -u benchhub-celery -f
```

`BENCHHUB_AUTO_MIGRATE=1` in `.env` runs `check_and_migrate_db()` on every boot, so a model-column ALTER applies on `restart`. Secrets live only in `~/benchhub/.env`.

**Fly.io is dead** — artifacts moved to `archive/fly/`; don't suggest any `fly *` command. `runner/{Dockerfile,harness.py,server.py}` stay in place (local sandbox tests `tests/test_sandbox_*` reference them).

## Data and config locations

- **DB + uploads live OUTSIDE the repo**, at `~/.dtofbenchmarking/` (`database.db` and `uploads/`). The empty `database.db` and `uploads/` in the repo are vestigial — do not assume they are the live ones.
- `local_config.py` (gitignored-style, may not exist on a fresh clone) provides `GIT_REPO_PATH` for the *external* git repo from which submission/dataset author info is extracted via `git log`. It is imported optionally; if missing, `GIT_REPO_PATH = None`.
- Global UI settings (column widths, theme) are stored as JSON in `~/.dtofbenchmarking/settings.json` via the `GlobalSettings` singleton.

## Architecture

### One-file Flask app
`app.py` (~6600 lines) holds nearly everything: models, all routes, ZIP processing, DB migrations, custom-field detection, viz rendering. Prefer editing `app.py` over new modules — there's no layered structure to slot into. Helpers outside:
- `metric_engine.py` — `evaluate_dynamic_metric` (exec's user code), `get_metric_context` (assembles metric kwargs), `sort_metrics_by_dependency` (Kahn topo sort so metric B consumes metric A's output).
- `tasks.py` — Celery tasks. **Circular-import shape**: `tasks.py` imports from `app`; `app.py` lazily imports `tasks` inside route handlers (`tasks.process_submission.delay`). Don't move task defs into `app.py` or rearrange imports without understanding this.
- `metric_routes.py` — orphaned legacy snippets (uses `@app.route` without importing `app`); dead code, equivalents live in `app.py`.

### Domain model (`app.py` ~270–510)
- **Project** → has many **Leaderboard**s. `Project` is just a namespace; URLs are prefixed with `/<project_name>/...` and resolved via `@app.before_request load_project_context` (cookie-fallback to `active_project_id`).
- **Dataset** is **global** (not project-scoped, despite older code comments). Linked to leaderboards via the `leaderboard_datasets` association table — a leaderboard can have multiple datasets. The legacy `Leaderboard.dataset_id` column is deprecated but still populated for back-compat.
- **Sample** belongs to a Dataset. `HistogramData`, `SignalShape`, `ConfigData` are legacy per-sample tables; new data flows through `CustomField` instead. The `Sample.histogram_data` / `Sample.signal_shape` Python @properties shadow the SQLAlchemy relationships and transparently fall back to `CustomField` rows — be aware when querying.
- **CustomField** — unified dynamically-typed bag for per-sample (Dataset) and per-(submission,sample) data. `field_type ∈ {image,scalar,metric,histogram,depth,json,text}`. Per-sample metric *outputs* also land here as `name=f"lm_{leaderboard_metric_id}"`, which lets `reaggregate_submission_metrics` re-pool without re-running user code.
- **GlobalMetric**/**GlobalVisualization** store user Python source. **LeaderboardMetric**/**LeaderboardVisualization** link a global def to an LB with `arg_mappings` (arg → context key), `target_name` (dependency-chaining alias), `pooling_type` (`mean|median|percentile|min|max`), `sort_direction`, `tag_filter`.
- **MetricResult** stores the final aggregated scalar per (submission, leaderboard_metric).

### Submission processing pipeline
1. Upload (`upload_submission`) → `process_submission_zip` extracts to `uploads/submissions/<id>/`, runs `detect_custom_fields`.
2. `tasks.process_submission.delay(sub.id)` enqueues.
3. `_process_submission_impl` (`tasks.py`): builds per-sample context via `get_metric_context` (GT + submission CFs + on-the-fly histogram entropy); topo-sorts `LeaderboardMetric`s (`sort_metrics_by_dependency`) and merges outputs back into context; per-sample metrics write each value to a `CustomField` (so re-aggregation skips re-exec) then pool via `pooling_type`; aggregated metrics get the full value list for non-aggregated deps / scalar for aggregated deps; pre-caches aggregated viz; updates `Submission.processing_status` granularly (`Pending`→`Processing: Metric N/M`→`Generating Visualizations`→`Processed`/`Error:`).
4. Batch recalc uses `process_submissions_batch_sequential` — one-at-a-time on purpose (concurrency rolled back in `8a77b48`; don't re-add `group()`/`chord()` without checking why).

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

`_folder_name_prefix_kind(folder_name)` is **authoritative** when the prefix matches; content-peek heuristics (`_classify_image_path`, `_classify_npz`) run only for unrecognised prefixes (legacy back-compat). `tags` folder is hardcoded text. Mirrored on HF import: `_infer_mapping` emits `target_field=<kind>_<col>` (not double-prefixed if the col already starts with the kind). `git_info.json`/`git.info` at ZIP root → commit metadata; missing `author` ⇒ `get_author_from_git_commit` shells `git -C $GIT_REPO_PATH log origin/<branch>`.

### Frontend
Server-rendered Jinja templates in `templates/` (no framework, vanilla JS). The big screens are `leaderboard.html`, `comparison.html`, `dataset_view.html`, `edit_leaderboard.html`. Static assets are minimal (`static/css/`, `static/js/`).

### DLP-safe code path
Some networks block `.py` uploads. The metric editor encodes code as `BASE64:<...>` client-side; `handle_dlp_safe_code` decodes server-side. `scripts/obfuscator.html` + `scripts/obfuscator_gui.py` are standalone helpers. Preserve this when touching metric upload/edit endpoints.

### DB migrations
No Alembic. `check_and_migrate_db()` (called from `if __name__ == '__main__':`) runs raw SQLite `PRAGMA table_info` + `ALTER TABLE ... ADD COLUMN` against `~/.dtofbenchmarking/database.db` on every startup — add a block here for every new model column or existing installs break. SQLite opened `journal_mode=WAL`, 120s `busy_timeout`.

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
(Much of this names Phase-A-era HF/PWC code — verify against current `app.py` before relying on it.)
- **`_HF_SPLIT_PREFERENCE = ['test','validation','val','dev','train']`**. `_resolve_hf_split_and_load(att, load_fn)` walks the order, probes row 0 to verify mapped GT cols aren't all null, falls back to first loadable split, and **persists the resolved split via `_persist_resolved_split`** so the LB badge tells the truth.
- **`_infer_mapping(features)`**: string→text, Audio→audio, else→json. `Value:unknown` (HF's flattened nested types, e.g. DocRED `sents`/`vertexSet`/`labels`) → json, not skip. Changing it ⇒ check `_persist_hf_eval_snapshots` + `_virtual_sample_from_hf_row` persist the new kind.
- **`_pwc_task_to_category`** strips domain prefixes (Medical, Aerial, Few-Shot, …) before classifying ("Medical Image Segmentation" → "Vision/Image Segmentation"). New prefixes → `_DOMAIN_PREFIXES`, shortest-first.
- **`populate_lb_samples` has a 5-min `soft_time_limit`** — `suggest_hf_repo` can land on a monolithic 100GB+ HDF5 repo and hang the worker. Flask+Celery+Redis share one box; don't bulk-enqueue populate tasks (site goes unresponsive) — use the per-LB "Populate samples" button or rate-limit.

## Input vs GT roles on dataset columns
Each mapping entry carries optional `role`: `input` (conditioning given to submitter, NOT predicted) or `gt` (held server-side, prediction target). Default `gt` (back-compat). `_pwc_task_input_kinds(task_name)` returns the `target_kind`s to flag `input`. `_lb_submission_pred_fields` drops pred fields whose GT col is `input` (so `label_pred` won't appear in an Image-Generation contract). Owner-editable on `/edit_leaderboard/<id>` → Prediction-fields → "Dataset field roles"; frozen once verified subs exist. `LeaderboardMetric.arg_mappings` must reference a GT-role column or the metric has no valid pred field (`.tag_input_gt.py` one-shot rewrites these per task).

## Metric / Visualization input-kind declarations
`GlobalMetric.input_kinds` / `GlobalVisualization.input_kinds` are nullable JSON arrays of accepted `target_kind`s in arg order (NULL = unconstrained). `/metrics?selected=<id>` shows an "Accepts: <kind>×<kind>" row; backfilled for 18 metrics in `.backfill_input_kinds.py` (add patterns to `KIND_HINTS` there). The LB→metric binding UI doesn't yet *enforce* kinds.

## User-registered data types (`DataTypeDef`) + the decode hook
- Register a new `kind` BenchHub doesn't ship (NIfTI, point clouds, EEG, …) via **`/datatypes/new`** (route `datatype_new`; `/supported_types` "Add a data type" links here) or `client.create_datatype(...)`. POST `/datatypes/create` (`create_datatype_web`) → back to `/datatypes/new` on error, `/supported_types` on success; `/datatypes` 302s to `/supported_types`.
- `DataTypeDef` columns: `name` (globally unique, shares the `DTYPES` namespace), `file_ext` (NULL ⇒ inline text), `viz_mime`, `visualize_code` (`def visualize(blob, params)->PIL.Image`), optional **`decode_code`** (`def decode(blob, params)->object`), `owner_user_id`, `visibility`. Storage is **bytes-verbatim** (encode=identity); `visualize`/`decode` run **only in the sandbox**.
- **Decode hook = deserialize side of the contract.** With `decode_code` set, a consuming metric gets the decoded object (mirrors `bh.Depth.array`); absent ⇒ raw bytes. Wiring: `metric_engine.RegisteredBlob` (`kind,blob,params,decode_code`) is the carrier — `get_metric_context` emits it for any GT/input/pred CustomField whose `data_type` ∉ `DTYPES` (via `_registered_blob_for_cf`, lazy `from app import DataTypeDef`). Sandbox: `_jsonify_kwarg` → `{"__dtype__","decode","params","b64"}`, `runner/harness.py:_decode_arg` runs `decode` inside the metric's own container. In-process: `evaluate_dynamic_metric` resolves via `_resolve_registered_blob`.
- **Import admission**: `benchhub.manifest` can't see `DataTypeDef`, so `validate_manifest`/`load_manifest`/`expected_file_path`/`import_typed_dataset` take optional **`extra_kinds={name: file_ext}`** (accepted, stored verbatim, no preview). Server passes `app._registered_extra_kinds(owner_user_id)` (public + owner's own) at 4 sites: `_ingest_typed_dataset_zip`, `admin_import_typed_dataset` (request scope → `g`), `tasks.run_hf_import`/`tasks.run_file_tree_import` (Celery — **lazy-import** inside the task; do NOT add to top-level `from app import`, circular).
- **Registered-kind predictions (bytes-in).** Both GT/input AND pred support registered kinds. Submitter serializes their output; client packs **verbatim** — deliberately **no `encode` hook** (producer owns serialization; `decode` is the only hook since only the *server* deserializes to score). Client: `benchhub.RawPrediction(kind, data, *, file_ext=None, params=None)` (`.from_file(...)`); `SubmissionBuilder.predict(sample, field=RawPrediction(...))` packs bytes under the field's ext, which comes from the LB **contract** (`/contract` entries enriched with `file_ext` via `_kind_file_ext`) — so `set_contract()`/`fetch_contract()` (or explicit `file_ext=`) required or `build_zip` raises. Server: `validate_submission_manifest`/`import_typed_submission` take `extra_kinds`; submit route passes `_registered_extra_kinds(lb.owner_user_id)`. `check_submission_matches_contract` is kind-string only (no change); `_enforce_shape_constraint` skips registered kinds.
- A registered kind used by a public LB can't be deleted or made private (`_datatype_used_by_public_lb` guard).

## Editing the LB pred-field schema
- Owner/admin can edit each LB's prediction-field schema on `/edit_leaderboard/<id>` → "Prediction fields" tab. Each row: name (`<x>_pred`), kind (`image`/`mask`/`depth`/`audio`/`scalar`/`text`/`json`/`histogram`), description, remove. Add-row button for extras.
- Frozen once **verified** submissions exist (mirrored PWC submissions don't count) — changing kinds afterwards would silently re-interpret existing prediction files through the wrong decoder. Delete the verified subs to unlock.
- Writes the list to `Leaderboard.required_pred_fields_json`; `_lb_submission_pred_fields` already merges that as an authoritative override of metric-derived entries.
- `_create_lb_from_pwc_benchmark` picks the gt_field for arg_mappings via `_pwc_task_pred_kind_priority(task_name)` — task-aware ordering so image-generation tasks land on the image kind, segmentation on mask, depth on depth, etc. Falls back to the default `(scalar > depth > image > mask > text)` order. Add new patterns there when a future bulk import lands a task type that's not covered.

## Mask vs image disambiguation
Both upload paths route segmentation masks to `target_kind='mask'` (deterministic-hue palette + IoU-family defaults), not 'image':
- **HF** (`_infer_mapping`): an `Image` column whose name contains `mask`/`segmentation`/`segment_map`/`seg_map`/`annotation`/`panoptic`/`label_map`/`semseg` → mask (`_HF_MASK_TOKENS`, via `_col_name_looks_like_mask`).
- **BH ZIP** (`detect_custom_fields`): folder-name token short-circuits to mask; else `_classify_image_path` peeks first file (downsampled 256×256): PIL mode `P` → mask; `L`/`I` with ≤32 unique values → mask; `RGB`/`RGBA` with ≤32 unique colors → mask; else image.

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
- `_compute_explorable_lb_ids(lb_ids)` = LB IDs whose GT is cached: BH dataset Sample rows OR LB-scoped CustomField rows (sample_id+submission_id both NULL — HF-stub markers). Drives the green/yellow pill on `/leaderboards`,`/home`,`/landing` + the "Explore samples" label. (`/explore` → 302 to `/leaderboards` since `21b5222`; all links use `url_for('leaderboards', ...)`.)
- An LB with `canonical_for_repo IS NOT NULL` + zero GT CFs is **broken** — show the owner-only "Populate samples" button, not an empty Explore page.
- **`/datasets` has two sections**: regular `Dataset` rows + "Cached HuggingFace datasets" from distinct `Attachment.hf_repo_id` whose owning LB is explorable (each links to the first LB's Explore view — filter is intentional, else dead links).

## Metric authoring
- **LLM-authored metrics (`_llm_generate_metric_code`) aren't safe verbatim** for non-trivial cases (rank-based, span-overlap, BLEU) — they mix scalar-vs-list logic and return the wrong number. Rank LBs (Link Prediction) needed manual `_rank_of_gt(gt, pred_list)` rewrites; spot-check new ones.
- Metrics stored as source on `GlobalMetric.python_code`, exec'd in `evaluate_dynamic_metric` with `numpy as np`. Replace cleanly by overwriting `python_code` (no migration).

## Migration patterns
- **Every model column add needs an `ALTER TABLE ... ADD COLUMN` block in `check_and_migrate_db()`** (no Alembic). Recent: `Leaderboard.category` (two-level "Area/Task").
- **Idempotent data backfill goes in the same block** after the ALTER, gated on "any rows still NULL". Template: PWC-category backfill at `--- 3b. ---` — probe optional resources, best-effort match, swallow exceptions so fresh installs aren't blocked.

## Tests
**Add a test next to the closest existing one when you fix a bug.** Map of touched code → test file: `_pwc_task_to_category`/`_DOMAIN_PREFIXES` → `test_pwc_category.py`; `_resolve_hf_split_and_load` → `test_hf_split_resolver.py`; `_infer_mapping` → `test_pwc_import.py`/`test_hf_features_fallback.py`; `_compute_explorable_lb_ids` → `test_explorable.py`; `_VirtualSample` dispatch → `test_attachment_iter.py`; `get_metric_context` text/json/topk_list → `test_metric_context_arrays.py`; samples-only `comparison_view` → `test_routes_comparison.py`. Run `pytest tests/` (not bare `pytest` — avoids root-level `test_chain*.py`).

## User-owned content visibility
- `GlobalMetric`/`GlobalVisualization` rows default to `visibility='private'` for non-admins, `public` for admins (BENCHHUB_ADMIN_EMAILS / `is_admin`). Owners flip via the detail-pane select on `/metrics?selected=<id>` / `/visualizations?selected=<id>` (`set_global_metric_visibility`, `set_global_visualization_visibility`).
- **Name uniqueness is two-tier** (names no longer globally unique — two users can each have a private `my_iou`). Two SQLite indexes in `check_and_migrate_db`: `uq_<table>_name_public` (partial unique on `name` WHERE `visibility='public'`) + `uq_<table>_name_per_owner` (composite on `(owner_user_id, name)`).
- **Promote-to-public collision UX**: the visibility route detects a public-name collision pre-flip and redirects to `resolve_name_collision.html`, which proposes `<name>_<N>` via `_suggest_unique_public_name()`; the `/visibility/confirm` second hop re-checks before committing.
- `Leaderboard.canonicality` **dropped** — column stays in DB unread; visibility (public/private/unlisted) is the only catalog-membership flag. `admin_promote_leaderboard` is a back-compat alias that just flips `visibility` (owner-OR-admin now). `Leaderboard.canonical_for_repo` is informational ("tracks X HF repo", multiple LBs may share); admin-only `admin_set_canonical_for_repo` edits it without touching visibility.

## FeatureRequest
- New `FeatureRequest` table backs `/feature_requests` (user-facing form + list of own submissions) and `/admin/feature_requests` (admin triage with status + note). Used for new-data-type asks now that we're NOT shipping a user-pluggable field-type system this round.
- Statuses: `open` (default), `planned`, `in_progress`, `resolved`, `declined`. Admin can attach an `admin_note` visible to the requester.

## OAuth
- GitHub + Google via Authlib (`oauth.github`, `oauth.google`), configured with `GITHUB_CLIENT_ID/SECRET` + `GOOGLE_CLIENT_ID/SECRET` (env vars). Google's Cloud Console redirect URI must be `<site>/oauth/callback/google`.
- Apple sign-in NOT wired (needs signing key + team/key ID from the Apple portal). To add: register `oauth.apple` with a JWT-generated `client_secret` + `/login/apple` + `/oauth/callback/apple` mirroring Google.

## Depth visualization
- Depth-kind GT thumbs cache as **8-bit grayscale PNG** (normalized 0..255 of the source range). Don't burn a colormap at cache time.
- `/api/gt_viz/<lb_id>/<col>/<sample_name>?cmap=<name>` recolors the gray PNG at view time. Supported names: `turbo`, `jet`, `viridis`, `magma`, `inferno`, `plasma`, `gray`, `normal`. Unknown names fall back to turbo. `normal` computes a Sobel-based tangent-space surface normal map; it's qualitative, not metric (no depth-unit calibration).
- The comparison view's depth column header has a colormap select that rewrites every `.depth-img[data-col-key]` src in that column on change. `serve_custom_field_image` forwards `?cmap=` through its redirect to `serve_gt_viz` since Flask doesn't carry query args across `redirect()` automatically.

## Working notes for future sessions
- **Write decisions down as they happen.** If a fix involves a non-obvious gotcha (CSS leak, framework default, schema quirk, ordering trap), append a bullet under the appropriate section in this file *during the change*, not after. Section anchors: "Things to be careful with", "Frontend conventions", "HF dataset attachment patterns", "Comparison view gotchas", "Metric authoring", "Migration patterns".
- **Treat CLAUDE.md as the durable memory.** Commit messages document one change; this file documents what to know to make the NEXT change.
