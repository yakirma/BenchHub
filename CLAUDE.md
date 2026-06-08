# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo. **This file loads into context every session — keep it lean** (see "Keeping this file lean" at the bottom). Deep per-subsystem detail lives in subdirectory `CLAUDE.md` files that load only when you touch that subtree.

## Detailed guidance (loads on demand)
Claude Code loads a subdirectory `CLAUDE.md` only when you read/edit files in that subtree, so these cost no context on a typical session:
- **`benchhub/CLAUDE.md`** — typed-contract package internals: `types`/`DTYPES`, the metric-authoring signature convention, `manifest`/`client`/`preview`/`lb_materialize`, the `DataTypeDef` decode hook + `RawPrediction`.
- **`runner/CLAUDE.md`** — the metric/viz sandbox (harness, decode-in-container, the `BENCHHUB_SANDBOX_METRICS` execution model).
- **`tests/CLAUDE.md`** — conftest fixtures, cache isolation, the bug→test-file map.
- **`templates/CLAUDE.md`** — palette, theme/layout conventions, comparison-view gotchas.

## Project overview
BenchHub is a Flask + Celery + SQLite web app for benchmarking model predictions against curated datasets. The legacy folder-name ZIP-ingest path is gone; the **typed contract** in `benchhub.types` is the spine of the system. App runs on `http://localhost:6060`.

## Typed contract (the spine) — app-side summary
Package internals live in `benchhub/CLAUDE.md`. What you need when editing `app.py`/routes:
- 10 `DataType` kinds (`Image`,`Mask`,`Depth`,`Audio`,`Text`,`BBoxes`,`Label`,`LabelList`,`Scalar`,`Json`) + the `DTYPES` registry; `/supported_types` is driven from `DTYPES` so it can't drift.
- Pipeline: admin imports a typed dataset (`manifest.json` + `<field>/<sample>.<ext>`) → LB declares `required_pred_fields_json` (`{name,kind,params,role}`, `role∈{input,gt,pred}`) → `benchhub-client` validates + packs a ZIP + POSTs `/api/submit/<lb_id>` → server validates manifest vs contract, writes Submission + CustomField pred rows, enqueues `tasks.process_submission` → metric engine builds per-sample context with primitive (`gt_depth_pred`) + typed (`__typed__gt_depth_pred`) entries. Metrics declaring `GlobalMetric.input_kinds` get `bh.<Kind>` instances; legacy metrics keep the primitive.

## Hybrid storage (preview tier + per-LB materialisation)
Catalog defaults to a lightweight preview tier; full-res bytes only land for LBs that bind a subset and materialise. Internals (preview.py, lb_materialize.py, the materialize task) → `benchhub/CLAUDE.md`.
- **Preview** (always): `uploads/datasets/<id>/<field>/<sample>.<ext>` — downscaled+JPG image/mask/depth (≤512px, q85), waveform PNG audio, inline text/json/scalar/label. `Dataset.preview_only=True`, ~30–50 KB/sample. dataset_view renders from here.
- **Materialised** (per-LB): `uploads/lb_materializations/<lb_id>/<field>/<sample>.<ext>` — full-res for the LB's chosen subset; counts against the LB owner's quota.
- **Per-LB selection**: `/create_lb_for_dataset` (+ `/create_lb_chooser`) wizard (`sample_cap`, `sampling` head/random/stratified, `stratify_field`, `sampling_seed`; random default, stratified auto when a `label` field exists). POST `/create_leaderboard` writes a `LeaderboardMaterialization` + `.delay()`s `tasks.materialize_leaderboard`. Failures → `status='failed'` + `error_message` + Retry (`/leaderboard/<id>/materialize/retry`).
- **Path resolution at scoring**: file-backed `gt_<field>` go through `extract_viz_arg_value(..., leaderboard_id=None)` → `materialized_or_preview_path()` (materialised wins, preview fallback; inline kinds unaffected). `execute_visualization` passes `leaderboard_id=lv.leaderboard_id` so LB viz renders full-res on a preview-only row.
- **Quotas (split-bucket, Phase 13)**: two caps on `User` — `quota_public_max_bytes` (**50 GB** default, was 100 GB; charged when a `visibility=='public'` row is created/grown — public Datasets + LB materialisations) and `quota_private_max_bytes` (**10 GB**, for `private`/`unlisted`/NULL). `check_quota(user, *, kind, incoming_bytes, visibility=...)` reads the bucket implied by the row (`visibility` **required** for new code, default `'private'` = fail-safe small bucket); `storage_used_bytes(user, *, visibility=...)` partitions (`None`=legacy total). Helpers cluster in `app.py` ~1763–1944 (`_visibility_bucket`, `storage_used_bytes`, `quota_cap_for`, `check_quota`). Publish-flip pre-flight (`set_dataset_visibility`/`set_leaderboard_visibility`) rejects private→public when the public bucket can't absorb the bytes (admins bypass via `is_admin()`). Submission ZIPs aren't charged. Legacy `quota_max_storage_bytes` column ignored. Migrations (`Dataset.preview_only`, `leaderboard_materialization`) in `check_and_migrate_db`.

## ⚠️ Pre-existing deletions (Phase A delete pile)
Big chunks of legacy machinery were removed (commits `6707189`, `97f4b6c`, `66ffcc6`). Don't reintroduce:
- Old HuggingFace/PWC import stack (`import_from_hf*`, `admin_pwc_*`, `admin_lb_sota*`, `_VirtualSample`, `_infer_mapping`, `_resolve_hf_split_and_load`, `_create_lb_from_pwc_benchmark`, `pwc_client.py`).
- SOTA/Colab notebook generation (`*_colab_*`, `_llm_generate_*`, `_llm_propose_*`, `leaderboard_colab_*`).
- `canonicality`: `admin_promote_leaderboard` route + UI form. Column stays in DB unread; `canonical_for_repo` likewise dead.
- Folder-prefix ZIP ingest (`detect_custom_fields`, `_classify_image_path`, `process_*_zip`, `upload_dataset`/`upload_submission` routes). Upload UI → "paused" placeholder pointing at `/supported_types`.
- Legacy per-sample model classes `HistogramData`/`SignalShape`/`ConfigData` (SQLite tables stay; `Sample.histogram_data`/`.signal_shape`/`.config_data` resolve to `None`). 26 test files removed. `/explore` → 302 to `/leaderboards`.

⚠️ **This list is stale**: several named symbols (`_infer_mapping`, `_VirtualSample`, `_HF_SPLIT_PREFERENCE`, `_pwc_task_to_category`, `populate_lb_samples`, `process_submission_zip`, `detect_custom_fields`, `_llm_generate_metric_code`) reappear in current `app.py`/`metric_engine.py` — a later import system (agent-mode + file-tree, see `scripts/import_hf_agent.py`, `benchhub/file_tree_import.py`) was built on the cleared ground. Verify against code before trusting any "deleted"/"live" claim.

## Running the app
```bash
redis-server                                          # broker + result backend (6379)
celery -A app.celery worker --loglevel=info           # worker, in repo root
python app.py                                          # Flask app
```
Tests under `tests/` (60+ files): `pytest tests/` (not bare `pytest`). Fixtures + the bug→test map → `tests/CLAUDE.md`. No lint/build wired up. Deps: `pip install -r requirements.txt` (Flask, Flask-SQLAlchemy, celery, redis, numpy, scipy, matplotlib, Pillow, h5py, soundfile, …); `pytest` isn't pinned — install separately.

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
`BENCHHUB_AUTO_MIGRATE=1` in `.env` runs `check_and_migrate_db()` on every boot, so a model-column ALTER applies on `restart`. Secrets live only in `~/benchhub/.env`. **Fly.io is dead** — artifacts in `archive/fly/`; don't suggest any `fly *` command. (`runner/{Dockerfile,harness.py,server.py}` are NOT Fly — see `runner/CLAUDE.md`.)

## Data and config locations
- **DB + uploads live OUTSIDE the repo**, at `~/.dtofbenchmarking/` (`database.db`, `uploads/`). The empty `database.db`/`uploads/` in the repo are vestigial.
- `local_config.py` (may not exist on a fresh clone) provides `GIT_REPO_PATH` for the *external* git repo that submission/dataset author info is extracted from via `git log`. Imported optionally; missing ⇒ `GIT_REPO_PATH = None`.
- Global UI settings (column widths, theme) stored as JSON in `~/.dtofbenchmarking/settings.json` via the `GlobalSettings` singleton.

## Architecture
### One-file Flask app
`app.py` (~6600 lines) holds nearly everything: models, all routes, ZIP processing, DB migrations, custom-field detection, viz rendering. Prefer editing `app.py` over new modules — there's no layered structure to slot into. Helpers outside:
- `metric_engine.py` — `evaluate_dynamic_metric` (exec's user code), `get_metric_context` (assembles metric kwargs), `sort_metrics_by_dependency` (Kahn topo sort so metric B consumes metric A's output).
- `tasks.py` — Celery tasks. **Circular-import shape**: `tasks.py` imports from `app`; `app.py` lazily imports `tasks` inside route handlers (`tasks.process_submission.delay`). Don't move task defs into `app.py` or rearrange imports without understanding this.
- `metric_routes.py` — orphaned legacy snippets (uses `@app.route` without importing `app`); dead code, equivalents live in `app.py`.

### Domain model (`app.py` ~270–510)
- **Project** → has many **Leaderboard**s. `Project` is just a namespace; URLs are prefixed `/<project_name>/...` and resolved via `@app.before_request load_project_context` (cookie-fallback to `active_project_id`).
- **Dataset** is **global** (not project-scoped, despite older comments). Linked to LBs via the `leaderboard_datasets` association table (an LB can have multiple datasets). Legacy `Leaderboard.dataset_id` deprecated but still populated.
- **Sample** belongs to a Dataset. `HistogramData`/`SignalShape`/`ConfigData` are legacy per-sample tables; new data flows through `CustomField`. `Sample.histogram_data`/`.signal_shape` Python @properties shadow the SQLAlchemy relationships and fall back to `CustomField` rows — be aware when querying.
- **CustomField** — unified dynamically-typed bag for per-sample (Dataset) and per-(submission,sample) data. `field_type ∈ {image,scalar,metric,histogram,depth,json,text}`. Per-sample metric *outputs* also land here as `name=f"lm_{leaderboard_metric_id}"`, which lets `reaggregate_submission_metrics` re-pool without re-running user code.
- **GlobalMetric**/**GlobalVisualization** store user Python source. **LeaderboardMetric**/**LeaderboardVisualization** link a global def to an LB with `arg_mappings` (arg → context key), `target_name` (dependency-chaining alias), `pooling_type` (`mean|median|percentile|min|max`), `sort_direction`, `tag_filter`.
- **MetricResult** stores the final aggregated scalar per (submission, leaderboard_metric).

### Submission processing pipeline
1. Upload (`upload_submission`) → `process_submission_zip` extracts to `uploads/submissions/<id>/`, runs `detect_custom_fields`.
2. `tasks.process_submission.delay(sub.id)` enqueues.
3. `_process_submission_impl` (`tasks.py`): builds per-sample context via `get_metric_context` (GT + submission CFs + on-the-fly histogram entropy); topo-sorts `LeaderboardMetric`s (`sort_metrics_by_dependency`) and merges outputs back into context; per-sample metrics write each value to a `CustomField` (so re-aggregation skips re-exec) then pool via `pooling_type`; aggregated metrics get the full value list for non-aggregated deps / scalar for aggregated deps; pre-caches aggregated viz; updates `Submission.processing_status` granularly (`Pending`→`Processing: Metric N/M`→`Generating Visualizations`→`Processed`/`Error:`).
4. Batch recalc uses `process_submissions_batch_sequential` — one-at-a-time on purpose (concurrency rolled back in `8a77b48`; don't re-add `group()`/`chord()` without checking why).

### Folder convention for ZIPs (`<type>_<field_name>`)
Canonical naming for any dataset/submission folder is `<type>_<field_name>`. Recognised type prefixes live in `_FIELD_TYPE_PREFIXES`:

| Type | Folder example | File(s) |
|------|----------------|---------|
| `image` | `image_rgb` | `<sample>.png`/`.jpg`/`.jpeg`/`.bmp`/`.tiff` |
| `mask` | `mask_annotation` | `<sample>.png` (single-channel class IDs or low-color RGB) |
| `depth` | `depth_gt` | `<sample>.npz` (key `depth`, HxW float) |
| `audio` | `audio_clip` | `<sample>.wav`/`.mp3`/`.flac` |
| `scalar` | `scalar_score` | `<sample>.txt` (one float) |
| `text` | `text_caption` | `<sample>.txt` |
| `json` | `json_bbox` | `<sample>.json` |
| `histogram` | `histogram_dtof` | `<sample>.npz` (`bins`, `counts`) |
| `metric` | `metric_iou` | `<sample>.txt` (pre-computed) |

`_folder_name_prefix_kind(folder_name)` is **authoritative** when the prefix matches; content-peek heuristics (`_classify_image_path`, `_classify_npz`) run only for unrecognised prefixes (legacy back-compat). `tags` folder is hardcoded text. Mirrored on HF import: `_infer_mapping` emits `target_field=<kind>_<col>` (not double-prefixed if the col already starts with the kind). `git_info.json`/`git.info` at ZIP root → commit metadata; missing `author` ⇒ `get_author_from_git_commit` shells `git -C $GIT_REPO_PATH log origin/<branch>`.

### Frontend
Server-rendered Jinja in `templates/` (vanilla JS, no framework). Palette, theme/layout conventions, and comparison-view gotchas → `templates/CLAUDE.md`.

### DLP-safe code path
Some networks block `.py` uploads. The metric editor encodes code as `BASE64:<...>` client-side; `handle_dlp_safe_code` decodes server-side. `scripts/obfuscator.html` + `scripts/obfuscator_gui.py` are standalone helpers. Preserve this when touching metric upload/edit endpoints.

### DB migrations
No Alembic. `check_and_migrate_db()` (called from `if __name__ == '__main__':`) runs raw SQLite `PRAGMA table_info` + `ALTER TABLE ... ADD COLUMN` against `~/.dtofbenchmarking/database.db` on every startup — add a block here for every new model column or existing installs break. SQLite opened `journal_mode=WAL`, 120s `busy_timeout`. **Idempotent data backfill** goes in the same block after the ALTER, gated on "any rows still NULL"; template = PWC-category backfill at `--- 3b. ---` (probe optional resources, best-effort match, swallow exceptions so fresh installs aren't blocked).

## Things to be careful with
- The `Sample` class redefines `histogram_data`/`signal_shape` as @properties *after* declaring them as relationships; the Python descriptor wins at attribute access. Don't "clean this up" without verifying every read site.
- `Attachment.kind` is a Python `@property` returning `'bh'`/`'hf'` from `dataset_id IS NULL`. **NOT a DB column** — `.filter(Attachment.kind == 'hf')` matches zero rows. Filter on the underlying columns (`Attachment.hf_repo_id.isnot(None)` for HF, `Attachment.dataset_id.isnot(None)` for BH); use `att.kind` only when iterating rows in Python.
- `secret_key = 'supersecretkey'` is hardcoded; fine for local dev, assume no auth/CSRF protection.
- `evaluate_dynamic_metric` calls `exec()` on user-supplied Python — by design (sandboxed in prod, see `runner/CLAUDE.md`). Treat the in-process path as trusted-local-network only.
- `app.py` uses `@app.url_value_preprocessor` + `@app.url_defaults` + a monkey-patch of `werkzeug.routing.Map.is_endpoint_expecting` to inject `project_name` into every URL. New routes taking `<project_name>` get the value injected on `url_for(...)` without passing it.

## HF dataset attachment patterns
(Much of this names Phase-A-era HF/PWC code — verify against current `app.py` before relying on it.)
- **`_HF_SPLIT_PREFERENCE = ['test','validation','val','dev','train']`**. `_resolve_hf_split_and_load(att, load_fn)` walks the order, probes row 0 to verify mapped GT cols aren't all null, falls back to first loadable split, **persists via `_persist_resolved_split`** so the LB badge tells the truth.
- **`_infer_mapping(features)`**: string→text, Audio→audio, else→json. `Value:unknown` (HF's flattened nested types, e.g. DocRED `sents`/`vertexSet`/`labels`) → json, not skip. Changing it ⇒ check `_persist_hf_eval_snapshots` + `_virtual_sample_from_hf_row`.
- **`_pwc_task_to_category`** strips domain prefixes (Medical, Aerial, Few-Shot, …) before classifying ("Medical Image Segmentation" → "Vision/Image Segmentation"). New prefixes → `_DOMAIN_PREFIXES`, shortest-first.
- **`populate_lb_samples` has a 5-min `soft_time_limit`** — `suggest_hf_repo` can land on a monolithic 100GB+ HDF5 repo and hang the worker. Flask+Celery+Redis share one box; don't bulk-enqueue populate tasks — use the per-LB "Populate samples" button or rate-limit.

## Input vs GT roles on dataset columns
Each mapping entry carries optional `role`: `input` (conditioning given to submitter, NOT predicted) or `gt` (held server-side, prediction target). Default `gt` (back-compat). `_pwc_task_input_kinds(task_name)` returns the `target_kind`s to flag `input`. `_lb_submission_pred_fields` drops pred fields whose GT col is `input` (so `label_pred` won't appear in an Image-Generation contract). Owner-editable on `/edit_leaderboard/<id>` → Prediction-fields → "Dataset field roles"; frozen once verified subs exist. `LeaderboardMetric.arg_mappings` must reference a GT-role column or the metric has no valid pred field (`.tag_input_gt.py` one-shot rewrites these per task).

## Metric / Visualization input-kind declarations
`GlobalMetric.input_kinds` / `GlobalVisualization.input_kinds` are nullable JSON arrays of accepted `target_kind`s in arg order (NULL = unconstrained). `/metrics?selected=<id>` shows an "Accepts: <kind>×<kind>" row; backfilled for 18 metrics in `.backfill_input_kinds.py` (add patterns to `KIND_HINTS` there). The LB→metric binding UI doesn't yet *enforce* kinds.

## Editing the LB pred-field schema
- Owner/admin edits each LB's prediction-field schema on `/edit_leaderboard/<id>` → "Prediction fields" tab. Each row: name (`<x>_pred`), kind, description, remove. Add-row for extras.
- Frozen once **verified** submissions exist (mirrored PWC subs don't count) — changing kinds afterwards would re-interpret existing prediction files through the wrong decoder. Delete verified subs to unlock.
- Writes to `Leaderboard.required_pred_fields_json`; `_lb_submission_pred_fields` merges it as an authoritative override of metric-derived entries.
- `_create_lb_from_pwc_benchmark` picks the gt_field for arg_mappings via `_pwc_task_pred_kind_priority(task_name)` (task-aware: image-gen→image, segmentation→mask, depth→depth); falls back to `(scalar > depth > image > mask > text)`.

## Mask vs image disambiguation
Both upload paths route segmentation masks to `target_kind='mask'` (deterministic-hue palette + IoU-family defaults), not 'image':
- **HF** (`_infer_mapping`): an `Image` column whose name contains `mask`/`segmentation`/`segment_map`/`seg_map`/`annotation`/`panoptic`/`label_map`/`semseg` → mask (`_HF_MASK_TOKENS`, via `_col_name_looks_like_mask`).
- **BH ZIP** (`detect_custom_fields`): folder-name token short-circuits to mask; else `_classify_image_path` peeks first file (downsampled 256×256): PIL mode `P` → mask; `L`/`I` with ≤32 unique values → mask; `RGB`/`RGBA` with ≤32 unique colors → mask; else image.

## Field-type taxonomy (CustomField.field_type)

| field_type | Storage | Comparison cell | Notes |
|---|---|---|---|
| `scalar` | `value_float` | `gt_scalar_value`, smart_num-formatted (int → no `.0000`) | Togglable as a column. |
| `text` | `value_text` | `gt_text_value`, scrollable card | Default for any string column. |
| `metric` | `value_float` | Goes through `per_sample_metrics` chart panel | NOT togglable as a normal column. |
| `image`/`depth`/`mask` | marker row + bench_cache | `<img>` → `serve_custom_field_image` → `serve_gt_viz` | Bytes don't live on the volume. |
| `audio` | marker row + bench_cache (waveform PNG) | Same `<img>` path; route sniffs PNG magic | HF Audio decode needs `soundfile`. |
| `json` | `value_text` (serialised JSON) | JSON scroll box | Dicts / Seq-of-dict / bboxes / Translation. `get_metric_context` json-decodes. |
| `topk_list` | `value_text` (JSON array) | Falls back to text render | Ranked-list preds for Hits@N / MRR. Deserialised by `get_metric_context`. |
| `histogram` | `value_blob` | Sparkline / chart | Fixed-length int sequences. |

## "Explorable" status
- `_compute_explorable_lb_ids(lb_ids)` = LB IDs whose GT is cached: BH dataset Sample rows OR LB-scoped CustomField rows (sample_id+submission_id both NULL — HF-stub markers). Drives the green/yellow pill on `/leaderboards`,`/home`,`/landing` + the "Explore samples" label. (`/explore` → 302 to `/leaderboards` since `21b5222`; all links use `url_for('leaderboards', ...)`.)
- An LB with `canonical_for_repo IS NOT NULL` + zero GT CFs is **broken** — show the owner-only "Populate samples" button, not an empty Explore page.
- **`/datasets` has two sections**: regular `Dataset` rows + "Cached HuggingFace datasets" from distinct `Attachment.hf_repo_id` whose owning LB is explorable (each links to the first LB's Explore view — filter is intentional, else dead links).

## Metric authoring
- **LLM-authored metrics (`_llm_generate_metric_code`) aren't safe verbatim** for non-trivial cases (rank-based, span-overlap, BLEU) — they mix scalar-vs-list logic and return the wrong number. Rank LBs (Link Prediction) needed manual `_rank_of_gt(gt, pred_list)` rewrites; spot-check new ones.
- Metrics stored as source on `GlobalMetric.python_code`, exec'd in `evaluate_dynamic_metric` with `numpy as np`. Replace cleanly by overwriting `python_code` (no migration). The signature-→`input_kinds` authoring convention is in `benchhub/CLAUDE.md`.

## User-owned content visibility
- `GlobalMetric`/`GlobalVisualization` rows default `visibility='private'` for non-admins, `public` for admins (BENCHHUB_ADMIN_EMAILS / `is_admin`). Owners flip via the detail-pane select on `/metrics?selected=<id>` / `/visualizations?selected=<id>` (`set_global_metric_visibility`, `set_global_visualization_visibility`).
- **Name uniqueness is two-tier** (names no longer globally unique — two users can each have a private `my_iou`). Two SQLite indexes in `check_and_migrate_db`: `uq_<table>_name_public` (partial unique on `name` WHERE `visibility='public'`) + `uq_<table>_name_per_owner` (composite on `(owner_user_id, name)`).
- **Promote-to-public collision UX**: the visibility route detects a public-name collision pre-flip and redirects to `resolve_name_collision.html`, which proposes `<name>_<N>` via `_suggest_unique_public_name()`; the `/visibility/confirm` second hop re-checks before committing.
- `Leaderboard.canonicality` **dropped** — column stays in DB unread; visibility (public/private/unlisted) is the only catalog-membership flag. `admin_promote_leaderboard` is a back-compat alias that just flips `visibility` (owner-OR-admin now). `Leaderboard.canonical_for_repo` is informational ("tracks X HF repo", multiple LBs may share); admin-only `admin_set_canonical_for_repo` edits it without touching visibility.

## FeatureRequest
- `FeatureRequest` table backs `/feature_requests` (user form + own submissions) and `/admin/feature_requests` (admin triage with status + note). Statuses: `open` (default), `planned`, `in_progress`, `resolved`, `declined`. Admin can attach an `admin_note` visible to the requester.

## OAuth
- GitHub + Google via Authlib (`oauth.github`, `oauth.google`), configured with `GITHUB_CLIENT_ID/SECRET` + `GOOGLE_CLIENT_ID/SECRET` (env vars). Google's Cloud Console redirect URI must be `<site>/oauth/callback/google`.
- Apple sign-in NOT wired (needs signing key + team/key ID from the Apple portal). To add: register `oauth.apple` with a JWT-generated `client_secret` + `/login/apple` + `/oauth/callback/apple` mirroring Google.

## Depth visualization
- Depth-kind GT thumbs cache as **8-bit grayscale PNG** (normalized 0..255 of the source range). Don't burn a colormap at cache time.
- `/api/gt_viz/<lb_id>/<col>/<sample_name>?cmap=<name>` recolors the gray PNG at view time. Names: `turbo`,`jet`,`viridis`,`magma`,`inferno`,`plasma`,`gray`,`normal` (unknown → turbo). `normal` is a Sobel-based tangent-space surface-normal map — qualitative, not metric. (The comparison-view colormap `<select>` UI is in `templates/CLAUDE.md`.)

## Keeping docs current
- **Update docs in the same change, not later — stale docs are a bug in the change.** When a change alters a user-facing flow (import options, LB creation, submission/scoring, settings, quotas), update the in-app docs page under `templates/docs/*.html`. When it changes a subsystem's internals or a gotcha, update that subtree's `CLAUDE.md`. When it changes ops/deploy, update `docs/SELFHOST_RUNBOOK.md`.

## Keeping this file lean
- **This file loads into context every session.** It hit the 40k-char limit once; keep it to the always-on app.py/operational core. Subsystem-deep detail belongs in the subdirectory `CLAUDE.md` files indexed at the top (they load only when you touch that subtree).
- **Where a new note goes:**
  - Durable cross-cutting gotcha (bites anywhere in `app.py`/migrations/ops) → the right section *here*, and **update the existing bullet instead of appending a duplicate**.
  - Subsystem-specific detail → the matching `benchhub/` · `runner/` · `tests/` · `templates/` `CLAUDE.md`.
  - Why one specific change was made → the commit message (git history), not here.
  - Personal working-style preferences → auto-memory (`~/.claude/.../memory/`).
- **Prune archaeology.** Once a migration has run everywhere and the old path is gone, delete its note rather than compressing it forever. **`@import` does NOT save context** (imported files load eagerly) — don't reach for it to shrink this file; move content to a subdirectory file instead.
