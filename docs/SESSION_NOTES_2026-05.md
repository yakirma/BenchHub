# BenchHub session recap — 2026-05-07/08

A take-home brief for picking up the project in another session. Covers
what we built, the architectural insights we surfaced, the current
shape of the codebase, and the open follow-ups.

---

## 1. What landed in this session

Roughly chronological. Each entry is a self-contained commit on `main`.

### Colab submission rails

- **Gist URL format fix.** Colab's importer wants
  `colab.research.google.com/gist/<owner>/<gist_id>`; bare-id form
  fails with "Unexpected GitHub Gist path". `_ensure_colab_gist` now
  captures `owner.login` from the GitHub API response and persists
  it on the cache wrapper so PATCH-on-revisit reuses the same path.
- **Per-user API-token autofill.** New `UserColabGist(user_id,
  leaderboard_id, gist_id, gist_owner, sig)` table. The LB-level
  `Leaderboard.colab_notebook_cache` stores the *generic* notebook;
  per-user gists carry the personalized copy with `API_TOKEN`
  substituted in. `_personalize_notebook_for_user` does the
  substitution before push to the gist.
- **Submission ↔ Colab back-link.** New `Submission.source_colab_url`
  column; the API upload endpoint accepts a `source_colab_url` form
  field (sent by the colab notebook), with auto-fallback to the
  user's `UserColabGist` for the LB. LB page renders a rocket-icon
  link per row.
- **Direct downloads on LB header.** Split-button dropdown next to
  "Submit via Colab" exposes `colab_notebook.ipynb` and a new
  `colab_bootstrap.py` endpoint that serves a self-contained Python
  bootstrap (no Jupyter required to start a local run).
- **Notebook v6 template.** Three big shifts:
  - Runtime choice up to the user (no `metadata.accelerator` pin).
  - "Run locally" bootstrap cell at the top (no-op on Colab,
    copy-pasteable to `bootstrap.py`).
  - Predictions go to **bare-name folders** (`<col>_pred/<sample>.<ext>`),
    NOT under `metric_*`. Per-kind extension dispatch in the
    prediction loop: `.txt` for scalars, `.png` for images/masks,
    `_<W>x<H>.npz` for depth.
  - Cached notebooks self-invalidate via a template version stamp
    folded into the LB structure signature.

### Auto-LB pipeline (the metric/viz proposer)

The big architectural shift in this session was a meaningful
proposer rewrite.

- **Originally**: walked `metric_*` GT fields, attached a single
  `mae_<col>` or `accuracy_<col>` metric. Three problems:
  1. Misused the `metric_*` namespace (which is for user-precomputed
     metric *values*, not labels).
  2. Couldn't handle anything but scalar GT.
  3. Metric names like `mae_score` polluted the GlobalMetric library.

- **Now (Option B)**: walks ALL GT columns and dispatches per kind.
  | GT kind | Metrics proposed | Viz |
  | ------- | ---------------- | --- |
  | scalar (ClassLabel-shaped, sibling `<col>_class`) | `top1` | `confusion_matrix` (aggregated) |
  | scalar (numeric) | `mae` | — |
  | depth | `rmse` + `abs_rel` + `a1` | `depth_error_heatmap` (aggregated) |
  | image (RGB) | `psnr` | — |
  | image (mask-named: `mask`/`seg`/`label`/`parsing`/`class`) | `miou` | — |

- **Engine plumbing.** `get_metric_context` in `metric_engine.py`
  loads `image` + `depth` GT custom fields lazily as numpy arrays
  (None on failure). Submission-folder scanner picks up bare-name
  `<col>_pred/` folders and reads `.txt` / `.png` / `.npz` per the
  same kind dispatch.
- **Strict-name reuse.** GlobalMetric / GlobalVisualization names
  are kind-only (`mae`, `rmse`, `top1`, `psnr`, `miou`, `abs_rel`,
  `a1`, `confusion_matrix`, `depth_error_heatmap`). One row per
  kind shared across LBs and columns. `target_name` carries the
  per-LB display detail (`MAE (score)`, `RMSE (depth_map)`).
- **Preview-and-edit flow.** `/create_leaderboard` with
  `auto_assign_metrics=1` renders `auto_lb_preview.html` instead of
  creating immediately. Per-row keep checkbox, editable display
  name, sort-direction dropdown, editable Python code (badged
  "reuse existing" / "LLM-authored" / "static fallback").
  `/create_leaderboard/auto_finalize` persists kept rows.
- **Submission contract on LB page.** New card listing each
  `<col>_pred/` folder, its kind, its GT field, a description, and
  the metrics + viz that consume it. Same data appears as a chip
  row on the auto-LB preview before commit.

### Tag pipeline simplification

Old auto-tagger emitted up to 6 vague tags per dataset (union of HF
metadata + LLM suggestions). Tightened to:

- **Primary tag** from a fixed vocabulary: `depth`, `segmentation`,
  `classification`, `detection`, `language`, `audio`, `generation`,
  `regression`, `tabular`, `multimodal`, `tracking`, `pose`,
  `reconstruction`.
- **Optional qualifier** (≤ 1) — kebab-case modifier like `stereo`,
  `medical`, `indoor`, `multi-label`. Total cap: 2 tags per dataset.
- LLM-first when an API key is set; deterministic fallback maps
  known HF task tags onto the vocabulary so installs without an
  API key still get a sensible primary.

### HF auto-import resilience

- **Streaming-fallback** for `_hf_fetch_features`. Many community
  repos return 200 from `/api/datasets/<repo>` but have no
  `cardData.dataset_info`. Falls back to opening the dataset via
  `datasets.load_dataset` in streaming mode and reading
  `ds.features` directly. Last-resort: peek a single row and infer
  types from Python values (`PIL.Image` → image, `int` → Value:int64).
- **`datasets<3.0` pin.** v3.x removed support for script-backed
  datasets (NYU Depth V2 etc. ship a `<repo>.py` loader). Pinned to
  v2.x and pass `trust_remote_code=True` to every `load_dataset`
  call.
- **Numeric-class names handled.** When ClassLabel.names is just
  stringified indices (`['0','1','2',...]`), skip the redundant
  `<col>_class` side field. Tag still emitted.
- **Tags-folder text pin.** `detect_custom_fields` forces the
  reserved `tags/` folder to text type so a numeric class name like
  `"5"` doesn't get misfiled as a scalar.

### Identity & UI polish

- **OAuth thumbnails.** Dataset / LB / comparison / datasets-list
  pages now drive avatars from `User.display_name` + `User.avatar_url`
  instead of the legacy `git_author` string + `AuthorProfile`
  lookup. New `_user_avatar.html` Jinja macro server-renders the
  avatar; the old `author_avatars.js` bootstraps + global
  `AUTHOR_PROFILES` JSON injection are gone.
- **LB-card thumbnails on `/explore`.** Match `/home`'s treatment.
- **LB-page filters collapsed by default.** Auto-opens when any
  filter is active, "Clear filters" shortcut.
- **Text custom-field columns visible.** Dataset-view's
  available-display-options injector skipped `text` fields entirely
  — fixed so AG News `text`, NLI `premise`/`hypothesis`, captions,
  etc. surface as proper columns.

### Seeding mission

Production target: 10 datasets per domain (depth / segmentation /
language / denoising) → 40 LBs.

- **`scripts/seed_datasets.py`.** Batch HF auto-import + auto-LB.
  Per-entry failure isolated. `--skip-existing` for re-runnability.
- **`seed_data/{depth,segmentation,llm,denoising}.json`.** Curated
  starter configs, 10 verified-live HF repos each.
- **`scripts/prune_seed_configs.py`.** One-shot pruner that drops
  entries `_hf_fetch_features` can't read, with `.bak` rollback.
- **`scripts/wipe_seeded.py`.** Nuclear-option clean slate before a
  fresh seed run. Two-pass: dry-run prints counts; `--yes` deletes.
  Wipes Datasets / Samples / Leaderboards / Submissions /
  CustomFields / GlobalMetrics / GlobalVisualizations / Tags /
  UserColabGists + on-disk `uploads/datasets` + `uploads/submissions`.
  Users / OAuth / API tokens / settings preserved.
- **`seed_data/seed_baselines.py` + 4 domain stubs.** Generic
  upload-mechanic boilerplate that walks a model spec list, runs
  predictions, packages bare-name `<field>/<sample>.<ext>` ZIPs,
  POSTs to `/api/leaderboard/<id>/submission/upload` with
  `source_colab_url`. Domain stubs (`baselines_depth.py`,
  `baselines_segmentation.py`, `baselines_language.py`,
  `baselines_denoising.py`) ship 10 HF model IDs each + naive
  `predictor_fn` stubs.

### Volume capacity (Fly)

- Old: 1 GB attached volume in `iad`. Filled before depth seed
  finished.
- Now: extended to 20 GB. ~$3/mo extra.

### Tests

640+ passing. Coverage spans:
- HF features fallback paths (streaming, row-peek, datasets-lib-missing).
- Engine context array-loaders for image / depth / scalar / depth+pred.
- Each per-kind metric proposal (top1, mae, rmse, abs_rel, a1, psnr, miou) — including exec'ing the generated `fallback_code` and asserting sane numeric outputs.
- Submission ↔ Colab back-link (form-supplied URL, UserColabGist fallback, NULL when neither).
- Auto-LB preview/finalize flow (kept-only, edits land, no-LB when nothing kept).
- OAuth-driven thumbnails on dataset/LB/comparison surfaces.
- Tag vocabulary lockdown.
- Text custom-field column visibility.

---

## 2. Insights (the load-bearing ones)

### `metric_*` is for user-precomputed metric values, not labels.

The original convention: a submission ships `metric_<name>/<sample>.txt`
with a per-sample metric value the LB just averages and displays
(BenchHub doesn't run any code). This pattern still works.

What it's NOT: a place to store GT class labels. Earlier auto-import
mistakenly wrote ClassLabel indices to `metric_label/`, which both
polluted the namespace and meant submissions couldn't be evaluated
against them. The fix: ClassLabel + numeric Value HF columns now go
to bare-name `<col>/` folders (field_type='scalar'). Submissions
ship `<col>_pred/` (predicted value), and the LB's auto-attached
metric (`top1` / `mae` / etc.) compares the two.

This is the cleanest split:
- `<col>/` (GT) ↔ `<col>_pred/` (submission prediction) → comparison metric.
- `metric_<name>/` (submission) → user-precomputed metric value, just averaged.

### HF datasets are messy.

The ecosystem looks tidy from the website but fights you under
automation:

- **~30% of repos** lack `cardData.dataset_info` in the API. Streaming
  fallback works for most of them; the rest need manual schema.
- **Some repos** (NYU Depth V2 et al.) ship loader scripts that
  `datasets` 3.x dropped support for. Pinning to v2.x + `trust_remote_code=True` brings them back.
- **Pair-split datasets** (Kaggle's Denoising Dirty Documents:
  `-train` is the input, `-cleaned` is the GT, two separate HF
  repos) can't auto-build an LB without cross-repo joining.
- **Repo IDs rotate.** Even repos confirmed live last week 404 today.
  Always dry-run before a real seed.

### Cloning HF datasets is the wrong long-term shape.

Currently the importer copies (capped) sample bytes to
`uploads/datasets/<name>/`. For HF auto-imports specifically, this
duplicates data that's free + canonical on HF, costs Fly disk rent,
and goes stale.

**Pointer mode** (the next major refactor — see "Future directions")
would store row references instead of bytes. Lazy-fetch GT from HF
at metric-eval time. Same disk for ~50× more data, deterministic via
revision pinning.

### Auto-proposer needs structured-GT awareness.

A proposer that only walks scalar GT works for classification but
silently produces empty LBs for ~70% of vision benchmarks (depth maps,
segmentation masks, denoising image pairs). The Option B refactor
fixes this by dispatching per GT kind, but the engine context loader
had to be extended to load arrays from disk before the metric code
could see them.

### LLM-driven mappers benefit from explicit forbids.

Earliest LLM mapping prompts let Claude split one source column into
multiple BenchHub fields ("label" → metric_label + label_class +
tag). Adding **"NEVER split one source column. Output EXACTLY ONE
entry per source column."** plus a defensive dedupe in the cleaning
step fixed the CIFAR-shape bloat. Same shape for the auto-tag
prompt: locked to a fixed vocabulary, capped at 2 tags, off-vocab
primary → reject.

### Sandboxed metric eval is real but not on by default.

`metric_engine.py` has both an in-process `evaluate_dynamic_metric`
(uses `exec()`) and a docker-sandboxed `evaluate_in_sandbox` path
gated by `BENCHHUB_SANDBOX_METRICS=1`. Option B's array-in-context
pattern works for the in-process path (fast); sandboxed mode would
need bytes-over-JSON which we haven't built. Acceptable for now
(public site is single-tenant), but a real concern before opening
to untrusted submissions.

---

## 3. Project structure (load-bearing files)

### Top-level

```
app.py                  ~10k lines. Flask app, all routes, models, importers, the auto-LB pipeline, the colab notebook generator. The thing.
metric_engine.py        Metric + visualization eval. get_metric_context (engine-side data loader), evaluate_dynamic_metric (in-process exec), evaluate_in_sandbox (docker-shell-out path).
tasks.py                Celery tasks. process_submission, batch recalculation. Imports from app — circular shape on purpose, see CLAUDE.md.
requirements.txt        Pinned: `datasets>=2.20,<3.0`. Don't bump past 2.x without re-doing the script-loader story.
fly.toml                2 GB VM, 20 GB volume.
```

### Templates (`templates/`)

- `base.html` — chrome.
- `home.html` — owned datasets + LBs + thumbnails.
- `explore.html` — public LB grid (now with thumbnails).
- `dataset_view.html` — per-dataset page with the column-visibility model + sample table.
- `leaderboard.html` — per-LB page. Submission contract card, per-row Colab back-link, collapsed filter widgets, split-button submit dropdown.
- `auto_lb_preview.html` — preview-and-edit before auto-LB commit.
- `_user_avatar.html` — server-rendered OAuth avatar macro (no JS).
- `hf_import_preview.html` — HF auto-import field-mapping preview.

### Auxiliary helpers

- `scripts/seed_datasets.py` — batch HF auto-import.
- `scripts/wipe_seeded.py` — nuclear clean slate.
- `scripts/prune_seed_configs.py` — drop unloadable entries from a config.
- `seed_data/*.json` — curated 10-per-domain HF repo lists.
- `seed_data/seed_baselines.py` — generic baseline-runner boilerplate.
- `seed_data/baselines_<domain>.py` — domain stubs (model lists +
  `predictor_fn` + `load_inputs`).

### Database (SQLite, Fly-volume-backed)

The schema you'll touch most:

```
User                — OAuth identity (display_name, avatar_url, api_token, hf_token, is_admin)
Dataset             — global namespace; owner_user_id, visibility, source_kind ('hf-auto'|...)
Sample              — per-dataset
CustomField         — flexible per-sample data: image|scalar|metric|histogram|depth|json|text
Leaderboard         — owns metrics + visualizations + submissions
Submission          — owner_user_id, source_colab_url, processing_status
LeaderboardMetric   — link table: arg_mappings (JSON), target_name, sort_direction, pooling_type
GlobalMetric        — name (now kind-only after the suffix-drop), python_code
LeaderboardVisualization, GlobalVisualization — analogues for viz
UserColabGist       — per-user gist mapping (user_id, lb_id) → (gist_id, gist_owner, sig)
Tag                 — many-to-many with Dataset / Leaderboard / Sample
```

`check_and_migrate_db()` runs on every boot, so adding a new column
means: model definition + a migration entry in the
`_ownership_migrations` list. SQLite-only DDL, no Alembic.

### CLAUDE.md is the orientation doc.

Don't skip it before starting work. It documents the circular
import shape between `app.py` and `tasks.py`, the upload folder
layout, the `metric_*` convention pre-rewrite, and other things
that aren't obvious from the code.

---

## 4. Future directions (in order of expected leverage)

### A. Pointer-mode HF dataset storage (the big one)

**Problem.** Currently HF auto-imports clone bytes to disk. For
40 datasets × 200 samples × ~1 MB/sample = ~8 GB. That's tractable
but you're capped to "fits-on-Fly" sizes; real benchmarks (ImageNet,
LAION, full COCO) are 100 GB+.

**Design.**

- New `Dataset.storage_mode` enum: `'local' | 'hf-pointer'`. Existing
  rows stay `'local'`.
- New `Sample.source_ref_json`: `{repo_id, revision, split, row_idx}`.
- `_import_hf_auto` for pointer mode: harvest schema + ClassLabel
  names + tags only (cheap metadata-only stream), create Sample rows
  with row indices, write zero image bytes.
- `metric_engine.get_metric_context`: when Sample is pointer-backed,
  lazy-fetch the row via
  `datasets.load_dataset(...).skip(idx).take(1)` and feed
  `_load_gt_array`. Add a small Redis cache keyed by
  `(repo_id, revision, idx)` so re-running a metric across a
  recalculation doesn't hit HF for every row.
- `download_dataset` endpoint: for hf-pointer, generate the ZIP
  on-the-fly OR redirect to HF.
- Visualization paths (per-sample image rendering): same lazy-fetch
  + cache.
- Colab bootstrap: pull GT directly from HF (`datasets.load_dataset(repo_id)`) for hf-pointer datasets — BenchHub never had the bytes
  to ship.

**Estimated effort.** 2–3 days. Mostly mechanical once the
abstraction settles. ~95% storage reduction, unbounded benchmark
size, GT pinned to a revision = deterministic eval.

### B. Pair-split dataset support

Make `Dataset.linked_gt_dataset_id` (nullable FK) so one BenchHub
Dataset can reference inputs from repo A and GT from repo B (the
Kaggle `denoising-dirty-documents` shape). Maybe ~2 hours of work;
worth it if you hit 3+ pair-split datasets you actually want to
host.

### C. Replace placeholder `predictor_fn` stubs

The `seed_data/baselines_<domain>.py` stubs return naive scalars
(mean depth, dominant class, mean intensity) just to verify the
upload pipe. For real benchmarks:

- `baselines_depth.py`: predict full depth map; ship as
  `<col>_pred/<sample>_<W>x<H>.npz`. The auto-LB's
  RMSE/abs-rel/a1 metrics already expect this shape.
- `baselines_segmentation.py`: predict per-pixel mask; ship as PNG.
- `baselines_denoising.py`: predict full denoised image; ship as
  PNG; the `psnr` metric already handles uint8 / float-normalized.
- `baselines_language.py`: text-classification flavor is fine as-is;
  add summarization/QA variants when the LBs need them.

### D. Sandboxed eval for the public deployment

Right now `evaluate_dynamic_metric` does `exec()` on user-uploaded
metric Python. Acceptable for a single-tenant deployment, dangerous
for an open one. The docker-sandboxed path
(`evaluate_in_sandbox`, gated by `BENCHHUB_SANDBOX_METRICS=1`)
exists but doesn't yet support the array-in-context pattern Option
B introduced. Before opening to public submissions: bytes-over-JSON
serialization for image/depth arrays + Redis cache for the GT
fetch.

### E. Submission-side SOURCE_COLAB_URL self-discovery

Today the gist URL is baked in via `_personalize_notebook_for_user`.
The cleaner version: the colab notebook detects its own URL via
`google.colab.runtime.get_notebook_url()` (or similar) at upload
time, so even hand-edited copies of the notebook back-link
correctly. Minor polish.

### F. LLM-proposed pred-field naming

The proposer hardcodes `<col>_pred`. For some columns the natural
name differs (e.g. an image-restoration LB might prefer `clean_pred`
even when the GT column is `target`). Could let the LLM mapping
output suggest the pred-field name during the auto-LB preview.

### G. Multi-modal proposers

- Audio (`Audio` field type): WER for speech-to-text, MOS-style
  comparison for audio-to-audio.
- Video (sequences of frames): per-frame metrics with temporal
  pooling.

Lower priority — not in the seeding mission's domain.

### H. The `Dataset.storage_bytes` cache is approximate

`process_dataset_zip` writes `Dataset.storage_bytes = _path_size_bytes(dataset_dir)` once. If the underlying directory changes (a CustomField swap, a tag re-import), the cache goes stale. For pointer mode (A) this becomes irrelevant; otherwise consider invalidating on
mutating writes.

---

## 5. Open follow-ups handed off

- **Pointer-mode refactor (A above).** I committed a memory note to
  surface this when seeding finishes; if you start a fresh session
  and say "seeding's done" I'll bring it up automatically.
- **Stale repo IDs in seed configs.** The 2026-05-08 verification
  pass landed live entries, but expect periodic decay. Re-running
  `scripts/prune_seed_configs.py` before a re-seed catches them.
- **Per-domain real metrics.** The auto-LB attaches `mae` for
  numeric scalars, `rmse`/`abs_rel`/`a1` for depth, etc. — they're
  *baselines*, not always the canonical metric for the dataset. Edit
  via the LB edit page when seeding has settled.

---

## 6. How to pick this back up

1. Start with `CLAUDE.md` for the architecture orientation.
2. Read this doc.
3. Skim `git log --oneline -50` to see the most recent surface area.
4. The auto-LB pipeline is the area with the most leverage right now;
   the storage-mode refactor (Option A) is the highest-impact next
   step.
5. Tests live in `tests/test_*.py`. ~640 passing. Run
   `python -m pytest -q` from repo root before any non-trivial
   change.
6. Deploy is `flyctl deploy --remote-only` from repo root. The fly
   machine ID is in `flyctl machine list -a benchhub`.
