# BenchHub session handoff — 2026-05-08

A take-home brief for picking up the project in another Claude
session. Covers the current active workstream, what shipped earlier
this session, the load-bearing insights, the project structure, and
the queued follow-ups.

---

## 1. Where we are right now (active workstream)

**Pivot:** the 40-dataset seeding mission is **paused**, intentionally.
Cloning capped HF subsets onto Fly was always the wrong long-term
shape — see "Insights" below. Instead we're refactoring to
**live-streaming GT from HF + decentralized submission storage**,
backed by a single bounded-LRU disk cache.

### Active refactor: pieces in flight

| Piece | Status | Where |
| ----- | ------ | ----- |
| **Disk-bounded LRU cache** (`bench_cache.py`) | ✅ landed | `bench_cache.py`, `CacheEntry` model, 15 tests |
| **Pointer-mode for HF datasets** (no byte cloning) | ✅ landed | `Dataset.storage_mode`, `Sample.source_ref_json`, `_import_hf_pointer`, `_pointer_gt_resolver`, 10 tests |
| **Remote submissions** (HF Hub + raw URL, hash-pinned) | ✅ landed | `Submission.storage_mode/remote_url/content_hash`, `_fetch_remote_submission_zip`, `/api/leaderboard/<id>/submission/from_url`, 9 tests |
| **Paired-dataset support via LB settings** | ✅ landed | `leaderboard_datasets.role`, `_make_paired_gt_provider`, role dropdown on LB edit page, 12 tests |
| **Hash-mismatch enforcement on re-eval** | 🟡 next | recalc path verifies `Submission.content_hash` against re-fetched bytes |
| **Evict extracted submission folder after eval** | 🟡 after | recalc re-extracts from cached ZIP; closes the disk-savings loop |

Architecture sketch:

```
                                                     ┌──────────────────────┐
GT bytes ──── streamed from HF on demand ───────────►│                      │
                                                     │   bench_cache        │
                                                     │  (LRU, bytes-bounded,│
Submission bytes ── streamed from user's HF/URL ────►│   per-key file lock) │
                                                     └──────┬───────────────┘
                                                            │ load array
                                                            ▼
                                                    metric_engine eval
                                                            │
                                                            ▼
                                                  MetricResult (kept forever)
```

**Eviction policy:** submissions evict before GT (cheap to re-fetch
from the user's external store; HF GT is rate-limited). Both LRU
within their tier.

**Reproducibility (decided):** strict hash-pinning. Submissions
capture SHA-256 at first eval; mismatch on re-fetch surfaces a
"submission modified — please resubmit" warning rather than silently
re-ranking.

### What I just landed (this turn)

- `bench_cache.py` — module: `cache_get`, `cache_put`, `cache_gc`,
  `cache_clear`, `cache_stats`, `resolve_budget_bytes`. Per-key file
  locks via `fcntl.flock`. Atomic writes via tmp+rename+fsync.
- `CacheEntry` SQLAlchemy model (`cache_key`, `size_bytes`, `origin`,
  `last_accessed_at`, `created_at`) + migration block.
- 15 tests in `tests/test_bench_cache.py`: round-trip, eviction
  priority (submissions before GT), LRU within tier, atomic-write
  rollback on writer exception, dedup on existing key, stats.
- Total suite: 658 passing.

### Concrete next steps (pick up here)

The original three-piece refactor + paired-datasets are all in.
Two follow-ups remain to close the storage-savings loop:

1. **Hash-mismatch enforcement on re-eval (remote submissions).**
   - On recalc path: re-fetch via `_fetch_remote_submission_zip`
     (which is cache-aware) → compare returned hash against
     `Submission.content_hash` → on mismatch, set
     `processing_status='Error: submission file changed; please
     resubmit'` and bail.
   - Surface a clear UI badge on the LB row when this happens.
   - Estimated effort: ~1-2 hours.

2. **Evict extracted submission folder after eval (close the
   disk-savings loop for remote subs).**
   - Currently a remote submission's bytes live in two places: the
     cached ZIP AND the extracted `uploads/submissions/<id>/`
     folder. Real disk savings need eviction of the extracted form.
   - Add a context manager `_with_extracted_submission(submission)`
     that, for remote submissions, extracts the cached ZIP into a
     transient folder, yields the path, and removes the folder on
     exit. Local submissions yield `uploads/submissions/<id>/`
     unchanged.
   - Move tasks.py + the recalc paths to use the context manager.
   - Estimated effort: ~3-4 hours.

After 1-3 land, the storage refactor is feature-complete: GT bytes,
submission bytes, and per-row caching all live in `bench_cache`,
which is bounded + LRU + origin-prioritized. Disk usage scales with
recently-accessed-bytes, not total-imported-bytes.

---

## 2. What shipped earlier in this session (already on `main`)

Roughly chronological, each a self-contained commit.

### Colab submission rails

- **Gist URL format fix.** Colab's importer wants
  `colab.research.google.com/gist/<owner>/<gist_id>`; bare-id fails.
- **Per-user API-token autofill.** New `UserColabGist` table; the
  LB-level cache holds the generic notebook, per-user gists carry
  the personalized copy with `API_TOKEN` substituted.
- **Submission ↔ Colab back-link.** New `Submission.source_colab_url`;
  rocket-icon link per submission row on the LB page.
- **Direct downloads on LB header.** Split-button dropdown surfaces
  `colab_notebook.ipynb` and a new `colab_bootstrap.py` standalone
  endpoint.
- **Notebook v6 template.** No metadata.accelerator pin (user picks
  GPU/CPU). Top-of-notebook bootstrap cell. Predictions go to
  bare-name folders (`<col>_pred/`), per-kind extension dispatch
  (`.txt` / `.png` / `.npz`).

### Auto-LB pipeline (Option B refactor)

The big architectural shift mid-session.

- **Engine plumbing.** `metric_engine.get_metric_context` loads
  `image` + `depth` GT custom fields lazily as numpy arrays.
  Submission-folder scanner picks up bare-name `<col>_pred/`
  folders for image / depth / scalar predictions.
- **Per-kind metric proposers** dispatched by GT type:

  | GT kind | Metrics | Viz |
  | ------- | ------- | --- |
  | scalar (ClassLabel) | `top1` | `confusion_matrix` |
  | scalar (numeric) | `mae` | — |
  | depth | `rmse` + `abs_rel` + `a1` | `depth_error_heatmap` |
  | image (RGB) | `psnr` | — |
  | image (mask-named) | `miou` | — |

- **Strict-name reuse.** GlobalMetric / GlobalVisualization names
  are kind-only (`mae`, not `mae_score`). One row per kind shared
  across all LBs. `target_name` carries per-LB display detail.
- **Preview-and-edit flow.** `/create_leaderboard` with auto-assign
  renders `auto_lb_preview.html` (per-row keep checkbox, editable
  display name, sort-direction dropdown, editable Python code).
  `/create_leaderboard/auto_finalize` persists kept rows.
- **Submission contract on LB page.** New card lists each
  `<col>_pred/` folder, kind, GT field, description, and the
  metrics/viz that consume it.

### Tag pipeline simplification

Old: union of HF tags + LLM, up to 6 vague tags. New:
- **Primary** from a fixed vocabulary (`depth`, `segmentation`,
  `classification`, `detection`, `language`, `audio`, `generation`,
  `regression`, `tabular`, `multimodal`, `tracking`, `pose`,
  `reconstruction`).
- **Optional qualifier** (≤ 1) — kebab-case modifier.
- Total cap: 2 tags.

### HF auto-import resilience

- **Streaming-fallback for `_hf_fetch_features`.** Many community
  repos return 200 but no `cardData.dataset_info`. Falls back to
  `datasets.load_dataset(streaming=True)` and reads `ds.features`
  directly. Last resort: peek a row, infer from Python types.
- **`datasets<3.0` pin** + `trust_remote_code=True`. v3.x removed
  script-backed loaders (NYU Depth V2 etc.).
- **Numeric-class names handled.** When ClassLabel.names is just
  `['0','1','2',...]`, skip the redundant `<col>_class` side field.
- **Tags-folder text pin** so a `"5"` class name doesn't get
  misfiled as a scalar.

### Identity & UI polish

- **OAuth thumbnails on dataset/LB/comparison/datasets-list pages.**
  Server-rendered `_user_avatar.html` macro replaces the legacy
  `git_author` + `AuthorProfile` JS pathway.
- **LB-card thumbnails on `/explore`.**
- **LB-page filters collapsed by default.** Auto-opens when active.
- **Text custom-field columns visible** on dataset view (was being
  silently dropped — AG News `text`, NLI `premise`/`hypothesis`).

### Seeding mission scaffolding (paused, not deleted)

- `scripts/seed_datasets.py` — batch HF auto-import + auto-LB.
- `seed_data/{depth,segmentation,llm,denoising}.json` — 10 verified-
  live HF repos per domain.
- `scripts/prune_seed_configs.py` — drops entries `_hf_fetch_features`
  can't read.
- `scripts/wipe_seeded.py` — clean-slate before a fresh seed run.
- `seed_data/seed_baselines.py` + 4 domain stubs — generic baseline
  runner the user runs on Colab.

These stay around but are dormant until the pointer-mode refactor
lands. Once it lands, "seeding" becomes ~free (metadata only) and
this scaffolding can be re-pointed at a much larger dataset list.

### Volume capacity (Fly)

- Extended from 1 GB to 20 GB during the abandoned seeding attempt.
  Now overkill for pointer-mode but harmless.

---

## 3. Insights (the load-bearing ones)

### `metric_*` is for user-precomputed metric values, not labels.

The original convention: a submission ships `metric_<name>/<sample>.txt`
with a per-sample metric value the LB just averages. What it's NOT:
GT class labels. The auto-import used to write ClassLabel indices to
`metric_label/`, polluting the namespace and meaning submissions
couldn't be evaluated. Cleanest split:
- `<col>/` (GT) ↔ `<col>_pred/` (submission prediction) → comparison metric.
- `metric_<name>/` (submission) → user-precomputed metric value, just averaged.

### HF datasets are messy under automation.

- ~30% of repos lack `cardData.dataset_info` in the API. Streaming
  fallback works for most.
- Some repos ship loader scripts that `datasets` 3.x dropped. Pinned
  to 2.x + `trust_remote_code=True` brings them back.
- Pair-split datasets (Kaggle's denoising-dirty-documents:
  `-train` is input, `-cleaned` is GT) don't work as a single LB
  without cross-repo support. → Solving this via the `role` column
  on `leaderboard_datasets`, see "Active refactor" §3.
- Repo IDs rotate. Always dry-run before a real seed.

### Cloning HF datasets is the wrong long-term shape.

(The crystallization that triggered this session's pivot.)

The current importer copies (capped) sample bytes to
`uploads/datasets/<name>/`. For HF auto-imports, this duplicates
data that's free + canonical on HF, costs Fly disk rent, and goes
stale. Real benchmarks (ImageNet, LAION, full COCO) are 100 GB+ —
we'd be capped to "fits-on-Fly" sizes forever.

**Pointer mode** stores row references, fetches on-demand, caches
LRU. ~95% storage reduction; unbounded benchmark size; deterministic
via revision pinning.

### Submissions don't need to live on Fly either.

Same logic as GT: BenchHub is a metric-evaluation service, not a
file host. Submissions can live on the user's own HF Hub repo (or
any URL), authed via the user's saved token, fetched on-demand,
cached, evicted. Per-user storage scales with each user (instead of
centralizing on Fly), and BenchHub never owns user model output
bytes long-term.

The user's "give access for this specific files" instinct points
at general OAuth-per-file flows (Drive, Dropbox, etc.) — defer
those. **HF Hub + direct URL** covers 95% of users with two endpoints
and zero per-provider maintenance.

### Auto-proposer needs structured-GT awareness.

A scalar-only proposer silently produces empty LBs for ~70% of
vision benchmarks. Option B's per-kind dispatch (depth/image/mask)
is the fix. The engine context loader had to be extended to load
arrays from disk before the metric code could see them — that
extension is now also the foundation for the cache integration.

### LLM-driven mappers benefit from explicit forbids.

Earlier prompts let Claude split one source column into multiple
BenchHub fields. Adding "NEVER split one source column. Output
EXACTLY ONE entry per source column." plus a defensive dedupe in
the cleaning step fixed it. Same shape for the auto-tag prompt:
locked vocabulary, hard cap, off-vocab → reject.

### Sandboxed metric eval is real but not on by default.

`metric_engine.evaluate_dynamic_metric` does `exec()` on user code.
Acceptable single-tenant; dangerous public. The docker-sandboxed
path (`evaluate_in_sandbox`, gated by `BENCHHUB_SANDBOX_METRICS=1`)
exists but doesn't yet support the Option B array-in-context
pattern. Before opening to public submissions: bytes-over-JSON
serialization for image/depth + Redis cache for the GT fetch.

---

## 4. Project structure

### Top-level

```
app.py                  ~10k lines. Flask app, all routes, models, importers, the auto-LB pipeline, the colab notebook generator. The thing.
metric_engine.py        Metric + visualization eval. get_metric_context (engine-side data loader), evaluate_dynamic_metric (in-process exec), evaluate_in_sandbox (docker-shell-out path).
bench_cache.py          NEW. Disk-bounded LRU cache. Pointer-mode + remote-submission both go through it.
tasks.py                Celery tasks. process_submission, batch recalculation. Imports from app — circular shape on purpose, see CLAUDE.md.
requirements.txt        Pinned: `datasets>=2.20,<3.0`. Don't bump past 2.x without re-doing the script-loader story.
fly.toml                2 GB VM, 20 GB volume.
```

### Templates (`templates/`)

- `base.html` — chrome.
- `home.html` — owned datasets + LBs + thumbnails.
- `explore.html` — public LB grid (with thumbnails).
- `dataset_view.html` — per-dataset page (column-visibility model + sample table).
- `leaderboard.html` — per-LB page (submission contract card, Colab back-link, collapsed filters, split-button submit dropdown).
- `auto_lb_preview.html` — preview-and-edit before auto-LB commit.
- `_user_avatar.html` — server-rendered OAuth avatar macro.
- `hf_import_preview.html` — HF auto-import field-mapping preview.

### Auxiliary helpers

- `scripts/seed_datasets.py` — batch HF auto-import (paused).
- `scripts/wipe_seeded.py` — clean slate.
- `scripts/prune_seed_configs.py` — drop unloadable entries from a config.
- `seed_data/*.json` — curated 10-per-domain HF repo lists.
- `seed_data/seed_baselines.py` — generic baseline-runner.
- `seed_data/baselines_<domain>.py` — domain stubs.

### Database (SQLite, Fly-volume-backed)

```
User                — OAuth identity (display_name, avatar_url, api_token, hf_token, is_admin)
Dataset             — global namespace; owner_user_id, visibility, source_kind ('hf-auto'|...).
                      NEXT: storage_mode ('local' | 'hf-pointer').
Sample              — per-dataset.
                      NEXT: source_ref_json (nullable, populated for hf-pointer).
CustomField         — flexible per-sample data: image|scalar|metric|histogram|depth|json|text.
Leaderboard         — owns metrics + visualizations + submissions.
Submission          — owner_user_id, source_colab_url, processing_status.
                      NEXT: storage_mode, remote_url, remote_auth_ref, content_hash.
LeaderboardMetric   — link table: arg_mappings (JSON), target_name, sort_direction, pooling_type.
GlobalMetric        — name (kind-only after the suffix-drop), python_code.
LeaderboardVisualization, GlobalVisualization — analogues for viz.
UserColabGist       — per-user gist mapping (user_id, lb_id) → (gist_id, gist_owner, sig).
CacheEntry          — NEW. Bench cache entries (cache_key PK, size_bytes, origin, last_accessed_at, created_at).
Tag                 — many-to-many with Dataset / Leaderboard / Sample.
leaderboard_datasets — NEXT: add `role` column ('primary' | 'gt_source').
```

`check_and_migrate_db()` runs on every boot. Adding a new column =
model definition + migration entry in the `_ownership_migrations`
list. Adding a new table = explicit `CREATE TABLE` block earlier in
that function. SQLite-only DDL, no Alembic.

### CLAUDE.md is the orientation doc.

Don't skip it before starting work. Documents the circular import
shape between `app.py` and `tasks.py`, the upload folder layout, the
`metric_*` convention pre-rewrite, and other things that aren't
obvious from the code.

---

## 5. Queued follow-ups (ordered by leverage)

After the active refactor (pointer-mode + remote submissions +
paired datasets), in expected-leverage order:

### A. Sandboxed eval that supports Option B's arrays

Today the sandbox path can't accept numpy-array inputs. Bytes-over-
JSON for image/depth so the docker container can run depth/image
metrics. Required before opening submissions to untrusted users.

### B. Replace placeholder `predictor_fn` stubs in the baseline runners

`seed_data/baselines_<domain>.py` returns naive scalars (mean depth,
dominant class, mean intensity) just to verify the upload pipe.
Real benchmarks need full depth maps / masks / images. Wait until
seeding is unpaused.

### C. Submission self-discovery of SOURCE_COLAB_URL

Today the gist URL is baked in via `_personalize_notebook_for_user`.
Cleaner: notebook detects its own URL via `google.colab.runtime.*`
at upload time so hand-edited copies still back-link correctly.

### D. LLM-proposed pred-field naming

Proposer hardcodes `<col>_pred`. Sometimes a different name reads
better (e.g. `restored_pred` for an `original` clean-image GT).
Could let the LLM mapping output suggest pred-field names during
the auto-LB preview.

### E. Multi-modal proposers

- Audio: WER / MOS-style.
- Video: per-frame metrics with temporal pooling.

Lower priority; not in the current domain set.

### F. `Dataset.storage_bytes` cache invalidation

Currently set once at import. After pointer-mode lands this is
mostly irrelevant (dataset bytes go to ~zero), but for any remaining
local datasets we should invalidate on mutating writes.

---

## 6. How to pick this back up in another session

1. Read `CLAUDE.md` first (architecture orientation).
2. Read this doc.
3. `git log --oneline -30` for the most recent surface area.
4. **Active workstream:** the pointer-mode refactor. Cache layer is
   in `bench_cache.py`; next is wiring it to the dataset import path
   (Section 1, "Concrete next steps").
5. Tests live in `tests/test_*.py`. **658 passing.** Run
   `python -m pytest -q` from repo root before any non-trivial
   change.
6. Deploy: `flyctl deploy --remote-only`. Fly machine ID via
   `flyctl machine list -a benchhub`.
7. Don't seed yet. The seeding scaffolding still works but the
   pivot rationale is: don't fill the volume with cloned bytes
   we're about to refactor away.

### Memory note

There's an active memory pin to surface the storage refactor when
the user mentions seeding being done. After the refactor lands,
check the memory entry's status — likely time to delete it since
the project will have moved on.
