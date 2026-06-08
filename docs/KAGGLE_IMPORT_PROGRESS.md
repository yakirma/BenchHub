# Kaggle Import — Implementation Progress / Handoff

**Date:** 2026-06-08. **Companion to:** `docs/KAGGLE_IMPORT_PLAN.md` (the design).
This file is the *state* of the in-progress implementation so a fresh session
can continue without re-doing the reconnaissance.

---

## 0. Status at a glance

| Piece | State |
|---|---|
| Recon of reuse machinery | ✅ done (captured below) |
| Design forks (plan §11) | ✅ decided (below) |
| `benchhub/kaggle_convert.py` | ✅ done — **53 tests green** |
| `tests/test_kaggle_convert.py` | ✅ done (fixed the `test_rle_rows_tuple_form` fixture — 3×2 so a bg pixel survives) |
| `benchhub/kaggle_detect.py` | ✅ done — `tests/test_kaggle_detect.py` (22 tests green) |
| `benchhub/kaggle_client.py` | ✅ done — REST adapter + `classify_license` + 429 backoff |
| `benchhub/kaggle_search.py` | ✅ done — search/card/trending (1h TTL); `tests/test_kaggle_client.py` (48 tests, covers both, green) |
| `file_tree_import.py` conversion-loader hooks | ✅ done — `rle`/`coco`/`voc`/`yolo` loaders + `index:true` CSV/parquet/rle/coco; `tests/test_kaggle_file_tree.py` (8 tests green) |
| `tasks.py: run_kaggle_import` | ✅ done — mirrors `run_file_tree_import` |
| `app.py` routes + Dataset cols + migration + license gate | ✅ done — 7 routes, 4 cols, ALTER block, gate in `set_dataset_visibility` (verified via guarded import) |
| `templates/import_from_kaggle*.html` | ✅ done — `import_from_kaggle.html` (search+trending) + `import_from_kaggle_map.html` (detect+spec+preview+commit); linked from datasets.html + import_from_files.html |
| Docs (in-app + CLAUDE.md notes) | ✅ done — `templates/docs/importing.html` Kaggle section + loaders; root + `benchhub/CLAUDE.md`; plan flipped to IN PROGRESS |

**131 Kaggle unit tests green** (`tests/test_kaggle_{convert,detect,file_tree,client}.py`).
Verified offline + via a guarded `app` import on `~/benchhub/.venv` (routes register,
model cols present, migration adds them, task registered, license classifier correct,
entry page renders). Full app test suite can't run in the dev conda env (no flask/pandas).

### Still TODO (deferred / needs live access)
- **Phase 0 spike (the one real unknown):** verify the `EP_*` REST paths in
  `kaggle_client.py` against a live token (none was available) — they're from the
  public CLI swagger and may have drifted. Module constants, fix in one place.
- **Per-LB full-res re-materialize for kaggle datasets** — `lb_materialize.py` is
  HF-specific; a kaggle LB currently scores against the preview tier (downscaled).
  Wire a kaggle re-fetch (cache the zip, re-run materialize for the subset).
- Per-file fetch for sampling (v1 is whole-zip-once); palette/instance-seg auto-detect
  (J/K — converters exist, detect doesn't emit them); DICOM/NIfTI (O); competition
  train-split import; a `KaggleMetaCache` DB table (in-memory TTL used for now).
- `.env`/runbook: document `KAGGLE_USERNAME`/`KAGGLE_KEY` for the service account.

---

## 1. Environment facts (verified this session)

- Python **3.13.5**, numpy **2.4.6** (numpy 2.x — `np.unique(..., return_inverse=True)`
  inverse-shape quirk handled in `kaggle_convert.palette_to_labelmap` via `.reshape(-1)`).
- **pytest 9.0.3 installed** this session (was missing; `pip install pytest`).
- requests **2.32.3**, Pillow **12.2.0** present.
- **`kaggle` pip package is NOT installed** and **no Kaggle creds** (`~/.kaggle/kaggle.json`
  absent, `KAGGLE_USERNAME`/`KAGGLE_KEY` unset). ⇒ Adapter decision below.
- `kaggle` is **not** in `requirements.txt`.

---

## 2. Design forks — DECIDED (plan §11)

1. **Adapter transport:** build `kaggle_client.py` as a **`requests`-only REST wrapper**
   over `https://www.kaggle.com/api/v1` (requests already a dep) — **do NOT depend on the
   `kaggle` pip package**. Reason: package missing + no creds + REST is fully mockable for
   offline tests. Auth = HTTP Basic (`KAGGLE_USERNAME`/`KAGGLE_KEY` from env or
   `~/.kaggle/kaggle.json`), lazily resolved so import never fails without creds.
   ⚠️ The exact REST paths (`/datasets/list`, `/datasets/view/{owner}/{slug}`,
   `/datasets/list/{owner}/{slug}` for files, `/datasets/download/...`) should be kept as
   module-level constants and **verified against the live API in a Phase-0 spike** once a
   token exists — they're from the public Kaggle CLI swagger but may have drifted.
2. **Auth model:** service-account (single token in env), discovery + download share it.
   BYO-per-user deferred.
3. **License gate: STRICT.** Only **redistributable** licenses may go public/materialize for
   other users: CC0, CC-BY, CC-BY-SA, ODbL, PDDL, DbCL, CDLA-Permissive, U.S. Government
   Works, Apache/MIT/BSD. **Restricted** (never public, importer-private only): CC-BY-NC*,
   CC-BY-ND*, GPL-with-conditions, "Other", "Unknown", "© original authors", World Bank ToU,
   Reddit ToS. Persist `license_name` + `license_redistributable`; wire into the
   visibility/quota/publish-flip guard so a restricted dataset can't be flipped public.
4. **Hidden-GT guard:** refuse to build GT fields from `test.csv` / `sample_submission.csv`
   / unlabeled test dirs; build the eval split from labeled `train` data only. No usable GT
   ⇒ clear "not benchmarkable" status (not an empty-GT LB).
5. **Download:** **whole-zip-once** for v1 (`dataset_download_files` + unzip into a cache dir
   keyed by `owner__slug__vN`, then `fetch = lambda rel: <cache>/<rel>`). Per-file fetch for
   sampling deferred.
6. **Scope this pass:** Phase 1 clean shapes (tabular A/A′/A″, ImageFolder C, image+CSV D,
   paired image/mask H, restoration pairs N) **+** Phase 3 conversion primitives (already
   written in `kaggle_convert.py`). DICOM/NIfTI, time-series, competitions deferred.
7. **Importer flow:** auto-detect + confirm wizard, reusing the file-tree mapping UI.

---

## 3. Reuse machinery — exact signatures (so you don't re-recon)

### The source-agnostic engine — reuse VERBATIM
`benchhub/file_tree_import.py`:
```python
materialize_file_tree(spec, files, fetch, staging_dir, *,
    sample_cap=-1, sample_offset=0, dataset_name='dataset',
    token_filter=None, progress_cb=None) -> summary_dict
```
- `files`: flat list of repo-relative path **strings**.
- `fetch(repo_relpath) -> local_filesystem_path` (the ONLY source-specific callable).
- `staging_dir`: writes `manifest.json` + `<field>/<sample>.<ext>`.
- Returns `{name, samples, fields, total_rows_in_split, rows_written}`.

`spec` is a list of field dicts (see file_tree_import.py:11-36 docstring):
```python
{"name","kind","role"('input'|'gt'),"loader"('file'|'npz'|'json'|'csv'|'parquet'|
 'hdf5'|'zip'|'tar'|'gz'|'token'|'sequence'),"pattern":"dir/{id}.ext", ...loader-specific}
```
Helpers to reuse: `inspect_repo(files)`, `analyze_levels(files)`,
`generate_spec_from_roles(files, roles)`, `_EXT_KIND_LOADER`, `_data_files` (junk filter).
**Conversion-loader hook points:** `_transcode_to_canonical(kind, raw, dest_noext)` and
`_stage_value(kind, raw, dest_noext)` (file_tree_import.py:503 / 542) — add new branches
here (or a new loader in the `materialize_file_tree` elif chain ~line 816) that call the
`kaggle_convert` primitives for rle/bbox/palette kinds.

### Staging → DB rows + preview tier — reuse VERBATIM
`benchhub/manifest.py`:
```python
import_typed_dataset(staging_dir, *, db_session, Dataset, Sample, CustomField,
    upload_folder, owner_user_id=None, visibility='public', DatasetField=None,
    existing_dataset=None, tolerate_incomplete=False, preview_only=False,
    extra_kinds=None) -> (dataset_id, summary)
```
`preview_only=True` renders downscaled JPG/PNG/waveform + sets `Dataset.preview_only=True`.

### Typed output contracts (what the converters must produce) — `benchhub/types.py`
- **Mask** = INTEGER label map (H,W); PNG mode L (≤255) or I;16. `_stage_value('mask', ndarray, dest)` handles writing.
- **Depth** = float32 (H,W), `.npz` key `depth`, unit ∈ {meters, millimeters, unitless}.
- **BBoxes** = JSON `{"boxes":[[x1,y1,x2,y2],...],"format":"xyxy"|"xywh"|"cxcywh","labels"?,"scores"?}`.
- **CocoDetections** = list of `{"category_id","bbox":[x,y,w,h] (XYWH!),"segmentation","area","iscrowd"}`.
- **Label** inline; `label` token loader auto-builds a categorical vocab (`params.names`).

### HF path to MIRROR (app.py / tasks.py / hf_search.py)
- Routes (app.py): `/admin/import_from_hf` main (7377), `/search` (7397), `/card` (7408),
  `/trending` (7421), `/preview` (7712), `/decode_preview` (7840), `/commit` (7438).
  All `@login_required` (any user). Non-admin import → `visibility='private'`;
  **one-import-at-a-time guard** = `Dataset.query.filter_by(owner_user_id=..., import_status='importing').count()`.
- Task to mirror: `tasks.py:run_file_tree_import` (~196-311) — pre-create Dataset row
  (`import_status='importing'` + `import_progress_json`), list files, build `_fetch`,
  `materialize_file_tree` → quota check on staged bytes
  (`check_quota(owner, kind='dataset_create', incoming_bytes=, visibility=)`) →
  `import_typed_dataset(preview_only=True)` → stamp `source_kind`/`source_url`/`source_metadata`
  → `import_status='ready'`. Failure → `import_status='failed'` + `import_error` + rmtree, no re-raise.
  **No Celery soft/hard time limit** (deliberate — see benchhub/CLAUDE.md).
- `benchhub/hf_search.py` cache pattern to mirror for `kaggle_search.py`: module-level dict
  cache, **1-hour TTL** on `card_summary`/`trending_by_domain`; `search_datasets` uncached.
  Return shapes: search → `[{id,downloads,likes,description,gated}]`; trending →
  `{domain: [items]}`.

### Dataset model + migration convention (app.py)
- Import-relevant cols: `source_kind`(459), `source_url`(436), `source_metadata`(460 JSON),
  `preview_only`(472), `import_status`(437,'ready'), `import_progress_json`(445),
  `import_error`(446), `import_task_id`(447), `visibility`(420), `owner_user_id`(417),
  `storage_bytes`(426), `card_description`(481).
- **New cols to add** (per plan §12): `kaggle_slug`, `kaggle_version`, `license_name`,
  `license_redistributable`.
- Migration: no Alembic. Add ALTER blocks in `check_and_migrate_db()` (~app.py:18029+),
  guarded by `PRAGMA table_info(dataset)` membership check, wrapped in try/except. Example
  loop pattern at ~18238. `BENCHHUB_AUTO_MIGRATE=1` runs it on boot.

---

## 4. What `benchhub/kaggle_convert.py` already provides (DONE)

Pure numpy/PIL converters (offline-tested by `tests/test_kaggle_convert.py`):
- `decode_kaggle_rle(rle,h,w, order='F', one_indexed=True)` — **column-major, 1-indexed by
  default** (the #1 gotcha: C-order silently transposes). `encode_kaggle_rle` (inverse, for
  round-trip). `parse_rle`.
- `rle_rows_to_labelmap(rows,h,w, value_key, class_key=None, overlap='last')` — composite
  multiple RLE rows → integer label map (per-class or per-instance).
- `coco_uncompressed_rle_to_mask(counts,h,w)` — COCO alternating-run-length RLE.
- `bbox_to_xyxy(box, fmt, img_w=, img_h=)` (fmt: xywh/xyxy/xyxy_voc/cxcywh/cxcywh_norm),
  `yolo_line_to_xyxy`, `voc_box_to_xyxy` — all → canonical abs 0-indexed xyxy.
- `palette_to_labelmap(img, legend=)` — mode-P / RGB-legend / grayscale → integer label map.
- `composite_instance_masks(masks, overlap=)` — instance stack → instance-id map (uint16 >255).
- `depth16_to_float(img, scale=)` — 16-bit depth raster → float32 in unit.
- `downcast_labelmap` — uint8 (≤255) else uint16, matching Mask.encode().

---

## 5. Remaining build order (resume here)

1. **Run** `python -m pytest tests/test_kaggle_convert.py -q`; fix failures.
2. **`benchhub/kaggle_detect.py`** — fingerprint the extracted tree (CSV cols, dir-per-class,
   paired dirs, RLE column, COCO/VOC/YOLO files, `.dcm`/`.nii`) → emit a `spec` (extends
   `inspect_repo`). Implements the hidden-GT guard. + `tests/test_kaggle_detect.py` (synthetic
   file lists, no network).
3. **`benchhub/kaggle_client.py`** — REST-over-requests: `list_files(slug,version)`,
   whole-zip `download(slug,version)->cache_dir` + `fetch` factory, `view(slug)` metadata,
   429 backoff, **`classify_license(name) -> (bucket, redistributable: bool)`** (pure,
   test this hard). + `benchhub/kaggle_search.py` (search/card/trending, 1h TTL). Tests inject
   a fake `requests`/session — no live calls.
4. **Extend `file_tree_import.py`** additively: a loader (or `_stage_value` branch) that runs
   the `kaggle_convert` primitives for rle/bbox/palette source columns. + tests.
5. **`tasks.py: run_kaggle_import`** — mirror `run_file_tree_import` (download cached zip →
   `materialize_file_tree` → quota → `import_typed_dataset(preview_only=True)` →
   `source_kind='kaggle'`). Lazy-import `_registered_extra_kinds` inside the task (circular).
6. **`app.py`** — Dataset cols + `check_and_migrate_db` ALTER block; routes
   `/import_from_kaggle` + `/search /card /trending /preview /decode_preview /commit`
   (mirror HF, `@login_required`, private-for-non-admin, one-import guard); the **license
   gate** overlaid on visibility/publish-flip.
7. **`templates/import_from_kaggle*.html`** — mirror `admin_import_from_hf*.html`.
8. **Docs** — in-app `templates/docs/*.html` import page; note in root `CLAUDE.md` +
   `benchhub/CLAUDE.md`; flip `KAGGLE_IMPORT_PLAN.md` status to "in progress".
9. **Test gate:** `python -m pytest tests/test_kaggle_*.py -q` green. For any app.py import
   sanity check, set `BENCHHUB_DATA_DIR` to a temp dir FIRST (memory: importing `app`
   binds the prod DB).

---

## 6. Risk / gotcha reminders
- **RLE column-major** is the dominant correctness trap — already guarded in the converter.
- **numpy 2.x** `np.unique` inverse shape — handled; keep the `.reshape(-1)`.
- **Don't import `app` without `BENCHHUB_DATA_DIR`** set (nukes prod DB on create_all/drop_all).
- **app.py is one 6600-line file** — keep edits additive/surgical; verify with a guarded import.
- Kaggle datasets can be **100GB+** — keep the one-import-at-a-time guard; no bulk enqueue.
- Live Kaggle testing needs a token (none present) — until then, mock the REST layer.
