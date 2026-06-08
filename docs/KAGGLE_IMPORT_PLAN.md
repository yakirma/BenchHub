# Kaggle Dataset Import & Caching — Implementation Plan

**Status:** IN PROGRESS — Phase 1 (clean shapes) + Phase 3 conversion primitives implemented and unit-tested offline; live Kaggle verification (Phase 0 spike) still pending (no token was available). See `KAGGLE_IMPORT_PROGRESS.md` for the build log. **Date:** 2026-06-08.
**Author:** synthesized from a research pass (Kaggle API + Kaggle format taxonomy + the HF-import lessons) plus firsthand reading of the existing import code.

This plan reuses BenchHub's existing import machinery wherever possible and applies every hard-won HuggingFace-import lesson. It is the Kaggle analogue of the HF path under `/admin/import_from_hf` and the self-service `/import_from_files`.

---

## 0. TL;DR

The headline finding: **BenchHub's import engine is already source-agnostic, so a Kaggle importer is mostly an adapter, not a rewrite.** `benchhub/file_tree_import.py:materialize_file_tree(spec, files, fetch, staging)` → `benchhub/manifest.py:import_typed_dataset(preview_only=True)` does all the heavy lifting and knows nothing about HuggingFace. In `tasks.py:run_file_tree_import` the *only* HF-specific lines are:

```python
files  = api.list_repo_files(repo_id, repo_type='dataset', token=…)   # list
_fetch = lambda rel: hf_hub_download(repo_id, rel, repo_type='dataset', token=…)  # fetch one file
materialize_file_tree(spec, files, _fetch, staging, …)               # source-agnostic
…
existing.source_kind = 'hf'; existing.source_url = …                  # cosmetic stamp
```

**A Kaggle importer = implement `list_files` + `fetch(relpath)->localpath` for Kaggle, supply a license gate, add a discovery UX, and add ~5 computer-vision conversion primitives.** Everything else — sampling, quota-on-staged-bytes, the typed manifest, the preview tier, per-LB materialization, the smart mapping UI, decode-preview cycling — is inherited.

Three things Kaggle introduces that HF did not:
1. **A download wrinkle** — HF gives cheap per-file fetch; Kaggle's per-file download is awkward (often gzip-wrapped; large datasets ship as one monolithic zip), so the `fetch` callable needs a deliberate strategy.
2. **A hard legal gate** — Kaggle datasets carry *per-dataset* licenses, many non-redistributable. Caching + re-serving bytes to other users is redistribution and must be gated.
3. **Hidden-GT competition data** — competition `test.csv`/`sample_submission.csv` have no labels; only labeled `train` data is usable as ground truth.

---

## 1. Goals & non-goals

**Goals**
- Let a user **find** good Kaggle datasets (search, filter, sort by Kaggle's quality signals).
- Let them **explore/understand** a dataset *before* importing (metadata, file list, license, decoded sample preview) — zero bytes downloaded until they commit.
- Let them **import** it into a benchmark-ready typed Dataset with auto-detected field→kind→role mapping.
- **Cache** imported datasets (preview tier always; per-LB full-res on demand) and respect Kaggle rate limits + licenses.

**Non-goals (v1)**
- Kaggle **Competitions** with hidden test labels (only labeled train splits are usable).
- Kaggle **Models** hub (weights, not eval data).
- A first-class **time-series** kind.
- Delegated ("app-on-behalf-of-user") OAuth — Kaggle does not offer it.

---

## 2. What we reuse (verified firsthand)

| Capability | Existing code | Reuse for Kaggle |
|---|---|---|
| Decode any file tree → typed staging dir | `file_tree_import.materialize_file_tree(spec, files, fetch, staging, …)` | **verbatim** — feed it a Kaggle file list + fetch |
| Staging dir → Dataset+Sample+CustomField rows + preview tier | `manifest.import_typed_dataset(…, preview_only=True)` | **verbatim** |
| Smart auto-mapping (ext histogram, `<dir>/{id}.<ext>` patterns, label-vs-modality folder toggle, shared-archive/CSV detection, mask-name heuristics) | `file_tree_import.inspect_repo / analyze_levels / generate_spec_from_roles`, `_EXT_KIND_LOADER` | **verbatim** — operates on a `files` list |
| Canonical-ext transcode at staging | `file_tree_import._transcode_to_canonical / _stage_value`, `benchhub/preview.py` | **extend** with CV primitives (§4) |
| Sampling (full split, `sample_cap=-1`), variant fan-out, decode-preview cycling (`sample_offset`) | `materialize_file_tree` params; `/admin/import_from_hf/decode_preview` | **verbatim** |
| Quota check on actual staged bytes | `run_file_tree_import` → `check_quota(kind='dataset_create', …)` | **verbatim** |
| Discovery helpers (search / card / trending w/ 1-h TTL) | `benchhub/hf_search.py` + routes `/admin/import_from_hf/{search,card,trending}` | **mirror** as `benchhub/kaggle_search.py` |
| Preview/decode-before-commit + commit flow | `/admin/import_from_hf/{preview,decode_preview,commit}` | **mirror** |
| Async import w/ progress, one-import-at-a-time guard, failed-placeholder + Retry | `tasks.run_file_tree_import`, the HF commit route's guards | **mirror** as `run_kaggle_import` |
| Self-service access model (any user → private + private quota; admin → public) | HF commit route | **mirror**, overlaid with the license gate |

**Net:** the source-specific surface is **two functions + a license gate + a discovery UX + CV primitives**. Estimated >70% reuse.

### HF concept → Kaggle equivalent
| HF | Kaggle |
|---|---|
| `repo_id` (`owner/name`) | dataset ref `owner_slug/dataset_slug` (+ optional `datasetVersionNumber`) |
| `HfApi.list_repo_files` | `KaggleApi.dataset_list_files` (cheap, no bytes) |
| `hf_hub_download(one file)` | `dataset_download_file` (per-file, awkward) **or** `dataset_download_files`+unzip (whole zip) |
| `/api/datasets?search=` | `/api/v1/datasets/list` (search/filter/sort) |
| dataset card (README + tags + gated) | `dataset_view` / `metadata_get` (title, license, usability, file list, column descs) |
| gated repo (`401` on download) | non-redistributable **license** (legal gate, not access) |
| per-user HF token | per-user Kaggle token (`KAGGLE_USERNAME`/`KAGGLE_KEY`) — **no** app-on-behalf OAuth |

---

## 3. Kaggle integration layer (the adapter)

New module **`benchhub/kaggle_client.py`**:

- **Client:** the official `kaggle` pip package (`from kaggle.api.kaggle_api_extended import KaggleApi`) *or* a thin REST wrapper over `https://www.kaggle.com/api/v1`. **Pin the version** — the client recently changed (page-token pagination, `parquet`/`published` enum additions, auth methods); verify the installed signature in Phase 0. Wrap every call in a backoff layer (retry on `429`; Kaggle does not publish rate limits).
- **Auth (decision needed — see §11):** Kaggle is **per-user token only; no 3-legged OAuth**. Two viable models, mirroring how the HF path takes per-user creds:
  - (a) **Service account** — one token in `.env` (`KAGGLE_USERNAME`/`KAGGLE_KEY`, or `~/.kaggle/kaggle.json` mounted). Simple, shared, headless-friendly. All discovery + downloads run as this account.
  - (b) **BYO** — each user pastes their own `kaggle.json`. More correct attribution, but UX friction + token storage risk.
  - **Recommendation:** service account for **discovery/metadata** (cheap, shared) in v1; allow optional BYO token for **downloads** later.
- **`list_files(slug, version=None)`** → `[{name, totalBytes, fileType, columns}]` via `dataset_list_files`. Cheap, no byte transfer; feeds `inspect_repo` directly.
- **`fetch` strategy (the wrinkle — decision in §11):**
  - **Per-file** (`dataset_download_file`) — ideal for sampling (only fetch the N files a sample needs), but Kaggle often gzip-wraps single files and very large datasets are served only as one zip.
  - **Whole-zip-once** — `dataset_download_files`+unzip into a cache dir, then `fetch = lambda rel: <cache>/<rel>`. Always works, simplest, but downloads everything (bad for 100 GB datasets).
  - **Recommendation:** **whole-zip-once for v1** (correct + simple; the preview tier downscales afterward), add per-file fetch for sampling in Phase 3. Cache the extracted bytes keyed by `slug + version`.
- **Versioning:** capture `datasetVersionNumber` on every import for reproducibility and cache-busting (Kaggle datasets are mutable across versions).

---

## 4. Dataset shape taxonomy → detect-and-convert

Kaggle datasets cluster into ~19 recurring shapes. New module **`benchhub/kaggle_detect.py`** fingerprints the extracted tree (CSV columns, dir-per-class, paired dirs, RLE column, COCO/VOC/YOLO annotation files, `.dcm`/`.nii`) and emits a `spec` (the same structure `materialize_file_tree` consumes) — i.e. it *extends* `inspect_repo`.

### Shape → kind/role matrix
| # | Shape | Detect | Map (input → gt) | Task | Suitability |
|---|---|---|---|---|---|
| A | Single tabular CSV/parquet/xlsx — classification | one table; low-cardinality/`label`-named target col | feature cols → input (Scalar/Text/Label/Json); target → **Label** | tab/text classification | **clean** |
| A′ | …regression | target col continuous | features → input; target → **Scalar** | regression | **clean** |
| A″ | …text classification | long free-text col + label col | text → **Text**(input); label → **Label**(gt) | sentiment/NLP | **clean** |
| B | Multi-CSV competition (`train/test/sample_submission`) | `sample_submission.csv` present, or test missing target | build from **train.csv only** | per train.csv | **partial — train only** |
| C | ImageFolder (`root/<class>/img`) | leaf dirs = classes, only images | **Image**(input) → **Label**(gt, names=sorted classes) | image classification | **clean** |
| D | Image + CSV label map | image dir + CSV joining filename→label | **Image** → **Label/LabelList/Scalar** | classification/regression | **clean** |
| E | Detection — COCO JSON | JSON w/ `images`+`annotations`+`categories` | **Image** → **CocoDetections**/BBoxes(`xywh`) | detection / inst-seg | **clean** |
| F | Detection — Pascal VOC XML | per-image `.xml` `<bndbox>` | **Image** → **BBoxes**(`xyxy`, abs **1-indexed**) | detection | **clean** (convert) |
| G | Detection — YOLO TXT | per-image `.txt` 5 floats + names file | **Image** → **BBoxes** (denormalize ×W,H; `cxcywh`→`xyxy`; class 0-indexed) | detection | **clean** (convert) |
| H | Segmentation — paired image+mask | parallel `images/`+`masks/` dirs | **Image** → **Mask** (integer label map) | semantic seg | **clean** |
| I | Segmentation — **RLE in CSV** | CSV w/ `EncodedPixels`-like column | decode RLE → **Mask**; **Image** input | semantic/instance seg | **clean after decode** ⚠️ |
| J | Segmentation — palette/legend mask | PIL mode `P`, or RGB ≤~32 colors | palette/color → class id → **Mask** | semantic seg | **clean** (decode) |
| K | Segmentation — instance/Sequence stack | many per-instance masks / `Sequence(Image)` | **composite** into one instance-id **Mask** | instance seg | **clean after composite** |
| L | Audio + labels CSV | audio files + CSV join | **Audio**(input) → **Label/LabelList/Text** | audio cls / ASR | **clean** (transcode) |
| M | Depth | `rgb/`+`depth/` (16-bit PNG / `.npy`/`.npz`/`.exr`) | **Image** → **Depth** (apply scale, declare unit) | mono depth | **clean** |
| N | Restoration pairs (dirty→clean) | two parallel image dirs, name tokens dirty/clean | **Image**(input) → **Image**(gt) | denoise/SR/deblur/colorize | **clean** |
| O | Medical DICOM/NIfTI | `.dcm` magic / `.nii(.gz)` | DataTypeDef (lossless) **or** transcode → Image/Depth/Sequence | cls / seg | **needs DataTypeDef or transcode** |
| P | Time series | timestamp + numeric value col(s) | window → **Json/Scalar** (no native kind) | forecasting | **partial** |
| Q | Mixed / nested zips | archive-in-archive, heterogeneous | recurse, classify leaves, filter junk | varies | **partial — decompose** |
| — | Competition test / `sample_submission` | hidden labels | **UNSUITABLE as GT** | — | **reject as GT** |

### Conversion primitives (new module `benchhub/kaggle_convert.py`)
1. **RLE decode** → Mask. ⚠️ **#1 gotcha: the dominant Kaggle convention is COLUMN-MAJOR (`order='F'`), 1-indexed** (Airbus Ship, Carvana, SIIM-ACR, TGS Salt, Severstal, Sartorius, HuBMAP). Build a flat 0/1 vector from `(start-1, length)` pairs of size `H*W`, reshape `order='F'`. Decoding C-order **silently transposes every mask** — looks plausible, IoU is garbage. Multiple rows per image (per-class/instance) → composite into one label map; empty/NaN `EncodedPixels` = empty.
2. **Palette / color-legend → integer label map** (mode `P` direct; RGB needs a legend).
3. **BBox coordinate normalizer** — `xywh` top-left / `xyxy` abs-1-indexed / `cxcywh` normalized 0-indexed → BBoxes/CocoDetections (reconcile class index bases).
4. **Canonical-ext transcode** — jpg/bmp/tiff/gif/webp/jxl→PNG; mp3/flac/ogg→wav (soundfile); 16-bit depth PNG→float32 `.npz` (key `depth`) + unit. (Extends existing `_transcode_to_canonical`.)
5. **Sequence/instance-stack → single instance-id Mask** compositing (template: `scripts/add_ade20k_instance_field.py`; deterministic overlap ordering; uint16 path if >255 instances).

> Reminder: **BenchHub `Mask` is an integer label map** (mode `L`/`P`/`I;16`), *not* an RGB image. RLE/palette/legend masks must all become integer class-id arrays or IoU/Dice is meaningless.

---

## 5. The "hidden GT" guard

A guard in the importer that **refuses to build GT fields** from `test.csv`, `sample_submission.csv`, or unlabeled test dirs, and builds the eval split from labeled data only. When no usable GT exists, surface a clear **"no usable ground truth — not benchmarkable"** status rather than producing an empty-GT leaderboard. (Decision §11: import such dumps as catalog-only Datasets, or reject?)

---

## 6. Find → Explore → Understand → Import (UX)

Mirror the HF flow; lean on Kaggle's quality signals. All discovery runs on **cheap metadata calls — zero bytes** until commit.

- **FIND** — `/import_from_kaggle` search page → `kaggle_search.search_datasets` (`dataset_list` with `search`, `sort_by ∈ {hottest,votes,updated,active,published}`, `file_type ∈ {csv,parquet,json,sqlite,bigQuery,all}`, `license_name ∈ {cc,gpl,odb,other,all}`, `min/max_size` bytes). Show **usabilityRating**, votes, downloads, total size, **license badge**, last-updated. A trending-by-domain grid (Vision/NLP/Audio/Tabular) behind a ~1-h TTL cache (mirror HF trending).
- **UNDERSTAND** — a card view (`dataset_view`/`metadata_get`): title, subtitle, description, **license + redistributable badge**, usability, total size, **file list with sizes/types**, per-column descriptions for tabular, version count. Plus the **auto-detected shape + suggested field→kind→role mapping** for review.
- **EXPLORE** — decoded preview of one/few real samples (mirror `import_from_hf_preview` + `decode_preview` with `sample_offset` cycling). Downloads only the needed file(s) (or first zip member) so the user *sees* an image/mask/row before committing.
- **IMPORT/COMMIT** — editable mapping UI (reuse the file-tree mapping UI + `inspect_repo` suggestions + label/modality toggle), sampling choice, public/private selector **gated by license**, then enqueue `run_kaggle_import`.
- **Access model** — mirror HF: any signed-in user (self-service caching); non-admins → import lands **private**, charged to the **private** quota bucket, **one-import-at-a-time** guard; admins → public. The **license gate overlays this**: a non-redistributable dataset cannot be made public or materialized for other users (§8).

---

## 7. Caching & storage (four tiers)

1. **Metadata/search cache** — a small `KaggleMetaCache` table (or the existing TTL pattern) keyed by query/slug, ~1-h TTL. Cheap calls, but cache to respect rate limits and keep the browse UX snappy.
2. **Downloaded-bytes cache** — working dir `~/.dtofbenchmarking/kaggle_cache/<owner>__<slug>__v<N>/` holding the extracted zip; **dedup by slug+version**; size-budgeted LRU eviction. This is the raw download, distinct from the preview tier.
3. **Preview tier** — inherited (downscaled ≤512px JPG/PNG, waveform PNG, inline text/json; always present, global).
4. **Per-LB materialization** — inherited (full-res for the LB's chosen subset; counts against the LB owner's quota).

**Cache-busting:** key every tier on `datasetVersionNumber` (Kaggle datasets mutate across versions).

---

## 8. Legal / licensing — highest-priority risk

Kaggle datasets carry **per-dataset** licenses that vary widely (CC0 → CC-BY → CC-BY-SA → **CC-BY-NC** → GPL → ODbL → **"Other" / "Unknown" / "© original authors"**). **Caching bytes and re-serving them to other BenchHub users is redistribution** and is license-dependent. The `licenseName` is in metadata, so the gate can be automated:

- **Always** store + display the license and a Kaggle source link on the Dataset.
- Map license → bucket: **redistributable** (CC0, CC-BY, CC-BY-SA, ODbL…) vs **restricted** (NC, GPL-with-conditions, Other, Unknown).
- **Gate:**
  - Restricted/Unknown → **do not cache+re-serve to others.** Either (a) catalog stub that links out to Kaggle, or (b) allow a **private import for the importer's own use only** — never public, never materialized for other users.
  - Non-commercial (NC) → allow but flag; depends on whether BenchHub counts as "commercial" (**decision §11**).
- Persist `license_name` + `license_redistributable` on the Dataset; wire into the existing visibility/quota/dependency-guard logic so a restricted dataset **cannot be flipped public**.

---

## 9. Ops / scale / reliability

- **Async** — `tasks.run_kaggle_import` mirroring `run_file_tree_import`: pre-create the `Dataset` row (`importing`), list files, download (cached), `materialize_file_tree` → quota check on staged bytes → `import_typed_dataset(preview_only=True)`, stamp `source_kind='kaggle'`. Progress via `import_progress_json`.
- **Single-box discipline** — Kaggle datasets can be **100 GB+**. Reuse the **one-import-at-a-time per non-admin** guard; generous `soft_time_limit` (≈3600 s like the file-tree task — recall HF's 5-min-limit worker-hang gotcha); bound the download; never bulk-enqueue.
- **Rate limits** — backoff on `429`; the metadata cache (§7) cuts call volume; throttle discovery.
- **`data_dir` isolation** — never `import app` in a script without setting `BENCHHUB_DATA_DIR` (memory lesson; would bind the prod DB).
- **Failure/retry** — a failed import leaves an `import_status='failed'` placeholder (existing gotcha) — surface a Retry button + cleanup, exactly as HF does.
- **Junk filtering** — skip `._*`, `.DS_Store`, `Thumbs.db`, notebooks, READMEs (`inspect_repo`/file-tree already filter; extend for Kaggle bundle cruft).

---

## 10. Phasing (de-risk order)

- **Phase 0 — spike / de-risk (do first).** Verify the installed `kaggle` package signature + headless auth; probe **3 real datasets** (one tabular, one ImageFolder, one RLE-segmentation): `list_files` → `fetch` → decode-preview. Confirm the fetch strategy and that `inspect_repo` produces a sane spec from a Kaggle file list. *This is the single biggest unknown (API version drift + the fetch wrinkle) — resolve it before building UX.*
- **Phase 1 — MVP.** Self-service Kaggle import for the **clean shapes** (tabular A/A′/A″, ImageFolder C, image+CSV D, paired image+mask H, restoration pairs N) via the Kaggle adapter + whole-zip-once download + the hidden-GT guard + the license gate + a basic search/card/preview/commit UX reusing the file-tree mapping UI.
- **Phase 2 — discovery polish.** Trending-by-domain, filters, usability/votes sort, decode-preview cycling, Kaggle-tuned mapping suggestions.
- **Phase 3 — CV depth.** The conversion primitives (RLE column-major, bbox normalizer, palette decode, instance compositing), detection E/F/G, palette/instance seg J/K, depth M with scale/unit. Add per-file fetch for sampling.
- **Phase 4 — advanced.** Medical O (DataTypeDef vs transcode), time-series P decision, competition train-split import.

---

## 11. Open decisions (need product-owner input)

1. **Auth model** — service-account token (simple, shared) vs per-user BYO `kaggle.json` (correct attribution, friction) — or hybrid (service account for discovery, BYO for downloads)?
2. **License policy** — how strict? Only redistributable may go public? Allow restricted as private-only / link-out? **Is BenchHub "commercial"** (governs CC-BY-NC)?
3. **Non-benchmarkable dumps** (hidden GT / no GT) — import as **catalog-only** Datasets, or **reject** outright?
4. **DICOM/NIfTI** — lossless **DataTypeDef** (needs sandboxed decode per metric) vs lossy **transcode-at-import** (windowed Image / Depth / Sequence) as the default?
5. **Importer flow** — single auto-detect importer vs **auto-detect + confirm-the-shape** wizard? *(Recommend the latter, reusing the file-tree mapping UI.)*
6. **Time-series** in scope for v1? *(Recommend no.)*
7. **Download default** — whole-zip-once (simple) vs per-file (sampling)? *(Recommend whole-zip-once for v1.)*

---

## 12. Concrete new files / touch-points

**New**
- `benchhub/kaggle_client.py` — auth + `list_files` + `fetch` + 429 backoff (the adapter).
- `benchhub/kaggle_search.py` — search / card / trending (mirror `hf_search.py`).
- `benchhub/kaggle_detect.py` — shape fingerprinting → `spec` (extends `inspect_repo`).
- `benchhub/kaggle_convert.py` — RLE / bbox / palette / transcode / instance-composite primitives.
- `tasks.py: run_kaggle_import` — mirror `run_file_tree_import`.
- `templates/import_from_kaggle*.html` — mirror `admin_import_from_hf*.html`.

**Modified**
- `app.py` — routes `/import_from_kaggle` + `/search` `/card` `/trending` `/preview` `/decode_preview` `/commit` (mirror `/admin/import_from_hf*`).
- `app.py` model + `check_and_migrate_db` — `Dataset` columns: `kaggle_slug`, `kaggle_version`, `license_name`, `license_redistributable` (+ `source_kind='kaggle'`, `source_url`). ALTER blocks per the no-Alembic convention.
- `.env` — `KAGGLE_USERNAME` / `KAGGLE_KEY` (or mounted `~/.kaggle/kaggle.json`) for the service account.
- Cache dir `~/.dtofbenchmarking/kaggle_cache/` + a `KaggleMetaCache` table.
- `benchhub/file_tree_import.py` — extend `_transcode_to_canonical` / `_stage_value` to call the new CV primitives; teach `inspect_repo` the Kaggle annotation/RLE fingerprints (or keep that in `kaggle_detect.py`).

---

### Appendix — key references
- Kaggle API docs: <https://www.kaggle.com/docs/api>; CLI source: <https://github.com/Kaggle/kaggle-cli>
- RLE column-major decode reference: <https://www.kaggle.com/code/inversion/run-length-decoding-quick-start>
- Competitions (hidden test): <https://www.kaggle.com/docs/competitions>
- Token management: <https://www.kaggle.com/settings/api>
- Internal: `benchhub/file_tree_import.py`, `benchhub/manifest.py`, `benchhub/types.py`, `tasks.py:run_file_tree_import`, `app.py:/admin/import_from_hf*`, `scripts/import_hf_agent.py`, `scripts/add_ade20k_instance_field.py`, `docs/SESSION_NOTES_2026-05.md` (line ~217, Kaggle pair-split note).
