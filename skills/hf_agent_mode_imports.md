---
name: hf-agent-mode-imports
description: "Lessons + heuristics for importing HuggingFace datasets via the file-tree (agent-mode) path in scripts/import_hf_agent.py — what works, what to skip, all the per-format conversions and naming quirks I hit."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

When the user asks to import HuggingFace datasets "in agent mode"
(i.e. without relying on Croissant), the canonical script is
`scripts/import_hf_agent.py`. This file collects every quirk I had
to learn while writing it; touch it back up whenever a new edge
case surfaces.

## What "agent mode" means here

**It is NOT just "skip Croissant."** It is "treat every import like a
research task: read the dataset card, list the actual file tree, probe
a real shard/parquet, THEN write the importer config." Skipping the
investigation step and guessing from convention (shard URL patterns,
WebDataset extension splits, README claims that don't match the tree)
is what causes silent partial imports and the "you missed the shards"
class of bugs.

Mandatory steps before writing any per-dataset code:

1. **Fetch the dataset card** (`README.md` + frontmatter) and READ it
   end-to-end. Look for `## Dataset Structure`, `## Loading`, and any
   mention of shards, splits, configs, or "the parquet table is a
   subset of …" caveats. Frontmatter `configs:` / `data_files:` glob
   patterns are authoritative — far more so than the file tree.
2. **List the actual tree** via
   `/api/datasets/<repo>/tree/main?recursive=1` and reconcile against
   what the card claims. If the card says "100 shards under
   `shards/`" and the tree has them, build a shard-based importer
   (not parquet). If the card mentions an additional modality that
   isn't in the parquet preview, find it on disk before deciding the
   schema is parquet-only.
3. **Probe ONE shard / parquet / image file end-to-end** — actually
   open it, walk its entries, confirm the regex you intend to use
   actually matches. For WebDataset shards: open the first tar with
   `tarfile.open(mode='r|*')`, print 20 entry names, derive the
   key/modality split from the real names, NOT from "WebDataset
   convention says X." The `pixparse/cc3m-wds` failure on
   2026-05-26 was exactly this: I wrote the regex from
   convention and the actual entries used a different extension
   structure.
4. **For multi-config repos**, the card's frontmatter `configs:`
   table tells you which configs hold which modalities. Don't assume
   the default config has everything.

Counter-examples (do NOT do this):
- Hardcoding a guessed shard URL like `shard-0000.tar` without
  listing the directory first (broke
  `AnonymouScientist/SiliciclasticReservoirs` — actual filenames
  didn't follow that pattern).
- Importing only what's in the `dataset_info` features dict when the
  README explicitly lists additional shard-only modalities (was the
  ego-1k miss — README's "Dataset structure" section described 12
  cameras under `shards/`, but the parquet preview only had table
  metadata).
- Trusting the HF dataset viewer's column list as the complete
  schema. It shows what the uploader registered with
  `dataset_info`, not what's on disk.

Compare:
- Croissant-based importer (hf-croissant-import is what
  `tasks.run_hf_import` and the admin /import_from_hf UI use)
  only sees columns the dataset uploader exposed in `dataset_info`.
- Most popular vision benchmarks ship as raw file trees (one folder
  per modality, sample ids in filenames) with no Croissant doc.
- Agent mode walks `/api/datasets/<repo>/tree/main?recursive=1`
  directly, pairs files into multi-modality samples, and imports
  via the same typed-manifest pipeline (`import_typed_dataset`).
- It can also run via `scripts/import_one_per_domain.py` which
  walks the BH task-domain list and picks the most-downloaded
  candidate per domain that matches one of the layouts.

## We are a vision-oriented benchmark — shards are not optional

When a vision dataset ships any portion of its image bytes in
WebDataset tar shards (under `shards/`, `wds/`, or globbed via
`configs: → data_files: "*.tar"` in the README frontmatter),
**import the shards too**, not just the parquet/arrow tables that
sit alongside them. Parquet alongside shards is usually metadata
(captions, labels, splits) and the shards are the actual pixels.
Importing the parquet alone gives us a "dataset" with no visual
asset, which defeats the point of cataloguing it.

Implementation pattern lives in `/tmp/import_shards.py`
(2026-05-26): stream the tar via `urllib.urlopen` + open with
`tarfile.open(mode='r|*')`, group entries by sample key, materialise
into a typed-staging dir, hand off to `import_typed_dataset`. Run
multiple datasets in parallel via `ProcessPoolExecutor(max_workers=3)`
— HF CDN tolerates that fine; more workers start hitting per-IP
rate limits.

The shard-importer regex for splitting entry name into
`(sample_key, modality)` is per-repo. Probe the actual entries
first (step 3 above) — there is no universal pattern.

## Skip image/video-generation datasets entirely

**Policy (2026-05-26):** datasets tagged `text-to-image`,
`text-to-video`, `image-to-video`, `unconditional-image-generation`,
`video-generation`, or `image-generation` are NOT useful for the
catalog. The user dropped `OpenRaiser/CoW-Bench` specifically and
asked to "exclude any image/video generation datasets."

**Why these don't fit:**
- The "ground truth" for generation is an evaluator's judgment of
  the generated artifact (FID, CLIP-score, human eval). There's no
  per-sample target to compare against in a classic metric loop.
- The repos commonly redistribute **other models' generated
  outputs** as JPGs (see CoW-Bench's `model_outputs/` and `video_cut/`
  — both contain pre-extracted frames from Sora/Kling/etc., not
  source videos). That's commentary on the generation models, not a
  benchmark we can run new submissions against.
- The "input" → "expected output" pattern doesn't hold: the input
  is a text prompt and the output is supposed to be created by the
  submitter, not predicted from a fixed sample.

**How to apply:**
- Remove `text-to-image`, `text-to-video`, `image-to-video`,
  `unconditional-image-generation`, `video-generation`,
  `image-generation`, `image-to-3d` from any TASKS list the bulk
  loop walks.
- In the LLM classifier system prompt, add to the SKIP list:
  "the repo's primary task is image / video / 3d generation
  (any `*-generation` tag, `text-to-image`, `text-to-video`,
  `image-to-video`, `image-to-3d`)."
- Existing imports under those tags should be removed.

Edge case: a dataset tagged BOTH `depth-estimation` AND
`image-to-3d` (e.g. ego-1k) — keep, because the depth side is a
classical prediction task. Decide by the dominant task tag + what
the README's "Expected output" / "Recommended usage" section says.

## No sampling — take ALL rows from the chosen eval split

**Policy (2026-05-26):** every import takes the full eval split,
no cap. `sample_cap=-1`, `max_samples=None`. Previously we capped
at 50/dataset for speed; that produced misleading messages ("50 of
0 rows ... via head sampling") and meant leaderboards ranked
submissions on a tiny holdout slice that didn't represent the
dataset. The user explicitly asked to remove the cap.

Implications:
- A dataset's full eval split lands on disk. Sizes vary wildly —
  cifar10 test = 10k samples, ImageNet val = 50k, CC validation =
  15k. Watch the 200 GB ceiling.
- Some splits are millions of rows (rare on `test`/`validation` but
  common when the only available split is something weird like
  `audit`). When the row count looks unreasonable (>500k) for a
  benchmark, decide per-dataset whether to skip or downsample.
- The bulk loop's pre-flight should print the row count BEFORE
  starting the download so we can sanity-check vs disk space.

Implementation:
- `try_parquet` in the bulk LLM loop: pass `sample_cap=-1` to
  `materialize_hf_to_typed_dir`.
- Per-repo agent-mode scripts (ego-1k, ECCV, MDE, SPIN-UV, CC, ...):
  remove the `MAX_SAMPLES` slice / set the limit to a large number
  (10_000) since some are bounded by the natural split anyway.
- Record `total_rows_in_split` in `source_metadata` so the
  dataset-view template shows the full "Imported X of Y" line.

See [sampling-policy-no-cap](sampling_policy_no_cap.md) for the rationale memory.

## Column-completeness rule — every upstream column lands SOMEWHERE

Every per-repo planner (the functions in `/tmp/import_*.py` PLANS
registries, or any custom importer you write next) MUST start by
enumerating EVERY upstream column / file kind, then explicitly
decide what to do with each:

- **Map it** to a typed field, OR
- **Discard it intentionally** with a one-line comment naming the
  reason (binary blob we can't decode, license string we don't
  care about, internal upstream id, etc.)

The trap: planners tend to only wire the columns the author
remembered. Silently-dropped columns mean the imported dataset
looks complete in the catalog but is missing useful metadata that
the upstream data has. The bug isn't visible until a user asks
"where's the shelfmark / category_name / source_round?"

Audit on 2026-05-26 found that **medieval-segmentation** dropped
`shelfmark, century, project`; **SPIN-UV** dropped `width, height,
date_captured, file_name`; **cmevs-erp-eval** dropped `source_round,
frame_count, original_scene_id`. All backfilled via one-shot scripts.
The per-repo planners in `/tmp/import_imagefolder.py` updated to
include them on any future re-import.

**How to apply (mandatory step in agent-mode imports):**

1. List the upstream columns explicitly in your planner's docstring:
   ```
   Upstream columns (per row of test/metadata.jsonl):
     file_name, width, height, objects, shelfmark, century, project
   ```
2. For each column, the planner either constructs a field for it or
   adds a comment explaining the drop:
   ```python
   # license: int FK into coco_categories, not useful at sample level
   # date_captured: keep
   ```
3. At the END of the planner, sanity-check: number of fields in the
   manifest >= 80% of upstream column count (modulo declared drops).
   If not, surface a warning before importing.
4. The "discard log" is critical — without it, a re-import six
   months later will silently drop the same columns again.

When backfilling later (column X turns out to be useful after all):
- New `DatasetField` row + per-sample `CustomField` rows via a
  one-shot like `/tmp/backfill_*.py`.
- Update the per-repo planner so future re-imports don't drop it
  again. The DB-side fix alone is not durable.

## Three-source modality discovery (card + name + tree)

Before deciding what fields to import, triangulate the expected
modalities from three independent sources. If any one of them
contradicts your manifest, stop and investigate.

1. **Dataset card / README** — read it. Look for:
   - YAML frontmatter `task_categories:` and `tags:` (most reliable).
   - A `## Dataset Structure` / `## Directory Layout` / `## File
     Semantics` section that describes the on-disk shape.
   - "Expected input" / "Expected output" / "Recommended usage"
     prose — those name the modalities by purpose.
   - A `## Data Preparation` section that references files NOT
     visible in the surface tree (e.g. ECCV's `depth.npz`
     was buried two levels deeper than the root listing showed).
2. **Repo name + pretty_name** — `*_Depth_Estimation`, `*_Segmentation`,
   `*_Captioning`, `*-cable` (MVTec defect), `ego-1k` (egocentric
   multiview), etc. If the name implies a modality, look for it.
3. **File layout** — actual paths from
   `/api/datasets/<repo>/tree/main?recursive=1` (and recursive
   sub-queries if the root truncates at ~1000 files). Match the
   layout against the README description; mismatches usually mean
   the upload is partial or the README documents a planned final
   form.

The three sources should agree. When they don't:
- Card says depth-estimation, name has "Depth", tree has no
  `depth.npz` → recurse into subdirs; HF list may be truncated.
- Card promises 12 cameras but tree only shows 1 → likely WebDataset
  shards under a different path (e.g. ego-1k's `shards/`).
- Name says "Segmentation" but the tree has only `*.jpg` + a
  jsonl with polygon coords → the GT IS the json (polygons), not
  a mask PNG. Don't synthesize a missing mask; use the json as
  the gt-kind field and consider whether to re-categorize.

Encode the conclusion as a list of `(field_name, kind, role,
expected_glob)` tuples in your importer config — the glob is what
makes the code reviewable (someone can verify the glob against the
tree without re-reading the README).

## Category dictates required modalities — if missing, the import is broken

A dataset's task tag / category sets a contract on which modalities
MUST exist after import. If the category says depth-estimation but
your import produced no `depth`-kind field, the import is broken
even if the row count looks right and the page renders. The user
will (rightly) flag it. The audit on 2026-05-26 found ECCV depth
imported with rgb-only because depth.npz was nested two levels
deeper than the surface tree listing returned.

**Required-modality contracts** (a field of the listed kind MUST be
present after import for the dataset to count as correctly imported):

| Category / task tag                             | Required kind(s)              |
|-------------------------------------------------|-------------------------------|
| depth-estimation, monocular-depth-estimation    | `depth`                       |
| image-segmentation, semantic-segmentation,      | `mask`                        |
| instance-segmentation, mask-generation          |                               |
| object-detection                                | `json` (bboxes)               |
| image-classification, zero-shot-image-class.    | `label` OR `text` GT          |
| image-to-text, image-captioning                 | `text` GT                     |
| pose-estimation, keypoint-detection             | `json` (keypoints)            |
| normal-estimation                               | `image` (normal map) or `json`|
| optical-flow                                    | `depth`-like / `json`         |
| novel-view-synthesis, multiview                 | ≥2 `image` fields (multi-cam) |

The `image` input field is implicit — every vision task needs at
least one image-kind input.

**Enforce it at import time:** before calling `import_typed_dataset`,
check the manifest fields against the expected kind set for the
dataset's category. If a required kind is missing, raise — don't
silently produce a phantom row. Add the check to per-dataset
importers AND to the bulk loop's `try_*` paths.

**Also enforce it at scan time:** a periodic sweep over the catalog
should flag any Dataset where `category` is in the table above but
no DatasetField of the required kind exists. Such rows should be
re-imported (or marked broken), not left in the catalog.

When the file-tree probe doesn't surface a required modality, do
NOT give up — recurse deeper. HF's tree API has a per-call file
cap (looks like ~1000), so a recursive=1 query on the root can
silently truncate. Drill into each scene/sample subdir individually
(`/api/datasets/<repo>/tree/main/<subpath>?recursive=1`) and reconcile.

## URL-only datasets (e.g. conceptual_captions) — need a crawler

Some HF datasets store `image_url` strings instead of image bytes
(documented in their dataset card). Conceptual Captions is the
canonical example: parquet has `image_url` + `caption`, pixels
hot-linked from the open web. Importing the parquet alone gives a
table of URLs, not a visual dataset.

**The fix is a per-URL crawl** with strict timeouts + content-type
validation. Walk the URLs, keep the first N that return a 2xx with
an image content-type, and decode through PIL to filter out
corrupted-but-served bytes. Expect ~35-40% dead URLs for any crawl
that's >5 years old.

Pattern lives in `/tmp/import_cc.py` (2026-05-26): for each URL,
8-second timeout, `Accept: image/*` header, 20 MB cap per image,
re-encode to canonical PNG before staging. Keep the original URL
as a `text` field for provenance.

Heuristic for the LLM-agent loop: dataset card mentions
"image_url" / "hot-linked" / "URLs not bytes" → don't route through
`parquet`/`file_tree`, hand off to a URL-crawler.

## `*_path` columns point at repo files — fetch + load as the typed kind

A parquet/metadata column named `image_path` / `*_path` / `file_name`
usually holds a path RELATIVE to the repo (or to a sub-root), not the
bytes. Importing it as a `text` field gives a column of paths, not an
image — the user (rightly) flags "you should have parsed them as
bh.Image." This is distinct from URL-only datasets (external http
URLs, need a crawler) — here the file is IN the same HF repo, fetch it
with `hf_hub_download`.

`zabir1996/mip-bench` (ds 70) was imported parquet-full with
`image_path` stored as text (`"Lecture 1/Images/Slide1.JPG"`). The
actual files live at `Lectures/Lecture 1/Images/Slide1.JPG` — the
column is relative to a `Lectures/` root that the value omits. Fixed
2026-06-02 by a one-off: for each sample, resolve `repo_path =
"Lectures/" + image_path` (fall back to the bare path), `hf_hub_download`
it, `benchhub.preview.image_preview(bytes)` → write
`uploads/datasets/<id>/image/<sample>.jpg`, add a `DatasetField(kind=
'image', role='gt')` + per-sample `CustomField(data_type='image',
value_text=<rel path>)`. 210/211 loaded (1 slide genuinely absent in
the repo).

**How to apply:**
- When a planner sees a `*_path`/`file_name`/`image_path` column, check
  whether the value matches a real file in the repo tree (try the bare
  value AND common root prefixes like `Lectures/`, `images/`, `data/`).
  If it does, treat it as a file-backed kind (image/audio/...), not text.
- Keep the original path as a `text` field too if useful for provenance.
- The repo-relative root is per-dataset — list the tree and diff one
  sample's `image_path` against the real paths to find the prefix.

## imagefolder format is the dominant failure mode (NOT shards)

An audit of 28 parquet-imported datasets on 2026-05-26 found
**zero** tar-shard misses besides ego-1k, but **10 high-impact
misses where the dataset is `imagefolder`-format** — i.e. the
README frontmatter looks like:

```yaml
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train/**/*.png
      - split: test
        path: data/test/**/*.png
```

…and there is no parquet at all (or there's a small parquet with
only labels/metadata, and the pixels live in sidecar PNG/JPG
trees). Our parquet-via-datasets-server importer silently grabs
just the `dataset_info` schema for these and ingests zero pixels —
giving us a "dataset" row with the right field names and no actual
data on disk.

**How to detect**: in the README YAML frontmatter, if `data_files`
glob ends in `*.png`, `*.jpg`, `*.jpeg`, `*.tif`, `*.zip`, or
`*.json[l]`, it's NOT a parquet repo. If there's no `data_files`
entry at all and the tree has no `*.parquet`, same story. Probe
the tree (step 2 of [agent-mode-mandate](agent_mode_mandate.md)) and look at extensions.

**Common layouts inside imagefolder repos:**

- **`<split>/<class>/<id>.jpg`** — classification (beans, MVTec
  test images). Join: folder-name → class label.
- **`<split>/<class>/<id>.png` + `ground_truth/<class>/<id>_mask.png`**
  — anomaly detection with paired masks (MVTec-AD family). Join:
  class folder + zero-padded index.
- **`<split>/<scene>/<asset>.{png,npz,json}`** — scene-level multi-
  modal (event/depth, KITTI-raw, ECCV depth est.). Join: per-scene
  manifest or timestamp.
- **`<split>/metadata.jsonl`** + sibling image files referenced by
  `file_name` (HF datasets-script-less convention). Join: jsonl row
  `file_name` → file on disk.
- **Hierarchical nested folders** (`p<N>/d<M>/L<k>/<idx>.png` for
  GroMo25; `<date>/<drive>/image_02/data/<10digits>.png` for KITTI-
  raw) — deeper than our 3-level layout detector handles. Each
  needs per-repo glob.

**Implementation pattern for imagefolder ingest** (apply alongside
the existing parquet/shard paths):

1. Use HF tree API (`/api/datasets/<repo>/tree/main?recursive=1`)
   to enumerate files. Filter to image extensions + the metadata
   sidecar (`metadata.jsonl`, `<class>.csv`).
2. Build a sample-id → modality map. For class-folder layouts, the
   modality is "image" and the class becomes a `label` field. For
   paired-modality (image + mask, image + depth) it's two fields.
3. Download a capped subset (50 samples) of the image bytes via
   `huggingface_hub.hf_hub_download` per-file (slow but reliable);
   for hundreds of files prefer a single `snapshot_download` with
   `allow_patterns` for the chosen subset.
4. Stage into the typed-manifest format + hand off to
   `import_typed_dataset`.
5. The role of the modality is usually `input`; `label`/`mask`/
   `depth` sidecars are `gt`. Reuse `_guess_role`.

**Datasets to attack first** (HIGH impact, simpler layouts):
- `MSherbinii/mvtec-ad-cable` — paired image + mask, per-class subdirs.
- `AI-Lab-Makerere/beans` — flat class folders (currently
  imported from the parquet-dup track; the original ZIPs work too).
- `CATMuS/medieval-segmentation` — flat image + metadata.jsonl.
- `MrigLabIITRopar/GroMo25` — image + per-crop CSV labels.

**Tricky (per-repo custom code needed):**
- `Kai-Yin-UoA/Monocular_Depth_Essentials` — KITTI + NYUDv2 nested.
- `ruikle123/SPIN-UV` — COCO seg + per-camera sequences + IR/depth.
- `COIN-Research-Group/sawhill-dataset` — video frames + .npy
  reference embeddings.
- `zabir1996/mip-bench` — lecture slides + transcripts, JSON split
  index.
- `Ethanliang99/ECCV_Event_Video_Depth_Estimation` — per-scene
  timestamp pairing (low/normal/event/depth).
- `OpenRaiser/CoW-Bench` — parquet has image input, but videos
  ARE in sidecar `video/`/`video_cut/` folders + 20+ model outputs.

The full audit punch list with file paths + join keys lives in
session 0f5a8564-be84-43d2-bf54-da49e47cc27b, search for
"HF Import Audit — Punch List".

## Three layouts the detector recognises

A. `<modality>/<sample_id>.<ext>` — modality is the first dir
   component, leaf stem must be numeric.

B. `<...path...>/<prefix>_<id>(_<sub>)?.<ext>` — id is `\d{3,}`;
   parent path + prefix (+ optional sub-modality after id) becomes
   the modality. Captured by `_parse_stem`.

C. `<split>/<modality>/<sample_id>.<ext>` — like A but one level
   deeper, prefers a test/validation/val split.

Each branch detects modalities, then runs `_has_paired_modalities`
twice (before AND after collapse — collapse can fold N variants of
"the same modality" into one bucket and invalidate the pairing
requirement).

## File-naming patterns to handle

- `cam_1_10002434.png` — modality `cam_1`, id `10002434`.
  Regex must match the LAST `_<digits>` (greedy from end).
- `panorama_0000.png` + `panorama_0000_depth.npy` —
  same base, sub-modality differentiator AFTER the id. Critical
  for cmevs-erp-eval and any HF dataset that pairs RGB with NPY
  depth. `_parse_stem` returns `('panorama/depth', '0000')` for
  the second one so collapse separates the buckets.
- `<10digits>.png` inside `image_02/data/` (KITTI raw) — too
  deeply nested for any of A/B/C. Skip such datasets; per-dataset
  custom code is needed.

## Files to skip in layout detection

- `._<name>` — AppleDouble metadata files from Mac-zipped repos
  (cmevs-erp-eval had these). They parse cleanly as "modalities"
  and create fake pairings that the field-completeness check
  later rejects, dropping all samples. `_is_junk_filename` filters
  these along with `.DS_Store`, `Thumbs.db`, any dotfile.

## Multi-modality vs flat-classification trees

A flat tree like LFW (`train/images/000/<person>/*.jpg`) looks
like multi-modality to a naive detector — every person folder is a
"modality" sharing image ids. Two filters needed:

1. **Sample-id overlap**: the two biggest modalities must share
   ≥ 20 sample ids. Most flat trees fail this since each person
   has only a few images.
2. **Semantic-category diversity** (the real fix): modalities
   must span ≥ 2 distinct semantic categories — `image-like`,
   `depth-like`, `mask-like`, `pose-like`, `flow-like`, etc.
   Category comes from the modality NAME, not the file extension,
   so PanoCity (`pano` + `pano_depth`, both PNG) still pairs.
   `_SEMANTIC_CATEGORY_TOKENS` is the bucket table;
   `_semantic_category` returns the LAST matching token because
   names tend to be `<container>_<modality>` (`pano_depth`,
   `cam_1_segmentation`).

## Canonical extensions are NOT advisory

`benchhub.types.DTYPES[kind].file_ext` is what
`import_typed_dataset` looks for on disk per `<field>/<sample>`:

| kind  | canonical |
|-------|-----------|
| image | `.png` |
| mask  | `.png` |
| depth | `.npz` |
| audio | `.wav` |
| text  | `.txt` |
| json  | `.json` |

The importer's per-sample existence check is strict — if your
staged file is named `<sample>.tiff` for an image-kind field, the
importer raises "manifest references missing files" and refuses
the whole dataset. Source-vs-canonical extension handling in
`_stage_dataset`:

- **image/mask**: rename to canonical `.png` regardless of source.
  PIL sniffs magic bytes so a TIFF stored as `.png` decodes fine.
- **depth from PNG/TIFF/etc.**: decode + repack as `.npz`
  (`_png_depth_to_npz`). Handles PIL modes I/I;16/F/L/RGB.
- **depth from `.npy`**: rewrap as `.npz` with `depth` key
  (`_npy_to_npz`). bh.Depth only loads `.npz`.
- **everything else with canonical ≠ src**: skip the file.
  The post-staging completeness check drops samples missing a
  field, so don't worry about partial data — the manifest is
  pruned to just the complete samples.

## Role-guessing convention

Default `role='gt'` for everything is wrong for multi-modality
benchmarks. `_guess_role` returns `'input'` when the modality is
`image-like` semantic + kind=`image` (rgb / rgba / photo / pano /
panorama / image / images / color); everything else stays `gt`.

- PanoCity: pano=input, pano_depth=gt ✓
- IntuitivePhysics: rgba=input, others (depth/normal/seg/flow)=gt
- cmevs-erp-eval: panorama=input, depth/pose=gt
- cifar10 (Croissant-imported, not agent-mode): img=input, label=gt

**Runtime safety net (not just import-time):** `_resolve_lb_input_samples`
in app.py defaults to treating image-kind field(s) as the input when
NOTHING else resolves to input (no field role + no per-LB
field_roles_json override). So "an image field present ⇒ it's the model
input" holds even for manual/parquet imports that never ran `_guess_role`
(cifar100, mip-bench were both gt-only until this). It only fires when no
explicit input exists, so image-to-image benchmarks (predicted image is
gt, rgb is the declared input) are unaffected. Added 2026-06-02.

## Collapse-by-suffix dedupe

Datasets like PanoCity ship one modality pair per
city/block/camera, producing 100+ modalities that are really
`pano` + `pano_depth` repeated. The collapser groups by the FINAL
path component after `/` (the modality token), dedupes by sample
id with FIRST-occurrence-wins. We lose the other blocks' samples
for any colliding id, but the cross-modality pairing (the
ENTIRE POINT of the dataset) survives. Namespacing by parent path
fails here because `pano/0001` and `pano_depth/0001` end up with
different namespaces and stop intersecting.

## Other gotchas

- **Gated repos** (`gated=manual` or `gated=auto` in
  `/api/datasets/<repo>`): tree-walk works anonymously but
  individual file downloads return 401. Filter on `gated` in the
  per-domain driver.
- **Full tree walks are slow**: PanoCity has 99k files = ~100
  pages = 2-3 min just to enumerate. The spot probe in
  `import_one_per_domain.py` caps at `max_files=3000,
  max_pages=3` to fail-fast on unmatchable candidates; the
  per-import call still walks the full tree once committed.
- **Pre-create the Dataset row** with `import_status='importing'`
  BEFORE downloads, set `'ready'` on success or `'failed'` with
  `import_error` on exception. Lets /datasets show the in-flight
  row + matches the async-import UX.
- **Run from the repo root** (`cd ~/benchhub` first) OR rely on
  the `_REPO_ROOT` sys.path insertion at the top of the script.
- **Quota / data-dir safety**: scripts refuse to write to
  `~/.dtofbenchmarking` without `--i-know-what-im-doing`. Always
  test with `BENCHHUB_DATA_DIR=$(mktemp -d)` first — see
  [data-dir-isolation](data_dir_isolation.md) for the painful reason.

## When agent mode WON'T help

- KITTI-raw nested layouts (`<drive>/<sync>/image_02/data/N.png`):
  modality is 3-4 levels deep, deeper than any of A/B/C. Would
  need a per-repo parser.
- Parquet / Arrow-only datasets: use the Croissant importer
  (hf-croissant-import). Most popular HF datasets in NLP and
  classification are parquet — agent mode finds nothing for those
  task domains.
- Datasets with sample ids encoded as filenames (UUID / arbitrary
  strings) instead of digits: my regex requires `\d{3,}`. The
  pattern catches most numeric-id benchmarks but skips
  string-keyed ones.

## What we actually imported (May 2026)

A clean baseline after wiping non-cifar10 and running the
per-domain driver + a follow-up parquet-fallback pass. Useful for
testing UI changes against real multi-modality data.

| id | repo | fields | task tag | path |
|---|---|---|---|---|
| 1 | uoft-cs/cifar10 | img + label | image-classification | parquet |
| 5 | YijingGuo/PanoCity | pano + pano_depth | depth-estimation | file-tree |
| 6 | worldbenchmark/IntuitivePhysics | rgba + depth + normal + forward_flow + segmentation | visual-question-answering | file-tree |
| 7 | anon-cmevs-2026/cmevs-erp-eval | panorama + depth + pose | image-to-image | file-tree |
| 8 | Tengpaz/worldrenderer-dataset-test | rgb + depth + normal | image-to-image | file-tree |
| 9 | CATMuS/medieval-segmentation | image + 6 metadata | image-segmentation / mask-generation | parquet |
| 10 | ruikle123/SPIN-UV | image + label | image-segmentation | parquet |
| 11 | MSherbinii/mvtec-ad-cable | image + label | zero-shot-image-classification | parquet |
| 12 | COIN-Research-Group/sawhill-dataset | image + label | image-feature-extraction | parquet |
| 13 | detection-datasets/coco | image + 4 metadata | object-detection | parquet |
| 14 | gksriharsha/chitralekha | image + text | image-to-text (multi-config) | parquet |

## Parquet fallback (for datasets with no file-tree pairing)

When the file-tree detector returns no layout, fall through to
the same code path the admin /import_from_hf form uses:

1. `benchhub.hf_search.fetch_dataset_info(repo_id)` — get the
   datasets-server `/info` features dict + splits.
2. `benchhub.hf_croissant.schema_from_hf_features(features)` —
   convert HF features to BH `CroissantField` list. No Croissant
   document required; the features dict alone is enough.
3. `materialize_hf_to_typed_dir(repo_id, split, fields=…)` — the
   existing helper. Now also accepts `config_name=` for
   multi-config repos.
4. `import_typed_dataset` on the staging dir, same as the
   file-tree path.

This is essentially `tasks.run_hf_import` minus the Celery wrapper
and minus the admin form data. The agent provides `fields` by
walking features.

**One-shot script lives at `/tmp/import_with_schema.py`** —
prototype, not committed. Should land in `scripts/` if we keep
using it. Args: `--repo`, `--split`, `--config-name`, `--sample-cap`.

## Text + JSON CustomFields need content in value_text, not the path

`bh.Text` and `bh.Json` are file-backed (`.txt`/`.json` ext), so
the natural `import_typed_dataset` path stores the relative file
path in `CustomField.value_text`. But every catalog view reads
`value_text` directly as the CONTENT (e.g. CATMuS `shelfmark`
should render the shelfmark string, not
`datasets/9/shelfmark/00000.txt`). Commit `4f15907` fixed this:
after the file copy, also read the bytes back into `value_text`
for kinds `text` and `json`. If an existing import shows file
paths in a text column, retro-patch:

```python
for cf in CustomField.query.filter(CustomField.data_type.in_(['text','json'])):
    if (cf.value_text or '').startswith(('datasets/', 'submissions/')):
        try:
            cf.value_text = open(
                os.path.join(UPLOAD, cf.value_text), encoding='utf-8'
            ).read().rstrip('\n')
        except (OSError, UnicodeDecodeError):
            pass
db.session.commit()
```

## Things to watch for in the parquet path

- **HF cache `.incomplete` files** left over from interrupted
  downloads land with permissions `000` (no read, no write) and
  block all retries with `PermissionError`. Nuke them before
  retrying:
  `find ~/.cache/huggingface/datasets/downloads/ -name '*.incomplete' -delete`.
- **Multi-config repos** (`arsaporta/symile-m3`, `gksriharsha/chitralekha`):
  `load_dataset` refuses to pick a config. Pass
  `config_name=<one of the configs>`. The error message lists the
  available configs.
- **Datasets with malformed metadata** (e.g. `vankey/RealText-V2`'s
  `ValueError: 'file_name' must be present as dictionary key in
  metadata files`) — skip; not our problem.
- **HF rate-limiting / connection drops** — `httpx.RemoteProtocolError:
  peer closed connection`. Retry, or pick a smaller dataset.
- **Stale placeholder rows** — every failed import leaves a
  `Dataset(import_status='failed')` row that blocks re-imports
  via the unique-name constraint. Cleanup snippet:
  ```python
  Dataset.query.filter(Dataset.import_status.in_(['importing','failed'])).delete()
  ```

## Domains we couldn't cover

Four task tags are essentially empty on HF — zero non-gated
candidates with `/info` schemas in top 30:
- `semantic-segmentation` (community uses `image-segmentation` tag)
- `instance-segmentation` (same — covered by SPIN-UV / CATMuS in
  the `image-segmentation` bucket)
- `monocular-depth-estimation` (covered by `depth-estimation`)
- `pose-estimation` (community uses `keypoint-detection`)

Don't bother probing these tags directly. If asked to cover them,
re-tag an existing dataset's category on the dataset settings
page rather than chasing nonexistent imports.

## Updating this file

Append new sections under "Other gotchas" whenever a fresh dataset
surfaces a quirk — same way [deploy-runbook](../docs/SELFHOST_RUNBOOK.md) grows. Don't bother
removing entries once the underlying code handles them; future
me/you can grep code to confirm.
