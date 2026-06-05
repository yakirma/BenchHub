---
name: file-tree-importer
description: "The user-declared file-tree HF importer — how it works, the source-descriptor spec, loaders, and where to extend it. Use this instead of writing bespoke per-dataset import scripts for paired-file / packed-npz repos."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 5c7cb6c2-c6d2-4a9f-9ba4-8a84e62b77b0
---

For HF repos that aren't tabular/parquet (Croissant can't read them) —
trees of paired files, packed `.npz`, per-sequence/quality folders —
there is now a **self-service, UI-driven importer**, not just bespoke
scripts. Prefer it over writing another one-off like
`scripts/import_hf_agent.py` / the ADE20K / mip-bench one-offs.

**Engine:** `benchhub/file_tree_import.py`. A spec is a list of field
descriptors; each declares a BH `kind`, a `role`, a `loader`, and a
`pattern` with `{token}` placeholders:

- `loader: file` — one file per sample; pattern e.g.
  `train/{seq}/normal/{id}.png`. The FIRST `file` field is the sample
  **index** (matching its pattern enumerates samples + captures tokens);
  other fields join by shared tokens. Samples sorted by filename.
- `loader: npz` — `shared=False` → one archive per sample
  (`{id}.npz` + `key`); `shared=True` → ONE archive holding all samples
  stacked along `axis`, each sample taking its frame by ordinal within
  its token-group **in sorted-filename order** (the ECCV `depth.npz`
  case: `(N,H,W,1)` → per-sample `(H,W)`).

Adapters stage each kind into the typed-manifest dir that
`import_typed_dataset(preview_only=True)` already ingests, so all the
downstream (preview tier, quota, dataset_view) is reused.

**Unified with the tabular importer.** One entry button; the tabular
(Croissant/parquet) preview hands off to the file-tree importer when it
finds no schema (auto-redirect to `/import_from_files/inspect?repo_id=`)
or via a "mapping wrong? → file-tree" link, and both entry forms have a
"Map raw files instead" submit (formaction → inspect). The file-tree map
page links back to the schema importer. `import_from_files_inspect`
accepts GET for these hand-offs.

**Flow / routes:** `/import_from_files` → `/inspect` (lists tree, ext
histogram, suggested `<dir>/{id}.<ext>` patterns) → mapping builder →
`/decode_preview` (decodes the FIRST sample per field → thumbnails so the
user confirms the interpretation — this catches the "arr_0 is events not
depth" trap) → `/commit` → `tasks.run_file_tree_import` (async,
hf_hub_download, quota-checked, soft_time_limit). Linked from the
`/datasets` sidebar. Same guardrails as the Croissant importer: full
split (no row cap — quota is the bound), private for non-admins,
one-import-at-a-time.

**Loaders shipped: file, token, npz, json, csv, parquet, hdf5, zip, tar, gz.**
- `token`: the field value IS a captured path token (no file read) —
  `<class>/{id}.png` → a `label` field from `{class}`. For label kind it
  builds a vocab → stores an int index + `names` (renders as the name).
  The index field's `file` pattern must capture that token. (Benjy
  typed_digital_signatures case.)
- `zip`/`tar`(/`.tar.gz`)/`gz`: container loaders — point at a `member`
  inside (with `{id}`/`{ordinal}`); extracted bytes run the per-kind
  adapter. A zip/tar whose `member` uses `{id}` can be the sample INDEX
  (members enumerate samples) — resolve_samples takes an optional `fetch`
  to list members. `gz` = single decompress (one member).
- **Subset/split filter**: commit's `filter_token`+`filter_value` → a
  single-value `token_filter` (import only where token==value) — the
  file-tree analog of the tabular split dropdown. Distinct from
  `variant_token` (fan-out → many datasets).
- `json`: dotted `pointer` with `{id}`/`{ordinal}` substitution
  (`frames.{id}.pose`, `ordered.{ordinal}.q`); per-sample file or shared
  doc (`shared=True`).
- `csv` / `parquet`: `column` selected per row, matched by `id_column`
  (vs the `id` token) or by row order when blank (share `_table_row`).
- `hdf5`: `key` = dataset path within the `.h5` (`group/depth`);
  per-sample whole array, or shared stacked array split along `axis` like
  npz. h5 handles cached + closed after the run.
**Variant automation shipped:** commit accepts `variant_token`; fans out
into one dataset per distinct value (`<name>_<value>`, cap 12), each via
`token_filter={token: value}` passed to the engine. The mapping UI also
has a collapsible visual file-tree browser (clicking a file fills the
pattern, leaf → `{id}`).

**Sequence/clip kind (Phase 1 shipped).** `benchhub.types.Sequence`
(kind="sequence", file_ext=".zip") holds an ordered list of homogeneous
frames (params `item_kind`=image/depth/mask, `fps`); stored as a ZIP of
per-frame encodings; `visualize()` muxes to mp4 via the **system ffmpeg**
(`/usr/bin/ffmpeg`; animated-GIF fallback if absent). The importer
`sequence` loader groups frames by the pattern tokens minus `{frame}`
(`clips/{id}/{frame}.png` → one clip per `{id}`); a sequence whose pattern
uses `{frame}` can be the sample index. dataset_view + comparison render
a `<video>` streamed from `/api/viz/<cf_id>` (generic decode+visualize).
**Phase 2 shipped:** the client is generic — `iter_samples` yields an
iterable `bh.Sequence` for a sequence input, and `predict(...,
clip=bh.Sequence([...]))` packs as `<field>/<sample>.zip`. Required two
server whitelist fixes so a sequence INPUT is fetchable: add `sequence`
to the `/samples` url-vs-inline branch AND the `api_leaderboard_inputs_archive`
file_fields list (both were `('image','mask','depth','audio')`). Pred
kind allowed via `DTYPES` in the schema editor; also added to
`_EXTRA_PRED_KINDS`.

**Path-level role wizard (shipped).** `analyze_levels(files)` +
`generate_spec_from_roles(files, roles)` power a "Describe the structure"
step (route `/import_from_files/from_roles`): the user tags each path
level id/modality/property/split/group/fixed and the field rows are
auto-generated (modality fans out per value; property → image + token
label; file level multi-ext fans out per ext; split/group → tokens). The
name-vocab (`_MODALITY_WORDS`, exact case-insensitive whole-name match)
now only seeds the *default* role per level.

**Bounded listing for huge repos (shipped).** The request-path routes
(inspect/from_roles/decode_preview/commit-validation) MUST NOT call
`HfApi().list_repo_files` directly — on a giant repo it walks the whole
tree (CT-RATE `ibrahimhamamci/CT-RATE`: 251k `.nii.gz` across 150k+ dirs
→ 125s) and hangs the request ("import preview takes forever"). Use
`app._list_hf_repo_files(repo_id, token=...)` → `(files, truncated)`
instead: it tries HF's server-paginated `list_repo_tree(recursive=True)`
under a 12s budget + 50k cap, and — key gotcha — that recursive listing
**emits the ENTIRE directory tree before any file**, so for a dir-first
giant it yields 0 files in the budget; the helper then bails (after a 6s
probe) to `_dfs_sample_repo_files` (non-recursive per-dir DFS, ~0.5s/dir,
reaches leaf files fast) for a representative sample. The Celery task
keeps the full `list_repo_files` (background, time-tolerant). `inspect`
also: cheap `_hf_repo_used_storage` size probe (`dataset_info(expand=
['usedStorage'])`, ~0.3s) for messaging; BLOCKS (flash+redirect) when no
files reachable; renders a "very large repo" truncation banner + an
unsupported-kind warning. `_file_tree_unsupported_warning(files)` matches
**compound** suffixes (`.nii.gz` etc. → "NIfTI medical volumes") because
the bare `.gz` last-ext is a legit container loader. Tests in
`tests/test_import_from_files_routes.py` (stub now provides
`list_repo_tree`+`RepoFile`+`dataset_info`, not just `list_repo_files`).

**Split/subfolder chooser before preview (shipped).** A repo bigger than
`_FILE_TREE_SCOPE_BYTES` (10 GB, from the `used_storage` probe) renders a
chooser (`import_from_files_scope.html`) BEFORE the heavy preview: cheap
non-recursive `_list_hf_subdirs(repo_id, prefix=...)` (~0.2s/level) lists
immediate subfolders with Browse (drill: `inspect?prefix=<dir>`) +
"Preview & map" (`inspect?prefix=<dir>&scoped=1`) actions + breadcrumb.
The chosen `prefix` threads through everything: `_list_hf_repo_files(...,
path_prefix=)` (scopes `list_repo_tree(path_in_repo=)`), from_roles +
decode_preview JSON bodies, the commit form hidden `path_prefix`, and
`run_file_tree_import(..., path_prefix=)` (scoped recursive listing in the
task, not full `list_repo_files`). KEY WIN: scoping bypasses the dir-first
pathology — CT-RATE whole = 125s/0-files; `dataset/valid` scoped = 2.4s,
3039 files, COMPLETE (recursive walk of a split's subtree finishes in
budget). The too-large block bounces to the PARENT chooser (never the same
scope → no redirect loop). `scoped=1` skips the chooser ("preview as-is").

**All planned loaders shipped.** To add another kind/loader: extend
`_STAGE_EXT`, the loader branch + ordinal precompute in
`materialize_file_tree`, the parser `_parse_file_tree_spec`, the mapping
template (`lo-<loader>` field classes + select option + collectSpec).

Verified on `Ethanliang99/ECCV_Event_Video_Depth_Estimation`. See
[hf-agent-mode-imports](hf_agent_mode_imports.md), [category-modality-contract](category_modality_contract.md),
[hf-sequence-of-masks](hf_sequence_of_masks.md).
