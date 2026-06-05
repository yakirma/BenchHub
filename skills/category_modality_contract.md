---
name: category-modality-contract
description: "A dataset's task tag implies required modalities. depth-estimation must yield a depth field, segmentation must yield a mask field, etc. Missing required modalities = broken import."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

When importing an HF dataset, the task tag or category is a contract
on what modalities must exist on disk after the import completes.

**Why:** User on 2026-05-26 — "in Ethanliang99__ECCV_Event_Video_Depth_Estimation
you missed the depth modality, it was stored in depth.npz file …
this is a depth dataset you should expect depth modality!" The
HF tree API capped the root recursive listing at ~1000 files and
truncated before reaching `train/<scene>/normal/depth.npz`. The
importer happily produced 50 rgb-only samples and the catalog
listed it as a depth-estimation dataset with no depth field. Same
class of bug as the shards/imagefolder misses but caught by a
different signal — the dataset's stated purpose.

**How to apply:**

1. At step 1 of [agent-mode-mandate](agent_mode_mandate.md) (read the README), record
   the `task_categories:` and `tags:` from the YAML frontmatter.
2. Map them to required output kinds — see the table in
   [hf-agent-mode-imports](hf_agent_mode_imports.md) under "Category dictates required
   modalities."
3. After the file-tree probe at step 2, if the surface listing
   doesn't show a path matching the required kind (look for
   `depth.npz`, `*_depth.*`, `mask*.png`, `segmentation*.png`,
   `bbox*.json`, etc.), **recurse deeper** — drill into each
   scene/sample subdir with its own `tree/main/<subpath>?recursive=1`
   call. HF's recursive listing truncates at ~1000 files; don't
   trust it as exhaustive on big repos.
4. After the import staging completes, assert that the manifest
   includes at least one field of each required kind. Raise if not
   — phantom-import-but-wrong-shape is worse than a clean failure.
5. When scanning the existing catalog, treat any
   `category in {Depth Estimation, Segmentation, ...}` Dataset
   with zero matching-kind fields as broken — re-import or mark.

See also [vision-shards-mandate](vision_shards_mandate.md) (vision needs the actual pixels)
and [agent-mode-mandate](agent_mode_mandate.md) (read + list + probe before coding).
