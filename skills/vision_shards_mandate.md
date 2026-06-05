---
name: vision-shards-mandate
description: "BenchHub is vision-oriented — when an HF vision dataset ships images in WebDataset shards, always import the shards AND any parquet metadata, merged into the same Dataset."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

For any HF vision dataset that ships images in WebDataset tar shards
(under `shards/`, `wds/`, or matched by `configs: → data_files:
"*.tar"` in the README frontmatter), the import MUST include the
shard bytes. If a parquet/arrow table sits alongside, **merge it
into the same Dataset** — the shards carry the actual pixels, the
parquet typically carries captions / labels / split metadata, and
either alone is incomplete.

**Why:** User on 2026-05-26 — "we are vision oriented benchmarking
site. we must have vision asset" — and immediately after my
shards-only re-import of facebook/ego-1k: "facebook__ego-1k has
only the shards data. it should have it and the parquet data."
Importing one half of a shards+parquet dataset gives us either
pixel-less metadata or unlabeled pixels; both defeat the point of
cataloguing the dataset on a vision benchmark.

**How to apply:**

- At step 1 of [agent-mode-mandate](agent_mode_mandate.md) (reading the README), always
  ask "are there shards?" and "is there a parquet alongside?".
  Frontmatter `configs:` / `data_files:` glob patterns are
  authoritative — `"*.tar"` means shards, `"*.parquet"` means
  parquet, both = merge.
- Identify the join key between parquet rows and shard sample keys.
  WebDataset convention is `__key__` in the parquet column matches
  the shard tar entry stem; if the dataset diverges from this,
  the README usually says so.
- Merge into a single typed-staging dir: one folder per modality
  (shard kinds: image/depth/etc.), one folder per parquet column
  (kinds: text/json/scalar/label), and one set of sample IDs
  across all of them.
- If a dataset legitimately has ONLY parquet (e.g. classification
  with embedded image bytes) or ONLY shards (e.g. raw video
  collections) — that's fine. The rule is: when both exist, take
  both.
