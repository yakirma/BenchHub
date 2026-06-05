---
name: sampling-policy-no-cap
description: HF imports take the full eval split. No 50-sample cap. Sample_cap=-1 across every importer.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

Every HF import takes the full eval split — no per-dataset row cap.

**Why:** Until 2026-05-26 every importer (the bulk LLM loop, the
per-repo agent-mode scripts, the URL crawler) capped at 50 samples.
That made the dataset-view template print misleading lines like
"Imported 50 of 0 rows ... via head sampling (seed )" and — more
importantly — leaderboards ranked submissions on a 50-sample
holdout that didn't represent the underlying dataset. User
explicitly said: "for the existing datasets re-import with all
samples (no sampling). add it to the skill file."

**How to apply:**

1. In any HF import path, set `sample_cap=-1` (or remove the cap
   entirely). The materializer treats -1 as unbounded.
2. In per-repo agent-mode scripts, drop the `samples[:MAX_SAMPLES]`
   slice or set `MAX_SAMPLES` to a number larger than any natural
   eval split (e.g. 10_000).
3. Before downloading, probe the split row count and log it. If
   the row count is >500k for what should be a benchmark eval
   split, surface the size and ask before proceeding — that's
   usually a sign the dataset is weirdly structured (only `train`
   split exposed, or the eval split was incorrectly tagged).
4. Always record `total_rows_in_split` in `source_metadata` so the
   dataset-view template can show "Imported N of M" cleanly. With
   no cap, N == M and the template renders "Imported all M rows
   from the X split."

Disk-space watch: with no cap, the 200 GB target ceiling is hit
much faster. The bulk loop should keep its used-bytes check and
stop when over budget.

See [hf-agent-mode-imports](hf_agent_mode_imports.md) for the full importer playbook;
[vision-shards-mandate](vision_shards_mandate.md) for the shards-not-optional rule that
intersects with this when a vision dataset has parquet+shards.
