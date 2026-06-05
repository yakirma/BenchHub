---
name: agent-mode-mandate
description: "Every HF dataset import MUST be done in agent mode — read the dataset card, list the actual file tree, probe a real shard/parquet, then write the importer. No guessing from convention."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

Every HF dataset import is a research task, not a templating task.

**Why:** On 2026-05-26 I imported `facebook/ego-1k` from only the
parquet table even though the dataset card's "Dataset Structure"
section described 12-camera images under `shards/`. The user flagged
that miss explicitly: "you missed the images that was under shards
directory. it was explained under the dataset card tab. you should
run in agent mode and read this card every time to understand the
dataset structure." Then when I built `/tmp/import_shards.py` for the
follow-up, I hardcoded shard URLs and WebDataset regex patterns from
"convention" instead of probing the actual tree — and 2 of 3
re-imports failed because the conventions didn't hold. The user
called that out too: "did you operate in agent mode, for the dataset
importer? always act in agent mode and try understanding the actual
data structure, from description and actual file layouts."

**How to apply:** Before writing any import code for a new HF repo,
do all of:

1. `WebFetch` (or `curl`) the README and read it end-to-end — at
   minimum the YAML frontmatter (`configs:`, `data_files:`,
   `task_categories:`) and any `## Dataset Structure` / `## Loading`
   section.
2. `curl /api/datasets/<repo>/tree/main?recursive=1` (or use the
   helper in `import_hf_agent.py`) and reconcile what's actually on
   disk against what the card claims.
3. Open ONE concrete artifact end-to-end — list the entries in the
   first tar shard, or `pq.read_metadata` the first parquet — and
   derive the per-entry regex / column-mapping from the real names,
   not from "WebDataset convention says X."
4. Only after all 3 steps, write the per-repo config.

The full lessons live in [hf-agent-mode-imports](hf_agent_mode_imports.md) under the "What
'agent mode' means here" section — that's the durable expansion.
This memory file is the rule; the skill file is the playbook.

See also [vision-shards-mandate](vision_shards_mandate.md) — for vision datasets the shards
question is mandatory at step 1.
