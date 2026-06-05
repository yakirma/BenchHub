---
name: hf-sequence-of-masks
description: "HF datasets whose features include a Sequence(Image) of per-object/per-layer masks (e.g. ADE20K `instances`) need compositing into one instance-id mask, not dropping — the agent importer silently drops Sequence-of-Image fields."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5c7cb6c2-c6d2-4a9f-9ba4-8a84e62b77b0
---

When an HF dataset exposes a `Sequence(feature=Image(...))` column — a
variable-length **stack of masks** — the agent importer tends to keep
only one element (or drop the column entirely). ADE20K (`1aurent/ADE20K`,
ds 92) is the canonical case: `segmentations` (Sequence, len 1) became
`mask`, but `instances` (Sequence, len 7..31 — one binary layer per
object, values {0,128,255}) was **dropped**.

**Why:** the user expects every source field represented. A
Sequence-of-Image instance field is real GT, not noise.

**How to apply:** composite the per-object binary layers into ONE
instance-id map — `out[layer==255] = k` for the k-th layer (later layers
win on overlap) — and store it as a single `instance` mask field, the
same way `mask` is stored: a 16-bit `I;16` `.classid.png` (instance ids)
+ palette `.jpg` from `benchhub.preview.mask_preview`, one CustomField
(data_type='mask', value_text points at the `.jpg`) per sample, plus a
`DatasetField(kind='mask', role='gt')`. The backfill template is
`scripts/add_ade20k_instance_field.py`. Stream the same split in order —
row i aligns with sample `s_{i:05d}` (verify via the stored `filename`
CF). Run scripts with `PYTHONPATH=<repo root>` so `import app` resolves.
See [category-modality-contract](category_modality_contract.md) and [hf-agent-mode-imports](hf_agent_mode_imports.md).
