#!/bin/bash
# Prune the Hugging Face `datasets` cache so it never silently rebuilds to TBs.
# BenchHub serves from ~/.dtofbenchmarking/uploads (the preview tier), so the
# datasets/ Arrow + downloads cache is transient scratch from load_dataset()
# during imports — orphaned once the preview is extracted. Re-derived on demand.
#
# Safe by construction:
#   - the raw downloads/ scratch is always disposable -> cleared every run;
#   - processed Arrow entries are removed only when untouched for >2 days, so a
#     long-running import (mtime keeps updating while it writes) is never hit;
#   - the model `hub/` cache is left alone (weights are expensive + reused).
set -u
D="$HOME/.cache/huggingface/datasets"
[ -d "$D" ] || exit 0
before=$(df -h / | awk 'NR==2{print $4}')
rm -rf "$D/downloads"/* 2>/dev/null || true
find "$D" -mindepth 1 -maxdepth 1 -type d ! -name downloads -mtime +1 -exec rm -rf {} + 2>/dev/null || true
echo "[$(date '+%F %T')] pruned HF datasets cache (free: $before -> $(df -h / | awk 'NR==2{print $4}'))"
