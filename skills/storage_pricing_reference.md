---
name: storage-pricing-reference
description: "Per-GB-month price points for the storage providers the user has used or is comparing against, for BenchHub catalog sizing decisions."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

Per-GB-month pricing the user is sizing BenchHub against:

| Provider              | Approx $/GB-month | Notes |
|-----------------------|-------------------|-------|
| **Fly.io volumes**    | ~$0.15            | What the user paid before self-hosting. Persistent SSD volumes attached to a Fly machine. Source: user recollection. |
| **Backblaze B2**      | **~$0.007** (verified 2026-05-26) | $6.95/TB-month per Backblaze's pricing page. First 10 GB free. Egress free up to 3× monthly stored (so 200 GB stored = 600 GB free download/month); above that $0.01/GB. Class D API calls cost $0.004 per 10k. ~21× cheaper than Fly. |
| Self-hosted (current) | effectively 0     | The 200 GB target on the home Ubuntu box at runbenchhub.com sits on already-paid-for disk. The cost is electricity + the time-cost of a disk replacement when one dies. See [deploy-runbook](../docs/SELFHOST_RUNBOOK.md). |

Rough math for the 200 GB catalog target:
- Fly: 200 × $0.15 = **$30/month** (Fly is dead — archive/fly/ — but it
  frames why moving off was worth it).
- B2: 200 × $0.007 = **~$1.40/month**. The free-egress envelope
  (3× stored bytes/month) means restoring a full 200 GB catalog
  in a recovery scenario stays free. Useful tier if we ever want
  off-box cold backup of `~/.dtofbenchmarking/uploads/`.
- Self-hosted: ~free at the margin.

When pricing decisions come up:
- For HOT serving (the catalog the site renders), stay on the home
  box — latency + zero recurring cost win.
- For COLD backup of dataset bytes, B2 is the cheap option. The
  daily DB backup (`ops/benchhub_db_backup.py`) covers metadata
  but doesn't ship the `uploads/` blob — at 60+ GB it's worth
  considering a periodic B2 sync if storage growth continues.
