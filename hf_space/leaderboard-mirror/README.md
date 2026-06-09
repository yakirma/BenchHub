---
title: BenchHub Leaderboards (Mirror)
emoji: 🏆
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
hf_oauth: false
---

# BenchHub Leaderboards — read-only mirror

A read-only mirror of **public** leaderboard standings from
**[runbenchhub.com](https://runbenchhub.com)**, the source of truth.

- **View** results here; **submit** on BenchHub — every "Submit" button links
  back to the leaderboard's submission page on runbenchhub.com.
- This Space contains **no** ground-truth samples, **no** predictions, and
  **no** submission UI. It reads a derived standings dataset
  (`HF_RESULTS_REPO`, default `benchhub/leaderboards`) that BenchHub publishes.

To point at a different standings dataset repo, set the `HF_RESULTS_REPO` Space
variable. The dataset repo is public, so the Space needs no token.
