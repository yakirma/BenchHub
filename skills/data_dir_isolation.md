---
name: data-dir-isolation
description: Never run a python -c that imports app without setting BENCHHUB_DATA_DIR first — app.py binds SQLAlchemy to the prod database path on import.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

When investigating bugs locally on the BenchHub box, **never** run a
`python -c "import app; ..."` snippet that touches the DB without
first setting `BENCHHUB_DATA_DIR=/tmp/somewhere`. The module
unconditionally resolves the data dir from that env var (falling
back to `~/.dtofbenchmarking`) at import time and binds
SQLAlchemy's engine to whatever it found. Any `db.drop_all()`,
`db.create_all()`, or even a stray write you make against the
session is going at the live production DB at
`~/.dtofbenchmarking/database.db`.

**Why:** I destroyed the production dataset / leaderboard /
CustomField / sample tables on 2026-05-24 by running debug
one-liners that called `db.drop_all(); db.create_all()` to repro a
pred-fields test failure. Conftest fixtures redirect
`BENCHHUB_DATA_DIR` to a tempdir before importing app, so pytest
runs are safe — ad-hoc `python -c` invocations completely bypass
that protection.

**How to apply:**
- For one-off DB poking, use pytest (which already isolates) or
  wrap the invocation in `env BENCHHUB_DATA_DIR=$(mktemp -d) python -c '...'`.
- Daily snapshots now run via the user systemd timer
  `benchhub-db-backup.timer` and keep 14 days in
  `~/.dtofbenchmarking/db_backups/` — recovery procedure is in
  [deploy-runbook](../docs/SELFHOST_RUNBOOK.md) under the "DB backups" section.
- If a destructive command is unavoidable, take an extra manual
  snapshot first: `systemctl --user start benchhub-db-backup.service`.
