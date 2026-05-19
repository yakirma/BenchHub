# Archived: Fly.io deployment

Production was on [Fly.io](https://fly.io) until early 2026, when it was
migrated to a self-hosted Ubuntu box at `runbenchhub.com` and the Fly app
was destroyed. The artifacts are kept here so a future Fly redeploy can
be reconstructed from this directory without git archaeology.

## Contents

| File | What |
|---|---|
| `DEPLOY.md` | End-to-end Fly + Cloudflare Access runbook. |
| `Dockerfile` | BenchHub web image (gunicorn + celery + redis in one container, see `start.sh`). |
| `.dockerignore` | Build context exclusions for the image above. |
| `fly.toml` | Fly app config for the main service. |
| `start.sh` | Container entrypoint — boots redis, celery, then gunicorn. |
| `entrypoint.sh` | Root → `app` user drop wrapper. |
| `runner/fly.toml` | Fly app config for the sandbox runner (the runner code itself stays in `/runner` — local tests still use it). |

## Current deployment

See [`../../docs/SELFHOST_RUNBOOK.md`](../../docs/SELFHOST_RUNBOOK.md) for
the live operational runbook on the home box.
