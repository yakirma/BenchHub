# BenchHub

BenchHub is an open-source benchmarking platform: pick a dataset, define
metrics in Python, upload predictions, and see how your model ranks. Live
at **https://runbenchhub.com**.

Originally built as a private dTOF SPAD pipeline benchmarking tool, then
generalized into a public, multi-tenant web app.

## Features

- **OAuth sign-in (GitHub)** — no passwords; one-click account creation.
- **Datasets and leaderboards** are global — no project namespace.
- **Per-row visibility** (`public` / `unlisted` / `private`) on datasets,
  leaderboards, and metric/visualization library entries.
- **HuggingFace import**: pull a structured HF dataset repo as a one-click
  alternative to a ZIP upload (see `scripts/seed_nyu_v2_curated.py` for
  an example workflow).
- **User-defined metrics in Python** — bring your own scoring code; the
  metric engine resolves dependencies and runs them per-sample or
  aggregated. Sandbox-isolated when `BENCHHUB_SANDBOX_METRICS=1`.
- **Asynchronous processing** with Celery (Redis broker).
- **Per-user quotas**: 200 MB storage, 5 datasets, 50 submissions / 24h
  by default. Free-tier safe to expose to the open internet.
- **API tokens** for programmatic uploads (`/settings/api_tokens`).
- **Account deletion** (GDPR right-to-be-forgotten) with cascading cleanup.
- **Public landing page** at `/`, `/leaderboards` for browsing the catalog,
  `/u/<id>` for public profile pages.

## Prerequisites

- Python 3.10+
- Redis (broker + result backend, default port 6379)

## Installation

```bash
git clone <repository-url>
cd BenchHub
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

Three terminals:

```bash
# 1. Redis
redis-server

# 2. Celery worker
celery -A app.celery worker --loglevel=info

# 3. Flask app
python app.py
```

Then open `http://localhost:6060`.

Data lives outside the repo at `~/.dtofbenchmarking/` (database + uploads).
Override with `BENCHHUB_DATA_DIR=/some/path`.

## Tests

```bash
pytest
```

429 tests, ~3-4 seconds. Coverage gate is configured in `pytest.ini`.

## Dataset / submission ZIP convention

Folders are auto-detected by prefix:

| Prefix          | Type      | Files                                  |
| --------------- | --------- | -------------------------------------- |
| `metric_`       | metric    | `<sample>.txt` containing a float      |
| `hist_` / `raw_histogram` / `hist` | histogram | `<sample>.npz` (`bins`, `counts`) |
| `raw_`          | depth/map | `<sample>_<W>x<H>.npz`                 |
| (anything else) | image / scalar / json / text | by file extension          |

`git_info.json` (or `git.info`) at the ZIP root attaches commit metadata
to the resulting dataset/submission row.

## DLP-safe code uploads

Some networks block `.py` uploads. The metric editor encodes user code
as `BASE64:<...>` client-side; the server decodes. Standalone helpers:

- `scripts/obfuscator.html` — portable browser tool
- `scripts/obfuscator_gui.py` — Tkinter GUI

## Deployment

The production app is self-hosted on a home Ubuntu 24.04 box (RTX 5090,
128 GB RAM, 8 TB) reachable at https://runbenchhub.com. gunicorn + celery
+ redis run directly under systemd; nginx + certbot terminate TLS; the
domain is on Cloudflare in DNS-only mode (no proxy) with `ddclient`
keeping the A record pointed at the home WAN IP.

**Operational runbook: [`docs/SELFHOST_RUNBOOK.md`](docs/SELFHOST_RUNBOOK.md)**
— code-push procedure, `.env` keys, log tailing, DDNS, TLS renewal,
rollback, and the breakages we've already hit.

Fly.io is deprecated: the app was destroyed after the cutover to the home
box. The Fly artifacts (`fly.toml`, `Dockerfile`, `DEPLOY.md`, …) are
archived under [`archive/fly/`](archive/fly/) for the case where a future
Fly redeploy needs to be reconstructed.

## License

(Choose and add a license file — repository currently has no LICENSE.)
