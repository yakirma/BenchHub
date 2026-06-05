# BenchHub

BenchHub is an open-source benchmarking platform: pick a dataset, define
metrics in Python, upload predictions, and see how your model ranks. Live
at **https://runbenchhub.com**.

Originally built as a private dTOF SPAD pipeline benchmarking tool, then
generalized into a public, multi-tenant web app.

## Features

- **Passwordless sign-in** — GitHub, Google, or a one-time email code.
- **Strict typed contract** — every field has a *kind* (image, mask, depth,
  audio, label, bboxes, scalar, json, **sequence/video**, …) shared by the
  dataset, the `benchhub-client`, and the metric engine (`benchhub/types.py`).
- **Self-service HuggingFace import** — one button: the **tabular** importer
  (Croissant/parquet, inferred + editable) falls back to a **file-tree
  mapper** for repos of paired files / packed archives / video clips. The
  mapper has a "describe the structure" role wizard, loaders for
  file/npz/json/csv/parquet/hdf5/zip/tar/gz/token/sequence, a decode preview,
  variant fan-out, and draft autosave.
- **Two-tier storage** — datasets cache as a cheap preview tier; each
  leaderboard materializes a chosen sample subset at full resolution.
- **Datasets and leaderboards are global**; per-row visibility
  (`public` / `unlisted` / `private`) on datasets, leaderboards, and
  metric/visualization library entries.
- **User-defined metrics & visualizations in Python** — typed signatures,
  per-sample + aggregated, pooling, dependency chaining. All user code runs
  in a **hardened, network-isolated, read-only sandbox container** (one
  short-lived sandbox per job) — never in-process on the server.
- **User-registered data types** — declare a new `kind` (its storage + a
  `visualize(blob, params)` that runs in the sandbox) via
  `client.create_datatype(...)`; it joins the global kind namespace.
- **`benchhub-client` + dev kit** — `iter_samples` (decoded typed inputs
  incl. iterable video clips) → predict → submit; programmatic dataset
  creation; `client.create_metric` / `create_visualization` /
  `create_datatype`; and `benchhub.author.test_metric` / `test_visualization`
  to iterate locally before uploading.
- **Asynchronous processing** with Celery (Redis broker).
- **Split-bucket quotas** — 50 GB public + 10 GB private per user by default.
- **API tokens** (`/settings/api_tokens`), account deletion with cascading
  cleanup, public landing (`/`), catalog (`/leaderboards`, `/datasets`),
  profiles (`/u/<id>`).

## Documentation

Full user docs live in-app at **[`/docs`](https://runbenchhub.com/docs)**
(templates under `templates/docs/`): overview, core concepts, importing data,
data types, leaderboards, writing metrics & visualizations, submitting
predictions, the API/client reference, and step-by-step tutorials. A
high-level pipeline diagram is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (editable drawio source under
[`docs/diagrams/`](docs/diagrams/)). Architecture/dev notes are in
[`CLAUDE.md`](CLAUDE.md); the session-by-session dev history is under
[`docs/`](docs/) (e.g. `SESSION_NOTES_2026-05.md`).

Feature requests + bug reports → **[GitHub issues](https://github.com/yakirma/BenchHub/issues)**.

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
pytest tests/
```

~1000+ tests. (Run `pytest tests/`, not bare `pytest`, so it skips the
ad-hoc root-level `test_chain*.py` experiments.)

## Datasets & the typed contract

Datasets are typed: a directory with a `manifest.json` declaring
`fields[]` (`{name, kind, role, params}`) plus one folder per field holding
`<sample>.<ext>`. You rarely build this by hand — the **HuggingFace
importers** and the client's **`BHDatasetCreator`** produce it for you. The
kinds and the import flows are documented in-app at
[`/docs`](https://runbenchhub.com/docs) (Data Types, Importing Data). The
legacy folder-name-prefix ZIP path has been removed.

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
