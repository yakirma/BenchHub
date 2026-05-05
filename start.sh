#!/bin/sh
# Start both the Celery worker and gunicorn inside the same Fly VM.
#
# Why one VM: Fly volumes are 1:1 with machines, but the worker (writes
# CustomField rows, reads/writes /data/uploads) and the web process (reads
# the same SQLite DB + serves uploads) MUST share a filesystem. Two process
# groups in fly.toml would each get their own volume → out-of-sync state.
# Single machine, single volume, both processes is the right shape for the
# Cloudflare-Access-gated single-user deploy.

set -eu

# Trap SIGTERM/SIGINT and forward to both children for clean Fly shutdown.
shutdown() {
  echo "[start.sh] shutting down…"
  if [ -n "${WORKER_PID:-}" ]; then kill -TERM "$WORKER_PID" 2>/dev/null || true; fi
  if [ -n "${WEB_PID:-}"    ]; then kill -TERM "$WEB_PID"    2>/dev/null || true; fi
  wait
  exit 0
}
trap shutdown TERM INT

celery -A app.celery worker --loglevel=info --concurrency=1 &
WORKER_PID=$!

gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 120 app:app &
WEB_PID=$!

# Exit when either child exits — Fly will restart the whole VM, which is
# fine because both processes are intended to live together.
wait -n "$WORKER_PID" "$WEB_PID"
EXIT=$?
echo "[start.sh] one process exited with $EXIT — terminating the other"
shutdown
