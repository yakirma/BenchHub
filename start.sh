#!/bin/bash
# Start redis-server + Celery worker + gunicorn inside the same Fly VM.
#
# Why one VM: Fly volumes are 1:1 with machines, but the worker (writes
# CustomField rows, reads/writes /data/uploads) and the web process (reads
# the same SQLite DB + serves uploads) MUST share a filesystem. Two process
# groups in fly.toml would each get their own volume → out-of-sync state.
# Single machine, single volume, all three processes is the right shape
# for the Cloudflare-Access-gated single-user deploy.
#
# Why local redis: previously the broker was Upstash's free tier (500K
# req/month), and the celery worker kept exhausting the quota and
# crash-looping. Loopback redis on the same VM has no quota, no network
# variance, and the worker is the only consumer — so this is strictly
# simpler.
#
# Resilience: gunicorn is the user-facing process. If celery dies (e.g.
# a bad task crashes the worker), we keep gunicorn alive so the web UI
# stays up — submissions can't evaluate, but the user can still browse,
# manage settings, etc. Re-arm celery in a loop with an exponential
# backoff so it reattaches automatically.
#
# NOTE: must be bash, not sh — we use `wait -n` which is a bashism. The
# python:3.13-slim image ships /bin/bash, so this is fine.

set -u

shutdown() {
  echo "[start.sh] shutting down…"
  if [ -n "${WORKER_PID:-}" ]; then kill -TERM "$WORKER_PID" 2>/dev/null || true; fi
  if [ -n "${WEB_PID:-}"    ]; then kill -TERM "$WEB_PID"    2>/dev/null || true; fi
  if [ -n "${REDIS_PID:-}"  ]; then kill -TERM "$REDIS_PID"  2>/dev/null || true; fi
  wait
  exit 0
}
trap shutdown TERM INT

# Redis: loopback-only, no persistence (it's purely a Celery broker —
# tasks in flight at restart are unfortunate but acceptable, same shape
# as Upstash would have given us). --daemonize no keeps it in foreground
# so we can manage its PID directly.
echo "[start.sh] starting redis-server (loopback, no persistence)"
redis-server \
    --bind 127.0.0.1 \
    --port 6379 \
    --protected-mode yes \
    --daemonize no \
    --save "" \
    --appendonly no \
    --loglevel notice &
REDIS_PID=$!

# Wait for redis to be ready before celery tries to connect — redis-cli
# is shipped with redis-server, so we use it instead of nc.
for i in {1..30}; do
  if redis-cli -h 127.0.0.1 -p 6379 PING 2>/dev/null | grep -q PONG; then
    echo "[start.sh] redis-server up after ${i} attempts"
    break
  fi
  sleep 0.2
done

# Celery in a self-restarting subshell: if it crashes for any reason,
# the subshell sleeps and tries again. Backoff is bounded.
(
  backoff=10
  while true; do
    echo "[start.sh] starting celery worker (backoff=${backoff}s)"
    # --without-mingle / --without-gossip / --without-heartbeat: skip
    # the multi-worker coordination chatter (we have one worker on one
    # machine; mingling is N redis polls per second for nothing).
    celery -A app.celery worker --loglevel=info --concurrency=1 \
        --without-mingle --without-gossip --without-heartbeat || \
      echo "[start.sh] celery exited; will retry after ${backoff}s"
    sleep "$backoff"
    if [ "$backoff" -lt 600 ]; then backoff=$((backoff * 2)); fi
  done
) &
WORKER_PID=$!

# Gunicorn — the canonical user-facing process. Its exit ends the VM.
# --timeout 360: a few admin endpoints (SOTA notebook generation
# in particular) call Claude with max_tokens=16k, which can run 60-180 s.
# Default 30 s would SIGKILL the worker mid-call and the user gets a
# generic 502. The two workers mean other requests aren't blocked
# while one's tied up on the slow path.
gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 360 app:app &
WEB_PID=$!

# Block on gunicorn ONLY. If celery's retry loop dies, ignore — web stays up.
wait "$WEB_PID"
EXIT=$?
echo "[start.sh] gunicorn exited with $EXIT"
shutdown
