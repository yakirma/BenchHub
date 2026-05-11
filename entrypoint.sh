#!/bin/bash
# Root-stage entrypoint: repair Fly-volume ownership, then drop to `app`
# and exec start.sh.
#
# Why this exists: an older revision of the container briefly ran as
# root, which created /data/uploads/submissions/ owned by root. When we
# later switched to USER app in the Dockerfile, processes running as
# uid 1000 couldn't os.makedirs() under that directory — submission
# uploads silently failed inside an `except Exception` and the row sat
# in "Queued" with no extracted folder. Persistent volume ownership
# survives image rebuilds, so the only sustainable fix is a startup
# `chown` that runs as root.
#
# This is also the safety net for any future ownership drift (a manual
# `flyctl ssh console` that writes to /data as root, etc).
set -eu

if [ "$(id -u)" = "0" ]; then
  echo "[entrypoint] repairing /data ownership (app:app)"
  chown -R app:app /data 2>/dev/null || true
  echo "[entrypoint] dropping to 'app' and exec start.sh"
  # `su -c` doesn't replace the shell, but with `exec` inside the -c
  # string the child shell becomes start.sh. Signals (SIGTERM from Fly
  # on stop) reach start.sh, which has its own trap.
  exec su app -c 'exec /app/start.sh'
else
  # Container already started as non-root (e.g. during `flyctl ssh
  # console -C ...`) — just run start.sh directly.
  exec /app/start.sh
fi
