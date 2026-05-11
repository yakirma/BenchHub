# BenchHub container — runs gunicorn AND celery inside one VM via start.sh.
# Why not two Fly process groups: Fly volumes are 1:1 with machines, but the
# web + worker need to share /data (SQLite DB + uploads). Single VM keeps
# them on one volume.

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BENCHHUB_DATA_DIR=/data \
    PORT=8080

# matplotlib needs libgomp1 + libfreetype/libpng (numpy/scipy ship manylinux
# wheels, but matplotlib's font cache wants these at runtime).
# redis-server is the Celery broker — we run it loopback-only on this same
# VM. Upstash's free tier kept blowing through its 500K monthly request cap
# and crash-looping the worker; an in-VM Redis sidesteps the external
# dependency entirely.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        libfreetype6 \
        libpng16-16 \
        redis-server \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Non-root user; data dir bind-mounted at /data via Fly volume.
# The container starts as root via the entrypoint script below so we can
# repair ownership on the persistent Fly volume (an older revision that
# briefly ran as root left /data/uploads/submissions/ owned by root, and
# subsequent runs as `app` silently failed to extract submission ZIPs).
# entrypoint then drops to `app` via `su` for the actual processes.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data \
 && chmod +x /app/start.sh /app/entrypoint.sh \
 && chown -R app:app /app \
 && chown app:app /data

EXPOSE 8080

# entrypoint.sh runs as root to fix any stale ownership on the Fly volume,
# then drops to `app` via `su` to exec start.sh (which boots redis +
# celery + gunicorn).
CMD ["./entrypoint.sh"]
