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
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        libfreetype6 \
        libpng16-16 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Non-root user; data dir bind-mounted at /data via Fly volume.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data \
 && chmod +x /app/start.sh \
 && chown -R app:app /app /data
USER app

EXPOSE 8080

# start.sh runs celery + gunicorn together with proper SIGTERM forwarding.
CMD ["./start.sh"]
