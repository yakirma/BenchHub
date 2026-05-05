# BenchHub container — runs the same image as both web (gunicorn) and worker
# (celery). The process is selected by Fly's [processes] table in fly.toml,
# which becomes the container's CMD.

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
 && chown -R app:app /app /data
USER app

EXPOSE 8080

# Default CMD is the web process; Fly overrides this for the worker process.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
