# Sandbox runner image for user-supplied metric / visualization code.
#
# Two ways to use this image:
#
# 1. CLI (one-shot, used by the docker-subprocess path in
#    metric_engine.evaluate_in_sandbox locally):
#       docker run --rm --network=none --read-only --tmpfs /tmp:size=64m
#           --memory=512m --cpus=1 -i benchhub-runner python /app/harness.py
#       (writes JSON job on stdin, reads result on stdout)
#
# 2. HTTP server (used by the runner-as-separate-Fly-app deploy — see
#    runner/fly.toml). This is the default CMD: gunicorn serves
#    runner/server.py which wraps harness.run_job. Boundary is the
#    separate VM (no DB, no /data); per-request interpreter freshness
#    comes from --max-requests=100 worker recycling.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Runtime libs for matplotlib (numpy/scipy ship manylinux wheels). Install
# all at once so apt's metadata cache layer is small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        libfreetype6 \
        libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

# Pre-install the canonical scientific-python stack so user code can import
# them without needing internet (the container runs with --network=none).
# Flask + gunicorn for the HTTP server mode.
RUN pip install \
        'numpy>=1.26' \
        'scipy>=1.11' \
        'matplotlib>=3.8' \
        'Pillow>=10' \
        'Flask>=3,<4' \
        'gunicorn>=22'

WORKDIR /app
COPY harness.py /app/harness.py
COPY server.py /app/server.py

# matplotlib's font cache wants a writable home; --read-only at runtime
# blocks /home/runner, so point HOME at /tmp (which we'll mount tmpfs).
RUN useradd --create-home --uid 1000 runner \
 && chown -R runner:runner /app
USER runner
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.matplotlib \
    PORT=8080

EXPOSE 8080

# Default = HTTP server. CLI mode is `... python /app/harness.py`.
# --max-requests recycles each worker after 100 jobs so leaked state
# (sys.modules pollution, dangling threads) can't accumulate forever.
# --timeout 60 matches metric_engine.evaluate_in_sandbox's caller-side
# timeout default; gunicorn kills the worker if a single job takes longer.
CMD ["gunicorn", \
     "-b", "0.0.0.0:8080", \
     "--workers", "2", \
     "--max-requests", "100", \
     "--max-requests-jitter", "20", \
     "--timeout", "60", \
     "server:app"]
