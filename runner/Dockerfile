# Sandbox runner image for user-supplied metric / visualization code.
#
# This image is designed to be run with:
#   docker run --rm
#       --network=none
#       --read-only
#       --tmpfs /tmp:size=64m
#       --memory=512m --cpus=1
#       -i benchhub-runner
#
# stdin: a single JSON job (see runner/harness.py for the schema).
# stdout: a single JSON line of results.
#
# The container has no network, no writable filesystem outside /tmp, and a
# small memory cap. Caller is responsible for setting a wall-clock timeout
# (subprocess.run(..., timeout=60)).

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
RUN pip install \
        'numpy>=1.26' \
        'scipy>=1.11' \
        'matplotlib>=3.8' \
        'Pillow>=10'

WORKDIR /app
COPY harness.py /app/harness.py

# matplotlib's font cache wants a writable home; --read-only at runtime
# blocks /home/runner, so point HOME at /tmp (which we'll mount tmpfs).
RUN useradd --create-home --uid 1000 runner \
 && chown -R runner:runner /app
USER runner
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.matplotlib

ENTRYPOINT ["python", "/app/harness.py"]
