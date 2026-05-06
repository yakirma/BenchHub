# BenchHub developer tasks.
#
# Convenience wrappers around pytest / coverage / dev-server. Uses the
# in-tree .venv if present, otherwise falls back to system python.

PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PYTEST := $(PY) -m pytest

# Coverage threshold — see .coveragerc for source/exclusions.
# Current baseline is 54.0% across app.py + metric_engine.py + tasks.py.
# Bump this when you raise coverage; do not lower it.
COV_FAIL_UNDER := 50

.PHONY: help test test-fast test-docker cov cov-html clean dev worker redis-check runner-image

help:
	@echo "Targets:"
	@echo "  test         Run the full test suite (skips docker-marked tests)."
	@echo "  test-fast    Stop on first failure, no captured output."
	@echo "  test-docker  Run the docker-marked integration tests (requires docker)."
	@echo "  cov          Run with coverage; fails if below $(COV_FAIL_UNDER)%."
	@echo "  cov-html     Coverage report as browsable HTML in .coverage_html/."
	@echo "  runner-image Build the sandbox runner image (benchhub-runner:local)."
	@echo "  dev          Start the Flask app on :6060 (assumes Redis is running)."
	@echo "  worker       Start a Celery worker (logs to stdout)."
	@echo "  redis-check  Verify Redis is reachable on the default port."
	@echo "  clean        Remove pytest/coverage caches."

test:
	$(PYTEST)

test-fast:
	$(PYTEST) -x -s

test-docker:
	$(PYTEST) -m docker --no-cov

cov:
	$(PYTEST) --cov --cov-report=term --cov-fail-under=$(COV_FAIL_UNDER)

runner-image:
	docker build -t benchhub-runner:local runner/

cov-html:
	$(PYTEST) --cov --cov-report=html
	@echo "Open .coverage_html/index.html"

dev:
	$(PY) app.py

worker:
	$(PY) -m celery -A app.celery worker --loglevel=info

redis-check:
	@redis-cli ping > /dev/null 2>&1 && echo "Redis OK" || (echo "Redis NOT reachable on default port — start it with 'redis-server'."; exit 1)

clean:
	rm -rf .pytest_cache .coverage .coverage_html
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
