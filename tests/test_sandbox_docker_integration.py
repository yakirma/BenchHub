"""Real-Docker integration test for the sandbox runner.

Skipped by default (`make test`) because it (a) needs `docker` on PATH and
a working daemon, and (b) is slow — the first run builds the image. Opt
in via `make test-docker` (or `pytest -m docker`).

Proves the wrapper, the harness, and the Dockerfile actually agree on
the JSON contract end-to-end:
- happy path: simple metric, batch of contexts, all succeed
- numpy is available inside the container
- --network=none really blocks outbound connections
- --read-only really blocks writes outside /tmp
"""
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from metric_engine import evaluate_in_sandbox

# Every test in this module is part of the slow docker-only suite.
pytestmark = pytest.mark.docker


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_DOCKERFILE = os.path.join(_REPO_ROOT, 'runner', 'Dockerfile')
_IMAGE_TAG = 'benchhub-runner:integ-test'


def _docker_available():
    """`docker version` succeeds → daemon reachable. Just `which docker` is
    not enough; the binary exists on macOS without Docker Desktop running."""
    if shutil.which('docker') is None:
        return False
    try:
        proc = subprocess.run(
            ['docker', 'version'], capture_output=True, timeout=10,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Fail fast in collect when docker isn't reachable so we don't waste minutes
# trying to build an image we can't run.
if not _docker_available():
    pytest.skip(
        "docker not available — skipping sandbox integration tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def runner_image():
    """Build the sandbox image once per pytest session. Re-runs are cheap
    because Docker's layer cache short-circuits when nothing has changed
    in runner/."""
    # Build from the repo root (context) with the runner Dockerfile so the
    # vendored benchhub package is in scope; a repo-root .dockerignore keeps
    # the context tiny.
    proc = subprocess.run(
        ['docker', 'build', '-f', _DOCKERFILE, '-t', _IMAGE_TAG, _REPO_ROOT],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"docker build failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return _IMAGE_TAG


def make_metric(code):
    return SimpleNamespace(name='integ', python_code=code)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_real_container_executes_simple_metric(runner_image):
    """Tightest possible end-to-end: 3 contexts, results in matching order."""
    results = evaluate_in_sandbox(
        make_metric("def m(x):\n    return x * 2\n"),
        contexts=[{'x': 1}, {'x': 2}, {'x': 3}],
        arg_mappings_json='{"x": "x"}',
        image=runner_image,
    )
    assert results == [(2.0, None), (4.0, None), (6.0, None)]


def test_real_container_can_use_numpy(runner_image):
    """numpy is pre-installed in the image and injected as `np` via harness."""
    results = evaluate_in_sandbox(
        make_metric("def m(arr):\n    return float(np.mean(arr))\n"),
        contexts=[{'arr': [1, 2, 3, 4]}, {'arr': [10, 20]}],
        arg_mappings_json='{"arr": "arr"}',
        image=runner_image,
    )
    assert results == [(2.5, None), (15.0, None)]


def test_real_container_per_call_error_does_not_break_batch(runner_image):
    """A single bad context returns its own error; the rest of the batch
    still produces values."""
    code = (
        "def m(x):\n"
        "    if x == 0:\n"
        "        raise ValueError('zero')\n"
        "    return 100 / x\n"
    )
    results = evaluate_in_sandbox(
        make_metric(code),
        contexts=[{'x': 4}, {'x': 0}, {'x': 5}],
        arg_mappings_json='{"x": "x"}',
        image=runner_image,
    )
    assert results[0] == (25.0, None)
    assert results[1][0] is None
    assert 'ValueError' in (results[1][1] or '')
    assert results[2] == (20.0, None)


# ---------------------------------------------------------------------------
# Hardening: prove the docker flags actually do what we expect
# ---------------------------------------------------------------------------


def test_real_container_blocks_outbound_network(runner_image):
    """--network=none means even a direct socket() to a public IP fails.
    The metric tries it and the per-call error must mention an OS-level
    block (no route / network unreachable / etc)."""
    code = (
        "def m():\n"
        "    import socket\n"
        "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    s.settimeout(2)\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    return 1.0\n"
    )
    results = evaluate_in_sandbox(
        make_metric(code),
        contexts=[{}],
        arg_mappings_json='{}',
        image=runner_image,
    )
    assert results[0][0] is None
    err = results[0][1] or ''
    # OS error wording varies by kernel — match the broad shape.
    assert any(token in err for token in (
        'Network is unreachable', 'No route to host', 'OSError', 'timed out',
    )), err


def test_real_container_blocks_writes_outside_tmp(runner_image):
    """--read-only on the rootfs means writes anywhere except /tmp (which
    is a tmpfs we mount explicitly) get EROFS / Permission denied."""
    code = (
        "def m():\n"
        "    with open('/etc/benchhub_pwn', 'w') as f:\n"
        "        f.write('hi')\n"
        "    return 1.0\n"
    )
    results = evaluate_in_sandbox(
        make_metric(code),
        contexts=[{}],
        arg_mappings_json='{}',
        image=runner_image,
    )
    assert results[0][0] is None
    err = results[0][1] or ''
    assert any(token in err for token in (
        'Read-only file system', 'Permission denied', 'OSError',
    )), err


def test_real_container_allows_writes_to_tmp(runner_image):
    """The tmpfs at /tmp is intentionally writable — matplotlib's font
    cache lives there, and a metric that legitimately wants scratch
    space should work."""
    code = (
        "def m():\n"
        "    with open('/tmp/scratch.txt', 'w') as f:\n"
        "        f.write('ok')\n"
        "    with open('/tmp/scratch.txt') as f:\n"
        "        return float(len(f.read()))\n"
    )
    results = evaluate_in_sandbox(
        make_metric(code),
        contexts=[{}],
        arg_mappings_json='{}',
        image=runner_image,
    )
    assert results == [(2.0, None)]


# ---------------------------------------------------------------------------
# Fatal failures from the harness side
# ---------------------------------------------------------------------------


def test_real_container_syntax_error_propagates_per_context(runner_image):
    results = evaluate_in_sandbox(
        make_metric("def m(:\n    return 1\n"),
        contexts=[{}, {}, {}],
        arg_mappings_json='{}',
        image=runner_image,
    )
    assert all(v is None for v, _ in results)
    assert all('SyntaxError' in (e or '') for _, e in results)
