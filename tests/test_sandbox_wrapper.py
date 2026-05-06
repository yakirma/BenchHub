"""Tests for metric_engine.evaluate_in_sandbox.

The wrapper shells out to docker. We mock subprocess so the suite never
actually needs a docker daemon. Coverage spans:

- the docker invocation receives the right hardening flags
- the JSON job sent on stdin has the right shape
- per-context (value, error) tuples come back in the right order
- fatal failures (image missing, timeout, non-zero exit, malformed JSON)
  are surfaced uniformly as (None, <reason>) for every input context
"""
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from metric_engine import _build_kwargs, evaluate_in_sandbox, sandbox_evaluate_one


def make_metric(code='def m(x):\n    return x\n', name='m'):
    return SimpleNamespace(name=name, python_code=code)


def fake_proc(stdout='', stderr='', returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _build_kwargs (parity with the in-process path)
# ---------------------------------------------------------------------------


class TestBuildKwargs:
    def test_context_lookup(self):
        out = _build_kwargs({"a": "x", "b": "y"}, {"x": 1, "y": 2})
        assert out == {"a": 1, "b": 2}

    def test_scalar_int_literal(self):
        assert _build_kwargs({"a": "SCALAR:7"}, {}) == {"a": 7}

    def test_scalar_float_literal(self):
        assert _build_kwargs({"a": "SCALAR:3.14"}, {}) == {"a": pytest.approx(3.14)}

    def test_scalar_string_fallback(self):
        assert _build_kwargs({"a": "SCALAR:hello"}, {}) == {"a": "hello"}

    def test_missing_context_key_passes_none(self):
        assert _build_kwargs({"a": "missing"}, {}) == {"a": None}


# ---------------------------------------------------------------------------
# evaluate_in_sandbox — happy path
# ---------------------------------------------------------------------------


def test_returns_per_context_tuples_in_order():
    payload = {
        'fatal': None,
        'results': [
            {'value': 1.0, 'error': None},
            {'value': 2.0, 'error': None},
            {'value': None, 'error': 'boom'},
        ],
    }
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))):
        results = evaluate_in_sandbox(
            make_metric(),
            contexts=[{'x': 1}, {'x': 2}, {'x': 3}],
            arg_mappings_json='{"x": "x"}',
        )

    assert results == [(1.0, None), (2.0, None), (None, 'boom')]


def test_command_uses_required_hardening_flags():
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))) as mock_run:
        evaluate_in_sandbox(make_metric(), contexts=[{}], arg_mappings_json='{}')

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == 'docker'
    assert cmd[1:3] == ['run', '--rm']
    assert '--network=none' in cmd
    assert '--read-only' in cmd
    assert any(c.startswith('--memory=') for c in cmd)
    assert any(c.startswith('--cpus=') for c in cmd)
    # no-new-privileges hardens against setuid escalation inside the image.
    nnp_idx = cmd.index('--security-opt')
    assert cmd[nnp_idx + 1] == 'no-new-privileges'
    # tmpfs at /tmp so matplotlib / numpy can write caches even under --read-only.
    tmpfs_idx = cmd.index('--tmpfs')
    assert cmd[tmpfs_idx + 1].startswith('/tmp')
    # The image must appear in the command line. The arguments after it
    # (`python /app/harness.py`) override the image's default CMD which
    # is now gunicorn (HTTP server mode).
    assert 'benchhub-runner' in cmd
    assert cmd[-2:] == ['python', '/app/harness.py']


def test_passes_job_json_on_stdin():
    payload = {'fatal': None, 'results': [{'value': 5.0, 'error': None}]}
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))) as mock_run:
        evaluate_in_sandbox(
            make_metric(code='def m(x, y):\n    return x + y\n'),
            contexts=[{'a': 2, 'b': 3}],
            arg_mappings_json='{"x": "a", "y": "b"}',
        )

    job = json.loads(mock_run.call_args.kwargs['input'])
    assert 'def m(x, y)' in job['code']
    assert job['kwargs_list'] == [{'x': 2, 'y': 3}]
    assert job['include_numpy'] is True


def test_image_overridable_via_kwarg_and_env(monkeypatch):
    # Make sure no leftover URL routes us to the HTTP backend.
    monkeypatch.delenv('BENCHHUB_SANDBOX_URL', raising=False)
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}

    # Explicit kwarg wins.
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))) as mock_run:
        evaluate_in_sandbox(make_metric(), [{}], '{}', image='custom/runner:42')
    assert 'custom/runner:42' in mock_run.call_args.args[0]

    # Falls back to env var.
    monkeypatch.setenv('BENCHHUB_SANDBOX_IMAGE', 'env/runner:9')
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))) as mock_run:
        evaluate_in_sandbox(make_metric(), [{}], '{}')
    assert 'env/runner:9' in mock_run.call_args.args[0]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def _all_have_fatal(results, expected_substr):
    """Helper: every context in `results` must report the same fatal-style error."""
    assert all(v is None for v, _ in results)
    for _, err in results:
        assert err is not None
        assert expected_substr in err, err


def test_docker_missing_returns_fatal_for_every_context():
    with patch('metric_engine.subprocess.run', side_effect=FileNotFoundError):
        results = evaluate_in_sandbox(make_metric(), contexts=[{}, {}, {}], arg_mappings_json='{}')

    assert len(results) == 3
    _all_have_fatal(results, 'docker not found')


def test_timeout_returns_fatal_for_every_context():
    with patch('metric_engine.subprocess.run', side_effect=subprocess.TimeoutExpired(cmd=[], timeout=60)):
        results = evaluate_in_sandbox(make_metric(), contexts=[{}, {}], arg_mappings_json='{}', timeout_seconds=60)
    _all_have_fatal(results, 'timed out')


def test_container_nonzero_with_no_stdout_surfaces_stderr():
    with patch('metric_engine.subprocess.run', return_value=fake_proc(
        stdout='', stderr='Unable to find image benchhub-runner:latest\n', returncode=125,
    )):
        results = evaluate_in_sandbox(make_metric(), contexts=[{}], arg_mappings_json='{}')
    assert results[0][0] is None
    assert 'rc=125' in results[0][1]
    assert 'Unable to find image' in results[0][1]


def test_malformed_json_from_container_is_fatal():
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout='not json at all')):
        results = evaluate_in_sandbox(make_metric(), contexts=[{}, {}], arg_mappings_json='{}')
    _all_have_fatal(results, 'non-JSON')


def test_harness_fatal_is_propagated():
    payload = {'fatal': 'SyntaxError: bad code', 'results': []}
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))):
        results = evaluate_in_sandbox(make_metric(), contexts=[{}, {}, {}], arg_mappings_json='{}')
    _all_have_fatal(results, 'SyntaxError')


def test_short_results_array_is_padded():
    """If the harness returns fewer results than contexts (shouldn't happen,
    but defend against it), missing slots get ``(None, "missing result")``."""
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))):
        results = evaluate_in_sandbox(make_metric(), contexts=[{}, {}, {}], arg_mappings_json='{}')

    assert len(results) == 3
    assert results[0] == (1.0, None)
    assert results[1] == (None, 'missing result')
    assert results[2] == (None, 'missing result')


# ---------------------------------------------------------------------------
# sandbox_evaluate_one — single-context wrapper
# ---------------------------------------------------------------------------


def test_sandbox_evaluate_one_returns_first_tuple():
    payload = {'fatal': None, 'results': [{'value': 7.0, 'error': None}]}
    with patch('metric_engine.subprocess.run', return_value=fake_proc(stdout=json.dumps(payload))):
        v, err = sandbox_evaluate_one(make_metric(), {'x': 1}, '{"x": "x"}')
    assert v == 7.0
    assert err is None
