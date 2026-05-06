"""Tests for the HTTP-backed path of metric_engine.evaluate_in_sandbox.

The function dispatches between the docker-subprocess backend (existing,
covered by tests/test_sandbox_wrapper.py) and the HTTP-POST-to-runner-app
backend (new, covered here). Selection is by env var or kwarg.

requests.post is mocked end-to-end; no real network.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from metric_engine import evaluate_in_sandbox


def make_metric(code='def m(x):\n    return x\n'):
    return SimpleNamespace(name='m', python_code=code)


def fake_response(payload, status=200):
    """Minimal stand-in for requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    return resp


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_url_kwarg_routes_to_http_backend(monkeypatch):
    """Explicit url= must pick the HTTP path even if BENCHHUB_SANDBOX_URL
    isn't set."""
    monkeypatch.delenv('BENCHHUB_SANDBOX_URL', raising=False)
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}

    with patch('metric_engine._requests') as mod, \
         patch('metric_engine.subprocess.run') as docker_mock:
        mod.post.return_value = fake_response(payload)
        evaluate_in_sandbox(make_metric(), [{}], '{}', url='http://runner.internal:8080/run')

    docker_mock.assert_not_called()
    mod.post.assert_called_once()


def test_env_var_routes_to_http_backend(monkeypatch):
    """When BENCHHUB_SANDBOX_URL is set, the HTTP path wins without needing
    the explicit kwarg. Pinned so misconfigured deploys don't silently fall
    back to docker (which would fail on Fly machines that lack a docker
    daemon)."""
    monkeypatch.setenv('BENCHHUB_SANDBOX_URL', 'http://runner.internal:8080/run')
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}

    with patch('metric_engine._requests') as mod, \
         patch('metric_engine.subprocess.run') as docker_mock:
        mod.post.return_value = fake_response(payload)
        evaluate_in_sandbox(make_metric(), [{}], '{}')

    docker_mock.assert_not_called()
    mod.post.assert_called_once()
    assert mod.post.call_args.args[0] == 'http://runner.internal:8080/run'


def test_no_url_falls_back_to_docker(monkeypatch):
    monkeypatch.delenv('BENCHHUB_SANDBOX_URL', raising=False)
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}

    with patch('metric_engine._requests') as mod, \
         patch('metric_engine.subprocess.run') as docker_mock:
        # docker subprocess returns a CompletedProcess with stdout=JSON
        proc = MagicMock(returncode=0, stdout=json.dumps(payload), stderr='')
        docker_mock.return_value = proc
        evaluate_in_sandbox(make_metric(), [{}], '{}')

    docker_mock.assert_called_once()
    mod.post.assert_not_called()


# ---------------------------------------------------------------------------
# HTTP request shape
# ---------------------------------------------------------------------------


def test_http_post_carries_correct_job_payload(monkeypatch):
    monkeypatch.delenv('BENCHHUB_SANDBOX_URL', raising=False)
    payload = {'fatal': None, 'results': [{'value': 5.0, 'error': None}]}

    with patch('metric_engine._requests') as mod:
        mod.post.return_value = fake_response(payload)
        evaluate_in_sandbox(
            make_metric(code='def m(x, y): return x + y\n'),
            contexts=[{'a': 2, 'b': 3}],
            arg_mappings_json='{"x": "a", "y": "b"}',
            url='http://runner.internal:8080/run',
        )

    call = mod.post.call_args
    body = call.kwargs['json']
    assert 'def m(x, y)' in body['code']
    assert body['kwargs_list'] == [{'x': 2, 'y': 3}]
    assert body['include_numpy'] is True


def test_http_post_uses_caller_supplied_timeout(monkeypatch):
    monkeypatch.delenv('BENCHHUB_SANDBOX_URL', raising=False)
    payload = {'fatal': None, 'results': [{'value': 1.0, 'error': None}]}

    with patch('metric_engine._requests') as mod:
        mod.post.return_value = fake_response(payload)
        evaluate_in_sandbox(
            make_metric(), [{}], '{}',
            url='http://runner.internal:8080/run',
            timeout_seconds=10,
        )

    assert mod.post.call_args.kwargs['timeout'] == 10


def test_http_results_returned_in_order(monkeypatch):
    payload = {
        'fatal': None,
        'results': [
            {'value': 1.0, 'error': None},
            {'value': None, 'error': 'boom'},
            {'value': 3.0, 'error': None},
        ],
    }
    with patch('metric_engine._requests') as mod:
        mod.post.return_value = fake_response(payload)
        results = evaluate_in_sandbox(
            make_metric(), [{}, {}, {}], '{}',
            url='http://runner.internal:8080/run',
        )

    assert results == [(1.0, None), (None, 'boom'), (3.0, None)]


# ---------------------------------------------------------------------------
# HTTP error paths
# ---------------------------------------------------------------------------


def _all_have_fatal(results, expected_substr):
    assert all(v is None for v, _ in results)
    for _, err in results:
        assert err is not None
        assert expected_substr in err, err


def test_http_timeout_propagates_as_per_context_fatal():
    import requests as real_requests
    with patch('metric_engine._requests') as mod:
        mod.exceptions = real_requests.exceptions
        mod.post.side_effect = real_requests.exceptions.Timeout()
        results = evaluate_in_sandbox(
            make_metric(), [{}, {}, {}], '{}',
            url='http://runner.internal:8080/run',
            timeout_seconds=30,
        )
    _all_have_fatal(results, 'timed out')


def test_http_connection_error_surfaces_url():
    import requests as real_requests
    with patch('metric_engine._requests') as mod:
        mod.exceptions = real_requests.exceptions
        mod.post.side_effect = real_requests.exceptions.ConnectionError("conn refused")
        results = evaluate_in_sandbox(
            make_metric(), [{}], '{}',
            url='http://runner.internal:8080/run',
        )
    assert results[0][0] is None
    err = results[0][1]
    assert 'unreachable' in err
    assert 'runner.internal' in err


def test_http_4xx_with_json_body_surfaces_fatal_field():
    """Server returns 413 / 400 with {fatal: <msg>, results: []}. The wrapper
    pulls out the fatal string and propagates it to every context."""
    payload = {'fatal': 'job payload too large', 'results': []}
    with patch('metric_engine._requests') as mod:
        mod.post.return_value = fake_response(payload, status=413)
        results = evaluate_in_sandbox(
            make_metric(), [{}, {}], '{}',
            url='http://runner.internal:8080/run',
        )
    _all_have_fatal(results, 'job payload too large')


def test_http_5xx_without_json_body_falls_back_to_status():
    resp = MagicMock()
    resp.status_code = 502
    resp.json.side_effect = ValueError("not json")
    resp.text = '<html>bad gateway</html>'
    with patch('metric_engine._requests') as mod:
        mod.post.return_value = resp
        results = evaluate_in_sandbox(
            make_metric(), [{}], '{}',
            url='http://runner.internal:8080/run',
        )
    _all_have_fatal(results, 'HTTP 502')


def test_http_200_with_non_json_body_is_fatal():
    """If the runner is hijacked / replaced with an HTML-emitting server,
    the wrapper must flag the response as malformed rather than swallow it."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("Expecting value")
    resp.text = '<html>oops</html>'
    with patch('metric_engine._requests') as mod:
        mod.post.return_value = resp
        results = evaluate_in_sandbox(
            make_metric(), [{}, {}], '{}',
            url='http://runner.internal:8080/run',
        )
    _all_have_fatal(results, 'non-JSON')


def test_http_harness_fatal_field_propagates():
    """200 OK + harness-level fatal (e.g. SyntaxError in user code) — every
    context inherits the same error, matching the docker path's behavior."""
    payload = {'fatal': 'SyntaxError: bad metric', 'results': []}
    with patch('metric_engine._requests') as mod:
        mod.post.return_value = fake_response(payload)
        results = evaluate_in_sandbox(
            make_metric(), [{}, {}, {}], '{}',
            url='http://runner.internal:8080/run',
        )
    _all_have_fatal(results, 'SyntaxError')
