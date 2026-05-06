"""Tests for runner/server.py — the HTTP wrapper around harness.

Drives the Flask app through Flask's test client. No Docker, no real
network. Covers the request/response shape, error handling, and the body-
size guard.
"""
import json
import os
import sys

import pytest


# Make `runner/server.py` importable. The `runner` directory has its own
# isolation from the BenchHub Flask app — sharing a path entry just lets
# the test runner import it.
_RUNNER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runner')
if _RUNNER_DIR not in sys.path:
    sys.path.insert(0, _RUNNER_DIR)


@pytest.fixture
def runner_client():
    # Import inside the fixture so the path setup above is in effect.
    import server as runner_server
    runner_server.app.config['TESTING'] = True
    return runner_server.app.test_client()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200_ok(runner_client):
    resp = runner_client.get('/health')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True}


# ---------------------------------------------------------------------------
# /run — happy path
# ---------------------------------------------------------------------------


def test_run_executes_metric_and_returns_per_call_results(runner_client):
    job = {
        'code': 'def m(x):\n    return x + 100\n',
        'kwargs_list': [{'x': 1}, {'x': 2}, {'x': 3}],
    }
    resp = runner_client.post('/run', json=job)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['fatal'] is None
    assert [r['value'] for r in body['results']] == [101.0, 102.0, 103.0]


def test_run_propagates_harness_fatal(runner_client):
    """Syntax error → harness returns fatal:<msg>, results:[]. Wrapper
    relays that as-is over HTTP 200 — the *transport* succeeded, the
    *job* failed."""
    resp = runner_client.post('/run', json={
        'code': 'def m(:\n    return 1\n',
        'kwargs_list': [{}, {}],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['results'] == []
    assert 'SyntaxError' in body['fatal']


def test_run_per_call_error_returned_in_results(runner_client):
    """A runtime error in one context doesn't sink the batch."""
    resp = runner_client.post('/run', json={
        'code': "def m(x):\n    return 1.0 / x\n",
        'kwargs_list': [{'x': 2}, {'x': 0}],
    })
    body = resp.get_json()
    assert body['fatal'] is None
    assert body['results'][0]['value'] == 0.5
    assert body['results'][1]['value'] is None
    assert 'ZeroDivisionError' in body['results'][1]['error']


# ---------------------------------------------------------------------------
# /run — error responses
# ---------------------------------------------------------------------------


def test_run_400_when_body_is_not_a_json_object(runner_client):
    """Lists, strings, numbers — anything that's not a dict — get rejected
    at the wrapper before the harness sees them."""
    resp = runner_client.post('/run', json=['not', 'a', 'dict'])
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['results'] == []
    assert 'JSON object' in body['fatal']


def test_run_400_when_body_is_missing(runner_client):
    """No JSON body at all → 400 with a clear fatal."""
    resp = runner_client.post('/run', data='not json',
                              content_type='application/json')
    assert resp.status_code == 400
    assert 'JSON object' in resp.get_json()['fatal']


def test_run_413_when_body_is_too_large(runner_client, monkeypatch):
    """The MAX_CONTENT_LENGTH guard short-circuits oversized POSTs at the
    WSGI layer. Pin the wording so callers can detect it programmatically."""
    # Shrink the cap to make the test fast.
    import server as runner_server
    monkeypatch.setitem(runner_server.app.config, 'MAX_CONTENT_LENGTH', 100)

    huge_code = 'def m(): return 1\n' + ('# pad\n' * 200)  # > 100 bytes
    resp = runner_client.post('/run', json={'code': huge_code, 'kwargs_list': [{}]})
    assert resp.status_code == 413
    assert resp.get_json()['fatal'] == 'job payload too large'


# ---------------------------------------------------------------------------
# In-process state isolation reminder
# ---------------------------------------------------------------------------


def test_run_does_not_leak_globals_across_requests(runner_client):
    """Each call to harness.run_job uses a fresh exec namespace, so a
    metric that scribbles on `globals()` can't influence the next request.
    Pinning the soft-boundary contract documented in server.py.
    (This is NOT a strong guarantee against sys.modules pollution — that's
    what gunicorn --max-requests is for in production. See server.py.)"""
    # First request: monkey-patches a local global. Should not survive.
    runner_client.post('/run', json={
        'code': "shared = {'tampered': True}\ndef m(): return 1\n",
        'kwargs_list': [{}],
    })
    # Second request: tries to read the global the first one created.
    resp = runner_client.post('/run', json={
        'code': (
            "def m():\n"
            "    try:\n"
            "        return float(shared.get('tampered', 0))\n"
            "    except NameError:\n"
            "        return -1.0\n"
        ),
        'kwargs_list': [{}],
    })
    body = resp.get_json()
    # Either NameError-handled (-1.0) or the variable is gone — never 1.0.
    assert body['results'][0]['value'] == -1.0
