"""Tests for runner/harness.py — the in-container metric worker.

The harness is pure Python and dependency-light, so we drive it directly
without spinning up Docker. Same tests would pass when run inside the
container; they exercise the contract.
"""
import io
import json
import math
import os
import sys

import pytest


# Make `runner/harness.py` importable without packaging.
_RUNNER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runner')
sys.path.insert(0, _RUNNER_DIR)
import harness  # noqa: E402


# ---------------------------------------------------------------------------
# run_job — pure-Python entry point
# ---------------------------------------------------------------------------


def test_happy_path_single_kwargs():
    out = harness.run_job({
        'code': 'def f(x):\n    return x * 2\n',
        'kwargs_list': [{'x': 3}],
    })
    assert out['fatal'] is None
    assert out['results'] == [{'value': 6.0, 'error': None}]


def test_batch_of_kwargs_keeps_order():
    out = harness.run_job({
        'code': 'def f(x):\n    return x + 10\n',
        'kwargs_list': [{'x': 1}, {'x': 2}, {'x': 3}],
    })
    assert [r['value'] for r in out['results']] == [11.0, 12.0, 13.0]
    assert all(r['error'] is None for r in out['results'])


def test_function_name_picks_specific_callable():
    code = (
        "def helper(x):\n    return x\n"
        "def metric(x):\n    return x * 100\n"
    )
    out = harness.run_job({
        'code': code,
        'function_name': 'metric',
        'kwargs_list': [{'x': 2}],
    })
    assert out['results'][0]['value'] == 200.0


def test_function_name_unknown_returns_fatal():
    out = harness.run_job({
        'code': 'def f(): return 1\n',
        'function_name': 'does_not_exist',
        'kwargs_list': [{}],
    })
    assert out['results'] == []
    assert "does_not_exist" in out['fatal']


def test_syntax_error_is_fatal():
    out = harness.run_job({
        'code': 'def f(:\n    return 1\n',
        'kwargs_list': [{}],
    })
    assert out['results'] == []
    assert 'SyntaxError' in out['fatal']


def test_no_callable_is_fatal():
    out = harness.run_job({
        'code': 'x = 5\n',
        'kwargs_list': [{}],
    })
    assert out['results'] == []
    assert 'No callable' in out['fatal']


def test_runtime_error_is_per_call_not_fatal():
    code = (
        "def f(x):\n"
        "    if x == 0:\n"
        "        raise ValueError('zero')\n"
        "    return 1.0 / x\n"
    )
    out = harness.run_job({
        'code': code,
        'kwargs_list': [{'x': 2}, {'x': 0}, {'x': 4}],
    })
    assert out['fatal'] is None
    assert out['results'][0] == {'value': 0.5, 'error': None}
    assert out['results'][1]['value'] is None
    assert 'ValueError' in out['results'][1]['error']
    assert out['results'][2]['value'] == 0.25


def test_nan_result_becomes_error():
    out = harness.run_job({
        'code': "def f():\n    return float('nan')\n",
        'kwargs_list': [{}],
    })
    assert out['results'][0]['value'] is None
    assert 'NaN' in out['results'][0]['error']


def test_inf_result_becomes_error():
    out = harness.run_job({
        'code': "def f():\n    return float('inf')\n",
        'kwargs_list': [{}],
    })
    assert out['results'][0]['value'] is None
    assert 'Inf' in out['results'][0]['error']


def test_non_numeric_result_becomes_error():
    out = harness.run_job({
        'code': "def f():\n    return 'hello'\n",
        'kwargs_list': [{}],
    })
    assert out['results'][0]['value'] is None
    assert out['results'][0]['error'] is not None


def test_kwargs_list_must_be_a_list():
    out = harness.run_job({
        'code': 'def f(): return 1\n',
        'kwargs_list': 'not a list',
    })
    assert out['fatal'] is not None
    assert 'list' in out['fatal']


def test_individual_kwargs_must_be_dicts():
    out = harness.run_job({
        'code': 'def f(): return 1\n',
        'kwargs_list': [{}, 'not a dict', {}],
    })
    # First and third succeed; second is per-call error.
    assert out['fatal'] is None
    assert out['results'][0]['value'] == 1.0
    assert out['results'][1]['value'] is None
    assert 'dict' in out['results'][1]['error']


def test_numpy_is_injected_by_default():
    out = harness.run_job({
        'code': 'def f(arr):\n    return float(np.mean(arr))\n',
        'kwargs_list': [{'arr': [1.0, 2.0, 3.0, 4.0]}],
    })
    assert out['results'][0]['value'] == pytest.approx(2.5)


def test_numpy_can_be_disabled():
    out = harness.run_job({
        'code': "def f():\n    return np.mean([1, 2])\n",
        'kwargs_list': [{}],
        'include_numpy': False,
    })
    # With numpy not injected, `np` is undefined → NameError → per-call error.
    assert out['results'][0]['value'] is None
    assert 'np' in out['results'][0]['error'] or 'NameError' in out['results'][0]['error']


# ---------------------------------------------------------------------------
# main() — stdin/stdout JSON contract
# ---------------------------------------------------------------------------


def test_main_reads_stdin_writes_stdout():
    job = {
        'code': 'def f(x): return x + 1\n',
        'kwargs_list': [{'x': 41}],
    }
    stdin = io.StringIO(json.dumps(job))
    stdout = io.StringIO()
    rc = harness.main(stdin=stdin, stdout=stdout)
    assert rc == 0
    payload = json.loads(stdout.getvalue())
    assert payload['fatal'] is None
    assert payload['results'][0]['value'] == 42.0


def test_main_returns_2_on_invalid_json_input():
    stdin = io.StringIO('not json')
    stdout = io.StringIO()
    rc = harness.main(stdin=stdin, stdout=stdout)
    assert rc == 2
    payload = json.loads(stdout.getvalue())
    assert payload['fatal'] is not None
    assert 'JSON' in payload['fatal']
    assert payload['results'] == []
