"""Typed-arg + visualization support in the sandbox harness (steps 1-3 of
the sandboxing work). No docker needed — exercises the JSON round-trip
(metric_engine._jsonify_kwarg → harness._decode_arg) and the harness's
metric/visualization job kinds in-process."""
import base64
import json
import sys
from pathlib import Path

import numpy as np

import benchhub as bh
from metric_engine import _jsonify_kwarg, evaluate_viz_in_sandbox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'runner'))
import harness  # noqa: E402


def test_jsonify_kwarg_typed_instance_is_json_safe():
    enc = _jsonify_kwarg(bh.Depth(np.zeros((3, 3), np.float32), unit='meters'))
    assert enc['__bh__'] == 'depth'
    # The whole thing must survive json.dumps (the bug that broke the old path).
    json.dumps(enc)


def test_typed_metric_round_trips_through_harness():
    code = ("def rmse(gt: bh.Depth, pred: bh.Depth):\n"
            "    import numpy as np\n"
            "    return float(np.sqrt(np.mean((gt.array - pred.array) ** 2)))\n")
    gt = bh.Depth(np.zeros((4, 4), np.float32), unit='meters')
    pred = bh.Depth(np.full((4, 4), 3.0, np.float32), unit='meters')
    out = harness.run_job({
        'kind': 'metric', 'code': code, 'include_benchhub': True,
        'kwargs_list': [{'gt': _jsonify_kwarg(gt), 'pred': _jsonify_kwarg(pred)}],
    })
    assert out['fatal'] is None
    assert out['results'][0] == {'value': 3.0, 'error': None}


def test_label_vocab_survives_round_trip():
    dec = harness._decode_arg(_jsonify_kwarg(bh.Label('dog', names=['cat', 'dog'])))
    assert isinstance(dec, bh.Label)
    assert dec.value == 'dog' and dec.names == ['cat', 'dog']


def test_nested_list_of_typed_round_trips():
    items = [bh.Scalar(1.0), bh.Scalar(2.0)]
    dec = harness._decode_arg(_jsonify_kwarg(items))
    assert [round(float(x.value), 3) for x in dec] == [1.0, 2.0]


def test_visualization_job_returns_png():
    code = ("def v(x):\n"
            "    from PIL import Image\n"
            "    return Image.new('RGB', (8, 8), (10, 20, 30))\n")
    out = harness.run_job({
        'kind': 'visualization', 'code': code,
        'kwargs_list': [{'x': 1}],
    })
    r = out['results'][0]
    assert r['error'] is None
    assert base64.b64decode(r['png_b64'])[:8] == b'\x89PNG\r\n\x1a\n'


def test_visualization_non_image_return_is_error():
    out = harness.run_job({
        'kind': 'visualization', 'code': "def v(x):\n    return 42\n",
        'kwargs_list': [{'x': 1}],
    })
    r = out['results'][0]
    assert r['png_b64'] is None
    assert 'PIL.Image' in r['error']


def test_primitive_metric_unaffected():
    out = harness.run_job({'code': 'def m(x):\n    return x + 1\n',
                           'kwargs_list': [{'x': 41}]})
    assert out['results'][0] == {'value': 42.0, 'error': None}


def test_evaluate_viz_in_sandbox_shapes_fatal_string():
    # No docker call: a fatal-string payload (e.g. image missing) maps to
    # (None, error). Force it by pointing at a bogus docker binary.
    png, err = evaluate_viz_in_sandbox(
        "def v(x):\n    return None\n", {'x': 1}, docker_path='/nonexistent-docker')
    assert png is None and err
