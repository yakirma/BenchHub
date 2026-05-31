"""Phase A: typed-instance metric context.

Locks the new `__typed__<key>` parallel path that `get_metric_context`
adds for each CustomField, and the `input_kinds`-driven swap in
`evaluate_dynamic_metric` that hands the typed instance to opt-in
metrics. Legacy metrics (no `input_kinds`) keep getting primitives —
that contract is exercised by test_metric_context_arrays.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import benchhub as bh
from app import (
    CustomField,
    Dataset,
    GlobalMetric,
    Sample,
    db,
)
from metric_engine import (
    _metric_wants_typed,
    _stash_typed,
    _typed_for_cf,
    evaluate_dynamic_metric,
    get_metric_context,
)


@pytest.fixture
def sample(db_session):
    ds = Dataset(name='typed_ds', visibility='public')
    db.session.add(ds)
    db.session.flush()
    s = Sample(dataset_id=ds.id, name='s1')
    db.session.add(s)
    db.session.commit()
    return s


# ---------------------------------------------------------------------------
# CustomField.data_params accessors
# ---------------------------------------------------------------------------

def test_get_params_returns_empty_dict_when_null():
    cf = CustomField(name='x', data_type='depth', data_params=None)
    assert cf.get_params() == {}


def test_set_params_roundtrip():
    cf = CustomField(name='x', data_type='depth')
    cf.set_params({'unit': 'millimeters'})
    assert cf.data_params == '{"unit":"millimeters"}'
    assert cf.get_params() == {'unit': 'millimeters'}


def test_set_params_empty_clears_column():
    cf = CustomField(name='x', data_type='depth')
    cf.set_params({'unit': 'meters'})
    cf.set_params({})
    assert cf.data_params is None


def test_get_params_returns_empty_dict_on_invalid_json():
    cf = CustomField(name='x', data_type='depth', data_params='not-json')
    assert cf.get_params() == {}


# ---------------------------------------------------------------------------
# _typed_for_cf wraps primitive values in their DataType class
# ---------------------------------------------------------------------------

def test_typed_for_cf_scalar():
    cf = CustomField(name='snr', data_type='scalar')
    inst = _typed_for_cf(cf, 42.5)
    assert isinstance(inst, bh.Scalar)
    assert inst.value == 42.5


def test_typed_for_cf_text():
    cf = CustomField(name='cap', data_type='text')
    inst = _typed_for_cf(cf, 'a quick brown fox')
    assert isinstance(inst, bh.Text)
    assert inst.text == 'a quick brown fox'


def test_typed_for_cf_depth_picks_up_unit_from_params():
    cf = CustomField(name='d', data_type='depth')
    cf.set_params({'unit': 'millimeters'})
    arr = np.ones((4, 5), dtype=np.float32) * 1500
    inst = _typed_for_cf(cf, arr)
    assert isinstance(inst, bh.Depth)
    assert inst.unit == 'millimeters'
    np.testing.assert_array_equal(inst.array, arr)


def test_typed_for_cf_depth_defaults_to_meters():
    cf = CustomField(name='d', data_type='depth')
    inst = _typed_for_cf(cf, np.zeros((3, 3), dtype=np.float32))
    assert inst.unit == 'meters'


def test_typed_for_cf_image_wraps_uint8_array():
    cf = CustomField(name='rgb', data_type='image')
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    inst = _typed_for_cf(cf, arr)
    assert isinstance(inst, bh.Image)
    assert inst.array.shape == (8, 8, 3)


def test_typed_for_cf_json():
    cf = CustomField(name='meta', data_type='json')
    payload = {'relations': [{'h': 0, 't': 1}]}
    inst = _typed_for_cf(cf, payload)
    assert isinstance(inst, bh.Json)
    assert inst.data == payload


def test_typed_for_cf_label_with_names_vocab():
    """A label field carrying a `names` vocab must still wrap to a
    bh.Label (regression: Label didn't accept `names`, so cls(value,
    **params) raised TypeError, got swallowed, and typed metrics fell
    back to a raw int — making accuracy assert-fail to 0.0)."""
    cf = CustomField(name='label', data_type='label')
    cf.set_params({'names': ['airplane', 'automobile', 'bird']})
    inst = _typed_for_cf(cf, 3)
    assert isinstance(inst, bh.Label)
    assert inst.value == 3
    assert inst.names == ['airplane', 'automobile', 'bird']


def test_typed_for_cf_ignores_params_constructor_does_not_accept():
    """Defense-in-depth: a stray param a kind doesn't take is dropped,
    not fatal — the typed instance is still built."""
    cf = CustomField(name='s', data_type='scalar')
    cf.set_params({'bogus': 123, 'whatever': 'x'})
    inst = _typed_for_cf(cf, 0.9)
    assert isinstance(inst, bh.Scalar)
    assert inst.value == 0.9


def test_typed_for_cf_unknown_kind_returns_none():
    cf = CustomField(name='legacy', data_type='metric')  # 'metric' isn't in DTYPES
    assert _typed_for_cf(cf, 0.5) is None


def test_typed_for_cf_none_value_returns_none():
    cf = CustomField(name='x', data_type='depth')
    assert _typed_for_cf(cf, None) is None


# ---------------------------------------------------------------------------
# _stash_typed mirrors the primitive write with __typed__ key
# ---------------------------------------------------------------------------

def test_stash_typed_writes_both_keys():
    cf = CustomField(name='snr', data_type='scalar')
    ctx = {}
    _stash_typed(ctx, 'gt_snr', cf, 7.0)
    assert ctx['gt_snr'] == 7.0
    assert isinstance(ctx['__typed__gt_snr'], bh.Scalar)


def test_stash_typed_skips_typed_key_for_unknown_kind():
    cf = CustomField(name='lm_3', data_type='metric')
    ctx = {}
    _stash_typed(ctx, 'lm_3', cf, 0.91)
    assert ctx['lm_3'] == 0.91
    assert '__typed__lm_3' not in ctx


# ---------------------------------------------------------------------------
# get_metric_context populates __typed__ entries from sample.custom_fields
# ---------------------------------------------------------------------------

def test_get_metric_context_adds_typed_scalar(db_session, sample):
    db.session.add(CustomField(
        sample_id=sample.id, name='snr', data_type='scalar', value_float=42.0,
    ))
    db.session.commit()
    ctx = get_metric_context(sample)
    assert ctx['gt_snr'] == 42.0
    assert isinstance(ctx['__typed__gt_snr'], bh.Scalar)
    assert ctx['__typed__gt_snr'].value == 42.0


def test_get_metric_context_adds_typed_depth_with_params(db_session, sample, tmp_path):
    arr = np.array([[1.5, 2.0], [3.0, 4.5]], dtype=np.float32)
    npz = tmp_path / 'depth' / 's1.npz'
    npz.parent.mkdir(parents=True)
    np.savez_compressed(npz, depth=arr)

    cf = CustomField(
        sample_id=sample.id, name='depth_gt', data_type='depth',
        value_text=os.path.relpath(npz, tmp_path),
    )
    cf.set_params({'unit': 'millimeters'})
    db.session.add(cf)
    db.session.commit()

    ctx = get_metric_context(sample, upload_folder=str(tmp_path))
    np.testing.assert_array_equal(ctx['gt_depth_gt'], arr)
    typed = ctx['__typed__gt_depth_gt']
    assert isinstance(typed, bh.Depth)
    assert typed.unit == 'millimeters'
    np.testing.assert_array_equal(typed.array, arr)


# ---------------------------------------------------------------------------
# evaluate_dynamic_metric: typed-vs-primitive dispatch on input_kinds
# ---------------------------------------------------------------------------

def test_metric_wants_typed_false_for_null():
    gm = GlobalMetric(name='legacy', python_code='', input_kinds=None)
    assert _metric_wants_typed(gm) is False


def test_metric_wants_typed_false_for_empty_array():
    gm = GlobalMetric(name='legacy', python_code='', input_kinds='[]')
    assert _metric_wants_typed(gm) is False


def test_metric_wants_typed_true_for_declared_kinds():
    gm = GlobalMetric(name='typed', python_code='', input_kinds='["depth","depth"]')
    assert _metric_wants_typed(gm) is True


def test_evaluate_metric_passes_primitives_when_no_input_kinds():
    """Legacy metric — input_kinds NULL — receives the raw float."""
    gm = GlobalMetric(
        name='double_legacy',
        python_code='def double(x): return x * 2',
        input_kinds=None,
    )
    ctx = {}
    _stash_typed(
        ctx, 'gt_snr',
        CustomField(name='snr', data_type='scalar'), 21.0,
    )
    value, err = evaluate_dynamic_metric(gm, ctx, '{"x": "gt_snr"}')
    assert err is None
    assert value == 42.0


def test_evaluate_metric_passes_typed_when_input_kinds_declared():
    """Opt-in metric — input_kinds set — receives a Depth instance and
    reaches into `.array` + `.unit` to compute its result."""
    gm = GlobalMetric(
        name='depth_mean_meters',
        python_code=(
            'def f(gt):\n'
            '    arr = gt.array\n'
            '    if gt.unit == "millimeters":\n'
            '        arr = arr / 1000.0\n'
            '    return float(arr.mean())\n'
        ),
        input_kinds='["depth"]',
    )
    cf = CustomField(name='d', data_type='depth')
    cf.set_params({'unit': 'millimeters'})
    ctx = {}
    _stash_typed(ctx, 'gt_d', cf, np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32))

    value, err = evaluate_dynamic_metric(gm, ctx, '{"gt": "gt_d"}')
    assert err is None
    assert value == pytest.approx(2.5)


def test_evaluate_metric_typed_falls_back_to_primitive_when_no_typed_entry():
    """A typed metric pointed at a key that wasn't typed-stashed (e.g.
    the field has a `data_type` not in DTYPES) still receives the
    primitive — `__typed__<key>` is absent, the bare key wins."""
    gm = GlobalMetric(
        name='reads_value',
        python_code='def f(x): return x * 3',
        input_kinds='["scalar"]',
    )
    ctx = {'gt_lm_5': 4.0}  # no '__typed__gt_lm_5' present
    value, err = evaluate_dynamic_metric(gm, ctx, '{"x": "gt_lm_5"}')
    assert err is None
    assert value == 12.0
