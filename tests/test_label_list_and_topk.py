"""bh.LabelList typed kind + top-1 / top-5 reference metrics.

LabelList is the dataset-contract shape for a ranked top-K
classification prediction. The dataset declares a pred field with
`kind=label_list` and (optionally) `params={"k": 5, "names": [...]}`
so the typed-manifest importer + metric engine know what to expect.
"""
from __future__ import annotations

import json

import pytest

import benchhub as bh
from benchhub.types import DTYPES


# ---------------------------------------------------------------------------
# Type-class behaviour
# ---------------------------------------------------------------------------

def test_labellist_is_registered_in_dtypes():
    assert 'label_list' in DTYPES
    assert DTYPES['label_list'] is bh.LabelList


def test_labellist_encode_decode_roundtrip_preserves_values():
    inst = bh.LabelList([3, 5, 8, 1, 0], k=5)
    blob = inst.encode()
    decoded = bh.LabelList.decode(blob, {'k': 5})
    assert decoded.values == [3, 5, 8, 1, 0]
    assert decoded.k == 5


def test_labellist_decode_carries_names_from_params():
    blob = json.dumps([3, 1]).encode('utf-8')
    decoded = bh.LabelList.decode(blob, {'k': 2, 'names': ['cat', 'dog', 'fish']})
    assert decoded.names == ['cat', 'dog', 'fish']


def test_labellist_rejects_non_int_str_values():
    with pytest.raises(ValueError, match='int or str'):
        bh.LabelList([1.5, 'cat'], k=2)


def test_labellist_requires_k():
    with pytest.raises(TypeError):
        bh.LabelList([1, 2, 3])  # missing required keyword `k`


def test_labellist_rejects_invalid_k():
    with pytest.raises(ValueError, match='k must be'):
        bh.LabelList([1, 2], k=0)


def test_labellist_decode_requires_k_in_params():
    blob = json.dumps([1, 2]).encode('utf-8')
    with pytest.raises(ValueError, match='requires `k` in params'):
        bh.LabelList.decode(blob, {})


def test_labellist_validate_rejects_wrong_length():
    """Exact-length contract: validate() raises whenever values
    isn't exactly k long. Top-K submissions are not allowed to
    short-fill or over-fill."""
    inst = bh.LabelList([1, 2, 3, 4], k=5)
    with pytest.raises(ValueError, match='exact match required'):
        inst.validate()
    inst = bh.LabelList([1, 2, 3, 4, 5, 6], k=5)
    with pytest.raises(ValueError, match='exact match required'):
        inst.validate()


def test_labellist_validate_accepts_exact_length():
    inst = bh.LabelList([1, 2, 3, 4, 5], k=5)
    inst.validate()  # no raise


def test_labellist_visualize_uses_vocab_names():
    inst = bh.LabelList([3, 0], k=2, names=['airplane', 'automobile', 'bird', 'cat'])
    body, mime = inst.visualize()
    assert body.decode() == '3 cat, 0 airplane'
    assert mime == 'text/plain; charset=utf-8'


def test_labellist_visualize_falls_back_when_no_vocab():
    inst = bh.LabelList([3, 0, 9], k=3)
    body, _ = inst.visualize()
    assert body.decode() == '3, 0, 9'


# ---------------------------------------------------------------------------
# Reference metric behaviour — top-1 / top-5
# ---------------------------------------------------------------------------

def _run_metric(python_code: str, gt, pred):
    """Exec the source like the metric engine does and call the
    first function with (gt, pred). Mirrors evaluate_dynamic_metric
    just enough to test the seeded snippets."""
    import benchhub as _bh
    import numpy as _np
    scope = {'np': _np, 'bh': _bh, 'benchhub': _bh}
    exec(python_code, scope)
    func = next(v for k, v in scope.items()
                if callable(v) and not k.startswith('_') and k not in ('np', 'bh', 'benchhub'))
    return func(gt, pred)


def test_top1_accuracy_returns_one_when_top_pick_matches():
    from scripts.seed_reference_metrics import _TOP_1_ACC
    gt = bh.Label(3)
    pred = bh.LabelList([3, 7, 2, 5, 1], k=5)
    assert _run_metric(_TOP_1_ACC, gt, pred) == 1.0


def test_top1_accuracy_returns_zero_when_top_pick_wrong():
    from scripts.seed_reference_metrics import _TOP_1_ACC
    gt = bh.Label(3)
    pred = bh.LabelList([7, 3, 2, 5, 1], k=5)  # 3 is at index 1 — not top-1
    assert _run_metric(_TOP_1_ACC, gt, pred) == 0.0


def test_top5_accuracy_returns_one_when_gt_in_top_five():
    from scripts.seed_reference_metrics import _TOP_5_ACC
    gt = bh.Label(3)
    pred = bh.LabelList([7, 2, 5, 1, 3], k=5)  # 3 at index 4 — inside top-5
    assert _run_metric(_TOP_5_ACC, gt, pred) == 1.0


def test_top5_accuracy_returns_zero_when_gt_beyond_top_five():
    from scripts.seed_reference_metrics import _TOP_5_ACC
    gt = bh.Label(99)
    pred = bh.LabelList([1, 2, 3, 4, 5], k=5)  # 99 is not in top-5
    assert _run_metric(_TOP_5_ACC, gt, pred) == 0.0


def test_top5_accuracy_asserts_pred_is_labellist():
    from scripts.seed_reference_metrics import _TOP_5_ACC
    gt = bh.Label(0)
    with pytest.raises(AssertionError, match='pred must be bh.LabelList'):
        _run_metric(_TOP_5_ACC, gt, bh.Label(0))  # wrong type for pred


def test_top1_accuracy_asserts_gt_is_label():
    from scripts.seed_reference_metrics import _TOP_1_ACC
    pred = bh.LabelList([1, 2, 3], k=3)
    with pytest.raises(AssertionError, match='gt must be bh.Label'):
        _run_metric(_TOP_1_ACC, bh.LabelList([1], k=1), pred)


# ---------------------------------------------------------------------------
# HF materializer accepts the new kind
# ---------------------------------------------------------------------------

def test_row_value_to_typed_coerces_list_into_labellist():
    from benchhub.hf_materialize import _row_value_to_typed
    inst = _row_value_to_typed([3, 5, 1, 0, 8], 'label_list',
                               {'k': 5, 'names': ['a', 'b', 'c']})
    assert isinstance(inst, bh.LabelList)
    assert inst.values == [3, 5, 1, 0, 8]
    assert inst.k == 5
    assert inst.names == ['a', 'b', 'c']


def test_row_value_to_typed_returns_none_for_non_list_value():
    from benchhub.hf_materialize import _row_value_to_typed
    assert _row_value_to_typed(3, 'label_list', {'k': 5}) is None


def test_row_value_to_typed_raises_when_k_missing():
    """The contract requires `k` in params — the materializer must
    surface the missing declaration rather than silently picking
    a default."""
    from benchhub.hf_materialize import _row_value_to_typed
    # `_row_value_to_typed` catches the ValueError + returns None
    # via its outer except — so we observe the same "skip this
    # field" signal the caller gets.
    out = _row_value_to_typed([1, 2, 3], 'label_list', {})
    assert out is None


def test_validate_manifest_requires_k_on_label_list_field():
    from benchhub.manifest import validate_manifest
    manifest = {
        'name': 'tk',
        'fields': [{'name': 'lp', 'kind': 'label_list', 'role': 'pred',
                    'params': {}}],
        'samples': ['s0'],
    }
    with pytest.raises(ValueError, match='requires params.k'):
        validate_manifest(manifest)
    # With k declared the same manifest validates cleanly.
    manifest['fields'][0]['params'] = {'k': 5}
    validate_manifest(manifest)  # no raise
