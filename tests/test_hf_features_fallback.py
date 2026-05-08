"""Streaming-fallback for `_hf_fetch_features`.

Many HuggingFace datasets (community uploads, image folders, script-
backed) return 200 from /api/datasets/<repo> but do NOT populate
`cardData.dataset_info` — so the API-only path comes back empty and
the importer skips them. The fallback opens the dataset via the
`datasets` library in streaming mode and reads `ds.features` (or
peeks a single row). Pin the conversions so a future refactor can't
silently drop the fallback.
"""
import sys
import types
from unittest.mock import patch

import pytest

from app import (
    _hf_fetch_features,
    _hf_features_via_streaming,
    _features_from_datasets_features,
    _describe_feature,
    _features_from_example,
)


# ---------------------------------------------------------------------------
# _describe_feature: walk the datasets.Features object tree
# ---------------------------------------------------------------------------


class _FakeImage: pass
class _FakeAudio: pass


class _FakeClassLabel:
    def __init__(self, names):
        self.names = names


class _FakeValue:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeSequence:
    def __init__(self, feature, length=-1):
        self.feature = feature
        self.length = length


def test_describe_image():
    f = _FakeImage(); f.__class__.__name__ = 'Image'
    assert _describe_feature(f) == {'type': 'Image'}


def test_describe_classlabel_carries_names():
    f = _FakeClassLabel(names=['cat', 'dog'])
    f.__class__.__name__ = 'ClassLabel'
    assert _describe_feature(f) == {'type': 'ClassLabel',
                                    'names': ['cat', 'dog']}


def test_describe_value_carries_dtype():
    f = _FakeValue('int64'); f.__class__.__name__ = 'Value'
    assert _describe_feature(f) == {'type': 'Value:int64'}


def test_describe_sequence_unwraps_inner_value_dtype():
    """`Sequence(Value('int32'))` should normalize to 'Sequence:int32'
    (matching the REST-API shape) — not 'Sequence:Value:int32'."""
    inner = _FakeValue('int32'); inner.__class__.__name__ = 'Value'
    seq = _FakeSequence(feature=inner, length=1024)
    seq.__class__.__name__ = 'Sequence'
    out = _describe_feature(seq)
    assert out == {'type': 'Sequence:int32', 'length': 1024}


def test_describe_unknown_falls_back():
    class _Weird: pass
    f = _Weird()
    assert _describe_feature(f) == {'type': 'unknown'}


# ---------------------------------------------------------------------------
# _features_from_datasets_features: dict-walking
# ---------------------------------------------------------------------------


def test_features_from_datasets_features_walks_each_column():
    img = _FakeImage(); img.__class__.__name__ = 'Image'
    label = _FakeClassLabel(names=['a', 'b', 'c']); label.__class__.__name__ = 'ClassLabel'
    feats = {'image': img, 'label': label}
    out = _features_from_datasets_features(feats)
    assert out == {
        'image': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['a', 'b', 'c']},
    }


# ---------------------------------------------------------------------------
# _features_from_example: last-resort row peek
# ---------------------------------------------------------------------------


def test_features_from_example_infers_basic_python_types():
    from PIL import Image as _PILImage
    row = {
        'rgb':   _PILImage.new('RGB', (4, 4)),
        'flag':  True,
        'count': 42,
        'score': 0.5,
        'name':  'foo',
        'seq':   [1, 2, 3],
    }
    out = _features_from_example(row)
    assert out['rgb']   == {'type': 'Image'}
    assert out['flag']  == {'type': 'Value:bool'}
    assert out['count'] == {'type': 'Value:int64'}
    assert out['score'] == {'type': 'Value:float32'}
    assert out['name']  == {'type': 'Value:string'}
    assert out['seq']['type'].startswith('Sequence')


# ---------------------------------------------------------------------------
# _hf_features_via_streaming: end-to-end via a fake `datasets` lib
# ---------------------------------------------------------------------------


def _install_fake_datasets(monkeypatch, fake_features=None, fake_rows=None):
    """Install a stand-in `datasets` module exposing load_dataset that
    returns a streaming-ish dataset with the supplied features/rows."""
    mod = types.ModuleType('datasets')

    class _FakeDS:
        features = fake_features
        def __iter__(self):
            return iter(fake_rows or [])

    mod.load_dataset = lambda *a, **kw: _FakeDS()
    monkeypatch.setitem(sys.modules, 'datasets', mod)


def test_streaming_fallback_returns_features_when_declared(monkeypatch):
    img = _FakeImage(); img.__class__.__name__ = 'Image'
    label = _FakeClassLabel(['cat', 'dog']); label.__class__.__name__ = 'ClassLabel'
    _install_fake_datasets(monkeypatch, fake_features={'image': img, 'label': label})
    out = _hf_features_via_streaming('fake/community')
    assert out['image']['type'] == 'Image'
    assert out['label']['type'] == 'ClassLabel'
    assert out['label']['names'] == ['cat', 'dog']


def test_streaming_fallback_peeks_row_when_no_features(monkeypatch):
    """No declared schema → peek one row."""
    from PIL import Image as _PILImage
    rows = [{'image': _PILImage.new('RGB', (2, 2)), 'count': 5}]
    _install_fake_datasets(monkeypatch, fake_features=None, fake_rows=rows)
    out = _hf_features_via_streaming('fake/no-schema')
    assert out['image'] == {'type': 'Image'}
    assert out['count'] == {'type': 'Value:int64'}


def test_streaming_fallback_returns_empty_when_load_dataset_fails(monkeypatch):
    mod = types.ModuleType('datasets')
    def _raise(*a, **kw):
        raise RuntimeError('repo offline')
    mod.load_dataset = _raise
    monkeypatch.setitem(sys.modules, 'datasets', mod)
    assert _hf_features_via_streaming('fake/dead') == {}


def test_streaming_fallback_returns_empty_when_datasets_lib_missing(monkeypatch):
    """Local dev without the `datasets` package → graceful no-op."""
    monkeypatch.setitem(sys.modules, 'datasets', None)
    # ImportError raises when load_dataset is accessed; fallback should
    # return {} rather than blow up the import flow.
    assert _hf_features_via_streaming('fake/anything') == {}


# ---------------------------------------------------------------------------
# _hf_fetch_features: API-first, falls back to streaming
# ---------------------------------------------------------------------------


def test_hf_fetch_features_uses_api_when_available(monkeypatch):
    """When the REST API exposes features, the streaming path is never
    invoked — pin this so we don't pay an extra HF round-trip on every
    healthy repo."""
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'cardData': {'dataset_info': [{
                'features': {'x': {'_type': 'Value', 'dtype': 'int32'}},
            }]}}

    streaming_calls = []
    def fake_streaming(*a, **kw):
        streaming_calls.append(a)
        return {'fallback': 'should-not-be-called'}

    with patch('requests.get', return_value=_Resp()), \
         patch('app._hf_features_via_streaming', side_effect=fake_streaming):
        out = _hf_fetch_features('owner/repo')

    assert out == {'x': {'type': 'Value:int32'}}
    assert streaming_calls == []


def test_hf_fetch_features_falls_back_to_streaming_when_api_blank(monkeypatch):
    """API returns no dataset_info → streaming path is consulted."""
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'cardData': {}}  # no dataset_info

    expected_streaming = {'image': {'type': 'Image'}}
    with patch('requests.get', return_value=_Resp()), \
         patch('app._hf_features_via_streaming',
               return_value=expected_streaming) as streaming_mock:
        out = _hf_fetch_features('owner/no-schema-repo')

    assert out == expected_streaming
    streaming_mock.assert_called_once_with(
        'owner/no-schema-repo', revision=None, hf_token=None,
    )


def test_hf_fetch_features_returns_empty_when_both_paths_blank(monkeypatch):
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'cardData': {}}

    with patch('requests.get', return_value=_Resp()), \
         patch('app._hf_features_via_streaming', return_value={}):
        assert _hf_fetch_features('owner/dead') == {}
