"""Pointer-mode HF dataset import + the bench_cache-backed resolver
that loads image / depth GT lazily during metric eval.

The big-picture contract:
- `_import_hf_pointer` writes Sample rows with `source_ref_json` and
  CustomField rows with `source_column` set + value_text NULL for
  image / depth (bytes never touch the volume).
- Inline metadata (scalar values, text, ClassLabel sidecars, tags)
  IS stored on the CustomField rows so the dataset page renders
  without any HF round-trip.
- `_pointer_gt_resolver(sample, cf)` fetches + caches the missing
  bytes on demand. Returns numpy arrays.
"""
import io
import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

import bench_cache
from app import (
    CacheEntry, CustomField, Dataset, Sample, db,
    _import_hf_pointer, _pointer_gt_resolver,
)


# ---------------------------------------------------------------------------
# Fake `datasets` module — yields a tiny fixed set of rows.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hf_pointer_dataset(monkeypatch):
    """Two-row dataset shaped like a depth benchmark: one RGB image
    column + one depth NPZ-equivalent column (numpy array) + one
    ClassLabel column."""

    class _ClassLabel:
        def __init__(self, names):
            self.names = names

    rows = [
        {
            'image': Image.new('RGB', (4, 4), (10, 20, 30)),
            'depth_map': np.full((4, 4), 1.5, dtype=np.float32),
            'label': 0,
        },
        {
            'image': Image.new('RGB', (4, 4), (40, 50, 60)),
            'depth_map': np.full((4, 4), 2.5, dtype=np.float32),
            'label': 1,
        },
    ]

    class _IterableDS:
        features = {
            'image': object(),
            'depth_map': object(),
            'label': _ClassLabel(['cat', 'dog']),
        }
        def __iter__(self):
            return iter(rows)

    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: _IterableDS()
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)
    return rows


@pytest.fixture
def hf_token_unset(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    # Block the upstream HF metadata round-trip in `_auto_tags_for_hf`
    # so the importer doesn't actually hit the network.
    import requests
    monkeypatch.setattr('requests.get', lambda *a, **kw: type('R', (), {
        'raise_for_status': lambda self: None,
        'json': lambda self: {'tags': [], 'description': ''},
    })())


# ---------------------------------------------------------------------------
# _import_hf_pointer — what lands on disk vs in the DB.
# ---------------------------------------------------------------------------


def test_pointer_import_writes_no_image_or_depth_bytes(
    db_session, tmp_path, fake_hf_pointer_dataset, hf_token_unset, logged_in_user,
    monkeypatch,
):
    """The whole point of pointer mode: image + depth columns produce
    CustomField rows with NULL value_text. Bytes stay on HF."""
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
        {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
    ]
    features = {
        'image': {'type': 'Image'},
        'depth_map': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog']},
    }
    ok, msg, ds_id = _import_hf_pointer(
        'fake/depth-bench', 'pointer_ds', mapping,
        sample_cap=2, owner_user_id=logged_in_user.id, features=features,
    )
    assert ok, msg

    ds = Dataset.query.get(ds_id)
    assert ds.storage_mode == 'hf-pointer'
    assert ds.source_kind == 'hf-pointer'

    samples = Sample.query.filter_by(dataset_id=ds.id).order_by(Sample.name).all()
    assert [s.name for s in samples] == ['s00000', 's00001']

    # Each sample carries source_ref_json with the row index.
    refs = [json.loads(s.source_ref_json) for s in samples]
    assert refs[0]['row_idx'] == 0 and refs[0]['repo_id'] == 'fake/depth-bench'
    assert refs[1]['row_idx'] == 1

    # Image + depth CustomFields exist with value_text NULL +
    # source_column populated.
    for s in samples:
        img_cf = CustomField.query.filter_by(
            sample_id=s.id, name='image_image',
        ).first()
        depth_cf = CustomField.query.filter_by(
            sample_id=s.id, name='raw_depth_map',
        ).first()
        assert img_cf is not None
        assert img_cf.field_type == 'image'
        assert img_cf.value_text is None       # ← bytes NOT cloned
        assert img_cf.source_column == 'image'
        assert depth_cf is not None
        assert depth_cf.field_type == 'depth'
        assert depth_cf.value_text is None
        assert depth_cf.source_column == 'depth_map'


def test_pointer_import_keeps_classlabel_sidecars_inline(
    db_session, tmp_path, fake_hf_pointer_dataset, hf_token_unset, logged_in_user,
):
    """The ClassLabel scalar value, the human class-name sidecar, and
    the per-sample tag string all live inline on the DB so dataset
    page renders are HF-fetch-free."""
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
        {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
    ]
    features = {
        'image': {'type': 'Image'}, 'depth_map': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog']},
    }
    _, _, ds_id = _import_hf_pointer(
        'fake/cls', 'pointer_cls', mapping, sample_cap=2,
        owner_user_id=logged_in_user.id, features=features,
    )
    samples = Sample.query.filter_by(dataset_id=ds_id).order_by(Sample.name).all()

    # Scalar + class-name sidecar inline (value present, source_column
    # set so the engine could re-resolve from HF if it wanted to).
    label_0 = CustomField.query.filter_by(
        sample_id=samples[0].id, name='label').first()
    assert label_0.value_float == 0.0
    assert label_0.source_column == 'label'
    cn_0 = CustomField.query.filter_by(
        sample_id=samples[0].id, name='label_class').first()
    assert cn_0 is not None and cn_0.value_text == 'cat'
    cn_1 = CustomField.query.filter_by(
        sample_id=samples[1].id, name='label_class').first()
    assert cn_1.value_text == 'dog'

    # Per-sample tag string includes the class name.
    assert 'cat' in (samples[0].tags or '')
    assert 'dog' in (samples[1].tags or '')


def test_pointer_import_refuses_duplicate_dataset_name(
    db_session, fake_hf_pointer_dataset, hf_token_unset, logged_in_user,
):
    db.session.add(Dataset(name='dup_ptr', visibility='public'))
    db.session.commit()
    ok, msg, ds_id = _import_hf_pointer(
        'fake/x', 'dup_ptr', [], sample_cap=1,
        owner_user_id=logged_in_user.id,
    )
    assert not ok
    assert ds_id is None
    assert 'already exists' in msg.lower()


# ---------------------------------------------------------------------------
# _pointer_gt_resolver — lazy fetch + cache + read-back.
# ---------------------------------------------------------------------------


def test_pointer_resolver_fetches_image_and_caches(
    client, db_session, tmp_path, fake_hf_pointer_dataset,
    hf_token_unset, logged_in_user, monkeypatch,
):
    """First call: cache miss → writer runs → bytes land in cache.
    Second call: cache hit → writer DOES NOT run."""
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
        {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
    ]
    features = {
        'image': {'type': 'Image'}, 'depth_map': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog']},
    }
    _, _, ds_id = _import_hf_pointer(
        'fake/depth-bench', 'resolver_ds', mapping, sample_cap=2,
        owner_user_id=logged_in_user.id, features=features,
    )
    sample = Sample.query.filter_by(
        dataset_id=ds_id, name='s00000').first()
    img_cf = CustomField.query.filter_by(
        sample_id=sample.id, name='image_image').first()

    # Count fetches by patching `datasets.load_dataset` with a wrapper
    # that increments a counter.
    fetches = {'n': 0}
    real_load = sys.modules['datasets'].load_dataset
    def _counting_load(*a, **kw):
        fetches['n'] += 1
        return real_load(*a, **kw)
    sys.modules['datasets'].load_dataset = _counting_load

    arr1 = _pointer_gt_resolver(sample, img_cf)
    arr2 = _pointer_gt_resolver(sample, img_cf)

    sys.modules['datasets'].load_dataset = real_load

    assert arr1 is not None and arr1.shape == (4, 4, 3)
    np.testing.assert_array_equal(arr1, arr2)
    # Exactly one HF round-trip — second call hit the cache.
    assert fetches['n'] == 1
    assert CacheEntry.query.filter_by(origin='gt').count() >= 1


def test_pointer_resolver_fetches_depth_array(
    client, db_session, tmp_path, fake_hf_pointer_dataset,
    hf_token_unset, logged_in_user, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
        {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
    ]
    features = {
        'image': {'type': 'Image'}, 'depth_map': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog']},
    }
    _, _, ds_id = _import_hf_pointer(
        'fake/depth-bench', 'resolver_depth_ds', mapping, sample_cap=2,
        owner_user_id=logged_in_user.id, features=features,
    )
    sample = Sample.query.filter_by(
        dataset_id=ds_id, name='s00001').first()
    depth_cf = CustomField.query.filter_by(
        sample_id=sample.id, name='raw_depth_map').first()

    arr = _pointer_gt_resolver(sample, depth_cf)
    assert arr is not None
    assert arr.shape == (4, 4)
    # row 1 of the fixture is full of 2.5; pointer fetch should agree.
    assert float(arr.mean()) == pytest.approx(2.5)


def test_pointer_resolver_returns_none_when_not_pointer_mode(db_session):
    """Local-mode samples (no source_ref_json) return None — caller
    falls back to the on-disk loader."""
    ds = Dataset(name='not_pointer', visibility='public', storage_mode='local')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0')  # no source_ref_json
    db.session.add(s); db.session.flush()
    cf = CustomField(sample_id=s.id, name='img', field_type='image',
                     value_text=None, source_column='img')
    db.session.add(cf); db.session.commit()
    assert _pointer_gt_resolver(s, cf) is None


def test_pointer_resolver_returns_none_without_source_column(
    db_session, fake_hf_pointer_dataset, hf_token_unset, logged_in_user,
):
    ds = Dataset(name='no_sc', visibility='public', storage_mode='hf-pointer')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0',
               source_ref_json=json.dumps({
                   'repo_id': 'fake/x', 'revision': 'main',
                   'split': 'train', 'row_idx': 0,
               }))
    db.session.add(s); db.session.flush()
    cf = CustomField(sample_id=s.id, name='img', field_type='image',
                     value_text=None, source_column=None)
    db.session.add(cf); db.session.commit()
    assert _pointer_gt_resolver(s, cf) is None


# ---------------------------------------------------------------------------
# get_metric_context end-to-end with the resolver wired in.
# ---------------------------------------------------------------------------


def test_get_metric_context_uses_pointer_resolver_for_image_and_depth(
    client, db_session, tmp_path, fake_hf_pointer_dataset,
    hf_token_unset, logged_in_user, monkeypatch,
):
    """Pin the integration: get_metric_context, when handed the
    resolver, populates `gt_image_image` + `gt_raw_depth_map` from the
    streamed HF row instead of returning None."""
    from metric_engine import get_metric_context
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
        {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
    ]
    features = {
        'image': {'type': 'Image'}, 'depth_map': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog']},
    }
    _, _, ds_id = _import_hf_pointer(
        'fake/depth-bench', 'ctx_pointer_ds', mapping, sample_cap=2,
        owner_user_id=logged_in_user.id, features=features,
    )
    sample = Sample.query.filter_by(
        dataset_id=ds_id, name='s00000').first()

    ctx = get_metric_context(sample, pointer_resolver=_pointer_gt_resolver)
    assert ctx['gt_image_image'] is not None
    assert ctx['gt_image_image'].shape == (4, 4, 3)
    assert ctx['gt_raw_depth_map'] is not None
    assert ctx['gt_raw_depth_map'].shape == (4, 4)
    # Inline scalar still flows through too.
    assert ctx['gt_label'] == 0.0


def test_get_metric_context_without_resolver_leaves_gt_arrays_none(
    client, db_session, tmp_path, fake_hf_pointer_dataset,
    hf_token_unset, logged_in_user, monkeypatch,
):
    """Resolver is opt-in. If a caller forgets to pass one, the
    pointer-mode samples surface as None in the context — explicit
    failure mode rather than a silent crash on missing files."""
    from metric_engine import get_metric_context
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
    ]
    features = {'image': {'type': 'Image'}, 'depth_map': {'type': 'Image'}}
    _, _, ds_id = _import_hf_pointer(
        'fake/depth-bench', 'ctx_no_resolver_ds', mapping, sample_cap=1,
        owner_user_id=logged_in_user.id, features=features,
    )
    sample = Sample.query.filter_by(
        dataset_id=ds_id, name='s00000').first()
    ctx = get_metric_context(sample)  # no pointer_resolver
    assert ctx.get('gt_image_image') is None
    assert ctx.get('gt_raw_depth_map') is None
