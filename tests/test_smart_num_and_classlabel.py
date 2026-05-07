"""Smart-render of integer-valued scalars + ClassLabel side-files."""
import os

import pytest

from app import _smart_num, _import_hf_auto, app as flask_app


# ---------------------------------------------------------------------------
# _smart_num filter
# ---------------------------------------------------------------------------


def test_smart_num_drops_dot_zero_for_ints():
    assert _smart_num(1.0) == '1'
    assert _smart_num(42.0) == '42'
    assert _smart_num(-3.0) == '-3'
    assert _smart_num(0.0) == '0'


def test_smart_num_keeps_fractions():
    assert _smart_num(1.5) == 1.5
    assert _smart_num(0.123) == 0.123


def test_smart_num_passes_through_non_numeric():
    assert _smart_num('hello') == 'hello'
    assert _smart_num(None) is None


def test_smart_num_handles_nan():
    import math
    assert _smart_num(float('nan')) != _smart_num(float('nan'))  # NaN stays NaN


def test_smart_num_registered_as_jinja_filter():
    assert 'smart_num' in flask_app.jinja_env.filters
    assert flask_app.jinja_env.filters['smart_num'] is _smart_num


# ---------------------------------------------------------------------------
# ClassLabel handling in _import_hf_auto
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_classlabel_dataset(monkeypatch):
    """Stub `datasets.load_dataset` to yield 3 rows with image + ClassLabel."""
    import sys, types
    from PIL import Image as _PILImage
    import io as _io

    rows = [
        {'image': _PILImage.new('RGB', (8, 8), (255, 0, 0)),  'label': 0},
        {'image': _PILImage.new('RGB', (8, 8), (0, 255, 0)),  'label': 1},
        {'image': _PILImage.new('RGB', (8, 8), (0, 0, 255)),  'label': 2},
    ]

    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: iter(rows)
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)


def test_classlabel_import_writes_class_name_field_and_per_sample_tags(
    client, db_session, fake_classlabel_dataset, logged_in_user,
):
    """When a column is a ClassLabel with names, the importer writes:
    - metric_<col>/<id>.txt with the integer index
    - <col>_class/<id>.txt with the human class name
    - tags/<id>.txt with the class name as a sample-level tag
    """
    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'label', 'target_kind': 'metric', 'target_field': 'metric_label'},
    ]
    features = {
        'image': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog', 'bird']},
    }
    success, message, ds_id = _import_hf_auto(
        'fake/cls', 'cls_subset', mapping,
        sample_cap=3, owner_user_id=logged_in_user.id,
        features=features,
    )
    assert success, message

    from app import Sample, CustomField, Dataset
    ds = Dataset.query.get(ds_id)
    samples = Sample.query.filter_by(dataset_id=ds.id).all()
    sample_names = sorted(s.name for s in samples)
    assert sample_names == ['s00000', 's00001', 's00002']

    # The integer label survived as the metric_label custom field.
    label_fields = {s.name: CustomField.query.filter_by(
        sample_id=s.id, name='metric_label').first() for s in samples}
    # All present and integer-valued.
    assert all(cf is not None for cf in label_fields.values())
    assert sorted(int(cf.value_float) for cf in label_fields.values()) == [0, 1, 2]

    # Class-name parallel column got created with the human-readable string.
    class_fields = {s.name: CustomField.query.filter_by(
        sample_id=s.id, name='label_class').first() for s in samples}
    assert all(cf is not None for cf in class_fields.values())
    assert {cf.value_text for cf in class_fields.values()} == {'cat', 'dog', 'bird'}

    # Per-sample tags include the class name (Sample.tags is comma-separated).
    by_name = {s.name: s for s in samples}
    # 0 → cat, 1 → dog, 2 → bird (mapped to s00000..s00002 in order).
    assert 'cat' in (by_name['s00000'].tags or '')
    assert 'dog' in (by_name['s00001'].tags or '')
    assert 'bird' in (by_name['s00002'].tags or '')


def test_classlabel_names_inferred_from_ds_features_when_api_blank(
    client, db_session, monkeypatch, logged_in_user,
):
    """Modern parquet datasets (cifar10, mnist) often don't expose
    ClassLabel.names through HF's /api/datasets endpoint. The
    `datasets` library DOES resolve them on ds.features, so we read
    from there as a fallback. Pin: empty `features` arg + ds.features
    with a ClassLabel object → class names + tags still applied."""
    import sys, types
    from PIL import Image as _PILImage

    # Fake ClassLabel that mirrors the datasets lib's API.
    class _FakeClassLabel:
        def __init__(self, names):
            self.names = names

    rows = [
        {'image': _PILImage.new('RGB', (4, 4), (255, 0, 0)), 'label': 0},
        {'image': _PILImage.new('RGB', (4, 4), (0, 255, 0)), 'label': 1},
    ]

    class _FakeIterableDataset:
        # Mimic datasets.IterableDataset: iterable + .features
        features = {
            'image': object(),  # Image() — no names
            'label': _FakeClassLabel(['airplane', 'automobile']),
        }
        def __iter__(self):
            return iter(rows)

    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: _FakeIterableDataset()
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)

    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'label', 'target_kind': 'metric', 'target_field': 'metric_label'},
    ]
    success, message, ds_id = _import_hf_auto(
        'fake/cifar', 'cifar_subset', mapping,
        sample_cap=2, owner_user_id=logged_in_user.id,
        features={},  # API returned no names
    )
    assert success, message

    from app import Sample, CustomField, Dataset
    ds = Dataset.query.get(ds_id)
    samples = Sample.query.filter_by(dataset_id=ds.id).all()
    by_name = {s.name: s for s in samples}

    # Class-name parallel column written.
    cf0 = CustomField.query.filter_by(sample_id=by_name['s00000'].id, name='label_class').first()
    cf1 = CustomField.query.filter_by(sample_id=by_name['s00001'].id, name='label_class').first()
    assert cf0 is not None and cf0.value_text == 'airplane'
    assert cf1 is not None and cf1.value_text == 'automobile'

    # Per-sample tags carry the class name.
    assert 'airplane' in (by_name['s00000'].tags or '')
    assert 'automobile' in (by_name['s00001'].tags or '')


def test_classlabel_skips_redundant_class_field_when_names_are_just_indices(
    client, db_session, monkeypatch, logged_in_user,
):
    """When ClassLabel.names is just stringified indices (['0','1','2',...]),
    the `<col>_class` side field would mirror the metric column with the
    same digit on every row — skip it. The per-sample tag still gets the
    name string so users can filter by class downstream."""
    import sys, types
    from PIL import Image as _PILImage

    rows = [
        {'image': _PILImage.new('RGB', (4, 4), (255, 0, 0)), 'label': 5},
        {'image': _PILImage.new('RGB', (4, 4), (0, 255, 0)), 'label': 7},
    ]
    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: iter(rows)
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)

    mapping = [
        {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        {'column': 'label', 'target_kind': 'metric', 'target_field': 'metric_label'},
    ]
    features = {
        'image': {'type': 'Image'},
        # Names list is just stringified integer indices (the case the
        # user hit on a CIFAR-shaped dataset where the API didn't expose
        # human class names).
        'label': {'type': 'ClassLabel',
                  'names': [str(i) for i in range(10)]},
    }
    success, message, ds_id = _import_hf_auto(
        'fake/numlabels', 'numlabels', mapping,
        sample_cap=2, owner_user_id=logged_in_user.id,
        features=features,
    )
    assert success, message

    from app import Sample, CustomField, Dataset
    ds = Dataset.query.get(ds_id)
    samples = Sample.query.filter_by(dataset_id=ds.id).all()
    # The integer label survived as the metric_label custom field …
    label_cfs = [CustomField.query.filter_by(
        sample_id=s.id, name='metric_label').first() for s in samples]
    assert all(cf is not None for cf in label_cfs)
    # … but no `label_class` side field was written for any sample.
    assert all(
        CustomField.query.filter_by(
            sample_id=s.id, name='label_class').first() is None
        for s in samples
    )
    # Per-sample tag still carries the (numeric) class name as a string.
    by_name = {s.name: s for s in samples}
    assert '5' in (by_name['s00000'].tags or '')
    assert '7' in (by_name['s00001'].tags or '')


def test_non_classlabel_metric_still_int_when_int_valued(
    client, db_session, monkeypatch, logged_in_user,
):
    """Numeric columns whose values happen to be whole numbers also
    render as ints downstream — verified via the smart_num filter."""
    import sys, types
    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: iter([{'count': 5}, {'count': 17}])
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)

    mapping = [{'column': 'count', 'target_kind': 'metric', 'target_field': 'metric_count'}]
    success, _, ds_id = _import_hf_auto(
        'fake/counts', 'count_ds', mapping,
        sample_cap=2, owner_user_id=logged_in_user.id,
        features={'count': {'type': 'Value:int64'}},
    )
    assert success
    from app import Sample, CustomField, Dataset
    ds = Dataset.query.get(ds_id)
    samples = Sample.query.filter_by(dataset_id=ds.id).all()
    cfs = [CustomField.query.filter_by(sample_id=s.id, name='metric_count').first() for s in samples]
    # Stored as float (DB column type); but smart_num renders the .0-clean form.
    rendered = sorted(_smart_num(cf.value_float) for cf in cfs)
    assert rendered == ['17', '5']
