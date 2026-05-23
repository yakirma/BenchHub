"""Dataset-view's Filter-by-field control.

Filters samples by the value of a chosen text / label / scalar /
metric custom field. Text-ish kinds use a case-insensitive
substring match; numeric kinds accept a bare number or an
operator-prefixed form (`>1.5`, `<=10`, `!=0`).
"""
from __future__ import annotations

import os

from app import (
    CustomField,
    Dataset,
    DatasetField,
    Sample,
    app as flask_app,
    db,
)


def _make_label_dataset():
    """Five samples, label values 0 / 1 / 2 / 0 / 1 with a 3-class vocab."""
    ds = Dataset(name='filter_label_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    df = DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt')
    df.set_params({'names': ['cat', 'dog', 'fish']})
    db.session.add(df)
    for i, lbl in enumerate([0, 1, 2, 0, 1]):
        s = Sample(dataset_id=ds.id, name=f's{i}')
        db.session.add(s); db.session.flush()
        db.session.add(CustomField(sample_id=s.id, name='label',
                                   data_type='label', value_text=str(lbl)))
    db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    return ds


def _make_scalar_dataset():
    """Five samples, scalar `score` ∈ {0.1, 0.4, 0.5, 0.7, 0.9}."""
    ds = Dataset(name='filter_scalar_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='score', kind='scalar', role='gt'))
    for i, val in enumerate([0.1, 0.4, 0.5, 0.7, 0.9]):
        s = Sample(dataset_id=ds.id, name=f's{i}')
        db.session.add(s); db.session.flush()
        db.session.add(CustomField(sample_id=s.id, name='score',
                                   data_type='scalar', value_float=val))
    db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    return ds


def test_filter_by_label_substring(client, db_session):
    ds = _make_label_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=label&filter_value=0').data.decode()
    # Only the two label=0 rows (s0, s3) survive — s1/s2/s4 don't.
    assert 'data-sample-name="s0"' in body
    assert 'data-sample-name="s3"' in body
    for n in ('s1', 's2', 's4'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_by_scalar_greater_than(client, db_session):
    ds = _make_scalar_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=score&filter_value=>0.5').data.decode()
    # 0.7 and 0.9 survive; 0.5 is excluded by strict >.
    assert 'data-sample-name="s3"' in body
    assert 'data-sample-name="s4"' in body
    for n in ('s0', 's1', 's2'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_by_scalar_lte(client, db_session):
    ds = _make_scalar_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=score&filter_value=<=0.4').data.decode()
    assert 'data-sample-name="s0"' in body
    assert 'data-sample-name="s1"' in body
    for n in ('s2', 's3', 's4'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_by_scalar_exact_equal(client, db_session):
    ds = _make_scalar_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=score&filter_value=0.5').data.decode()
    assert 'data-sample-name="s2"' in body
    for n in ('s0', 's1', 's3', 's4'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_by_label_class_name_translates_to_index(client, db_session):
    """Typing the class name (`cat`, `dog`) into the filter must
    look up the matching index in the vocab and constrain by it."""
    ds = _make_label_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=label&filter_value=cat').data.decode()
    # label index 0 → "cat" → s0 and s3.
    assert 'data-sample-name="s0"' in body
    assert 'data-sample-name="s3"' in body
    for n in ('s1', 's2', 's4'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_by_label_index_plus_name_string(client, db_session):
    """The cell render format `<idx> <name>` (e.g. `1 dog`) is also
    accepted as a filter value — same lookup as the bare name."""
    ds = _make_label_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=label&filter_value=1+dog').data.decode()
    # label index 1 → "dog" → s1 and s4.
    assert 'data-sample-name="s1"' in body
    assert 'data-sample-name="s4"' in body
    for n in ('s0', 's2', 's3'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_by_sample_name(client, db_session):
    """The synthetic `sample_name` filter does a case-insensitive
    substring match on Sample.name."""
    ds = _make_scalar_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=sample_name&filter_value=s3').data.decode()
    assert 'data-sample-name="s3"' in body
    for n in ('s0', 's1', 's2', 's4'):
        assert f'data-sample-name="{n}"' not in body


def test_filter_value_suggestions_for_label_field(client, db_session):
    """When the filter_field is a label with a vocab, the
    <datalist> for the value input enumerates `<idx> <name>` entries
    so the browser can autocomplete what the user types."""
    ds = _make_label_dataset()
    body = client.get(f'/dataset/{ds.id}?filter_field=label').data.decode()
    assert '<datalist id="filter-value-suggestions">' in body
    assert '<option value="0 cat">' in body
    assert '<option value="1 dog">' in body
    assert '<option value="2 fish">' in body


def test_filter_dropdown_includes_sample_name(client, db_session):
    """`sample_name` is a synthetic filterable field surfaced at
    the top of the dropdown so the admin can substring-filter by
    sample name without touching the URL by hand."""
    ds = _make_scalar_dataset()
    body = client.get(f'/dataset/{ds.id}').data.decode()
    assert 'sample_name (sample_name)' in body


def test_filter_dropdown_lists_text_scalar_label_fields(client, db_session):
    """The Filter-by dropdown must include text / label / scalar /
    metric fields and skip image / mask / depth / audio etc."""
    ds = Dataset(name='dropdown_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='img', kind='image', role='input'),
        DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'),
        DatasetField(dataset_id=ds.id, name='score', kind='scalar', role='gt'),
        DatasetField(dataset_id=ds.id, name='caption', kind='text', role='gt'),
    ])
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add_all([
        CustomField(sample_id=s.id, name='img', data_type='image', value_text='img/s0.png'),
        CustomField(sample_id=s.id, name='label', data_type='label', value_text='0'),
        CustomField(sample_id=s.id, name='score', data_type='scalar', value_float=0.5),
        CustomField(sample_id=s.id, name='caption', data_type='text', value_text='a cat'),
    ])
    db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)

    body = client.get(f'/dataset/{ds.id}').data.decode()
    # Filter dropdown surfaces label / scalar / text. img must NOT
    # appear inside the filter_field_select.
    assert 'name="filter_field"' in body
    # The control labels show "<name> (<kind>)"; check for the kind suffixes.
    assert 'label (label)' in body
    assert 'score (scalar)' in body
    assert 'caption (text)' in body
    # img is not filterable — must not show up as a filter_field option.
    assert 'img (image)' not in body
