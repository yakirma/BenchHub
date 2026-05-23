"""GT Stats / Submission Stats panels are suppressed when the
underlying dataset has no scalar/metric custom fields to show.

The GT-side panel renders dataset-scope scalar/metric CustomField
values; the Submission-side panel renders the same for submission-
scope CFs. Either side disappears independently when there's
nothing to show — the dataset-view drops it from View Options,
and the comparison view skips both <th> and <td> for that side.
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


def _make_dataset_with_label_only():
    """Image input + Label gt. No scalar / metric fields anywhere."""
    ds = Dataset(name='no_stats_dataset', visibility='public')
    db.session.add(ds)
    db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(sample_id=s.id, name='label',
                               data_type='label', value_text='3'))
    db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    return ds


def _make_dataset_with_scalar_gt():
    """Image input + scalar gt. GT Stats panel must appear."""
    ds = Dataset(name='has_scalar_gt', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='score', kind='scalar', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(sample_id=s.id, name='score',
                               data_type='scalar', value_float=0.42))
    db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    return ds


def test_dataset_view_hides_gt_stats_when_no_scalar_or_metric(client, db_session):
    ds = _make_dataset_with_label_only()
    body = client.get(f'/dataset/{ds.id}').data.decode('utf-8')
    # No "GT Stats" label in View Options; no per_source_stats column header.
    assert 'GT Stats' not in body
    assert 'per_source_stats' not in body


def test_dataset_view_shows_gt_stats_when_scalar_gt_present(client, db_session):
    ds = _make_dataset_with_scalar_gt()
    body = client.get(f'/dataset/{ds.id}').data.decode('utf-8')
    # The View Options checkbox / column header lives under the
    # `per_source_stats` key — present iff there's at least one
    # scalar / metric custom field.
    assert 'per_source_stats' in body
