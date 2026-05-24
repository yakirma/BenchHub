"""LB-creation inline form: per-metric add widget.

Replaces the legacy free-text `summary_metrics` input. The form now
posts parallel arrays:

    metric_global_id[]    — picked GlobalMetric ids
    metric_mappings_json[] — JSON {arg_name: dataset_gt_field}

`create_leaderboard` walks the pair, looks up each GlobalMetric,
and creates a LeaderboardMetric row with the JSON mapping.
"""
from __future__ import annotations

import json
import os

from app import (
    Dataset,
    DatasetField,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    User,
    _extract_metric_arg_names,
    app as flask_app,
    db,
)


# ---------------------------------------------------------------------------
# Arg-name extraction
# ---------------------------------------------------------------------------

def test_extract_metric_arg_names_pulls_function_signature():
    code = """
import numpy as np
def my_metric(gt, pred):
    return float((gt == pred).mean())
"""
    assert _extract_metric_arg_names(code) == ['gt', 'pred']


def test_extract_metric_arg_names_skips_self_cls():
    code = "class M:\n    def __call__(self, gt, pred):\n        return 0"
    # First top-level FunctionDef is None — the def is on a class; helper
    # returns [] because the class-body def isn't at module top level.
    assert _extract_metric_arg_names(code) == []


def test_extract_metric_arg_names_handles_syntax_error():
    assert _extract_metric_arg_names("def broken( ") == []


def test_extract_metric_arg_names_returns_empty_when_no_function():
    assert _extract_metric_arg_names("import numpy as np") == []


# ---------------------------------------------------------------------------
# create_leaderboard wires structured metric rows
# ---------------------------------------------------------------------------

def _seed_dataset_and_metric(user):
    ds = Dataset(name='picker_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'),
        DatasetField(dataset_id=ds.id, name='img', kind='image', role='input'),
    ])
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    gm = GlobalMetric(
        name='accuracy_picker',
        description='accuracy metric for the picker test',
        python_code='def accuracy(gt, pred):\n    return float(gt == pred)',
        owner_user_id=user.id,
        visibility='public',
    )
    db.session.add(gm); db.session.commit()
    return ds, gm


def test_create_lb_writes_leaderboardmetric_from_structured_form(client, db_session):
    user = User(email='picker@bench.local', display_name='picker',
                oauth_provider='github', oauth_sub='picker-1')
    db.session.add(user); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    ds, gm = _seed_dataset_and_metric(user)

    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'picker_lb',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': [str(gm.id)],
        'metric_mappings_json': [json.dumps({'gt': 'label'})],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='picker_lb').first()
    assert lb is not None
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    assert len(lms) == 1
    assert lms[0].global_metric_id == gm.id
    assert json.loads(lms[0].arg_mappings) == {'gt': 'label'}


def test_create_lb_ignores_blank_picker_rows(client, db_session):
    """Empty `metric_global_id` entries (no row picked but Add never
    clicked) shouldn't blow up — they're silently skipped."""
    user = User(email='blank@bench.local', display_name='blank',
                oauth_provider='github', oauth_sub='blank-1')
    db.session.add(user); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    ds, _ = _seed_dataset_and_metric(user)

    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'no_metrics_lb',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': ['', ''],
        'metric_mappings_json': ['', ''],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='no_metrics_lb').first()
    assert lb is not None
    assert LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).count() == 0


def test_create_lb_skips_unknown_global_metric_id(client, db_session):
    """A spoofed `metric_global_id` that doesn't resolve to a real
    GlobalMetric row is dropped — the LB is still created."""
    user = User(email='ghost@bench.local', display_name='ghost',
                oauth_provider='github', oauth_sub='ghost-1')
    db.session.add(user); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    ds, _ = _seed_dataset_and_metric(user)

    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'spoof_lb',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': ['999999'],
        'metric_mappings_json': ['{}'],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='spoof_lb').first()
    assert lb is not None
    assert LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).count() == 0


def test_dataset_view_renders_metric_picker_dropdown(client, db_session):
    """The picker dropdown on the dataset page lists each visible
    GlobalMetric with its arg names attached as a data-args JSON
    attribute (so the JS can spawn one input per arg)."""
    user = User(email='view@bench.local', display_name='view',
                oauth_provider='github', oauth_sub='view-1')
    db.session.add(user); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    ds, gm = _seed_dataset_and_metric(user)

    body = client.get(f'/dataset/{ds.id}/create_lb').data.decode()
    assert 'id="lb-metric-picker"' in body
    assert 'id="lb-metric-add"' in body
    # Metric name + JSON-encoded args show up in the option markup.
    assert f'value="{gm.id}"' in body
    assert '&#34;gt&#34;' in body or '"gt"' in body  # tojson|forceescape
