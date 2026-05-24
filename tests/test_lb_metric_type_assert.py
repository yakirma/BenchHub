"""Type/role assertion on metrics during LB creation.

A GlobalMetric can declare:
  - `input_kinds`  — parallel JSON array of BH kinds per arg
  - `input_roles`  — parallel JSON array of dataset roles per arg

The LB-creation picker on /dataset/<id> filters metrics to those
whose declared kinds (and roles) can be satisfied by the attached
dataset, and the per-arg field dropdowns are filtered to fields
matching the expected (kind, role) pair. On commit the server
re-asserts: bindings that don't match are rejected and the
remaining metrics get created normally.
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
    app as flask_app,
    db,
)


def _seed_user_and_dataset():
    user = User(email='typed@bench.local', display_name='typed',
                oauth_provider='github', oauth_sub='typed-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='typed_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='label',  kind='label',  role='gt'),
        DatasetField(dataset_id=ds.id, name='img',    kind='image',  role='input'),
        DatasetField(dataset_id=ds.id, name='depth_gt', kind='depth', role='gt'),
    ])
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    db.session.commit()
    return user, ds


def _seed_metric(name, *, kinds, roles, code='def f(gt, pred):\n    return 0\n', user=None):
    gm = GlobalMetric(
        name=name,
        description=name,
        python_code=code,
        owner_user_id=(user.id if user else None),
        visibility='public',
        input_kinds=(json.dumps(kinds) if kinds is not None else None),
        input_roles=(json.dumps(roles) if roles is not None else None),
    )
    db.session.add(gm); db.session.commit()
    return gm


def _login(client, user):
    with client.session_transaction() as sess:
        sess['user_id'] = user.id


def test_picker_hides_metrics_that_cant_fit_the_dataset(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    # accuracy on label×label preds — satisfiable: dataset has a label gt.
    _seed_metric('accuracy_typed', kinds=['label', 'label'], roles=['gt', 'pred'], user=user)
    # FID on image×image — NOT satisfiable: dataset has an `img` input
    # but no role=gt image field.
    _seed_metric('fid_typed', kinds=['image', 'image'], roles=['gt', 'pred'], user=user)

    body = client.get(f'/dataset/{ds.id}').data.decode()
    assert 'accuracy_typed' in body
    assert 'fid_typed' not in body


def test_picker_shows_metric_when_kind_matches_via_input_role(client, db_session):
    """Metric arg with role=input must find a same-kind role=input
    field on the dataset. Dataset has `img` as image input → a
    metric expecting image+input qualifies."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    _seed_metric('image_check', kinds=['image'], roles=['input'],
                 code='def f(img):\n    return 0\n', user=user)
    body = client.get(f'/dataset/{ds.id}').data.decode()
    assert 'image_check' in body


def test_commit_rejects_wrong_kind_binding(client, db_session):
    """A metric expecting `label` for its gt arg can't bind to the
    image input field. The server drops that metric (with a flash)
    while still creating the LB."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    gm = _seed_metric('label_acc', kinds=['label', 'label'],
                      roles=['gt', 'pred'], user=user)

    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'typed_lb_bad',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': [str(gm.id)],
        # Map `gt` to `img` — wrong kind (image not label) AND wrong
        # role (input not gt).
        'metric_mappings_json': [json.dumps({'gt': 'img'})],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='typed_lb_bad').first()
    assert lb is not None
    # The bad binding gets dropped → no LeaderboardMetric row written.
    assert LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).count() == 0


def test_commit_keeps_correct_binding(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    gm = _seed_metric('label_acc_ok', kinds=['label', 'label'],
                      roles=['gt', 'pred'], user=user)

    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'typed_lb_ok',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': [str(gm.id)],
        'metric_mappings_json': [json.dumps({'gt': 'label', 'pred': 'label'})],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='typed_lb_ok').first()
    assert lb is not None
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    assert len(lms) == 1
    assert json.loads(lms[0].arg_mappings) == {'gt': 'label', 'pred': 'label'}


def test_unconstrained_metric_falls_through_to_legacy_behaviour(client, db_session):
    """A metric with NULL input_kinds + NULL input_roles is the
    legacy contract — the picker still shows it, the per-arg
    dropdown lists every GT field, and any binding is accepted."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    gm = _seed_metric('legacy_metric', kinds=None, roles=None, user=user)
    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'legacy_lb',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': [str(gm.id)],
        'metric_mappings_json': [json.dumps({'gt': 'label'})],
    }, follow_redirects=False)
    lb = Leaderboard.query.filter_by(name='legacy_lb').first()
    assert lb is not None
    assert LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).count() == 1


def test_edit_global_metric_writes_input_kinds_and_roles(client, db_session):
    user, _ = _seed_user_and_dataset()
    _login(client, user)
    gm = _seed_metric('to_edit', kinds=None, roles=None, user=user)
    r = client.post(f'/metrics/{gm.id}/edit', data={
        'name': gm.name,
        'description': 'updated',
        'python_code': 'def to_edit(gt, pred):\n    return 0\n',
        'input_kinds': 'label, label',
        'input_roles': 'gt, pred',
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    db.session.refresh(gm)
    assert json.loads(gm.input_kinds) == ['label', 'label']
    assert json.loads(gm.input_roles) == ['gt', 'pred']
