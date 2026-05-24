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
    # Pred fields are part of the dataset contract — declare them
    # explicitly. The picker / commit-validation no longer invents
    # `<gt>_pred` virtual entries, so tests that need a pred-side
    # binding must seed an explicit role=pred DatasetField.
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='label',       kind='label', role='gt'),
        DatasetField(dataset_id=ds.id, name='img',         kind='image', role='input'),
        DatasetField(dataset_id=ds.id, name='depth_gt',    kind='depth', role='gt'),
        DatasetField(dataset_id=ds.id, name='label_pred',  kind='label', role='pred'),
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


def test_picker_disables_metrics_that_cant_fit_the_dataset(client, db_session):
    """Unsatisfiable metrics show up in the picker but are
    disabled with a `(needs …)` hint so the user can see what's
    missing rather than just discover an empty dropdown."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    # accuracy on label×label preds — satisfiable.
    _seed_metric('accuracy_typed', kinds=['label', 'label'], roles=['gt', 'pred'], user=user)
    # FID on image×image — NOT satisfiable (no role=gt image field).
    _seed_metric('fid_typed', kinds=['image', 'image'], roles=['gt', 'pred'], user=user)

    body = client.get(f'/dataset/{ds.id}').data.decode()
    # accuracy renders as enabled.
    assert 'accuracy_typed' in body
    assert 'data-satisfiable="1"' in body
    # fid renders too, but as disabled with the (needs ...) suffix.
    assert 'fid_typed' in body
    assert 'data-satisfiable="0"' in body
    assert 'needs kind=image' in body


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
    """`pred` arg must bind to the synthesized pred-contract name
    (`label_pred`), not the GT name. The runtime engine reads
    submission-side CustomFields keyed by that name."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    gm = _seed_metric('label_acc_ok', kinds=['label', 'label'],
                      roles=['gt', 'pred'], user=user)

    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'typed_lb_ok',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': [str(gm.id)],
        'metric_mappings_json': [json.dumps({'gt': 'label', 'pred': 'label_pred'})],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='typed_lb_ok').first()
    assert lb is not None
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    assert len(lms) == 1
    assert json.loads(lms[0].arg_mappings) == {'gt': 'label', 'pred': 'label_pred'}


def test_commit_rejects_pred_arg_bound_to_gt_field(client, db_session):
    """Binding the `pred` arg to a role=gt field (like `label`
    instead of `label_pred`) must be rejected — the role mismatch
    would mean the runtime reads the wrong CustomField."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    gm = _seed_metric('label_acc_role', kinds=['label', 'label'],
                      roles=['gt', 'pred'], user=user)
    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'role_mismatch_lb',
        'dataset_ids': [str(ds.id)],
        'metric_global_id': [str(gm.id)],
        'metric_mappings_json': [json.dumps({'gt': 'label', 'pred': 'label'})],
    }, follow_redirects=False)
    lb = Leaderboard.query.filter_by(name='role_mismatch_lb').first()
    assert lb is not None
    assert LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).count() == 0


def test_picker_disables_metric_when_dataset_has_no_matching_pred_field(client, db_session):
    """If a metric has a pred-role arg of kind X and the dataset
    has no explicit role=pred field of kind X, the metric is
    unsatisfiable — picker shows it as disabled with the missing-
    field hint so the admin knows what to add to the dataset."""
    user = User(email='noPred@bench.local', display_name='nopred',
                oauth_provider='github', oauth_sub='nopred-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='no_pred_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='label',
                                kind='label', role='gt'))
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    db.session.commit()
    _login(client, user)
    _seed_metric('accuracy_strict', kinds=['label', 'label'],
                 roles=['gt', 'pred'], user=user)
    body = client.get(f'/dataset/{ds.id}').data.decode()
    assert 'accuracy_strict' in body
    # Disabled with the (needs kind=label role=pred) hint.
    assert 'data-satisfiable="0"' in body
    assert 'needs kind=label role=pred' in body


def test_picker_surfaces_explicit_pred_field_for_pred_args(client, db_session):
    """When the dataset DOES declare a role=pred field, it's the
    real source of truth for the pred-arg dropdown."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    _seed_metric('label_acc_pred', kinds=['label', 'label'],
                 roles=['gt', 'pred'], user=user)
    body = client.get(f'/dataset/{ds.id}').data.decode()
    # `label_pred` is an explicit DatasetField row → must appear.
    assert 'label_pred' in body


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


def test_create_metric_autoderives_kinds_from_signature(client, db_session):
    """A metric authored with `gt: bh.Label, pred: bh.Label` style
    annotations has its input_kinds auto-derived from the AST at
    save time — admin can leave the comma-sep field blank."""
    user, _ = _seed_user_and_dataset()
    _login(client, user)
    code = (
        "import benchhub as bh\n"
        "def acc(gt: bh.Label, pred: bh.Label):\n"
        "    assert isinstance(gt, bh.Label)\n"
        "    assert isinstance(pred, bh.Label)\n"
        "    return float(gt.value == pred.value)\n"
    )
    r = client.post('/metrics/create', data={
        'name': 'auto_typed_acc',
        'description': 'auto-typed',
        'python_code': code,
        # input_kinds / input_roles intentionally left blank — they
        # should fall back to the signature + arg-name heuristic.
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    gm = GlobalMetric.query.filter_by(name='auto_typed_acc').first()
    assert gm is not None
    assert json.loads(gm.input_kinds) == ['label', 'label']
    assert json.loads(gm.input_roles) == ['gt', 'pred']


def test_create_metric_autoderives_with_attribute_alias(client, db_session):
    """`benchhub.Depth` style annotations resolve the same as the
    short alias `bh.Depth` — we walk attribute access either way."""
    user, _ = _seed_user_and_dataset()
    _login(client, user)
    code = (
        "import benchhub\n"
        "def rmse(gt: benchhub.Depth, pred: benchhub.Depth):\n"
        "    return float(((gt.array - pred.array) ** 2).mean() ** 0.5)\n"
    )
    r = client.post('/metrics/create', data={
        'name': 'auto_typed_rmse',
        'description': '',
        'python_code': code,
    }, follow_redirects=False)
    gm = GlobalMetric.query.filter_by(name='auto_typed_rmse').first()
    assert gm is not None
    assert json.loads(gm.input_kinds) == ['depth', 'depth']
    assert json.loads(gm.input_roles) == ['gt', 'pred']


def test_create_metric_without_annotations_leaves_kinds_null(client, db_session):
    """Legacy untyped signatures stay unconstrained — NULL columns."""
    user, _ = _seed_user_and_dataset()
    _login(client, user)
    code = "def my(a, b):\n    return float(a == b)\n"
    r = client.post('/metrics/create', data={
        'name': 'untyped',
        'description': '',
        'python_code': code,
    }, follow_redirects=False)
    gm = GlobalMetric.query.filter_by(name='untyped').first()
    assert gm is not None
    assert gm.input_kinds is None
    assert gm.input_roles is None


def test_explicit_kinds_override_signature(client, db_session):
    """An explicit `input_kinds` form field wins over the
    annotation-derived value, so admins can override a misleading
    annotation without rewriting the source."""
    user, _ = _seed_user_and_dataset()
    _login(client, user)
    code = "def m(gt: int, pred: int):\n    return float(gt == pred)\n"
    r = client.post('/metrics/create', data={
        'name': 'override',
        'description': '',
        'python_code': code,
        'input_kinds': 'label, label',
        'input_roles': 'gt, pred',
    }, follow_redirects=False)
    gm = GlobalMetric.query.filter_by(name='override').first()
    assert json.loads(gm.input_kinds) == ['label', 'label']
    assert json.loads(gm.input_roles) == ['gt', 'pred']


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
