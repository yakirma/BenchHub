"""Post-import editor for dataset pred fields.

The HF preview's Prediction-fields section is the primary place
to declare role=pred DatasetField rows, but it requires re-importing
the dataset. The editor on /dataset/<id>/settings adds + removes
pred fields in-place — useful when realising mid-experiment that
you need a top-K LabelList pred on an existing cifar10 import.
"""
from __future__ import annotations

import json
import os

from app import (
    Dataset,
    DatasetField,
    User,
    app as flask_app,
    db,
)


def _login(client, user):
    with client.session_transaction() as sess:
        sess['user_id'] = user.id


def _seed_user_and_dataset():
    user = User(email='predfield@bench.local', display_name='predfield',
                oauth_provider='github', oauth_sub='predfield-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='predfield_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='img',   kind='image', role='input'),
        DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'),
    ])
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    db.session.commit()
    return user, ds


def test_add_pred_field_appends_a_role_pred_datasetfield(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/pred_fields/add', data={
        'name': 'label_pred',
        'kind': 'label',
        'params': '',
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    df = next(
        (f for f in ds.fields if f.name == 'label_pred' and f.role == 'pred'),
        None,
    )
    assert df is not None
    assert df.kind == 'label'


def test_add_label_list_pred_requires_k_in_params(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/pred_fields/add', data={
        'name': 'label_topk_pred',
        'kind': 'label_list',
        'params': '{}',   # missing k
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    # No field created — params validation failed.
    assert not any(f.name == 'label_topk_pred' for f in ds.fields)


def test_add_label_list_pred_with_valid_k_succeeds(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/pred_fields/add', data={
        'name': 'label_topk_pred',
        'kind': 'label_list',
        'params': '{"k": 5}',
    }, follow_redirects=False)
    df = next(
        (f for f in ds.fields if f.name == 'label_topk_pred'),
        None,
    )
    assert df is not None
    assert df.kind == 'label_list'
    assert json.loads(df.params) == {'k': 5}


def test_add_pred_field_rejects_collision_with_existing_field(client, db_session):
    """Can't shadow an existing field name (gt, input, or pred)."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/pred_fields/add', data={
        'name': 'label',  # collides with the gt field
        'kind': 'label',
        'params': '',
    }, follow_redirects=False)
    # No second `label` field added.
    assert sum(1 for f in ds.fields if f.name == 'label') == 1


def test_add_pred_field_rejects_unknown_kind(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/pred_fields/add', data={
        'name': 'whatever_pred',
        'kind': 'bogus_kind',
        'params': '',
    }, follow_redirects=False)
    assert not any(f.name == 'whatever_pred' for f in ds.fields)


def test_delete_pred_field_removes_it(client, db_session):
    user, ds = _seed_user_and_dataset()
    db.session.add(DatasetField(dataset_id=ds.id, name='label_pred',
                                kind='label', role='pred'))
    db.session.commit()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/pred_fields/label_pred/delete',
                    follow_redirects=False)
    assert not any(f.name == 'label_pred' for f in ds.fields)


def test_delete_only_targets_role_pred(client, db_session):
    """Trying to delete a gt-role field via the pred-field route is
    a no-op — the route filters by role='pred'."""
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    client.post(f'/dataset/{ds.id}/pred_fields/label/delete',
                follow_redirects=False)
    # The gt-role `label` is still there.
    assert any(f.name == 'label' and f.role == 'gt' for f in ds.fields)


def test_dataset_settings_renders_pred_field_editor(client, db_session):
    user, ds = _seed_user_and_dataset()
    _login(client, user)
    body = client.get(f'/dataset/{ds.id}/settings').data.decode()
    assert 'Pred fields' in body
    assert 'name="kind"' in body
    # The full DTYPES list is in the kind dropdown.
    assert '>label_list<' in body
    assert '>bboxes<' in body
