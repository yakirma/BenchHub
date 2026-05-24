"""LB-level field role overrides + cascade delete.

The dataset is role-neutral now: its `DatasetField.role` is a
default hint, but each LB can re-interpret the same field
differently. A depth column can be `gt` on a depth-estimation LB
and `input` on a colorization LB built off the same dataset.

LBs persist their overrides in `Leaderboard.field_roles_json`
(JSON dict `{field_name: role}`). The `_effective_field_role`
helper resolves the LB override first, falling back to the
dataset's declared role.

Dataset deletion now cascades to attached LBs — leaving them
attached to a deleted dataset would orphan their contract.
"""
from __future__ import annotations

import json
import os

from app import (
    Dataset,
    DatasetField,
    Leaderboard,
    User,
    _effective_field_role,
    _lb_field_roles,
    app as flask_app,
    db,
)


def _login(client, user):
    with client.session_transaction() as sess:
        sess['user_id'] = user.id


def _seed_dataset_and_user():
    user = User(email='roles@bench.local', display_name='roles',
                oauth_provider='github', oauth_sub='roles-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='roles_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='img',   kind='image', role='gt'),
        DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'),
        DatasetField(dataset_id=ds.id, name='depth', kind='depth', role='gt'),
    ])
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    db.session.commit()
    return user, ds


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------

def test_lb_field_roles_returns_empty_dict_when_unset(db_session):
    lb = Leaderboard(name='no_overrides', visibility='public')
    db.session.add(lb); db.session.commit()
    assert _lb_field_roles(lb) == {}


def test_lb_field_roles_parses_json_dict(db_session):
    lb = Leaderboard(
        name='with_overrides', visibility='public',
        field_roles_json=json.dumps({
            'img': 'input',
            'label': 'gt',
            'depth': 'skip',
        }),
    )
    db.session.add(lb); db.session.commit()
    assert _lb_field_roles(lb) == {
        'img': 'input',
        'label': 'gt',
        'depth': 'skip',
    }


def test_lb_field_roles_filters_invalid_role_values(db_session):
    """Unknown role strings are dropped so a corrupt JSON value
    can't leak into the picker."""
    lb = Leaderboard(
        name='bad_roles', visibility='public',
        field_roles_json=json.dumps({
            'img': 'input',
            'label': 'banana',  # not a valid role
        }),
    )
    db.session.add(lb); db.session.commit()
    assert _lb_field_roles(lb) == {'img': 'input'}


def test_effective_field_role_prefers_lb_override(db_session):
    _, ds = _seed_dataset_and_user()
    lb = Leaderboard(
        name='effective_lb', visibility='public',
        field_roles_json=json.dumps({'depth': 'input'}),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    depth = next(f for f in ds.fields if f.name == 'depth')
    assert _effective_field_role(lb, depth) == 'input'


def test_effective_field_role_falls_back_to_dataset_role(db_session):
    """No override → use the field's declared role."""
    _, ds = _seed_dataset_and_user()
    lb = Leaderboard(name='plain_lb', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    label = next(f for f in ds.fields if f.name == 'label')
    assert _effective_field_role(lb, label) == 'gt'


# ---------------------------------------------------------------------------
# create_leaderboard captures roles + pred entries from the form
# ---------------------------------------------------------------------------

def test_create_leaderboard_persists_field_role_overrides(client, db_session):
    user, ds = _seed_dataset_and_user()
    _login(client, user)
    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'roles_lb',
        'dataset_ids': [str(ds.id)],
        # The LB form posts one field_role_<name> per dataset field.
        'field_role_img':   'input',
        'field_role_label': 'gt',
        'field_role_depth': 'skip',
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='roles_lb').first()
    assert lb is not None
    assert _lb_field_roles(lb) == {
        'img': 'input',
        'label': 'gt',
        'depth': 'skip',
    }


def test_create_leaderboard_persists_pred_fields_from_form(client, db_session):
    user, ds = _seed_dataset_and_user()
    _login(client, user)
    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'preds_lb',
        'dataset_ids': [str(ds.id)],
        'pred_field_name':   ['label_pred', 'depth_pred'],
        'pred_field_kind':   ['label', 'depth'],
        'pred_field_params': ['', '{"unit": "meters"}'],
    }, follow_redirects=False)
    assert r.status_code in (200, 302)
    lb = Leaderboard.query.filter_by(name='preds_lb').first()
    assert lb is not None
    contract = json.loads(lb.required_pred_fields_json or '[]')
    assert {e['name'] for e in contract} == {'label_pred', 'depth_pred'}
    by_name = {e['name']: e for e in contract}
    assert by_name['depth_pred']['kind'] == 'depth'
    assert by_name['depth_pred']['params'] == {'unit': 'meters'}


def test_create_leaderboard_ignores_unknown_field_role_values(client, db_session):
    user, ds = _seed_dataset_and_user()
    _login(client, user)
    r = client.post('/create_leaderboard', data={
        'leaderboard_name': 'badroles_lb',
        'dataset_ids': [str(ds.id)],
        'field_role_img': 'bogus',  # not a valid role — dropped
        'field_role_label': 'gt',
    }, follow_redirects=False)
    lb = Leaderboard.query.filter_by(name='badroles_lb').first()
    assert _lb_field_roles(lb) == {'label': 'gt'}


# ---------------------------------------------------------------------------
# Cascade delete: dataset deletion drops attached LBs
# ---------------------------------------------------------------------------

def test_delete_dataset_cascades_to_attached_leaderboards(client, db_session):
    user, ds = _seed_dataset_and_user()
    lb_a = Leaderboard(name='a_on_ds', visibility='public', owner_user_id=user.id)
    lb_b = Leaderboard(name='b_on_ds', visibility='public', owner_user_id=user.id)
    lb_a.datasets.append(ds)
    lb_b.datasets.append(ds)
    db.session.add_all([lb_a, lb_b]); db.session.commit()
    a_id, b_id = lb_a.id, lb_b.id

    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/delete', follow_redirects=False)
    assert r.status_code in (200, 302)
    assert Dataset.query.get(ds.id) is None
    assert Leaderboard.query.get(a_id) is None
    assert Leaderboard.query.get(b_id) is None


def test_delete_dataset_with_no_attached_lbs_still_works(client, db_session):
    user, ds = _seed_dataset_and_user()
    _login(client, user)
    r = client.post(f'/dataset/{ds.id}/delete', follow_redirects=False)
    assert Dataset.query.get(ds.id) is None


def test_dataset_view_inline_lb_form_has_role_table_and_pred_section(client, db_session):
    """The LB-creation panel on /dataset/<id> renders the new
    'Field roles' table (one row per declared dataset field) +
    'Prediction fields' table + auto-pred JS scaffolding."""
    user, ds = _seed_dataset_and_user()
    _login(client, user)
    body = client.get(f'/dataset/{ds.id}').data.decode()
    # Field roles table renders one row per dataset field.
    assert 'Field roles' in body
    assert 'name="field_role_img"' in body
    assert 'name="field_role_label"' in body
    assert 'name="field_role_depth"' in body
    # Prediction fields section + auto-row template.
    assert 'Prediction fields' in body
    assert 'id="lb-pred-fields-body"' in body
    assert 'id="lb-pred-add-row"' in body
    assert 'id="lb-pred-row-template"' in body


def test_dataset_settings_warns_about_attached_lbs(client, db_session):
    """The Danger zone on /dataset/<id>/settings lists the attached
    LBs the cascade delete will remove, so the admin sees the
    blast radius before clicking."""
    user, ds = _seed_dataset_and_user()
    lb = Leaderboard(name='attached_warn_lb', visibility='public',
                     owner_user_id=user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    _login(client, user)
    body = client.get(f'/dataset/{ds.id}/settings').data.decode()
    # LB name appears in the danger-zone listing.
    assert 'attached_warn_lb' in body
    # Count is mentioned in the confirm-prompt's onsubmit JS.
    assert '1 attached leaderboard(s)' in body
