"""Coverage for the per-field type + role editors on /dataset/<id>/settings.

Policy: while the dataset is unbound (no LB references it), the owner
can re-classify any field's kind or flip its role. Once any LB picks
up the dataset, those edits are locked — the LB's metric mappings
would silently re-route bytes those metrics already scored against.
"""
import json

import pytest

from app import (
    CustomField, Dataset, DatasetField, Leaderboard, Sample, User, db,
)


@pytest.fixture
def owner(db_session):
    u = User(email='ds_schema@bench.local', display_name='o',
             oauth_provider='github', oauth_sub='ds-schema-1')
    db.session.add(u); db.session.commit()
    return u


def _mk_dataset_with_field(owner, *, field_name='label', kind='label', role='gt'):
    ds = Dataset(name=f'schema_ds_{owner.id}', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(
        dataset_id=ds.id, name=field_name, kind=kind, role=role,
    ))
    sample = Sample(dataset_id=ds.id, name='s001')
    db.session.add(sample); db.session.flush()
    db.session.add(CustomField(
        sample_id=sample.id, name=field_name, data_type=kind,
        value_text='cat',
    ))
    db.session.commit()
    return ds


def test_type_edit_changes_kind_when_no_lb_uses_dataset(client, db_session, owner):
    """While the dataset is unbound, POST /field/<n>/type updates both
    DatasetField.kind and every CustomField.data_type row."""
    ds = _mk_dataset_with_field(owner, kind='scalar')
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(f'/dataset/{ds.id}/field/label/type',
                    data={'data_type': 'label'})
    assert r.status_code == 302
    df = DatasetField.query.filter_by(dataset_id=ds.id, name='label').first()
    assert df.kind == 'label'
    cf = CustomField.query.join(Sample).filter(
        Sample.dataset_id == ds.id, CustomField.name == 'label').first()
    assert cf.data_type == 'label'


def test_role_edit_changes_role_when_no_lb_uses_dataset(client, db_session, owner):
    """Same gate for role. gt → input flip is what most users want
    (default role on HF import is gt; you only know the conditioning
    inputs once you've looked at the data)."""
    ds = _mk_dataset_with_field(owner, role='gt')
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(f'/dataset/{ds.id}/field/label/role',
                    data={'role': 'input'})
    assert r.status_code == 302
    df = DatasetField.query.filter_by(dataset_id=ds.id, name='label').first()
    assert df.role == 'input'


def test_type_edit_blocked_when_lb_attached(client, db_session, owner):
    """A leaderboard binding this dataset locks the kind editor — the
    DatasetField.kind stays put."""
    ds = _mk_dataset_with_field(owner, kind='scalar')
    lb = Leaderboard(name='binding_lb', summary_metrics='',
                     owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    client.post(f'/dataset/{ds.id}/field/label/type',
                data={'data_type': 'label'})
    df = DatasetField.query.filter_by(dataset_id=ds.id, name='label').first()
    assert df.kind == 'scalar', "kind must stay put while an LB is bound"


def test_role_edit_blocked_when_lb_attached(client, db_session, owner):
    """Role editor is gated by the same rule."""
    ds = _mk_dataset_with_field(owner, role='gt')
    lb = Leaderboard(name='binding_lb_role', summary_metrics='',
                     owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    client.post(f'/dataset/{ds.id}/field/label/role',
                data={'role': 'input'})
    df = DatasetField.query.filter_by(dataset_id=ds.id, name='label').first()
    assert df.role == 'gt'


def test_type_edit_accepts_typed_registry_kinds(client, db_session, owner):
    """The old `_VALID_FIELD_TYPES` allow-list was missing
    label/label_list/bboxes/mask/audio; sourcing from DTYPES means
    all typed kinds are accepted out of the box."""
    ds = _mk_dataset_with_field(owner, kind='scalar')
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    for kind in ('label', 'label_list', 'bboxes', 'mask', 'audio'):
        r = client.post(f'/dataset/{ds.id}/field/label/type',
                        data={'data_type': kind})
        assert r.status_code == 302
        df = DatasetField.query.filter_by(dataset_id=ds.id, name='label').first()
        assert df.kind == kind, (
            f"expected kind={kind!r} after POST; got {df.kind!r}"
        )


def test_settings_page_renders_lock_banner_with_lb_names(client, db_session, owner):
    """When the dataset is bound to LBs, the settings page surfaces
    each LB name so the owner knows what to detach to unlock."""
    ds = _mk_dataset_with_field(owner)
    lb = Leaderboard(name='blocker_lb', summary_metrics='',
                     owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    body = client.get(f'/dataset/{ds.id}/settings').data.decode('utf-8')
    assert 'Locked' in body
    assert 'blocker_lb' in body


def test_settings_page_renders_role_selector_when_unbound(client, db_session, owner):
    """No LBs → role <select> appears alongside the kind one."""
    ds = _mk_dataset_with_field(owner)
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    body = client.get(f'/dataset/{ds.id}/settings').data.decode('utf-8')
    assert 'name="role"' in body
    assert 'name="data_type"' in body
