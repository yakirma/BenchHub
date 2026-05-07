"""Editable field types on the dataset settings page + HF source badge."""
import json

import pytest

from app import CustomField, Dataset, Sample, User, db


@pytest.fixture
def owned_dataset_with_field(db_session, logged_in_user):
    ds = Dataset(name='editable_field_ds', owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name='s1')
    db.session.add(sample); db.session.flush()
    cf = CustomField(name='accuracy', field_type='scalar',
                     value_float=0.9, sample_id=sample.id)
    db.session.add(cf); db.session.commit()
    return ds


def test_settings_page_lists_field_types(auth_client, owned_dataset_with_field):
    resp = auth_client.get(f'/dataset/{owned_dataset_with_field.id}/settings')
    assert resp.status_code == 200
    body = resp.data
    # Field name + current type rendered.
    assert b'accuracy' in body
    assert b'scalar' in body
    # Selector is present.
    assert b'name="field_type"' in body


def test_update_field_type_reclassifies_all_rows(
    auth_client, owned_dataset_with_field, db_session,
):
    # Add a second sample with another row of the same field name.
    sample2 = Sample(dataset_id=owned_dataset_with_field.id, name='s2')
    db.session.add(sample2); db.session.flush()
    cf2 = CustomField(name='accuracy', field_type='scalar',
                      value_float=0.7, sample_id=sample2.id)
    db.session.add(cf2); db.session.commit()

    resp = auth_client.post(
        f'/dataset/{owned_dataset_with_field.id}/field/accuracy/type',
        data={'field_type': 'metric'},
        follow_redirects=False,
    )
    assert resp.status_code == 302

    # Both rows updated.
    rows = CustomField.query.filter_by(name='accuracy').all()
    assert all(r.field_type == 'metric' for r in rows)


def test_update_field_type_rejects_bogus(auth_client, owned_dataset_with_field):
    resp = auth_client.post(
        f'/dataset/{owned_dataset_with_field.id}/field/accuracy/type',
        data={'field_type': 'DROP TABLE'},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b'Invalid field type' in resp.data


def test_update_field_type_blocks_non_owner(
    auth_client, db_session,
):
    other = User(
        email='ft-other@example.com', display_name='X',
        oauth_provider='github', oauth_sub='ft-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(name='ft_blocked', owner_user_id=other.id)
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s1')
    db.session.add(s); db.session.flush()
    cf = CustomField(name='foo', field_type='scalar', value_float=1.0,
                     sample_id=s.id)
    db.session.add(cf); db.session.commit()

    resp = auth_client.post(
        f'/dataset/{ds.id}/field/foo/type', data={'field_type': 'metric'},
    )
    assert resp.status_code == 403


# --- HF source badge ----


def test_hf_source_badge_renders_on_dataset_detail(client, db_session):
    ds = Dataset(
        name='hf_badged', visibility='public',
        source_kind='hf-parquet',
        source_metadata=json.dumps({
            'repo_id': 'org/some-hf-dataset',
            'revision': 'main',
            'samples_written': 50,
        }),
    )
    db.session.add(ds); db.session.commit()

    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    body = resp.data
    assert b'org/some-hf-dataset' in body
    assert b'huggingface.co/datasets/org/some-hf-dataset' in body
    assert b'huggingface_logo' in body  # the SVG src


def test_hf_source_badge_absent_for_zip_uploads(client, db_session):
    """Datasets without HF provenance shouldn't render the badge."""
    ds = Dataset(name='vanilla_zip', visibility='public')
    db.session.add(ds); db.session.commit()
    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    assert b'huggingface_logo' not in resp.data


def test_settings_page_hf_card_renders(auth_client, logged_in_user, db_session):
    ds = Dataset(
        name='hf_settings_card', owner_user_id=logged_in_user.id,
        source_kind='hf-parquet',
        source_metadata=json.dumps({
            'repo_id': 'cool/repo', 'revision': 'v2',
            'samples_written': 200, 'sample_cap': 200,
        }),
    )
    db.session.add(ds); db.session.commit()

    resp = auth_client.get(f'/dataset/{ds.id}/settings')
    assert resp.status_code == 200
    assert b'Imported from HuggingFace' in resp.data
    assert b'cool/repo' in resp.data
    # Deep link includes the revision.
    assert b'/tree/v2' in resp.data
