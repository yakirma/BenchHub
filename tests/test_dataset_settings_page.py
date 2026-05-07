"""Dataset settings page (Danger zone + Sharing live here now)."""
import pytest

from app import Dataset, User, db


def test_settings_page_requires_login(client, db_session):
    ds = Dataset(name='settings_anon')
    db.session.add(ds); db.session.commit()
    resp = client.get(f'/dataset/{ds.id}/settings', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_settings_page_blocks_non_owner(auth_client, logged_in_user, db_session):
    other = User(
        email='other-set@example.com', display_name='X',
        oauth_provider='github', oauth_sub='os-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(name='not_yours_settings', owner_user_id=other.id)
    db.session.add(ds); db.session.commit()
    resp = auth_client.get(f'/dataset/{ds.id}/settings')
    assert resp.status_code == 403


def test_settings_page_renders_for_owner_with_danger_zone(
    auth_client, logged_in_user, db_session,
):
    ds = Dataset(name='owned_settings', owner_user_id=logged_in_user.id,
                 visibility='private')
    db.session.add(ds); db.session.commit()

    resp = auth_client.get(f'/dataset/{ds.id}/settings')
    assert resp.status_code == 200
    body = resp.data
    assert b'Danger zone' in body
    assert b'Delete this dataset' in body
    # Sharing card visible because not public.
    assert b'Sharing' in body


def test_settings_page_hides_sharing_when_public(auth_client, logged_in_user, db_session):
    ds = Dataset(name='public_settings', owner_user_id=logged_in_user.id,
                 visibility='public')
    db.session.add(ds); db.session.commit()
    resp = auth_client.get(f'/dataset/{ds.id}/settings')
    assert resp.status_code == 200
    # Danger zone always shown to owner; Sharing hidden for public.
    assert b'Danger zone' in resp.data
    assert b'i bi-people' not in resp.data  # icon used by sharing card


def test_dataset_view_links_to_settings_for_owner(
    auth_client, logged_in_user, db_session,
):
    ds = Dataset(name='linked_settings', owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.commit()
    resp = auth_client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    assert f'/dataset/{ds.id}/settings'.encode() in resp.data
