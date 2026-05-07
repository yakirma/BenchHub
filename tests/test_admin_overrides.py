"""Admin (BENCHHUB_ADMIN_EMAILS) bypass on @owner_required + @visibility_required."""
import pytest

from app import Dataset, User, db


@pytest.fixture
def admin_user_session(db_session, monkeypatch):
    """Stranger user who is on the admin allow-list. Returns the user row."""
    monkeypatch.setenv('BENCHHUB_ADMIN_EMAILS', 'admin@example.com')
    u = User(
        email='admin@example.com',
        display_name='Admin',
        oauth_provider='github',
        oauth_sub='admin-1',
    )
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def admin_client(client, admin_user_session):
    with client.session_transaction() as sess:
        sess['user_id'] = admin_user_session.id
    return client


def test_admin_can_delete_others_dataset(admin_client, db_session):
    """Owner is some other user; admin (allow-listed) hits POST /delete and
    succeeds — non-admins would get 403."""
    other = User(
        email='owner-x@example.com', display_name='Owner',
        oauth_provider='github', oauth_sub='ox-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(name='not_mine_to_delete', owner_user_id=other.id)
    db.session.add(ds); db.session.commit()
    ds_id = ds.id

    resp = admin_client.post(f'/dataset/{ds_id}/delete', follow_redirects=False)
    assert resp.status_code == 302
    assert Dataset.query.get(ds_id) is None


def test_non_admin_cannot_delete_others_dataset(auth_client, db_session, logged_in_user):
    """Sanity: the same gate still blocks non-admin non-owners."""
    other = User(
        email='someone-else@example.com', display_name='SE',
        oauth_provider='github', oauth_sub='se-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(name='still_not_mine', owner_user_id=other.id)
    db.session.add(ds); db.session.commit()
    ds_id = ds.id

    resp = auth_client.post(f'/dataset/{ds_id}/delete', follow_redirects=False)
    assert resp.status_code == 403
    assert Dataset.query.get(ds_id) is not None


def test_admin_can_view_others_private_dataset(admin_client, db_session):
    """Private datasets normally 404 to non-owners (don't leak existence).
    Admin gets through."""
    other = User(
        email='priv-owner@example.com', display_name='Priv',
        oauth_provider='github', oauth_sub='po-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(
        name='secret_ds', owner_user_id=other.id, visibility='private',
    )
    db.session.add(ds); db.session.commit()

    resp = admin_client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    assert b'secret_ds' in resp.data


def test_non_admin_gets_404_on_private_dataset(client, db_session):
    other = User(
        email='priv-owner-2@example.com', display_name='Priv2',
        oauth_provider='github', oauth_sub='po-2',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(
        name='hidden_ds', owner_user_id=other.id, visibility='private',
    )
    db.session.add(ds); db.session.commit()

    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 404


def test_current_user_is_admin_in_template_context(admin_client, admin_user_session, db_session):
    """The context processor exposes current_user_is_admin so templates can
    show admin-only chrome."""
    # /home is auth'd and renders base.html (which doesn't directly use the
    # flag — but a passing render proves the context processor doesn't
    # explode for an admin user).
    resp = admin_client.get('/home')
    assert resp.status_code == 200
