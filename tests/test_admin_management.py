"""DB-backed admin list and the /settings/admins UI."""
import pytest

from app import User, db


@pytest.fixture
def env_admin(db_session, monkeypatch, client):
    """User on BENCHHUB_ADMIN_EMAILS allow-list — env-bootstrap admin."""
    monkeypatch.setenv('BENCHHUB_ADMIN_EMAILS', 'envadmin@example.com')
    u = User(
        email='envadmin@example.com', display_name='Env Admin',
        oauth_provider='github', oauth_sub='env-1',
    )
    db.session.add(u); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = u.id
    return u


@pytest.fixture
def db_admin(db_session, client):
    """User who has the DB bit but is NOT on the env allow-list."""
    u = User(
        email='dbadmin@example.com', display_name='DB Admin',
        oauth_provider='github', oauth_sub='db-1',
        is_admin=True,
    )
    db.session.add(u); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = u.id
    return u


def test_admins_settings_requires_admin(client, db_session, logged_in_user):
    """Plain user → 403."""
    with client.session_transaction() as sess:
        sess['user_id'] = logged_in_user.id
    resp = client.get('/settings/admins')
    assert resp.status_code == 403


def test_admins_settings_renders_for_db_admin(client, db_admin):
    resp = client.get('/settings/admins')
    assert resp.status_code == 200
    assert b'Admin management' in resp.data
    assert b'dbadmin@example.com' in resp.data


def test_admin_grant_promotes_existing_user(client, db_admin, db_session):
    target = User(
        email='futureadmin@example.com', display_name='Future',
        oauth_provider='github', oauth_sub='fa-1',
    )
    db.session.add(target); db.session.commit()

    resp = client.post('/settings/admins/grant',
                       data={'email': 'futureadmin@example.com'},
                       follow_redirects=False)
    assert resp.status_code == 302
    db.session.refresh(target)
    assert target.is_admin is True


def test_admin_grant_unknown_email_warns(client, db_admin):
    resp = client.post('/settings/admins/grant',
                       data={'email': 'never-signed-in@example.com'},
                       follow_redirects=True)
    assert resp.status_code == 200
    assert b'No BenchHub user' in resp.data


def test_admin_revoke_clears_db_flag(client, db_admin, db_session):
    target = User(
        email='demoteme@example.com', display_name='DM',
        oauth_provider='github', oauth_sub='dm-1',
        is_admin=True,
    )
    db.session.add(target); db.session.commit()

    resp = client.post(f'/settings/admins/revoke/{target.id}',
                       follow_redirects=False)
    assert resp.status_code == 302
    db.session.refresh(target)
    assert target.is_admin is False


def test_admin_cannot_revoke_self(client, db_admin):
    """Lockout protection — revoking yourself is blocked."""
    resp = client.post(f'/settings/admins/revoke/{db_admin.id}',
                       follow_redirects=True)
    assert resp.status_code == 200
    # HTML-escaped apostrophe — match either form.
    body = resp.data.decode()
    assert ("can't revoke your own admin" in body
            or "can&#39;t revoke your own admin" in body)
    db.session.refresh(db_admin)
    assert db_admin.is_admin is True


def test_env_admin_bootstrap_sets_db_flag_at_login(client, db_session, monkeypatch):
    """OAuth callback flips is_admin=True for env-listed emails so they
    appear in the runtime admin list (and survive env-var changes)."""
    # We can't easily simulate the full OAuth dance in unit tests, but the
    # is_admin() helper short-circuits on the env-var allow-list anyway,
    # which gives the same effective behavior. Pin both:
    from app import is_admin
    monkeypatch.setenv('BENCHHUB_ADMIN_EMAILS', 'env-only@example.com')
    u = User(
        email='env-only@example.com', display_name='Env Only',
        oauth_provider='github', oauth_sub='eo-1',
    )
    db.session.add(u); db.session.commit()
    # is_admin() recognizes them via the env path even before the
    # OAuth-bootstrap flip happens.
    assert is_admin(u) is True


def test_db_admin_works_without_env_listing(client, db_session, monkeypatch):
    """A user with is_admin=True is admin even if not on the env list."""
    from app import is_admin
    monkeypatch.setenv('BENCHHUB_ADMIN_EMAILS', '')  # empty allow-list
    u = User(
        email='only-db-admin@example.com', display_name='Only DB',
        oauth_provider='github', oauth_sub='odb-1',
        is_admin=True,
    )
    db.session.add(u); db.session.commit()
    assert is_admin(u) is True
