"""Tests for the Phase 1 auth foundation: User model, GitHub OAuth flow,
session, current_user helper, and login_required decorator.

Authlib's HTTP layer is mocked so the suite never hits GitHub.
"""
from unittest.mock import MagicMock, patch

import pytest

from app import User, app as flask_app, db, login_required


# --- Test-only routes registered at import time ---
# Flask refuses route registration after the first request; the session-scoped
# `app` fixture handles requests across many tests, so test-local routes have
# to land at module load.

# Mounted under /api/ so the load_project_context middleware (which redirects
# unscoped paths to /projects) lets them through to our login_required gate.

@flask_app.route("/api/_test_protected")
@login_required
def _test_protected():
    return "secret"


@flask_app.route("/api/_test_protected_2")
@login_required
def _test_protected_2():
    return "secret-payload"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_github_oauth(profile, emails=None):
    """Return a context manager that fakes Authlib's OAuth client.

    profile: dict returned by GET /user
    emails:  list of dicts returned by GET /user/emails (only consulted when
             `profile['email']` is falsy)
    """
    token = {'access_token': 'fake-token'}
    profile_resp = MagicMock(json=lambda: profile)
    emails_resp = MagicMock(json=lambda: (emails or []))

    def fake_get(path, *_, **__):
        if path == 'user':
            return profile_resp
        if path == 'user/emails':
            return emails_resp
        raise AssertionError(f"Unexpected GitHub API call: {path}")

    return patch.multiple(
        'app.oauth.github',
        authorize_access_token=MagicMock(return_value=token),
        get=MagicMock(side_effect=fake_get),
    )


GITHUB_CREDS = {'GITHUB_CLIENT_ID': 'fake-id', 'GITHUB_CLIENT_SECRET': 'fake-secret'}


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


def test_user_oauth_identity_is_unique(db_session):
    db_session.add(User(
        email="a@example.com",
        display_name="A",
        oauth_provider="github",
        oauth_sub="42",
    ))
    db_session.commit()

    db_session.add(User(
        email="b@example.com",
        display_name="B",
        oauth_provider="github",
        oauth_sub="42",  # same provider+sub → must collide
    ))
    with pytest.raises(Exception):
        db_session.commit()
    db_session.rollback()


def test_user_email_is_unique(db_session):
    db_session.add(User(
        email="dup@example.com",
        display_name="A",
        oauth_provider="github",
        oauth_sub="1",
    ))
    db_session.commit()

    db_session.add(User(
        email="dup@example.com",  # collision
        display_name="B",
        oauth_provider="github",
        oauth_sub="2",
    ))
    with pytest.raises(Exception):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# /login (form)
# ---------------------------------------------------------------------------


def test_login_page_renders(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Continue with GitHub" in resp.data


def test_login_page_preserves_next_param(client):
    resp = client.get("/login?next=/some/protected/page")
    assert resp.status_code == 200
    assert b"/some/protected/page" in resp.data


# ---------------------------------------------------------------------------
# /login/github → OAuth start
# ---------------------------------------------------------------------------


def test_login_github_503_when_creds_missing(client, monkeypatch):
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("GITHUB_CLIENT_SECRET", raising=False)

    resp = client.get("/login/github")
    assert resp.status_code == 503
    assert b"GITHUB_CLIENT_ID" in resp.data


def test_login_github_redirects_to_provider(client, monkeypatch):
    for k, v in GITHUB_CREDS.items():
        monkeypatch.setenv(k, v)

    with patch('app.oauth.github.authorize_redirect') as redirect_mock:
        from flask import redirect as _flask_redirect
        redirect_mock.return_value = _flask_redirect("https://github.com/login/oauth/authorize?fake=1")
        resp = client.get("/login/github?next=/x")

    assert resp.status_code == 302
    assert "github.com/login/oauth/authorize" in resp.headers["Location"]
    redirect_mock.assert_called_once()


# ---------------------------------------------------------------------------
# In-app browser (WebView) OAuth block — Google/GitHub deny OAuth there, so the
# start routes must bounce back to /login instead of dead-ending on the provider.
# ---------------------------------------------------------------------------

_IG_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
          "AppleWebKit/605.1.15 Instagram 300.0.0.0")


def test_login_google_in_webview_redirects_back_to_login(client, monkeypatch):
    # Even with creds present, a WebView UA must NOT be sent to Google.
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"):
        monkeypatch.setenv(k, "x")
    with patch('app.oauth.google.authorize_redirect') as redirect_mock:
        resp = client.get("/login/google?next=/datasets",
                          headers={"User-Agent": _IG_UA})
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    assert "google.com" not in resp.headers["Location"]
    assert "/datasets" in resp.headers["Location"]   # next preserved
    redirect_mock.assert_not_called()                # never hit the provider


def test_login_github_in_webview_redirects_back_to_login(client, monkeypatch):
    for k, v in GITHUB_CREDS.items():
        monkeypatch.setenv(k, v)
    with patch('app.oauth.github.authorize_redirect') as redirect_mock:
        resp = client.get("/login/github", headers={"User-Agent": _IG_UA})
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    redirect_mock.assert_not_called()


def test_login_page_in_webview_leads_with_email(client):
    resp = client.get("/login", headers={"User-Agent": _IG_UA})
    assert resp.status_code == 200
    assert b"in-app browser" in resp.data        # the webview-aware banner
    assert b"Copy link" in resp.data             # open-in-browser helper


# ---------------------------------------------------------------------------
# /oauth/callback/github → user upsert + session
# ---------------------------------------------------------------------------


def test_callback_creates_new_user_and_starts_session(client):
    profile = {
        'id': 12345,
        'login': 'octocat',
        'name': 'The Octocat',
        'email': 'octo@example.com',
        'avatar_url': 'https://avatars.githubusercontent.com/u/12345',
    }
    with _patch_github_oauth(profile):
        resp = client.get("/oauth/callback/github")
    assert resp.status_code == 302  # redirected post-login

    user = User.query.filter_by(oauth_provider='github', oauth_sub='12345').first()
    assert user is not None
    assert user.email == 'octo@example.com'
    assert user.display_name == 'The Octocat'
    assert user.avatar_url.endswith('/12345')
    assert user.last_login_at is not None

    # Session cookie set; following a request shows the user is logged in.
    with client.session_transaction() as sess:
        assert sess['user_id'] == user.id


def test_callback_does_not_duplicate_existing_user(client, db_session):
    # Pre-existing user from a prior login.
    existing = User(
        email="octo@example.com",
        display_name="Old Name",
        avatar_url="https://old.example.com/avatar.png",
        oauth_provider="github",
        oauth_sub="12345",
    )
    db_session.add(existing)
    db_session.commit()
    existing_id = existing.id

    # GitHub now returns updated profile fields.
    profile = {
        'id': 12345,
        'login': 'octocat',
        'name': 'The Octocat (renamed)',
        'email': 'octo@example.com',
        'avatar_url': 'https://new.example.com/avatar.png',
    }
    with _patch_github_oauth(profile):
        client.get("/oauth/callback/github")

    db.session.expire_all()
    # Same row, refreshed denormalized fields.
    assert User.query.count() == 1
    refreshed = User.query.get(existing_id)
    assert refreshed.display_name == "The Octocat (renamed)"
    assert refreshed.avatar_url == "https://new.example.com/avatar.png"


def test_callback_falls_back_to_user_emails_when_profile_email_missing(client):
    """GitHub omits email from /user when the user has it set to private. The
    callback must consult /user/emails to find a verified primary."""
    profile = {
        'id': 99,
        'login': 'private_octo',
        'name': 'Private Octo',
        'email': None,
        'avatar_url': None,
    }
    emails = [
        {'email': 'noise@example.com', 'primary': False, 'verified': True},
        {'email': 'real@example.com', 'primary': True, 'verified': True},
    ]
    with _patch_github_oauth(profile, emails=emails):
        resp = client.get("/oauth/callback/github")
    assert resp.status_code == 302

    user = User.query.filter_by(oauth_sub='99').first()
    assert user.email == 'real@example.com'


def test_callback_redirects_to_login_when_no_email_anywhere(client):
    profile = {'id': 7, 'login': 'no_email', 'email': None}
    with _patch_github_oauth(profile, emails=[]):
        resp = client.get("/oauth/callback/github")

    # Sent back to /login with a flash; no user created.
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    assert User.query.filter_by(oauth_sub='7').count() == 0


def test_callback_handles_authlib_exchange_failure(client):
    """If GitHub returns an error during the token exchange, send the user
    back to /login rather than 500-ing."""
    with patch('app.oauth.github.authorize_access_token',
               side_effect=Exception("user denied access")):
        resp = client.get("/oauth/callback/github")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# /logout
# ---------------------------------------------------------------------------


def test_logout_clears_session(client, db_session):
    user = User(email="x@example.com", display_name="X",
                oauth_provider="github", oauth_sub="11")
    db_session.add(user)
    db_session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = user.id

    resp = client.post("/logout")
    assert resp.status_code == 302

    with client.session_transaction() as sess:
        assert 'user_id' not in sess


# ---------------------------------------------------------------------------
# current_user injection + login_required
# ---------------------------------------------------------------------------


def test_current_user_is_none_for_anonymous_request(client):
    """Hit any public endpoint and confirm g.current_user is unset."""
    # /login is public and renders, doesn't redirect.
    resp = client.get("/login")
    assert resp.status_code == 200
    # The nav widget should show "Log in" rather than a username.
    assert b"Log in" in resp.data
    assert b'class="dropdown-toggle"' not in resp.data or b"avatar" not in resp.data


def test_current_user_is_populated_when_session_user_id_set(client, db_session):
    user = User(email="logged@example.com", display_name="Logged In User",
                oauth_provider="github", oauth_sub="55")
    db_session.add(user)
    db_session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = user.id

    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Logged In User" in resp.data


def test_login_required_redirects_when_anonymous(client, db_session):
    resp = client.get("/api/_test_protected", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    assert "next=" in resp.headers["Location"]


def test_login_required_allows_when_authenticated(client, db_session):
    user = User(email="u@example.com", display_name="U",
                oauth_provider="github", oauth_sub="77")
    db_session.add(user)
    db_session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id

    resp = client.get("/api/_test_protected_2")
    assert resp.status_code == 200
    assert b"secret-payload" in resp.data


# ---------------------------------------------------------------------------
# Migration block
# ---------------------------------------------------------------------------


def test_check_and_migrate_db_creates_user_table_on_old_install(app, monkeypatch):
    """Simulate an upgrade from a pre-User schema by dropping the user table,
    then run the migration and assert it gets re-created."""
    import sqlite3
    from app import check_and_migrate_db

    db_uri = app.config["SQLALCHEMY_DATABASE_URI"]
    db_path = db_uri.replace("sqlite:///", "")

    db.session.remove()
    db.engine.dispose()

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS user")
    conn.commit()
    conn.close()

    check_and_migrate_db()

    conn = sqlite3.connect(db_path)
    rows = conn.execute("PRAGMA table_info(user)").fetchall()
    conn.close()

    cols = {row[1] for row in rows}
    assert {'id', 'email', 'display_name', 'oauth_provider', 'oauth_sub'} <= cols


# --- Passwordless email sign-in (verification code) ---

def _latest_code_hash(email):
    from app import EmailLoginCode
    row = (EmailLoginCode.query.filter_by(email=email, consumed=False)
           .order_by(EmailLoginCode.id.desc()).first())
    return row


def test_email_login_full_flow_creates_user_and_session(client, db_session):
    from app import User, EmailLoginCode
    import hashlib

    # Step 1: request a code (SMTP unconfigured → TESTING shows it in flash).
    r = client.post('/login/email', data={'email': 'New.User@Example.com'},
                    follow_redirects=False)
    assert r.status_code == 302 and '/login/email/verify' in r.headers['Location']
    row = EmailLoginCode.query.filter_by(email='new.user@example.com').first()
    assert row is not None and not row.consumed

    # Recover the plaintext code by brute-checking the flashed hash isn't
    # exposed — instead, mint our own known code path: re-derive by trying
    # the 6-digit space is silly, so read it from the dev flash instead.
    # The dev/test branch flashes "your code is NNNNNN".
    with client.session_transaction() as s:
        flashes = dict(s.get('_flashes', []))
    msg = ' '.join(flashes.values())
    import re
    m = re.search(r'code is (\d{6})', msg)
    assert m, f'code not surfaced in test mode: {msg!r}'
    code = m.group(1)
    assert hashlib.sha256(code.encode()).hexdigest() == row.code_hash

    # Step 2: verify.
    r2 = client.post('/login/email/verify', data={'code': code},
                     follow_redirects=False)
    assert r2.status_code == 302
    u = User.query.filter_by(email='new.user@example.com').first()
    assert u is not None and u.oauth_provider == 'email'
    with client.session_transaction() as s:
        assert s.get('user_id') == u.id


def test_email_login_wrong_code_rejected(client, db_session):
    client.post('/login/email', data={'email': 'a@b.com'})
    r = client.post('/login/email/verify', data={'code': '000000'},
                    follow_redirects=False)
    # Wrong code → back to verify page, no session.
    assert '/login/email/verify' in r.headers['Location']
    with client.session_transaction() as s:
        assert 'user_id' not in s


def test_email_login_invalid_email_rejected(client, db_session):
    from app import EmailLoginCode
    r = client.post('/login/email', data={'email': 'notanemail'},
                    follow_redirects=False)
    assert '/login' in r.headers['Location']
    assert EmailLoginCode.query.count() == 0


def test_email_login_links_to_existing_oauth_account(client, db_session):
    """An email that already belongs to a GitHub/Google user logs into
    THAT account rather than creating a duplicate."""
    from app import User, EmailLoginCode
    existing = User(email='dual@example.com', display_name='Dual',
                    oauth_provider='github', oauth_sub='gh-dual')
    db.session.add(existing); db.session.commit()

    client.post('/login/email', data={'email': 'dual@example.com'})
    with client.session_transaction() as s:
        import re
        code = re.search(r'code is (\d{6})', ' '.join(dict(s['_flashes']).values())).group(1)
    client.post('/login/email/verify', data={'code': code})

    assert User.query.filter_by(email='dual@example.com').count() == 1
    with client.session_transaction() as s:
        assert s.get('user_id') == existing.id


def test_send_email_tolerates_malformed_smtp_port(monkeypatch):
    """A copy-paste typo in SMTP_PORT (inline comment) must not raise an
    uncaught ValueError — it should fall back / fail soft to False."""
    from app import _send_email
    monkeypatch.setenv('SMTP_HOST', 'smtp.invalid.localhost')
    monkeypatch.setenv('SMTP_PORT', '465  # implicit SSL')   # the exact bad paste
    monkeypatch.setenv('MAIL_FROM', 'no-reply@runbenchhub.com  # comment')
    # Unreachable host → send fails, but the function must return False
    # (caught), never propagate an exception.
    assert _send_email('x@example.com', 'subj', 'body') is False


def test_send_email_unconfigured_returns_false(monkeypatch):
    from app import _send_email
    monkeypatch.delenv('SMTP_HOST', raising=False)
    assert _send_email('x@example.com', 'subj', 'body') is False


def test_email_login_notifies_on_bounce(client, db_session, monkeypatch):
    """When Resend reports the address bounced (doesn't exist / rejected),
    the user is sent back to /login with a clear notice — not to the
    code-entry page — and the code is invalidated."""
    import app as _app
    from app import EmailLoginCode
    monkeypatch.setattr(_app, '_resend_send', lambda *a, **k: 'fake-id')
    monkeypatch.setattr(_app, '_resend_wait_event', lambda *a, **k: 'bounced')

    r = client.post('/login/email', data={'email': 'ghost@nope.example'},
                    follow_redirects=False)
    assert r.status_code == 302
    assert '/login/email/verify' not in r.headers['Location']  # NOT the code page
    with client.session_transaction() as s:
        assert 'user_id' not in s
        flashes = ' '.join(v for _, v in s.get('_flashes', []))
    assert "couldn't deliver" in flashes and 'ghost@nope.example' in flashes
    # Code invalidated so it can't be used later.
    row = EmailLoginCode.query.filter_by(email='ghost@nope.example').first()
    assert row is not None and row.consumed is True


def test_email_login_proceeds_on_delivered(client, db_session, monkeypatch):
    """A delivered (non-bounce) send takes the user to the verify page."""
    import app as _app
    monkeypatch.setattr(_app, '_resend_send', lambda *a, **k: 'fake-id')
    monkeypatch.setattr(_app, '_resend_wait_event', lambda *a, **k: 'delivered')

    r = client.post('/login/email', data={'email': 'real@example.com'},
                    follow_redirects=False)
    assert '/login/email/verify' in r.headers['Location']
