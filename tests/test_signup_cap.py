"""Public-launch registration cap: new sign-ups refused past the user
cap; existing users + admins are unaffected."""
from datetime import datetime, timedelta

import app as _app
from app import User, db


def _mk_user(email, sub):
    u = User(email=email, display_name=email.split('@')[0],
             oauth_provider='github', oauth_sub=sub)
    db.session.add(u); db.session.commit()
    return u


def test_signup_not_blocked_under_cap(db_session, monkeypatch):
    monkeypatch.setattr(_app, '_max_total_users', lambda: 200)
    _mk_user('a@x.io', 'a')
    assert _app._signup_blocked('new@x.io') is False


def test_signup_blocked_at_cap(db_session, monkeypatch):
    monkeypatch.setattr(_app, '_max_total_users', lambda: 2)
    _mk_user('a@x.io', 'a')
    _mk_user('b@x.io', 'b')
    assert _app._signup_blocked('c@x.io') is True


def test_admin_email_bypasses_cap(db_session, monkeypatch):
    monkeypatch.setattr(_app, '_max_total_users', lambda: 1)
    monkeypatch.setattr(_app, '_admin_emails', lambda: {'boss@x.io'})
    _mk_user('a@x.io', 'a')
    # Non-admin blocked at cap, admin allowed through.
    assert _app._signup_blocked('rando@x.io') is True
    assert _app._signup_blocked('boss@x.io') is False


def test_email_login_refuses_new_user_at_cap(client, db_session, monkeypatch):
    """The email-login verify route bounces a brand-new email to /login
    with a closed-signups flash once the cap is hit, creating no user."""
    monkeypatch.setattr(_app, '_max_total_users', lambda: 1)
    _mk_user('a@x.io', 'a')
    before = User.query.count()
    # Drive the verify route with a valid pending code for a new email.
    email = 'newcomer@x.io'
    code = '123456'
    row = _app.EmailLoginCode(
        email=email, code_hash=_app._hash_login_code(code),
        expires_at=datetime.utcnow() + timedelta(minutes=10))
    db.session.add(row); db.session.commit()
    with client.session_transaction() as s:
        s['email_login_pending'] = email
    r = client.post('/login/email/verify', data={'code': code},
                    follow_redirects=False)
    assert r.status_code == 302 and '/login' in r.headers['Location']
    assert User.query.count() == before          # no new account created
    assert User.query.filter_by(email=email).first() is None


def test_new_user_default_public_quota_is_50gb(db_session):
    u = _mk_user('quota@x.io', 'q')
    assert u.quota_public_max_bytes == 50 * 1024 ** 3
