"""Per-user HuggingFace token: auto-save on import + manual settings page."""
from unittest.mock import patch

import pytest

from app import User, db, _resolve_hf_token


# ---------------------------------------------------------------------------
# _resolve_hf_token: form > saved > None
# ---------------------------------------------------------------------------


def test_resolve_uses_form_token_and_saves_it(auth_client, logged_in_user, db_session):
    """When the form supplies a token, _resolve_hf_token returns it AND
    writes it to user.hf_token so future requests skip the input."""
    assert logged_in_user.hf_token is None
    # Run inside an app/request context so g.current_user is wired up.
    from app import app as flask_app
    with flask_app.test_request_context('/'):
        from flask import g
        g.current_user = logged_in_user
        result = _resolve_hf_token('hf_freshly_typed')
    assert result == 'hf_freshly_typed'
    db_session.refresh(logged_in_user)
    assert logged_in_user.hf_token == 'hf_freshly_typed'


def test_resolve_falls_back_to_saved_token_when_form_blank(
    auth_client, logged_in_user, db_session,
):
    logged_in_user.hf_token = 'hf_persisted'
    db.session.commit()

    from app import app as flask_app
    with flask_app.test_request_context('/'):
        from flask import g
        g.current_user = logged_in_user
        result = _resolve_hf_token('')  # blank form value
    assert result == 'hf_persisted'


def test_resolve_returns_none_when_neither_set(auth_client, logged_in_user, db_session):
    from app import app as flask_app
    with flask_app.test_request_context('/'):
        from flask import g
        g.current_user = logged_in_user
        assert _resolve_hf_token(None) is None
        assert _resolve_hf_token('   ') is None


# ---------------------------------------------------------------------------
# /settings/hf_token UI
# ---------------------------------------------------------------------------


def test_hf_token_settings_requires_login(client):
    resp = client.get('/settings/hf_token', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_hf_token_settings_renders_save_form_when_none(auth_client, logged_in_user):
    resp = auth_client.get('/settings/hf_token')
    assert resp.status_code == 200
    body = resp.data
    assert b'No HuggingFace token on file' in body
    assert b'Save token' in body


def test_hf_token_settings_shows_masked_when_present(auth_client, logged_in_user, db_session):
    logged_in_user.hf_token = 'hf_supersecrettokenvalue'
    db.session.commit()
    resp = auth_client.get('/settings/hf_token')
    assert resp.status_code == 200
    # Mask renders prefix + suffix only.
    assert b'hf_supe' in resp.data
    assert b'alue' in resp.data
    # Full token is not echoed.
    assert b'hf_supersecrettokenvalue' not in resp.data


def test_hf_token_save_persists(auth_client, logged_in_user, db_session):
    resp = auth_client.post('/settings/hf_token/save',
                            data={'hf_token': 'hf_via_settings'},
                            follow_redirects=False)
    assert resp.status_code == 302
    db.session.refresh(logged_in_user)
    assert logged_in_user.hf_token == 'hf_via_settings'


def test_hf_token_save_rejects_blank(auth_client, logged_in_user, db_session):
    logged_in_user.hf_token = 'hf_existing'
    db.session.commit()
    resp = auth_client.post('/settings/hf_token/save',
                            data={'hf_token': '   '},
                            follow_redirects=True)
    assert resp.status_code == 200
    assert b'Empty token' in resp.data
    db.session.refresh(logged_in_user)
    # Untouched.
    assert logged_in_user.hf_token == 'hf_existing'


def test_hf_token_remove_clears(auth_client, logged_in_user, db_session):
    logged_in_user.hf_token = 'hf_about_to_go'
    db.session.commit()
    resp = auth_client.post('/settings/hf_token/remove', follow_redirects=False)
    assert resp.status_code == 302
    db.session.refresh(logged_in_user)
    assert logged_in_user.hf_token is None


# ---------------------------------------------------------------------------
# Integration: HF preview path picks up saved token
# ---------------------------------------------------------------------------


def test_preview_uses_saved_token_when_form_blank(
    auth_client, logged_in_user, db_session,
):
    """User has a token saved; preview-route POST omits hf_token; the
    fetch helper should still get called with the saved token."""
    logged_in_user.hf_token = 'hf_saved_for_user'
    db.session.commit()

    seen = {}
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'cardData': {'dataset_info': [
                {'features': {'image': {'_type': 'Image'}}}
            ]}}

    def fake_get(url, headers=None, *a, **kw):
        seen['headers'] = headers or {}
        return _R()

    with patch('requests.get', side_effect=fake_get):
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'foo/bar'},
                                follow_redirects=True)
    assert resp.status_code == 200
    # The saved token rode along on the upstream call.
    assert seen.get('headers', {}).get('Authorization') == 'Bearer hf_saved_for_user'
