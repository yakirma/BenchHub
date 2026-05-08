"""HF-token discoverability surfaces.

When a logged-in user without a saved hf_token lands on the import
flows, we want a quiet but visible "save your token" banner with a
direct link to /settings/hf_token. When they DO have one saved, the
banner stays out of the way. The from_url submission endpoint
likewise points at the settings URL on auth-shaped failures.
"""
import json
from unittest.mock import patch

import pytest

from app import (
    Dataset, Leaderboard, Submission, db,
)


# ---------------------------------------------------------------------------
# /datasets HF tab — banner shows / hides based on saved token.
# ---------------------------------------------------------------------------


def test_datasets_page_shows_token_banner_when_user_has_no_token(
    auth_client, logged_in_user, db_session,
):
    logged_in_user.hf_token = None
    db.session.commit()
    resp = auth_client.get('/datasets')
    body = resp.data.decode()
    assert 'No HuggingFace token saved' in body
    # Direct link to the settings page so the user can act in one click.
    assert '/settings/hf_token' in body


def test_datasets_page_hides_token_banner_when_token_saved(
    auth_client, logged_in_user, db_session,
):
    logged_in_user.hf_token = 'hf_test_token'
    db.session.commit()
    resp = auth_client.get('/datasets')
    body = resp.data.decode()
    # Banner gone; the manage-token link from the navbar might still
    # be there but the warning copy is the unique fingerprint.
    assert 'No HuggingFace token saved' not in body


def test_datasets_page_no_banner_for_anonymous_users(client, db_session):
    """Banner is logged-in-only — anon users see the picker but
    aren't nagged about a token (they have to sign in first anyway)."""
    resp = client.get('/datasets')
    body = resp.data.decode()
    assert 'No HuggingFace token saved' not in body


# ---------------------------------------------------------------------------
# Inline access-token field hint reflects auto-save behavior.
# ---------------------------------------------------------------------------


def test_datasets_page_inline_field_hints_auto_save(auth_client, db_session):
    resp = auth_client.get('/datasets')
    body = resp.data.decode()
    # Updated placeholder copy is the discoverable signal that the
    # form-level token will be persisted for next time.
    assert 'auto-saved' in body or 'auto-save' in body.lower()


# ---------------------------------------------------------------------------
# from_url submission endpoint — auth-shaped HF errors point at /settings.
# ---------------------------------------------------------------------------


@pytest.fixture
def lb_with_token(auth_client, logged_in_user, db_session):
    from app import generate_api_token
    logged_in_user.api_token = generate_api_token()
    db.session.commit()
    ds = Dataset(name='aff_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='aff_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    return lb, logged_in_user


def test_from_url_401_on_hf_url_includes_token_settings_link(
    client, lb_with_token, db_session, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    lb, user = lb_with_token
    with patch('app._fetch_remote_submission_zip',
               side_effect=RuntimeError('401 Client Error: Unauthorized')):
        resp = client.post(
            f'/api/leaderboard/{lb.id}/submission/from_url',
            json={'url': 'hf://gated/repo/preds.zip'},
            headers={'Authorization': f'Bearer {user.api_token}'},
        )
    assert resp.status_code == 400
    body = resp.get_json()
    assert '/settings/hf_token' in body['error']
    # Machine-readable field for clients that want to render a
    # friendlier UI than parsing the error string.
    assert '/settings/hf_token' in body['token_settings_url']


def test_from_url_non_auth_failure_message_unchanged(
    client, lb_with_token, db_session, tmp_path, monkeypatch,
):
    """Network errors / other non-auth failures still get the
    plain 'fetch failed: ...' message (no spurious settings link)."""
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    lb, user = lb_with_token
    with patch('app._fetch_remote_submission_zip',
               side_effect=RuntimeError('connection reset by peer')):
        resp = client.post(
            f'/api/leaderboard/{lb.id}/submission/from_url',
            json={'url': 'https://example.test/x.zip'},
            headers={'Authorization': f'Bearer {user.api_token}'},
        )
    assert resp.status_code == 400
    body = resp.get_json()
    assert '/settings/hf_token' not in body['error']
    # No sidecar field either.
    assert 'token_settings_url' not in body
