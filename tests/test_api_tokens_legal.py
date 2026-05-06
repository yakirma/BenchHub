"""Phase 8 — API token gate on /api/* uploads + legal stub pages."""
import io

import pytest

from app import (
    Dataset,
    Project,
    User,
    db,
    generate_api_token,
)


# ===========================================================================
# Token generation + decorator
# ===========================================================================


def test_generate_api_token_returns_unique_high_entropy_string():
    a = generate_api_token()
    b = generate_api_token()
    assert a != b
    # secrets.token_urlsafe(32) → ~43 chars; never less than 32.
    assert len(a) >= 32
    # URL-safe base64 alphabet only.
    assert all(c.isalnum() or c in '-_' for c in a)


# ===========================================================================
# /api/dataset/upload — Bearer-token gate
# ===========================================================================


@pytest.fixture
def token_user(db_session):
    u = User(
        email='tok@example.com',
        display_name='Token User',
        oauth_provider='github',
        oauth_sub='tok-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def test_dataset_upload_api_rejects_anon(client, db_session):
    """Phase 8: was previously open. With the gate, anon hits 401."""
    resp = client.post('/api/dataset/upload',
                       data={'dataset_zip': (io.BytesIO(b'x'), 'x.zip')},
                       content_type='multipart/form-data')
    assert resp.status_code == 401
    assert b'API token required' in resp.data


def test_dataset_upload_api_rejects_bad_token(client, db_session):
    resp = client.post('/api/dataset/upload',
                       headers={'Authorization': 'Bearer not-a-real-token'},
                       data={'dataset_zip': (io.BytesIO(b'x'), 'x.zip')},
                       content_type='multipart/form-data')
    assert resp.status_code == 401
    assert b'Invalid API token' in resp.data


def test_dataset_upload_api_accepts_valid_token(client, db_session, token_user, make_zip):
    """Smoke: valid token gets past the auth gate. We don't assert
    success on the upload itself (process_dataset_zip needs structured
    content) — only that we *don't* hit the 401."""
    zip_path = make_zip("api_ok.zip", {
        "metric_score/s1.txt": "0.5",
    }, root_folder="api_ok")
    with open(zip_path, 'rb') as fh:
        resp = client.post('/api/dataset/upload',
                           headers={'Authorization': f'Bearer {token_user.api_token}'},
                           data={'dataset_zip': (fh, 'api_ok.zip'),
                                 'dataset_name': 'api_ok'},
                           content_type='multipart/form-data')
    assert resp.status_code != 401
    # Owner attribution: the dataset row carries the token user as owner.
    ds = Dataset.query.filter_by(name='api_ok').first()
    if ds is not None:
        assert ds.owner_user_id == token_user.id


def test_dataset_upload_api_quota_returns_429(client, db_session, token_user, make_zip):
    """Authenticated path now respects quotas — over-cap returns 429."""
    token_user.quota_max_datasets = 0
    db.session.commit()
    zip_path = make_zip("over_quota.zip", {
        "metric_score/s1.txt": "0.5",
    }, root_folder="over_quota")
    with open(zip_path, 'rb') as fh:
        resp = client.post('/api/dataset/upload',
                           headers={'Authorization': f'Bearer {token_user.api_token}'},
                           data={'dataset_zip': (fh, 'over_quota.zip'),
                                 'dataset_name': 'over_quota'},
                           content_type='multipart/form-data')
    assert resp.status_code == 429
    assert b'limit' in resp.data.lower() or b'reached' in resp.data.lower()


# ===========================================================================
# /settings/api_tokens UI
# ===========================================================================


def test_api_tokens_page_requires_login(client, db_session):
    resp = client.get('/settings/api_tokens', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_api_tokens_regenerate_creates_token(auth_client, logged_in_user, db_session):
    assert logged_in_user.api_token is None
    resp = auth_client.post('/settings/api_tokens/regenerate', follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(logged_in_user)
    assert logged_in_user.api_token is not None
    assert len(logged_in_user.api_token) >= 32


def test_api_tokens_regenerate_rotates_value(auth_client, logged_in_user, db_session):
    logged_in_user.api_token = generate_api_token()
    db.session.commit()
    old = logged_in_user.api_token

    resp = auth_client.post('/settings/api_tokens/regenerate', follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(logged_in_user)
    assert logged_in_user.api_token != old


def test_api_tokens_revoke_clears_value(auth_client, logged_in_user, db_session):
    logged_in_user.api_token = generate_api_token()
    db.session.commit()
    resp = auth_client.post('/settings/api_tokens/revoke', follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(logged_in_user)
    assert logged_in_user.api_token is None


# ===========================================================================
# Legal stubs reachable anonymously
# ===========================================================================


def test_terms_page_reachable_anon(client):
    resp = client.get('/terms')
    assert resp.status_code == 200
    assert b'Terms of Service' in resp.data


def test_privacy_page_reachable_anon(client):
    resp = client.get('/privacy')
    assert resp.status_code == 200
    assert b'Privacy Policy' in resp.data


def test_footer_links_to_legal_pages(client):
    body = client.get('/').data
    assert b'href="/terms"' in body
    assert b'href="/privacy"' in body


# ===========================================================================
# Admin gate + curate endpoint
# ===========================================================================


@pytest.fixture
def admin_user(db_session, monkeypatch):
    monkeypatch.setenv('BENCHHUB_ADMIN_EMAILS', 'admin@example.com')
    u = User(
        email='admin@example.com',
        display_name='Admin',
        oauth_provider='github',
        oauth_sub='admin-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def non_admin_user(db_session):
    u = User(
        email='nobody@example.com',
        display_name='Nobody',
        oauth_provider='github',
        oauth_sub='nobody-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def test_admin_curate_requires_admin(client, db_session, non_admin_user, monkeypatch):
    """Valid token but not on the admin allow-list → 403, not 200."""
    monkeypatch.setenv('BENCHHUB_ADMIN_EMAILS', 'admin@example.com')
    from app import Dataset, db as _db
    ds = Dataset(name='dx')
    _db.session.add(ds); _db.session.commit()

    resp = client.post(
        f'/api/admin/datasets/{ds.id}/curate',
        headers={'Authorization': f'Bearer {non_admin_user.api_token}'},
    )
    assert resp.status_code == 403


def test_admin_curate_flips_flag_and_reassigns_owner(client, db_session, admin_user):
    """Admin token + existing dataset → is_curated=True, owner = system user."""
    from app import Dataset, ensure_curated_seed, db as _db
    ensure_curated_seed()  # so the system user/project exist
    ds = Dataset(name='to_curate', visibility='public')
    _db.session.add(ds); _db.session.commit()
    ds_id = ds.id

    resp = client.post(
        f'/api/admin/datasets/{ds_id}/curate',
        headers={'Authorization': f'Bearer {admin_user.api_token}'},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['is_curated'] is True

    refreshed = Dataset.query.get(ds_id)
    assert refreshed.is_curated is True
    sys_user = User.query.filter_by(email='curated@benchhub.local').first()
    assert refreshed.owner_user_id == sys_user.id


def test_admin_curate_404_for_unknown_dataset(client, db_session, admin_user):
    resp = client.post(
        '/api/admin/datasets/9999/curate',
        headers={'Authorization': f'Bearer {admin_user.api_token}'},
    )
    assert resp.status_code == 404


def test_admin_uncurate_flips_flag_back(client, db_session, admin_user):
    from app import Dataset, db as _db
    ds = Dataset(name='already_curated', is_curated=True)
    _db.session.add(ds); _db.session.commit()

    resp = client.post(
        f'/api/admin/datasets/{ds.id}/uncurate',
        headers={'Authorization': f'Bearer {admin_user.api_token}'},
    )
    assert resp.status_code == 200
    refreshed = Dataset.query.get(ds.id)
    assert refreshed.is_curated is False


def test_admin_endpoint_no_token_returns_401(client, db_session):
    """Anon (no Authorization header) → 401 from require_api_token,
    not a 403 from require_admin. The two layers stack correctly."""
    resp = client.post('/api/admin/datasets/1/curate')
    assert resp.status_code == 401
