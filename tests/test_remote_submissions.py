"""Remote submissions: ZIP lives at https:// or hf://owner/repo/path,
BenchHub fetches via bench_cache, hash-pins on first eval.

Pin:
- `_fetch_remote_submission_zip` resolves both URL schemes, caches
  via bench_cache (origin='submission'), returns (path, sha256).
- `/api/leaderboard/<id>/submission/from_url` builds a Submission
  with storage_mode='remote', remote_url, content_hash populated.
- Hash matches what `_fetch_remote_submission_zip` returns.
- Re-fetching the same URL is a cache hit.
"""
import hashlib
import io
import json
import os
import sys
import zipfile
from unittest.mock import patch

import pytest

from app import (
    CacheEntry, Dataset, Leaderboard, Submission, db,
    _fetch_remote_submission_zip,
)


def _build_zip_bytes():
    """Same shape the local-upload tests use — two top-level entries
    so process_submission_zip doesn't auto-rename."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('metric_dummy/s00000.txt', '0.5')
        zf.writestr('README.md', 'remote-submission')
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def lb_with_token(auth_client, logged_in_user, db_session):
    from app import generate_api_token
    logged_in_user.api_token = generate_api_token()
    db.session.commit()
    ds = Dataset(name='remote_sub_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='remote_sub_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb, logged_in_user


# ---------------------------------------------------------------------------
# _fetch_remote_submission_zip — both URL schemes go through bench_cache.
# ---------------------------------------------------------------------------


def test_fetch_https_caches_and_returns_hash(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    payload = _build_zip_bytes()
    expected_hash = hashlib.sha256(payload).hexdigest()

    class _Resp:
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_content(self, chunk_size=None):
            yield payload

    with patch('requests.get', return_value=_Resp()):
        path, returned_hash = _fetch_remote_submission_zip(
            'https://example.test/remote.zip',
        )

    assert open(path, 'rb').read() == payload
    assert returned_hash == expected_hash
    # Cache row registered as origin='submission' so eviction prefers it.
    rows = CacheEntry.query.filter_by(origin='submission').all()
    assert len(rows) >= 1


def test_fetch_https_second_call_is_cache_hit(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    payload = _build_zip_bytes()

    fetches = {'n': 0}
    class _Resp:
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_content(self, chunk_size=None):
            fetches['n'] += 1
            yield payload

    with patch('requests.get', return_value=_Resp()):
        _fetch_remote_submission_zip('https://example.test/remote.zip')
        _fetch_remote_submission_zip('https://example.test/remote.zip')
    # Second call hits the cache; only one HTTP fetch.
    assert fetches['n'] == 1


def test_fetch_hf_url_uses_hf_hub_download(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    payload = _build_zip_bytes()
    fake_local = tmp_path / 'fake_hf_download.zip'
    fake_local.write_bytes(payload)

    captured = {}
    def _fake_hf_dl(**kwargs):
        captured.update(kwargs)
        return str(fake_local)

    fake_mod = type(sys)('huggingface_hub')
    fake_mod.hf_hub_download = _fake_hf_dl
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_mod)

    path, h = _fetch_remote_submission_zip(
        'hf://acme/preds/sub123.zip', hf_token='tok',
    )
    assert open(path, 'rb').read() == payload
    assert h == hashlib.sha256(payload).hexdigest()
    assert captured['repo_id'] == 'acme/preds'
    assert captured['filename'] == 'sub123.zip'
    assert captured['repo_type'] == 'dataset'
    assert captured['token'] == 'tok'


def test_fetch_hf_url_with_revision_pin(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    fake_local = tmp_path / 'rev_pinned.zip'
    fake_local.write_bytes(_build_zip_bytes())

    captured = {}
    def _fake_hf_dl(**kwargs):
        captured.update(kwargs)
        return str(fake_local)
    fake_mod = type(sys)('huggingface_hub')
    fake_mod.hf_hub_download = _fake_hf_dl
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_mod)

    _fetch_remote_submission_zip('hf://acme/preds/sub.zip@v1.2.3')
    assert captured['revision'] == 'v1.2.3'


def test_fetch_rejects_unknown_scheme(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    with pytest.raises(Exception):  # bench_cache wraps it
        _fetch_remote_submission_zip('ftp://oldschool/path.zip')


# ---------------------------------------------------------------------------
# /api/leaderboard/<id>/submission/from_url — end-to-end.
# ---------------------------------------------------------------------------


def test_from_url_endpoint_records_remote_storage_and_hash(
    client, lb_with_token, db_session, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    lb, user = lb_with_token
    payload = _build_zip_bytes()
    expected_hash = hashlib.sha256(payload).hexdigest()

    class _Resp:
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_content(self, chunk_size=None):
            yield payload

    with patch('requests.get', return_value=_Resp()):
        resp = client.post(
            f'/api/leaderboard/{lb.id}/submission/from_url',
            json={
                'url': 'https://example.test/remote.zip',
                'submission_name': 'remote_sub_v1',
                'source_colab_url': 'https://colab.research.google.com/gist/u/x',
            },
            headers={'Authorization': f'Bearer {user.api_token}'},
        )
    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert body['success'] is True
    assert body['content_hash'] == expected_hash

    sub = Submission.query.get(body['submission_id'])
    assert sub is not None
    assert sub.storage_mode == 'remote'
    assert sub.remote_url == 'https://example.test/remote.zip'
    assert sub.content_hash == expected_hash
    assert sub.source_colab_url == 'https://colab.research.google.com/gist/u/x'


def test_from_url_endpoint_400_when_url_missing(client, lb_with_token):
    lb, user = lb_with_token
    resp = client.post(
        f'/api/leaderboard/{lb.id}/submission/from_url',
        json={},
        headers={'Authorization': f'Bearer {user.api_token}'},
    )
    assert resp.status_code == 400


def test_from_url_endpoint_returns_400_on_fetch_failure(
    client, lb_with_token, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    lb, user = lb_with_token
    with patch('requests.get', side_effect=RuntimeError('network down')):
        resp = client.post(
            f'/api/leaderboard/{lb.id}/submission/from_url',
            json={'url': 'https://example.test/x.zip'},
            headers={'Authorization': f'Bearer {user.api_token}'},
        )
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'fetch failed' in body['error'].lower()


def test_from_url_endpoint_requires_api_token(client, lb_with_token):
    lb, _ = lb_with_token
    resp = client.post(
        f'/api/leaderboard/{lb.id}/submission/from_url',
        json={'url': 'https://example.test/x.zip'},
    )
    # No Authorization header → 401 from @require_api_token.
    assert resp.status_code == 401
