"""HuggingFace dataset picker (Round C scrape endpoint)."""
from unittest.mock import patch

import pytest


def test_api_hf_datasets_is_public(client):
    """The picker just relays public HF data; no auth required.
    Was @login_required before — anonymous users got a 302 to /login
    that the JS fetch couldn't parse, surfacing as 'Failed to load'."""
    with patch('requests.get') as mock:
        mock.return_value.raise_for_status = lambda: None
        mock.return_value.json = lambda: []
        resp = client.get('/api/hf/datasets', follow_redirects=False)
    assert resp.status_code == 200
    assert resp.headers.get('Content-Type', '').startswith('application/json')


def test_api_hf_datasets_returns_normalized_rows(auth_client):
    """Mock requests.get to return a stub HF API payload; the endpoint
    should reshape it into the picker's row format."""
    fake_payload = [
        {'id': 'org/some-dataset', 'downloads': 1234, 'likes': 56,
         'lastModified': '2026-01-02', 'tags': ['task:depth-estimation', 'language:en']},
        {'id': 'no-download-counts', 'downloads': None, 'likes': None,
         'lastModified': None, 'tags': None},
        {},  # missing id → dropped
    ]

    class _Resp:
        def __init__(self, data): self._data = data
        def raise_for_status(self): pass
        def json(self): return self._data

    with patch('app.requests.get' if False else 'requests.get',
               return_value=_Resp(fake_payload)) as get_mock:
        # Bust the cache by tweaking sort+q
        resp = auth_client.get('/api/hf/datasets?sort=downloads&q=zzz')
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['sort'] == 'downloads'
        assert body['q'] == 'zzz'
        assert len(body['rows']) == 2  # missing-id row dropped
        first = body['rows'][0]
        assert first['id'] == 'org/some-dataset'
        assert first['downloads'] == 1234
        assert first['likes'] == 56
        assert 'task:depth-estimation' in first['tags']
        # Robust to missing fields
        second = body['rows'][1]
        assert second['downloads'] == 0
        assert second['likes'] == 0
        assert second['tags'] == []


def test_api_hf_datasets_invalid_sort_falls_back(auth_client):
    """Anything not in {likes, downloads, trending} → 'likes'."""
    class _Resp:
        def raise_for_status(self): pass
        def json(self): return []
    with patch('requests.get', return_value=_Resp()):
        resp = auth_client.get('/api/hf/datasets?sort=DROP+TABLE')
    assert resp.status_code == 200
    assert resp.get_json()['sort'] == 'likes'


def test_api_hf_datasets_returns_empty_on_network_error(auth_client):
    """Any exception from the upstream call returns an empty list (UI
    degrades to manual entry) instead of 500ing the page."""
    with patch('requests.get', side_effect=RuntimeError("upstream down")):
        # Use a fresh sort+q to bypass the in-process cache.
        resp = auth_client.get('/api/hf/datasets?sort=trending&q=netfail')
    assert resp.status_code == 200
    assert resp.get_json()['rows'] == []
