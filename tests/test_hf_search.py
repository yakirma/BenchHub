"""Tests for benchhub.hf_search — HF Hub /api/datasets wrappers + cache."""
from __future__ import annotations

import urllib.request

import pytest

from benchhub import hf_search


@pytest.fixture(autouse=True)
def _drop_cache():
    """Clear the trending cache so monkeypatched fetches don't leak
    across tests."""
    hf_search._clear_cache()
    yield
    hf_search._clear_cache()


def _patch_fetch(monkeypatch, by_url):
    """Replace `urllib.request.urlopen` with a lookup against a
    `{substring_in_url: response_body_bytes}` map. Convenient for
    asserting both that we called the right endpoint AND that we
    parsed the response correctly."""
    class _FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def _fake_open(req, **kw):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        for needle, body in by_url.items():
            if needle in url:
                return _FakeResp(body)
        raise AssertionError(f"unexpected URL {url!r}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_open)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_projects_to_dropdown_shape():
    raw = {
        "id": "uoft-cs/cifar10",
        "downloads": 12345, "likes": 200, "gated": False,
        "description": "a\nmulti-line\nblob   with whitespace",
        "sha": "ignored", "createdAt": "ignored",
    }
    out = hf_search._normalize(raw)
    assert out == {
        "id": "uoft-cs/cifar10",
        "downloads": 12345, "likes": 200, "gated": False,
        "description": "a multi-line blob   with whitespace",
    }


def test_normalize_truncates_long_descriptions():
    raw = {"id": "x", "description": "x" * 1000}
    assert len(hf_search._normalize(raw)["description"]) == 200


def test_normalize_handles_missing_keys():
    out = hf_search._normalize({})
    assert out == {"id": "", "downloads": 0, "likes": 0,
                   "description": "", "gated": False}


# ---------------------------------------------------------------------------
# search_datasets
# ---------------------------------------------------------------------------

def test_search_short_circuits_on_empty_query(monkeypatch):
    """No URL fetch should happen for an empty query."""
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("should not have called urlopen")))
    assert hf_search.search_datasets("") == []
    assert hf_search.search_datasets("   ") == []


def test_search_calls_hf_api_and_normalises(monkeypatch):
    body = b'[{"id":"a/x","downloads":7,"likes":1},{"id":"b/y"}]'
    _patch_fetch(monkeypatch, {"/api/datasets?": body})
    out = hf_search.search_datasets("cifar", limit=2)
    assert [d["id"] for d in out] == ["a/x", "b/y"]
    assert out[0]["downloads"] == 7


def test_search_returns_empty_on_network_failure(monkeypatch):
    def _boom(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert hf_search.search_datasets("anything") == []


def test_search_returns_empty_on_non_json_body(monkeypatch):
    _patch_fetch(monkeypatch, {"/api/datasets?": b"<html>nope</html>"})
    assert hf_search.search_datasets("anything") == []


def test_search_returns_empty_when_response_is_not_a_list(monkeypatch):
    _patch_fetch(monkeypatch, {"/api/datasets?": b'{"error": "rate limited"}'})
    assert hf_search.search_datasets("anything") == []


# ---------------------------------------------------------------------------
# trending_by_domain — cache + per-domain dispatch
# ---------------------------------------------------------------------------

def test_trending_groups_by_domain_with_first_call_uncached(monkeypatch):
    body = b'[{"id":"foo/bar","downloads":1000}]'
    _patch_fetch(monkeypatch, {"/api/datasets?": body})
    out = hf_search.trending_by_domain(limit_per_domain=1)
    assert set(out) == {"Vision", "NLP", "Audio", "Tabular"}
    # Same fixture used for every domain → every group has the row.
    for domain, items in out.items():
        assert items[0]["id"] == "foo/bar"


def test_trending_caches_within_ttl(monkeypatch):
    """A second call within the TTL window doesn't fan out HTTP again."""
    calls = []

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return b'[{"id":"cached/ds"}]'

    def _record(req, **kw):
        calls.append(req.full_url if hasattr(req, 'full_url') else str(req))
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _record)

    first = hf_search.trending_by_domain(limit_per_domain=1)
    n_after_first = len(calls)
    second = hf_search.trending_by_domain(limit_per_domain=1)
    assert second == first
    assert len(calls) == n_after_first  # no new HTTP


def test_trending_refetches_when_cache_expired(monkeypatch):
    """Manually age the cache past TTL and confirm we re-fetch."""
    _patch_fetch(monkeypatch, {"/api/datasets?": b'[{"id":"first/ds"}]'})
    hf_search.trending_by_domain(limit_per_domain=1)

    # Rewrite cache timestamps to before-the-TTL-window.
    import time
    expired = time.time() - hf_search._TRENDING_TTL_SECONDS - 1
    for domain in list(hf_search._TRENDING_CACHE):
        ts, items = hf_search._TRENDING_CACHE[domain]
        hf_search._TRENDING_CACHE[domain] = (expired, items)

    _patch_fetch(monkeypatch, {"/api/datasets?": b'[{"id":"refreshed/ds"}]'})
    out = hf_search.trending_by_domain(limit_per_domain=1)
    for items in out.values():
        assert items[0]["id"] == "refreshed/ds"


# ---------------------------------------------------------------------------
# fetch_split_row_counts — datasets-server /size endpoint
# ---------------------------------------------------------------------------

def test_fetch_split_row_counts_extracts_per_split_num_rows(monkeypatch):
    body = b'''{
        "size": {
            "dataset": [],
            "configs": [],
            "splits": [
                {"config":"plain_text","split":"train","num_rows":50000},
                {"config":"plain_text","split":"test","num_rows":10000}
            ]
        }
    }'''
    _patch_fetch(monkeypatch, {"datasets-server.huggingface.co/size": body})
    assert hf_search.fetch_split_row_counts("uoft-cs/cifar10") == {
        "train": 50000, "test": 10000,
    }


def test_fetch_split_row_counts_skips_malformed_entries(monkeypatch):
    body = b'''{
        "size": {
            "splits": [
                {"split":"train","num_rows":100},
                {"split":"test"},
                {"num_rows":50},
                "garbage"
            ]
        }
    }'''
    _patch_fetch(monkeypatch, {"datasets-server.huggingface.co/size": body})
    assert hf_search.fetch_split_row_counts("anything") == {"train": 100}


def test_fetch_split_row_counts_returns_empty_on_network_failure(monkeypatch):
    def _boom(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert hf_search.fetch_split_row_counts("anything") == {}


def test_fetch_split_row_counts_returns_empty_on_non_dict_response(monkeypatch):
    _patch_fetch(monkeypatch, {"datasets-server.huggingface.co/size": b"[1, 2, 3]"})
    assert hf_search.fetch_split_row_counts("anything") == {}


def test_fetch_split_row_counts_empty_repo_id_short_circuits():
    """No network call should happen for a blank repo_id."""
    # If urllib is touched the test fails since we haven't patched it.
    assert hf_search.fetch_split_row_counts("") == {}


# ---------------------------------------------------------------------------
# fetch_class_label_vocabs — datasets-server /info endpoint
# ---------------------------------------------------------------------------

def test_fetch_class_label_vocabs_extracts_names(monkeypatch):
    body = b'''{
        "dataset_info": {
            "plain_text": {
                "features": {
                    "img": {"_type": "Image"},
                    "label": {
                        "_type": "ClassLabel",
                        "num_classes": 3,
                        "names": ["airplane", "automobile", "bird"]
                    }
                }
            }
        }
    }'''
    _patch_fetch(monkeypatch, {"datasets-server.huggingface.co/info": body})
    out = hf_search.fetch_class_label_vocabs("uoft-cs/cifar10")
    assert out == {"label": ["airplane", "automobile", "bird"]}


def test_fetch_class_label_vocabs_handles_multi_label_columns(monkeypatch):
    body = b'''{
        "dataset_info": {
            "default": {
                "features": {
                    "coarse_label": {"_type":"ClassLabel","names":["a","b"]},
                    "fine_label":   {"_type":"ClassLabel","names":["x","y","z"]}
                }
            }
        }
    }'''
    _patch_fetch(monkeypatch, {"datasets-server.huggingface.co/info": body})
    out = hf_search.fetch_class_label_vocabs("anything")
    assert out == {
        "coarse_label": ["a", "b"],
        "fine_label": ["x", "y", "z"],
    }


def test_fetch_class_label_vocabs_skips_non_classlabel_features(monkeypatch):
    body = b'''{
        "dataset_info": {
            "default": {
                "features": {
                    "score": {"_type":"Value","dtype":"float32"},
                    "label": {"_type":"ClassLabel","names":["a","b"]}
                }
            }
        }
    }'''
    _patch_fetch(monkeypatch, {"datasets-server.huggingface.co/info": body})
    assert hf_search.fetch_class_label_vocabs("any") == {"label": ["a", "b"]}


def test_fetch_class_label_vocabs_returns_empty_on_network_failure(monkeypatch):
    def _boom(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert hf_search.fetch_class_label_vocabs("anything") == {}


def test_fetch_class_label_vocabs_empty_repo_id_short_circuits():
    assert hf_search.fetch_class_label_vocabs("") == {}


def test_trending_per_domain_uses_distinct_filter(monkeypatch):
    """Each domain's call should carry its own task_categories filter
    on the URL — confirms we're not collapsing everything into one
    bucket on the server side."""
    calls = []

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"[]"

    def _record(req, **kw):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        calls.append(url)
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _record)
    hf_search.trending_by_domain(limit_per_domain=1)

    # Every configured domain produces a distinct URL with its filter.
    filters_in_urls = [
        "image-classification", "text-classification",
        "automatic-speech-recognition", "tabular-classification",
    ]
    for token in filters_in_urls:
        assert any(token in url for url in calls), f"missing {token!r} call"
