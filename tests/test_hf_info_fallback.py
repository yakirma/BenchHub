"""HF-import fallback path: datasets-server /info when Croissant is absent.

Croissant covers ~most-but-not-all HF datasets (loader-script
repos, older community uploads, and anything without proper YAML
metadata don't have it). The /info endpoint covers the long tail
since it indexes anything HF can stream.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from benchhub import hf_croissant as hfc
from benchhub import hf_search as hfs


# ---------------------------------------------------------------------------
# fetch_dataset_info — /info endpoint shape
# ---------------------------------------------------------------------------

def _patch_fetch(monkeypatch, body):
    class _FakeResp:
        def __init__(self, payload): self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, **kw: _FakeResp(body),
    )


def test_fetch_dataset_info_extracts_features_and_splits(monkeypatch):
    body = json.dumps({
        "dataset_info": {
            "default": {
                "features": {
                    "img": {"_type": "Image"},
                    "label": {"_type": "ClassLabel", "names": ["a", "b"]},
                    "caption": {"_type": "Value", "dtype": "string"},
                },
                "splits": {
                    "train": {"num_examples": 50000},
                    "test": {"num_examples": 10000},
                },
            }
        }
    }).encode()
    _patch_fetch(monkeypatch, body)
    info = hfs.fetch_dataset_info("uoft-cs/cifar10")
    assert set(info["features"].keys()) == {"img", "label", "caption"}
    assert set(info["splits"]) == {"train", "test"}


def test_fetch_dataset_info_handles_list_splits(monkeypatch):
    """Some configs ship splits as a list of dicts instead of a dict."""
    body = json.dumps({
        "dataset_info": {
            "default": {
                "features": {"x": {"_type": "Value", "dtype": "int32"}},
                "splits": [{"name": "train"}, {"name": "validation"}],
            }
        }
    }).encode()
    _patch_fetch(monkeypatch, body)
    info = hfs.fetch_dataset_info("any")
    assert info["splits"] == ["train", "validation"]


def test_fetch_dataset_info_returns_none_on_network_failure(monkeypatch):
    def _boom(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert hfs.fetch_dataset_info("any") is None


def test_fetch_dataset_info_returns_none_when_no_dataset_info(monkeypatch):
    _patch_fetch(monkeypatch, b'{"size": {}}')
    assert hfs.fetch_dataset_info("any") is None


def test_fetch_dataset_info_returns_none_on_empty_repo_id():
    assert hfs.fetch_dataset_info("") is None


# ---------------------------------------------------------------------------
# schema_from_hf_features — feature dict → CroissantSchema
# ---------------------------------------------------------------------------

def test_schema_from_hf_features_maps_image_and_classlabel():
    schema = hfc.schema_from_hf_features({
        "img": {"_type": "Image"},
        "label": {"_type": "ClassLabel", "names": ["cat", "dog"]},
    }, splits=["train", "test"], name="myrepo/x")
    by_name = {f.name: f for f in schema.fields}
    assert by_name["img"].kind == "image"
    assert by_name["label"].kind == "label"
    assert schema.splits == ["train", "test"]
    assert schema.name == "myrepo/x"


def test_schema_from_hf_features_handles_value_dtypes():
    schema = hfc.schema_from_hf_features({
        "caption":     {"_type": "Value", "dtype": "string"},
        "score":       {"_type": "Value", "dtype": "float32"},
        "active":      {"_type": "Value", "dtype": "bool"},
        "pixel_count": {"_type": "Value", "dtype": "int32"},
    })
    by_name = {f.name: f for f in schema.fields}
    assert by_name["caption"].kind == "text"
    assert by_name["score"].kind == "scalar"
    assert by_name["active"].kind == "scalar"
    assert by_name["pixel_count"].kind == "scalar"


def test_schema_from_hf_features_name_heuristic_upgrades_int_to_label():
    """Same upgrade Croissant gets: a Value(int*) column called
    `label` / `target` / `class_id` / ... is treated as a
    classification target, not a free scalar."""
    schema = hfc.schema_from_hf_features({
        "label":     {"_type": "Value", "dtype": "int64"},
        "fine_label": {"_type": "Value", "dtype": "int32"},
        "target_id": {"_type": "Value", "dtype": "int64"},  # not in token set
    })
    by_name = {f.name: f for f in schema.fields}
    assert by_name["label"].kind == "label"
    assert by_name["fine_label"].kind == "label"
    assert by_name["target_id"].kind == "scalar"


def test_schema_from_hf_features_unknown_type_falls_through_to_json():
    schema = hfc.schema_from_hf_features({
        "weird": {"_type": "SomeUnknownThing"},
        "seq":   {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "int32"}},
        "tx":    {"_type": "Translation", "languages": ["en", "fr"]},
    })
    by_name = {f.name: f for f in schema.fields}
    assert by_name["weird"].kind == "json"
    assert by_name["seq"].kind == "json"
    assert by_name["tx"].kind == "json"


def test_schema_from_hf_features_drops_split_indicator_columns():
    """Per-row split column is metadata, not data — same skip
    Croissant parser does."""
    schema = hfc.schema_from_hf_features({
        "img":   {"_type": "Image"},
        "split": {"_type": "Value", "dtype": "string"},
        "split_name": {"_type": "Value", "dtype": "string"},
    })
    names = {f.name for f in schema.fields}
    assert names == {"img"}


def test_schema_carries_hf_prefix_on_croissant_type():
    """Each field's `croissant_type` is `hf:<HF _type>` so the
    preview UI can tell which path produced this schema."""
    schema = hfc.schema_from_hf_features({"img": {"_type": "Image"}})
    assert schema.fields[0].croissant_type == "hf:Image"


# ---------------------------------------------------------------------------
# Preview route falls back to /info when Croissant fails
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_client(client, db_session):
    from app import User, db
    admin = User(email='hfinfo@bench.local', display_name='hf',
                 oauth_provider='github', oauth_sub='hf-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id
    return client


def test_preview_falls_back_to_info_when_croissant_missing(admin_client, monkeypatch):
    """When fetch_croissant raises (the typical "no Croissant
    document" 404), the preview form should still render — the
    /info-derived schema feeds the rest of the flow identically."""
    def _no_croissant(repo_id, **kw):
        raise hfc.CroissantFetchError(f"no Croissant document for {repo_id!r}")
    monkeypatch.setattr(hfc, 'fetch_croissant', _no_croissant)
    monkeypatch.setattr(hfs, 'fetch_dataset_info', lambda repo_id, **kw: {
        'features': {
            'img': {'_type': 'Image'},
            'depth': {'_type': 'Array2D', 'shape': [480, 640], 'dtype': 'float32'},
        },
        'splits': ['train', 'test'],
    })
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})
    monkeypatch.setattr(hfs, 'fetch_class_label_vocabs', lambda repo_id, **kw: {})

    r = admin_client.post('/admin/import_from_hf/preview',
                          data={'repo_id': '0jl/NYUv2'})
    assert r.status_code == 200
    body = r.data.decode()
    # Schema rows show up — kind dropdowns are pre-populated.
    assert 'value="img"' in body  # field hidden input
    assert 'value="depth"' in body
    # Image kind selected on the img row.
    assert 'selected>image</option>' in body or 'value="image" selected' in body


def test_preview_redirects_when_both_sources_fail(admin_client, monkeypatch):
    """If both Croissant + /info return nothing, the preview
    redirects back with a danger flash so the admin sees the
    failure clearly."""
    def _no_croissant(repo_id, **kw):
        raise hfc.CroissantFetchError(f"no Croissant document for {repo_id!r}")
    monkeypatch.setattr(hfc, 'fetch_croissant', _no_croissant)
    monkeypatch.setattr(hfs, 'fetch_dataset_info', lambda repo_id, **kw: None)
    r = admin_client.post('/admin/import_from_hf/preview',
                          data={'repo_id': 'broken/repo'},
                          follow_redirects=False)
    assert r.status_code == 302
