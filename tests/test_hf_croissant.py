"""Tests for benchhub.hf_croissant — parse HF Croissant docs into a BH schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchhub.hf_croissant import (
    CroissantFetchError,
    CroissantField,
    CroissantSchema,
    _map_kind,
    _coerce_type,
    parse_croissant,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# Type mapping — the table is the contract, lock it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("croissant_type, expected_kind", [
    ("sc:ImageObject",  "image"),
    ("sc:AudioObject",  "audio"),
    ("sc:VideoObject",  "json"),     # no video kind yet
    ("sc:Text",         "text"),
    ("sc:Integer",      "scalar"),
    ("sc:Float",        "scalar"),
    ("sc:Number",       "scalar"),
    ("sc:Boolean",      "scalar"),
    ("sc:Date",         "text"),
    ("sc:URL",          "text"),
    ("cr:Audio",        "audio"),
    ("cr:Image",        "image"),
    ("cr:BoundingBox",  "bboxes"),
    ("cr:Label",        "label"),
    ("not-a-real-type", "json"),    # unknown → json fallback
])
def test_map_kind_table(croissant_type, expected_kind):
    assert _map_kind(croissant_type) == expected_kind


def test_coerce_type_picks_first_known_from_list():
    """Croissant fields can declare multiple dataTypes — we pick the
    first one we know how to map."""
    assert _coerce_type(["unknown:Thing", "sc:ImageObject"]) == "sc:ImageObject"


def test_coerce_type_empty_list_returns_empty_string():
    assert _coerce_type([]) == ""


# ---------------------------------------------------------------------------
# parse_croissant — basic shape
# ---------------------------------------------------------------------------

def test_parse_rejects_non_dataset_docs():
    with pytest.raises(ValueError, match="@type=sc:Dataset"):
        parse_croissant({"@type": "sc:Person"})


def test_parse_cifar10_extracts_image_and_label_fields():
    schema = parse_croissant(_load("croissant_cifar10.json"))
    assert isinstance(schema, CroissantSchema)
    assert schema.name == "cifar10"
    by_name = {f.name: f for f in schema.fields}
    # Per-row split indicator is metadata; parser drops it.
    assert "split" not in by_name
    assert "img" in by_name
    assert by_name["img"].kind == "image"
    assert by_name["img"].croissant_type == "sc:ImageObject"
    assert by_name["img"].source_column == "img"
    assert "label" in by_name
    # `sc:Integer` would normally map to `scalar`, but the column name
    # is `label` — the parser upgrades that to `label` so the admin
    # doesn't have to flip the dropdown by hand for every classification
    # dataset. They can still override on the preview form.
    assert by_name["label"].kind == "label"
    assert by_name["label"].croissant_type == "sc:Integer"
    assert by_name["label"].source_column == "label"


def test_parse_mnist_extracts_fields_with_id_only():
    """ylecun/mnist's Croissant doesn't set field `name` — only `@id`.
    The parser falls back to the local segment of `@id`."""
    schema = parse_croissant(_load("croissant_mnist.json"))
    names = {f.name for f in schema.fields}
    # Whatever the per-row fields end up named, the split-indicator
    # field is dropped because it references the splits enum.
    assert "split" not in names
    assert "split_name" not in names
    # Two real fields: image + label. The label is sc:Integer named
    # `label` → upgraded to kind=label by the name heuristic.
    kinds = sorted(f.kind for f in schema.fields)
    assert kinds == ["image", "label"]


def test_parse_extracts_splits_from_splits_record_set():
    schema = parse_croissant(_load("croissant_mnist.json"))
    assert set(schema.splits) == {"train", "test"}


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------

def _ds(fields):
    """Build a minimal Croissant doc with one main recordSet."""
    return {
        "@type": "sc:Dataset",
        "name": "synthetic",
        "recordSet": [{
            "@id": "rs",
            "@type": "cr:RecordSet",
            "field": fields,
        }],
    }


def test_unknown_data_type_maps_to_json():
    doc = _ds([
        {"@id": "rs/blob", "@type": "cr:Field", "dataType": "ex:WeirdShape",
         "source": {"extract": {"column": "blob"}}},
    ])
    schema = parse_croissant(doc)
    assert schema.fields[0].kind == "json"


def test_field_without_source_column_still_parsed():
    """Synthetic / computed fields may have no extract.column."""
    doc = _ds([
        {"@id": "rs/derived", "@type": "cr:Field", "dataType": "sc:Float"},
    ])
    schema = parse_croissant(doc)
    assert schema.fields[0].name == "derived"
    assert schema.fields[0].source_column is None


@pytest.mark.parametrize("colname", [
    "label", "labels", "class", "classes", "category",
    "target", "class_id", "fine_label", "coarse_label",
])
def test_integer_column_with_label_like_name_suggests_label_kind(colname):
    """`sc:Integer` named with a label-y token is suggested as
    kind=label, not scalar — so classification datasets don't force
    the admin to flip the dropdown by hand."""
    doc = _ds([
        {"@id": f"rs/{colname}", "@type": "cr:Field", "dataType": "sc:Integer",
         "source": {"extract": {"column": colname}}},
    ])
    schema = parse_croissant(doc)
    assert schema.fields[0].kind == "label"


def test_integer_column_with_arbitrary_name_stays_scalar():
    """The heuristic is name-gated: a numeric column with no label-y
    name (e.g. `pixel_count`) is left as scalar."""
    doc = _ds([
        {"@id": "rs/pixel_count", "@type": "cr:Field", "dataType": "sc:Integer",
         "source": {"extract": {"column": "pixel_count"}}},
    ])
    schema = parse_croissant(doc)
    assert schema.fields[0].kind == "scalar"


def test_label_name_heuristic_doesnt_override_non_integer_kinds():
    """If the raw type is already a structured kind (cr:Image, etc.),
    the name heuristic must NOT downgrade it to label."""
    doc = _ds([
        {"@id": "rs/label", "@type": "cr:Field", "dataType": "sc:ImageObject",
         "source": {"extract": {"column": "label"}}},
    ])
    schema = parse_croissant(doc)
    assert schema.fields[0].kind == "image"


def test_references_captured():
    doc = _ds([
        {"@id": "rs/label_id", "@type": "cr:Field", "dataType": "sc:Integer",
         "source": {"extract": {"column": "label_id"}},
         "references": {"field": {"@id": "labels/name"}}},
    ])
    schema = parse_croissant(doc)
    assert schema.fields[0].references == "labels/name"


def test_picks_record_set_with_most_extractable_columns():
    """Two recordSets — the one with column-extracts wins."""
    doc = {
        "@type": "sc:Dataset",
        "name": "twins",
        "recordSet": [
            # Decoy with no extracts.
            {"@id": "metadata", "@type": "cr:RecordSet",
             "field": [
                 {"@id": "metadata/x", "@type": "cr:Field", "dataType": "sc:Text"},
             ]},
            # Real one.
            {"@id": "main", "@type": "cr:RecordSet",
             "field": [
                 {"@id": "main/a", "@type": "cr:Field", "dataType": "sc:Float",
                  "source": {"extract": {"column": "a"}}},
                 {"@id": "main/b", "@type": "cr:Field", "dataType": "sc:Float",
                  "source": {"extract": {"column": "b"}}},
             ]},
        ],
    }
    schema = parse_croissant(doc)
    assert schema.record_set_id == "main"
    assert sorted(f.name for f in schema.fields) == ["a", "b"]


def test_dataset_with_no_record_sets_raises():
    with pytest.raises(ValueError, match="no usable recordSet"):
        parse_croissant({"@type": "sc:Dataset", "name": "empty", "recordSet": []})


# ---------------------------------------------------------------------------
# fetch_croissant — error path
# ---------------------------------------------------------------------------

def test_fetch_croissant_propagates_error_body(monkeypatch):
    """If the HF API returns `{"error": "..."}` instead of a Croissant doc,
    fetch_croissant surfaces a CroissantFetchError (not a confused parse)."""
    import urllib.request

    class _FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *a, **kw: _FakeResp(b'{"error":"private repo"}'),
    )
    from benchhub.hf_croissant import fetch_croissant
    with pytest.raises(CroissantFetchError, match="private repo"):
        fetch_croissant("private/repo")


def test_fetch_croissant_handles_non_json(monkeypatch):
    import urllib.request

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<html>not json</html>"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
    from benchhub.hf_croissant import fetch_croissant
    with pytest.raises(CroissantFetchError, match="not JSON"):
        fetch_croissant("nonsense/repo")
