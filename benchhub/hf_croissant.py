"""Parse Croissant JSON-LD into a BenchHub-friendly schema.

Every HuggingFace dataset repo exposes a machine-readable
Croissant document at
`https://huggingface.co/api/datasets/<repo>/croissant`. The document
declares field types, file references, and splits — everything we
need to drop the old name-token heuristics that the deleted HF
importer relied on. What it does NOT declare is **roles** (input
vs GT) or per-instance params (depth unit, label vocab, bbox
format); those come from the admin via the preview form.

Public API:

    fetch_croissant(repo_id) -> dict        # raw JSON-LD via HF API
    parse_croissant(doc)     -> CroissantSchema

`CroissantSchema.fields` is a list of `CroissantField`s with
deterministic kind mapping (no name-token guessing). The admin form
can override `kind` (e.g. flip an `image` to `mask` for a
segmentation column) and supply `role` + `params` per field.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


_CROISSANT_URL_TEMPLATE = "https://huggingface.co/api/datasets/{repo_id}/croissant"


# Croissant dataType (or its trailing segment) → BH kind. Built from
# the MLCommons + schema.org vocab Croissant uses. Anything outside
# this table lands as `json` and the admin can override on the
# preview form.
_TYPE_MAP: dict[str, str] = {
    # schema.org
    "sc:ImageObject": "image",
    "sc:AudioObject": "audio",
    "sc:VideoObject": "json",      # no video kind yet
    "sc:Text": "text",
    "sc:Integer": "scalar",
    "sc:Float": "scalar",
    "sc:Number": "scalar",
    "sc:Boolean": "scalar",
    "sc:Date": "text",
    "sc:DateTime": "text",
    "sc:URL": "text",
    # Croissant-native
    "cr:Audio": "audio",
    "cr:Image": "image",
    "cr:BoundingBox": "bboxes",
    "cr:Label": "label",
}


# Column names that strongly imply a classification-label field
# rather than an arbitrary integer. Used to bump sc:Integer columns
# from `scalar` to `label` in the suggested kind on the preview form.
# HF's Croissant export doesn't expose ClassLabel.names — that only
# shows up at materialize time — so we lean on the column name here.
_LABEL_NAME_TOKENS: set[str] = {
    "label", "labels",
    "class", "classes",
    "category", "categories",
    "target", "target_class",
    "class_id", "class_label", "classlabel",
    "coarse_label", "fine_label",
}


@dataclass
class CroissantField:
    """One column in the dataset."""

    name: str
    kind: str                       # BH kind: image, mask, depth, ..., or json
    croissant_type: str             # raw `dataType` string for transparency
    source_column: str | None = None    # name of the column in the parquet shard
    references: str | None = None       # `@id` of another field this references


@dataclass
class CroissantSchema:
    """Parsed Croissant document, distilled to what the importer needs."""

    name: str                       # dataset name
    description: str
    record_set_id: str              # which recordSet these fields came from
    fields: list[CroissantField] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class CroissantFetchError(Exception):
    """Raised when the Croissant endpoint returns no usable JSON-LD."""


def fetch_croissant(repo_id: str, *, timeout: int = 20) -> dict:
    """GET the Croissant document for an HF dataset repo.

    Returns the parsed JSON dict on success. Raises CroissantFetchError
    on HTTP non-200, network failure, or when the response is the API's
    `{"error": "..."}` shape instead of a Croissant document.
    """
    url = _CROISSANT_URL_TEMPLATE.format(repo_id=repo_id)
    req = urllib.request.Request(url, headers={"User-Agent": "benchhub/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise CroissantFetchError(
            f"HTTP {e.code} fetching Croissant for {repo_id!r}: {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise CroissantFetchError(
            f"network error fetching Croissant for {repo_id!r}: {e}"
        ) from e
    try:
        doc = json.loads(body)
    except json.JSONDecodeError as e:
        raise CroissantFetchError(
            f"Croissant response for {repo_id!r} was not JSON: {e}"
        ) from e
    if not isinstance(doc, dict) or doc.get("@type") != "sc:Dataset":
        # HF returns `{"error": "..."}` for repos that don't have a
        # Croissant export (gated, private, or non-conformant). Surface
        # that as a clean error rather than a confused parse later.
        err = doc.get("error") if isinstance(doc, dict) else None
        raise CroissantFetchError(
            f"no Croissant document for {repo_id!r}"
            + (f" (server said: {err})" if err else "")
        )
    return doc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _local_name(qid: str | None, prefix: str | None = None) -> str:
    """Strip the recordSet-prefix from an `@id`. Croissant docs vary
    in whether they fill `name` or rely on `@id` like `<rs>/<field>`."""
    if not qid:
        return ""
    if prefix and qid.startswith(prefix + "/"):
        return qid[len(prefix) + 1 :]
    if "/" in qid:
        return qid.rsplit("/", 1)[-1]
    return qid


def _coerce_type(raw: Any) -> str:
    """Croissant occasionally lists multiple types (e.g. ["sc:ImageObject",
    "cr:Image"]); pick the first that we know how to map."""
    if isinstance(raw, list):
        for candidate in raw:
            if isinstance(candidate, str) and candidate in _TYPE_MAP:
                return candidate
        return raw[0] if raw and isinstance(raw[0], str) else ""
    return raw if isinstance(raw, str) else ""


def _map_kind(croissant_type: str) -> str:
    """Croissant dataType → BH kind. Anything we don't know maps to `json`."""
    return _TYPE_MAP.get(croissant_type, "json")


def _record_sets(doc: dict) -> list[dict]:
    rs = doc.get("recordSet") or []
    return [r for r in rs if isinstance(r, dict)]


def _is_splits_record_set(rs: dict) -> bool:
    """Heuristic-free: a recordSet is the splits enum when it inlines
    `data` and every field has a `split` / `split_name` local name."""
    if not rs.get("data"):
        return False
    fields = rs.get("field") or []
    if not fields:
        return False
    rs_id = rs.get("@id") or ""
    locals_ = [_local_name(f.get("@id"), prefix=rs_id) for f in fields]
    return all(name in {"split", "split_name", "name"} for name in locals_)


def _split_names_from(rs: dict) -> list[str]:
    """Pull the list of split names out of a splits-enum recordSet's
    inline `data`."""
    rows = rs.get("data") or []
    rs_id = rs.get("@id") or ""
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if not isinstance(v, str):
                continue
            local = _local_name(k, prefix=rs_id)
            if local in {"split_name", "split", "name"}:
                out.append(v)
                break
    return out


def _pick_main_record_set(record_sets: list[dict]) -> dict | None:
    """Choose the recordSet that holds the actual columns (not the
    splits enum). Heuristic-free: the one with the most fields whose
    `source.extract.column` is set."""
    best: dict | None = None
    best_count = -1
    for rs in record_sets:
        if _is_splits_record_set(rs):
            continue
        fields = rs.get("field") or []
        n = sum(
            1
            for f in fields
            if isinstance(f.get("source"), dict)
            and isinstance(f["source"].get("extract"), dict)
            and "column" in f["source"]["extract"]
        )
        if n > best_count:
            best = rs
            best_count = n
    return best


def _source_column(field_dict: dict) -> str | None:
    src = field_dict.get("source")
    if not isinstance(src, dict):
        return None
    extract = src.get("extract")
    if not isinstance(extract, dict):
        return None
    col = extract.get("column")
    return col if isinstance(col, str) else None


def _references_id(field_dict: dict) -> str | None:
    refs = field_dict.get("references")
    if not isinstance(refs, dict):
        return None
    f = refs.get("field")
    if isinstance(f, dict):
        return f.get("@id")
    if isinstance(f, str):
        return f
    return None


def parse_croissant(doc: dict) -> CroissantSchema:
    """Parse a Croissant JSON-LD doc into a BH-friendly schema."""
    if doc.get("@type") != "sc:Dataset":
        raise ValueError("not a Croissant Dataset document (missing @type=sc:Dataset)")

    name = doc.get("name") or ""
    description = doc.get("description") or ""
    record_sets = _record_sets(doc)
    main = _pick_main_record_set(record_sets)
    if main is None:
        raise ValueError("no usable recordSet found in Croissant document")

    rs_id = main.get("@id") or main.get("name") or ""
    fields: list[CroissantField] = []
    for f in main.get("field") or []:
        if not isinstance(f, dict):
            continue
        raw_type = _coerce_type(f.get("dataType"))
        kind = _map_kind(raw_type)
        # Croissant docs are inconsistent: some set `name` to the bare
        # field name, others to the full `<recordSet>/<field>` form.
        # Strip the prefix either way so we always get the local segment.
        local = _local_name(
            f.get("name") or f.get("@id"),
            prefix=rs_id,
        )
        # Name-based upgrade for classification labels: HF's Croissant
        # export types ClassLabel columns as `sc:Integer`, which we
        # otherwise map to `scalar`. If the column name strongly
        # implies a class label (`label`, `class`, `target`, ...),
        # suggest `label` instead so the admin doesn't have to flip
        # the dropdown by hand. They can still override on the form.
        if kind == "scalar" and local and local.lower() in _LABEL_NAME_TOKENS:
            kind = "label"
        # When the field references a splits/labels enum AND its raw
        # type is text/integer, treat it as a label by default — the
        # admin can override to scalar if the reference is just a split
        # indicator (which we should skip anyway).
        ref = _references_id(f)
        # Skip the per-row split indicator; it's metadata, not data.
        rs_prefix = rs_id + "/" if rs_id else ""
        if local in {"split", "split_name"} and ref:
            continue
        if not local:
            continue
        fields.append(
            CroissantField(
                name=local,
                kind=kind,
                croissant_type=raw_type,
                source_column=_source_column(f),
                references=ref,
            )
        )

    splits: list[str] = []
    for rs in record_sets:
        if _is_splits_record_set(rs):
            splits.extend(_split_names_from(rs))
    # De-dupe but keep first-seen order.
    seen: dict[str, None] = {}
    splits = [s for s in splits if not seen.setdefault(s, None)]

    return CroissantSchema(
        name=name,
        description=description,
        record_set_id=rs_id,
        fields=fields,
        splits=splits,
    )


def fetch_and_parse(repo_id: str, *, timeout: int = 20) -> CroissantSchema:
    """Convenience: fetch + parse in one call."""
    return parse_croissant(fetch_croissant(repo_id, timeout=timeout))
