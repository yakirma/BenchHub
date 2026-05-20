"""Materialise an HF dataset split into the BH typed-manifest layout.

The admin's preview-form selections (parsed Croissant schema +
their role/params/kind choices) become a typed `manifest.json` +
per-(field, sample) files on disk. The standard `import_typed_dataset`
takes it from there.

This module hosts the HF-library-touching code so the route stays thin
and easy to test with a stubbed materialiser. Importing `datasets`
inside the function (not at module top) lets the rest of the code be
imported on machines that don't have the HF library installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import benchhub as bh
from benchhub.types import DTYPES


def _row_value_to_typed(value: Any, kind: str, params: dict) -> bh.DataType | None:
    """Coerce a single HF row value into a typed BH instance.

    The conversion is best-effort: when the kind doesn't match the
    value (e.g. admin set kind=mask on a scalar column) we return
    None and the caller skips the file. Importers see the missing
    file as a hard error, so the admin gets a clean failure message
    instead of a corrupt import.
    """
    if value is None:
        return None
    try:
        if kind == "image":
            from PIL import Image as PILImage
            if isinstance(value, PILImage.Image):
                arr = np.asarray(value.convert("RGB"))
            else:
                arr = np.asarray(value)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return bh.Image(arr)
        if kind == "mask":
            arr = np.asarray(value)
            return bh.Mask(arr, **params)
        if kind == "depth":
            arr = np.asarray(value, dtype=np.float32)
            return bh.Depth(arr, **({"unit": "meters"} | params))
        if kind == "audio":
            # HF's Audio feature decodes to {'array': np.ndarray,
            # 'sampling_rate': int}; some custom features return a
            # bare path string. Handle both.
            if isinstance(value, dict) and "array" in value:
                return bh.Audio(value["array"], value.get("sampling_rate", 16000))
            return None
        if kind == "text":
            return bh.Text(str(value))
        if kind == "bboxes":
            return bh.BBoxes(value if isinstance(value, list) else [], **params)
        if kind == "label":
            if isinstance(value, (int, str)):
                return bh.Label(value)
            return bh.Label(int(value))
        if kind == "scalar":
            return bh.Scalar(float(value))
        if kind == "json":
            return bh.Json(value if isinstance(value, (dict, list)) else {"value": value})
    except Exception as e:
        print(f"DEBUG: _row_value_to_typed kind={kind} failed: {e}")
        return None
    return None


def materialize_hf_to_typed_dir(
    repo_id: str,
    *,
    split: str | None,
    sample_cap: int,
    staging_dir: str,
    dataset_name: str,
    fields: list[dict],
    revision: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Download up to `sample_cap` rows of `repo_id[:split]` and lay
    them out under `staging_dir` in the typed-manifest format.

    `fields` is a list of dicts shaped:
        {
          "name":           str,    # BH field name (becomes the subdir + manifest entry)
          "source_column":  str,    # HF row dict key
          "kind":           str,    # BH kind, must be in benchhub.types.DTYPES
          "role":           str,    # 'input' or 'gt' (role='skip' should be filtered out before call)
          "params":         dict,   # per-instance metadata: depth unit, mask ignore_index, ...
        }

    Returns a summary dict with row counts. Raises ValueError on
    obvious manifest problems (no fields, unknown kind).
    """
    if not fields:
        raise ValueError("no fields selected for materialisation")
    for f in fields:
        if f["kind"] not in DTYPES:
            raise ValueError(f"unknown kind {f['kind']!r} for field {f['name']!r}")

    from datasets import load_dataset  # heavy import — gate to call time.

    ds_kwargs: dict[str, Any] = {"trust_remote_code": False}
    if revision:
        ds_kwargs["revision"] = revision
    if hf_token:
        ds_kwargs["token"] = hf_token
    if split:
        ds_kwargs["split"] = split

    ds = load_dataset(repo_id, **ds_kwargs)
    # `load_dataset` returns a `DatasetDict` when split is unspecified.
    # For the admin form path we always pick a split upstream, but be
    # forgiving: if a Dict comes back, pick the first split.
    if hasattr(ds, "keys") and not hasattr(ds, "__getitem__") or (
        hasattr(ds, "keys") and not hasattr(ds, "num_rows")
    ):
        first_split = next(iter(ds.keys()))
        ds = ds[first_split]

    total = len(ds)
    n = min(sample_cap, total)

    root = Path(staging_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": dataset_name,
        "version": "1.0",
        "fields": [
            {
                "name": f["name"],
                "kind": f["kind"],
                "role": f["role"],
                "params": f.get("params") or {},
            }
            for f in fields
        ],
        "samples": [f"s{i:06d}" for i in range(n)],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for f in fields:
        (root / f["name"]).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[str] = []
    for i in range(n):
        row = ds[i]
        sample_name = f"s{i:06d}"
        for f in fields:
            col = f.get("source_column") or f["name"]
            value = row.get(col) if isinstance(row, dict) else None
            inst = _row_value_to_typed(value, f["kind"], f.get("params") or {})
            if inst is None:
                skipped.append(f"{sample_name}/{f['name']}")
                continue
            cls = DTYPES[f["kind"]]
            ext = cls.file_ext or ".txt"
            (root / f["name"] / f"{sample_name}{ext}").write_bytes(inst.encode())
            written += 1

    return {
        "samples": n,
        "fields": len(fields),
        "rows_written": written,
        "rows_skipped": len(skipped),
        "skipped_sample_field_pairs": skipped[:20],  # cap noise in flash
    }
