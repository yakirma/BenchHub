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
import random
from pathlib import Path
from typing import Any

import numpy as np

import benchhub as bh
from benchhub.types import DTYPES


_VALID_STRATEGIES = {"head", "uniform", "stratified"}


def _pick_indices(
    ds,
    n: int,
    fields: list[dict],
    *,
    strategy: str,
    seed: int,
) -> list[int]:
    """Return `n` row indices into `ds` per the requested sampling strategy.

    - ``head``       — `[0, 1, …, n-1]`. Fastest, deterministic, but
                       sensitive to row order (sorted-by-class datasets
                       end up with one class dominating).
    - ``uniform``    — seeded `random.sample(range(len(ds)), n)`,
                       sorted ascending. Same byte-for-byte output
                       across runs given the same seed.
    - ``stratified`` — group rows by the first field whose role is
                       ``gt`` and kind is ``label``; allocate
                       ``n // num_classes`` per class with the
                       remainder spread across the first few classes;
                       random.sample inside each bucket. Falls back to
                       uniform when no eligible label field is
                       declared.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"unknown sampling strategy {strategy!r}; expected one of "
            f"{sorted(_VALID_STRATEGIES)}"
        )
    total = len(ds)
    if n >= total:
        return list(range(total))

    if strategy == "head":
        return list(range(n))

    rng = random.Random(int(seed))

    if strategy == "uniform":
        return sorted(rng.sample(range(total), n))

    # stratified --------------------------------------------------------
    label_field = next(
        (f for f in fields
         if f.get("role") == "gt" and f.get("kind") == "label"),
        None,
    )
    if label_field is None:
        # No label to stratify on — silently fall back to uniform.
        return sorted(rng.sample(range(total), n))

    col = label_field.get("source_column") or label_field["name"]
    buckets: dict[Any, list[int]] = {}
    for i in range(total):
        row = ds[i]
        v = row.get(col) if isinstance(row, dict) else None
        if v is None:
            continue
        # Class values must be hashable; cast tuples/lists to str so
        # they can be dict keys without raising.
        try:
            hash(v)
            key = v
        except TypeError:
            key = str(v)
        buckets.setdefault(key, []).append(i)
    if not buckets:
        return sorted(rng.sample(range(total), n))

    sorted_buckets = sorted(buckets.items(), key=lambda kv: str(kv[0]))
    n_classes = len(sorted_buckets)
    per_class = n // n_classes
    extra = n - per_class * n_classes

    out: list[int] = []
    for i, (_label, idxs) in enumerate(sorted_buckets):
        quota = per_class + (1 if i < extra else 0)
        if len(idxs) <= quota:
            out.extend(idxs)
        else:
            out.extend(rng.sample(idxs, quota))

    # If short (a class had fewer rows than its quota), top up
    # uniformly from the unused indices.
    if len(out) < n:
        taken = set(out)
        remaining = [i for i in range(total) if i not in taken]
        shortfall = n - len(out)
        if remaining:
            out.extend(rng.sample(remaining, min(shortfall, len(remaining))))
    return sorted(out)


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
    sampling: str = "head",
    seed: int = 42,
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

    # For label fields, lift the class-name vocab off the HF
    # `ClassLabel` feature (e.g. ['airplane', 'automobile', ...] for
    # cifar10) into the field's params so it lands in DatasetField /
    # CustomField.data_params downstream. The dataset view uses it to
    # render the legend + map integer values to class names.
    feats = getattr(ds, "features", None) or {}
    for f in fields:
        if f.get("kind") != "label":
            continue
        col = f.get("source_column") or f["name"]
        feat = feats.get(col) if isinstance(feats, dict) else feats.get(col, None) if feats else None
        names = getattr(feat, "names", None)
        if not names:
            continue
        params = dict(f.get("params") or {})
        params.setdefault("names", list(names))
        f["params"] = params

    total = len(ds)
    # sample_cap <= 0 means "import every row" — the admin set max
    # samples to -1 (or left it blank). Pre-materialize quota check
    # in the route is what's keeping unlimited imports from blowing
    # past the user's storage cap.
    n = total if sample_cap is None or sample_cap <= 0 else min(sample_cap, total)
    indices = _pick_indices(ds, n, fields, strategy=sampling, seed=seed)
    n = len(indices)  # may have been clamped by _pick_indices

    root = Path(staging_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Sample names are based on enumeration position, not source row
    # index, so the on-disk layout stays compact (`s000000..s00000N`)
    # regardless of which rows we picked from the source.
    sample_names = [f"s{i:06d}" for i in range(n)]
    # Pred fields are schema-only declarations — the HF source has
    # no column for them. They go into the manifest so the
    # downstream `import_typed_dataset` writes a DatasetField row,
    # but we don't try to materialise per-sample files for them.
    data_bearing = [f for f in fields if f.get("role") != "pred"]

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
        "samples": sample_names,
        # Track the source-row mapping for traceability — useful when
        # debugging a stratified subset later.
        "source": {
            "repo_id": repo_id,
            "split": split,
            "sampling": sampling,
            "seed": int(seed),
            "row_indices": indices,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for f in data_bearing:
        (root / f["name"]).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[str] = []
    for sample_idx, source_idx in enumerate(indices):
        row = ds[source_idx]
        sample_name = sample_names[sample_idx]
        for f in data_bearing:
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
        "sampling": sampling,
        "seed": int(seed),
        "split": split,
        "total_rows_in_split": total,
    }
