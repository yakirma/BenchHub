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
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import benchhub as bh
from benchhub.types import DTYPES


_VALID_STRATEGIES = {"head", "uniform", "stratified"}

# Committed parquet shard naming: "<split>-NNNNN-of-MMMMM[-NNNNN].parquet".
# The optional trailing "-NNNNN" group covers repos that sub-shard each
# logical shard (e.g. OpenFake's core/train-00000-of-00032-00000.parquet).
_SHARD_RE = re.compile(r"^(?P<split>.+?)-\d{3,6}-of-\d{3,6}(?:-\d{3,6})?\.parquet$")


def list_parquet_shards(repo_id, *, config_name=None, revision=None,
                        hf_token=None):
    """Map each split → its ordered list of parquet shard repo-paths.

    Understands the *committed* parquet layouts HF datasets use:
    ``<config>/<split>-NNNNN-of-MMMMM.parquet`` (multi-config repos like
    OpenFake), the same without a config dir, and single-shard
    ``<split>.parquet``. Returns ``{}`` (so the caller falls back to a
    normal full ``load_dataset``) for layouts it can't address —
    loader-script repos, the ``refs/convert/parquet`` auto-export
    branch, exotic directory names. Lexicographic file order matches the
    ``-NNNNN-of-`` numbering, so "first K shards" is deterministic.
    """
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset",
                                    revision=revision, token=hf_token or None)
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        # With a config, only files directly under that config dir count
        # (the <config>/<split>-*.parquet convention); without one, take
        # whatever parquet the repo exposes.
        if config_name:
            if len(parts) < 2 or parts[0] != config_name:
                continue
        base = parts[-1]
        m = _SHARD_RE.match(base)
        split = m.group("split") if m else base[: -len(".parquet")]
        groups[split].append(f)
    return {s: sorted(v) for s, v in groups.items()}


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
    # Fetch the label column in ONE shot rather than `ds[i]` per row —
    # row indexing decodes every column (including any Image), so a
    # row-by-row scan would decode the whole image set just to read the
    # class id. `ds[col]` materialises only the label column (ints for a
    # ClassLabel feature). Fall back to row access if bulk fails.
    try:
        col_values = list(ds[col])
    except Exception:
        col_values = None
    buckets: dict[Any, list[int]] = {}
    for i in range(total):
        if col_values is not None:
            v = col_values[i]
        else:
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


def _to_jsonable(v):
    """Recursive coercion of HF row values to JSON-serializable form.

    HF parquet rows often carry PIL Image / bytes / numpy / pathlib
    types inside a column we typed as 'json'. json.dumps() will choke
    on those. This walker preserves structure, falls back to a small
    placeholder dict for unknown opaque types, and stringifies the
    common scalars (datetimes, paths)."""
    import numpy as _np
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, bytes):
        return {"__bytes__": True, "len": len(v)}
    if _PILImage is not None and isinstance(v, _PILImage.Image):
        return {"__pil__": True, "mode": v.mode, "size": list(v.size)}
    if isinstance(v, _np.ndarray):
        return {"__ndarray__": True, "shape": list(v.shape),
                "dtype": str(v.dtype)}
    if isinstance(v, (_np.integer,)): return int(v)
    if isinstance(v, (_np.floating,)): return float(v)
    if isinstance(v, _np.bool_): return bool(v)
    # Datetime, pathlib, anything else — stringify.
    try:
        return str(v)
    except Exception:
        return {"__opaque__": type(v).__name__}


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
        if kind == "label_list":
            if not isinstance(value, list):
                return None
            # `k` is required on the LabelList contract. The
            # materialize caller is the manifest's `params` block —
            # see `validate_manifest` for the up-front check.
            if "k" not in params:
                raise ValueError(
                    "label_list field requires params.k (top-K depth) "
                    "to materialize; declare it on the dataset's pred field"
                )
            return bh.LabelList(value, **params)
        if kind == "scalar":
            return bh.Scalar(float(value))
        if kind == "json":
            # HF rows can land non-JSON-able values in a 'json' column —
            # PIL Image instances when the row carries embedded images,
            # raw bytes, numpy arrays, sets, etc. Recursively coerce
            # them to JSON-friendly placeholders so the encode() doesn't
            # crash mid-materialize. The placeholder preserves the kind
            # so a downstream consumer can tell what got dropped.
            return bh.Json(_to_jsonable(
                value if isinstance(value, (dict, list)) else {"value": value}
            ))
    except Exception as e:
        print(f"DEBUG: _row_value_to_typed kind={kind} failed: {e}")
        return None
    return None


_SAMPLE_NAME_SAFE_RE = None  # populated lazily; avoids importing `re` at module top


def _sanitize_sample_name(raw: str) -> str:
    """Cap length, replace path / shell-hostile chars with `_`, and
    strip leading/trailing whitespace + dots. Empty results return
    '' so the caller can fall back to enumeration."""
    import re as _re
    global _SAMPLE_NAME_SAFE_RE
    if _SAMPLE_NAME_SAFE_RE is None:
        _SAMPLE_NAME_SAFE_RE = _re.compile(r"[^A-Za-z0-9._-]+")
    cleaned = _SAMPLE_NAME_SAFE_RE.sub("_", str(raw)).strip("._ \t\r\n")
    # The filename pieces written under <field>/<sample>.<ext> need
    # to stay reasonable. 80 chars is plenty of room for human
    # identifiers without bloating directory listings.
    return cleaned[:80]


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
    sample_name_from: str | None = None,
    config_name: str | None = None,
    shard_cap: int = -1,
    progress_cb=None,
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

    if progress_cb is not None:
        try:
            progress_cb({"phase": "downloading", "current": 0, "total": 0,
                         "message": f"Streaming {repo_id}…"})
        except Exception:
            pass

    from datasets import load_dataset  # heavy import — gate to call time.

    ds_kwargs: dict[str, Any] = {"trust_remote_code": False}
    if config_name:
        ds_kwargs["name"] = config_name
    if revision:
        ds_kwargs["revision"] = revision
    if hf_token:
        ds_kwargs["token"] = hf_token
    if split:
        ds_kwargs["split"] = split

    # Shard cap (only when the split is multi-shard parquet): download a
    # bounded prefix of shards instead of the whole split, then load
    # those. shard_cap > 0 → the first N shards; shard_cap == 0 → "auto",
    # download just enough shards (in order) to cover `sample_cap` rows;
    # shard_cap < 0 → off (full load). Bounds download bytes + time — the
    # real constraint on large repos — at the cost of a possibly skewed
    # slice when rows aren't shuffled across shards. Falls back to a
    # normal full load when the layout isn't shard-addressable.
    shards_used = shards_total = None
    all_shards: list[str] = []
    if shard_cap is not None and shard_cap >= 0 and split:
        try:
            all_shards = list_parquet_shards(
                repo_id, config_name=config_name, revision=revision,
                hf_token=hf_token).get(split) or []
        except Exception:
            all_shards = []

    _auto = (shard_cap == 0 and bool(sample_cap) and sample_cap > 0)
    _firstn = (shard_cap > 0 and shard_cap < len(all_shards))
    if len(all_shards) > 1 and (_auto or _firstn):
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as _pq
        shards_total = len(all_shards)
        limit = shard_cap if _firstn else len(all_shards)
        local_paths: list[str] = []
        rows_so_far = 0
        for i, sp in enumerate(all_shards[:limit]):
            if progress_cb is not None:
                try:
                    progress_cb({"phase": "downloading", "current": i,
                                 "total": (shard_cap if _firstn else 0),
                                 "message": (f"Downloading shard {i + 1} of "
                                             f"{shards_total} — {repo_id}…")})
                except Exception:
                    pass
            lp = hf_hub_download(repo_id, sp, repo_type="dataset",
                                 revision=revision, token=hf_token or None)
            local_paths.append(lp)
            if _auto:
                # Stop as soon as the shards downloaded hold enough rows.
                # Read the count from the parquet footer — cheap, no full
                # parse. (load_dataset over `local_paths` reads them once.)
                try:
                    rows_so_far += _pq.ParquetFile(lp).metadata.num_rows
                except Exception:
                    rows_so_far = sample_cap  # can't introspect → stop here
                if rows_so_far >= sample_cap:
                    break
        shards_used = len(local_paths)
        ds = load_dataset("parquet", data_files=local_paths, split="train")
    else:
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

    # --- Classification coverage policy --------------------------------
    # A dataset whose GT is a class label must cache EVERY class with a
    # healthy number of examples, or the preview/eval set silently drops
    # classes — head sampling on a class-sorted split (the common HF
    # layout) grabs only the first few classes. So for any capped import
    # with a `label`/`labellist` GT field: force *stratified* sampling
    # and raise the cap to max(2000, 10 × n_classes) — at least 10 images
    # per class, or ~2000 spread uniformly across classes when there are
    # few enough to fit. Untouched when the admin imports the whole split
    # (cap <= 0 already covers every class).
    _label_field = next(
        (f for f in fields
         if (f.get("role") or "gt") == "gt"
         and f.get("kind") in ("label", "labellist")),
        None,
    )
    if _label_field is not None and sample_cap is not None and sample_cap > 0:
        _names = (_label_field.get("params") or {}).get("names")
        _n_classes = len(_names) if _names else None
        if _n_classes:
            _policy_cap = max(2000, 10 * _n_classes)
            if sample_cap < _policy_cap:
                sample_cap = _policy_cap
        # Stratify so per-class counts are balanced and complete,
        # independent of how the source split is ordered.
        sampling = "stratified"

    # sample_cap <= 0 means "import every row" — the admin set max
    # samples to -1 (or left it blank). Pre-materialize quota check
    # in the route is what's keeping unlimited imports from blowing
    # past the user's storage cap.
    n = total if sample_cap is None or sample_cap <= 0 else min(sample_cap, total)
    indices = _pick_indices(ds, n, fields, strategy=sampling, seed=seed)
    n = len(indices)  # may have been clamped by _pick_indices

    root = Path(staging_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Sample names: by default, enumeration-based (`s000000..s00000N`)
    # to keep the on-disk layout compact regardless of which rows
    # we picked from the source. Admin can override by passing
    # `sample_name_from=<column>` — usually a text column carrying
    # human-readable identifiers (image filename, captions, qid).
    # Duplicates + empty / unsanitisable values fall back to the
    # enumerated form for that row, so the resulting list is always
    # unique without dropping samples.
    fallback_names = [f"s{i:06d}" for i in range(n)]
    if sample_name_from:
        sample_names: list[str] = []
        seen: set[str] = set()
        n_collisions = 0
        for fallback_idx, source_idx in enumerate(indices):
            row = ds[source_idx]
            raw = row.get(sample_name_from) if isinstance(row, dict) else None
            cleaned = _sanitize_sample_name(raw) if raw is not None else ""
            name = cleaned or fallback_names[fallback_idx]
            if name in seen:
                n_collisions += 1
                # Disambiguate with a stable numeric suffix tied to
                # the source row index, NOT a running counter, so
                # the same raw value always lands on the same name
                # across re-imports with the same seed.
                name = f"{name}__{source_idx}"
                # Pathological case: even that collides. Fall back
                # to the enumerated default.
                if name in seen:
                    name = fallback_names[fallback_idx]
            seen.add(name)
            sample_names.append(name)
    else:
        sample_names = fallback_names
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
    for f in data_bearing:
        (root / f["name"]).mkdir(parents=True, exist_ok=True)

    def _emit(phase, **kw):
        if progress_cb is None:
            return
        try:
            progress_cb({"phase": phase, **kw})
        except Exception:
            pass  # progress reporting is best-effort

    _emit("materializing", current=0, total=n,
          message=f"Writing {n} sample(s) × {len(data_bearing)} field(s)…")

    written = 0
    skipped: list[str] = []
    present_counts: dict[str, int] = {f["name"]: 0 for f in data_bearing}
    # Heartbeat the progress callback every PROGRESS_EVERY rows.
    # Too frequent → wasteful DB writes; too sparse → bar feels frozen.
    PROGRESS_EVERY = max(1, n // 100) if n > 100 else 1
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
            present_counts[f["name"]] += 1
        if (sample_idx + 1) % PROGRESS_EVERY == 0 or sample_idx == n - 1:
            _emit("materializing", current=sample_idx + 1, total=n,
                  message=f"Wrote sample {sample_idx + 1}/{n}")

    # Mark *sparse* fields optional so the importer tolerates the gaps
    # instead of failing its missing-file pre-flight. A field present for
    # SOME-but-not-all samples is legitimately sparse (e.g. OpenFake's
    # generation `prompt`, null for real images). A field present for
    # NONE is almost always a column/kind mis-map — leave it required so
    # the importer surfaces it loudly rather than silently dropping it.
    sparse = {name for name, c in present_counts.items() if 0 < c < n}
    if sparse:
        for fe in manifest["fields"]:
            if fe["name"] in sparse:
                fe["optional"] = True
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "samples": n,
        "fields": len(fields),
        "rows_written": written,
        "rows_skipped": len(skipped),
        "skipped_sample_field_pairs": skipped[:20],  # cap noise in flash
        "sampling": sampling,
        "seed": int(seed),
        "split": split,
        # When shard-capped, `total` is the rows in the downloaded shards,
        # not the whole split — shards_used/total make the cap explicit.
        "total_rows_in_split": total,
        "shards_used": shards_used,
        "shards_total": shards_total,
    }
