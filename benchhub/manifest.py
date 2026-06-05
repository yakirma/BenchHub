"""Typed dataset manifest — the disk format BenchHub now ingests.

A dataset is a directory containing a `manifest.json` plus one
sub-directory per declared field. Each field's bytes live at
`<field_name>/<sample_name>.<ext>` where `<ext>` follows the type
class's `file_ext`. Inline kinds (`Scalar`, `Label`) use `.txt`;
their value is the result of `DataType.encode()` decoded as UTF-8.

Manifest schema (all keys required unless noted):

```json
{
  "name": "cifar10-test",
  "version": "1.0",
  "description": "optional human description",
  "fields": [
    {"name": "image", "kind": "image", "role": "input",  "params": {}},
    {"name": "label", "kind": "label", "role": "gt",     "params": {"vocab": ["airplane", ...]}}
  ],
  "samples": ["sample_0", "sample_1", ...]
}
```

`role` is `input` (given to submitters at inference time) or `gt`
(held server-side, target of prediction). The same shape drives
the LB's `required_pred_fields_json` (with role `pred`).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from benchhub.types import DTYPES


def _render_preview_from_file(kind: str, src: Path) -> tuple[bytes, str, dict | None]:
    """Read a staged source file + render it to preview bytes per
    kind. Returns (bytes, file_ext, meta_or_None). The meta dict is
    populated for depth (carries the original min/max so hover can
    recover metric values from the lossy colormap encode). Lazy-imports
    benchhub.preview so a non-preview import doesn't pull numpy/PIL
    machinery.
    """
    from benchhub.preview import render_preview, depth_meta
    import numpy as _np
    import io as _io
    from PIL import Image as _PIL
    raw = src.read_bytes()
    if kind == 'image':
        return (*render_preview('image', _PIL.open(_io.BytesIO(raw))), None)
    if kind == 'mask':
        # mask kind stores a class-id PNG on disk; pass through.
        # Also return the raw class-id array as a "meta" payload that the
        # caller writes as a single-channel PNG sidecar, used at hover
        # time so the tooltip can read class ids even after the display
        # JPG is palette-applied + JPEG-compressed.
        arr = _np.asarray(_PIL.open(_io.BytesIO(raw)))
        ids = arr
        if ids.ndim == 3 and ids.shape[-1] == 1:
            ids = ids[..., 0]
        if ids.ndim == 3 and ids.shape[-1] >= 3:
            # Grayscale-stored-as-RGB (R=G=B=class_id) — collapse.
            if (_np.array_equal(ids[..., 0], ids[..., 1])
                    and _np.array_equal(ids[..., 1], ids[..., 2])
                    and int(ids.max()) < 256):
                ids = ids[..., 0]
            else:
                ids = None  # genuine RGB colour mask, no class ids to save
        if ids is not None and ids.ndim == 2:
            classid_meta = {'_classid_array': ids.astype(_np.uint16)}
        else:
            classid_meta = None
        bytes_, ext = render_preview('mask', arr)
        return bytes_, ext, classid_meta
    if kind == 'depth':
        # Canonical depth file_ext is .npz with key 'depth'.
        with _np.load(_io.BytesIO(raw)) as z:
            keys = list(z.keys())
            arr = z[keys[0] if 'depth' not in keys else 'depth']
        arr32 = arr.astype(_np.float32)
        bytes_, ext = render_preview('depth', arr32)
        return bytes_, ext, depth_meta(arr32)
    if kind == 'audio':
        # Lazy import soundfile so the dep stays optional unless
        # someone actually imports audio in preview mode.
        import soundfile as _sf
        with _io.BytesIO(raw) as buf:
            samples, sr = _sf.read(buf, always_2d=False)
        return (*render_preview('audio', samples), None)
    raise ValueError(f'no preview for kind={kind!r}')


_INLINE_EXT = ".txt"
# Roles a dataset field can take. `input` and `gt` carry per-sample
# data on disk; `pred` is schema-only — it declares the wire shape
# the LB accepts from submissions but stores no values at the dataset
# level (those land per-Submission via /api/submit/<lb>).
_VALID_ROLES = {"input", "gt", "pred"}
# Roles whose per-sample files MUST exist on disk + write CustomField
# rows during import. `pred` is intentionally absent here.
_DATA_BEARING_ROLES = {"input", "gt"}


def _field_ext(kind: str, extra_kinds: dict | None = None) -> str:
    """File extension a field of this kind writes on disk. `extra_kinds`
    maps a server-registered kind name → its on-disk extension (or None /
    '' for inline), letting datasets carry user-registered kinds the
    standalone package has no DataType class for."""
    cls = DTYPES.get(kind)
    if cls is not None:
        return cls.file_ext or _INLINE_EXT
    if extra_kinds is not None and kind in extra_kinds:
        return extra_kinds[kind] or _INLINE_EXT
    raise ValueError(f"unknown kind {kind!r}")


def validate_manifest(manifest: dict, extra_kinds: dict | None = None) -> None:
    """Raise ValueError on a malformed manifest. Checks types + roles +
    field-name uniqueness; does NOT verify that the files exist on
    disk — that's the importer's job. `extra_kinds` (name → on-disk ext)
    widens the accepted-kind set with server-registered kinds."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    for required in ("name", "fields", "samples"):
        if required not in manifest:
            raise ValueError(f"manifest missing required key {required!r}")
    if not isinstance(manifest["name"], str) or not manifest["name"].strip():
        raise ValueError("manifest.name must be a non-empty string")
    if not isinstance(manifest["fields"], list) or not manifest["fields"]:
        raise ValueError("manifest.fields must be a non-empty list")
    if not isinstance(manifest["samples"], list) or not manifest["samples"]:
        raise ValueError("manifest.samples must be a non-empty list")
    seen: set[str] = set()
    for i, f in enumerate(manifest["fields"]):
        if not isinstance(f, dict):
            raise ValueError(f"fields[{i}] must be an object")
        for key in ("name", "kind", "role"):
            if key not in f:
                raise ValueError(f"fields[{i}] missing {key!r}")
        if f["name"] in seen:
            raise ValueError(f"duplicate field name {f['name']!r}")
        seen.add(f["name"])
        if f["kind"] not in DTYPES and not (
                extra_kinds is not None and f["kind"] in extra_kinds):
            allowed = sorted(set(DTYPES) | set(extra_kinds or {}))
            raise ValueError(
                f"fields[{i}].kind={f['kind']!r} not an accepted kind "
                f"{allowed}"
            )
        if f["role"] not in _VALID_ROLES:
            raise ValueError(
                f"fields[{i}].role={f['role']!r} must be one of {sorted(_VALID_ROLES)}"
            )
        params = f.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"fields[{i}].params must be an object")
        # Kind-specific param requirements. LabelList is the only
        # one right now: every label_list field MUST declare an
        # integer `k` (top-K depth) so submissions can be validated
        # against an exact-length contract.
        if f["kind"] == "label_list":
            k = params.get("k")
            if not isinstance(k, int) or k < 1:
                raise ValueError(
                    f"fields[{i}] kind=label_list requires params.k as a "
                    f"positive integer; got {k!r}"
                )


def load_manifest(manifest_path: str | os.PathLike,
                  extra_kinds: dict | None = None) -> dict:
    """Read + validate a manifest from disk. Returns the parsed dict."""
    path = Path(manifest_path)
    with open(path) as f:
        data = json.load(f)
    validate_manifest(data, extra_kinds)
    return data


# ---------------------------------------------------------------------------
# Importer — wired up via app.py to materialise Dataset / Sample / CustomField rows.
# ---------------------------------------------------------------------------

def expected_file_path(source_root: str | os.PathLike, field: dict, sample_name: str,
                       extra_kinds: dict | None = None) -> Path:
    """Where the on-disk file for this (field, sample) is expected.
    Caller checks existence + reads."""
    ext = _field_ext(field["kind"], extra_kinds)
    return Path(source_root) / field["name"] / f"{sample_name}{ext}"


def import_typed_dataset(
    source_root: str | os.PathLike,
    *,
    db_session,
    Dataset,
    Sample,
    CustomField,
    upload_folder: str | os.PathLike,
    owner_user_id: int | None = None,
    visibility: str = "public",
    DatasetField=None,
    existing_dataset=None,
    tolerate_incomplete: bool = False,
    preview_only: bool = False,
    extra_kinds: dict | None = None,
) -> tuple[int, dict]:
    """Materialise a typed dataset into the DB + uploads volume.

    Reads `<source_root>/manifest.json`, then for every (field, sample)
    pair copies the file into `<upload_folder>/datasets/<dataset_id>/
    <field_name>/<sample_name>.<ext>` and creates a CustomField row.
    Inline kinds (scalar/label) decode the file content via the type
    class and store the value directly in CustomField.value_float /
    .value_text — no on-disk copy.

    When `existing_dataset` is provided, attach Sample/DatasetField/
    CustomField rows to that pre-created row instead of creating a
    new Dataset. Used by the async import flow that creates the
    Dataset eagerly (with `import_status='importing'`) so it appears
    on /datasets while the background task is still working.

    Returns (dataset_id, summary_dict).
    The caller is responsible for db_session.commit().
    """
    source_root = Path(source_root)
    extra_kinds = extra_kinds or {}
    manifest = load_manifest(source_root / "manifest.json", extra_kinds)

    # Pre-flight: every data-bearing (field, sample) file must exist.
    # `role='pred'` is schema-only — no per-sample files expected.
    # Default behaviour (`tolerate_incomplete=False`) is strict: any
    # missing file raises. The HF bulk importer passes True so a
    # single bad parquet row doesn't kill the whole import.
    data_fields = [f for f in manifest["fields"]
                   if f.get("role", "gt") in _DATA_BEARING_ROLES]
    if tolerate_incomplete:
        kept_samples: list[str] = []
        dropped: list[str] = []
        for s in manifest["samples"]:
            sample_missing = [f["name"] for f in data_fields
                              if not expected_file_path(source_root, f, s, extra_kinds).exists()]
            if sample_missing:
                dropped.append(f"{s}({','.join(sample_missing)})")
            else:
                kept_samples.append(s)
        if not kept_samples:
            raise FileNotFoundError(
                f"every sample missing at least one field; first 5 drops: "
                f"{dropped[:5]}"
            )
        if dropped:
            manifest["samples"] = kept_samples
            manifest.setdefault("source", {})["dropped_incomplete"] = dropped[:50]
    else:
        missing: list[str] = []
        for f in data_fields:
            for s in manifest["samples"]:
                p = expected_file_path(source_root, f, s, extra_kinds)
                if not p.exists():
                    missing.append(str(p.relative_to(source_root)))
        if missing:
            raise FileNotFoundError(
                f"manifest references missing files: {missing[:5]}"
                + ("…" if len(missing) > 5 else "")
            )

    if existing_dataset is not None:
        dataset = existing_dataset
    else:
        dataset = Dataset(
            name=manifest["name"],
            owner_user_id=owner_user_id,
            visibility=visibility,
        )
        db_session.add(dataset)
    db_session.flush()  # need dataset.id for the upload path

    dataset_dir = Path(upload_folder) / "datasets" / str(dataset.id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Schema rows — single source of truth for kind / params / role.
    # The matching per-sample CustomField rows still carry the same
    # values (lookup cost), but DatasetField is what defines the
    # leaderboard's pred contract.
    if DatasetField is not None:
        for f in manifest["fields"]:
            params = f.get("params") or {}
            df = DatasetField(
                dataset_id=dataset.id,
                name=f["name"],
                kind=f["kind"],
                role=f.get("role", "gt"),
            )
            df.set_params(params)
            db_session.add(df)

    samples_by_name: dict[str, Any] = {}
    for s_name in manifest["samples"]:
        s = Sample(dataset_id=dataset.id, name=s_name)
        db_session.add(s)
        samples_by_name[s_name] = s
    db_session.flush()

    n_field_rows = 0
    n_files_copied = 0
    for f in manifest["fields"]:
        kind = f["kind"]
        # Registered (server-side) kinds have no DataType class in this
        # standalone package — store their bytes verbatim, keyed off the
        # extension the caller passed in extra_kinds.
        cls = DTYPES.get(kind)
        params = f.get("params") or {}
        if cls is not None:
            is_inline = cls.file_ext is None
        else:
            is_inline = not (extra_kinds.get(kind))
        # Pred fields are declared but carry no data at the dataset
        # level — skip the per-sample CustomField writes for them.
        # The DatasetField row above is still created so the LB can
        # discover the pred contract.
        if f.get("role", "gt") not in _DATA_BEARING_ROLES:
            continue

        field_dir = dataset_dir / f["name"]
        if not is_inline:
            field_dir.mkdir(parents=True, exist_ok=True)

        for s_name in manifest["samples"]:
            src = expected_file_path(source_root, f, s_name, extra_kinds)
            cf = CustomField(
                sample_id=samples_by_name[s_name].id,
                name=f["name"],
                data_type=kind,
            )
            if params:
                cf.set_params(params)

            if is_inline:
                # Inline kinds: decode the file bytes back into the
                # primitive value and stash in value_float / value_text.
                blob = src.read_bytes()
                if cls is None:
                    # Registered inline kind — store the content verbatim
                    # as text; the kind's own decode() (server-side) turns
                    # it back into an object for metrics.
                    cf.value_text = blob.decode("utf-8", "replace").rstrip("\n")
                else:
                    inst = cls.decode(blob, params)
                    if kind == "scalar":
                        cf.value_float = float(inst.value)
                    else:
                        cf.value_text = blob.decode("utf-8").rstrip("\n")
            else:
                # File-backed kinds: copy under uploads/, store
                # relative-to-uploads path in value_text.
                # In preview_only mode, render a downscaled +
                # colormapped JPG (or PNG waveform for audio) and
                # write that instead of the source bytes. The kind
                # stays the same — preview is just a different file
                # format under the same path.
                if preview_only and kind in ('image', 'mask', 'depth', 'audio'):
                    try:
                        prev_bytes, prev_ext, prev_meta = _render_preview_from_file(
                            kind, src
                        )
                        dst = field_dir / (Path(src.name).stem + prev_ext)
                        dst.write_bytes(prev_bytes)
                        # Sidecar handling:
                        #   depth → {'min','max','shape'} dict → .meta.json
                        #   mask  → {'_classid_array': uint16 (H,W)} → .classid.png
                        # The display JPG is lossy palette/colormap; the
                        # sidecar preserves the precise per-pixel value
                        # so hover tooltips and ?raw=1 can serve real
                        # class ids / depth meta even for preview-only
                        # imports.
                        if prev_meta is not None:
                            if kind == 'mask' and '_classid_array' in prev_meta:
                                from PIL import Image as _PIL2
                                arr = prev_meta['_classid_array']
                                # Canvas getImageData drops 16-bit PNGs to
                                # 8-bit on the way in. Save 'L' (uint8)
                                # when class ids fit; only fall back to
                                # I;16 for >256-class datasets so hover
                                # still reads accurate ids in the typical
                                # case.
                                import numpy as _np2
                                if int(arr.max()) < 256:
                                    out_arr = arr.astype(_np2.uint8)
                                    mode = 'L'
                                else:
                                    out_arr = arr.astype(_np2.uint16)
                                    mode = 'I;16'
                                _PIL2.fromarray(out_arr, mode=mode).save(
                                    field_dir / (Path(src.name).stem + '.classid.png')
                                )
                            elif kind == 'depth':
                                import json as _json
                                (field_dir / (Path(src.name).stem + '.meta.json')
                                ).write_text(_json.dumps(prev_meta))
                    except Exception as e:
                        # Fall back to full copy if preview generation
                        # fails — partial preview > broken sample.
                        dst = field_dir / src.name
                        shutil.copy2(src, dst)
                        print(f'  preview generation failed for '
                              f'{f["name"]}/{s_name} ({kind}): {e}; copied original')
                else:
                    dst = field_dir / src.name
                    shutil.copy2(src, dst)
                cf.value_text = str(dst.relative_to(upload_folder))
                n_files_copied += 1
                # `text` / `json` are file-backed for layout
                # symmetry, but every code path that renders them
                # (comparison view, dataset view's text scrollbox)
                # reads CustomField.value_text directly as the
                # CONTENT. Without this override, value_text holds
                # the file path instead and users see e.g.
                # "datasets/9/shelfmark/00000.txt" rendered as the
                # shelfmark.
                if kind in ("text", "json"):
                    try:
                        cf.value_text = (
                            dst.read_text(encoding="utf-8").rstrip("\n")
                        )
                    except (OSError, UnicodeDecodeError):
                        # Leave the path as a fallback if the file
                        # isn't decodable — better something than a
                        # crashed import.
                        pass
            db_session.add(cf)
            n_field_rows += 1

    # Refresh the cached storage_bytes counter so quota math + the
    # /home dashboard stay accurate without a separate `du` pass.
    bytes_on_disk = sum(
        p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file()
    )
    dataset.storage_bytes = bytes_on_disk
    if preview_only and hasattr(dataset, 'preview_only'):
        dataset.preview_only = True

    summary = {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "samples": len(manifest["samples"]),
        "fields": len(manifest["fields"]),
        "custom_field_rows": n_field_rows,
        "files_copied": n_files_copied,
        "bytes_on_disk": bytes_on_disk,
    }
    return dataset.id, summary


# ---------------------------------------------------------------------------
# Submission manifest — same shape as a dataset, with `predictions[]` in
# place of `fields[]`. Each entry's role is implicit (`pred`).
# ---------------------------------------------------------------------------

def validate_submission_manifest(manifest: dict) -> None:
    """Raise ValueError on a malformed submission manifest."""
    if not isinstance(manifest, dict):
        raise ValueError("submission manifest must be a JSON object")
    for required in ("name", "predictions", "samples"):
        if required not in manifest:
            raise ValueError(f"submission manifest missing required key {required!r}")
    if not isinstance(manifest["name"], str) or not manifest["name"].strip():
        raise ValueError("submission manifest.name must be a non-empty string")
    if not isinstance(manifest["predictions"], list) or not manifest["predictions"]:
        raise ValueError("submission manifest.predictions must be a non-empty list")
    if not isinstance(manifest["samples"], list) or not manifest["samples"]:
        raise ValueError("submission manifest.samples must be a non-empty list")
    seen: set[str] = set()
    for i, p in enumerate(manifest["predictions"]):
        if not isinstance(p, dict):
            raise ValueError(f"predictions[{i}] must be an object")
        for key in ("name", "kind"):
            if key not in p:
                raise ValueError(f"predictions[{i}] missing {key!r}")
        if p["name"] in seen:
            raise ValueError(f"duplicate prediction name {p['name']!r}")
        seen.add(p["name"])
        if p["kind"] not in DTYPES:
            raise ValueError(
                f"predictions[{i}].kind={p['kind']!r} not in DTYPES "
                f"{sorted(DTYPES)}"
            )
        params = p.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"predictions[{i}].params must be an object")


def check_submission_matches_contract(
    submission_manifest: dict, contract: list[dict]
) -> None:
    """Raise ValueError if any required pred field is missing or has
    the wrong kind. `contract` is the parsed Leaderboard.required_pred_fields_json
    (a list of `{name, kind, params, role?}` entries — only `pred`-role
    entries are checked here)."""
    contract_by_name = {
        c["name"]: c for c in contract
        if c.get("role", "pred") == "pred"
    }
    if not contract_by_name:
        # LB hasn't declared a contract; accept whatever the submitter
        # sent. The metric arg_mappings will surface mismatches at
        # eval time.
        return
    pred_by_name = {p["name"]: p for p in submission_manifest["predictions"]}
    missing = sorted(set(contract_by_name) - set(pred_by_name))
    if missing:
        raise ValueError(
            f"submission missing required prediction fields: {missing}"
        )
    for name, want in contract_by_name.items():
        got = pred_by_name[name]
        if got["kind"] != want["kind"]:
            raise ValueError(
                f"prediction {name!r} kind={got['kind']!r} != contract {want['kind']!r}"
            )


def _spatial_shape(inst) -> tuple[int, int] | None:
    """Spatial (H, W) shape of an image/mask/depth-typed instance.
    Returns None when the instance has no array attribute or it's
    sub-2D — used purely as the cross-check key for `shape_match`."""
    arr = getattr(inst, "array", None)
    if arr is None or getattr(arr, "ndim", 0) < 2:
        return None
    return tuple(arr.shape[:2])


def _coerce_fixed_shape(raw) -> tuple[int, int] | None:
    """Validate that `params.shape` is a 2-element `[H, W]` of
    positive ints. Returns the tuple on success, None when absent.
    Raises ValueError on a malformed value so the admin gets a
    clear error at contract-author time instead of a silent skip.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"params.shape must be a 2-element [H, W] list; got {raw!r}"
        )
    try:
        h, w = int(raw[0]), int(raw[1])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"params.shape entries must be ints; got {raw!r}"
        ) from e
    if h <= 0 or w <= 0:
        raise ValueError(f"params.shape must be positive; got {(h, w)}")
    return (h, w)


def _enforce_shape_constraint(
    *,
    source_root: Path,
    manifest: dict,
    contract: list[dict],
    get_input_shape,
) -> None:
    """Per-(sample, pred) shape check.

    Two mutually-exclusive constraint forms in a pred field's
    ``params``:

      * ``shape_match: "<input_field_name>"`` — pred's spatial
        ``(H, W)`` must match that input field's shape per-sample.
        Needs a `get_input_shape(sample, field)` resolver from the
        caller (the on-box submit route wires this).
      * ``shape: [H, W]`` — pred's spatial ``(H, W)`` must equal
        this fixed value for every sample. Doesn't need the
        resolver — the constraint travels with the contract.

    Setting both on the same pred field raises ValueError. No-op
    when neither is set. Inline kinds (scalar/label/...) skip the
    check cleanly since they don't carry a spatial shape.
    """
    by_name = {c["name"]: c for c in contract if isinstance(c, dict)}
    for p in manifest["predictions"]:
        spec = by_name.get(p["name"])
        if not spec:
            continue
        params = spec.get("params") or {}
        shape_match = params.get("shape_match")
        fixed_shape = _coerce_fixed_shape(params.get("shape"))
        # Enforce the "pick one" contract.
        if shape_match and fixed_shape is not None:
            raise ValueError(
                f"pred {p['name']!r}: params.shape and params.shape_match "
                f"can't be set together; pick one."
            )
        if not (shape_match or fixed_shape):
            continue
        if shape_match and not isinstance(shape_match, str):
            raise ValueError(
                f"pred {p['name']!r}: params.shape_match must be the "
                f"input field name as a string; got {shape_match!r}"
            )
        cls = DTYPES.get(p["kind"])
        if cls is None or cls.file_ext is None:
            # Inline kinds (scalar/label) don't have a spatial shape;
            # ignore shape constraints on those.
            continue
        pred_params = p.get("params") or {}
        for s_name in manifest["samples"]:
            path = expected_file_path(source_root, p, s_name)
            try:
                pred_inst = cls.decode(path.read_bytes(), pred_params)
            except Exception as e:
                raise ValueError(
                    f"sample {s_name!r} pred {p['name']!r}: failed to "
                    f"decode for shape check ({e})"
                ) from e
            pred_shape = _spatial_shape(pred_inst)
            if pred_shape is None:
                continue
            if fixed_shape is not None:
                if tuple(pred_shape) != fixed_shape:
                    raise ValueError(
                        f"sample {s_name!r} pred {p['name']!r}: shape "
                        f"{tuple(pred_shape)} != contract shape "
                        f"{tuple(fixed_shape)}"
                    )
                continue
            # shape_match path.
            if get_input_shape is None:
                # No resolver supplied — can't verify, fall through.
                continue
            expected = get_input_shape(s_name, shape_match)
            if expected is None:
                continue
            if tuple(pred_shape) != tuple(expected):
                raise ValueError(
                    f"sample {s_name!r} pred {p['name']!r}: shape "
                    f"{tuple(pred_shape)} != input {shape_match!r} shape "
                    f"{tuple(expected)} (contract requires shape_match)"
                )


# Back-compat alias — earlier code/tests call _enforce_shape_match.
_enforce_shape_match = _enforce_shape_constraint


def import_typed_submission(
    source_root: str | os.PathLike,
    *,
    leaderboard,
    submission_name: str,
    db_session,
    Submission,
    CustomField,
    upload_folder: str | os.PathLike,
    owner_user_id: int | None = None,
    contract: list[dict] | None = None,
    get_input_shape=None,
) -> tuple[int, dict]:
    """Materialise a typed submission: validate against the LB's
    contract, create a Submission row + per-prediction CustomField
    rows, copy file-backed kinds into uploads/submissions/<id>/.

    Returns (submission_id, summary_dict). The caller commits.
    """
    source_root = Path(source_root)
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("submission manifest.json missing at archive root")

    with open(manifest_path) as f:
        manifest = json.load(f)
    validate_submission_manifest(manifest)

    # Caller-supplied contract wins. Falls back to the LB's legacy
    # `required_pred_fields_json` column for back-compat with older
    # LBs that haven't been re-derived from their datasets'
    # `DatasetField` schema.
    if contract is None:
        contract_raw = leaderboard.required_pred_fields_json or "[]"
        try:
            contract = json.loads(contract_raw)
            if not isinstance(contract, list):
                contract = []
        except (TypeError, ValueError):
            contract = []
    check_submission_matches_contract(manifest, contract)

    # Shape-match: pred fields declared with
    # `params.shape_match=<input_field>` must produce per-sample arrays
    # whose spatial (H, W) matches that input. Runs only when the
    # caller hands in a resolver — the on-box import path does, the
    # unit tests stub it.
    _enforce_shape_constraint(
        source_root=Path(source_root),
        manifest=manifest,
        contract=contract,
        get_input_shape=get_input_shape,
    )

    # Pre-flight: every (pred_field, sample) file must exist.
    missing: list[str] = []
    for p in manifest["predictions"]:
        for s in manifest["samples"]:
            path = expected_file_path(source_root, p, s)
            if not path.exists():
                missing.append(str(path.relative_to(source_root)))
    if missing:
        raise FileNotFoundError(
            f"submission references missing prediction files: {missing[:5]}"
            + ("…" if len(missing) > 5 else "")
        )

    submission = Submission(
        leaderboard_id=leaderboard.id,
        name=submission_name or manifest["name"],
        owner_user_id=owner_user_id,
    )
    db_session.add(submission)
    db_session.flush()

    sub_dir = Path(upload_folder) / "submissions" / str(submission.id)
    sub_dir.mkdir(parents=True, exist_ok=True)

    n_pred_rows = 0
    n_files_copied = 0
    for p in manifest["predictions"]:
        kind = p["kind"]
        cls = DTYPES[kind]
        params = p.get("params") or {}
        is_inline = cls.file_ext is None

        field_dir = sub_dir / p["name"]
        if not is_inline:
            field_dir.mkdir(parents=True, exist_ok=True)

        for s_name in manifest["samples"]:
            src = expected_file_path(source_root, p, s_name)
            cf = CustomField(
                submission_id=submission.id,
                sample_name=s_name,
                name=p["name"],
                data_type=kind,
            )
            if params:
                cf.set_params(params)

            if is_inline:
                blob = src.read_bytes()
                inst = cls.decode(blob, params)
                if kind == "scalar":
                    cf.value_float = float(inst.value)
                else:
                    cf.value_text = blob.decode("utf-8").rstrip("\n")
            else:
                dst = field_dir / src.name
                shutil.copy2(src, dst)
                cf.value_text = str(dst.relative_to(upload_folder))
                n_files_copied += 1
                # See the dataset-import counterpart: text/json are
                # file-backed but the comparison view renders
                # value_text as content. Override with the file's
                # actual content so we don't show submitters the
                # file path as their prediction.
                if kind in ("text", "json"):
                    try:
                        cf.value_text = (
                            dst.read_text(encoding="utf-8").rstrip("\n")
                        )
                    except (OSError, UnicodeDecodeError):
                        pass
            db_session.add(cf)
            n_pred_rows += 1

    summary = {
        "submission_id": submission.id,
        "name": submission.name,
        "leaderboard_id": leaderboard.id,
        "predictions": len(manifest["predictions"]),
        "samples": len(manifest["samples"]),
        "custom_field_rows": n_pred_rows,
        "files_copied": n_files_copied,
    }
    return submission.id, summary
