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


_INLINE_EXT = ".txt"
_VALID_ROLES = {"input", "gt", "pred"}


def _field_ext(kind: str) -> str:
    """File extension a field of this kind writes on disk."""
    cls = DTYPES.get(kind)
    if cls is None:
        raise ValueError(f"unknown kind {kind!r}")
    return cls.file_ext or _INLINE_EXT


def validate_manifest(manifest: dict) -> None:
    """Raise ValueError on a malformed manifest. Checks types + roles +
    field-name uniqueness; does NOT verify that the files exist on
    disk — that's the importer's job."""
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
        if f["kind"] not in DTYPES:
            raise ValueError(
                f"fields[{i}].kind={f['kind']!r} not in DTYPES "
                f"{sorted(DTYPES)}"
            )
        if f["role"] not in _VALID_ROLES:
            raise ValueError(
                f"fields[{i}].role={f['role']!r} must be one of {sorted(_VALID_ROLES)}"
            )
        params = f.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"fields[{i}].params must be an object")


def load_manifest(manifest_path: str | os.PathLike) -> dict:
    """Read + validate a manifest from disk. Returns the parsed dict."""
    path = Path(manifest_path)
    with open(path) as f:
        data = json.load(f)
    validate_manifest(data)
    return data


# ---------------------------------------------------------------------------
# Importer — wired up via app.py to materialise Dataset / Sample / CustomField rows.
# ---------------------------------------------------------------------------

def expected_file_path(source_root: str | os.PathLike, field: dict, sample_name: str) -> Path:
    """Where the on-disk file for this (field, sample) is expected.
    Caller checks existence + reads."""
    ext = _field_ext(field["kind"])
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
) -> tuple[int, dict]:
    """Materialise a typed dataset into the DB + uploads volume.

    Reads `<source_root>/manifest.json`, then for every (field, sample)
    pair copies the file into `<upload_folder>/datasets/<dataset_id>/
    <field_name>/<sample_name>.<ext>` and creates a CustomField row.
    Inline kinds (scalar/label) decode the file content via the type
    class and store the value directly in CustomField.value_float /
    .value_text — no on-disk copy.

    Returns (dataset_id, summary_dict).
    The caller is responsible for db_session.commit().
    """
    source_root = Path(source_root)
    manifest = load_manifest(source_root / "manifest.json")

    # Pre-flight: every declared (field, sample) file must exist.
    missing: list[str] = []
    for f in manifest["fields"]:
        for s in manifest["samples"]:
            p = expected_file_path(source_root, f, s)
            if not p.exists():
                missing.append(str(p.relative_to(source_root)))
    if missing:
        raise FileNotFoundError(
            f"manifest references missing files: {missing[:5]}"
            + ("…" if len(missing) > 5 else "")
        )

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
        cls = DTYPES[kind]
        params = f.get("params") or {}
        is_inline = cls.file_ext is None

        field_dir = dataset_dir / f["name"]
        if not is_inline:
            field_dir.mkdir(parents=True, exist_ok=True)

        for s_name in manifest["samples"]:
            src = expected_file_path(source_root, f, s_name)
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
                inst = cls.decode(blob, params)
                if kind == "scalar":
                    cf.value_float = float(inst.value)
                else:
                    cf.value_text = blob.decode("utf-8").rstrip("\n")
            else:
                # File-backed kinds: copy under uploads/, store
                # relative-to-uploads path in value_text.
                dst = field_dir / src.name
                shutil.copy2(src, dst)
                cf.value_text = str(dst.relative_to(upload_folder))
                n_files_copied += 1
            db_session.add(cf)
            n_field_rows += 1

    # Refresh the cached storage_bytes counter so quota math + the
    # /home dashboard stay accurate without a separate `du` pass.
    bytes_on_disk = sum(
        p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file()
    )
    dataset.storage_bytes = bytes_on_disk

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
