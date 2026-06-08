"""Tests for benchhub.manifest — the typed-dataset disk format + importer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import benchhub as bh
from app import CustomField, Dataset, Sample, db
from benchhub.manifest import (
    expected_file_path,
    import_typed_dataset,
    load_manifest,
    validate_manifest,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _ok_manifest() -> dict:
    return {
        "name": "tiny",
        "version": "1.0",
        "fields": [
            {"name": "image", "kind": "image", "role": "input"},
            {"name": "label", "kind": "label", "role": "gt"},
        ],
        "samples": ["s0", "s1"],
    }


def test_validate_accepts_minimal_well_formed():
    validate_manifest(_ok_manifest())


@pytest.mark.parametrize("missing", ["name", "fields", "samples"])
def test_validate_rejects_missing_top_level_key(missing):
    m = _ok_manifest()
    del m[missing]
    with pytest.raises(ValueError, match=missing):
        validate_manifest(m)


def test_validate_rejects_unknown_kind():
    m = _ok_manifest()
    m["fields"][0]["kind"] = "not-a-real-kind"
    with pytest.raises(ValueError, match="not an accepted kind"):
        validate_manifest(m)


def test_validate_accepts_registered_extra_kind():
    m = _ok_manifest()
    m["fields"][0]["kind"] = "volume"
    # Rejected without extra_kinds, accepted when declared.
    with pytest.raises(ValueError, match="not an accepted kind"):
        validate_manifest(m)
    validate_manifest(m, extra_kinds={"volume": ".nii.gz"})


def test_validate_rejects_bad_role():
    m = _ok_manifest()
    m["fields"][0]["role"] = "predicted_by_god"
    with pytest.raises(ValueError, match="role"):
        validate_manifest(m)


def test_validate_rejects_duplicate_field_names():
    m = _ok_manifest()
    m["fields"][1]["name"] = "image"
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(m)


def test_validate_rejects_empty_samples():
    m = _ok_manifest()
    m["samples"] = []
    with pytest.raises(ValueError, match="samples"):
        validate_manifest(m)


# ---------------------------------------------------------------------------
# expected_file_path — extension comes from the type class's file_ext
# ---------------------------------------------------------------------------

def test_expected_path_for_image_kind_uses_png(tmp_path):
    f = {"name": "image", "kind": "image", "role": "input"}
    assert expected_file_path(tmp_path, f, "s0").name == "s0.png"


def test_expected_path_for_depth_kind_uses_npz(tmp_path):
    f = {"name": "depth", "kind": "depth", "role": "gt"}
    assert expected_file_path(tmp_path, f, "s0").name == "s0.npz"


def test_expected_path_for_inline_kind_uses_txt(tmp_path):
    f = {"name": "label", "kind": "label", "role": "gt"}
    assert expected_file_path(tmp_path, f, "s0").name == "s0.txt"


# ---------------------------------------------------------------------------
# load_manifest reads + validates from disk
# ---------------------------------------------------------------------------

def test_load_manifest_reads_and_validates(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_ok_manifest()))
    parsed = load_manifest(path)
    assert parsed["name"] == "tiny"


def test_load_manifest_propagates_validation_errors(tmp_path):
    path = tmp_path / "manifest.json"
    m = _ok_manifest()
    del m["fields"]
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="fields"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Importer end-to-end
# ---------------------------------------------------------------------------

def _build_tiny_dataset_on_disk(root: Path, *, depth_unit: str = "meters") -> None:
    """Two samples, three fields: image, depth (gt), label (gt)."""
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": "imp_tiny",
        "version": "1.0",
        "description": "End-to-end importer test fixture.",
        "fields": [
            {"name": "image",     "kind": "image", "role": "input"},
            {"name": "depth_gt",  "kind": "depth", "role": "gt", "params": {"unit": depth_unit}},
            {"name": "label",     "kind": "label", "role": "gt"},
        ],
        "samples": ["s0", "s1"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    (root / "image").mkdir()
    (root / "depth_gt").mkdir()
    (root / "label").mkdir()

    for s in ("s0", "s1"):
        img = bh.Image(np.zeros((4, 4, 3), dtype=np.uint8))
        (root / "image" / f"{s}.png").write_bytes(img.encode())

        depth = bh.Depth(np.ones((4, 4), dtype=np.float32), unit=depth_unit)
        (root / "depth_gt" / f"{s}.npz").write_bytes(depth.encode())

        label = bh.Label("cat" if s == "s0" else "dog")
        (root / "label" / f"{s}.txt").write_bytes(label.encode())


def test_import_typed_dataset_creates_rows(db_session, tmp_path):
    src = tmp_path / "src"
    _build_tiny_dataset_on_disk(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, summary = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
    )
    db.session.commit()

    assert summary["samples"] == 2
    assert summary["fields"] == 3
    assert summary["custom_field_rows"] == 6
    assert summary["files_copied"] == 4  # 2 image + 2 depth; labels are inline

    ds = Dataset.query.get(ds_id)
    assert ds.name == "imp_tiny"
    assert {s.name for s in ds.samples} == {"s0", "s1"}


def test_import_typed_dataset_inline_label_value_text(db_session, tmp_path):
    src = tmp_path / "src"
    _build_tiny_dataset_on_disk(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, _ = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
    )
    db.session.commit()

    s0 = Sample.query.filter_by(dataset_id=ds_id, name="s0").one()
    labels = {cf.name: cf for cf in s0.custom_fields if cf.data_type == "label"}
    assert labels["label"].value_text == '"cat"'


def test_import_typed_dataset_depth_params_persisted(db_session, tmp_path):
    src = tmp_path / "src"
    _build_tiny_dataset_on_disk(src, depth_unit="millimeters")
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, _ = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
    )
    db.session.commit()

    s0 = Sample.query.filter_by(dataset_id=ds_id, name="s0").one()
    depth = next(cf for cf in s0.custom_fields if cf.name == "depth_gt")
    assert depth.data_type == "depth"
    assert depth.get_params() == {"unit": "millimeters"}


def test_import_typed_dataset_copies_files_to_uploads(db_session, tmp_path):
    src = tmp_path / "src"
    _build_tiny_dataset_on_disk(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, _ = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
    )
    db.session.commit()

    layout = sorted(
        p.relative_to(uploads).as_posix()
        for p in (uploads / "datasets" / str(ds_id)).rglob("*")
        if p.is_file()
    )
    assert layout == [
        f"datasets/{ds_id}/depth_gt/s0.npz",
        f"datasets/{ds_id}/depth_gt/s1.npz",
        f"datasets/{ds_id}/image/s0.png",
        f"datasets/{ds_id}/image/s1.png",
    ]


def test_import_typed_dataset_missing_file_raises(db_session, tmp_path):
    src = tmp_path / "src"
    _build_tiny_dataset_on_disk(src)
    # Manifest references s0+s1 but only s0 is on disk for one field.
    (src / "image" / "s1.png").unlink()
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    with pytest.raises(FileNotFoundError, match="image/s1.png"):
        import_typed_dataset(
            src,
            db_session=db.session,
            Dataset=Dataset, Sample=Sample, CustomField=CustomField,
            upload_folder=str(uploads),
        )


# ---------------------------------------------------------------------------
# Sparse / optional fields (e.g. OpenFake's generation `prompt`, null for
# real images): the pre-flight must tolerate the per-sample gaps instead of
# failing the whole import, and keep the samples that DO have the field.
# ---------------------------------------------------------------------------

def _build_sparse_dataset_on_disk(root: Path) -> None:
    """Three samples; image present for all, optional `prompt` present
    only for s1 (mirrors a real-vs-synthetic dataset)."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "imp_sparse",
        "version": "1.0",
        "fields": [
            {"name": "image",  "kind": "image", "role": "input"},
            {"name": "prompt", "kind": "text",  "role": "input", "optional": True},
        ],
        "samples": ["s0", "s1", "s2"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "image").mkdir()
    (root / "prompt").mkdir()
    for s in ("s0", "s1", "s2"):
        img = bh.Image(np.zeros((4, 4, 3), dtype=np.uint8))
        (root / "image" / f"{s}.png").write_bytes(img.encode())
    # Only the synthetic sample carries a prompt.
    (root / "prompt" / "s1.txt").write_bytes(bh.Text("a cat").encode())


def test_validate_rejects_non_bool_optional():
    m = _ok_manifest()
    m["fields"][0]["optional"] = "yes"
    with pytest.raises(ValueError, match="optional must be a boolean"):
        validate_manifest(m)


def test_import_typed_dataset_optional_field_tolerates_gaps(db_session, tmp_path):
    src = tmp_path / "src"
    _build_sparse_dataset_on_disk(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, summary = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
    )
    db.session.commit()

    # No sample dropped: all three keep their image.
    ds = Dataset.query.get(ds_id)
    assert {s.name for s in ds.samples} == {"s0", "s1", "s2"}
    # The prompt CustomField exists only for the sample that had one.
    prompts = (
        CustomField.query.join(Sample, CustomField.sample_id == Sample.id)
        .filter(Sample.dataset_id == ds_id, CustomField.name == "prompt")
        .all()
    )
    assert len(prompts) == 1
    s1 = Sample.query.filter_by(dataset_id=ds_id, name="s1").one()
    assert prompts[0].sample_id == s1.id


def test_import_typed_dataset_required_field_still_strict(db_session, tmp_path):
    """A non-optional field with a hole must still hard-fail — the
    sparse tolerance is opt-in per field, not a blanket relaxation."""
    src = tmp_path / "src"
    _build_sparse_dataset_on_disk(src)
    # Drop the `optional` flag: now `prompt` is required and the s0/s2
    # gaps must raise.
    manifest = json.loads((src / "manifest.json").read_text())
    for f in manifest["fields"]:
        f.pop("optional", None)
    (src / "manifest.json").write_text(json.dumps(manifest))
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    with pytest.raises(FileNotFoundError, match="prompt/"):
        import_typed_dataset(
            src,
            db_session=db.session,
            Dataset=Dataset, Sample=Sample, CustomField=CustomField,
            upload_folder=str(uploads),
        )
