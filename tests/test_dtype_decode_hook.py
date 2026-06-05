"""End-to-end coverage for the user-registered-kind decode hook.

A registered data type carries an optional ``decode(blob, params)`` that is
the deserialize side of the contract: when present, a metric consuming that
kind receives the decoded object instead of the raw stored bytes (mirroring
how a built-in kind hands a metric a ``bh.Depth`` with ``.array``).

This exercises the whole GT/input path:
  register kind → import a dataset carrying it (bytes verbatim) →
  get_metric_context emits a RegisteredBlob → evaluate_dynamic_metric runs
  the decode hook in-process and the metric sees the decoded value.

The pure sandbox/harness JSON round-trip lives in test_sandbox_typed.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import CustomField, DataTypeDef, Dataset, Sample, db
from benchhub.manifest import import_typed_dataset
from metric_engine import (
    RegisteredBlob,
    evaluate_dynamic_metric,
    get_metric_context,
)

# A toy registered kind: stores "1,2,3,4" text bytes; decode() parses them
# into a list of floats so a metric can average them.
_DECODE = (
    "def decode(blob, params):\n"
    "    return [float(x) for x in blob.decode().split(',') if x]\n"
)


def _register_vec_kind():
    dt = DataTypeDef(name="vec", file_ext=".vec", viz_mime="image/png",
                     decode_code=_DECODE, visibility="public")
    db.session.add(dt)
    db.session.commit()
    return dt


def _build_dataset_with_vec_kind(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "vec_ds",
        "version": "1.0",
        "fields": [{"name": "vec_gt", "kind": "vec", "role": "gt"}],
        "samples": ["s0", "s1"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "vec_gt").mkdir()
    (root / "vec_gt" / "s0.vec").write_bytes(b"1,2,3,4")
    (root / "vec_gt" / "s1.vec").write_bytes(b"10,20,30")


def test_import_admits_registered_kind_and_stores_bytes_verbatim(db_session, tmp_path):
    _register_vec_kind()
    src = tmp_path / "src"
    _build_dataset_with_vec_kind(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, summary = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
        extra_kinds={"vec": ".vec"},
    )
    db.session.commit()

    assert summary["custom_field_rows"] == 2
    cf = (CustomField.query
          .join(Sample, CustomField.sample_id == Sample.id)
          .filter(Sample.dataset_id == ds_id, CustomField.name == "vec_gt",
                  Sample.name == "s0")
          .first())
    assert cf is not None and cf.data_type == "vec"
    # File-backed: value_text is the relative path; bytes copied verbatim.
    stored = (uploads / cf.value_text).read_bytes()
    assert stored == b"1,2,3,4"


def test_metric_context_emits_registered_blob(db_session, tmp_path):
    _register_vec_kind()
    src = tmp_path / "src"
    _build_dataset_with_vec_kind(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    ds_id, _ = import_typed_dataset(
        src, db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads), extra_kinds={"vec": ".vec"})
    db.session.commit()

    sample = Sample.query.filter_by(dataset_id=ds_id, name="s0").first()
    ctx = get_metric_context(sample, upload_folder=str(uploads))
    rb = ctx.get("gt_vec_gt")
    assert isinstance(rb, RegisteredBlob)
    assert rb.kind == "vec"
    assert rb.blob == b"1,2,3,4"
    assert rb.decode_code  # carries the kind's decode hook


def test_metric_receives_decoded_value_in_process(db_session, tmp_path):
    _register_vec_kind()
    src = tmp_path / "src"
    _build_dataset_with_vec_kind(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    ds_id, _ = import_typed_dataset(
        src, db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads), extra_kinds={"vec": ".vec"})
    db.session.commit()

    sample = Sample.query.filter_by(dataset_id=ds_id, name="s0").first()
    ctx = get_metric_context(sample, upload_folder=str(uploads))

    # A fake GlobalMetric: the in-process evaluator just needs .python_code
    # and .input_kinds. decode() yields [1,2,3,4]; the metric returns the mean.
    class _M:
        name = "vec_mean"
        python_code = ("def vec_mean(gt):\n"
                       "    return sum(gt) / len(gt)\n")
        input_kinds = json.dumps(["vec"])

    val, err = evaluate_dynamic_metric(_M(), ctx, json.dumps({"gt": "gt_vec_gt"}))
    assert err is None
    assert val == 2.5  # mean of 1,2,3,4


def test_registered_kind_without_decode_passes_raw_bytes(db_session, tmp_path):
    # A kind with no decode hook: the metric receives the raw bytes.
    dt = DataTypeDef(name="rawk", file_ext=".bin", viz_mime="image/png",
                     decode_code=None, visibility="public")
    db.session.add(dt)
    db.session.commit()

    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text(json.dumps({
        "name": "raw_ds", "version": "1.0",
        "fields": [{"name": "raw_gt", "kind": "rawk", "role": "gt"}],
        "samples": ["s0"],
    }))
    (src / "raw_gt").mkdir()
    (src / "raw_gt" / "s0.bin").write_bytes(b"\x00\x01ABC")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    ds_id, _ = import_typed_dataset(
        src, db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads), extra_kinds={"rawk": ".bin"})
    db.session.commit()

    sample = Sample.query.filter_by(dataset_id=ds_id, name="s0").first()
    ctx = get_metric_context(sample, upload_folder=str(uploads))

    class _M:
        name = "byte_len"
        python_code = "def byte_len(gt):\n    return len(gt)\n"
        input_kinds = json.dumps(["rawk"])

    val, err = evaluate_dynamic_metric(_M(), ctx, json.dumps({"gt": "gt_raw_gt"}))
    assert err is None
    assert val == 5.0  # len(b"\x00\x01ABC")
