"""Phase: schema-at-the-dataset-level.

`DatasetField` is the single source of truth for what kind of data
each named field holds and which role it plays (input vs gt). A
leaderboard's prediction wire-contract is derived from the union of
attached datasets' GT fields — `Leaderboard.required_pred_fields_json`
stays as an override for LBs that need a custom contract.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import benchhub as bh
from app import (
    CustomField,
    Dataset,
    DatasetField,
    Leaderboard,
    Sample,
    Submission,
    User,
    _lb_pred_contract_from_dataset_fields,
    db,
    generate_api_token,
)
from benchhub.manifest import import_typed_dataset


# ---------------------------------------------------------------------------
# import_typed_dataset writes DatasetField rows
# ---------------------------------------------------------------------------

def _write_synth_dataset(root: Path) -> None:
    """Two samples × three fields (image input + depth gt + label gt)."""
    manifest = {
        "name": "schema_synth",
        "version": "1.0",
        "fields": [
            {"name": "image",    "kind": "image", "role": "input"},
            {"name": "depth_gt", "kind": "depth", "role": "gt",
             "params": {"unit": "millimeters"}},
            {"name": "label",    "kind": "label", "role": "gt"},
        ],
        "samples": ["s0", "s1"],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "image").mkdir()
    (root / "depth_gt").mkdir()
    (root / "label").mkdir()
    for s in ("s0", "s1"):
        (root / "image" / f"{s}.png").write_bytes(
            bh.Image(np.zeros((4, 4, 3), dtype=np.uint8)).encode()
        )
        (root / "depth_gt" / f"{s}.npz").write_bytes(
            bh.Depth(np.ones((4, 4), dtype=np.float32),
                     unit="millimeters").encode()
        )
        (root / "label" / f"{s}.txt").write_bytes(
            bh.Label("cat" if s == "s0" else "dog").encode()
        )


def test_importer_writes_one_dataset_field_per_declared_field(db_session, tmp_path):
    src = tmp_path / "src"; uploads = tmp_path / "uploads"; uploads.mkdir()
    _write_synth_dataset(src)
    ds_id, _ = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        DatasetField=DatasetField,
        upload_folder=str(uploads),
    )
    db.session.commit()

    schema = DatasetField.query.filter_by(dataset_id=ds_id).all()
    by_name = {f.name: f for f in schema}
    assert set(by_name) == {"image", "depth_gt", "label"}

    assert by_name["image"].kind == "image"
    assert by_name["image"].role == "input"
    assert by_name["depth_gt"].kind == "depth"
    assert by_name["depth_gt"].role == "gt"
    assert by_name["depth_gt"].get_params() == {"unit": "millimeters"}
    assert by_name["label"].kind == "label"
    assert by_name["label"].role == "gt"


def test_dataset_field_uniqueness_per_dataset(db_session):
    ds = Dataset(name="uniq_ds", visibility="public")
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name="x", kind="scalar", role="gt"))
    db.session.commit()

    db.session.add(DatasetField(dataset_id=ds.id, name="x", kind="scalar", role="gt"))
    with pytest.raises(Exception):
        db.session.commit()


# ---------------------------------------------------------------------------
# LB pred contract derivation
# ---------------------------------------------------------------------------

def _seed_dataset_with_schema(name: str, fields: list[dict]) -> Dataset:
    ds = Dataset(name=name, visibility="public")
    db.session.add(ds); db.session.flush()
    for f in fields:
        df = DatasetField(
            dataset_id=ds.id, name=f["name"], kind=f["kind"],
            role=f.get("role", "gt"),
        )
        if f.get("params"):
            df.set_params(f["params"])
        db.session.add(df)
    db.session.commit()
    return ds


def test_lb_contract_collects_explicit_pred_fields(db_session):
    """The LB pred contract comes from explicit role=pred
    DatasetField rows. GT-only datasets get an empty contract —
    the engine never invents `<name>_pred` mirror entries."""
    ds = _seed_dataset_with_schema("c1_ds", [
        {"name": "image",    "kind": "image", "role": "input"},
        {"name": "depth_gt", "kind": "depth", "role": "gt",
         "params": {"unit": "meters"}},
        {"name": "label",    "kind": "label", "role": "gt"},
        {"name": "depth_gt_pred", "kind": "depth", "role": "pred",
         "params": {"unit": "meters"}},
        {"name": "label_pred",    "kind": "label", "role": "pred"},
    ])
    lb = Leaderboard(name="c1_lb", summary_metrics="", visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    contract = _lb_pred_contract_from_dataset_fields(lb)
    by_name = {e["name"]: e for e in contract}
    # Inputs are not turned into preds.
    assert "image_pred" not in by_name
    # GT-only fields without an explicit pred get nothing.
    assert by_name["depth_gt_pred"]["kind"] == "depth"
    assert by_name["depth_gt_pred"]["params"] == {"unit": "meters"}
    assert by_name["depth_gt_pred"]["role"] == "pred"
    assert by_name["label_pred"]["kind"] == "label"


def test_lb_contract_unions_explicit_pred_across_multiple_datasets(db_session):
    """LB attached to two datasets unions their explicit pred
    fields. GT-only fields contribute nothing to the contract."""
    ds_a = _seed_dataset_with_schema("u_a", [
        {"name": "label",      "kind": "label", "role": "gt"},
        {"name": "label_pred", "kind": "label", "role": "pred"},
    ])
    ds_b = _seed_dataset_with_schema("u_b", [
        {"name": "depth",      "kind": "depth", "role": "gt"},
        {"name": "depth_pred", "kind": "depth", "role": "pred",
         "params": {"unit": "meters"}},
    ])
    lb = Leaderboard(name="u_lb", summary_metrics="", visibility="public")
    lb.datasets.extend([ds_a, ds_b])
    db.session.add(lb); db.session.commit()

    names = {e["name"] for e in _lb_pred_contract_from_dataset_fields(lb)}
    assert names == {"label_pred", "depth_pred"}


def test_lb_contract_empty_when_no_explicit_pred_declared(db_session):
    """No GT-mirror fallback — a GT-only dataset gives an empty
    contract. Submissions against such an LB are rejected at
    manifest-validation time with a clear "no pred fields" error
    rather than the engine inventing a contract the dataset never
    promised."""
    ds = _seed_dataset_with_schema("no_pred_ds", [
        {"name": "label", "kind": "label", "role": "gt"},
    ])
    lb = Leaderboard(name="no_pred_lb", summary_metrics="", visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    assert _lb_pred_contract_from_dataset_fields(lb) == []


def test_lb_explicit_required_pred_fields_overrides_derivation(db_session):
    """An LB with `required_pred_fields_json` set keeps full control —
    the dataset-derived contract is ignored. Useful when the LB needs
    to restrict / rename pred fields differently from the underlying
    dataset's GT schema."""
    ds = _seed_dataset_with_schema("o_ds", [
        {"name": "depth_gt", "kind": "depth", "role": "gt"},
        {"name": "label",    "kind": "label", "role": "gt"},
    ])
    lb = Leaderboard(
        name="o_lb", summary_metrics="", visibility="public",
        required_pred_fields_json=json.dumps([
            {"name": "depth_gt_pred", "kind": "depth", "params": {"unit": "millimeters"}, "role": "pred"},
        ]),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    contract = _lb_pred_contract_from_dataset_fields(lb)
    assert [e["name"] for e in contract] == ["depth_gt_pred"]
    assert contract[0]["params"] == {"unit": "millimeters"}


def test_lb_with_no_attached_datasets_has_empty_contract(db_session):
    lb = Leaderboard(name="e_lb", summary_metrics="", visibility="public")
    db.session.add(lb); db.session.commit()
    assert _lb_pred_contract_from_dataset_fields(lb) == []


# ---------------------------------------------------------------------------
# End-to-end: typed submission against a dataset-derived contract
# ---------------------------------------------------------------------------

@pytest.fixture
def api_user(db_session):
    u = User(
        email='schemaapi@bench.local', display_name='schema',
        oauth_provider='github', oauth_sub='schema-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def test_typed_submit_validates_against_dataset_derived_contract(
    client, api_user, db_session,
):
    """LB has NO required_pred_fields_json; the dataset's explicit
    role=pred fields are the contract. A submission with the wrong
    kind for that field gets a 400."""
    ds = _seed_dataset_with_schema("e2e_ds", [
        {"name": "depth_gt",      "kind": "depth", "role": "gt"},
        {"name": "depth_gt_pred", "kind": "depth", "role": "pred"},
    ])
    db.session.add(Sample(dataset_id=ds.id, name="s0"))
    lb = Leaderboard(name="e2e_lb", summary_metrics="", visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    # Build a submission ZIP with the wrong kind (scalar) for depth_gt_pred.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({
            "name": "wrong-kind",
            "predictions": [
                {"name": "depth_gt_pred", "kind": "scalar"},
            ],
            "samples": ["s0"],
        }))
        zf.writestr("depth_gt_pred/s0.txt", bh.Scalar(0.5).encode())

    resp = client.post(
        f"/api/submit/{lb.id}",
        data={"submission_zip": (io.BytesIO(buf.getvalue()), "sub.zip")},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"kind" in resp.data
