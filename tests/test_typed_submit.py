"""End-to-end tests for the typed-submission ingest path (Phase B Task 11)."""
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
    Leaderboard,
    Sample,
    Submission,
    User,
    db,
    generate_api_token,
)
from benchhub.manifest import (
    check_submission_matches_contract,
    validate_submission_manifest,
)


# ---------------------------------------------------------------------------
# Submission manifest schema validation
# ---------------------------------------------------------------------------

def _ok_sub_manifest() -> dict:
    return {
        "name": "my-model",
        "predictions": [
            {"name": "depth_pred", "kind": "depth", "params": {"unit": "meters"}},
        ],
        "samples": ["s0", "s1"],
    }


def test_validate_sub_manifest_accepts_well_formed():
    validate_submission_manifest(_ok_sub_manifest())


@pytest.mark.parametrize("missing", ["name", "predictions", "samples"])
def test_validate_sub_manifest_rejects_missing_top_level(missing):
    m = _ok_sub_manifest()
    del m[missing]
    with pytest.raises(ValueError, match=missing):
        validate_submission_manifest(m)


def test_validate_sub_manifest_rejects_unknown_kind():
    m = _ok_sub_manifest()
    m["predictions"][0]["kind"] = "not-a-kind"
    with pytest.raises(ValueError, match="not in DTYPES"):
        validate_submission_manifest(m)


def test_validate_sub_manifest_rejects_duplicate_pred_names():
    m = _ok_sub_manifest()
    m["predictions"].append(dict(m["predictions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission_manifest(m)


# ---------------------------------------------------------------------------
# Contract checker
# ---------------------------------------------------------------------------

def test_contract_check_passes_when_kinds_match():
    contract = [{"name": "depth_pred", "kind": "depth", "role": "pred"}]
    manifest = _ok_sub_manifest()
    check_submission_matches_contract(manifest, contract)


def test_contract_check_rejects_missing_required_pred():
    contract = [
        {"name": "depth_pred",  "kind": "depth", "role": "pred"},
        {"name": "second_pred", "kind": "scalar", "role": "pred"},
    ]
    with pytest.raises(ValueError, match="missing required prediction fields"):
        check_submission_matches_contract(_ok_sub_manifest(), contract)


def test_contract_check_rejects_kind_mismatch():
    contract = [{"name": "depth_pred", "kind": "mask", "role": "pred"}]
    with pytest.raises(ValueError, match="kind"):
        check_submission_matches_contract(_ok_sub_manifest(), contract)


def test_contract_check_ignores_non_pred_roles():
    """Input + GT entries in required_pred_fields_json don't constrain the submission."""
    contract = [
        {"name": "image",      "kind": "image", "role": "input"},
        {"name": "gt_depth",   "kind": "depth", "role": "gt"},
        {"name": "depth_pred", "kind": "depth", "role": "pred"},
    ]
    check_submission_matches_contract(_ok_sub_manifest(), contract)


def test_contract_check_empty_contract_is_permissive():
    """LBs that haven't declared a pred contract accept any submission."""
    check_submission_matches_contract(_ok_sub_manifest(), [])


# ---------------------------------------------------------------------------
# Route-level end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def lb_with_pred_contract(db_session):
    ds = Dataset(name='subroute_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    lb = Leaderboard(
        name='subroute_lb', summary_metrics='', visibility='public',
        required_pred_fields_json=json.dumps([
            {"name": "depth_pred", "kind": "depth", "params": {"unit": "meters"}, "role": "pred"},
        ]),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb


@pytest.fixture
def api_user(db_session):
    u = User(
        email='api@bench.local', display_name='api',
        oauth_provider='github', oauth_sub='api-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def _build_submission_zip(*, predictions: list[dict], samples: list[str], values_by_field: dict) -> bytes:
    """Pack a submission manifest + per-(field, sample) files into an
    in-memory ZIP and return its bytes."""
    manifest = {
        "name": "test-sub",
        "predictions": predictions,
        "samples": samples,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for p in predictions:
            cls = bh.DTYPES[p["kind"]]
            ext = cls.file_ext or ".txt"
            for s in samples:
                inst = values_by_field[p["name"]][s]
                zf.writestr(f"{p['name']}/{s}{ext}", inst.encode())
    return buf.getvalue()


def test_typed_submit_happy_path(client, lb_with_pred_contract, api_user):
    """Valid submission lands in DB, files copied, response is 201."""
    arr = np.ones((4, 4), dtype=np.float32)
    body = _build_submission_zip(
        predictions=[
            {"name": "depth_pred", "kind": "depth", "params": {"unit": "meters"}},
        ],
        samples=["s0", "s1"],
        values_by_field={
            "depth_pred": {
                "s0": bh.Depth(arr, unit="meters"),
                "s1": bh.Depth(arr * 2, unit="meters"),
            },
        },
    )
    resp = client.post(
        f"/api/submit/{lb_with_pred_contract.id}",
        data={"submission_zip": (io.BytesIO(body), "sub.zip"), "name": "rn50"},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.data
    payload = resp.get_json()
    assert payload["leaderboard_id"] == lb_with_pred_contract.id
    assert payload["predictions"] == 1
    assert payload["samples"] == 2

    sub = Submission.query.get(payload["submission_id"])
    assert sub is not None
    assert sub.name == "rn50"
    cfs = CustomField.query.filter_by(submission_id=sub.id).all()
    assert len(cfs) == 2
    assert {cf.sample_name for cf in cfs} == {"s0", "s1"}
    assert all(cf.data_type == "depth" for cf in cfs)
    assert all(cf.get_params() == {"unit": "meters"} for cf in cfs)


def test_typed_submit_requires_api_token(client, lb_with_pred_contract):
    resp = client.post(
        f"/api/submit/{lb_with_pred_contract.id}",
        data={"submission_zip": (io.BytesIO(b""), "x.zip")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401


def test_typed_submit_rejects_kind_mismatch(client, lb_with_pred_contract, api_user):
    body = _build_submission_zip(
        predictions=[
            # Contract requires depth, submitter sent scalar.
            {"name": "depth_pred", "kind": "scalar"},
        ],
        samples=["s0"],
        values_by_field={"depth_pred": {"s0": bh.Scalar(0.5)}},
    )
    resp = client.post(
        f"/api/submit/{lb_with_pred_contract.id}",
        data={"submission_zip": (io.BytesIO(body), "sub.zip")},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"kind" in resp.data


def test_typed_submit_rejects_missing_pred_field(client, lb_with_pred_contract, api_user):
    body = _build_submission_zip(
        predictions=[
            # Contract requires `depth_pred`; we send `other_pred` instead.
            {"name": "other_pred", "kind": "scalar"},
        ],
        samples=["s0"],
        values_by_field={"other_pred": {"s0": bh.Scalar(0.5)}},
    )
    resp = client.post(
        f"/api/submit/{lb_with_pred_contract.id}",
        data={"submission_zip": (io.BytesIO(body), "sub.zip")},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"missing required prediction fields" in resp.data


def test_typed_submit_rejects_corrupt_zip(client, lb_with_pred_contract, api_user):
    resp = client.post(
        f"/api/submit/{lb_with_pred_contract.id}",
        data={"submission_zip": (io.BytesIO(b"not a zip"), "sub.zip")},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"ZIP" in resp.data
