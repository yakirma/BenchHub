"""Tests for benchhub.client — the typed submission builder + transport."""
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


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def test_client_picks_up_env_token(monkeypatch):
    monkeypatch.setenv("BENCHHUB_API_TOKEN", "tok-from-env")
    monkeypatch.setenv("BENCHHUB_BASE_URL", "https://example.test")
    c = bh.Client()
    assert c.token == "tok-from-env"
    assert c.base_url == "https://example.test"


def test_client_ctor_overrides_env(monkeypatch):
    monkeypatch.setenv("BENCHHUB_API_TOKEN", "tok-env")
    c = bh.Client(token="tok-ctor", base_url="http://localhost:6060/")
    assert c.token == "tok-ctor"
    assert c.base_url == "http://localhost:6060"  # trailing slash stripped


def test_client_without_token_raises_on_submit():
    c = bh.Client(token="", base_url="http://x")
    sub = c.submission(leaderboard_id=1)
    sub.predict("s0", x=bh.Scalar(0.5))
    with pytest.raises(ValueError, match="API token"):
        sub.submit()


# ---------------------------------------------------------------------------
# SubmissionBuilder — staging + manifest + zip
# ---------------------------------------------------------------------------

def test_builder_predict_rejects_non_datatype():
    sub = bh.Client(token="t", base_url="http://x").submission(leaderboard_id=1)
    with pytest.raises(TypeError, match="DataType"):
        sub.predict("s0", x=0.5)  # bare float


def test_builder_predict_validates_each_instance():
    """Calling .predict() validates the typed value — bad shape → ValueError."""
    sub = bh.Client(token="t", base_url="http://x").submission(leaderboard_id=1)
    bad_image = bh.Image(np.zeros((4, 4, 5), dtype=np.uint8))  # 5 channels: invalid
    with pytest.raises(ValueError):
        sub.predict("s0", img=bad_image)


def test_builder_manifest_minimal_round_trip():
    sub = bh.Client(token="t", base_url="http://x").submission(
        leaderboard_id=42, name="my-model",
    )
    sub.predict("s0", depth_pred=bh.Depth(np.zeros((4, 4), dtype=np.float32), unit="meters"))
    sub.predict("s1", depth_pred=bh.Depth(np.ones((4, 4), dtype=np.float32), unit="meters"))

    manifest = sub.build_manifest()
    assert manifest["name"] == "my-model"
    assert manifest["samples"] == ["s0", "s1"]
    assert len(manifest["predictions"]) == 1
    pred = manifest["predictions"][0]
    assert pred["name"] == "depth_pred"
    assert pred["kind"] == "depth"
    assert pred["params"] == {"unit": "meters"}


def test_builder_manifest_empty_raises():
    sub = bh.Client(token="t", base_url="http://x").submission(leaderboard_id=1)
    with pytest.raises(ValueError, match="no predictions"):
        sub.build_manifest()


def test_builder_manifest_rejects_mixed_kinds_per_field():
    sub = bh.Client(token="t", base_url="http://x").submission(leaderboard_id=1)
    sub.predict("s0", out=bh.Scalar(1.0))
    sub.predict("s1", out=bh.Label("cat"))
    with pytest.raises(ValueError, match="mixed types"):
        sub.build_manifest()


def test_builder_zip_layout_matches_server_expectations():
    sub = bh.Client(token="t", base_url="http://x").submission(
        leaderboard_id=42, name="sub",
    )
    sub.predict("s0", depth=bh.Depth(np.zeros((4, 4), dtype=np.float32), unit="meters"))
    sub.predict("s1", depth=bh.Depth(np.ones((4, 4), dtype=np.float32), unit="meters"))

    blob = sub.build_zip()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = sorted(zf.namelist())
    assert names == ["depth/s0.npz", "depth/s1.npz", "manifest.json"]


def test_builder_zip_inline_kinds_use_txt_ext():
    sub = bh.Client(token="t", base_url="http://x").submission(leaderboard_id=1)
    sub.predict("s0", label=bh.Label("cat"))
    blob = sub.build_zip()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert "label/s0.txt" in zf.namelist()


def test_builder_zip_rejects_partial_per_sample_field_set():
    """Every sample must supply every staged field — server-side validation."""
    sub = bh.Client(token="t", base_url="http://x").submission(leaderboard_id=1)
    sub.predict("s0", a=bh.Scalar(0.5), b=bh.Scalar(0.7))
    sub.predict("s1", a=bh.Scalar(0.6))  # missing `b`
    with pytest.raises(ValueError, match="missing prediction"):
        sub.build_zip()


# ---------------------------------------------------------------------------
# End-to-end via FlaskTestClientTransport — same code path as a real submit
# ---------------------------------------------------------------------------

@pytest.fixture
def lb_with_depth_pred(db_session):
    ds = Dataset(name='client_e2e_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    lb = Leaderboard(
        name='client_e2e_lb', summary_metrics='', visibility='public',
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
        email='clienttest@bench.local', display_name='ct',
        oauth_provider='github', oauth_sub='ct-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def test_end_to_end_submit_via_flask_transport(client, lb_with_depth_pred, api_user):
    bh_client = bh.Client(
        token=api_user.api_token,
        base_url='http://test',
        transport=bh.FlaskTestClientTransport(client),
    )
    sub = bh_client.submission(lb_with_depth_pred.id, name='resnet50-v1')
    arr = np.ones((4, 4), dtype=np.float32)
    sub.predict('s0', depth_pred=bh.Depth(arr, unit='meters'))
    sub.predict('s1', depth_pred=bh.Depth(arr * 2, unit='meters'))

    result = sub.submit()
    assert result["leaderboard_id"] == lb_with_depth_pred.id
    assert result["predictions"] == 1
    assert result["samples"] == 2

    persisted = Submission.query.get(result["submission_id"])
    assert persisted is not None
    assert persisted.name == "resnet50-v1"
    cfs = CustomField.query.filter_by(submission_id=persisted.id).all()
    assert {cf.sample_name for cf in cfs} == {"s0", "s1"}


def test_end_to_end_submit_propagates_contract_violation(client, lb_with_depth_pred, api_user):
    """A submission that violates the LB contract gets a 400 from the
    server; the client surfaces it as BenchHubAPIError."""
    bh_client = bh.Client(
        token=api_user.api_token,
        base_url='http://test',
        transport=bh.FlaskTestClientTransport(client),
    )
    sub = bh_client.submission(lb_with_depth_pred.id)
    # Contract demands `depth_pred` (kind=depth); we send `other_pred` (scalar).
    sub.predict('s0', other_pred=bh.Scalar(0.5))
    with pytest.raises(bh.BenchHubAPIError) as excinfo:
        sub.submit()
    assert excinfo.value.status_code == 400
    assert "missing required prediction fields" in excinfo.value.payload["error"]
