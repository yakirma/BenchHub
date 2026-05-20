"""shape_match constraint on image/mask/depth pred fields.

End-to-end coverage for both enforcement layers:

  * server-side via `import_typed_submission(get_input_shape=...)`
    refusing wrong-shape preds with a 400 from `/api/submit/<lb>`.
  * client-side via `SubmissionBuilder.set_contract` +
    `.set_input_shape(...)`, refusing before any upload.
  * `/api/leaderboard/<id>/contract` returns the live contract,
    including shape_match in params.
"""
from __future__ import annotations

import io
import json
import zipfile

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
    db,
    generate_api_token,
)


# ===========================================================================
# Server-side enforcement
# ===========================================================================

@pytest.fixture
def lb_with_shape_constrained_pred(db_session, tmp_path, monkeypatch):
    """LB attached to a dataset that declares:
        - image:    input  (kind=image)
        - depth_pred: pred (kind=depth, params={'shape_match': 'image'})
    Two samples (s0, s1) have on-disk image inputs of different sizes
    so a single 'one-size-fits-all' pred can never satisfy both."""
    from app import app as flask_app

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(uploads))

    ds = Dataset(name="shape_ds", visibility="public")
    db.session.add(ds); db.session.flush()

    # Per-sample input images with different shapes.
    shapes = {"s0": (16, 24), "s1": (8, 8)}
    for name, (h, w) in shapes.items():
        s = Sample(dataset_id=ds.id, name=name)
        db.session.add(s); db.session.flush()
        img_dir = uploads / "datasets" / str(ds.id) / "image"
        img_dir.mkdir(parents=True, exist_ok=True)
        png = bh.Image(np.zeros((h, w, 3), dtype=np.uint8)).encode()
        (img_dir / f"{name}.png").write_bytes(png)
        db.session.add(CustomField(
            sample_id=s.id, name="image", data_type="image",
            value_text=str((img_dir / f"{name}.png").relative_to(uploads)),
        ))

    # Schema: image (input) + depth_pred (pred, shape_match=image).
    db.session.add(DatasetField(
        dataset_id=ds.id, name="image", kind="image", role="input",
    ))
    pred_field = DatasetField(
        dataset_id=ds.id, name="depth_pred", kind="depth", role="pred",
    )
    pred_field.set_params({"shape_match": "image"})
    db.session.add(pred_field)

    lb = Leaderboard(name="shape_lb", summary_metrics="", visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb, shapes


@pytest.fixture
def api_user(db_session):
    u = User(
        email="shape@bench.local", display_name="sh",
        oauth_provider="github", oauth_sub="sh-1",
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def _build_submission_zip(predictions, samples, values_by_field):
    """Pack a submission manifest + per-(field, sample) typed bytes."""
    manifest = {
        "name": "shape-test", "predictions": predictions, "samples": samples,
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


def test_server_accepts_matching_shape(client, lb_with_shape_constrained_pred, api_user):
    """Both samples submit a depth pred whose shape matches the
    corresponding input — server accepts."""
    lb, shapes = lb_with_shape_constrained_pred
    preds = [{"name": "depth_pred", "kind": "depth", "params": {"shape_match": "image"}}]
    values = {"depth_pred": {
        "s0": bh.Depth(np.zeros(shapes["s0"], dtype=np.float32), unit="meters"),
        "s1": bh.Depth(np.zeros(shapes["s1"], dtype=np.float32), unit="meters"),
    }}
    body = _build_submission_zip(preds, ["s0", "s1"], values)
    r = client.post(
        f"/api/submit/{lb.id}",
        data={"submission_zip": (io.BytesIO(body), "sub.zip")},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert r.status_code == 201, r.data


def test_server_rejects_mismatched_shape(client, lb_with_shape_constrained_pred, api_user):
    """s0's depth pred uses the WRONG shape — server bounces 400."""
    lb, shapes = lb_with_shape_constrained_pred
    preds = [{"name": "depth_pred", "kind": "depth", "params": {"shape_match": "image"}}]
    bad = (shapes["s0"][0] + 4, shapes["s0"][1])  # wrong H
    values = {"depth_pred": {
        "s0": bh.Depth(np.zeros(bad, dtype=np.float32), unit="meters"),
        "s1": bh.Depth(np.zeros(shapes["s1"], dtype=np.float32), unit="meters"),
    }}
    body = _build_submission_zip(preds, ["s0", "s1"], values)
    r = client.post(
        f"/api/submit/{lb.id}",
        data={"submission_zip": (io.BytesIO(body), "sub.zip")},
        headers={"Authorization": f"Bearer {api_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert b"shape_match" in r.data
    assert b"s0" in r.data


def test_server_ignores_shape_match_when_pred_kind_is_inline(client, db_session, tmp_path, monkeypatch):
    """A pred declared with `shape_match` on an inline kind like
    `scalar` has no spatial shape — the importer skips that check
    rather than 400-ing every submission to such an LB."""
    from app import app as flask_app
    uploads = tmp_path / "uploads"; uploads.mkdir()
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(uploads))

    ds = Dataset(name="inline_shape", visibility="public")
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name="s0")
    db.session.add(s); db.session.flush()
    pred = DatasetField(dataset_id=ds.id, name="score_pred", kind="scalar", role="pred")
    pred.set_params({"shape_match": "image"})  # nonsensical but allowed
    db.session.add(pred)
    lb = Leaderboard(name="inline_shape_lb", summary_metrics="", visibility="public")
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    u = User(email='inline@bench.local', display_name='i',
             oauth_provider='github', oauth_sub='i-1', api_token=generate_api_token())
    db.session.add(u); db.session.commit()

    preds = [{"name": "score_pred", "kind": "scalar", "params": {"shape_match": "image"}}]
    body = _build_submission_zip(preds, ["s0"], {"score_pred": {"s0": bh.Scalar(0.5)}})
    r = client.post(f"/api/submit/{lb.id}",
                    data={"submission_zip": (io.BytesIO(body), "sub.zip")},
                    headers={"Authorization": f"Bearer {u.api_token}"},
                    content_type="multipart/form-data")
    assert r.status_code == 201


# ===========================================================================
# /api/leaderboard/<id>/contract endpoint
# ===========================================================================

def test_contract_endpoint_returns_shape_match_in_params(
    client, lb_with_shape_constrained_pred,
):
    lb, _ = lb_with_shape_constrained_pred
    r = client.get(f"/api/leaderboard/{lb.id}/contract")
    assert r.status_code == 200
    contract = r.get_json()
    by_name = {e["name"]: e for e in contract}
    assert by_name["depth_pred"]["kind"] == "depth"
    assert by_name["depth_pred"]["params"]["shape_match"] == "image"


def test_contract_endpoint_404s_on_private_lb_to_anon(client, db_session):
    owner = User(email='pl@bench.local', display_name='p',
                 oauth_provider='github', oauth_sub='pl-1')
    db.session.add(owner); db.session.flush()
    lb = Leaderboard(name="priv_lb", summary_metrics="", visibility="private",
                     owner_user_id=owner.id)
    db.session.add(lb); db.session.commit()
    r = client.get(f"/api/leaderboard/{lb.id}/contract")
    assert r.status_code == 404


# ===========================================================================
# Client-side enforcement via SubmissionBuilder
# ===========================================================================

def _local_client() -> bh.Client:
    return bh.Client(token="t", base_url="http://x")


def test_builder_validates_shape_match_locally_when_contract_and_input_known():
    """Contract says depth_pred.shape_match=image; user registers
    input shape; mismatched pred raises before any upload."""
    sub = _local_client().submission(leaderboard_id=1)
    sub.set_contract([{
        "name": "depth_pred", "kind": "depth", "role": "pred",
        "params": {"shape_match": "image"},
    }])
    sub.set_input_shape("s0", image=(32, 32))
    sub.predict(
        "s0",
        depth_pred=bh.Depth(np.zeros((16, 16), dtype=np.float32), unit="meters"),
    )
    with pytest.raises(ValueError, match="shape_match"):
        sub.build_zip()


def test_builder_accepts_matching_shape():
    sub = _local_client().submission(leaderboard_id=1)
    sub.set_contract([{
        "name": "depth_pred", "kind": "depth", "role": "pred",
        "params": {"shape_match": "image"},
    }])
    sub.set_input_shape("s0", image=(32, 32))
    sub.predict(
        "s0",
        depth_pred=bh.Depth(np.zeros((32, 32), dtype=np.float32), unit="meters"),
    )
    sub.build_zip()  # no raise


def test_builder_skips_check_when_input_shape_not_registered():
    """No registered input shape → no local check; server is the
    authority. Builder builds the ZIP and trusts the round-trip."""
    sub = _local_client().submission(leaderboard_id=1)
    sub.set_contract([{
        "name": "depth_pred", "kind": "depth", "role": "pred",
        "params": {"shape_match": "image"},
    }])
    sub.predict(
        "s0",
        depth_pred=bh.Depth(np.zeros((16, 16), dtype=np.float32), unit="meters"),
    )
    sub.build_zip()


def test_builder_set_input_shape_rejects_bad_shape_arg():
    sub = _local_client().submission(leaderboard_id=1)
    with pytest.raises(ValueError, match="2-tuple"):
        sub.set_input_shape("s0", image=(32,))  # 1-tuple
    with pytest.raises(ValueError, match="2-tuple"):
        sub.set_input_shape("s0", image="thirty-two")


def test_builder_fetch_contract_round_trip(
    client, lb_with_shape_constrained_pred, api_user,
):
    """`fetch_contract()` populates the builder from the live
    /api/leaderboard/<id>/contract endpoint."""
    lb, shapes = lb_with_shape_constrained_pred
    bh_client = bh.Client(
        token=api_user.api_token, base_url="http://test",
        transport=bh.FlaskTestClientTransport(client),
    )
    sub = bh_client.submission(lb.id)
    fetched = sub.fetch_contract()
    assert {e["name"] for e in fetched} == {"depth_pred"}

    # Now register the right shape and make sure the local validator
    # accepts a matching pred + rejects a mismatching one.
    sub.set_input_shape("s0", image=shapes["s0"])
    sub.predict("s0", depth_pred=bh.Depth(
        np.zeros(shapes["s0"], dtype=np.float32), unit="meters"))
    sub.set_input_shape("s1", image=shapes["s1"])
    sub.predict("s1", depth_pred=bh.Depth(
        np.zeros(shapes["s1"], dtype=np.float32), unit="meters"))
    # Builds + uploads cleanly.
    result = sub.submit()
    assert result["leaderboard_id"] == lb.id
