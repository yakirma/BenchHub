"""Tests for DataType.visualize() + the /api/viz/<cf_id> dispatch route."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

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
)


# ===========================================================================
# Per-type visualize() — bytes + mime contract
# ===========================================================================

def test_image_visualize_returns_png():
    img = bh.Image(np.zeros((8, 8, 3), dtype=np.uint8))
    body, mime = img.visualize()
    assert mime == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_mask_visualize_returns_png():
    m = bh.Mask(np.array([[0, 1], [2, 3]], dtype=np.uint8))
    body, mime = m.visualize()
    assert mime == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_depth_visualize_returns_png_via_colormap():
    arr = np.linspace(0.5, 5.0, 64, dtype=np.float32).reshape(8, 8)
    d = bh.Depth(arr, unit="meters")
    body, mime = d.visualize(cmap="turbo")
    assert mime == "image/png"
    decoded = PILImage.open(io.BytesIO(body))
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)


def test_depth_visualize_handles_all_nan_array():
    """Pathological case — array is fully NaN. Render a blank PNG
    instead of crashing on a min()/normalisation."""
    arr = np.full((8, 8), np.nan, dtype=np.float32)
    body, mime = bh.Depth(arr).visualize()
    assert mime == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_depth_visualize_unknown_cmap_falls_back_to_gray():
    """Unknown cmap name shouldn't 500 — the renderer falls back to
    plain grayscale so the client still gets a viewable PNG."""
    arr = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    body, mime = bh.Depth(arr).visualize(cmap="not-a-real-cmap")
    assert mime == "image/png"


def test_audio_visualize_returns_wav():
    sr = 16000
    wav = np.zeros(sr, dtype=np.float32)
    a = bh.Audio(wav, sr)
    body, mime = a.visualize()
    assert mime == "audio/wav"
    # WAV files start with RIFF....WAVE
    assert body[:4] == b"RIFF"
    assert body[8:12] == b"WAVE"


def test_text_visualize_returns_utf8_plain_text():
    body, mime = bh.Text("hello fox 🦊").visualize()
    assert mime.startswith("text/plain")
    assert body.decode("utf-8") == "hello fox 🦊"


def test_label_visualize_renders_bare_value_not_json_quoted():
    """`Label("cat").visualize()` should give `cat`, not `\"cat\"`."""
    body, mime = bh.Label("cat").visualize()
    assert mime.startswith("text/plain")
    assert body == b"cat"


def test_label_int_visualize():
    body, _ = bh.Label(3).visualize()
    assert body == b"3"


def test_scalar_visualize_returns_repr():
    body, mime = bh.Scalar(0.5).visualize()
    assert mime.startswith("text/plain")
    assert body == b"0.5"


def test_json_visualize_returns_application_json():
    body, mime = bh.Json({"foo": [1, 2]}).visualize()
    assert mime == "application/json"
    assert json.loads(body) == {"foo": [1, 2]}


# ---------------------------------------------------------------------------
# BBoxes — SVG dispatch + format coercion
# ---------------------------------------------------------------------------

def test_bboxes_visualize_returns_svg():
    body, mime = bh.BBoxes([[0, 0, 10, 10], [5, 5, 8, 9]]).visualize()
    assert mime == "image/svg+xml"
    s = body.decode("utf-8")
    assert s.startswith("<svg")
    # One <rect> per box.
    assert s.count("<rect") == 2


def test_bboxes_visualize_respects_canvas_dimensions():
    body, _ = bh.BBoxes([[0, 0, 1, 1]]).visualize(width=512, height=384)
    s = body.decode("utf-8")
    assert 'width="512"' in s and 'height="384"' in s
    assert 'viewBox="0 0 512 384"' in s


def test_bboxes_visualize_converts_xywh_to_xyxy():
    """The visualizer always emits the box in xyxy SVG coords. An
    xywh-declared box becomes (x, y, x+w, y+h)."""
    body, _ = bh.BBoxes([[10, 20, 30, 40]], format="xywh").visualize()
    s = body.decode("utf-8")
    assert 'x="10.00" y="20.00" width="30.00" height="40.00"' in s


def test_bboxes_visualize_drops_labels_as_text():
    body, _ = bh.BBoxes([[0, 0, 5, 5]], labels=["cat"]).visualize()
    assert b"<text" in body
    assert b">cat</text>" in body


def test_bboxes_visualize_escapes_label_html():
    """A label with `<` shouldn't break the SVG."""
    body, _ = bh.BBoxes([[0, 0, 5, 5]], labels=["<script>"]).visualize()
    s = body.decode("utf-8")
    assert "&lt;script&gt;" in s
    assert "<script>" not in s.replace("svg xmlns", "")  # script tag absent


# ===========================================================================
# Route: /api/viz/<cf_id>
# ===========================================================================

@pytest.fixture
def public_dataset_with_typed_fields(db_session, tmp_path, monkeypatch):
    """A public dataset on disk with one scalar + one image GT field,
    pointed at by a tempdir UPLOAD_FOLDER so the route can read files."""
    from app import app as flask_app

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(uploads))

    ds = Dataset(name="viz_ds", visibility="public")
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s0")
    db.session.add(sample); db.session.flush()

    # File-backed image GT.
    img_dir = uploads / "datasets" / str(ds.id) / "image"
    img_dir.mkdir(parents=True)
    img = bh.Image(np.full((4, 4, 3), 127, dtype=np.uint8))
    (img_dir / "s0.png").write_bytes(img.encode())
    img_cf = CustomField(
        sample_id=sample.id, name="image", data_type="image",
        value_text=str((img_dir / "s0.png").relative_to(uploads)),
    )
    db.session.add(img_cf)

    # Inline scalar GT.
    scal_cf = CustomField(
        sample_id=sample.id, name="snr", data_type="scalar",
        value_float=42.5,
    )
    db.session.add(scal_cf)

    # Inline label GT.
    lbl_cf = CustomField(
        sample_id=sample.id, name="label", data_type="label",
        value_text='"cat"',
    )
    db.session.add(lbl_cf)

    # File-backed depth GT (with millimeter unit in params).
    depth_dir = uploads / "datasets" / str(ds.id) / "depth"
    depth_dir.mkdir(parents=True)
    depth = bh.Depth(
        np.linspace(0.0, 4000.0, 16, dtype=np.float32).reshape(4, 4),
        unit="millimeters",
    )
    (depth_dir / "s0.npz").write_bytes(depth.encode())
    depth_cf = CustomField(
        sample_id=sample.id, name="depth", data_type="depth",
        value_text=str((depth_dir / "s0.npz").relative_to(uploads)),
    )
    depth_cf.set_params({"unit": "millimeters"})
    db.session.add(depth_cf)

    db.session.commit()
    return {
        "dataset": ds, "sample": sample,
        "image_cf": img_cf, "scalar_cf": scal_cf,
        "label_cf": lbl_cf, "depth_cf": depth_cf,
    }


def test_api_viz_serves_inline_scalar(client, public_dataset_with_typed_fields):
    cf = public_dataset_with_typed_fields["scalar_cf"]
    r = client.get(f"/api/viz/{cf.id}")
    assert r.status_code == 200
    assert r.content_type.startswith("text/plain")
    assert r.data == b"42.5"


def test_api_viz_serves_inline_label_as_bare_string(client, public_dataset_with_typed_fields):
    cf = public_dataset_with_typed_fields["label_cf"]
    r = client.get(f"/api/viz/{cf.id}")
    assert r.status_code == 200
    assert r.data == b"cat"  # not "\"cat\""


def test_api_viz_serves_file_backed_image(client, public_dataset_with_typed_fields):
    cf = public_dataset_with_typed_fields["image_cf"]
    r = client.get(f"/api/viz/{cf.id}")
    assert r.status_code == 200
    assert r.content_type == "image/png"
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_viz_depth_applies_cmap_query_param(client, public_dataset_with_typed_fields):
    cf = public_dataset_with_typed_fields["depth_cf"]
    r_turbo = client.get(f"/api/viz/{cf.id}?cmap=turbo")
    r_viridis = client.get(f"/api/viz/{cf.id}?cmap=viridis")
    assert r_turbo.status_code == 200 and r_viridis.status_code == 200
    assert r_turbo.content_type == "image/png"
    # Different colormaps produce different bytes (sanity that the
    # query param actually flows into visualize()).
    assert r_turbo.data != r_viridis.data


def test_api_viz_404_for_unknown_cf(client, db_session):
    r = client.get("/api/viz/99999")
    assert r.status_code == 404


def test_api_viz_404_when_file_missing(client, db_session, tmp_path, monkeypatch):
    """File-backed CustomField pointing at a path that doesn't exist
    → 404, not 500."""
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(tmp_path))

    ds = Dataset(name="missing_ds", visibility="public")
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s0")
    db.session.add(sample); db.session.flush()
    cf = CustomField(
        sample_id=sample.id, name="image", data_type="image",
        value_text="datasets/999/image/never_written.png",
    )
    db.session.add(cf); db.session.commit()

    r = client.get(f"/api/viz/{cf.id}")
    assert r.status_code == 404


def test_api_viz_private_dataset_blocks_anonymous(client, db_session, tmp_path, monkeypatch):
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(tmp_path))

    owner = User(
        email="owner@bench.local", display_name="owner",
        oauth_provider="github", oauth_sub="owner-1",
    )
    db.session.add(owner); db.session.flush()
    ds = Dataset(name="private_ds", visibility="private", owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s0")
    db.session.add(sample); db.session.flush()
    cf = CustomField(
        sample_id=sample.id, name="snr", data_type="scalar",
        value_float=7.0,
    )
    db.session.add(cf); db.session.commit()

    r = client.get(f"/api/viz/{cf.id}")
    assert r.status_code == 404  # don't leak existence


def test_api_viz_private_dataset_allowed_to_owner(client, db_session, tmp_path, monkeypatch):
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(tmp_path))

    owner = User(
        email="o2@bench.local", display_name="o2",
        oauth_provider="github", oauth_sub="o2-1",
    )
    db.session.add(owner); db.session.flush()
    ds = Dataset(name="priv2", visibility="private", owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s0")
    db.session.add(sample); db.session.flush()
    cf = CustomField(
        sample_id=sample.id, name="snr", data_type="scalar",
        value_float=9.0,
    )
    db.session.add(cf); db.session.commit()

    with client.session_transaction() as sess:
        sess["user_id"] = owner.id
    r = client.get(f"/api/viz/{cf.id}")
    assert r.status_code == 200
    assert r.data == b"9.0"


def test_api_viz_swallows_unknown_query_args(client, public_dataset_with_typed_fields):
    """Stray ?foo=bar shouldn't 500 — types accept extra kwargs via **_."""
    cf = public_dataset_with_typed_fields["scalar_cf"]
    r = client.get(f"/api/viz/{cf.id}?nonsense=1&foo=bar")
    assert r.status_code == 200
