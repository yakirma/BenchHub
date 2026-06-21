"""/api/flow_data — raw optical-flow (u, v) for the comparison hover readout.

Packs the two flow components as little-endian binary: int32 H, int32 W, then
H*W float32 u, then H*W float32 v. Used by the comparison page so hovering a
flow cell reads true pixel displacement.
"""
import json
import struct

import numpy as np
import pytest

from app import (
    CustomField,
    Dataset,
    GlobalVisualization,
    Leaderboard,
    LeaderboardVisualization,
    Sample,
    app as flask_app,
    db,
)


@pytest.fixture
def flow_board(db_session, tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    (uploads / "datasets" / "u").mkdir(parents=True)
    (uploads / "datasets" / "v").mkdir(parents=True)
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(uploads))

    # Tiny 2x2 signed flow fields.
    u = np.array([[1.0, -2.0], [3.5, 0.0]], dtype=np.float32)
    v = np.array([[0.5, 4.0], [-1.5, 2.0]], dtype=np.float32)
    np.savez(uploads / "datasets" / "u" / "s0.npz", depth=u)
    np.savez(uploads / "datasets" / "v" / "s0.npz", depth=v)

    ds = Dataset(name="flow_ds", visibility="public")
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s0")
    db.session.add(sample); db.session.flush()
    for nm, sub in (("flow_u", "u"), ("flow_v", "v")):
        cf = CustomField(sample_id=sample.id, name=nm, data_type="depth",
                         value_text=f"datasets/{sub}/s0.npz")
        db.session.add(cf)

    gv = GlobalVisualization(name="flow_color",
                             python_code="def flow_color(flow_u, flow_v): pass",
                             visibility="public")
    db.session.add(gv); db.session.flush()
    lb = Leaderboard(name="flow_lb", summary_metrics="", visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    lv = LeaderboardVisualization(
        leaderboard_id=lb.id, global_visualization_id=gv.id,
        arg_mappings=json.dumps({"flow_u": "gt_flow_u", "flow_v": "gt_flow_v"}),
        target_name="Flow", display_order=0)
    db.session.add(lv); db.session.commit()
    return {"lv": lv.id, "sample": sample.id, "u": u, "v": v}


def test_flow_data_returns_packed_uv(client, flow_board):
    r = client.get(f"/api/flow_data/{flow_board['lv']}/{flow_board['sample']}")
    assert r.status_code == 200
    assert r.mimetype == "application/octet-stream"
    buf = r.data
    H, W = struct.unpack_from("<ii", buf, 0)
    assert (H, W) == (2, 2)
    n = H * W
    u = np.frombuffer(buf, dtype="<f4", count=n, offset=8).reshape(H, W)
    v = np.frombuffer(buf, dtype="<f4", count=n, offset=8 + n * 4).reshape(H, W)
    assert np.allclose(u, flow_board["u"])
    assert np.allclose(v, flow_board["v"])
    # the pixel a user would hover (row 1, col 0) reads the true displacement
    assert u[1, 0] == pytest.approx(3.5)
    assert v[1, 0] == pytest.approx(-1.5)


def test_flow_data_404_on_non_flow_viz(client, db_session):
    # A viz whose arg_mappings have no flow_u/flow_v → 404 (nothing to serve).
    gv = GlobalVisualization(name="not_flow", python_code="def f(x): pass", visibility="public")
    db.session.add(gv); db.session.flush()
    lb = Leaderboard(name="lb2", summary_metrics="", visibility="public")
    db.session.add(lb); db.session.flush()
    ds = Dataset(name="d2", visibility="public"); db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name="s0"); db.session.add(s); db.session.flush()
    lv = LeaderboardVisualization(leaderboard_id=lb.id, global_visualization_id=gv.id,
                                  arg_mappings=json.dumps({"x": "gt_img"}),
                                  target_name="x", display_order=0)
    db.session.add(lv); db.session.commit()
    assert client.get(f"/api/flow_data/{lv.id}/{s.id}").status_code == 404


def test_flow_data_404_on_private_board_anon(client, db_session, tmp_path, monkeypatch):
    uploads = tmp_path / "up"; (uploads / "d").mkdir(parents=True)
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(uploads))
    np.savez(uploads / "d" / "s0.npz", depth=np.zeros((2, 2), np.float32))
    ds = Dataset(name="pd", visibility="private"); db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name="s0"); db.session.add(s); db.session.flush()
    for nm in ("flow_u", "flow_v"):
        db.session.add(CustomField(sample_id=s.id, name=nm, data_type="depth",
                                   value_text="d/s0.npz"))
    gv = GlobalVisualization(name="flow_color", python_code="def flow_color(flow_u, flow_v): pass",
                             visibility="public")
    db.session.add(gv); db.session.flush()
    lb = Leaderboard(name="priv_lb", summary_metrics="", visibility="private")
    lb.datasets.append(ds); db.session.add(lb); db.session.flush()
    lv = LeaderboardVisualization(leaderboard_id=lb.id, global_visualization_id=gv.id,
                                  arg_mappings=json.dumps({"flow_u": "gt_flow_u", "flow_v": "gt_flow_v"}),
                                  target_name="f", display_order=0)
    db.session.add(lv); db.session.commit()
    # anonymous client must not read a private board's flow field
    assert client.get(f"/api/flow_data/{lv.id}/{s.id}").status_code == 404
