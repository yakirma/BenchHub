"""Phase 4: per-sample viz thumbnail generation post-eval.

Verifies the writer functions produce reasonable PNGs, the generator
walks the LB's pred fields, the cap is respected, and the serving
route refuses path traversal.
"""
import io
import os

import numpy as np
import pytest
from PIL import Image

import app as app_mod
from app import (
    Dataset, Leaderboard, Sample, Submission, db,
    _generate_submission_viz_assets, _submission_viz_dir,
    _write_depth_viz_png, _write_image_viz_png,
)


# ---------------------------------------------------------------------------
# Writer unit tests
# ---------------------------------------------------------------------------


def test_depth_writer_produces_thumbnail_png(tmp_path):
    arr = np.linspace(0.5, 5.0, 64 * 96, dtype=np.float32).reshape(96, 64)
    out = tmp_path / 'd.png'
    assert _write_depth_viz_png(arr, str(out))
    img = Image.open(out)
    assert img.format == 'PNG'
    assert max(img.size) <= 256


def test_depth_writer_handles_nan_inf_arrays(tmp_path):
    arr = np.full((10, 10), np.nan, dtype=np.float32)
    arr[:5] = np.inf
    arr[5:7] = 1.0  # tiny finite slice — should not crash
    out = tmp_path / 'd.png'
    # Returns True because there is a finite slice; we just want no exception.
    _write_depth_viz_png(arr, str(out))


def test_depth_writer_rejects_all_nonfinite(tmp_path):
    arr = np.full((10, 10), np.nan, dtype=np.float32)
    out = tmp_path / 'd.png'
    assert not _write_depth_viz_png(arr, str(out))
    assert not out.exists()


def test_image_writer_handles_seg_mask(tmp_path):
    """2D mask of class IDs gets colorized via the small LUT."""
    mask = (np.arange(50 * 50).reshape(50, 50) % 12).astype(np.uint8)
    out = tmp_path / 'm.png'
    assert _write_image_viz_png(mask, str(out))
    img = Image.open(out)
    assert img.mode == 'RGB'  # colorized, not grayscale
    assert max(img.size) <= 256


def test_image_writer_handles_rgb(tmp_path):
    img_arr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    out = tmp_path / 'r.png'
    assert _write_image_viz_png(img_arr, str(out))
    img = Image.open(out)
    assert img.mode == 'RGB'
    # Aspect-preserving downscale: longer edge clamped to 256.
    assert max(img.size) == 256


# ---------------------------------------------------------------------------
# Generator + cap behavior
# ---------------------------------------------------------------------------


@pytest.fixture
def lb_with_depth_pred(client, db_session, tmp_path, monkeypatch):
    """Build an LB that consumes a `dense_depth_pred` field, plus a
    submission folder on disk with depth NPZ files for sample 0..N-1."""
    ds = Dataset(name='depth_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    samples = []
    for i in range(5):
        s = Sample(dataset_id=ds.id, name=f's{i}')
        db.session.add(s)
        samples.append(s)
    db.session.flush()

    lb = Leaderboard(
        name='depth_lb', summary_metrics='', visibility='public',
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    # _iter_lb_eval_samples reads from the new Attachment table, not
    # the legacy m2m. Wire one up explicitly.
    from app import Attachment
    db.session.add(Attachment(
        leaderboard_id=lb.id, dataset_id=ds.id, role='primary',
    ))
    db.session.flush()

    # Wire up a leaderboard_metric whose arg_mappings reference
    # sub_dense_depth_pred so _lb_submission_pred_fields infers the
    # dense field.
    from app import GlobalMetric, LeaderboardMetric
    gm = GlobalMetric(
        name='depth_mae', python_code='def f(*a, **k): return 0.0',
        is_aggregated=False,
    )
    db.session.add(gm); db.session.flush()
    db.session.add(LeaderboardMetric(
        leaderboard_id=lb.id, global_metric_id=gm.id,
        target_name='mae',
        arg_mappings='{"pred": "sub_dense_depth_pred", "gt": "gt_dense_depth"}',
        sort_direction='lower_is_better',
    ))

    # Mark the GT field on the first sample as 'depth' so the helper
    # classifies the pred as a depth pred.
    from app import CustomField
    db.session.add(CustomField(
        sample_id=samples[0].id, name='dense_depth',
        data_type='depth', value_text='datasets/dummy/dense_depth/s0.npz',
    ))
    db.session.commit()

    # Lay out submission ZIP-extracted folder with dense_depth/ predictions.
    sub_id = 99001
    sub_root = tmp_path / 'subs' / str(sub_id)
    pred_dir = sub_root / 'dense_depth'
    pred_dir.mkdir(parents=True)
    for s in samples:
        npz = io.BytesIO()
        np.savez_compressed(
            npz, depth=np.full((32, 32), float(int(s.name[1:])), dtype=np.float32),
        )
        (pred_dir / f'{s.name}.npz').write_bytes(npz.getvalue())

    sub = Submission(
        id=sub_id, name='depth_sub', leaderboard_id=lb.id,
    )
    db.session.add(sub); db.session.commit()

    # Override viz dir so we don't pollute the real data dir.
    viz_root = tmp_path / 'viz_target' / str(sub_id) / 'viz'
    monkeypatch.setattr(app_mod, '_submission_viz_dir',
                        lambda submission: str(viz_root))
    return lb, sub, str(sub_root), str(viz_root)


def test_generator_writes_one_png_per_sample(lb_with_depth_pred):
    lb, sub, sub_root, viz_root = lb_with_depth_pred
    n = _generate_submission_viz_assets(sub, lb, sub_root)
    assert n == 5
    files = sorted(os.listdir(os.path.join(viz_root, 'dense_depth')))
    assert files == ['s0.png', 's1.png', 's2.png', 's3.png', 's4.png']


def test_generator_respects_cap(lb_with_depth_pred, monkeypatch):
    lb, sub, sub_root, viz_root = lb_with_depth_pred
    monkeypatch.setattr(app_mod, 'SUBMISSION_VIZ_MAX_SAMPLES', 2)
    n = _generate_submission_viz_assets(sub, lb, sub_root)
    assert n == 2
    files = sorted(os.listdir(os.path.join(viz_root, 'dense_depth')))
    assert len(files) == 2


def test_generator_skips_when_no_dense_fields(client, db_session, tmp_path):
    """An LB with only scalar prediction fields shouldn't produce
    any viz directories — there's nothing to thumbnail."""
    ds = Dataset(name='scalar_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    lb = Leaderboard(name='scalar_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    sub = Submission(name='scalar_sub', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()
    n = _generate_submission_viz_assets(sub, lb, str(tmp_path))
    assert n == 0


# ---------------------------------------------------------------------------
# Serving route + path-traversal guard
# ---------------------------------------------------------------------------


def test_serve_route_404_for_missing_file(client, db_session):
    ds = Dataset(name='vds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='vlb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    sub = Submission(name='vsub', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()
    r = client.get(f'/api/submission_viz/{sub.id}/depth/s0.png')
    assert r.status_code == 404


def test_serve_route_blocks_path_traversal(client, db_session):
    ds = Dataset(name='trav_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='trav_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    sub = Submission(name='trav_sub', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()
    # Werkzeug normalizes some traversal attempts at the routing layer
    # before our handler runs, returning 404 instead of 400 — both are
    # safe (the request never reaches a sensitive file). Either is OK.
    r = client.get(
        f'/api/submission_viz/{sub.id}/..%2f..%2fetc%2fpasswd',
    )
    assert r.status_code in (400, 404)
