"""Engine-context array loaders for non-scalar metrics.

Pin the Option-B engine extension: `get_metric_context` now loads
`image` and `depth` GT custom fields into numpy arrays (or None on
failure) so RMSE / PSNR / IoU metrics can consume them directly.
The submission-side scanner picks up bare-name `<col>_pred/`
folders with .png / .npz / .txt files and exposes them under
`sub_<folder>` in the same shape.
"""
import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from app import CustomField, Sample, Submission, db
from metric_engine import (
    get_metric_context,
    _load_gt_array,
    _load_sub_pred_for_sample,
)


# ---------------------------------------------------------------------------
# _load_gt_array: image + depth + missing-file paths
# ---------------------------------------------------------------------------


def test_load_gt_array_image_returns_rgb_numpy(tmp_path):
    img = Image.new('RGB', (8, 4), (10, 20, 30))
    img.save(tmp_path / 'frame.png')

    cf = type('CF', (), {
        'name': 'frame', 'data_type': 'image',
        'value_text': 'frame.png',
    })()
    arr = _load_gt_array(cf, str(tmp_path))
    assert arr is not None
    assert arr.shape == (4, 8, 3)
    assert arr[0, 0, 0] == 10


def test_load_gt_array_depth_reads_depth_key(tmp_path):
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.savez(tmp_path / 'depth.npz', depth=depth)
    cf = type('CF', (), {
        'name': 'depth', 'data_type': 'depth', 'value_text': 'depth.npz',
    })()
    arr = _load_gt_array(cf, str(tmp_path))
    assert arr is not None
    np.testing.assert_array_equal(arr, depth)


def test_load_gt_array_depth_falls_back_to_first_key_when_no_depth(tmp_path):
    """Some legacy NPZs store the array under a different key — use
    the first one rather than failing."""
    arr_in = np.eye(5, dtype=np.float32)
    np.savez(tmp_path / 'd.npz', distance=arr_in)
    cf = type('CF', (), {
        'name': 'd', 'data_type': 'depth', 'value_text': 'd.npz',
    })()
    arr = _load_gt_array(cf, str(tmp_path))
    np.testing.assert_array_equal(arr, arr_in)


def test_load_gt_array_returns_none_for_missing_file(tmp_path):
    cf = type('CF', (), {
        'name': 'gone', 'data_type': 'image', 'value_text': 'no/such.png',
    })()
    assert _load_gt_array(cf, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _load_sub_pred_for_sample: scalar / image / depth detection
# ---------------------------------------------------------------------------


def test_sub_pred_scalar_text(tmp_path):
    pred_dir = tmp_path / 'label_pred'
    pred_dir.mkdir()
    (pred_dir / 's00000.txt').write_text('3')
    val = _load_sub_pred_for_sample(str(tmp_path), 'label_pred', 's00000')
    assert val == 3.0


def test_sub_pred_image_png(tmp_path):
    pred_dir = tmp_path / 'rgb_pred'
    pred_dir.mkdir()
    Image.new('RGB', (3, 2), (5, 5, 5)).save(pred_dir / 's1.png')
    arr = _load_sub_pred_for_sample(str(tmp_path), 'rgb_pred', 's1')
    assert arr is not None and arr.shape == (2, 3, 3)


def test_sub_pred_depth_with_dim_suffix(tmp_path):
    pred_dir = tmp_path / 'depth_pred'
    pred_dir.mkdir()
    arr_in = np.full((4, 4), 1.5, dtype=np.float32)
    np.savez(pred_dir / 's_4x4.npz', depth=arr_in)
    arr = _load_sub_pred_for_sample(str(tmp_path), 'depth_pred', 's')
    np.testing.assert_array_equal(arr, arr_in)


def test_sub_pred_depth_without_dim_suffix(tmp_path):
    """Bare-name `<sample>.npz` is still recognized as depth — the
    auto-LB pred contract doesn't strictly enforce the _<W>x<H> tail."""
    pred_dir = tmp_path / 'd_pred'
    pred_dir.mkdir()
    arr_in = np.zeros((2, 2), dtype=np.float32)
    np.savez(pred_dir / 's.npz', depth=arr_in)
    arr = _load_sub_pred_for_sample(str(tmp_path), 'd_pred', 's')
    np.testing.assert_array_equal(arr, arr_in)


def test_sub_pred_returns_none_when_nothing_matches(tmp_path):
    pred_dir = tmp_path / 'empty_pred'
    pred_dir.mkdir()
    assert _load_sub_pred_for_sample(str(tmp_path), 'empty_pred', 's0') is None


def test_sub_pred_returns_none_when_folder_missing(tmp_path):
    assert _load_sub_pred_for_sample(str(tmp_path), 'no_folder', 's0') is None


# ---------------------------------------------------------------------------
# get_metric_context end-to-end: depth GT + bare-name pred folder
# ---------------------------------------------------------------------------


def test_full_context_loads_depth_gt_and_depth_pred(
    db_session, tmp_path,
):
    """Pinpoint test for the canonical depth-LB shape: GT has a
    `depth_map` field, submission ships `depth_map_pred/<sample>_<W>x<H>.npz`.
    Both must arrive in the context as numpy arrays so the metric
    function can compute RMSE."""
    from app import Dataset, Sample as SampleModel, Submission, Leaderboard

    # GT: write a depth NPZ on disk and point a CustomField at it.
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    gt_dir = upload_dir / 'datasets' / 'ds' / 'depth_maps' / 'depth_map'
    gt_dir.mkdir(parents=True)
    gt_arr = np.full((4, 4), 5.0, dtype=np.float32)
    np.savez(gt_dir / 's00000_4x4.npz', depth=gt_arr)
    rel_gt_path = os.path.relpath(gt_dir / 's00000_4x4.npz', upload_dir)

    ds = Dataset(name='depth_ctx_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    sample = SampleModel(dataset_id=ds.id, name='s00000')
    db.session.add(sample); db.session.flush()
    db.session.add(CustomField(
        sample_id=sample.id, name='depth_map', data_type='depth',
        value_text=rel_gt_path,
    ))
    db.session.commit()

    # Submission: write a `depth_map_pred/` folder with a matching NPZ.
    lb = Leaderboard(name='depth_ctx_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(name='sub1', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()

    sub_folder = tmp_path / 'submission'
    pred_dir = sub_folder / 'depth_map_pred'
    pred_dir.mkdir(parents=True)
    pred_arr = np.full((4, 4), 6.0, dtype=np.float32)
    np.savez(pred_dir / 's00000_4x4.npz', depth=pred_arr)

    ctx = get_metric_context(
        sample, sub=sub, submission_folder=str(sub_folder),
        upload_folder=str(upload_dir),
    )
    assert 'gt_depth_map' in ctx
    np.testing.assert_array_equal(ctx['gt_depth_map'], gt_arr)
    assert 'sub_depth_map_pred' in ctx
    np.testing.assert_array_equal(ctx['sub_depth_map_pred'], pred_arr)


def test_full_context_loads_image_gt_and_image_pred(
    db_session, tmp_path,
):
    from app import Dataset, Sample as SampleModel, Submission, Leaderboard

    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    gt_dir = upload_dir / 'datasets' / 'ds' / 'images' / 'rgb'
    gt_dir.mkdir(parents=True)
    Image.new('RGB', (4, 4), (10, 10, 10)).save(gt_dir / 's0.png')
    rel = os.path.relpath(gt_dir / 's0.png', upload_dir)

    ds = Dataset(name='img_ctx_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    sample = SampleModel(dataset_id=ds.id, name='s0')
    db.session.add(sample); db.session.flush()
    db.session.add(CustomField(
        sample_id=sample.id, name='rgb', data_type='image', value_text=rel,
    ))
    db.session.commit()

    lb = Leaderboard(name='img_ctx_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(name='sub1', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()

    sub_folder = tmp_path / 'submission'
    pred_dir = sub_folder / 'rgb_pred'
    pred_dir.mkdir(parents=True)
    Image.new('RGB', (4, 4), (20, 20, 20)).save(pred_dir / 's0.png')

    ctx = get_metric_context(
        sample, sub=sub, submission_folder=str(sub_folder),
        upload_folder=str(upload_dir),
    )
    assert ctx['gt_rgb'] is not None and ctx['gt_rgb'].shape == (4, 4, 3)
    assert ctx['sub_rgb_pred'] is not None and ctx['sub_rgb_pred'].shape == (4, 4, 3)
    # GT mean ≈ 10, pred mean ≈ 20 — plenty of signal for a PSNR metric.
    assert ctx['gt_rgb'].mean() < ctx['sub_rgb_pred'].mean()


def test_submission_folder_metric_named_folder_loaded_as_scalar(
    db_session, tmp_path,
):
    """Folder prefixes (`metric_`, `hist_`, `raw_`) no longer carry
    special meaning. A folder named `metric_score` is loaded as a
    plain scalar prediction the same way `label_pred` is."""
    from app import Dataset, Sample as SampleModel, Submission, Leaderboard
    ds = Dataset(name='skip_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    sample = SampleModel(dataset_id=ds.id, name='s')
    db.session.add(sample); db.session.flush()
    lb = Leaderboard(name='skip_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(name='sub1', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()

    sub_folder = tmp_path / 'submission'
    (sub_folder / 'metric_score').mkdir(parents=True)
    (sub_folder / 'metric_score' / 's.txt').write_text('0.9')
    pred_dir = sub_folder / 'label_pred'
    pred_dir.mkdir()
    (pred_dir / 's.txt').write_text('7')

    ctx = get_metric_context(
        sample, sub=sub, submission_folder=str(sub_folder),
        upload_folder=str(tmp_path),
    )
    # Both bare-name folders now load uniformly.
    assert ctx.get('sub_label_pred') == 7.0
    assert ctx.get('sub_metric_score') == 0.9


def test_label_list_pred_reaches_context_as_typed_labellist(db_session, tmp_path):
    """A `label_list` submission CF (ranked top-K) must arrive in the
    context as the parsed list AND a typed bh.LabelList. Regression:
    get_metric_context had no label_list branch, so the field was
    dropped entirely → top-K metrics saw pred=None → 0.0."""
    import benchhub as bh
    from app import Dataset, Sample as SampleModel, Leaderboard

    ds = Dataset(name='topk_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    sample = SampleModel(dataset_id=ds.id, name='s0')
    db.session.add(sample); db.session.flush()
    lb = Leaderboard(name='topk_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(name='sub1', leaderboard_id=lb.id)
    db.session.add(sub); db.session.commit()

    cf = CustomField(
        submission_id=sub.id, sample_id=sample.id, sample_name='s0',
        name='label_topk_pred', data_type='label_list',
        value_text='[3, 5, 9, 6, 0]',
    )
    cf.set_params({'k': 5})
    db.session.add(cf); db.session.commit()

    ctx = get_metric_context(sample, sub=sub, upload_folder=str(tmp_path))
    # Parsed list at the bare + sub_ keys.
    assert ctx.get('sub_label_topk_pred') == [3, 5, 9, 6, 0]
    # Typed instance available for kind-aware metrics.
    typed = ctx.get('__typed__sub_label_topk_pred')
    assert isinstance(typed, bh.LabelList)
    assert typed.values[0] == 3  # NOT '[' (the old string-wrap bug)
