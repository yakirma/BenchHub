#!/usr/bin/env python
"""Rework lb83 (stereo-dataset) into a proper STEREO disparity board.

- Inputs: cam_01_first_frame (left) + cam_05_first_frame (right) — a stereo
  pair with a known baseline.
- GT: cam_01 depth. The parquet only ships a TURBO-colormapped depth
  visualisation (cam_01_depth_vis, an RGB image), so we INVERT the turbo
  colormap on each materialised npz to recover a single-channel relative
  depth map (fixes the GT viz + makes it scorable).
- Metrics: scale-invariant only (scale_inv_rmse_depth, affine_inv_rmse_depth,
  delta1_depth) — the stereo output is disparity, so plain metric RMSE is
  dropped.
- Clears the wrong monocular submissions.

All cam_XX fields were already materialised (role gt), so no re-materialise.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/rework_stereo_lb.py
"""
import os
import sys
import json

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

LB = 83
LEFT, RIGHT, GT_FIELD = 'cam_01_first_frame', 'cam_05_first_frame', 'cam_01_depth_vis'
KEEP_METRICS = ('scale_inv_rmse_depth', 'affine_inv_rmse_depth', 'delta1_depth')


def turbo_invert(rgb_img):
    """RGB (H,W,3) uint8/float turbo-colormapped image → (H,W) relative depth
    in [0,1], by nearest-neighbour lookup into the 256-entry turbo LUT."""
    import numpy as np
    import matplotlib.cm as cm
    lut = cm.get_cmap('turbo')(np.linspace(0, 1, 256))[:, :3].astype(np.float32)  # 256x3
    a = np.asarray(rgb_img, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    H, W = a.shape[:2]
    flat = a.reshape(-1, 3)
    # nearest LUT index per pixel (chunked to bound memory)
    idx = np.empty(flat.shape[0], dtype=np.int32)
    step = 20000
    for i in range(0, flat.shape[0], step):
        d = ((flat[i:i+step, None, :] - lut[None, :, :]) ** 2).sum(-1)
        idx[i:i+step] = d.argmin(1)
    return (idx.astype(np.float32) / 255.0).reshape(H, W)


def main():
    import numpy as np
    import app as A
    from app import (db, Leaderboard, LeaderboardMetric, GlobalMetric,
                     CustomField, MetricResult, Submission)
    from pathlib import Path

    import time

    def _commit():
        for attempt in range(60):
            try:
                db.session.commit(); return
            except Exception as e:
                if 'locked' in str(e).lower():
                    db.session.rollback(); time.sleep(2); continue
                raise
        raise RuntimeError('db locked after 60 retries')

    # 0. Decolorize the GT npz files FIRST (no DB needed, idempotent).
    gt_dir = Path(A.app.config['UPLOAD_FOLDER']) / 'lb_materializations' / str(LB) / GT_FIELD
    n = 0
    for npz in sorted(gt_dir.glob('*.npz')):
        d = np.load(npz)
        arr = d['depth']
        if arr.ndim == 3:
            rel = turbo_invert(arr)
            np.savez_compressed(npz, depth=rel.astype(np.float32))
            n += 1
    print(f'decolorized {n} GT depth maps in {gt_dir}')

    with A.app.app_context():
        lb = Leaderboard.query.get(LB)
        # 1. Roles + contract.
        lb.field_roles_json = json.dumps({LEFT: 'input', RIGHT: 'input', GT_FIELD: 'gt'})
        lb.required_pred_fields_json = json.dumps(
            [{"name": "disparity_pred", "kind": "depth", "params": {}, "role": "pred"}])

        # 2. Metrics: keep only the scale-invariant ones, point gt at cam_01.
        lms = LeaderboardMetric.query.filter_by(leaderboard_id=LB).all()
        keep_ids = []
        for lm in lms:
            if lm.target_name in KEEP_METRICS:
                lm.arg_mappings = json.dumps({"gt": f"gt_{GT_FIELD}", "pred": "sub_disparity_pred"})
                keep_ids.append(f'lm_{lm.id}')
            else:
                db.session.delete(lm)
        lb.summary_metrics = ','.join(keep_ids)
        _commit()
        print(f'reconfigured lb{LB}: inputs={LEFT}+{RIGHT} gt={GT_FIELD} metrics={keep_ids}')

        # 3. Clear the monocular submissions (+ their CFs / MetricResults).
        subs = Submission.query.filter_by(leaderboard_id=LB).all()
        for s in subs:
            CustomField.query.filter_by(submission_id=s.id).delete()
            MetricResult.query.filter_by(submission_id=s.id).delete()
            db.session.delete(s)
        _commit()
        print(f'cleared {len(subs)} monocular submissions')



if __name__ == '__main__':
    raise SystemExit(main())
