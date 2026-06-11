#!/usr/bin/env python
"""Decode CARLA RGB-encoded depth in lb85's materialised GT.

CARLA's depth camera encodes metric depth across the RGB channels:
    depth_m = (R + G*256 + B*256*256) / (256**3 - 1) * 1000
The carla_hd `raw_depth` column shipped this RGB image, and it was
materialised verbatim as a (H,W,3) array — so the metric read only the R
channel (garbage) and the GT depth rendered as the raw RGB. Decode each
materialised npz to a single-channel depth map in METERS, then re-score.
"""
import os
import sys
import glob

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

LB = 85
FIELD = 'raw_depth'


def main():
    import numpy as np
    import app as A
    gt_dir = os.path.join(A.app.config['UPLOAD_FOLDER'], 'lb_materializations', str(LB), FIELD)
    n = 0
    for f in sorted(glob.glob(os.path.join(gt_dir, '*.npz'))):
        a = np.load(f)['depth'].astype(np.float64)
        if a.ndim != 3:
            continue
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        depth_m = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1.0) * 1000.0
        np.savez_compressed(f, depth=depth_m.astype(np.float32))
        n += 1
    print(f'decoded {n} CARLA depth maps in {gt_dir}')

    # Re-score lb85's submissions against the corrected GT.
    import tasks
    from app import Submission
    with A.app.app_context():
        ids = [s.id for s in Submission.query.filter_by(leaderboard_id=LB).all()]
    for i in ids:
        tasks.process_submission.delay(i)
    print(f're-scoring {len(ids)} submissions: {ids}')


if __name__ == '__main__':
    raise SystemExit(main())
