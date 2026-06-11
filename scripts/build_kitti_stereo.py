#!/usr/bin/env python
"""Build a KITTI-2015 stereo leaderboard from the official scene-flow data.

KITTI ships left (image_2), right (image_3) and GT disparity (disp_occ_0, a
uint16 PNG = disparity*256, 0 = invalid). We materialise a FULL-RES typed
dataset (left/right image inputs + disparity GT as a depth-kind field with
is_inverse=True), create a public LB with the same 13 metrics as the synthetic
stereo board, and leave it ready for the classical + deep stereo submitters.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_kitti_stereo.py [n_samples]
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

KITTI = Path('/tmp/kitti2015/training')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DS_NAME = 'KITTI-2015-stereo'
LEFT, RIGHT, DISP = 'left', 'right', 'disparity'

# 3 scale-invariant + 10 Middlebury metrics (already-created GlobalMetrics).
METRICS = [
    ('scale_inv_rmse_depth', 'scale_inv_rmse_depth', 'lower_is_better'),
    ('affine_inv_rmse_depth', 'affine_inv_rmse_depth', 'lower_is_better'),
    ('delta1_depth', 'delta1_depth', 'higher_is_better'),
    ('mb_bad05', 'bad0.5', 'lower_is_better'), ('mb_bad1', 'bad1.0', 'lower_is_better'),
    ('mb_bad2', 'bad2.0', 'lower_is_better'), ('mb_bad4', 'bad4.0', 'lower_is_better'),
    ('mb_avgerr', 'avgerr', 'lower_is_better'), ('mb_rms', 'rms', 'lower_is_better'),
    ('mb_a50', 'A50', 'lower_is_better'), ('mb_a90', 'A90', 'lower_is_better'),
    ('mb_a95', 'A95', 'lower_is_better'), ('mb_a99', 'A99', 'lower_is_better'),
]


def build_staging(staging: Path):
    import numpy as np
    from PIL import Image
    for f in (LEFT, RIGHT, DISP):
        (staging / f).mkdir(parents=True, exist_ok=True)
    frames = sorted(p.name for p in (KITTI / 'image_2').glob('*_10.png'))[:N]
    names = []
    for fn in frames:
        name = fn.replace('.png', '')
        names.append(name)
        shutil.copy(KITTI / 'image_2' / fn, staging / LEFT / f'{name}.png')
        shutil.copy(KITTI / 'image_3' / fn, staging / RIGHT / f'{name}.png')
        d = np.asarray(Image.open(KITTI / 'disp_occ_0' / fn)).astype(np.float32) / 256.0
        d[d <= 0] = np.nan  # invalid pixels
        np.savez_compressed(staging / DISP / f'{name}.npz', depth=d)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': LEFT, 'kind': 'image', 'role': 'input', 'params': {}},
            {'name': RIGHT, 'kind': 'image', 'role': 'input', 'params': {}},
            {'name': DISP, 'kind': 'depth', 'role': 'gt', 'params': {'is_inverse': True, 'unit': 'unitless'}},
        ],
        'samples': names,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return names


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset

    with A.app.app_context():
        # Idempotent: skip if the dataset already exists.
        existing = Dataset.query.filter_by(name=DS_NAME).first()
        if existing is not None:
            print(f'dataset {DS_NAME} already exists (id={existing.id}); skipping import')
            ds = existing
        else:
            staging = Path(tempfile.mkdtemp(prefix='kitti_stereo_'))
            try:
                names = build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Vision/Stereo Depth Estimation'
                ds.source_url = 'https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php'
                ds.source_kind = 'local-stereo'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        # Leaderboard.
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Vision/Stereo Depth Estimation',
                required_pred_fields_json=json.dumps(
                    [{"name": "depth_pred", "kind": "depth", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({LEFT: 'input', RIGHT: 'input', DISP: 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        # Bind metrics.
        keep = []
        for gname, label, sort_dir in METRICS:
            gm = GlobalMetric.query.filter_by(name=gname).first()
            if gm is None:
                print(f'  WARN metric {gname} missing'); continue
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=label,
                    arg_mappings=json.dumps({"gt": f"gt_{DISP}", "pred": "sub_depth_pred"}),
                    pooling_type='mean', sort_direction=sort_dir)
                db.session.add(lm); db.session.commit()
            keep.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(dict.fromkeys(keep))
        db.session.commit()
        print(f'STEREO_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name} '
              f'inputs={LEFT}+{RIGHT} gt={DISP} metrics={len(keep)}')


if __name__ == '__main__':
    raise SystemExit(main())
