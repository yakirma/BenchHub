#!/usr/bin/env python
"""Import a SemanticKITTI slice as a typed BenchHub dataset (point_cloud + the
remapped point_labels) and create the 3D LiDAR Semantic Segmentation board
bound to point_miou. Run AFTER build_lidar.py (which registers the dtypes +
metric) and after the SemanticKITTI data is extracted under
~/.dtofbenchmarking/lidar_kitti/.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_lidar_lb.py [n_scans]
"""
import os
import sys
import json
import glob
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

import numpy as np

KITTI = os.path.expanduser('~/.dtofbenchmarking/lidar_kitti')
SEQ = '08'                                    # standard validation sequence
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
DS_NAME = 'SemanticKITTI-seq08'
CATEGORY = 'Vision/3D Semantic Segmentation'

# SemanticKITTI raw class id -> 0..19 learning id (standard semantic-kitti.yaml).
LEARNING_MAP = {0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,40:9,
                44:10,48:11,49:12,50:13,51:14,52:0,60:9,70:15,71:16,72:17,80:18,
                81:19,99:0,252:1,253:7,254:6,255:8,256:5,257:5,258:4,259:5}
_LUT = np.zeros(260, np.uint16)
for k, v in LEARNING_MAP.items():
    _LUT[k] = v


def remap(label_path):
    sem = (np.fromfile(label_path, np.uint32) & 0xFFFF)
    sem = np.where(sem < len(_LUT), sem, 0)
    return _LUT[sem].astype(np.uint16)


def build_staging(staging):
    vel = sorted(glob.glob(f'{KITTI}/dataset/sequences/{SEQ}/velodyne/*.bin'))
    lab = sorted(glob.glob(f'{KITTI}/data_odometry_labels/dataset/sequences/{SEQ}/labels/*.label'))
    assert len(vel) == len(lab) and vel, f'no matched scans ({len(vel)}/{len(lab)})'
    stride = max(1, len(vel) // N)
    pairs = list(zip(vel, lab))[::stride][:N]
    (staging / 'cloud').mkdir(parents=True, exist_ok=True)
    (staging / 'labels').mkdir(parents=True, exist_ok=True)
    out = []
    for vp, lp in pairs:
        name = f'{SEQ}_' + os.path.basename(vp).replace('.bin', '')
        shutil.copy(vp, staging / 'cloud' / f'{name}.bin')          # N×4 float32, verbatim
        remap(lp).tofile(staging / 'labels' / f'{name}.label')      # N uint16 learning ids
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'cloud',  'kind': 'point_cloud',  'role': 'input', 'params': {'channels': 4}},
            {'name': 'labels', 'kind': 'point_labels', 'role': 'gt',    'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    EXTRA = {'point_cloud': '.bin', 'point_labels': '.label'}
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='skitti_'))
            try:
                n = len(build_staging(staging))
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'], owner_user_id=2,
                    visibility='public', preview_only=False, extra_kinds=EXTRA)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = CATEGORY
                ds.source_url = 'https://semantic-kitti.org/'
                ds.source_kind = 'local-lidar'
                ds.card_description = (
                    'SemanticKITTI (Behley et al., 2019) — 3D LiDAR semantic '
                    f'segmentation, validation sequence 08 ({n} scans, every '
                    f'{max(1, 4071//N)}th). Per-point Velodyne scans (x,y,z,'
                    'intensity) + 19-class learning labels. Scored by point mIoU.')
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary.get("samples")} samples ({n})')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            print(f'dataset {DS_NAME} exists (id={ds.id})')

        gm = GlobalMetric.query.filter_by(name='point_miou').first()
        if gm is None:
            raise SystemExit('point_miou metric missing — run build_lidar.py first')
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public', category=CATEGORY,
                required_pred_fields_json=json.dumps(
                    [{'name': 'labels_pred', 'kind': 'point_labels', 'params': {}, 'role': 'pred'}]),
                field_roles_json=json.dumps({'cloud': 'input', 'labels': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb)
            db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='mIoU',
                arg_mappings=json.dumps({'gt': 'gt_labels', 'pred': 'sub_labels_pred'}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm)
            db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'LIDAR_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
