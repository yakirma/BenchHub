#!/usr/bin/env python
"""Import a nuScenes-lidarseg slice (official val scenes present in the
v1.0-trainval01 blob) as a typed BenchHub dataset (point_cloud + 16-class
challenge point_labels) and create the board bound to point_miou — the nuScenes
sibling of the SemanticKITTI board (lb138). Reuses the point_cloud/point_labels
dtypes + point_miou metric from build_lidar.py.

GT labels come from the panoptic .npz (semantic = value // 1000, 0..31 raw
nuScenes class), remapped to the 16 lidarseg-challenge classes (1..16, 0=ignore)
— the standard merge, matching WaffleIron's class order.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        python scripts/build_nuscenes_lb.py
"""
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

import numpy as np

NUSC = os.path.expanduser('~/lidar_seg/nuscenes')
VAL_TABLE = f'{NUSC}/val_table.json'
DS_NAME = 'nuScenes-lidarseg-part1-val'
CATEGORY = 'Vision/3D Semantic Segmentation'

# Standard nuScenes-lidarseg-challenge merge: 32 raw classes -> 16 (1..16),
# 0 = ignore. index = raw class id (category.json order), value = challenge id.
# 1 barrier 2 bicycle 3 bus 4 car 5 construction_vehicle 6 motorcycle
# 7 pedestrian 8 traffic_cone 9 trailer 10 truck 11 driveable_surface
# 12 other_flat 13 sidewalk 14 terrain 15 manmade 16 vegetation
CHALLENGE_LUT = np.array([
    0,   # 0  noise
    0,   # 1  animal
    7,   # 2  human.pedestrian.adult
    7,   # 3  human.pedestrian.child
    7,   # 4  human.pedestrian.construction_worker
    0,   # 5  human.pedestrian.personal_mobility
    7,   # 6  human.pedestrian.police_officer
    0,   # 7  human.pedestrian.stroller
    0,   # 8  human.pedestrian.wheelchair
    1,   # 9  movable_object.barrier
    0,   # 10 movable_object.debris
    0,   # 11 movable_object.pushable_pullable
    8,   # 12 movable_object.trafficcone
    0,   # 13 static_object.bicycle_rack
    2,   # 14 vehicle.bicycle
    3,   # 15 vehicle.bus.bendy
    3,   # 16 vehicle.bus.rigid
    4,   # 17 vehicle.car
    5,   # 18 vehicle.construction
    0,   # 19 vehicle.emergency.ambulance
    0,   # 20 vehicle.emergency.police
    6,   # 21 vehicle.motorcycle
    9,   # 22 vehicle.trailer
    10,  # 23 vehicle.truck
    11,  # 24 flat.driveable_surface
    12,  # 25 flat.other
    13,  # 26 flat.sidewalk
    14,  # 27 flat.terrain
    15,  # 28 static.manmade
    0,   # 29 static.other
    16,  # 30 static.vegetation
    0,   # 31 vehicle.ego
], dtype=np.uint16)


def build_staging(staging):
    table = json.load(open(VAL_TABLE))
    (staging / 'cloud').mkdir(parents=True, exist_ok=True)
    (staging / 'labels').mkdir(parents=True, exist_ok=True)
    out = []
    for r in table:
        tok = r['token']
        pc = np.fromfile(f"{NUSC}/samples/LIDAR_TOP/{r['file']}", dtype=np.float32)
        pc = pc.reshape(-1, 5)[:, :4].astype(np.float32)            # x,y,z,intensity
        sem = (np.load(f"{NUSC}/panoptic/v1.0-trainval/{tok}_panoptic.npz")['data'] // 1000)
        sem = np.where(sem < len(CHALLENGE_LUT), sem, 0)
        labels = CHALLENGE_LUT[sem].astype(np.uint16)               # 1..16, 0=ignore
        assert len(pc) == len(labels), f'{tok}: {len(pc)} pts vs {len(labels)} labels'
        pc.tofile(staging / 'cloud' / f'{tok}.bin')
        labels.tofile(staging / 'labels' / f'{tok}.label')
        out.append(tok)
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
            staging = Path(tempfile.mkdtemp(prefix='nusc_'))
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
                ds.source_url = 'https://www.nuscenes.org/nuscenes'
                ds.source_kind = 'local-lidar'
                ds.card_description = (
                    'nuScenes-lidarseg (Caesar et al., 2020 / Fong et al., 2022) — '
                    '3D LiDAR semantic segmentation. The 16-class lidarseg-challenge '
                    f'taxonomy over {n} keyframes from the {23} official validation '
                    'scenes present in the trainval part-1 blob (leakage-free: the '
                    'pretrained models trained only on the train split). Per-point '
                    '32-beam scans (x,y,z,intensity). Scored by point mIoU.')
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
        print(f'NUSCENES_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
