#!/usr/bin/env python
"""Re-point the 3D Panoptic board (lb141) at a DIVERSE eval set. It previously
shared the 250-consecutive-scan dataset with the 4D board (lb142), but PQ is a
per-scan (non-temporal) metric, so 250 adjacent frames of one ~25 s clip is a
poor, highly-correlated eval. This builds a strided sample across the WHOLE of
seq 08 (every Nth scan) — like the semantic-seg board — and swaps lb141 onto it.
The 4D boards (lb142 consecutive, lb143 sub-sequences) are untouched.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking python scripts/build_panoptic_3d.py [stride]
"""
import os, sys, json, glob, shutil, tempfile
from pathlib import Path
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'
import numpy as np

KITTI = os.path.expanduser('~/.dtofbenchmarking/lidar_kitti')
SEQ = '08'
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 8       # every Nth scan across the full sequence
DS_NAME = 'SemanticKITTI-seq08-panoptic-diverse'
CATEGORY = 'Vision/3D Panoptic Segmentation'
LB_ID = 141
OLD_DS_ID = 345
LEARNING_MAP = {0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,
                49:12,50:13,51:14,52:0,60:9,70:15,71:16,72:17,80:18,81:19,99:0,252:1,253:7,
                254:6,255:8,256:5,257:5,258:4,259:5}
_LUT = np.zeros(260, np.uint32)
for k, v in LEARNING_MAP.items(): _LUT[k] = v


def build_staging(staging):
    vel = sorted(glob.glob(f'{KITTI}/dataset/sequences/{SEQ}/velodyne/*.bin'))
    lab = sorted(glob.glob(f'{KITTI}/data_odometry_labels/dataset/sequences/{SEQ}/labels/*.label'))
    idxs = list(range(0, len(vel), STRIDE))
    (staging / 'cloud').mkdir(parents=True, exist_ok=True)
    (staging / 'panoptic').mkdir(parents=True, exist_ok=True)
    out = []
    for i in idxs:
        name = f'{SEQ}_' + os.path.basename(vel[i])[:-4]
        np.fromfile(vel[i], np.float32).reshape(-1, 4).tofile(staging / 'cloud' / f'{name}.bin')
        raw = np.fromfile(lab[i], np.uint32)
        (_LUT[raw & 0xFFFF] | (raw & 0xFFFF0000)).astype(np.uint32).tofile(staging / 'panoptic' / f'{name}.label')
        out.append(name)
    manifest = {'name': DS_NAME, 'version': '1.0', 'fields': [
        {'name': 'cloud', 'kind': 'point_cloud', 'role': 'input', 'params': {'channels': 4}},
        {'name': 'panoptic', 'kind': 'point_panoptic', 'role': 'gt', 'params': {}}], 'samples': out}
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField, Leaderboard, Submission)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='pano3d_'))
            try:
                out = build_staging(staging)
                ds_id, _ = import_typed_dataset(staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField, upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False,
                    extra_kinds={'point_cloud': '.bin', 'point_panoptic': '.label'})
                db.session.commit(); ds = Dataset.query.get(ds_id)
                ds.category = CATEGORY; ds.source_url = 'https://semantic-kitti.org/'
                ds.card_description = (f'SemanticKITTI seq 08 (val) — {len(out)} frames sampled every {STRIDE}th scan '
                    'across the WHOLE sequence (diverse, non-consecutive) for an unbiased per-scan panoptic (PQ) eval.')
                db.session.commit(); print(f'imported diverse dataset id={ds_id}: {len(out)} strided frames (every {STRIDE}th)')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            print(f'dataset exists id={ds.id}')

        lb = Leaderboard.query.get(LB_ID)
        # swap the dataset: drop the 250-consecutive one, attach the diverse one
        old = Dataset.query.get(OLD_DS_ID)
        if old in lb.datasets:
            lb.datasets.remove(old)
        if ds not in lb.datasets:
            lb.datasets.append(ds)
        # archive the old (consecutive-frame) submissions — their sample names no longer match
        for sub in Submission.query.filter_by(leaderboard_id=LB_ID, is_archived=False).all():
            sub.is_archived = True
        db.session.commit()
        print(f'lb{LB_ID} re-pointed to ds{ds.id} ({DS_NAME}); old consecutive submissions archived')
        print('PANO_3D_REPOINTED')


if __name__ == '__main__':
    main()
