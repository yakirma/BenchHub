#!/usr/bin/env python
"""Build a FULL-RES ADE20K semantic-segmentation leaderboard.

The catalog already has a preview-only ADE20K (id 92), but its masks are lossy
JPEGs — unusable as scoring GT. This imports a full-res eval subset from HF
`scene_parse_150` (the canonical ADE20K SceneParse150 benchmark, 150 classes)
with RAW class-id PNG masks, so submitted segmentation models can actually be
scored with mIoU.

Encoding: scene_parse_150 `annotation` is 0..150 where 0 = "other/unlabeled"
and 1..150 = the 150 ADE classes (official order). We remap 0 -> 255 (the
iou_mask metric's default ignore_index) and keep 1..150. A SegFormer model
(id2label 0..149, same class order) submits argmax+1, so pred class k -> GT
label k+1. Both stored at native image resolution so pred (resized to image
size) matches the GT shape the metric requires.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_ade20k_semseg.py [n_samples]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
DS_NAME = 'ADE20K-SceneParse150'
IMG, SEG = 'image', 'segmentation'
METRICS = [('iou_mask', 'mIoU', 'higher_is_better')]


def build_staging(staging: Path):
    import itertools
    import numpy as np
    from PIL import Image
    from datasets import load_dataset
    (staging / IMG).mkdir(parents=True, exist_ok=True)
    (staging / SEG).mkdir(parents=True, exist_ok=True)
    # Stream so we only fetch N samples, not the whole 2GB validation set.
    ds = load_dataset('scene_parse_150', split='validation', streaming=True,
                      trust_remote_code=True)
    names = []
    for i, row in enumerate(itertools.islice(ds, N)):
        img = row['image'].convert('RGB')
        ann = np.asarray(row['annotation'], dtype=np.int32)
        if ann.ndim == 3:
            ann = ann[..., 0]
        if ann.shape[:2] != (img.height, img.width):
            # keep them identical — resize annotation (nearest) to the image
            ann = np.asarray(Image.fromarray(ann.astype(np.uint8)).resize(
                (img.width, img.height), Image.NEAREST), dtype=np.int32)
        # 0 (other/unlabeled) -> 255 ignore; classes 1..150 stay.
        m = np.full(ann.shape, 255, dtype=np.uint8)
        keep = (ann >= 1) & (ann <= 150)
        m[keep] = ann[keep].astype(np.uint8)
        name = f's_{i:05d}'
        img.save(staging / IMG / f'{name}.png')
        Image.fromarray(m, mode='L').save(staging / SEG / f'{name}.png')
        names.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': IMG, 'kind': 'image', 'role': 'input', 'params': {}},
            {'name': SEG, 'kind': 'mask', 'role': 'gt', 'params': {'ignore_index': 255}},
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
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='ade20k_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('ADE_SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Vision/Semantic Segmentation'
                ds.source_url = 'https://huggingface.co/datasets/scene_parse_150'
                ds.source_kind = 'local-semseg'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Vision/Semantic Segmentation',
                required_pred_fields_json=json.dumps(
                    [{"name": "segmentation_pred", "kind": "mask", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({IMG: 'input', SEG: 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        keep = []
        for gname, label, sort_dir in METRICS:
            gm = GlobalMetric.query.filter_by(name=gname).first()
            if gm is None:
                print(f'  WARN metric {gname} missing'); continue
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=label,
                    arg_mappings=json.dumps({"gt": f"gt_{SEG}", "pred": "sub_segmentation_pred"}),
                    pooling_type='mean', sort_direction=sort_dir)
                db.session.add(lm); db.session.commit()
            keep.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(dict.fromkeys(keep))
        db.session.commit()
        print(f'SEMSEG_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name} metrics={len(keep)}')


if __name__ == '__main__':
    raise SystemExit(main())
