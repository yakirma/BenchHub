#!/usr/bin/env python
"""Build a FULL-RES object-detection leaderboard from a keremberke-style HF
detection repo (image + COCO-style `objects`), + bind the detection overlay.

The catalog copies are preview-only (downscaled JPGs; GT boxes in original
coords) → not scorable. This re-imports a full-res split (raw image + GT boxes
as {boxes:[x1,y1,x2,y2], labels:[name]} JSON, in image coords) so a matching
YOLOv8 model can be scored with mAP@0.5.

Env params:
    DET_REPO   HF repo (e.g. keremberke/blood-cell-object-detection)
    DET_NAME   dataset/LB base name (e.g. BloodCell)
    DET_SPLIT  HF split (default: test)

Usage:
    DET_REPO=keremberke/blood-cell-object-detection DET_NAME=BloodCell \
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_hf_detection.py [n_samples]
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

REPO = os.environ['DET_REPO']
DS_NAME = os.environ['DET_NAME'] + '-detection'
SPLIT = os.environ.get('DET_SPLIT', 'test')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
IMG, OBJ = 'image', 'objects'


def build_staging(staging: Path):
    import itertools
    from datasets import load_dataset
    (staging / IMG).mkdir(parents=True, exist_ok=True)
    (staging / OBJ).mkdir(parents=True, exist_ok=True)
    cfg = os.environ.get('DET_CONFIG', 'full')
    load = (lambda sp: load_dataset(REPO, split=sp, streaming=True)) if cfg in ('', 'none') \
        else (lambda sp: load_dataset(REPO, cfg, split=sp, streaming=True))
    try:
        ds = load(SPLIT)
    except Exception:
        ds = load('validation' if SPLIT != 'validation' else 'train')
    names = ds.features['objects'].feature['category'].names
    out = []
    for i, row in enumerate(itertools.islice(ds, N)):
        img = row['image'].convert('RGB')
        o = row['objects']
        # Most COCO-style repos use xywh; some (detection-datasets/*) use xyxy.
        xyxy = os.environ.get('DET_BBOX_FORMAT', 'xywh') == 'xyxy'
        boxes, labels = [], []
        for bb, cat in zip(o['bbox'], o['category']):
            a, b, c, d = [float(t) for t in bb]
            boxes.append([a, b, c, d] if xyxy else [a, b, a + c, b + d])
            labels.append(names[int(cat)])
        name = f'd_{i:06d}'
        img.save(staging / IMG / f'{name}.png')
        (staging / OBJ / f'{name}.json').write_text(json.dumps({'boxes': boxes, 'labels': labels}))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': IMG, 'kind': 'image', 'role': 'input', 'params': {}},
            {'name': OBJ, 'kind': 'json', 'role': 'gt', 'params': {}},
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
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='hfdet_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('DET_SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Vision/Object Detection'
                ds.source_url = f'https://huggingface.co/datasets/{REPO}'
                ds.source_kind = 'local-detection'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Vision/Object Detection',
                required_pred_fields_json=json.dumps(
                    [{"name": "detections_pred", "kind": "json", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({IMG: 'input', OBJ: 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        gm = GlobalMetric.query.filter_by(name='map50').first()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='mAP@0.5',
                arg_mappings=json.dumps({"gt": f"gt_{OBJ}", "pred": "sub_detections_pred"}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'DET_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
