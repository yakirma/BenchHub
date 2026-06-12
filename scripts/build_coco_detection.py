#!/usr/bin/env python
"""Build a FULL-RES COCO val object-detection leaderboard + a mAP@0.5 metric.

COCO val2017 is the standard detection benchmark; DETR/YOLOS/RT-DETR are all
COCO-trained, so their class space matches the GT exactly (we align by class
NAME). We import a full-res subset (raw image + GT boxes-as-JSON) so detectors
can be scored.

GT/pred JSON schema (absolute pixel xyxy, in the input image's coords):
    gt:   {"boxes": [[x1,y1,x2,y2], ...], "labels": ["person", ...]}
    pred: {"boxes": [...], "scores": [...], "labels": [...]}

Creates the `map50` GlobalMetric (per-sample mean AP@0.5 over GT classes,
greedy IoU>=0.5 matching, all-point AP; higher is better) if missing.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_coco_detection.py [n_samples]
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
DS_NAME = 'COCO-val2017-detection'
IMG, OBJ = 'image', 'objects'

MAP_CODE = r'''import numpy as np


def map50(gt, pred):
    """mAP@0.5 for object detection (higher is better). Per-sample mean AP
    over the classes present in GT; pred boxes are matched to GT greedily by
    IoU>=0.5 in score order. gt/pred are dicts: boxes [[x1,y1,x2,y2], ...] +
    labels [name, ...]; pred also has scores."""
    def unwrap(d):
        if d is None:
            return {}
        if hasattr(d, 'value'):
            d = d.value
        return d if isinstance(d, dict) else {}
    g = unwrap(gt); p = unwrap(pred)
    gboxes = np.asarray(g.get('boxes') or [], dtype=float).reshape(-1, 4)
    glabels = [str(x).strip().lower() for x in (g.get('labels') or [])]
    pboxes = np.asarray(p.get('boxes') or [], dtype=float).reshape(-1, 4)
    plabels = [str(x).strip().lower() for x in (p.get('labels') or [])]
    pscores = list(p.get('scores') or [1.0] * len(pboxes))
    if len(glabels) == 0:
        return float('nan')

    def iou(a, B):
        if len(B) == 0:
            return np.zeros((0,))
        x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
        x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
        area_B = (np.clip(B[:, 2] - B[:, 0], 0, None)
                  * np.clip(B[:, 3] - B[:, 1], 0, None))
        return inter / np.clip(area_a + area_B - inter, 1e-9, None)

    aps = []
    for cls in set(glabels):
        gidx = [i for i, l in enumerate(glabels) if l == cls]
        gb = gboxes[gidx]
        pidx = [i for i, l in enumerate(plabels) if l == cls]
        if not pidx:
            aps.append(0.0); continue
        order = sorted(pidx, key=lambda i: -pscores[i])
        matched = np.zeros(len(gb), dtype=bool)
        tp = np.zeros(len(order)); fp = np.zeros(len(order))
        for k, i in enumerate(order):
            ious = iou(pboxes[i], gb)
            if len(ious) and ious.max() >= 0.5 and not matched[int(ious.argmax())]:
                tp[k] = 1; matched[int(ious.argmax())] = True
            else:
                fp[k] = 1
        tpc = np.cumsum(tp); fpc = np.cumsum(fp)
        recall = tpc / len(gb)
        precision = tpc / np.clip(tpc + fpc, 1e-9, None)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
    return float(np.mean(aps)) if aps else float('nan')
'''


def build_staging(staging: Path):
    import itertools
    from datasets import load_dataset
    (staging / IMG).mkdir(parents=True, exist_ok=True)
    (staging / OBJ).mkdir(parents=True, exist_ok=True)
    ds = load_dataset('rafaelpadilla/coco2017', split='val', streaming=True)
    names = ds.features['objects'].feature['label'].names  # idx -> COCO name
    out = []
    for i, row in enumerate(itertools.islice(ds, N)):
        img = row['image'].convert('RGB')
        o = row['objects']
        boxes, labels = [], []
        for bb, lab in zip(o['bbox'], o['label']):
            nm = names[int(lab)]
            if nm in ('None', 'N/A'):
                continue
            x, y, w, h = bb
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            labels.append(nm)
        name = f'c_{int(row.get("image_id", i)):012d}'
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
        gm = GlobalMetric.query.filter_by(name='map50').first()
        if gm is None:
            gm = GlobalMetric(
                name='map50', description='mAP@0.5 for object detection (higher is better).',
                python_code=MAP_CODE, is_aggregated=0, accepts_aggregated_inputs=0,
                input_kinds=None, input_roles='["gt", "pred"]',
                owner_user_id=None, visibility='public')
            db.session.add(gm); db.session.commit()
            print('created map50 metric')

        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='coco_det_'))
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
                ds.source_url = 'https://huggingface.co/datasets/rafaelpadilla/coco2017'
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
