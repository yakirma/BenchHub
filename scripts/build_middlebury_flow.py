#!/usr/bin/env python
"""Build the Middlebury optical-flow leaderboard (training set, public GT).

The classic 8-pair optical-flow benchmark. Dense .flo GT (with Middlebury's
>1e9 "unknown flow" sentinel masked to NaN). Same representation as the other
flow boards (flow_u/flow_v depth fields, frame1/frame2 inputs, valid-aware EPE).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_middlebury_flow.py
"""
import os
import sys
import io
import json
import tempfile
import shutil
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

DS_NAME = 'Middlebury-flow'
GT_URL = 'https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip'
IMG_URL = 'https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip'
SCENES = ['Dimetrodon', 'Grove2', 'Grove3', 'Hydrangea', 'RubberWhale',
          'Urban2', 'Urban3', 'Venus']
UNKNOWN = 1e9

EPE_CODE = '''
def epe(gt_u, gt_v, pred_u, pred_v):
    """Optical-flow End-Point Error (lower is better): mean Euclidean flow-vector
    error over valid pixels (GT finite)."""
    import numpy as np

    def arr(x):
        if hasattr(x, 'array'):
            x = x.array
        return np.asarray(x, dtype=np.float32)

    gu, gv, pu, pv = arr(gt_u), arr(gt_v), arr(pred_u), arr(pred_v)
    if gu.ndim != 2 or gu.shape != gv.shape:
        return float('nan')

    def fit(p):
        if p.shape == gu.shape:
            return p
        H, W = gu.shape
        yi = np.linspace(0, p.shape[0] - 1, H).round().astype(int)
        xi = np.linspace(0, p.shape[1] - 1, W).round().astype(int)
        return p[yi][:, xi]

    pu, pv = fit(pu), fit(pv)
    m = np.isfinite(gu) & np.isfinite(gv)
    if not m.any():
        return float('nan')
    e = np.sqrt((gu - pu) ** 2 + (gv - pv) ** 2)
    return float(np.mean(e[m]))
'''


def read_flo(blob):
    import numpy as np
    f = io.BytesIO(blob)
    tag = np.frombuffer(f.read(4), np.float32)[0]
    if abs(tag - 202021.25) > 1e-1:
        raise ValueError(f'bad .flo tag {tag}')
    w = int(np.frombuffer(f.read(4), np.int32)[0])
    h = int(np.frombuffer(f.read(4), np.int32)[0])
    data = np.frombuffer(f.read(2 * w * h * 4), np.float32).reshape(h, w, 2)
    u = data[..., 0].astype('float32').copy()
    v = data[..., 1].astype('float32').copy()
    bad = (np.abs(u) >= UNKNOWN) | (np.abs(v) >= UNKNOWN)
    u[bad] = np.nan
    v[bad] = np.nan
    return u, v


def fetch(url, name):
    p = os.path.expanduser(f'~/.dtofbenchmarking/_cache_{name}')
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p + '.part'); os.replace(p + '.part', p)
    return zipfile.ZipFile(p)


def build_staging(staging: Path):
    import numpy as np
    zg = fetch(GT_URL, 'mb_gt.zip'); zi = fetch(IMG_URL, 'mb_img.zip')
    gnames = set(zg.namelist()); inames = set(zi.namelist())

    def find(names, scene, leaf):
        for n in names:
            if n.endswith(f'{scene}/{leaf}'):
                return n
        return None
    for d in ('frame1', 'frame2', 'flow_u', 'flow_v'):
        (staging / d).mkdir(parents=True, exist_ok=True)
    out = []
    for scene in SCENES:
        gp = find(gnames, scene, 'flow10.flo')
        i1 = find(inames, scene, 'frame10.png')
        i2 = find(inames, scene, 'frame11.png')
        if not (gp and i1 and i2):
            print('skip', scene, bool(gp), bool(i1), bool(i2)); continue
        u, v = read_flo(zg.read(gp))
        name = scene
        (staging / 'frame1' / f'{name}.png').write_bytes(zi.read(i1))
        (staging / 'frame2' / f'{name}.png').write_bytes(zi.read(i2))
        np.savez_compressed(staging / 'flow_u' / f'{name}.npz', depth=u)
        np.savez_compressed(staging / 'flow_v' / f'{name}.npz', depth=v)
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'frame1', 'kind': 'image', 'role': 'input', 'params': {}},
            {'name': 'frame2', 'kind': 'image', 'role': 'input', 'params': {}},
            {'name': 'flow_u', 'kind': 'depth', 'role': 'gt', 'params': {'unit': 'unitless'}},
            {'name': 'flow_v', 'kind': 'depth', 'role': 'gt', 'params': {'unit': 'unitless'}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(out)} Middlebury flow pairs')
    return out


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='mb_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('FLOW_SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Vision/Optical Flow'
                ds.source_url = 'https://vision.middlebury.edu/flow/'
                ds.source_kind = 'local-flow'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = GlobalMetric.query.filter_by(name='epe').first()
        ik = json.dumps(["depth", "depth", "depth", "depth"])
        if gm is None:
            gm = GlobalMetric(name='epe', python_code=EPE_CODE.strip(), owner_user_id=2,
                              visibility='public', is_aggregated=False, input_kinds=ik)
            db.session.add(gm); db.session.commit()
        elif not gm.input_kinds:
            gm.input_kinds = ik; db.session.commit()
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Vision/Optical Flow',
                required_pred_fields_json=json.dumps([
                    {"name": "flow_u_pred", "kind": "depth", "params": {"unit": "unitless"}, "role": "pred"},
                    {"name": "flow_v_pred", "kind": "depth", "params": {"unit": "unitless"}, "role": "pred"},
                ]),
                field_roles_json=json.dumps({'frame1': 'input', 'frame2': 'input',
                                             'flow_u': 'gt', 'flow_v': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='EPE',
                arg_mappings=json.dumps({"gt_u": "gt_flow_u", "gt_v": "gt_flow_v",
                                         "pred_u": "sub_flow_u_pred", "pred_v": "sub_flow_v_pred"}),
                pooling_type='mean', sort_direction='lower_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'FLOW_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
