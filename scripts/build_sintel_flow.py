#!/usr/bin/env python
"""Build an optical-flow leaderboard from MPI-Sintel (training split, public GT).

New task TYPE (Vision / Optical Flow). A flow field is HxWx2 (u,v); the typed
contract has no 2-channel float kind, so flow is represented as TWO depth-kind
fields, `flow_u` and `flow_v` (each HxW float32 npz) — reusing the proven
full-res depth pipeline. Inputs are the consecutive frame pair (frame1, frame2);
GT is the Middlebury .flo for frame1->frame2. The board scores End-Point Error
(EPE, lower is better). A RAFT model is scored by submit_flow.py.

One pass per run (clean|final); the ~5.6 GB complete zip is downloaded once
(HF cache) and reused.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_sintel_flow.py clean [n]
"""
import os
import sys
import io
import re
import json
import math
import tempfile
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

PASS = (sys.argv[1] if len(sys.argv) > 1 else 'clean').lower()
assert PASS in ('clean', 'final'), PASS
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DS_NAME = f'MPI-Sintel-{PASS}-flow'

EPE_CODE = '''
def epe(gt_u, gt_v, pred_u, pred_v):
    """Optical-flow End-Point Error (lower is better): mean over pixels of the
    Euclidean distance between predicted and ground-truth flow vectors. gt/pred
    are the u,v components (HxW float arrays). Prediction is nearest-resampled
    to the GT grid if shapes differ (no flow rescaling — the submitter is
    expected to emit GT-resolution flow; this is only a safety net)."""
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
        yi = (np.linspace(0, p.shape[0] - 1, H)).round().astype(int)
        xi = (np.linspace(0, p.shape[1] - 1, W)).round().astype(int)
        return p[yi][:, xi]

    pu, pv = fit(pu), fit(pv)
    err = np.sqrt((gu - pu) ** 2 + (gv - pv) ** 2)
    return float(np.mean(err))
'''


def read_flo(blob: bytes):
    import numpy as np
    f = io.BytesIO(blob)
    tag = np.frombuffer(f.read(4), np.float32)[0]
    if abs(tag - 202021.25) > 1e-1:
        raise ValueError(f'bad .flo tag {tag}')
    w = int(np.frombuffer(f.read(4), np.int32)[0])
    h = int(np.frombuffer(f.read(4), np.int32)[0])
    data = np.frombuffer(f.read(2 * w * h * 4), np.float32).reshape(h, w, 2)
    return data[..., 0].astype('float32'), data[..., 1].astype('float32')


def build_staging(staging: Path):
    import numpy as np
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download('ssbai/Sintel', 'MPI-Sintel-complete.zip', repo_type='dataset')
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    flo_re = re.compile(r'training/flow/([^/]+)/frame_(\d+)\.flo$')
    by_scene = defaultdict(list)
    for nm in names:
        m = flo_re.search(nm)
        if m:
            by_scene[m.group(1)].append((int(m.group(2)), nm))
    scenes = sorted(by_scene)
    per_scene = max(1, math.ceil(N / max(len(scenes), 1)))
    for d in ('frame1', 'frame2', 'flow_u', 'flow_v'):
        (staging / d).mkdir(parents=True, exist_ok=True)
    out = []
    for scene in scenes:
        frames = sorted(by_scene[scene])
        for idx, flo_name in frames[:per_scene]:
            if len(out) >= N:
                break
            img1 = f'training/{PASS}/{scene}/frame_{idx:04d}.png'
            img2 = f'training/{PASS}/{scene}/frame_{idx + 1:04d}.png'
            if img1 not in names or img2 not in names:
                continue
            try:
                u, v = read_flo(z.read(flo_name))
            except Exception:
                continue
            name = f'{scene}_{idx:04d}'
            (staging / 'frame1' / f'{name}.png').write_bytes(z.read(img1))
            (staging / 'frame2' / f'{name}.png').write_bytes(z.read(img2))
            np.savez_compressed(staging / 'flow_u' / f'{name}.npz', depth=u)
            np.savez_compressed(staging / 'flow_v' / f'{name}.npz', depth=v)
            out.append(name)
        if len(out) >= N:
            break
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
    print(f'{len(out)} frame pairs from {len(scenes)} scenes (pass={PASS})')
    return out


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='sintel_'))
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
                ds.source_url = 'http://sintel.is.tue.mpg.de/'
                ds.source_kind = 'local-flow'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = GlobalMetric.query.filter_by(name='epe').first()
        if gm is None:
            gm = GlobalMetric(name='epe', python_code=EPE_CODE.strip(), owner_user_id=2,
                              visibility='public', is_aggregated=False)
            db.session.add(gm); db.session.commit()
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
