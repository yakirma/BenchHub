#!/usr/bin/env python
"""Build the KITTI-2015 optical-flow leaderboard (training split, public GT).

Second flagship optical-flow benchmark after Sintel — real driving scenes with
sparse LiDAR/CAD-derived GT. Same representation as the Sintel boards (flow as
two depth fields flow_u/flow_v, inputs frame1/frame2), but GT is SPARSE: invalid
pixels are stored as NaN and the metrics ignore them. Binds the canonical
KITTI Fl-all (outlier %) as primary + EPE. RAFT C+T weights (leakage-free
generalization) are scored by submit_flow.py.

GT format: training/flow_occ/<id>_10.png — 16-bit 3-channel PNG, valid=ch2>0,
u=(ch0-2^15)/64, v=(ch1-2^15)/64.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_kitti_flow.py [n]
"""
import os
import sys
import io
import re
import json
import tempfile
import shutil
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

YEAR = os.environ.get('KITTI_YEAR', '2015')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
if YEAR == '2012':
    DS_NAME = 'KITTI-2012-flow'
    ZIP_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip'
    CACHE_ZIP = os.path.expanduser('~/.dtofbenchmarking/_cache_data_stereo_flow.zip')
    IMG_DIR = 'colored_0'           # KITTI-2012 frames live under colored_0/
else:
    DS_NAME = 'KITTI-2015-flow'
    ZIP_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip'
    CACHE_ZIP = os.path.expanduser('~/.dtofbenchmarking/_cache_data_scene_flow.zip')
    IMG_DIR = 'image_2'             # KITTI-2015 frames live under image_2/

# Valid-aware metrics shared with the Sintel boards (Sintel GT is all-finite, so
# the mask is a no-op there). epe is overwritten in place to the masked version.
EPE_CODE = '''
def epe(gt_u, gt_v, pred_u, pred_v):
    """Optical-flow End-Point Error (lower is better): mean Euclidean flow-vector
    error over VALID pixels (GT finite). Prediction nearest-resampled to the GT
    grid if shapes differ. gt/pred are u,v components (HxW)."""
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

FL_ALL_CODE = '''
def fl_all(gt_u, gt_v, pred_u, pred_v):
    """KITTI Fl-all outlier rate (%, lower is better): fraction of VALID pixels
    whose End-Point Error exceeds BOTH 3 px AND 5% of the GT flow magnitude."""
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
    mag = np.sqrt(gu ** 2 + gv ** 2)
    bad = (e > 3.0) & (e > 0.05 * mag)
    return float(100.0 * np.sum(bad & m) / np.sum(m))
'''


def fetch_zip():
    if not os.path.exists(CACHE_ZIP) or os.path.getsize(CACHE_ZIP) < 1_000_000_000:
        print('downloading data_scene_flow.zip (~1.68 GB)...', flush=True)
        urllib.request.urlretrieve(ZIP_URL, CACHE_ZIP + '.part')
        os.replace(CACHE_ZIP + '.part', CACHE_ZIP)
    return CACHE_ZIP


def decode_flow(blob):
    # KITTI flow GT is a 16-bit RGB PNG [u, v, valid]. Pillow-backed readers
    # (imageio/PIL) silently downconvert to 8-bit, zeroing the valid channel and
    # losing u/v precision, so decode the true 16 bits with pypng.
    import numpy as np
    import png
    w, h, rows, info = png.Reader(bytes=blob).read_flat()
    a = np.array(rows).reshape(h, w, info['planes'])  # uint16 HxWx3
    valid = a[..., 2] > 0
    u = (a[..., 0].astype(np.float32) - 2 ** 15) / 64.0
    v = (a[..., 1].astype(np.float32) - 2 ** 15) / 64.0
    u[~valid] = np.nan
    v[~valid] = np.nan
    return u, v


def build_staging(staging: Path):
    import numpy as np
    z = zipfile.ZipFile(fetch_zip())
    names = set(z.namelist())
    flo_re = re.compile(r'training/flow_occ/(\d+)_10\.png$')
    ids = sorted({flo_re.search(n).group(1) for n in names if flo_re.search(n)})
    for d in ('frame1', 'frame2', 'flow_u', 'flow_v'):
        (staging / d).mkdir(parents=True, exist_ok=True)
    out = []
    for sid in ids[:N]:
        img1 = f'training/{IMG_DIR}/{sid}_10.png'
        img2 = f'training/{IMG_DIR}/{sid}_11.png'
        gt = f'training/flow_occ/{sid}_10.png'
        if img1 not in names or img2 not in names or gt not in names:
            continue
        try:
            u, v = decode_flow(z.read(gt))
        except Exception as e:
            print('decode skip', sid, e); continue
        name = f'k15_{sid}'
        (staging / 'frame1' / f'{name}.png').write_bytes(z.read(img1))
        (staging / 'frame2' / f'{name}.png').write_bytes(z.read(img2))
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
    print(f'{len(out)} KITTI-2015 flow pairs')
    return out


def ensure_metric(db, GlobalMetric, name, code):
    # input_kinds=depth×4 → compact typed (npz) sandbox transport; the primitive
    # path JSON-encodes the dense flow arrays (~3 MB/arg) and OOMs the memory-
    # capped container (and can't carry the sparse-GT NaNs).
    ik = json.dumps(["depth", "depth", "depth", "depth"])
    gm = GlobalMetric.query.filter_by(name=name).first()
    if gm is None:
        gm = GlobalMetric(name=name, python_code=code.strip(), owner_user_id=2,
                          visibility='public', is_aggregated=False, input_kinds=ik)
        db.session.add(gm)
    else:
        gm.python_code = code.strip()   # refresh epe -> valid-aware
        gm.input_kinds = ik
    db.session.commit()
    return gm


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='kitti15_'))
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
                ds.source_url = 'https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php'
                ds.source_kind = 'local-flow'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm_fl = ensure_metric(db, GlobalMetric, 'fl_all', FL_ALL_CODE)
        gm_epe = ensure_metric(db, GlobalMetric, 'epe', EPE_CODE)
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
        sm = []
        for gm, tgt, prim in ((gm_fl, 'Fl-all (%)', True), (gm_epe, 'EPE', False)):
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=tgt,
                    arg_mappings=json.dumps({"gt_u": "gt_flow_u", "gt_v": "gt_flow_v",
                                             "pred_u": "sub_flow_u_pred", "pred_v": "sub_flow_v_pred"}),
                    pooling_type='mean', sort_direction='lower_is_better')
                db.session.add(lm); db.session.commit()
            if prim:
                sm.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(sm)
        db.session.commit()
        print(f'FLOW_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
