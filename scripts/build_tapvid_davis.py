#!/usr/bin/env python
"""Build a video point-tracking leaderboard from TAP-Vid-DAVIS.

New frontier task type (Tracking Any Point). 30 real videos; for each we track
a set of query points across all frames and predict their 2D trajectory +
visibility. Evaluated at 256x256 in the standard TAP-Vid 'first'-query protocol
with the official metrics: Average Jaccard (AJ), < delta_avg (position
accuracy), and Occlusion Accuracy (OA) — all authored here in pure numpy.

Representation:
  video         sequence (input)  — the 256x256 frames
  query_points  json (input)      — [N,3] (t, x, y) first-visible query per point
  target_points json (gt)         — [N,T,2] GT trajectory at 256x256
  occluded      json (gt)         — [N,T] 0/1 occlusion
  pred_tracks   json (pred)       — [N,T,2]
  pred_occluded json (pred)       — [N,T]

A CoTracker model is scored by submit_cotracker.py.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_tapvid_davis.py [n_videos]
"""
import os
import sys
import json
import zipfile
import pickle
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

# TAPVID_SET selects the TAP-Vid dataset (same pkl schema across all): DAVIS
# (top-level dict) or RGB-Stacking (top-level list of dicts).
_SET = os.environ.get('TAPVID_SET', 'davis')
_CFG = {
    'davis': ('TAP-Vid-DAVIS', '_cache_tapvid_davis.zip', 'd'),
    'rgb_stacking': ('TAP-Vid-RGB-Stacking', '_cache_tapvid_rgb_stacking.zip', 'rgb'),
}[_SET]
DS_NAME, _ZIP, _PFX = _CFG
N_VID = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
RES = 256
CACHE = os.path.expanduser(f'~/.dtofbenchmarking/{_ZIP}')

# --- TAP-Vid metrics (pure numpy). Query frame derived from gt_occluded
#     (first visible frame per point); that frame is excluded from eval. ---
_HELP = '''
    import numpy as np

    def arr(x, dt=float):
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'value'):
            x = x.value
        return np.asarray(x, dtype=dt)

    def setup(gt_occ, gt_tracks, pred_tracks):
        go = arr(gt_occ, bool); gt = arr(gt_tracks); pt = arr(pred_tracks)
        if gt.ndim != 3 or gt.shape != pt.shape:
            return None
        N, T = go.shape
        ev = np.ones((N, T), bool)
        for n in range(N):
            vis_t = np.where(~go[n])[0]
            if len(vis_t):
                ev[n, vis_t[0]] = False        # exclude the query (first-visible) frame
        return go, gt, pt, ev
'''

AJ_CODE = '''
def tapvid_aj(gt_occ, gt_tracks, pred_occ, pred_tracks):
    """TAP-Vid Average Jaccard (higher is better): mean over thresholds
    {1,2,4,8,16}px (at 256x256) of Jaccard = TP/(TP+FP+FN), combining position
    accuracy and visibility prediction. Query frame excluded."""
__HELP__
    s = setup(gt_occ, gt_tracks, pred_tracks)
    if s is None:
        return float('nan')
    go, gt, pt, ev = s
    po = arr(pred_occ, bool)
    vis = ~go; pvis = ~po
    js = []
    for th in (1, 2, 4, 8, 16):
        within = np.sum((pt - gt) ** 2, axis=-1) < th * th
        tp = np.sum(within & vis & pvis & ev)
        fp = np.sum(((~vis) | (~within)) & pvis & ev)
        fn = np.sum(vis & (~(within & pvis)) & ev)
        js.append(tp / max(tp + fp + fn, 1))
    return float(np.mean(js))
'''

DELTA_CODE = '''
def tapvid_delta(gt_occ, gt_tracks, pred_occ, pred_tracks):
    """TAP-Vid < delta_avg (higher is better): fraction of visible points whose
    prediction is within {1,2,4,8,16}px of GT, averaged over thresholds."""
__HELP__
    s = setup(gt_occ, gt_tracks, pred_tracks)
    if s is None:
        return float('nan')
    go, gt, pt, ev = s
    vis = ~go; fr = []
    for th in (1, 2, 4, 8, 16):
        within = np.sum((pt - gt) ** 2, axis=-1) < th * th
        fr.append(np.sum(within & vis & ev) / max(np.sum(vis & ev), 1))
    return float(np.mean(fr))
'''

OA_CODE = '''
def tapvid_oa(gt_occ, gt_tracks, pred_occ, pred_tracks):
    """TAP-Vid Occlusion Accuracy (higher is better): fraction of evaluated
    frames where predicted visibility matches GT."""
__HELP__
    s = setup(gt_occ, gt_tracks, pred_tracks)
    if s is None:
        return float('nan')
    go, gt, pt, ev = s
    po = arr(pred_occ, bool)
    return float(np.sum(np.equal(po, go) & ev) / max(np.sum(ev), 1))
'''


def build_staging(staging: Path):
    import numpy as np
    from PIL import Image as PILImage
    import benchhub as bh
    z = zipfile.ZipFile(CACHE)
    pkl = [n for n in z.namelist() if n.endswith('.pkl')][0]
    data = pickle.loads(z.read(pkl))
    for d in ('video', 'query_points', 'target_points', 'occluded'):
        (staging / d).mkdir(parents=True, exist_ok=True)
    items = list(data.items()) if isinstance(data, dict) else list(enumerate(data))
    out = []
    for vid_key, d in items[:N_VID]:
        vname = vid_key if isinstance(vid_key, str) else f'{_PFX}_{vid_key:03d}'
        video = np.asarray(d['video'])              # [T,H,W,3] uint8
        pts = np.asarray(d['points'], dtype=np.float32)   # [N,T,2] in [0,1] (x,y)
        occ = np.asarray(d['occluded'], dtype=bool)       # [N,T]
        T = video.shape[0]
        N = pts.shape[0]
        # resize frames to 256x256, scale points to 256
        frames = [bh.Image(np.asarray(PILImage.fromarray(video[t]).resize((RES, RES))))
                  for t in range(T)]
        tgt = (pts * RES).astype(np.float32)        # [N,T,2] at 256
        # query = first visible frame per point
        queries = []
        for n in range(N):
            vis_t = np.where(~occ[n])[0]
            tq = int(vis_t[0]) if len(vis_t) else 0
            queries.append([tq, float(tgt[n, tq, 0]), float(tgt[n, tq, 1])])
        name = vname.replace('/', '_')
        (staging / 'video' / f'{name}.zip').write_bytes(
            bh.Sequence(frames, item_kind='image', fps=12).encode())
        (staging / 'query_points' / f'{name}.json').write_text(json.dumps(queries))
        (staging / 'target_points' / f'{name}.json').write_text(json.dumps(tgt.tolist()))
        (staging / 'occluded' / f'{name}.json').write_text(json.dumps(occ.astype(int).tolist()))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'video', 'kind': 'sequence', 'role': 'input',
             'params': {'item_kind': 'image', 'fps': 12}},
            {'name': 'query_points', 'kind': 'json', 'role': 'input', 'params': {}},
            {'name': 'target_points', 'kind': 'json', 'role': 'gt', 'params': {}},
            {'name': 'occluded', 'kind': 'json', 'role': 'gt', 'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(out)} videos')
    return out


def ensure_metric(db, GlobalMetric, name, code):
    gm = GlobalMetric.query.filter_by(name=name).first()
    if gm is None:
        gm = GlobalMetric(name=name, python_code=code.strip(), owner_user_id=2,
                          visibility='public', is_aggregated=False)
        db.session.add(gm); db.session.commit()
    return gm


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='tapvid_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('SKIP no videos'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Vision/Point Tracking'
                ds.source_url = 'https://tapvid.github.io/'
                ds.source_kind = 'local-pointtrack'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        ks = {'tapvid_aj': AJ_CODE, 'tapvid_delta': DELTA_CODE, 'tapvid_oa': OA_CODE}
        gms = {n: ensure_metric(db, GlobalMetric, n, c.replace("__HELP__", _HELP)) for n, c in ks.items()}
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Vision/Point Tracking',
                required_pred_fields_json=json.dumps([
                    {"name": "pred_tracks", "kind": "json", "params": {}, "role": "pred"},
                    {"name": "pred_occluded", "kind": "json", "params": {}, "role": "pred"},
                ]),
                field_roles_json=json.dumps({'video': 'input', 'query_points': 'input',
                                             'target_points': 'gt', 'occluded': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        am = json.dumps({"gt_occ": "gt_occluded", "gt_tracks": "gt_target_points",
                         "pred_occ": "sub_pred_occluded", "pred_tracks": "sub_pred_tracks"})
        sm = []
        for nm, tgt, prim in (('tapvid_aj', 'AJ', True), ('tapvid_delta', '<δ_avg', False),
                              ('tapvid_oa', 'OA', False)):
            gm = gms[nm]
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=tgt,
                    arg_mappings=am, pooling_type='mean', sort_direction='higher_is_better')
                db.session.add(lm); db.session.commit()
            if prim:
                sm.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(sm)
        db.session.commit()
        print(f'TAPVID_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
