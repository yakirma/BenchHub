#!/usr/bin/env python
"""Build the per-SEQUENCE SemanticKITTI 4D Panoptic board — every SAMPLE is one
temporal sub-sequence (a continuous run of scans), scored by the official LSTQ
over that sub-sequence. The official benchmark evaluates whole sequences; seq 08
is the only held-out (leakage-free) one, so we split it into N continuous
sub-sequences to get many leakage-free temporal-sequence samples.

Each sample stores: panoptic GT (full-res, all scans concatenated) + a per-point
scan-index (so the metric reproduces the exact per-scan accumulation) + the
middle scan's cloud (for a card thumbnail). Metric `lstq_seq` is per-sample and
runs in-process (admin-trusted, oversized — see metric_engine).

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking python scripts/build_panoptic_seq.py [scans_per_seq]
"""
import os, sys, json, glob, shutil, tempfile
from pathlib import Path
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'
import numpy as np

KITTI = os.path.expanduser('~/.dtofbenchmarking/lidar_kitti')
SEQ = '08'
L = int(sys.argv[1]) if len(sys.argv) > 1 else 100          # scans per sub-sequence
DS_NAME = 'SemanticKITTI-seq08-4dpanoptic-seq'
CATEGORY = 'Vision/4D Panoptic Segmentation'
LEARNING_MAP = {0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,
                49:12,50:13,51:14,52:0,60:9,70:15,71:16,72:17,80:18,81:19,99:0,252:1,253:7,
                254:6,255:8,256:5,257:5,258:4,259:5}
_LUT = np.zeros(260, np.uint32)
for k, v in LEARNING_MAP.items(): _LUT[k] = v

LSTQ_SEQ = r'''
def lstq_seq(gt, pred, scan):
    """Per-sub-sequence LiDAR Seg & Tracking Quality (LSTQ / PQ4D), the official
    SemanticKITTI 4D-panoptic metric (higher is better), for a board where each
    SAMPLE is one temporal sub-sequence. gt/pred/scan are per-point arrays for the
    whole sub-sequence (concatenated across its scans): gt/pred are uint32
    point_panoptic (sem low16 | track-id high16); `scan` is the per-point scan
    index. We split by scan index and accumulate per-scan exactly as the official
    evaluator does, so this matches evaluate_4dpanoptic over the sub-sequence."""
    import numpy as np, math
    NC = 20; INCLUDE = list(range(1, 20)); MINPTS = 50; OFFSET = 1 << 32; EPS = 1e-15
    g = np.asarray(gt).ravel().astype(np.uint32); p = np.asarray(pred).ravel().astype(np.uint32)
    s = np.asarray(scan).ravel().astype(np.int64)
    n = min(len(g), len(p), len(s)); g, p, s = g[:n], p[:n], s[:n]
    conf = np.zeros((NC, NC), np.int64)
    preds = {}; gts = [{} for _ in range(NC)]; inter = [{} for _ in range(NC)]
    def upd(d, ids, cnts):
        for i, c in zip(ids, cnts):
            if i == 1: continue
            d[i] = d.get(i, 0) + c
    for sc in np.unique(s):                                   # per-scan accumulation (official)
        m = s == sc
        gs = (g[m] & 0xFFFF).astype(np.int64); gi = (g[m] >> 16).astype(np.int64) + 1
        ps = (p[m] & 0xFFFF).astype(np.int64); pi = (p[m] >> 16).astype(np.int64) + 1
        np.add.at(conf, (ps, gs), 1)
        keep = gs != 0
        gs, gi, ps, pi = gs[keep], gi[keep], ps[keep], pi[keep]
        for cl in INCLUDE:
            xic = pi * (ps == cl); yic = gi * (gs == cl)
            ug, cg = np.unique(yic[yic > 0], return_counts=True)
            k = cg > MINPTS
            upd(gts[cl], ug[k], cg[k])
            yic = yic * np.isin(yic, ug[k])
            up, cp = np.unique(xic[xic > 0], return_counts=True)
            upd(preds, up, cp)
            valid = (pi > 0) & (yic > 0)
            combo = pi[valid] + OFFSET * yic[valid]
            uc, cc = np.unique(combo, return_counts=True)
            upd(inter[cl], uc, cc)
    num_tubes = [0] * NC; aq_ovr = 0.0
    for cl in INCLUDE:
        num_tubes[cl] += len(gts[cl])
        for gid, gsz in gts[cl].items():
            inner = 0.0
            for pid, psz in preds.items():
                key = pid + OFFSET * gid
                if key in inter[cl]:
                    tpa = inter[cl][key]; inner += tpa * (tpa / (gsz + psz - tpa))
            aq_ovr += inner / float(gsz)
    S_assoc = aq_ovr / max(sum(num_tubes[1:9]), EPS)
    c = conf.copy(); c[0, :] = 0; c[:, 0] = 0
    tp = np.diag(c); fp = c.sum(1) - tp; fn = c.sum(0) - tp
    union = tp + fp + fn; present = union > 0
    iou = tp / np.maximum(union, EPS)
    S_cls = np.mean([iou[k] for k in INCLUDE if present[k]]) if present[1:].any() else 0.0
    return float(math.sqrt(max(S_assoc, 0.0) * max(S_cls, 0.0)))
'''


def build_staging(staging):
    vel = sorted(glob.glob(f'{KITTI}/dataset/sequences/{SEQ}/velodyne/*.bin'))
    lab = sorted(glob.glob(f'{KITTI}/data_odometry_labels/dataset/sequences/{SEQ}/labels/*.label'))
    nseq = len(vel) // L
    for d in ('cloud', 'panoptic', 'scan'):
        (staging / d).mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(nseq):
        frames = range(k * L, k * L + L)
        pan_parts, scan_parts = [], []
        for j, fi in enumerate(frames):
            raw = np.fromfile(lab[fi], np.uint32)
            pan_parts.append((_LUT[raw & 0xFFFF] | (raw & 0xFFFF0000)).astype(np.uint32))
            scan_parts.append(np.full(len(raw), j, np.uint16))
        name = f'{SEQ}_seq{k:02d}'
        np.concatenate(pan_parts).tofile(staging / 'panoptic' / f'{name}.label')
        np.concatenate(scan_parts).tofile(staging / 'scan' / f'{name}.label')
        # middle scan's cloud, for a card thumbnail
        mid = np.fromfile(vel[k * L + L // 2], np.float32).reshape(-1, 4)
        mid.tofile(staging / 'cloud' / f'{name}.bin')
        out.append(name)
    manifest = {'name': DS_NAME, 'version': '1.0', 'fields': [
        {'name': 'cloud', 'kind': 'point_cloud', 'role': 'input', 'params': {'channels': 4}},
        {'name': 'panoptic', 'kind': 'point_panoptic', 'role': 'gt', 'params': {}},
        {'name': 'scan', 'kind': 'point_labels', 'role': 'gt', 'params': {}}], 'samples': out}
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out, nseq


def main():
    import app as A
    from app import (db, GlobalMetric, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        if not GlobalMetric.query.filter_by(name='lstq_seq').first():
            db.session.add(GlobalMetric(name='lstq_seq', python_code=LSTQ_SEQ.strip(),
                                        owner_user_id=2, visibility='public', is_aggregated=False))
            db.session.commit()
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='panoseq_'))
            try:
                out, nseq = build_staging(staging)
                ds_id, _ = import_typed_dataset(staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField, upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False,
                    extra_kinds={'point_cloud': '.bin', 'point_panoptic': '.label', 'point_labels': '.label'})
                db.session.commit(); ds = Dataset.query.get(ds_id)
                ds.category = CATEGORY; ds.source_url = 'https://semantic-kitti.org/'
                ds.card_description = (f'SemanticKITTI seq 08 (val, leakage-free) split into {nseq} continuous '
                    f'temporal sub-sequences of {L} scans (~{L//10}s) each. Every SAMPLE is one sub-sequence; '
                    'scored by the official 4D LSTQ over its scans. Per-point panoptic + track-id GT.')
                db.session.commit(); print(f'imported dataset id={ds_id}: {len(out)} sub-sequences of {L} scans')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            print(f'dataset exists id={ds.id}')
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(name=lb_name, owner_user_id=2, visibility='public', category=CATEGORY,
                required_pred_fields_json=json.dumps([{'name': 'panoptic_pred', 'kind': 'point_panoptic', 'params': {}, 'role': 'pred'}]),
                field_roles_json=json.dumps({'cloud': 'input', 'panoptic': 'gt', 'scan': 'gt'}), summary_metrics='')
            lb.datasets.append(ds); db.session.add(lb); db.session.commit()
        gm = GlobalMetric.query.filter_by(name='lstq_seq').first()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(leaderboard_id=lb.id, global_metric_id=gm.id, target_name='LSTQ',
                arg_mappings=json.dumps({'gt': 'gt_panoptic', 'pred': 'sub_panoptic_pred', 'scan': 'gt_scan'}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        cur = set(c.strip() for c in (lb.hidden_comparison_display_columns or '').split(',') if c.strip())
        cur.update(['cloud', 'panoptic', 'scan', 'panoptic_pred']); lb.hidden_comparison_display_columns = ','.join(sorted(cur))
        db.session.commit()
        print(f'PANO_SEQ_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
