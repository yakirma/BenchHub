#!/usr/bin/env python
"""Add the Middlebury Stereo Evaluation v3 metrics to the stereo board (lb83).

Middlebury reports, over valid pixels: bad0.5 / bad1.0 / bad2.0 / bad4.0
(% pixels with disparity error above the threshold), avgerr, rms, and the
A50 / A90 / A95 / A99 error quantiles. They assume a GROUND-TRUTH DISPARITY in
pixels — but this board's GT is a turbo-decoded RELATIVE depth. So each metric
shares a preprocessing step: convert GT relative depth -> relative disparity,
normalise it to a canonical [0,64]px range (1st..99th pctile), resize + affine-
align the prediction's disparity to it, and compute the error in those units.
The thresholds are therefore relative to a normalised 64px max disparity.
"""
import os
import sys
import json

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

LB = 83
DMAX = 64.0  # canonical normalised max disparity (Middlebury-like)

HELP = '''
    def _arr(x):
        if x is None:
            return None
        a = x.array if hasattr(x, "array") else x
        a = np.asarray(a, dtype=np.float64)
        return a[..., 0] if a.ndim == 3 else a
    def _resize_nn(a, shape):
        if a.shape == shape:
            return a
        H, W = shape
        yi = np.clip(np.round(np.linspace(0, a.shape[0]-1, H)).astype(int), 0, a.shape[0]-1)
        xi = np.clip(np.round(np.linspace(0, a.shape[1]-1, W)).astype(int), 0, a.shape[1]-1)
        return a[yi][:, xi]
    def _err(gt, pred):
        g = _arr(gt); p = _arr(pred)
        if g is None or p is None:
            return None
        # GT: relative depth -> relative disparity (invert unless already inverse).
        gdisp = g if getattr(gt, "is_inverse", False) else 1.0 / np.clip(g, 1e-6, None)
        p = _resize_nn(p, gdisp.shape)
        pdisp = p if getattr(pred, "is_inverse", False) else 1.0 / np.clip(p, 1e-6, None)
        m = np.isfinite(gdisp) & np.isfinite(pdisp) & (g > 1e-3)
        if int(m.sum()) < 10:
            return None
        gg = gdisp[m]; pp = pdisp[m]
        lo, hi = np.percentile(gg, 1), np.percentile(gg, 99)
        if hi - lo < 1e-9:
            return None
        gg = np.clip((gg - lo) / (hi - lo), 0.0, 1.0) * %f
        A = np.stack([pp, np.ones_like(pp)], axis=1)
        sol, *_ = np.linalg.lstsq(A, gg, rcond=None)
        pa = sol[0] * pp + sol[1]
        return np.abs(pa - gg)
''' % DMAX


def _metric(fn, doc, stat):
    return ('import numpy as np\nimport benchhub as bh\n\n'
            f'def {fn}(gt: bh.Depth, pred: bh.Depth):\n'
            f'    """{doc}"""\n'
            + HELP +
            '    e = _err(gt, pred)\n'
            '    if e is None:\n        return float("nan")\n'
            + stat)


# (global_metric_name, display label, doc, statistic code, sort_dir)
SPECS = [
    ('mb_bad05', 'bad0.5', 'Middlebury bad0.5: %% of pixels with disparity error > 0.5px (normalised 64px scale; lower better).',
     '    return float(np.mean(e > 0.5) * 100.0)\n', 'lower_is_better'),
    ('mb_bad1', 'bad1.0', 'Middlebury bad1.0: %% of pixels with disparity error > 1px (lower better).',
     '    return float(np.mean(e > 1.0) * 100.0)\n', 'lower_is_better'),
    ('mb_bad2', 'bad2.0', 'Middlebury bad2.0: %% of pixels with disparity error > 2px (lower better).',
     '    return float(np.mean(e > 2.0) * 100.0)\n', 'lower_is_better'),
    ('mb_bad4', 'bad4.0', 'Middlebury bad4.0: %% of pixels with disparity error > 4px (lower better).',
     '    return float(np.mean(e > 4.0) * 100.0)\n', 'lower_is_better'),
    ('mb_avgerr', 'avgerr', 'Middlebury avgerr: mean absolute disparity error (normalised 64px scale; lower better).',
     '    return float(np.mean(e))\n', 'lower_is_better'),
    ('mb_rms', 'rms', 'Middlebury rms: root-mean-square disparity error (lower better).',
     '    return float(np.sqrt(np.mean(e ** 2)))\n', 'lower_is_better'),
    ('mb_a50', 'A50', 'Middlebury A50: 50th-percentile (median) disparity error (lower better).',
     '    return float(np.percentile(e, 50))\n', 'lower_is_better'),
    ('mb_a90', 'A90', 'Middlebury A90: 90th-percentile disparity error (lower better).',
     '    return float(np.percentile(e, 90))\n', 'lower_is_better'),
    ('mb_a95', 'A95', 'Middlebury A95: 95th-percentile disparity error (lower better).',
     '    return float(np.percentile(e, 95))\n', 'lower_is_better'),
    ('mb_a99', 'A99', 'Middlebury A99: 99th-percentile disparity error (lower better).',
     '    return float(np.percentile(e, 99))\n', 'lower_is_better'),
]


def main():
    import app as A
    from app import db, GlobalMetric, LeaderboardMetric, Leaderboard
    with A.app.app_context():
        lb = Leaderboard.query.get(LB)
        existing = {f'lm_{lm.id}' for lm in LeaderboardMetric.query.filter_by(leaderboard_id=LB).all()}
        keep = list(existing)  # keep current scale-invariant metrics
        for gname, label, doc, stat, sort_dir in SPECS:
            code = _metric(gname, doc, stat)
            gm = GlobalMetric.query.filter_by(name=gname).first()
            if gm is None:
                gm = GlobalMetric(name=gname, description=doc, python_code=code,
                                  is_aggregated=0, accepts_aggregated_inputs=0,
                                  input_kinds='["depth", "depth"]', input_roles='["gt", "pred"]',
                                  owner_user_id=None, visibility='public')
                db.session.add(gm)
            else:
                gm.python_code = code; gm.description = doc
            db.session.commit()
            # bind to lb83 if not already bound
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=LB, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=LB, global_metric_id=gm.id, target_name=label,
                    arg_mappings=json.dumps({"gt": "gt_cam_01_depth_vis", "pred": "sub_depth_pred"}),
                    pooling_type='mean', sort_direction=sort_dir)
                db.session.add(lm); db.session.commit()
            keep.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(dict.fromkeys(keep))
        db.session.commit()
        print('lb83 metrics now:', [(lm.target_name) for lm in LeaderboardMetric.query.filter_by(leaderboard_id=LB).all()])


if __name__ == '__main__':
    raise SystemExit(main())
