#!/usr/bin/env python
"""Render optical flow as a single Middlebury colour-wheel image instead of two
raw `depth` columns (flow_u, flow_v) per side.

Before: each flow board showed flow_u + flow_v as two grey "depth" columns for
the GT and for every prediction — so a prediction's flow appeared as two
columns ("seen twice") and was mislabelled "depth". After: one shared
`gtviz_` colour-wheel column for the GT flow and one `viz_` colour-wheel column
per submission, with the raw u/v depth columns hidden.

Binds to all five flow boards and sets hidden_comparison_display_columns so the
raw flow_u/flow_v/flow_u_pred/flow_v_pred columns no longer render. Idempotent.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/add_flow_viz.py 110 111 112 113 114
"""
import os
import sys
import json

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

VIZ_CODE = r'''def flow_color(flow_u, flow_v):
    """Middlebury colour-wheel visualization of a 2D optical-flow field. Hue
    encodes direction, saturation/brightness encodes magnitude (normalized per
    image to the 99th-percentile flow magnitude). flow_u / flow_v are HxW float
    depth arrays (horizontal / vertical displacement); NaN/invalid -> black."""
    import numpy as np
    from PIL import Image as PILImage

    def to_arr(x):
        if hasattr(x, 'array'):
            return np.asarray(x.array, dtype=np.float32)
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'value'):
            x = x.value
        return np.asarray(x, dtype=np.float32)

    u = to_arr(flow_u)
    v = to_arr(flow_v)
    u = np.squeeze(u)
    v = np.squeeze(v)
    if u.ndim != 2 or v.ndim != 2:
        return PILImage.new('RGB', (64, 64), (0, 0, 0))
    if u.shape != v.shape:
        h = min(u.shape[0], v.shape[0]); w = min(u.shape[1], v.shape[1])
        u = u[:h, :w]; v = v[:h, :w]

    mask = np.isfinite(u) & np.isfinite(v)
    u = np.where(mask, u, 0.0)
    v = np.where(mask, v, 0.0)

    # colour wheel (Baker et al. / standard flow_to_image)
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    cw = np.zeros((ncols, 3), np.float32)
    c = 0
    cw[0:RY, 0] = 255; cw[0:RY, 1] = np.floor(255 * np.arange(RY) / RY); c += RY
    cw[c:c+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG); cw[c:c+YG, 1] = 255; c += YG
    cw[c:c+GC, 1] = 255; cw[c:c+GC, 2] = np.floor(255 * np.arange(GC) / GC); c += GC
    cw[c:c+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB); cw[c:c+CB, 2] = 255; c += CB
    cw[c:c+BM, 2] = 255; cw[c:c+BM, 0] = np.floor(255 * np.arange(BM) / BM); c += BM
    cw[c:c+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR); cw[c:c+MR, 0] = 255

    rad = np.sqrt(u * u + v * v)
    valid_rad = rad[mask]
    maxr = np.percentile(valid_rad, 99) if valid_rad.size else 1.0
    maxr = max(float(maxr), 1e-5)
    un = u / maxr
    vn = v / maxr
    rn = np.sqrt(un * un + vn * vn)

    a = np.arctan2(-vn, -un) / np.pi          # [-1, 1]
    fk = (a + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    img = np.zeros(u.shape + (3,), np.uint8)
    for i in range(3):
        ch = cw[:, i]
        col0 = ch[k0] / 255.0
        col1 = ch[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        small = rn <= 1
        col[small] = 1 - rn[small] * (1 - col[small])   # increase saturation toward centre
        col[~small] = col[~small] * 0.75                # out-of-range -> darken
        img[:, :, i] = np.floor(255 * np.clip(col, 0, 1))
    img[~mask] = 0
    return PILImage.fromarray(img)
'''

GT_TARGET = 'GT flow'
PRED_TARGET = 'Predicted flow'
HIDE_COLS = ['flow_u', 'flow_v', 'flow_u_pred', 'flow_v_pred']


def main():
    lb_ids = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not lb_ids:
        print('usage: add_flow_viz.py <lb_id> [...]'); return 2
    import app as A
    from app import db, GlobalVisualization, LeaderboardVisualization, Leaderboard
    with A.app.app_context():
        gv = GlobalVisualization.query.filter_by(name='flow_color').first()
        if gv is None:
            gv = GlobalVisualization(
                name='flow_color',
                description='Middlebury colour-wheel visualization of a 2D optical-flow field (hue=direction, brightness=magnitude).',
                python_code=VIZ_CODE, is_aggregated=0, accepts_aggregated_inputs=0,
                input_kinds=json.dumps(['depth', 'depth']),
                owner_user_id=None, visibility='public')
            db.session.add(gv); db.session.commit()
            print(f'created flow_color viz id={gv.id}')
        else:
            gv.python_code = VIZ_CODE
            gv.input_kinds = json.dumps(['depth', 'depth'])
            db.session.commit()
            print(f'updated flow_color viz id={gv.id}')

        for lb_id in lb_ids:
            lb = Leaderboard.query.get(lb_id)
            if lb is None:
                print(f'  lb{lb_id}: NOT FOUND — skip'); continue
            for target, mapping in [
                (GT_TARGET, {'flow_u': 'gt_flow_u', 'flow_v': 'gt_flow_v'}),
                (PRED_TARGET, {'flow_u': 'sub_flow_u_pred', 'flow_v': 'sub_flow_v_pred'}),
            ]:
                lv = LeaderboardVisualization.query.filter_by(
                    leaderboard_id=lb_id, global_visualization_id=gv.id, target_name=target).first()
                if lv:
                    lv.arg_mappings = json.dumps(mapping); db.session.commit()
                    print(f'  lb{lb_id}: updated "{target}"')
                else:
                    lv = LeaderboardVisualization(
                        leaderboard_id=lb_id, global_visualization_id=gv.id,
                        arg_mappings=json.dumps(mapping), target_name=target)
                    db.session.add(lv); db.session.commit()
                    print(f'  lb{lb_id}: bound "{target}" (lv={lv.id})')

            # Hide the raw u/v depth columns (merge with anything already hidden).
            cur = set(c.strip() for c in (lb.hidden_comparison_display_columns or '').split(',') if c.strip())
            cur.update(HIDE_COLS)
            lb.hidden_comparison_display_columns = ','.join(sorted(cur))
            db.session.commit()
            print(f'  lb{lb_id}: hidden cols -> {lb.hidden_comparison_display_columns}')
        print('FLOW_VIZ_DONE')


if __name__ == '__main__':
    raise SystemExit(main())
