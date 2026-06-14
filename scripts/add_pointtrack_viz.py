#!/usr/bin/env python
"""Register a point-tracking overlay visualization and bind GT + per-model
predicted trajectories to TAP-Vid boards (mirrors add_detection_viz.py).

Draws each tracked point's trajectory (a colored trail + its last-visible
position) over the video's first frame. The GT-side binding (no sub_ field)
becomes one shared `gtviz_` column; the prediction binding becomes one
`viz_` column per submission.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/add_pointtrack_viz.py 117 118
"""
import os
import sys
import json

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

VIZ_CODE = r'''def point_track_overlay(image, tracks, occluded):
    """Overlay point trajectories (trail + last-visible dot, one colour per
    point) on the video's first frame. image=bh.Image (frame 0); tracks=[N,T,2];
    occluded=[N,T]."""
    import numpy as np
    import colorsys
    from PIL import Image as PILImage, ImageDraw

    def to_arr(x):
        if hasattr(x, 'array'):
            return np.asarray(x.array)
        return np.asarray(x)

    def to_list(x):
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'value'):
            x = x.value
        if isinstance(x, str):
            import json as _j
            try:
                x = _j.loads(x)
            except Exception:
                return []
        return x

    bg = to_arr(image)
    if bg.ndim == 2:
        bg = np.stack([bg] * 3, -1)
    bg = np.ascontiguousarray(bg[..., :3]).astype('uint8')
    im = PILImage.fromarray(bg).convert('RGB')
    dr = ImageDraw.Draw(im)

    tr = np.asarray(to_list(tracks), dtype=float)
    occ_l = to_list(occluded)
    occ = np.asarray(occ_l) if occ_l is not None and len(occ_l) else None
    if tr.ndim != 3 or tr.shape[0] == 0:
        return im
    N, T = tr.shape[0], tr.shape[1]
    for n in range(N):
        col = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(n / max(N, 1), 0.95, 1.0))
        pts = tr[n]
        vis = np.ones(T, bool)
        if occ is not None and occ.ndim == 2 and occ.shape[0] > n:
            vis = ~np.asarray(occ[n]).astype(bool)
        prev = None
        for t in range(T):
            x, y = float(pts[t, 0]), float(pts[t, 1])
            if prev is not None:
                dr.line([prev, (x, y)], fill=col, width=1)
            prev = (x, y)
        vt = np.where(vis)[0]
        if len(vt):
            x, y = float(pts[vt[-1], 0]), float(pts[vt[-1], 1])
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col, outline=(255, 255, 255))
    return im
'''


def main():
    lb_ids = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not lb_ids:
        print('usage: add_pointtrack_viz.py <lb_id> [...]'); return 2
    import app as A
    from app import db, GlobalVisualization, LeaderboardVisualization
    with A.app.app_context():
        gv = GlobalVisualization.query.filter_by(name='point_track_overlay').first()
        if gv is None:
            gv = GlobalVisualization(
                name='point_track_overlay',
                description='Draw point-tracking trajectories (trail + last-visible dot) over the first video frame.',
                python_code=VIZ_CODE, is_aggregated=0, accepts_aggregated_inputs=0,
                input_kinds=json.dumps(['image', 'json', 'json']),
                owner_user_id=None, visibility='public')
            db.session.add(gv); db.session.commit()
            print(f'created point_track_overlay viz id={gv.id}')
        else:
            gv.python_code = VIZ_CODE; db.session.commit()
            print(f'updated point_track_overlay viz id={gv.id}')

        for lb_id in lb_ids:
            for target, mapping in [
                ('GT tracks', {'image': 'gt_video', 'tracks': 'gt_target_points', 'occluded': 'gt_occluded'}),
                ('Predicted tracks', {'image': 'gt_video', 'tracks': 'sub_pred_tracks', 'occluded': 'sub_pred_occluded'}),
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
        print('POINTTRACK_VIZ_DONE')


if __name__ == '__main__':
    raise SystemExit(main())
