#!/usr/bin/env python
"""Class-colored bird's-eye-view visualizations for the 3D LiDAR segmentation
board: `lidar_seg` (cloud + labels -> per-class BEV) for the GT and each
prediction, and `lidar_error` (cloud + gt + pred -> green=correct/red=wrong).
Binds to the board + hides the raw cloud/labels columns. Idempotent.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/add_lidar_viz.py 138
"""
import os
import sys
import json

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

# Shared helpers (inlined into each viz so the sandbox run is self-contained).
_HELPERS = r'''
    import numpy as np
    from PIL import Image as PILImage
    PALETTE = np.array([
     (40,40,40),(245,150,100),(245,230,100),(150,60,30),(180,30,80),(255,0,0),
     (30,30,255),(200,40,255),(90,30,150),(255,0,255),(255,150,255),(75,0,75),
     (175,0,75),(50,120,255),(0,175,0),(0,60,135),(80,240,150),(150,240,255),
     (0,0,255),(245,150,100)],np.uint8)
    def _np(x, dt):
        if hasattr(x,'array'): x=x.array
        elif hasattr(x,'blob'): return np.frombuffer(x.blob, dt)   # RegisteredBlob (in-process)
        elif hasattr(x,'value'): x=x.value
        elif hasattr(x,'data'): x=x.data
        if isinstance(x,(bytes,bytearray)): return np.frombuffer(x, dt)
        return np.asarray(x)
    def _pts(c):
        c=_np(c, np.float32)
        return c.reshape(-1,4) if c.ndim==1 else c
    def _bev(pts, colors):
        rng,px=50.0,600
        u=((pts[:,0]+rng)/(2*rng)*px).astype(int); v=((pts[:,1]+rng)/(2*rng)*px).astype(int)
        m=(u>=0)&(u<px)&(v>=0)&(v<px)
        img=np.full((px,px,3),18,np.uint8)
        o=np.argsort(pts[:,2]); uu,vv,cc,mm=u[o],v[o],colors[o],m[o]
        img[vv[mm],uu[mm]]=cc[mm]
        return PILImage.fromarray(img[::-1])
'''

SEG_CODE = "def lidar_seg(cloud, labels):" + _HELPERS + r'''
    pts=_pts(cloud); lab=_np(labels, np.uint16).astype(np.int64).ravel()
    n=min(len(pts),len(lab)); pts,lab=pts[:n],lab[:n]
    col=PALETTE[np.clip(lab,0,len(PALETTE)-1)]
    return _bev(pts, col)
'''

ERR_CODE = "def lidar_error(cloud, gt, pred):" + _HELPERS + r'''
    pts=_pts(cloud); g=_np(gt, np.uint16).astype(np.int64).ravel(); p=_np(pred, np.uint16).astype(np.int64).ravel()
    n=min(len(pts),len(g),len(p)); pts,g,p=pts[:n],g[:n],p[:n]
    ok=g==p
    col=np.where(ok[:,None],np.array([0,200,0]),np.array([220,40,40])).astype(np.uint8)
    o=np.argsort(ok.astype(int))                  # draw errors on top
    return _bev(pts[o], col[o])
'''

HIDE = ['cloud', 'labels', 'labels_pred']


def main():
    lb_ids = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not lb_ids:
        print('usage: add_lidar_viz.py <lb_id> [...]'); return 2
    import app as A
    from app import db, GlobalVisualization, LeaderboardVisualization, Leaderboard
    with A.app.app_context():
        def gv(name, code, kinds, desc):
            v = GlobalVisualization.query.filter_by(name=name).first()
            if v is None:
                v = GlobalVisualization(name=name, description=desc, python_code=code,
                                        is_aggregated=0, accepts_aggregated_inputs=0,
                                        input_kinds=json.dumps(kinds), owner_user_id=None,
                                        visibility='public')
                db.session.add(v); db.session.commit(); print(f'created {name} viz id={v.id}')
            else:
                v.python_code=code; v.input_kinds=json.dumps(kinds); db.session.commit()
                print(f'updated {name} viz id={v.id}')
            return v

        seg = gv('lidar_seg', SEG_CODE, ['point_cloud', 'point_labels'],
                 'Class-colored bird’s-eye-view of a LiDAR semantic segmentation (points colored by class).')
        err = gv('lidar_error', ERR_CODE, ['point_cloud', 'point_labels', 'point_labels'],
                 'BEV error map for LiDAR segmentation — green = correct, red = wrong vs ground truth.')

        for lb_id in lb_ids:
            lb = Leaderboard.query.get(lb_id)
            if lb is None:
                print(f'  lb{lb_id}: NOT FOUND'); continue
            binds = [
                (seg, 'GT segmentation',        {'cloud': 'gt_cloud', 'labels': 'gt_labels'}),
                (seg, 'Predicted segmentation', {'cloud': 'gt_cloud', 'labels': 'sub_labels_pred'}),
                (err, 'Errors vs GT',           {'cloud': 'gt_cloud', 'gt': 'gt_labels', 'pred': 'sub_labels_pred'}),
            ]
            for v, target, mapping in binds:
                lv = LeaderboardVisualization.query.filter_by(
                    leaderboard_id=lb_id, global_visualization_id=v.id, target_name=target).first()
                if lv:
                    lv.arg_mappings = json.dumps(mapping); db.session.commit()
                    print(f'  lb{lb_id}: updated "{target}"')
                else:
                    lv = LeaderboardVisualization(leaderboard_id=lb_id, global_visualization_id=v.id,
                                                  arg_mappings=json.dumps(mapping), target_name=target)
                    db.session.add(lv); db.session.commit()
                    print(f'  lb{lb_id}: bound "{target}" (lv={lv.id})')
            cur = set(c.strip() for c in (lb.hidden_comparison_display_columns or '').split(',') if c.strip())
            cur.update(HIDE)
            lb.hidden_comparison_display_columns = ','.join(sorted(cur))
            db.session.commit()
        print('LIDAR_VIZ_DONE')


if __name__ == '__main__':
    raise SystemExit(main())
