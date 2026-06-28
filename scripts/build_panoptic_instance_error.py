#!/usr/bin/env python
"""Add instance-focused + temporal-error visualizations to the panoptic boards.

  - instance_seg   (static)   : things coloured per-instance, stuff dimmed grey —
                                isolates the object/instance segmentation. -> lb141 (3D).
  - instance_video (temporal) : the animated version; track-consistent instance
                                colours over the sub-sequence. -> lb143 (4D).
  - error_video    (temporal) : animated green=correct-class / red=wrong-class map
                                over the sub-sequence. -> lb143 (4D).

Reuses ds346's cloud_anim / scan_anim aux fields (no re-submission). Temporal viz
return animated GIFs (run in-process, served image/gif) and are flagged anim_img
in the route (name ends '_video') so they get Space/Enter play/pause.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking python scripts/build_panoptic_instance_error.py
"""
import os, sys, json
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

LB_3D = 141; LB_4D = 143

_HELP = r'''
    import numpy as np, io
    from PIL import Image as PILImage
    SEM = np.array([(40,40,40),(245,150,100),(245,230,100),(150,60,30),(180,30,80),(255,0,0),
     (30,30,255),(200,40,255),(90,30,150),(255,0,255),(255,150,255),(75,0,75),(175,0,75),
     (50,120,255),(0,175,0),(0,60,135),(80,240,150),(150,240,255),(0,0,255),(245,150,100)],np.uint8)
    THINGS=set(range(1,9)); S=6
    def _icol(i):
        i=int(i)*2654435761 & 0xFFFFFFFF
        return np.array([90+(i%150),60+((i//150)%180),80+((i//27000)%160)],np.uint8)
    def _np(x,dt):
        if hasattr(x,'array'): x=x.array
        elif hasattr(x,'blob'): return np.frombuffer(x.blob,dt)
        elif hasattr(x,'value'): x=x.value
        elif hasattr(x,'data'): x=x.data
        if isinstance(x,(bytes,bytearray)): return np.frombuffer(x,dt)
        return np.asarray(x)
    def _pts(c):
        c=_np(c,np.float32); return c.reshape(-1,4) if c.ndim==1 else c
    def _inscols(pan):
        sem=(pan & 0xFFFF); inst=(pan >> 16)
        col=np.full((len(pan),3),50,np.uint8)                     # dim everything (stuff/unlabeled)
        tm=np.isin(sem,list(THINGS)) & (inst>0)
        for u in np.unique(inst[tm]):
            m=(inst==u) & tm
            if m.any(): col[m]=_icol(u)                           # things: distinct per-instance colour
        return col
    def _bev(xyz,col,px=320,rng=50.0):
        u=((xyz[:,0]+rng)/(2*rng)*px).astype(int); v=((xyz[:,1]+rng)/(2*rng)*px).astype(int)
        m=(u>=0)&(u<px)&(v>=0)&(v<px)
        img=np.full((px,px,3),18,np.uint8); o=np.argsort(xyz[:,2])
        img[v[o][m[o]],u[o][m[o]]]=col[o][m[o]]
        return PILImage.fromarray(img[::-1])
    def _gif(frames):
        if not frames: return _bev(np.zeros((1,4),np.float32), np.array([[18,18,18]],np.uint8))
        buf=io.BytesIO()
        frames[0].save(buf,format='GIF',append_images=frames[1:],save_all=True,duration=110,loop=0,optimize=True)
        return buf.getvalue()
'''

INSTANCE_SEG = "def instance_seg(cloud, panoptic):" + _HELP + r'''
    pts=_pts(cloud); pan=_np(panoptic,np.uint32).ravel()
    n=min(len(pts),len(pan)); return _bev(pts[:n], _inscols(pan[:n]))
'''
INSTANCE_VIDEO = "def instance_video(cloud, scan_cloud, labels, scan_labels):" + _HELP + r'''
    cl=_pts(cloud); sca=_np(scan_cloud,np.uint16).ravel()
    lab=_np(labels,np.uint32).ravel(); scl=_np(scan_labels,np.uint16).ravel()
    nscans=int(sca.max())+1 if len(sca) else 0; frames=[]
    for s in range(0,nscans,2):
        cpts=cl[sca==s]; lb=lab[scl==s][::S]; n=min(len(cpts),len(lb))
        if n: frames.append(_bev(cpts[:n], _inscols(lb[:n])))
    return _gif(frames)
'''
ERROR_VIDEO = "def error_video(cloud, scan_cloud, gt, pred, scan_labels):" + _HELP + r'''
    cl=_pts(cloud); sca=_np(scan_cloud,np.uint16).ravel()
    g=_np(gt,np.uint32).ravel(); p=_np(pred,np.uint32).ravel(); scl=_np(scan_labels,np.uint16).ravel()
    nscans=int(sca.max())+1 if len(sca) else 0; frames=[]
    for s in range(0,nscans,2):
        cpts=cl[sca==s]; gs=(g[scl==s][::S] & 0xFFFF); ps=(p[scl==s][::S] & 0xFFFF)
        n=min(len(cpts),len(gs),len(ps))
        if not n: continue
        ok=gs[:n]==ps[:n]
        col=np.where(ok[:,None],np.array([0,200,0]),np.array([220,40,40])).astype(np.uint8)
        o=np.argsort(ok.astype(int))                              # draw the red errors on top
        frames.append(_bev(cpts[:n][o], col[o]))
    return _gif(frames)
'''


def main():
    import app as A
    from app import db, GlobalVisualization, LeaderboardVisualization, Leaderboard
    with A.app.app_context():
        def gv(name, code, kinds, desc):
            v = GlobalVisualization.query.filter_by(name=name).first()
            if v is None:
                v = GlobalVisualization(name=name, description=desc, python_code=code, is_aggregated=0,
                    accepts_aggregated_inputs=0, input_kinds=json.dumps(kinds), owner_user_id=None, visibility='public')
                db.session.add(v)
            else:
                v.python_code = code; v.description = desc
            db.session.commit(); return v

        iseg = gv('instance_seg', INSTANCE_SEG, ['point_cloud', 'point_panoptic'],
                  'Instance segmentation BEV — “thing” objects coloured per-instance, stuff dimmed, to isolate the object segmentation.')
        ivid = gv('instance_video', INSTANCE_VIDEO, ['point_cloud', 'point_labels', 'point_panoptic', 'point_labels'],
                  'Temporal instance VIDEO — objects coloured per-instance across the sub-sequence; stable colours over time = good tracking.')
        evid = gv('error_video', ERROR_VIDEO, ['point_cloud', 'point_labels', 'point_panoptic', 'point_panoptic', 'point_labels'],
                  'Temporal error VIDEO — green = correct class, red = wrong, across the sub-sequence.')

        def bind(lb_id, v, target, mapping, hide=()):
            lb = Leaderboard.query.get(lb_id)
            lv = LeaderboardVisualization.query.filter_by(leaderboard_id=lb_id, global_visualization_id=v.id, target_name=target).first()
            if lv:
                lv.arg_mappings = json.dumps(mapping)
            else:
                db.session.add(LeaderboardVisualization(leaderboard_id=lb_id, global_visualization_id=v.id, arg_mappings=json.dumps(mapping), target_name=target))
            if hide:
                cur = set(c.strip() for c in (lb.hidden_comparison_display_columns or '').split(',') if c.strip())
                cur.update(hide); lb.hidden_comparison_display_columns = ','.join(sorted(cur))
            db.session.commit()

        # 3D board (lb141): static instance seg (GT + pred)
        bind(LB_3D, iseg, 'GT instances', {'cloud': 'gt_cloud', 'panoptic': 'gt_panoptic'})
        bind(LB_3D, iseg, 'Predicted instances', {'cloud': 'gt_cloud', 'panoptic': 'sub_panoptic_pred'})
        # 4D board (lb143): temporal instance video (GT + pred) + temporal error video
        anim = {'cloud': 'gt_cloud_anim', 'scan_cloud': 'gt_scan_anim'}
        bind(LB_4D, ivid, 'GT instances — temporal video', {**anim, 'labels': 'gt_panoptic', 'scan_labels': 'gt_scan'})
        bind(LB_4D, ivid, 'Predicted instances — temporal video', {**anim, 'labels': 'sub_panoptic_pred', 'scan_labels': 'gt_scan'})
        bind(LB_4D, evid, 'Errors — temporal video', {**anim, 'gt': 'gt_panoptic', 'pred': 'sub_panoptic_pred', 'scan_labels': 'gt_scan'})
        print(f'INSTANCE_ERROR_VIZ_BOUND instance_seg={iseg.id} instance_video={ivid.id} error_video={evid.id}')


if __name__ == '__main__':
    main()
