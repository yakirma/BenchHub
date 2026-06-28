#!/usr/bin/env python
"""Add an animated-GIF temporal visualization to the per-sequence 4D board
(lb143). Each sample is a sub-sequence; this renders the scans as a VIDEO with
track-consistent instance colours, so you can SEE the tracking (stable colours
when tracked, strobing when not).

Adds two aux fields to ds346 — `cloud_anim` (every-6th point of each scan, for a
uniform BEV) + `scan_anim` (its per-point scan index) — then registers the
`panoptic_video` viz and binds GT + predicted versions. The viz runs in-process
(trusted + oversized) and returns GIF bytes; execute_visualization serves
image/gif. No re-submission needed (the pred video strides the full pred).

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking python scripts/build_panoptic_video.py
"""
import os, sys, json, glob
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'
import numpy as np

KITTI = os.path.expanduser('~/.dtofbenchmarking/lidar_kitti')
VEL = f'{KITTI}/dataset/sequences/08/velodyne'
DS_ID = 346; LB_ID = 143; L = 100; STRIDE = 6

_VHELP = r'''
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
    def _pancols(pan):
        sem=(pan & 0xFFFF); inst=(pan >> 16)
        col=SEM[np.clip(sem,0,len(SEM)-1)].copy()
        for u in np.unique(inst[inst>0]):
            m=(inst==u) & np.isin(sem,list(THINGS))
            if m.any(): col[m]=_icol(u)
        return col
    def _bev(xyz,col,px=320,rng=50.0):
        u=((xyz[:,0]+rng)/(2*rng)*px).astype(int); v=((xyz[:,1]+rng)/(2*rng)*px).astype(int)
        m=(u>=0)&(u<px)&(v>=0)&(v<px)
        img=np.full((px,px,3),18,np.uint8); o=np.argsort(xyz[:,2])
        img[v[o][m[o]],u[o][m[o]]]=col[o][m[o]]
        return PILImage.fromarray(img[::-1])
'''
PANO_VIDEO = "def panoptic_video(cloud, scan_cloud, labels, scan_labels):" + _VHELP + r'''
    cl=_pts(cloud); sca=_np(scan_cloud,np.uint16).ravel()
    lab=_np(labels,np.uint32).ravel(); scl=_np(scan_labels,np.uint16).ravel()
    nscans=int(sca.max())+1 if len(sca) else 0
    frames=[]
    for s in range(0, nscans, 2):                 # every 2nd scan -> ~50 frames
        cpts=cl[sca==s]
        lb=lab[scl==s][::S]                        # stride full labels by 6 -> aligns with cloud (velodyne[::6])
        n=min(len(cpts),len(lb))
        if n==0: continue
        frames.append(_bev(cpts[:n], _pancols(lb[:n])))
    if not frames:
        return _bev(np.zeros((1,4),np.float32), np.array([[18,18,18]],np.uint8))
    buf=io.BytesIO()
    frames[0].save(buf, format='GIF', append_images=frames[1:], save_all=True, duration=110, loop=0, optimize=True)
    return buf.getvalue()
'''


def main():
    import app as A
    from app import db, Dataset, Sample, CustomField, DatasetField, GlobalVisualization, LeaderboardVisualization, Leaderboard
    UP = A.app.config['UPLOAD_FOLDER']
    with A.app.app_context():
        ds = Dataset.query.get(DS_ID)
        # 1. aux fields (cloud_anim + scan_anim) — built per sub-sequence
        existing = {cf.name for cf in CustomField.query.filter(
            CustomField.sample_id.in_([s.id for s in ds.samples]), CustomField.name == 'cloud_anim').all()}
        for fld, kind, ext in [('cloud_anim', 'point_cloud', '.bin'), ('scan_anim', 'point_labels', '.label')]:
            if not DatasetField.query.filter_by(dataset_id=DS_ID, name=fld).first():
                db.session.add(DatasetField(dataset_id=DS_ID, name=fld, kind=kind, role='input',
                                            params=json.dumps({'channels': 4}) if kind == 'point_cloud' else None))
            os.makedirs(os.path.join(UP, 'datasets', str(DS_ID), fld), exist_ok=True)
        db.session.commit()
        if not existing:
            for s in ds.samples:
                k = int(s.name.split('seq')[-1])
                frames = range(k * L, k * L + L)
                clouds, scans = [], []
                for j, fi in enumerate(frames):
                    v = np.fromfile(f'{VEL}/{fi:06d}.bin', np.float32).reshape(-1, 4)[::STRIDE]
                    clouds.append(v.astype(np.float32)); scans.append(np.full(len(v), j, np.uint16))
                cpath = f'datasets/{DS_ID}/cloud_anim/{s.name}.bin'
                spath = f'datasets/{DS_ID}/scan_anim/{s.name}.label'
                np.concatenate(clouds).tofile(os.path.join(UP, cpath))
                np.concatenate(scans).tofile(os.path.join(UP, spath))
                db.session.add(CustomField(sample_id=s.id, name='cloud_anim', data_type='point_cloud', value_text=cpath))
                db.session.add(CustomField(sample_id=s.id, name='scan_anim', data_type='point_labels', value_text=spath))
            db.session.commit(); print(f'added cloud_anim + scan_anim to {len(ds.samples)} samples')
        else:
            print('aux fields already present')

        # 2. viz + bindings
        gv = GlobalVisualization.query.filter_by(name='panoptic_video').first()
        if gv is None:
            gv = GlobalVisualization(name='panoptic_video', python_code=PANO_VIDEO, is_aggregated=0,
                accepts_aggregated_inputs=0, input_kinds=json.dumps(['point_cloud', 'point_labels', 'point_panoptic', 'point_labels']),
                description='Temporal VIDEO (animated GIF) of a LiDAR panoptic sub-sequence — each scan as a track-coloured BEV; consistent colours across frames = good tracking.',
                owner_user_id=None, visibility='public')
            db.session.add(gv)
        else:
            gv.python_code = PANO_VIDEO
        db.session.commit()
        lb = Leaderboard.query.get(LB_ID)
        binds = [('GT — temporal video', {'cloud': 'gt_cloud_anim', 'scan_cloud': 'gt_scan_anim', 'labels': 'gt_panoptic', 'scan_labels': 'gt_scan'}),
                 ('Predicted — temporal video', {'cloud': 'gt_cloud_anim', 'scan_cloud': 'gt_scan_anim', 'labels': 'sub_panoptic_pred', 'scan_labels': 'gt_scan'})]
        for tname, mapping in binds:
            lv = LeaderboardVisualization.query.filter_by(leaderboard_id=LB_ID, global_visualization_id=gv.id, target_name=tname).first()
            if lv:
                lv.arg_mappings = json.dumps(mapping)
            else:
                db.session.add(LeaderboardVisualization(leaderboard_id=LB_ID, global_visualization_id=gv.id, arg_mappings=json.dumps(mapping), target_name=tname))
        cur = set(c.strip() for c in (lb.hidden_comparison_display_columns or '').split(',') if c.strip())
        cur.update(['cloud_anim', 'scan_anim']); lb.hidden_comparison_display_columns = ','.join(sorted(cur))
        db.session.commit()
        print(f'PANO_VIDEO_BOUND gv={gv.id} lb={LB_ID}')


if __name__ == '__main__':
    main()
