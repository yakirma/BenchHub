#!/usr/bin/env python
"""Build the SemanticKITTI 3D Panoptic + 4D Panoptic segmentation boards.

Registers a `point_panoptic` dtype (uint32 = semantic in low 16 bits, instance/
track id in high 16 — native SemanticKITTI format), seeds two AGGREGATED metrics
(`point_pq` = Panoptic Quality, `lstq` = LiDAR Seg & Tracking Quality / PQ4D),
imports a consecutive seq08 slice (cloud + panoptic GT) as one typed dataset, and
creates two boards on it: 3D Panoptic (->point_pq) and 4D Panoptic (->lstq) plus
the instance-coloured BEV visualizations. Idempotent.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking python scripts/build_panoptic.py [n_scans]
"""
import os, sys, json, glob, shutil, tempfile
from pathlib import Path
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'
import numpy as np

KITTI = os.path.expanduser('~/.dtofbenchmarking/lidar_kitti')
SEQ = '08'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250          # consecutive scans (needed for 4D tracking)
DS_NAME = 'SemanticKITTI-seq08-panoptic'
CAT_3D = 'Vision/3D Panoptic Segmentation'
CAT_4D = 'Vision/4D Panoptic Segmentation'
LEARNING_MAP = {0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,
                49:12,50:13,51:14,52:0,60:9,70:15,71:16,72:17,80:18,81:19,99:0,252:1,253:7,
                254:6,255:8,256:5,257:5,258:4,259:5}
_LUT = np.zeros(260, np.uint32)
for k, v in LEARNING_MAP.items(): _LUT[k] = v

# ---- dtype: point_panoptic (decode only; the spatial view is a GlobalVisualization) ----
PP_DECODE = '''
def decode(blob, params):
    import numpy as np
    return np.frombuffer(blob, np.uint32)
'''

# ---- metrics (validated byte-identical to the official PanopticEval / Panoptic4DEval) ----
POINT_PQ = open(os.path.expanduser('/tmp/claude-1000/-home-ymatri-Git-BenchHub/d6b433c8-6758-47ba-a6c5-0ecccf71c8b3/scratchpad/pq_metric.py')).read()
LSTQ = open(os.path.expanduser('/tmp/claude-1000/-home-ymatri-Git-BenchHub/d6b433c8-6758-47ba-a6c5-0ecccf71c8b3/scratchpad/lstq_metric.py')).read()

# ---- visualizations: instance-coloured BEV ----
_VHELP = r'''
    import numpy as np
    from PIL import Image as PILImage
    SEM = np.array([(0,0,0),(245,150,100),(245,230,100),(150,60,30),(180,30,80),(255,0,0),
     (30,30,255),(200,40,255),(90,30,150),(255,0,255),(255,150,255),(75,0,75),(175,0,75),
     (50,120,255),(0,175,0),(0,60,135),(80,240,150),(150,240,255),(0,0,255),(245,150,100)],np.uint8)
    THINGS=set(range(1,9))
    def _icol(i):
        i=int(i)*2654435761 & 0xFFFFFFFF
        return np.array([80+(i&0x7F),60+((i>>7)&0xBF),70+((i>>14)&0x9F)],np.uint8)
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
    def _bev(pts,colors):
        rng,px=50.0,600
        u=((pts[:,0]+rng)/(2*rng)*px).astype(int); v=((pts[:,1]+rng)/(2*rng)*px).astype(int)
        m=(u>=0)&(u<px)&(v>=0)&(v<px)
        img=np.full((px,px,3),18,np.uint8); o=np.argsort(pts[:,2])
        img[v[o][m[o]],u[o][m[o]]]=colors[o][m[o]]
        return PILImage.fromarray(img[::-1])
'''
PAN_SEG = "def panoptic_seg(cloud, panoptic):" + _VHELP + r'''
    pts=_pts(cloud); pan=_np(panoptic,np.uint32).ravel()
    n=min(len(pts),len(pan)); return _bev(pts[:n], _pancols(pan[:n]))
'''
PAN_ERR = "def panoptic_error(cloud, gt, pred):" + _VHELP + r'''
    pts=_pts(cloud); g=_np(gt,np.uint32).ravel(); p=_np(pred,np.uint32).ravel()
    n=min(len(pts),len(g),len(p)); pts,g,p=pts[:n],g[:n],p[:n]
    ok=(g & 0xFFFF)==(p & 0xFFFF)
    col=np.where(ok[:,None],np.array([0,200,0]),np.array([220,40,40])).astype(np.uint8)
    o=np.argsort(ok.astype(int)); return _bev(pts[o],col[o])
'''


def build_staging(staging):
    vel = sorted(glob.glob(f'{KITTI}/dataset/sequences/{SEQ}/velodyne/*.bin'))[:N]
    lab = sorted(glob.glob(f'{KITTI}/data_odometry_labels/dataset/sequences/{SEQ}/labels/*.label'))[:N]
    assert len(vel) == len(lab) == N, f'{len(vel)}/{len(lab)} != {N}'
    (staging / 'cloud').mkdir(parents=True, exist_ok=True)
    (staging / 'panoptic').mkdir(parents=True, exist_ok=True)
    out = []
    for vp, lp in zip(vel, lab):
        name = f'{SEQ}_' + os.path.basename(vp)[:-4]
        np.fromfile(vp, np.float32).reshape(-1, 4).tofile(staging / 'cloud' / f'{name}.bin')
        raw = np.fromfile(lp, np.uint32)
        pan = (_LUT[raw & 0xFFFF] | (raw & 0xFFFF0000)).astype(np.uint32)   # learning sem | instance
        pan.tofile(staging / 'panoptic' / f'{name}.label')
        out.append(name)
    manifest = {'name': DS_NAME, 'version': '1.0', 'fields': [
        {'name': 'cloud', 'kind': 'point_cloud', 'role': 'input', 'params': {'channels': 4}},
        {'name': 'panoptic', 'kind': 'point_panoptic', 'role': 'gt', 'params': {}}], 'samples': out}
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def main():
    import app as A
    from app import (db, DataTypeDef, GlobalMetric, GlobalVisualization, Dataset, Sample,
                     CustomField, DatasetField, Leaderboard, LeaderboardMetric, LeaderboardVisualization)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        # 1. dtype
        if not DataTypeDef.query.filter_by(name='point_panoptic').first():
            db.session.add(DataTypeDef(name='point_panoptic', file_ext='.label',
                description='Per-point panoptic labels: uint32, semantic class in low 16 bits, instance/track id in high 16 (SemanticKITTI format).',
                visualize_code=None, decode_code=PP_DECODE.strip(), owner_user_id=2, visibility='public'))
            db.session.commit()
        # 2. metrics
        for nm, code in [('point_pq', POINT_PQ), ('lstq', LSTQ)]:
            if not GlobalMetric.query.filter_by(name=nm).first():
                db.session.add(GlobalMetric(name=nm, python_code=code.strip(), owner_user_id=2,
                                            visibility='public', is_aggregated=True))
        # 3. visualizations
        def gv(name, code, kinds, desc):
            v = GlobalVisualization.query.filter_by(name=name).first()
            if not v:
                v = GlobalVisualization(name=name, description=desc, python_code=code, is_aggregated=0,
                    accepts_aggregated_inputs=0, input_kinds=json.dumps(kinds), owner_user_id=None, visibility='public')
                db.session.add(v)
            else:
                v.python_code = code
            db.session.commit(); return v
        seg = gv('panoptic_seg', PAN_SEG, ['point_cloud', 'point_panoptic'],
                 'Instance-coloured bird’s-eye-view of a LiDAR panoptic segmentation (things coloured per-instance, stuff by class).')
        err = gv('panoptic_error', PAN_ERR, ['point_cloud', 'point_panoptic', 'point_panoptic'],
                 'BEV semantic error map for panoptic segmentation — green = correct class, red = wrong.')
        db.session.commit()
        # 4. dataset
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='pano_'))
            try:
                n = len(build_staging(staging))
                ds_id, _ = import_typed_dataset(staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField, upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False,
                    extra_kinds={'point_cloud': '.bin', 'point_panoptic': '.label'})
                db.session.commit(); ds = Dataset.query.get(ds_id)
                ds.category = CAT_3D; ds.source_url = 'https://semantic-kitti.org/'
                ds.card_description = (f'SemanticKITTI (Behley et al.) — {n} consecutive scans of validation '
                    'sequence 08 with per-point panoptic labels (19 classes + instance/track ids). Backs both '
                    'the 3D Panoptic (PQ) and 4D Panoptic (LSTQ) boards.')
                db.session.commit(); print(f'imported dataset id={ds_id} ({n} scans)')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            print(f'dataset exists id={ds.id}')

        # 5. boards
        def board(lb_name, cat, metric_name, target):
            lb = Leaderboard.query.filter_by(name=lb_name).first()
            if lb is None:
                lb = Leaderboard(name=lb_name, owner_user_id=2, visibility='public', category=cat,
                    required_pred_fields_json=json.dumps([{'name': 'panoptic_pred', 'kind': 'point_panoptic', 'params': {}, 'role': 'pred'}]),
                    field_roles_json=json.dumps({'cloud': 'input', 'panoptic': 'gt'}), summary_metrics='')
                lb.datasets.append(ds); db.session.add(lb); db.session.commit()
            gm = GlobalMetric.query.filter_by(name=metric_name).first()
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(leaderboard_id=lb.id, global_metric_id=gm.id, target_name=target,
                    arg_mappings=json.dumps({'gt': 'gt_panoptic', 'pred': 'sub_panoptic_pred'}),
                    pooling_type='mean', sort_direction='higher_is_better')
                db.session.add(lm); db.session.commit()
            lb.summary_metrics = f'lm_{lm.id}'
            # bind viz
            binds = [(seg, 'GT panoptic', {'cloud': 'gt_cloud', 'panoptic': 'gt_panoptic'}),
                     (seg, 'Predicted panoptic', {'cloud': 'gt_cloud', 'panoptic': 'sub_panoptic_pred'}),
                     (err, 'Errors vs GT', {'cloud': 'gt_cloud', 'gt': 'gt_panoptic', 'pred': 'sub_panoptic_pred'})]
            for v, tname, mapping in binds:
                if not LeaderboardVisualization.query.filter_by(leaderboard_id=lb.id, global_visualization_id=v.id, target_name=tname).first():
                    db.session.add(LeaderboardVisualization(leaderboard_id=lb.id, global_visualization_id=v.id,
                        arg_mappings=json.dumps(mapping), target_name=tname))
            cur = set(c.strip() for c in (lb.hidden_comparison_display_columns or '').split(',') if c.strip())
            cur.update(['cloud', 'panoptic', 'panoptic_pred']); lb.hidden_comparison_display_columns = ','.join(sorted(cur))
            db.session.commit()
            print(f'  board lb_id={lb.id} {lb_name} -> {metric_name} (lm={lm.id})')
            return lb

        board(f'{DS_NAME}_benchmark', CAT_3D, 'point_pq', 'PQ')
        board(f'{DS_NAME.replace("panoptic","4dpanoptic")}_benchmark', CAT_4D, 'lstq', 'LSTQ')
        print('PANOPTIC_BUILD_DONE')


if __name__ == '__main__':
    main()
