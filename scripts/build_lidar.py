#!/usr/bin/env python
"""Register the 3D LiDAR-segmentation primitives: the `point_cloud` and
`point_labels` data types (DataTypeDef) + the `point_miou` metric. Idempotent.
The actual board + SemanticKITTI import is build_lidar_lb.py (after the data
download completes). Run:

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_lidar.py
"""
import os
import sys
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

# point_cloud: bytes = N×C float32 (x,y,z,intensity[,r,g,b]). Raw bird's-eye-view
# preview colored by height. Runs ONLY in the sandbox.
PC_VIZ = '''
def visualize(blob, params):
    import numpy as np
    from PIL import Image
    c = int((params or {}).get("channels", 4))
    pts = np.frombuffer(blob, dtype=np.float32)
    pts = pts[: (len(pts)//c)*c].reshape(-1, c)
    rng, px = 50.0, 600
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    u = ((x + rng) / (2*rng) * px).astype(int)
    v = ((y + rng) / (2*rng) * px).astype(int)
    m = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    img = np.full((px, px, 3), 18, np.uint8)
    t = np.clip((z + 3) / 6.0, 0, 1)
    col = (np.stack([t, np.clip(1.5 - np.abs(t-0.5)*3, 0, 1), 1 - t], 1) * 255).astype(np.uint8)
    o = np.argsort(z)
    uu, vv, cc, mm = u[o], v[o], col[o], m[o]
    img[vv[mm], uu[mm]] = cc[mm]
    return Image.fromarray(img[::-1])
'''
PC_DECODE = '''
def decode(blob, params):
    import numpy as np
    c = int((params or {}).get("channels", 4))
    a = np.frombuffer(blob, dtype=np.float32)
    return a[: (len(a)//c)*c].reshape(-1, c)
'''

# point_labels: bytes = N uint16 per-point class ids.
PL_DECODE = '''
def decode(blob, params):
    import numpy as np
    return np.frombuffer(blob, dtype=np.uint16)
'''

# Point-wise mean IoU over classes present in the GT (skip 0=unlabeled). gt/pred
# arrive decoded (numpy arrays) via the point_labels decode hook.
POINT_MIOU = '''
def point_miou(gt, pred):
    """Point-wise mean Intersection-over-Union for 3D semantic segmentation
    (higher is better). Mean over the classes present in the ground truth,
    ignoring class 0 (unlabeled)."""
    import numpy as np
    g = np.asarray(gt).ravel().astype(np.int64)
    p = np.asarray(pred).ravel().astype(np.int64)
    n = min(len(g), len(p))
    if n == 0:
        return 0.0
    g, p = g[:n], p[:n]
    ious = []
    for c in np.unique(g):
        if c == 0:
            continue
        inter = np.sum((g == c) & (p == c))
        union = np.sum((g == c) | (p == c))
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0
'''


def main():
    import app as A
    from app import db, DataTypeDef, GlobalMetric
    with A.app.app_context():
        def reg(name, file_ext, viz, dec, desc):
            dt = DataTypeDef.query.filter_by(name=name).first()
            if dt:
                print(f'datatype {name} exists (id={dt.id})')
                return dt
            dt = DataTypeDef(name=name, description=desc, file_ext=file_ext,
                             viz_mime='image/png', visualize_code=(viz.strip() if viz else None),
                             decode_code=(dec.strip() if dec else None),
                             owner_user_id=2, visibility='public')
            db.session.add(dt); db.session.commit()
            print(f'registered datatype {name} (id={dt.id})')
            return dt

        reg('point_cloud', '.bin', PC_VIZ, PC_DECODE,
            'A 3D point cloud — N×C float32 (x, y, z, intensity, optional r,g,b). '
            'LiDAR scans etc. Previewed as a bird’s-eye-view (height-colored).')
        reg('point_labels', '.label', None, PL_DECODE,
            'Per-point class ids for a point cloud — N uint16. Ground truth / '
            'predictions for 3D semantic segmentation.')

        gm = GlobalMetric.query.filter_by(name='point_miou').first()
        if gm is None:
            gm = GlobalMetric(name='point_miou', python_code=POINT_MIOU.strip(),
                              owner_user_id=2, visibility='public', is_aggregated=False)
            db.session.add(gm); db.session.commit()
            print(f'seeded metric point_miou (id={gm.id})')
        else:
            print(f'metric point_miou exists (id={gm.id})')


if __name__ == '__main__':
    main()
