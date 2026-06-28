#!/usr/bin/env python
"""Declare input_kinds on the 3 LiDAR metrics so they participate in the typed
contract (the /metrics 'Accepts:' row + future kind validation) instead of
silently relying on the legacy primitive path. Arg order matches each metric's
signature. Delivery is unchanged — registered-kind args already decode to arrays
via resolve_registered_kwargs — so scores are identical (verified separately).

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking ~/benchhub/.venv/bin/python scripts/declare_lidar_metric_kinds.py
"""
import os, sys, json
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

KINDS = {
    'point_miou': ['point_labels', 'point_labels'],                 # gt, pred
    'point_pq':   ['point_panoptic', 'point_panoptic'],             # gt, pred (lists)
    'lstq_seq':   ['point_panoptic', 'point_panoptic', 'point_labels'],  # gt, pred, scan
}

def main():
    import app as A
    from app import db, GlobalMetric
    with A.app.app_context():
        for name, kinds in KINDS.items():
            gm = GlobalMetric.query.filter_by(name=name).first()
            if gm is None:
                print(f'  ! {name}: not found'); continue
            before = gm.input_kinds
            gm.input_kinds = json.dumps(kinds)
            print(f'  {name}: {before!r} -> {gm.input_kinds}')
        db.session.commit()
        print('DONE')

if __name__ == '__main__':
    main()
