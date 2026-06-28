#!/usr/bin/env python
"""Bind the GT temporal-video viz to the 4D dataset ds346 via DatasetVisualization
so the temporal vis appears on the DATASET page (/dataset/346), not only on the
leaderboard (lb143). The dataset page renders DatasetVisualization-bound viz +
per-field dtype thumbnails; the temporal videos were only LeaderboardVisualization
(lb143-scoped), which is why they were absent from the dataset view.

GT-only (no submission on a dataset page) → bind panoptic_video + instance_video
with the GT arg mappings. error_video needs a prediction, so it's LB-only.
Reuses ds346's cloud_anim/scan_anim aux CustomFields. The upgraded
execute_dataset_visualization serves the animated GIF (image/gif).

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking ~/benchhub/.venv/bin/python scripts/bind_dataset_temporal_viz.py
"""
import os, sys, json
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

DS_ID = 346
GT_MAP = {'cloud': 'gt_cloud_anim', 'scan_cloud': 'gt_scan_anim',
          'labels': 'gt_panoptic', 'scan_labels': 'gt_scan'}
BINDS = [  # (viz_name, target_name, display_order)
    ('panoptic_video', 'Panoptic — temporal video (GT)', 0),
    ('instance_video', 'Instances — temporal video (GT)', 1),
]

def main():
    import app as A
    from app import db, Dataset, GlobalVisualization, DatasetVisualization
    with A.app.app_context():
        ds = Dataset.query.get(DS_ID)
        assert ds is not None, f'dataset {DS_ID} missing'
        for vname, target, order in BINDS:
            gv = GlobalVisualization.query.filter_by(name=vname).first()
            if gv is None:
                print(f'  ! {vname}: GlobalVisualization missing'); continue
            dv = DatasetVisualization.query.filter_by(
                dataset_id=DS_ID, global_visualization_id=gv.id, target_name=target).first()
            if dv:
                dv.arg_mappings = json.dumps(GT_MAP); dv.display_order = order
                print(f'  updated dsviz {dv.id}  {vname} -> {target}')
            else:
                dv = DatasetVisualization(dataset_id=DS_ID, global_visualization_id=gv.id,
                                          arg_mappings=json.dumps(GT_MAP), target_name=target,
                                          display_order=order)
                db.session.add(dv); db.session.flush()
                print(f'  created dsviz {dv.id}  {vname} -> {target}')
        db.session.commit()
        # report what the dataset page will now show
        rows = DatasetVisualization.query.filter_by(dataset_id=DS_ID).order_by(
            DatasetVisualization.display_order).all()
        print('DATASET_VIZ_BOUND', [(r.id, r.global_visualization.name, r.target_name) for r in rows])

if __name__ == '__main__':
    main()
