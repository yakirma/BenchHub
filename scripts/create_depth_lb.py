#!/usr/bin/env python
"""Create a depth-estimation leaderboard for a BenchHub dataset (idempotent).

Input = RGB image (role input); GT = a `depth` field. Contract = depth_pred
(depth); metric = aligned_rmse_depth (scale-and-shift-invariant RMSE, lower is
better — handles relative/metric output conventions + preview normalisation).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/create_depth_lb.py <dataset_id>

Prints:  LB_RESULT lb_id=<id> repo=<hf_repo> input_field=<name> n_classes=0
or `LB_SKIP <reason>`.
"""
import os
import sys
import json
import re

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))


def main():
    if len(sys.argv) < 2:
        print('LB_SKIP usage: create_depth_lb.py <dataset_id>'); return 2
    ds_id = int(sys.argv[1])
    import app as A
    from app import db, Leaderboard, LeaderboardMetric, GlobalMetric, Dataset, DatasetField

    with A.app.app_context():
        ds = Dataset.query.get(ds_id)
        if ds is None:
            print(f'LB_SKIP dataset {ds_id} not found'); return 1
        m = re.search(r'huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)', ds.source_url or '')
        if not m:
            print(f'LB_SKIP dataset {ds_id} ({ds.name}) has no HF source repo'); return 1
        repo = m.group(1)

        fields = DatasetField.query.filter_by(dataset_id=ds_id).all()
        img = next((f for f in fields if f.kind == 'image' and (f.role or '') == 'input'), None)
        img = img or next((f for f in fields if f.kind == 'image'), None)
        dep = next((f for f in fields if f.kind == 'depth' and (f.role or 'gt') == 'gt'), None)
        dep = dep or next((f for f in fields if f.kind == 'depth'), None)
        if img is None or dep is None:
            print(f'LB_SKIP dataset {ds_id} ({ds.name}) needs an image input + depth GT field'); return 1

        lb_name = f'{ds.name}_benchmark'
        existing = (Leaderboard.query.filter_by(name=lb_name).first()
                    or next((lb for lb in Leaderboard.query
                             .filter(Leaderboard.name.endswith('_benchmark')).all()
                             if any(d.id == ds_id for d in lb.datasets)), None))
        if existing is not None:
            print(f'LB_RESULT lb_id={existing.id} repo={repo} input_field={img.name} n_classes=0  (existing)')
            return 0

        contract = [{"name": "depth_pred", "kind": "depth", "params": {}, "role": "pred"}]
        lb = Leaderboard(
            name=lb_name, owner_user_id=2, visibility='public',
            category=(ds.category or 'Vision/Depth Estimation'),
            required_pred_fields_json=json.dumps(contract),
            field_roles_json=json.dumps({img.name: "input", dep.name: "gt"}),
            summary_metrics='',
        )
        lb.datasets.append(ds)
        db.session.add(lb)
        db.session.commit()

        lm_ids = []
        _metrics = [('rmse_depth', 'lower_is_better'),
                    ('affine_inv_rmse_depth', 'lower_is_better'),
                    ('scale_inv_rmse_depth', 'lower_is_better'),
                    ('delta1_depth', 'higher_is_better')]
        for mname, sort_dir in _metrics:
            gm = GlobalMetric.query.filter_by(name=mname).first()
            if gm is None:
                continue
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name=mname,
                arg_mappings=json.dumps({"gt": f"gt_{dep.name}", "pred": "sub_depth_pred"}),
                pooling_type='mean', sort_direction=sort_dir,
            )
            db.session.add(lm)
            db.session.commit()
            lm_ids.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(lm_ids)
        db.session.commit()
        print(f'LB_RESULT lb_id={lb.id} repo={repo} input_field={img.name} n_classes=0  (created)')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
