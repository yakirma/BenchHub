#!/usr/bin/env python
"""Build depth-estimation leaderboards for a set of datasets, like nyuv2.

For each dataset id: create a `<name>_benchmark` LB (image input + depth GT,
contract `depth_pred`, metrics rmse_depth / scale_inv_rmse_depth /
affine_inv_rmse_depth / delta1_depth), then create + trigger a per-LB
materialisation (full-res depth npz — preview depth is a colormapped JPG and
can't be scored). Idempotent: skips an LB / materialisation that already
exists.

Prints one `LB_BUILT lb_id=<id> ds=<id> input=<field> repo=<hf_repo>` line per
dataset (or `LB_SKIP <reason>`) — feed those to submit_depth.py.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_depth_lbs.py <ds_id> [<ds_id> ...]
"""
import os
import sys
import json
import re

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))

MAT_CAP = 50          # eval-set size, matching nyuv2 (lb75)
MAT_SAMPLING = 'random'
DEPTH_METRICS = [('rmse_depth', 'lower_is_better'),
                 ('affine_inv_rmse_depth', 'lower_is_better'),
                 ('scale_inv_rmse_depth', 'lower_is_better'),
                 ('delta1_depth', 'higher_is_better')]


def build_one(ds_id):
    import app as A
    from app import (db, Leaderboard, LeaderboardMetric, GlobalMetric, Dataset,
                     DatasetField, LeaderboardMaterialization)
    import tasks as _tasks

    with A.app.app_context():
        ds = Dataset.query.get(ds_id)
        if ds is None:
            print(f'LB_SKIP {ds_id} not found'); return
        m = re.search(r'huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)', ds.source_url or '')
        repo = m.group(1) if m else (ds.source_metadata and json.loads(ds.source_metadata).get('repo_id')) or ''

        fields = DatasetField.query.filter_by(dataset_id=ds_id).all()
        img = (next((f for f in fields if f.kind == 'image' and (f.role or '') == 'input'), None)
               or next((f for f in fields if f.kind == 'image'), None))
        dep = (next((f for f in fields if f.kind == 'depth' and (f.role or 'gt') == 'gt'), None)
               or next((f for f in fields if f.kind == 'depth'), None))
        if img is None or dep is None:
            print(f'LB_SKIP {ds_id} ({ds.name}) needs image input + depth GT'); return

        lb_name = f'{ds.name}_benchmark'
        lb = (Leaderboard.query.filter_by(name=lb_name).first()
              or next((l for l in Leaderboard.query.filter(Leaderboard.name.endswith('_benchmark')).all()
                       if any(d.id == ds_id for d in l.datasets)), None))
        if lb is None:
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
            for mname, sort_dir in DEPTH_METRICS:
                gm = GlobalMetric.query.filter_by(name=mname).first()
                if gm is None:
                    continue
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=mname,
                    arg_mappings=json.dumps({"gt": f"gt_{dep.name}", "pred": "sub_depth_pred"}),
                    pooling_type='mean', sort_direction=sort_dir)
                db.session.add(lm); db.session.commit(); lm_ids.append(f'lm_{lm.id}')
            lb.summary_metrics = ','.join(lm_ids)
            db.session.commit()
            created = 'created'
        else:
            created = 'existing'

        # Materialisation: full-res depth so the metrics can score. Only
        # create one if the LB has no ready/pending/importing materialisation.
        mat = LeaderboardMaterialization.query.filter_by(leaderboard_id=lb.id).first()
        if mat is None or mat.status == 'failed':
            if mat is not None:
                db.session.delete(mat); db.session.commit()
            mat = LeaderboardMaterialization(
                leaderboard_id=lb.id, sample_cap=MAT_CAP, sampling=MAT_SAMPLING,
                sampling_seed=42, shard_cap=-1, status='pending')
            db.session.add(mat); db.session.commit()
            _tasks.materialize_leaderboard.delay(lb.id)
            mat_state = 'queued'
        else:
            mat_state = mat.status

        print(f'LB_BUILT lb_id={lb.id} ds={ds_id} input={img.name} repo={repo} '
              f'({created}, mat={mat_state})')


def main():
    ids = [int(x) for x in sys.argv[1:] if x.strip()]
    if not ids:
        print('usage: build_depth_lbs.py <ds_id> [...]'); return 2
    for ds_id in ids:
        try:
            build_one(ds_id)
        except Exception as e:
            import traceback
            print(f'LB_FAIL {ds_id} {type(e).__name__}: {e}')
            traceback.print_exc()
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
