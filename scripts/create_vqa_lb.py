#!/usr/bin/env python
"""Create a VQA leaderboard for a BenchHub dataset (idempotent).

Inputs = image + question (both role input); GT = the answer field. Contract =
text_pred (text); metric = answer_match (normalised accuracy, higher is better).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/create_vqa_lb.py <dataset_id>

Prints:  LB_RESULT lb_id=<id> repo=<hf_repo> input_field=<image_field> n_classes=0
or `LB_SKIP <reason>`.
"""
import os
import sys
import json
import re

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))

_ANS_PREFS = ('answer', 'answers', 'label', 'gt_answer')


def main():
    if len(sys.argv) < 2:
        print('LB_SKIP usage: create_vqa_lb.py <dataset_id>'); return 2
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
        q = next((f for f in fields if f.kind == 'text' and f.name.lower() == 'question'), None)
        texts = [f for f in fields if f.kind == 'text']
        ans = next((f for f in texts if f.name.lower() in _ANS_PREFS), None)
        if img is None or q is None or ans is None:
            print(f'LB_SKIP dataset {ds_id} ({ds.name}) needs image + question + answer fields'); return 1

        lb_name = f'{ds.name}_benchmark'
        existing = (Leaderboard.query.filter_by(name=lb_name).first()
                    or next((lb for lb in Leaderboard.query
                             .filter(Leaderboard.name.endswith('_benchmark')).all()
                             if any(d.id == ds_id for d in lb.datasets)), None))
        if existing is not None:
            print(f'LB_RESULT lb_id={existing.id} repo={repo} input_field={img.name} n_classes=0  (existing)')
            return 0

        contract = [{"name": "text_pred", "kind": "text", "params": {}, "role": "pred"}]
        lb = Leaderboard(
            name=lb_name, owner_user_id=2, visibility='public',
            category=(ds.category or 'NLP/Visual Question Answering'),
            required_pred_fields_json=json.dumps(contract),
            field_roles_json=json.dumps({img.name: "input", q.name: "input", ans.name: "gt"}),
            summary_metrics='',
        )
        lb.datasets.append(ds)
        db.session.add(lb)
        db.session.commit()

        gm = GlobalMetric.query.filter_by(name='answer_match').first()
        lm = LeaderboardMetric(
            leaderboard_id=lb.id, global_metric_id=gm.id, target_name='answer_match',
            arg_mappings=json.dumps({"gt": f"gt_{ans.name}", "pred": "sub_text_pred"}),
            pooling_type='mean', sort_direction='higher_is_better',
        )
        db.session.add(lm)
        db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'LB_RESULT lb_id={lb.id} repo={repo} input_field={img.name} n_classes=0  (created)')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
