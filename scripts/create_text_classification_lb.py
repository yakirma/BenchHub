#!/usr/bin/env python
"""Create an NLP text-classification leaderboard for a BenchHub dataset (idempotent).

NLP analogue of create_classification_lb.py: the INPUT is a text field (the
sentence/review), the GT is a `label` field with class names. Contract =
label_pred (label) + label_topk_pred (label_list); metric = top_1_accuracy
(+ top_5_accuracy when class count > 5) — the SAME accuracy metric the image
boards use, since both compare a predicted label index to the GT label.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/create_text_classification_lb.py <dataset_id>

Prints:  LB_RESULT lb_id=<id> repo=<hf_repo> input_field=<name> n_classes=<n>
or `LB_SKIP <reason>`.
"""
import os
import sys
import json
import re

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))

# Text fields that look like the primary model input (vs ids / parses / metadata).
_INPUT_NAME_PREFS = ('text', 'sentence', 'review', 'content', 'document')


def main():
    if len(sys.argv) < 2:
        print('LB_SKIP usage: create_text_classification_lb.py <dataset_id>'); return 2
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
        texts = [f for f in fields if f.kind == 'text']
        # input = the primary text field by name preference, else the first text field
        inp = next((f for f in texts if f.name.lower() in _INPUT_NAME_PREFS), None)
        inp = inp or (texts[0] if texts else None)
        lab = next((f for f in fields if f.kind == 'label' and (f.role or 'gt') == 'gt'), None)
        lab = lab or next((f for f in fields if f.kind == 'label'), None)
        if inp is None or lab is None:
            print(f'LB_SKIP dataset {ds_id} ({ds.name}) needs a text input + label GT field'); return 1
        try:
            names = (json.loads(lab.params) if lab.params else {}).get('names')
        except Exception:
            names = None
        if not names or len(names) < 2:
            print(f'LB_SKIP dataset {ds_id} ({ds.name}) label field has no class names'); return 1

        lb_name = f'{ds.name}_benchmark'
        existing = (Leaderboard.query.filter_by(name=lb_name).first()
                    or next((lb for lb in Leaderboard.query
                             .filter(Leaderboard.name.endswith('_benchmark')).all()
                             if any(d.id == ds_id for d in lb.datasets)), None))
        if existing is not None:
            print(f'LB_RESULT lb_id={existing.id} repo={repo} input_field={inp.name} '
                  f'n_classes={len(names)}  (existing)'); return 0

        n = len(names)
        k = min(5, n)
        contract = [
            {"name": "label_pred", "kind": "label", "params": {"names": names}, "role": "pred"},
            {"name": "label_topk_pred", "kind": "label_list", "params": {"names": names, "k": k}, "role": "pred"},
        ]
        lb = Leaderboard(
            name=lb_name, owner_user_id=2, visibility='public',
            category=(ds.category or 'NLP/Text Classification'),
            required_pred_fields_json=json.dumps(contract),
            field_roles_json=json.dumps({inp.name: "input", lab.name: "gt"}),
            summary_metrics='',
        )
        lb.datasets.append(ds)
        db.session.add(lb)
        db.session.commit()

        gt_key = f'gt_{lab.name}'
        lm_ids = []
        wanted = ['top_1_accuracy'] + (['top_5_accuracy'] if n > 5 else [])
        for gm_name in wanted:
            gm = GlobalMetric.query.filter_by(name=gm_name).first()
            if gm is None:
                continue
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name=gm_name,
                arg_mappings=json.dumps({"gt": gt_key, "pred": "sub_label_topk_pred"}),
                pooling_type='mean', sort_direction='higher_is_better',
            )
            db.session.add(lm)
            db.session.commit()
            lm_ids.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(lm_ids)
        db.session.commit()
        print(f'LB_RESULT lb_id={lb.id} repo={repo} input_field={inp.name} n_classes={n}  (created)')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
