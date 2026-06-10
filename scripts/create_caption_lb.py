#!/usr/bin/env python
"""Create an image-captioning / OCR leaderboard (idempotent).

Input = image, GT = a text field (caption / transcription). Contract =
text_pred (text). Metric chosen by arg: `bleu` (captioning, higher is better)
or `cer` (OCR, lower is better).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/create_caption_lb.py <dataset_id> <bleu|cer>

Prints:  LB_RESULT lb_id=<id> repo=<hf_repo> input_field=<name> n_classes=0
or `LB_SKIP <reason>`.
"""
import os
import sys
import json
import re

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))

_GT_NAME_PREFS = ('caption', 'text', 'transcription', 'sentence', 'label', 'answer')
_METRIC = {'bleu': ('bleu_text', 'higher_is_better'), 'cer': ('cer_text', 'lower_is_better')}


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in _METRIC:
        print('LB_SKIP usage: create_caption_lb.py <dataset_id> <bleu|cer>'); return 2
    ds_id = int(sys.argv[1])
    metric_name, sort_dir = _METRIC[sys.argv[2]]
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
        # input image (OCR datasets mis-role the image as gt — treat any image as input)
        img = next((f for f in fields if f.kind == 'image' and (f.role or '') == 'input'), None)
        img = img or next((f for f in fields if f.kind == 'image'), None)
        # GT text: prefer caption/text/transcription, never a url field
        texts = [f for f in fields if f.kind == 'text' and 'url' not in f.name.lower()]
        gt = next((f for f in texts if f.name.lower() in _GT_NAME_PREFS), None)
        gt = gt or (texts[0] if texts else None)
        if img is None or gt is None:
            print(f'LB_SKIP dataset {ds_id} ({ds.name}) needs an image input + text GT field'); return 1

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
            category=(ds.category or 'Vision/Image Captioning'),
            required_pred_fields_json=json.dumps(contract),
            field_roles_json=json.dumps({img.name: "input", gt.name: "gt"}),
            summary_metrics='',
        )
        lb.datasets.append(ds)
        db.session.add(lb)
        db.session.commit()

        gm = GlobalMetric.query.filter_by(name=metric_name).first()
        lm = LeaderboardMetric(
            leaderboard_id=lb.id, global_metric_id=gm.id, target_name=metric_name,
            arg_mappings=json.dumps({"gt": f"gt_{gt.name}", "pred": "sub_text_pred"}),
            pooling_type='mean', sort_direction=sort_dir,
        )
        db.session.add(lm)
        db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'LB_RESULT lb_id={lb.id} repo={repo} input_field={img.name} n_classes=0  (created)')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
