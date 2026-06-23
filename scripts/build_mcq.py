#!/usr/bin/env python
"""Build a 4-way multiple-choice LLM board (HellaSwag or ARC) — same pinned-
protocol recipe as MMLU, reusing the `mmlu_accuracy` letter-match metric.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_mcq.py {hellaswag|arc} [n]
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

LETTERS = ['A', 'B', 'C', 'D']
KEY = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9

CONFIG = {
    'hellaswag': {
        'ds_name': 'HellaSwag-validation',
        'repo': 'Rowan/hellaswag',
        'parquet': 'data/validation-00000-of-00001.parquet',
        'category': 'NLP/Commonsense Reasoning',
        'source': 'https://huggingface.co/datasets/Rowan/hellaswag',
        'desc': 'HellaSwag (Zellers et al., 2019) — 10,042 commonsense '
                'sentence-completion questions (validation). Pinned zero-shot '
                'prompt; scored by letter exact match.',
        'instr': 'Choose the most plausible continuation of the passage. '
                 'Respond with only the letter (A, B, C, or D).',
        'stem': 'Passage',
    },
    'arc': {
        'ds_name': 'ARC-Challenge-test',
        'repo': 'allenai/ai2_arc',
        'parquet': 'ARC-Challenge/test-00000-of-00001.parquet',
        'category': 'NLP/Science QA',
        'source': 'https://huggingface.co/datasets/allenai/ai2_arc',
        'desc': 'ARC-Challenge (Clark et al., 2018) — grade-school science '
                'multiple-choice questions (test, 4-option). Pinned zero-shot '
                'prompt; scored by letter exact match.',
        'instr': 'Answer the multiple-choice science question. Respond with '
                 'only the letter (A, B, C, or D).',
        'stem': 'Question',
    },
}


def rows(df, key):
    """Yield (question_text, [4 choices], gold_idx) — only 4-option items."""
    for r in df.itertuples(index=False):
        d = r._asdict()
        if key == 'hellaswag':
            ch = list(d['endings'])
            if len(ch) != 4:
                continue
            yield str(d['ctx']), [str(c) for c in ch], int(d['label'])
        else:  # arc
            texts = list(d['choices']['text'])
            labels = list(d['choices']['label'])
            ak = str(d['answerKey'])
            if len(texts) != 4 or ak not in [str(x) for x in labels]:
                continue
            yield str(d['question']), [str(t) for t in texts], \
                [str(x) for x in labels].index(ak)


def build_staging(staging, cfg):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('prompt', 'answer'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(hf_hub_download(cfg['repo'], cfg['parquet'], repo_type='dataset'))
    out = []
    for i, (q, choices, gold) in enumerate(rows(df, KEY)):
        if i >= N:
            break
        name = f'q_{i:05d}'
        lettered = '\n'.join(f'{LETTERS[j]}. {c}' for j, c in enumerate(choices))
        (staging / 'prompt' / f'{name}.txt').write_text(
            f"{cfg['instr']}\n\n{cfg['stem']}: {q}\n{lettered}\nAnswer:")
        (staging / 'answer' / f'{name}.txt').write_text(LETTERS[gold])
        out.append(name)
    manifest = {
        'name': cfg['ds_name'], 'version': '1.0',
        'fields': [
            {'name': 'prompt', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'answer', 'kind': 'text', 'role': 'gt', 'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def main():
    cfg = CONFIG[KEY]
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=cfg['ds_name']).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix=f'{KEY}_'))
            try:
                build_staging(staging, cfg)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = cfg['category']
                ds.source_url = cfg['source']
                ds.source_kind = 'local-llm'
                ds.card_description = cfg['desc']
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = GlobalMetric.query.filter_by(name='mmlu_accuracy').first()
        if gm is None:
            raise SystemExit('mmlu_accuracy metric missing — run build_mmlu.py first')
        lb_name = f"{cfg['ds_name']}_benchmark"
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category=cfg['category'],
                required_pred_fields_json=json.dumps(
                    [{"name": "answer_pred", "kind": "text", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({'prompt': 'input', 'answer': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb)
            db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='Accuracy',
                arg_mappings=json.dumps({"gt": "gt_answer", "pred": "sub_answer_pred"}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm)
            db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'MCQ_LB_BUILT {KEY} lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
