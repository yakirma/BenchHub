#!/usr/bin/env python
"""Build an MMLU leaderboard — BenchHub's flagship LLM knowledge benchmark.

MMLU (Hendrycks et al., 2021): 14,042 four-way multiple-choice questions across
57 subjects. Same pinned-protocol approach as GSM8K — the exact prompt (question
+ lettered choices + answer instruction) is baked into the input, so every
submitter sends the identical string. GT is the correct letter; scored by
letter exact-match. Zero-shot, single-letter answer.

Seed with scripts/submit_llm.py (BenchClient, max_new_tokens=8).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_mmlu.py [n_samples]
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

DS_NAME = 'MMLU-test'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 14042   # full test split
LETTERS = ['A', 'B', 'C', 'D']

PROMPT_TMPL = (
    "Answer the following multiple-choice question. Respond with only the "
    "letter (A, B, C, or D) of the correct option.\n\n"
    "Question: {q}\n{choices}\nAnswer:"
)

# Letter exact-match: pull the first standalone A–D letter out of the model's
# output and compare to the gold letter. Tolerant of "B", "B.", "(B)", and
# "The answer is B". Runs in the no-network sandbox; stdlib only.
MMLU_CODE = '''
def mmlu_accuracy(gt, pred):
    """MMLU letter exact-match (higher is better). Extracts the chosen option
    letter from the model's generation and compares it to the gold letter;
    1.0 on match, else 0.0."""
    import re
    gold = str(gt).strip().upper()[:1]
    m = re.search(r'\\b([A-D])\\b', str(pred).upper())
    return 1.0 if (m and m.group(1) == gold) else 0.0
'''


def build_staging(staging: Path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('prompt', 'answer'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download('cais/mmlu', 'all/test-00000-of-00001.parquet',
                           repo_type='dataset')
    df = pd.read_parquet(path)
    out = []
    for i, row in enumerate(df.head(N).itertuples(index=False)):
        d = row._asdict()
        name = f'q_{i:05d}'
        choices = list(d['choices'])
        lettered = '\n'.join(f'{LETTERS[j]}. {c}' for j, c in enumerate(choices))
        (staging / 'prompt' / f'{name}.txt').write_text(
            PROMPT_TMPL.format(q=str(d['question']), choices=lettered))
        (staging / 'answer' / f'{name}.txt').write_text(LETTERS[int(d['answer'])])
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'prompt', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'answer', 'kind': 'text', 'role': 'gt', 'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def ensure_metric(db, GlobalMetric, name, code):
    gm = GlobalMetric.query.filter_by(name=name).first()
    if gm is None:
        gm = GlobalMetric(name=name, python_code=code.strip(), owner_user_id=2,
                          visibility='public', is_aggregated=False)
        db.session.add(gm)
        db.session.commit()
    return gm


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='mmlu_'))
            try:
                names = build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'NLP/Knowledge & Reasoning'
                ds.source_url = 'https://huggingface.co/datasets/cais/mmlu'
                ds.source_kind = 'local-llm'
                ds.card_description = (
                    'MMLU (Hendrycks et al., 2021) — 14,042 multiple-choice '
                    'questions across 57 subjects. Pinned zero-shot prompt; '
                    'scored by letter exact match.')
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = ensure_metric(db, GlobalMetric, 'mmlu_accuracy', MMLU_CODE)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='NLP/Knowledge & Reasoning',
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
        print(f'MMLU_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
