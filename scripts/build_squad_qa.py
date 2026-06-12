#!/usr/bin/env python
"""Build an extractive question-answering leaderboard from SQuAD v1.1.

New task TYPE for the catalog (NLP / Question Answering). Imports the SQuAD
validation split as text inputs (question + context) with the gold answer set
stored as JSON GT, authors the standard SQuAD EM + token-F1 metrics, builds
the LB, and binds them. A matching HF question-answering pipeline is scored by
submit_qa.py (BenchClient).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_squad_qa.py [n_samples]
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

DS_NAME = 'SQuAD-v1.1-validation'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

SQUAD_F1_CODE = '''
def squad_f1(gt, pred):
    """SQuAD token-level F1 (higher is better): max over gold answers of the
    word-overlap F1 between normalized prediction and gold. gt is the SQuAD
    answers dict {"text": [...]}; pred is the predicted answer string."""
    import string
    from collections import Counter

    def unwrap(x):
        if x is None:
            return x
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'text'):
            x = x.text
        if hasattr(x, 'value'):
            x = x.value
        return x

    def norm(s):
        s = str(s).lower()
        s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
        toks = [t for t in s.split() if t not in ('a', 'an', 'the')]
        return toks

    g = unwrap(gt)
    golds = g.get('text') if isinstance(g, dict) else g
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        golds = ['']
    p = unwrap(pred)
    pred_s = p if isinstance(p, str) else (p.get('text') if isinstance(p, dict) else str(p))

    def f1(gold):
        gt_t = norm(gold)
        pr_t = norm(pred_s)
        if not gt_t and not pr_t:
            return 1.0
        if not gt_t or not pr_t:
            return 0.0
        common = Counter(gt_t) & Counter(pr_t)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        prec = overlap / len(pr_t)
        rec = overlap / len(gt_t)
        return 2 * prec * rec / (prec + rec)

    return float(max(f1(gold) for gold in golds))
'''

SQUAD_EM_CODE = '''
def squad_em(gt, pred):
    """SQuAD exact match (higher is better): 1.0 if the normalized prediction
    equals any normalized gold answer, else 0.0."""
    import string

    def unwrap(x):
        if x is None:
            return x
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'text'):
            x = x.text
        if hasattr(x, 'value'):
            x = x.value
        return x

    def norm(s):
        s = str(s).lower()
        s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
        return ' '.join(t for t in s.split() if t not in ('a', 'an', 'the'))

    g = unwrap(gt)
    golds = g.get('text') if isinstance(g, dict) else g
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        golds = ['']
    p = unwrap(pred)
    pred_s = p if isinstance(p, str) else (p.get('text') if isinstance(p, dict) else str(p))
    npred = norm(pred_s)
    return 1.0 if any(npred == norm(g) for g in golds) else 0.0
'''


def build_staging(staging: Path):
    # datasets 2.21 in the import venv can't deserialize SQuAD's features
    # (generate_from_dict TypeError), so read the published parquet directly.
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('question', 'context', 'answers'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download('rajpurkar/squad',
                           'plain_text/validation-00000-of-00001.parquet',
                           repo_type='dataset')
    df = pd.read_parquet(path)
    out = []
    for i, row in enumerate(df.head(N).itertuples(index=False)):
        d = row._asdict()
        name = f'q_{i:06d}'
        (staging / 'question' / f'{name}.txt').write_text(str(d['question']))
        (staging / 'context' / f'{name}.txt').write_text(str(d['context']))
        golds = list(d['answers']['text'])
        (staging / 'answers' / f'{name}.json').write_text(
            json.dumps({'text': [str(g) for g in golds]}))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'question', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'context', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'answers', 'kind': 'json', 'role': 'gt', 'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def ensure_metric(db, GlobalMetric, name, code, target):
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
            staging = Path(tempfile.mkdtemp(prefix='squad_'))
            try:
                names = build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'NLP/Question Answering'
                ds.source_url = 'https://huggingface.co/datasets/rajpurkar/squad'
                ds.source_kind = 'local-qa'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm_f1 = ensure_metric(db, GlobalMetric, 'squad_f1', SQUAD_F1_CODE, 'F1')
        gm_em = ensure_metric(db, GlobalMetric, 'squad_em', SQUAD_EM_CODE, 'EM')
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='NLP/Question Answering',
                required_pred_fields_json=json.dumps(
                    [{"name": "answer_pred", "kind": "text", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({'question': 'input', 'context': 'input', 'answers': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb)
            db.session.commit()
        sm = []
        for gm, tgt, prim in ((gm_f1, 'F1', True), (gm_em, 'EM', False)):
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=tgt,
                    arg_mappings=json.dumps({"gt": "gt_answers", "pred": "sub_answer_pred"}),
                    pooling_type='mean', sort_direction='higher_is_better')
                db.session.add(lm)
                db.session.commit()
            if prim:
                sm.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(sm)
        db.session.commit()
        print(f'QA_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
