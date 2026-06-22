#!/usr/bin/env python
"""Build a GSM8K grade-school-math leaderboard — BenchHub's first LLM benchmark.

New task type for the catalog (NLP / Mathematical Reasoning). The differentiator
vs other LLM leaderboards: the eval *protocol* is pinned by baking the exact
prompt into the dataset `input`, so every submitter sends the identical string
to their model — the apples-to-apples promise actually holds. Imports the GSM8K
test split (1319 problems), stores the gold numeric answer as GT, authors a
final-answer exact-match metric, builds the LB, and binds it.

Submit baselines with scripts/submit_gsm8k.py (BenchClient).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_gsm8k.py [n_samples]
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

DS_NAME = 'GSM8K-test'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1319   # full test split

# The pinned eval protocol: every submission's model is given THIS exact string.
# Zero-shot, explicit final-answer format so extraction is unambiguous.
PROMPT_TMPL = (
    "Solve the following grade-school math word problem. Think step by step, "
    "then on the final line output only:\n#### <answer>\n\n"
    "Question: {q}\nAnswer:"
)

# Final-answer exact match. Extracts the last number from the model's output
# (handles $, commas, decimals) and compares to the gold number. This is the
# standard GSM8K scoring. Runs in the no-network sandbox; stdlib only.
GSM8K_CODE = '''
def gsm8k_accuracy(gt, pred):
    """GSM8K final-answer exact match (higher is better). Pulls the last number
    out of the model's generation and compares it to the gold answer; 1.0 on a
    numeric match, else 0.0. Tolerant of $, thousands-commas, and trailing
    periods so formatting doesn't cost a correct answer."""
    import re

    def last_number(s):
        s = str(s).replace('$', '').replace(',', '')
        # prefer the number after a '####' marker when present (gold + well-
        # formatted predictions), else the last number anywhere in the text.
        marked = re.findall(r'####\\s*(-?\\d+\\.?\\d*)', s)
        cands = marked if marked else re.findall(r'-?\\d+\\.?\\d*', s)
        if not cands:
            return None
        x = cands[-1].rstrip('.')
        try:
            return float(x)
        except ValueError:
            return None

    g = last_number(gt)
    p = last_number(pred)
    if g is None or p is None:
        return 0.0
    return 1.0 if abs(g - p) < 1e-6 else 0.0
'''


def build_staging(staging: Path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('prompt', 'answer'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download('openai/gsm8k', 'main/test-00000-of-00001.parquet',
                           repo_type='dataset')
    df = pd.read_parquet(path)
    out = []
    for i, row in enumerate(df.head(N).itertuples(index=False)):
        d = row._asdict()
        name = f'g_{i:05d}'
        (staging / 'prompt' / f'{name}.txt').write_text(
            PROMPT_TMPL.format(q=str(d['question'])))
        # GSM8K gold answer is CoT text ending in "#### <number>".
        gold = str(d['answer']).split('####')[-1].strip().replace(',', '')
        (staging / 'answer' / f'{name}.txt').write_text(gold)
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
            staging = Path(tempfile.mkdtemp(prefix='gsm8k_'))
            try:
                names = build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'NLP/Mathematical Reasoning'
                ds.source_url = 'https://huggingface.co/datasets/openai/gsm8k'
                ds.source_kind = 'local-llm'
                ds.card_description = (
                    'GSM8K (Cobbe et al., 2021) — 1,319 grade-school math word '
                    'problems. The prompt is pinned in the input; scored by '
                    'final-answer exact match.')
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = ensure_metric(db, GlobalMetric, 'gsm8k_accuracy', GSM8K_CODE)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='NLP/Mathematical Reasoning',
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
        print(f'GSM8K_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
