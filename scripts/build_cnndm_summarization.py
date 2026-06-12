#!/usr/bin/env python
"""Build an abstractive-summarization leaderboard from CNN/DailyMail (test).

New task TYPE (NLP / Summarization). Imports the test articles as a text input
with the reference highlights as text GT, authors pure-Python ROUGE-1/2/L F1
metrics (the sandbox has numpy + stdlib only — no rouge_score), builds the LB,
and binds them. A seq2seq summarizer is scored by submit_summarization.py.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_cnndm_summarization.py [n]
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

DS_NAME = 'CNN-DailyMail-test'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500

# Shared body injected into each ROUGE metric (each GlobalMetric is exec'd in
# isolation, so the helpers are inlined per metric).
_ROUGE_HELPERS = '''
    import string
    from collections import Counter

    def unwrap(x):
        if x is None:
            return ''
        if hasattr(x, 'text'):
            x = x.text
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'value'):
            x = x.value
        return x if isinstance(x, str) else str(x)

    def toks(s):
        s = unwrap(s).lower()
        s = ''.join(c if c not in string.punctuation else ' ' for c in s)
        return s.split()

    def ngrams(t, n):
        return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1)) if len(t) >= n else Counter()

    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
'''

ROUGE_N_TMPL = '''
def {name}(gt, pred):
    """ROUGE-{n} F1 (higher is better)."""
{helpers}
    r = toks(gt); h = toks(pred)
    rg = ngrams(r, {n}); hg = ngrams(h, {n})
    overlap = sum((rg & hg).values())
    if overlap == 0:
        return 0.0
    return float(f1(overlap / max(sum(hg.values()), 1), overlap / max(sum(rg.values()), 1)))
'''

ROUGE_L_CODE = '''
def rougeL_f(gt, pred):
    """ROUGE-L F1 (LCS-based, higher is better)."""
{helpers}
    a = toks(gt); b = toks(pred)
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else (dp[j] if dp[j] >= dp[j - 1] else dp[j - 1])
            prev = tmp
    lcs = dp[len(b)]
    if lcs == 0:
        return 0.0
    return float(f1(lcs / len(b), lcs / len(a)))
'''


def metric_codes():
    return {
        'rouge1_f': ROUGE_N_TMPL.format(name='rouge1_f', n=1, helpers=_ROUGE_HELPERS),
        'rouge2_f': ROUGE_N_TMPL.format(name='rouge2_f', n=2, helpers=_ROUGE_HELPERS),
        'rougeL_f': ROUGE_L_CODE.format(helpers=_ROUGE_HELPERS),
    }


def build_staging(staging: Path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('article', 'highlights'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download('abisee/cnn_dailymail',
                           '3.0.0/test-00000-of-00001.parquet', repo_type='dataset')
    df = pd.read_parquet(path, columns=['article', 'highlights']).head(N)
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        name = f'd_{i:06d}'
        (staging / 'article' / f'{name}.txt').write_text(str(row.article))
        (staging / 'highlights' / f'{name}.txt').write_text(str(row.highlights))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'article', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'highlights', 'kind': 'text', 'role': 'gt', 'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='cnndm_'))
            try:
                build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'NLP/Summarization'
                ds.source_url = 'https://huggingface.co/datasets/abisee/cnn_dailymail'
                ds.source_kind = 'local-summarization'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        codes = metric_codes()
        gms = {}
        for nm, code in codes.items():
            gm = GlobalMetric.query.filter_by(name=nm).first()
            if gm is None:
                gm = GlobalMetric(name=nm, python_code=code.strip(), owner_user_id=2,
                                  visibility='public', is_aggregated=False)
                db.session.add(gm); db.session.commit()
            gms[nm] = gm
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='NLP/Summarization',
                required_pred_fields_json=json.dumps(
                    [{"name": "summary_pred", "kind": "text", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({'article': 'input', 'highlights': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        sm = []
        for nm, tgt in (('rougeL_f', 'ROUGE-L'), ('rouge1_f', 'ROUGE-1'), ('rouge2_f', 'ROUGE-2')):
            gm = gms[nm]
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=tgt,
                    arg_mappings=json.dumps({"gt": "gt_highlights", "pred": "sub_summary_pred"}),
                    pooling_type='mean', sort_direction='higher_is_better')
                db.session.add(lm); db.session.commit()
            if nm == 'rougeL_f':
                sm.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(sm)
        db.session.commit()
        print(f'SUMM_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
