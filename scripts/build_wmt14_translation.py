#!/usr/bin/env python
"""Build a machine-translation leaderboard from WMT14 de-en (newstest2014).

New task TYPE (NLP / Translation). German source -> English reference. Imports
the test pairs (reads parquet directly), authors a pure-Python sentence BLEU-4
metric (numpy+stdlib sandbox — no sacrebleu), builds the LB, binds it. A de->en
model is scored by submit_translation.py.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_wmt14_translation.py [n]
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

DS_NAME = 'WMT14-de-en-newstest2014'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

BLEU_CODE = '''
def bleu(gt, pred):
    """Sentence BLEU-4 (higher is better), pooled per-sample by mean. Unigram
    precision is unsmoothed (no word overlap -> 0); orders 2-4 use add-1
    smoothing so a single missing higher-order n-gram doesn't zero the score.
    Brevity penalty vs the reference length. gt=reference, pred=hypothesis."""
    import math
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
        return unwrap(s).split()

    def ngr(t, n):
        return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

    ref = toks(gt); hyp = toks(pred)
    if not hyp or not ref:
        return 0.0
    log_p = 0.0
    for n in range(1, 5):
        h = ngr(hyp, n); r = ngr(ref, n)
        overlap = sum(min(c, r.get(g, 0)) for g, c in h.items())
        total = max(sum(h.values()), 1)
        if n == 1:
            if overlap == 0:
                return 0.0
            p = overlap / total
        else:
            p = (overlap + 1) / (total + 1)
        log_p += 0.25 * math.log(p)
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return float(bp * math.exp(log_p))
'''


def build_staging(staging: Path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('source', 'reference'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download('wmt/wmt14', 'de-en/test-00000-of-00001.parquet',
                           repo_type='dataset')
    df = pd.read_parquet(path).head(N)
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        tr = row.translation
        de, en = str(tr['de']), str(tr['en'])
        if not de.strip() or not en.strip():
            continue
        name = f't_{i:06d}'
        (staging / 'source' / f'{name}.txt').write_text(de)
        (staging / 'reference' / f'{name}.txt').write_text(en)
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'source', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'reference', 'kind': 'text', 'role': 'gt', 'params': {}},
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
            staging = Path(tempfile.mkdtemp(prefix='wmt_'))
            try:
                build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'NLP/Translation'
                ds.source_url = 'https://huggingface.co/datasets/wmt/wmt14'
                ds.source_kind = 'local-translation'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = GlobalMetric.query.filter_by(name='bleu').first()
        if gm is None:
            gm = GlobalMetric(name='bleu', python_code=BLEU_CODE.strip(), owner_user_id=2,
                              visibility='public', is_aggregated=False)
            db.session.add(gm); db.session.commit()
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='NLP/Translation',
                required_pred_fields_json=json.dumps(
                    [{"name": "translation_pred", "kind": "text", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({'source': 'input', 'reference': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='BLEU',
                arg_mappings=json.dumps({"gt": "gt_reference", "pred": "sub_translation_pred"}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'MT_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
