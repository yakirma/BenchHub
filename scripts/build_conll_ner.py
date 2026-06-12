#!/usr/bin/env python
"""Build a named-entity-recognition leaderboard from CoNLL-2003 (test split).

New task TYPE (NLP / Token Classification). Imports the pre-tokenized test
sentences as a JSON `tokens` input, converts the gold BIO `ner_tags` to a set
of typed entity spans (`entities` = [[type, start, end], ...]), authors an
entity-level micro-F1 metric, builds the LB, and binds it. A token-
classification model is scored by submit_ner.py (BenchClient).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_conll_ner.py [n_samples]
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

DS_NAME = 'CoNLL-2003-NER-test'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

NER_F1_CODE = '''
def ner_f1(gt, pred):
    """Entity-level micro-F1 (higher is better): exact-match of typed spans
    [type, start, end] between gold and predicted entity sets. Sentences with
    no entities on either side return NaN (excluded from the mean)."""
    def unwrap(x):
        if x is None:
            return []
        if hasattr(x, 'data'):
            x = x.data
        if hasattr(x, 'value'):
            x = x.value
        return x if isinstance(x, list) else []

    def spanset(x):
        out = set()
        for e in unwrap(x):
            if isinstance(e, (list, tuple)) and len(e) == 3:
                out.add((str(e[0]), int(e[1]), int(e[2])))
        return out

    gs = spanset(gt)
    ps = spanset(pred)
    tp = len(gs & ps)
    fp = len(ps - gs)
    fn = len(gs - ps)
    if tp + fp + fn == 0:
        return float('nan')
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))
'''


def tags_to_spans(tags, names):
    """BIO(1/2) integer tags -> [[type, start, end], ...] inclusive spans."""
    spans = []
    cur = None
    for i, t in enumerate(tags):
        lab = names[int(t)]
        if lab == 'O':
            if cur:
                spans.append(cur); cur = None
            continue
        pre, typ = (lab.split('-', 1) + [''])[:2]
        if pre == 'B' or cur is None or cur[0] != typ:
            if cur:
                spans.append(cur)
            cur = [typ, i, i]
        else:
            cur[2] = i
    if cur:
        spans.append(cur)
    return spans


def build_staging(staging: Path):
    import itertools
    from datasets import load_dataset
    for f in ('tokens', 'entities'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    ds = load_dataset('conll2003', split='test', trust_remote_code=True)
    names = ds.features['ner_tags'].feature.names
    out = []
    for i, row in enumerate(itertools.islice(ds, N)):
        name = f's_{i:06d}'
        (staging / 'tokens' / f'{name}.json').write_text(json.dumps(list(row['tokens'])))
        spans = tags_to_spans(row['ner_tags'], names)
        (staging / 'entities' / f'{name}.json').write_text(json.dumps(spans))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': 'tokens', 'kind': 'json', 'role': 'input', 'params': {}},
            {'name': 'entities', 'kind': 'json', 'role': 'gt', 'params': {}},
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
            staging = Path(tempfile.mkdtemp(prefix='ner_'))
            try:
                names = build_staging(staging)
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'NLP/Token Classification'
                ds.source_url = 'https://huggingface.co/datasets/conll2003'
                ds.source_kind = 'local-ner'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = GlobalMetric.query.filter_by(name='ner_f1').first()
        if gm is None:
            gm = GlobalMetric(name='ner_f1', python_code=NER_F1_CODE.strip(), owner_user_id=2,
                              visibility='public', is_aggregated=False)
            db.session.add(gm); db.session.commit()
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='NLP/Token Classification',
                required_pred_fields_json=json.dumps(
                    [{"name": "entities_pred", "kind": "json", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({'tokens': 'input', 'entities': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='Entity F1',
                arg_mappings=json.dumps({"gt": "gt_entities", "pred": "sub_entities_pred"}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'NER_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
