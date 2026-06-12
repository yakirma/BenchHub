#!/usr/bin/env python
"""Build a FULL-RES LibriSpeech ASR leaderboard + a WER metric.

LibriSpeech test-clean is the canonical English ASR benchmark. We import a
full-res subset (raw 16 kHz .wav + reference transcript text) so speech-to-text
models can be scored on Word Error Rate.

Creates the `wer` GlobalMetric (word-level Levenshtein after case/punct
normalisation; lower is better) if it doesn't exist, then the dataset + LB.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_librispeech_asr.py [n_samples]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DS_NAME = 'LibriSpeech-test-clean'
AUD, TXT = 'audio', 'transcript'

WER_CODE = '''import numpy as np
import benchhub as bh
import re


def wer(gt: bh.Text, pred: bh.Text):
    """Word Error Rate (lower is better): word-level Levenshtein distance
    between reference and hypothesis, normalised by reference length, after
    lower-casing + stripping punctuation."""
    if gt is None or pred is None:
        return float('nan')
    def norm(t):
        s = t.text if hasattr(t, 'text') else str(t)
        s = re.sub(r"[^a-z0-9' ]", ' ', s.lower())
        return s.split()
    r = norm(gt); h = norm(pred)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(h)]) / len(r)
'''


def build_staging(staging: Path):
    import itertools
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    (staging / AUD).mkdir(parents=True, exist_ok=True)
    (staging / TXT).mkdir(parents=True, exist_ok=True)
    ds = load_dataset('openslr/librispeech_asr', 'clean', split='test',
                      streaming=True, trust_remote_code=True)
    names = []
    for row in itertools.islice(ds, N):
        a = row['audio']
        arr = np.asarray(a['array'], dtype=np.float32)
        sr = int(a['sampling_rate'])
        name = str(row.get('id') or row.get('file') or f'utt_{len(names)}').replace('/', '_')
        sf.write(str(staging / AUD / f'{name}.wav'), arr, sr)
        (staging / TXT / f'{name}.txt').write_text(row['text'])
        names.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': AUD, 'kind': 'audio', 'role': 'input', 'params': {}},
            {'name': TXT, 'kind': 'text', 'role': 'gt', 'params': {}},
        ],
        'samples': names,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return names


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        # 1. WER metric
        gm = GlobalMetric.query.filter_by(name='wer').first()
        if gm is None:
            gm = GlobalMetric(
                name='wer', description='Word Error Rate (lower is better).',
                python_code=WER_CODE, is_aggregated=0, accepts_aggregated_inputs=0,
                input_kinds='["text", "text"]', input_roles='["gt", "pred"]',
                owner_user_id=None, visibility='public')
            db.session.add(gm); db.session.commit()
            print('created wer metric')

        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='librispeech_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('ASR_SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Speech & Audio/Speech Recognition'
                ds.source_url = 'https://huggingface.co/datasets/openslr/librispeech_asr'
                ds.source_kind = 'local-asr'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Speech & Audio/Speech Recognition',
                required_pred_fields_json=json.dumps(
                    [{"name": "transcript_pred", "kind": "text", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({AUD: 'input', TXT: 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='WER',
                arg_mappings=json.dumps({"gt": f"gt_{TXT}", "pred": "sub_transcript_pred"}),
                pooling_type='mean', sort_direction='lower_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'ASR_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
