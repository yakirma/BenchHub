#!/usr/bin/env python
"""Build a speech-emotion-recognition leaderboard from RAVDESS.

New Speech & Audio task type. RAVDESS has no canonical split, so we use a
speaker-independent test set (actors 21-24) — the standard leakage-minimizing
protocol for speaker-independent SER. 8 emotions. Reuses the audio+label
pipeline; scored by `accuracy` via submit_audio_label.py (name-maps each
model's emotion labels onto the board vocab).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_ravdess_emotion.py
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

DS_NAME = 'RAVDESS-speech-emotion'
AUD, LAB = 'audio', 'label'
TEST_ACTORS = {21, 22, 23, 24}
VOCAB = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def build_staging(staging: Path):
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    ds = load_dataset('xbgoose/ravdess', split='train')
    idx = {e: i for i, e in enumerate(VOCAB)}
    (staging / AUD).mkdir(parents=True, exist_ok=True)
    (staging / LAB).mkdir(parents=True, exist_ok=True)
    out = []
    for i, row in enumerate(ds):
        if row.get('vocal_channel') != 'speech' or int(row['actor']) not in TEST_ACTORS:
            continue
        emo = str(row['emotion']).strip().lower()
        if emo not in idx:
            continue
        a = row['audio']
        arr = np.asarray(a['array'], dtype=np.float32)
        sr = int(a['sampling_rate'])
        name = f'r_{i:05d}'
        sf.write(str(staging / AUD / f'{name}.wav'), arr, sr)
        (staging / LAB / f'{name}.txt').write_text(str(idx[emo]))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': AUD, 'kind': 'audio', 'role': 'input', 'params': {}},
            {'name': LAB, 'kind': 'label', 'role': 'gt', 'params': {'names': VOCAB}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(out)} RAVDESS test clips (actors {sorted(TEST_ACTORS)}), {len(VOCAB)} emotions')
    return out


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='ravdess_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Speech & Audio/Audio Classification'
                ds.source_url = 'https://huggingface.co/datasets/xbgoose/ravdess'
                ds.source_kind = 'local-audioclf'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Speech & Audio/Audio Classification',
                required_pred_fields_json=json.dumps(
                    [{"name": "label_pred", "kind": "label", "params": {"names": VOCAB}, "role": "pred"}]),
                field_roles_json=json.dumps({AUD: 'input', LAB: 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        gm = GlobalMetric.query.filter_by(name='accuracy').first()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='accuracy',
                arg_mappings=json.dumps({"gt": f"gt_{LAB}", "pred": "sub_label_pred"}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm); db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'RAVDESS_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
