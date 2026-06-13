#!/usr/bin/env python
"""Build a music-genre-classification leaderboard from GTZAN.

New Speech & Audio task type (music genre, distinct from ESC-50 env-sound).
GTZAN has no canonical split, so we replicate the HF audio-course split
(seed=42, shuffle, 10% test) — the split the public distilhubert/AST/HuBERT
gtzan models were trained against, so the held-out 10% is leakage-free for
them. Reuses the audio+label pipeline; scored by `accuracy` via
submit_audio_label.py.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_gtzan_genre.py
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

DS_NAME = 'GTZAN-music-genre'
AUD, LAB = 'audio', 'label'


def build_staging(staging: Path):
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    ds = load_dataset('sanchit-gandhi/gtzan', split='train')
    names = ds.features['genre'].names
    split = ds.train_test_split(seed=42, shuffle=True, test_size=0.1)['test']
    (staging / AUD).mkdir(parents=True, exist_ok=True)
    (staging / LAB).mkdir(parents=True, exist_ok=True)
    out = []
    for i, row in enumerate(split):
        a = row['audio']
        arr = np.asarray(a['array'], dtype=np.float32)
        sr = int(a['sampling_rate'])
        name = f'g_{i:04d}'
        sf.write(str(staging / AUD / f'{name}.wav'), arr, sr)
        (staging / LAB / f'{name}.txt').write_text(str(int(row['genre'])))
        out.append(name)
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': AUD, 'kind': 'audio', 'role': 'input', 'params': {}},
            {'name': LAB, 'kind': 'label', 'role': 'gt', 'params': {'names': names}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(out)} GTZAN test clips, {len(names)} genres: {names}')
    return out, names


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        vocab = None
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='gtzan_'))
            try:
                names, vocab = build_staging(staging)
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
                ds.source_url = 'https://huggingface.co/datasets/sanchit-gandhi/gtzan'
                ds.source_kind = 'local-audioclf'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        if vocab is None:
            import json as _j
            df = DatasetField.query.filter_by(dataset_id=ds.id, name=LAB).first()
            vocab = (_j.loads(df.params).get('names') if df and df.params else None) or []
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Speech & Audio/Audio Classification',
                required_pred_fields_json=json.dumps(
                    [{"name": "label_pred", "kind": "label", "params": {"names": vocab}, "role": "pred"}]),
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
        print(f'GTZAN_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
