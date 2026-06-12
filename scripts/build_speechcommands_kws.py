#!/usr/bin/env python
"""Build a keyword-spotting leaderboard from Speech Commands v0.02 (test),
in the canonical SUPERB KS 12-class formulation.

New task TYPE (Speech & Audio / Keyword Spotting). 10 core command words +
`_silence_` + `_unknown_` (every other word). Imports the test clips full-res
(16 kHz wav) with the 12-class GT, balanced per class, and binds the existing
`accuracy` metric. Models are scored by submit_audio_label.py — each predicted
label name maps to the 12-class vocab (any non-core word -> _unknown_), so both
SUPERB-KS models (12-class) and 35-word AST models share the board.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_speechcommands_kws.py [per_class]
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

DS_NAME = 'SpeechCommands-v2-KWS'
AUD, LAB = 'audio', 'label'
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SCAN_CAP = 12000

CORE = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
VOCAB = CORE + ['_silence_', '_unknown_']   # 12-class SUPERB KS order


def target_class(name):
    if name in CORE:
        return CORE.index(name)
    if name == '_silence_':
        return 10
    return 11  # any other word -> _unknown_


def build_staging(staging: Path):
    import itertools
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    (staging / AUD).mkdir(parents=True, exist_ok=True)
    (staging / LAB).mkdir(parents=True, exist_ok=True)
    ds = load_dataset('google/speech_commands', 'v0.02', split='test',
                      streaming=True, trust_remote_code=True)
    names = ds.features['label'].names
    per = {i: 0 for i in range(12)}
    samples = []
    for i, row in enumerate(itertools.islice(ds, SCAN_CAP)):
        cls = target_class(names[int(row['label'])])
        if per[cls] >= PER_CLASS:
            continue
        a = row['audio']
        arr = np.asarray(a['array'], dtype=np.float32)
        sr = int(a['sampling_rate'])
        name = f'k_{i:06d}'
        sf.write(str(staging / AUD / f'{name}.wav'), arr, sr)
        (staging / LAB / f'{name}.txt').write_text(str(cls))
        samples.append(name)
        per[cls] += 1
        if all(v >= PER_CLASS for v in per.values()):
            break
    if not samples:
        return None
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': AUD, 'kind': 'audio', 'role': 'input', 'params': {}},
            {'name': LAB, 'kind': 'label', 'role': 'gt', 'params': {'names': VOCAB}},
        ],
        'samples': samples,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('per-class counts:', per)
    return samples


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='kws_'))
            try:
                names = build_staging(staging)
                if not names:
                    print('KWS_SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Speech & Audio/Keyword Spotting'
                ds.source_url = 'https://huggingface.co/datasets/google/speech_commands'
                ds.source_kind = 'local-kws'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Speech & Audio/Keyword Spotting',
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
        print(f'KWS_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    raise SystemExit(main())
