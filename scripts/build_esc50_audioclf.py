#!/usr/bin/env python
"""Build a FULL-RES ESC-50 audio-classification leaderboard.

The catalog's audio datasets are preview-only (waveform PNGs — no usable
audio). This imports a full-res ESC-50 fold (raw .wav, 44.1 kHz) with the
50-class label vocab, so audio models can actually be scored on accuracy.

ESC-50 (`ashraq/esc50`): 2000 5-sec environmental-sound clips, 50 balanced
classes (`target` 0..49, `category` name), 5 folds. We take fold 1 (400 clips,
8 per class — all 50 classes covered) as the eval set.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_esc50_audioclf.py [fold]
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

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DS_NAME = 'ESC-50-audio'
AUD, LAB = 'audio', 'label'
METRICS = [('accuracy', 'accuracy', 'higher_is_better')]


def build_staging(staging: Path):
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    (staging / AUD).mkdir(parents=True, exist_ok=True)
    ds = load_dataset('ashraq/esc50', split='train', streaming=True)
    names = {}            # target -> category
    samples = []          # (name, target)
    for row in ds:
        if int(row['fold']) != FOLD:
            continue
        tgt = int(row['target'])
        names[tgt] = row['category']
        a = row['audio']
        arr = np.asarray(a['array'], dtype=np.float32)
        sr = int(a['sampling_rate'])
        name = Path(row['filename']).stem
        sf.write(str(staging / AUD / f'{name}.wav'), arr, sr)
        samples.append((name, tgt))
    if not samples:
        return None, None
    vocab = [names.get(i, f'class_{i}') for i in range(max(names) + 1)]
    # label GT is inline: write a manifest sample list + a sidecar values file.
    label_dir = staging / LAB
    label_dir.mkdir(parents=True, exist_ok=True)
    for name, tgt in samples:
        (label_dir / f'{name}.txt').write_text(str(tgt))
    manifest = {
        'name': DS_NAME, 'version': '1.0',
        'fields': [
            {'name': AUD, 'kind': 'audio', 'role': 'input', 'params': {}},
            {'name': LAB, 'kind': 'label', 'role': 'gt', 'params': {'names': vocab}},
        ],
        'samples': [n for n, _ in samples],
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return [n for n, _ in samples], vocab


def main():
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=DS_NAME).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix='esc50_'))
            try:
                names, vocab = build_staging(staging)
                if not names:
                    print('ESC_SKIP no samples'); return 1
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = 'Speech & Audio/Audio Classification'
                ds.source_url = 'https://huggingface.co/datasets/ashraq/esc50'
                ds.source_kind = 'local-audioclf'
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples, {len(vocab)} classes')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        lb_name = f'{DS_NAME}_benchmark'
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public',
                category='Speech & Audio/Audio Classification',
                required_pred_fields_json=json.dumps(
                    [{"name": "label_pred", "kind": "label", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({AUD: 'input', LAB: 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb); db.session.commit()
        keep = []
        for gname, label, sort_dir in METRICS:
            gm = GlobalMetric.query.filter_by(name=gname).first()
            if gm is None:
                print(f'  WARN metric {gname} missing'); continue
            lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
            if lm is None:
                lm = LeaderboardMetric(
                    leaderboard_id=lb.id, global_metric_id=gm.id, target_name=label,
                    arg_mappings=json.dumps({"gt": f"gt_{LAB}", "pred": "sub_label_pred"}),
                    pooling_type='mean', sort_direction=sort_dir)
                db.session.add(lm); db.session.commit()
            keep.append(f'lm_{lm.id}')
        lb.summary_metrics = ','.join(dict.fromkeys(keep))
        db.session.commit()
        print(f'AUDIOCLF_LB_BUILT lb_id={lb.id} ds={ds.id} name={lb_name} metrics={len(keep)}')


if __name__ == '__main__':
    raise SystemExit(main())
