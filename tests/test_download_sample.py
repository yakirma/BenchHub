"""Coverage for /sample/<id>/download — the per-sample ZIP bundle.

The download loop walks `Sample.custom_fields` and has to decide,
per data_type, whether the bytes live on disk (image / mask / depth
/ audio / json / bboxes / text) or inline in the row (scalar /
label / label_list). A prior version handled only scalar + image +
depth + json + text + histogram, so masks, audio, labels, bboxes
silently dropped from the bundle.
"""
import io
import os
import zipfile

import pytest

from app import CustomField, Dataset, Sample, app as flask_app, db


def _make_dataset_with_sample(tmp_path, monkeypatch):
    """Returns (dataset, sample, uploads_root) — the dataset has no
    fields yet; each test adds the ones it cares about."""
    monkeypatch.setitem(flask_app.config, 'UPLOAD_FOLDER', str(tmp_path))
    ds = Dataset(name="dl_ds")
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s001")
    db.session.add(sample); db.session.flush()
    return ds, sample, tmp_path


def _write_file(uploads_root, rel_path, content_bytes):
    full = os.path.join(str(uploads_root), rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'wb') as f:
        f.write(content_bytes)
    return rel_path


def _zip_names(resp):
    assert resp.status_code == 200, resp.data[:200]
    return set(zipfile.ZipFile(io.BytesIO(resp.data)).namelist())


def test_download_sample_includes_every_kind(client, db_session, tmp_path, monkeypatch):
    """One CustomField per supported kind; every one must appear in
    the resulting ZIP under `ground_truth/<field>/<sample>.<ext>` (or
    the file's original basename for file-backed kinds)."""
    ds, sample, uploads = _make_dataset_with_sample(tmp_path, monkeypatch)

    # Inline value kinds.
    db.session.add(CustomField(sample_id=sample.id, name='accuracy',
                               data_type='scalar', value_float=0.92))
    db.session.add(CustomField(sample_id=sample.id, name='caption',
                               data_type='label', value_text='cat'))
    db.session.add(CustomField(sample_id=sample.id, name='topk',
                               data_type='label_list', value_text='["cat","dog","fox"]'))

    # File-backed kinds. value_text holds the relative path under uploads/.
    img_rel   = _write_file(uploads, 'datasets/1/image/s001.png', b'\x89PNG\r\n\x1a\nfake')
    mask_rel  = _write_file(uploads, 'datasets/1/mask/s001.png',  b'\x89PNG\r\n\x1a\nfake')
    depth_rel = _write_file(uploads, 'datasets/1/depth/s001_512x256.npz', b'NPZ-fake')
    audio_rel = _write_file(uploads, 'datasets/1/audio/s001.wav',  b'RIFFfake')
    json_rel  = _write_file(uploads, 'datasets/1/meta/s001.json',  b'{"k":1}')
    bbox_rel  = _write_file(uploads, 'datasets/1/bbox/s001.json',  b'{"boxes":[]}')
    text_rel  = _write_file(uploads, 'datasets/1/notes/s001.txt',  b'hello')

    db.session.add(CustomField(sample_id=sample.id, name='image',
                               data_type='image', value_text=img_rel))
    db.session.add(CustomField(sample_id=sample.id, name='mask',
                               data_type='mask',  value_text=mask_rel))
    db.session.add(CustomField(sample_id=sample.id, name='depth',
                               data_type='depth', value_text=depth_rel))
    db.session.add(CustomField(sample_id=sample.id, name='audio',
                               data_type='audio', value_text=audio_rel))
    db.session.add(CustomField(sample_id=sample.id, name='meta',
                               data_type='json',  value_text=json_rel))
    db.session.add(CustomField(sample_id=sample.id, name='bbox',
                               data_type='bboxes', value_text=bbox_rel))
    db.session.add(CustomField(sample_id=sample.id, name='notes',
                               data_type='text',  value_text=text_rel))
    db.session.commit()

    names = _zip_names(client.get(f'/sample/{sample.id}/download'))

    # Inline kinds land under .txt / .json next to the sample name.
    assert 'ground_truth/accuracy/s001.txt' in names
    assert 'ground_truth/caption/s001.txt'  in names
    assert 'ground_truth/topk/s001.json'    in names

    # File-backed kinds preserve their original on-disk filename
    # (so depth's `_<W>x<H>` suffix survives the round-trip).
    assert 'ground_truth/image/s001.png'           in names
    assert 'ground_truth/mask/s001.png'            in names
    assert 'ground_truth/depth/s001_512x256.npz'   in names
    assert 'ground_truth/audio/s001.wav'           in names
    assert 'ground_truth/meta/s001.json'           in names
    assert 'ground_truth/bbox/s001.json'           in names
    assert 'ground_truth/notes/s001.txt'           in names


def test_download_sample_skips_metric_bookkeeping(client, db_session, tmp_path, monkeypatch):
    """Per-sample metric outputs (`lm_<id>` and any data_type='metric'
    row) are bookkeeping written by the engine — they shouldn't
    pollute a user-facing sample dump."""
    ds, sample, _ = _make_dataset_with_sample(tmp_path, monkeypatch)
    db.session.add(CustomField(sample_id=sample.id, name='lm_42',
                               data_type='scalar', value_float=0.5))
    db.session.add(CustomField(sample_id=sample.id, name='precomputed',
                               data_type='metric', value_float=0.7))
    db.session.add(CustomField(sample_id=sample.id, name='keep_me',
                               data_type='scalar', value_float=1.0))
    db.session.commit()

    names = _zip_names(client.get(f'/sample/{sample.id}/download'))
    assert 'ground_truth/keep_me/s001.txt' in names
    assert not any('lm_42' in n for n in names)
    assert not any('precomputed' in n for n in names)


def test_download_sample_handles_missing_file_backed_payload(client, db_session, tmp_path, monkeypatch):
    """If a file-backed CustomField's underlying file went missing
    (manual deletion, volume issue), the bundle must still build —
    just without that field. Other rows on the sample still come
    through."""
    ds, sample, _ = _make_dataset_with_sample(tmp_path, monkeypatch)
    db.session.add(CustomField(sample_id=sample.id, name='image',
                               data_type='image',
                               value_text='datasets/1/image/missing.png'))
    db.session.add(CustomField(sample_id=sample.id, name='score',
                               data_type='scalar', value_float=0.42))
    db.session.commit()

    names = _zip_names(client.get(f'/sample/{sample.id}/download'))
    assert 'ground_truth/score/s001.txt' in names
    assert not any(n.startswith('ground_truth/image/') for n in names)


def test_download_sample_materialized_prefers_full_res(client, db_session, tmp_path, monkeypatch):
    """With ?lb=<id>, file-backed fields are pulled from the LB's
    materialised dir (full-res) rather than the preview tier."""
    from app import Leaderboard
    from benchhub.lb_materialize import materialization_dir

    ds, sample, uploads = _make_dataset_with_sample(tmp_path, monkeypatch)
    lb = Leaderboard(name='dl_mat_lb', summary_metrics='')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    # Preview file (small) + materialised file (full-res) for the same field.
    preview_rel = _write_file(uploads, f'datasets/{ds.id}/image/s001.jpg', b'PREVIEW')
    db.session.add(CustomField(sample_id=sample.id, name='image',
                               data_type='image', value_text=preview_rel))
    db.session.commit()

    mat_dir = materialization_dir(str(uploads), lb.id)
    full = mat_dir / 'image' / 's001.jpg'
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b'FULL_RESOLUTION_BYTES')

    # Preview download → preview bytes.
    z_prev = zipfile.ZipFile(io.BytesIO(client.get(
        f'/sample/{sample.id}/download').data))
    assert z_prev.read('ground_truth/image/s001.jpg') == b'PREVIEW'

    # Materialised download → full-res bytes.
    resp = client.get(f'/sample/{sample.id}/download?lb={lb.id}')
    assert resp.headers['Content-Disposition'].count('materialized') == 1
    z_mat = zipfile.ZipFile(io.BytesIO(resp.data))
    assert z_mat.read('ground_truth/image/s001.jpg') == b'FULL_RESOLUTION_BYTES'
