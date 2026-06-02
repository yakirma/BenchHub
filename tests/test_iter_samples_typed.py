"""`Client.iter_samples` yields decoded `bh.<Kind>` instances, not
bare PIL images / scalars. This matches what the predict side expects
on the wire and what the generated submission notebook documents."""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image as PILImage

import benchhub as bh
from app import (
    CustomField,
    Dataset,
    DatasetField,
    Leaderboard,
    Sample,
    User,
    app as flask_app,
    db,
)
from benchhub.client import Client, FlaskTestClientTransport


def _seed_image_lb(client, *, vis='public'):
    user = User(email='its@bench.local', display_name='its',
                oauth_provider='github', oauth_sub='its-1',
                api_token='itstok')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='its_ds', visibility=vis, owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='img',
                                kind='image', role='input'))
    db.session.add(DatasetField(dataset_id=ds.id, name='label',
                                kind='label', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()

    # Drop a real (32x32x3) PNG on disk and register it as a CustomField.
    ds_dir = os.path.join(flask_app.config['UPLOAD_FOLDER'],
                          'datasets', str(ds.id), 'img')
    os.makedirs(ds_dir, exist_ok=True)
    img_path = os.path.join(ds_dir, 's0.png')
    arr = (np.arange(32 * 32 * 3) % 256).astype(np.uint8).reshape(32, 32, 3)
    PILImage.fromarray(arr).save(img_path)
    db.session.add(CustomField(
        sample_id=s.id, name='img', data_type='image',
        value_text=img_path,
    ))
    lb = Leaderboard(name='its_lb', visibility=vis, owner_user_id=user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return user, lb, arr


def test_iter_samples_yields_bh_image_for_image_field(client, db_session):
    _, lb, original_arr = _seed_image_lb(client)
    bhc = Client(token='itstok',
                 transport=FlaskTestClientTransport(client))
    out = list(bhc.iter_samples(lb.id))
    assert len(out) == 1
    sample_name, inputs = out[0]
    assert sample_name == 's0'
    assert 'img' in inputs
    # The value is a typed `bh.Image`, not a bare PIL.JpegImageFile.
    img = inputs['img']
    assert isinstance(img, bh.Image), f"expected bh.Image, got {type(img).__name__}"
    # `.array` round-trips to the canonical uint8 ndarray we wrote.
    assert img.array.shape == (32, 32, 3)
    assert img.array.dtype == np.uint8


def test_iter_samples_falls_back_to_image_for_unknown_blob_format(
        client, db_session, monkeypatch,
):
    """A preview-tier depth field is colormapped JPG, which doesn't
    round-trip through Depth.decode (expects .npz). The client falls
    back to wrapping as bh.Image so the user still gets a typed
    instance, not raw bytes."""
    user = User(email='itsd@bench.local', display_name='itsd',
                oauth_provider='github', oauth_sub='itsd-1',
                api_token='itsdtok')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='itsd_ds', visibility='public', owner_user_id=user.id,
                 preview_only=True)
    db.session.add(ds); db.session.flush()
    depth_field = DatasetField(dataset_id=ds.id, name='depth',
                               kind='depth', role='input')
    depth_field.set_params({'unit': 'meters'})
    db.session.add(depth_field)
    db.session.add(DatasetField(dataset_id=ds.id, name='label',
                                kind='label', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    ds_dir = os.path.join(flask_app.config['UPLOAD_FOLDER'],
                          'datasets', str(ds.id), 'depth')
    os.makedirs(ds_dir, exist_ok=True)
    # Save a JPG (a real preview-tier depth blob is colormapped JPG;
    # the format alone is enough to exercise the fallback path).
    jpg_path = os.path.join(ds_dir, 's0.jpg')
    PILImage.fromarray(
        np.full((16, 16, 3), 128, dtype=np.uint8)
    ).save(jpg_path, format='JPEG')
    db.session.add(CustomField(
        sample_id=s.id, name='depth', data_type='depth',
        value_text=jpg_path,
    ))
    lb = Leaderboard(name='itsd_lb', visibility='public', owner_user_id=user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    bhc = Client(token='itsdtok',
                 transport=FlaskTestClientTransport(client))
    out = list(bhc.iter_samples(lb.id))
    assert len(out) == 1
    _, inputs = out[0]
    val = inputs['depth']
    # Depth.decode rejected the JPG; fallback wraps as bh.Image so
    # the caller still gets a typed instance.
    assert isinstance(val, (bh.Depth, bh.Image)), (
        f"expected bh.Depth or fallback bh.Image, got {type(val).__name__}"
    )


def test_iter_samples_inline_kinds_unwrap_to_raw_values(client, db_session):
    """Inline (non-file-backed) kinds — scalar/label/text/json/label_list —
    still surface as their decoded Python value, not wrapped in a
    bh.<Kind>. (Wrapping inline values would force `.value` access
    everywhere for no benefit; raw is friendlier.)"""
    user = User(email='itsi@bench.local', display_name='itsi',
                oauth_provider='github', oauth_sub='itsi-1',
                api_token='itsitok')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='itsi_ds', visibility='public', owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='caption', kind='text',
                     role='input'),
        DatasetField(dataset_id=ds.id, name='label', kind='label',
                     role='gt'),
    ])
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='caption', data_type='text',
        value_text='a quick brown fox',
    ))
    lb = Leaderboard(name='itsi_lb', visibility='public', owner_user_id=user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    bhc = Client(token='itsitok',
                 transport=FlaskTestClientTransport(client))
    out = list(bhc.iter_samples(lb.id))
    _, inputs = out[0]
    assert inputs['caption'] == 'a quick brown fox'


def test_iter_samples_yields_bh_mask_from_raw_class_index_png(client, db_session):
    """A mask input packs the raw class-index PNG (mode L) in the bulk
    archive, so the client decodes a real bh.Mask — not the palette-RGB
    bh.Image the per-URL serve route would have produced."""
    user = User(email='itsm@bench.local', display_name='itsm',
                oauth_provider='github', oauth_sub='itsm-1',
                api_token='itsmtok')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='itsm_ds', visibility='public', owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='seg',
                                kind='mask', role='input'))
    db.session.add(DatasetField(dataset_id=ds.id, name='label',
                                kind='label', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    seg_dir = os.path.join(flask_app.config['UPLOAD_FOLDER'],
                           'datasets', str(ds.id), 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    rel = os.path.join('datasets', str(ds.id), 'seg', 's0.png')
    abs_path = os.path.join(flask_app.config['UPLOAD_FOLDER'], rel)
    # Mode-L class-index map: a 8x8 block of 0s and 1s.
    cls = np.zeros((8, 8), dtype=np.uint8)
    cls[4:, 4:] = 1
    PILImage.fromarray(cls, mode='L').save(abs_path)
    db.session.add(CustomField(
        sample_id=s.id, name='seg', data_type='mask', value_text=rel,
    ))
    lb = Leaderboard(name='itsm_lb', visibility='public', owner_user_id=user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    bhc = Client(token='itsmtok',
                 transport=FlaskTestClientTransport(client))
    _, inputs = list(bhc.iter_samples(lb.id))[0]
    seg = inputs['seg']
    assert isinstance(seg, bh.Mask), f"expected bh.Mask, got {type(seg).__name__}"
    assert seg.array.shape == (8, 8)
    assert set(np.unique(seg.array).tolist()) == {0, 1}


class _CountingTransport(FlaskTestClientTransport):
    """Wraps the test transport to count bulk-archive downloads."""

    def __init__(self, test_client):
        super().__init__(test_client)
        self.archive_downloads = 0
        self.per_sample_fetches = 0

    def download_inputs_archive(self, leaderboard_id, dest_path, token=None):
        self.archive_downloads += 1
        return super().download_inputs_archive(leaderboard_id, dest_path, token)

    def fetch_bytes(self, url, token=None):
        self.per_sample_fetches += 1
        return super().fetch_bytes(url, token)


def test_iter_samples_downloads_bulk_archive_once_and_caches(client, db_session):
    """First iter_samples downloads the bulk ZIP once (not per-sample);
    a second call reads from the on-disk cache with zero downloads."""
    _, lb, _ = _seed_image_lb(client)
    tr = _CountingTransport(client)
    bhc = Client(token='itstok', transport=tr)

    first = list(bhc.iter_samples(lb.id))
    assert len(first) == 1
    assert tr.archive_downloads == 1          # one bulk download
    assert tr.per_sample_fetches == 0         # NOT N per-sample GETs

    second = list(bhc.iter_samples(lb.id))
    assert len(second) == 1
    assert tr.archive_downloads == 1          # cache hit — no re-download
    # Bytes still decode to the same typed instance.
    assert isinstance(second[0][1]['img'], bh.Image)


def test_force_download_busts_the_cache(client, db_session):
    """force_download=True re-fetches the bulk archive even when a
    valid cache exists."""
    _, lb, _ = _seed_image_lb(client)
    tr = _CountingTransport(client)
    bhc = Client(token='itstok', transport=tr)

    list(bhc.iter_samples(lb.id))
    assert tr.archive_downloads == 1
    list(bhc.iter_samples(lb.id, force_download=True))
    assert tr.archive_downloads == 2          # forced re-download


def test_iter_samples_falls_back_to_per_sample_when_no_archive(client, db_session):
    """If the bulk archive route is unavailable (older server), the
    client transparently falls back to per-sample fetches."""
    _, lb, _ = _seed_image_lb(client)
    tr = _CountingTransport(client)

    def _no_archive(leaderboard_id, dest_path, token=None):
        from benchhub.client import BenchHubAPIError
        raise BenchHubAPIError(404, {"error": "no such route"})

    tr.download_inputs_archive = _no_archive  # type: ignore[assignment]
    bhc = Client(token='itstok', transport=tr)
    out = list(bhc.iter_samples(lb.id))
    assert len(out) == 1
    assert isinstance(out[0][1]['img'], bh.Image)
    assert tr.per_sample_fetches == 1          # legacy path kicked in


def test_samples_api_honors_lb_field_role_override(client, db_session):
    """A field declared role='gt' on the dataset but overridden to
    'input' via Leaderboard.field_roles_json is treated as an input by
    the /samples API (regression: cifar100's `img` was gt on the
    dataset)."""
    import json as _json
    import os as _os
    import numpy as _np
    from PIL import Image as _PILImage
    user = User(email='roleov@bench.local', display_name='ro',
                oauth_provider='github', oauth_sub='roleov-1', api_token='rotok')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='roleov_ds', visibility='public', owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    # img declared as gt on the dataset (like cifar100).
    db.session.add(DatasetField(dataset_id=ds.id, name='img', kind='image', role='gt'))
    db.session.add(DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0'); db.session.add(s); db.session.flush()
    img_dir = _os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id), 'img')
    _os.makedirs(img_dir, exist_ok=True)
    rel = _os.path.join('datasets', str(ds.id), 'img', 's0.png')
    _PILImage.fromarray(_np.zeros((8, 8, 3), dtype=_np.uint8)).save(
        _os.path.join(flask_app.config['UPLOAD_FOLDER'], rel))
    db.session.add(CustomField(sample_id=s.id, name='img', data_type='image', value_text=rel))
    lb = Leaderboard(name='roleov_lb', visibility='public', owner_user_id=user.id,
                     field_roles_json=_json.dumps({'img': 'input', 'label': 'gt'}))
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    payload = client.get(f'/api/leaderboard/{lb.id}/samples').get_json()
    assert [f['name'] for f in payload['input_fields']] == ['img']


def test_samples_api_defaults_image_to_input_when_no_explicit_input(client, db_session):
    """When no field is marked input (and no LB override), an image field
    is treated as the input by default — 'image present => input'."""
    import os as _os
    import numpy as _np
    from PIL import Image as _PILImage
    user = User(email='imgdef@bench.local', display_name='id',
                oauth_provider='github', oauth_sub='imgdef-1', api_token='idtok')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='imgdef_ds', visibility='public', owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    # Everything gt; no input, no field_roles override.
    db.session.add(DatasetField(dataset_id=ds.id, name='image', kind='image', role='gt'))
    db.session.add(DatasetField(dataset_id=ds.id, name='caption', kind='text', role='gt'))
    s = Sample(dataset_id=ds.id, name='s0'); db.session.add(s); db.session.flush()
    img_dir = _os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id), 'image')
    _os.makedirs(img_dir, exist_ok=True)
    rel = _os.path.join('datasets', str(ds.id), 'image', 's0.png')
    _PILImage.fromarray(_np.zeros((8, 8, 3), dtype=_np.uint8)).save(
        _os.path.join(flask_app.config['UPLOAD_FOLDER'], rel))
    db.session.add(CustomField(sample_id=s.id, name='image', data_type='image', value_text=rel))
    lb = Leaderboard(name='imgdef_lb', visibility='public', owner_user_id=user.id)
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    payload = client.get(f'/api/leaderboard/{lb.id}/samples').get_json()
    assert [f['name'] for f in payload['input_fields']] == ['image']
