"""Dynamic data types: register a kind (storage + sandboxed visualize),
the bytes-arg sandbox round-trip, and the public-LB dependency guard."""
import sys
from pathlib import Path

import pytest

import os

import benchhub as bh
import app as _app
from app import (CustomField, DataTypeDef, Dataset, DatasetField, Leaderboard,
                 Sample, User, db, generate_api_token)
from metric_engine import _jsonify_kwarg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'runner'))
import harness  # noqa: E402


@pytest.fixture
def api_user(db_session):
    u = User(email='dt@bench.local', display_name='dt', oauth_provider='github',
             oauth_sub='dt-1', is_admin=False, api_token=generate_api_token())
    db.session.add(u); db.session.commit()
    return u


def _client(client, user):
    return bh.Client(token=user.api_token, base_url="http://test",
                     transport=bh.FlaskTestClientTransport(client))


# --- bytes cross the sandbox JSON boundary (for visualize(blob, params)) ---

def test_bytes_kwarg_round_trips_through_harness():
    enc = _jsonify_kwarg(b'\x00\x01rawbytes')
    assert '__bytes__' in enc
    assert harness._decode_arg(enc) == b'\x00\x01rawbytes'


def test_dtype_visualize_job_renders_png_from_blob():
    code = ("def visualize(blob, params):\n"
            "    from PIL import Image\n"
            "    n = len(blob)\n"
            "    return Image.new('RGB', (max(1, n), 4), (n % 256, 0, 0))\n")
    out = harness.run_job({
        'kind': 'visualization', 'code': code, 'function_name': 'visualize',
        'kwargs_list': [{'blob': _jsonify_kwarg(b'abcd'), 'params': {'k': 1}}],
    })
    r = out['results'][0]
    assert r['error'] is None
    import base64
    assert base64.b64decode(r['png_b64'])[:8] == b'\x89PNG\r\n\x1a\n'


# --- register API + client.create_datatype --------------------------------

def test_create_datatype(client, api_user):
    c = _client(client, api_user)
    def visualize(blob, params):
        from PIL import Image
        return Image.new('RGB', (8, 8))
    res = c.create_datatype('volume', file_ext='.nii.gz', visualize_code=visualize,
                            description='3D medical volume')
    assert res['name'] == 'volume' and res['file_ext'] == '.nii.gz'
    assert res['visibility'] == 'private'
    dt = DataTypeDef.query.filter_by(name='volume').first()
    assert dt.owner_user_id == api_user.id and 'def visualize' in dt.visualize_code


def test_create_datatype_rejects_builtin_and_dupes(client, api_user):
    c = _client(client, api_user)
    with pytest.raises(bh.BenchHubAPIError) as e1:
        c.create_datatype('image')          # built-in kind
    assert e1.value.status_code == 409
    c.create_datatype('pointcloud', file_ext='.ply')
    with pytest.raises(bh.BenchHubAPIError) as e2:
        c.create_datatype('pointcloud')     # duplicate name
    assert e2.value.status_code == 409
    with pytest.raises(bh.BenchHubAPIError) as e3:
        c.create_datatype('Bad Name!')      # invalid format
    assert e3.value.status_code == 400


# --- guard: dtype used by a public LB can't be deleted / made private ------

def _setup_used_dtype(owner, *, lb_public):
    dt = DataTypeDef(name='volume', owner_user_id=owner.id, visibility='public',
                     file_ext='.nii.gz')
    ds = Dataset(name='volds', owner_user_id=owner.id, visibility='public')
    db.session.add_all([dt, ds]); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='scan', kind='volume', role='input'))
    lb = Leaderboard(name='vol-lb', summary_metrics='', owner_user_id=owner.id,
                     visibility='public' if lb_public else 'private')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return dt


def test_dtype_delete_blocked_when_used_by_public_lb(client, db_session):
    a = User(email='o@x.io', display_name='o', oauth_provider='github', oauth_sub='o1')
    db.session.add(a); db.session.commit()
    dt = _setup_used_dtype(a, lb_public=True)
    with client.session_transaction() as s:
        s['user_id'] = a.id
    client.post(f'/datatypes/{dt.id}/delete')
    assert db.session.get(DataTypeDef, dt.id) is not None          # not deleted
    # downgrade also blocked
    client.post(f'/datatypes/{dt.id}/visibility', data={'visibility': 'private'})
    assert db.session.get(DataTypeDef, dt.id).visibility == 'public'


def test_dtype_delete_allowed_when_only_private_lb(client, db_session):
    a = User(email='o2@x.io', display_name='o2', oauth_provider='github', oauth_sub='o2')
    db.session.add(a); db.session.commit()
    dt = _setup_used_dtype(a, lb_public=False)
    with client.session_transaction() as s:
        s['user_id'] = a.id
    client.post(f'/datatypes/{dt.id}/delete')
    assert db.session.get(DataTypeDef, dt.id) is None              # deleted


# --- end-to-end: render route ---------------------------------------------

def test_serve_custom_field_image_renders_registered_dtype(client, db_session,
                                                           tmp_path, monkeypatch):
    """A registered-kind CustomField is served as an image (rendered via the
    sandboxed visualize; falls back to an error-image PNG if the sandbox
    can't run here) — proving the dispatch branch, not a 400."""
    monkeypatch.setitem(_app.app.config, 'UPLOAD_FOLDER', str(tmp_path))
    db.session.add(DataTypeDef(
        name='volume', visibility='public', file_ext='.bin',
        visualize_code=("def visualize(blob, params):\n"
                        "    from PIL import Image\n"
                        "    return Image.new('RGB', (4, 4))\n")))
    ds = Dataset(name='vd', visibility='public'); db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0'); db.session.add(s); db.session.flush()
    rel = f'datasets/{ds.id}/volume/s0.bin'
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'wb') as fh:
        fh.write(b'rawbytes')
    cf = CustomField(sample_id=s.id, name='volume', data_type='volume', value_text=rel)
    db.session.add(cf); db.session.commit()
    r = client.get(f'/custom_field_image/{cf.id}')
    assert r.status_code == 200 and r.mimetype == 'image/png'


def test_register_datatype_via_web_form(client, db_session):
    u = User(email='w@x.io', display_name='w', oauth_provider='github', oauth_sub='w1')
    db.session.add(u); db.session.commit()
    with client.session_transaction() as s:
        s['user_id'] = u.id
    # the register form lives on the unified /supported_types page
    # (/datatypes now redirects there).
    assert client.get('/datatypes').status_code == 302
    body = client.get('/supported_types').data.decode()
    assert 'Register a data type' in body and 'visualize_code' in body
    # submit the form
    client.post('/datatypes/create', data={
        'name': 'pointcloud', 'file_ext': '.ply', 'description': 'xyz',
        'visualize_code': ("def visualize(blob, params):\n"
                            "    from PIL import Image\n"
                            "    return Image.new('RGB', (4, 4))\n")})
    dt = DataTypeDef.query.filter_by(name='pointcloud').first()
    assert dt is not None and dt.owner_user_id == u.id and dt.file_ext == '.ply'
    # bad name is rejected (no row), with a flash
    client.post('/datatypes/create', data={'name': 'Bad Name!'})
    assert DataTypeDef.query.filter_by(name='Bad Name!').first() is None
