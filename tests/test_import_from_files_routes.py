"""Routes for the user-declared file-tree importer."""
import json
import sys
import types

import pytest

from app import Dataset, User, db


@pytest.fixture
def user_client(client, db_session):
    u = User(email='ft@bench.local', display_name='ft',
             oauth_provider='github', oauth_sub='ft-1', is_admin=False)
    db.session.add(u); db.session.commit()
    with client.session_transaction() as s:
        s['user_id'] = u.id
    return client, u


def _stub_hfapi(monkeypatch, files):
    """Stub huggingface_hub.HfApi so list_repo_files returns `files`."""
    mod = types.ModuleType('huggingface_hub')

    class _Api:
        def list_repo_files(self, repo_id, **kw):
            return files
    mod.HfApi = _Api
    mod.hf_hub_download = lambda *a, **k: '/nonexistent'
    monkeypatch.setitem(sys.modules, 'huggingface_hub', mod)


def test_entry_page_requires_login(client, db_session):
    r = client.get('/import_from_files', follow_redirects=False)
    assert r.status_code == 302 and '/login' in r.headers['Location']


def test_inspect_renders_mapping_builder(user_client, monkeypatch):
    client, _ = user_client
    files = [f'train/i_0/normal/{i}.png' for i in range(5)] + \
            ['train/i_0/normal/depth.npz']
    _stub_hfapi(monkeypatch, files)
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_card', lambda r, **k: {})
    r = client.post('/import_from_files/inspect', data={'repo_id': 'a/b'})
    assert r.status_code == 200
    body = r.data.decode()
    assert 'Map the modalities' in body
    assert 'png' in body                 # ext histogram
    assert 'Add modality' in body
    assert 'map-form-errors' in body     # client-side validation hook
    assert 'validateRows' in body
    assert 'draft-banner' in body          # draft autosave/restore
    assert 'beforeunload' in body


def test_commit_enqueues_and_creates_dataset(user_client, monkeypatch):
    client, u = user_client
    files = [f'train/i_0/normal/{i}.png' for i in range(3)]
    _stub_hfapi(monkeypatch, files)

    import tasks as _tasks
    calls = {}
    monkeypatch.setattr(_tasks.run_file_tree_import, 'delay',
                        lambda **kw: calls.update(kw) or types.SimpleNamespace(id='t1'))

    r = client.post('/import_from_files/commit', data={
        'repo_id': 'a/b', 'dataset_name': 'eccv',
        'field_name': ['image'], 'field_kind': ['image'],
        'field_role': ['input'], 'field_loader': ['file'],
        'field_pattern': ['train/{seq}/normal/{id}.png'],
        'field_key': [''], 'field_shared': ['0'], 'field_axis': ['0'],
    }, follow_redirects=False)
    assert r.status_code == 302
    ds = Dataset.query.filter_by(name='eccv').first()
    assert ds is not None and ds.visibility == 'private'
    assert ds.import_status == 'importing'
    # Spec threaded to the task.
    assert calls['spec'][0]['pattern'] == 'train/{seq}/normal/{id}.png'
    assert calls['sample_cap'] == -1


def test_commit_rejects_unresolvable_pattern(user_client, monkeypatch):
    client, _ = user_client
    _stub_hfapi(monkeypatch, ['train/i_0/normal/0.png'])
    before = Dataset.query.count()
    r = client.post('/import_from_files/commit', data={
        'repo_id': 'a/b', 'dataset_name': 'x',
        'field_name': ['image'], 'field_kind': ['image'],
        'field_role': ['input'], 'field_loader': ['file'],
        'field_pattern': ['nope/{id}.png'],
        'field_key': [''], 'field_shared': ['0'], 'field_axis': ['0'],
    }, follow_redirects=False)
    assert r.status_code == 302
    assert Dataset.query.count() == before  # nothing created


def test_commit_variant_fanout_creates_one_dataset_per_value(user_client, monkeypatch):
    """A variant_token splits the import into one dataset per distinct
    token value, each with a token_filter."""
    client, u = user_client
    files = [f'train/i_0/{q}/{i}.png' for q in ('low', 'normal') for i in range(2)]
    _stub_hfapi(monkeypatch, files)

    import tasks as _tasks
    enq = []
    monkeypatch.setattr(_tasks.run_file_tree_import, 'delay',
                        lambda **kw: enq.append(kw) or types.SimpleNamespace(id='t'))

    r = client.post('/import_from_files/commit', data={
        'repo_id': 'a/b', 'dataset_name': 'eccv', 'variant_token': 'quality',
        'field_name': ['image'], 'field_kind': ['image'],
        'field_role': ['input'], 'field_loader': ['file'],
        'field_pattern': ['train/{seq}/{quality}/{id}.png'],
        'field_key': [''], 'field_shared': ['0'], 'field_axis': ['0'],
        'field_pointer': [''], 'field_column': [''], 'field_id_column': [''],
    }, follow_redirects=False)
    assert r.status_code == 302
    names = sorted(d.name for d in Dataset.query.all())
    assert names == ['eccv_low', 'eccv_normal']
    filters = sorted(json.dumps(k['token_filter']) for k in enq)
    assert filters == [json.dumps({'quality': 'low'}),
                       json.dumps({'quality': 'normal'})]


def test_commit_json_csv_spec_parsed(user_client, monkeypatch):
    """json pointer + csv column/id_column survive the form round-trip."""
    client, u = user_client
    files = ['seq/0.png', 'seq/manifest.json', 'seq/meta.csv']
    _stub_hfapi(monkeypatch, files)
    import tasks as _tasks
    enq = {}
    monkeypatch.setattr(_tasks.run_file_tree_import, 'delay',
                        lambda **kw: enq.update(kw) or types.SimpleNamespace(id='t'))

    r = client.post('/import_from_files/commit', data={
        'repo_id': 'a/b', 'dataset_name': 'm',
        'field_name': ['image', 'pose', 'score'],
        'field_kind': ['image', 'json', 'scalar'],
        'field_role': ['input', 'gt', 'gt'],
        'field_loader': ['file', 'json', 'csv'],
        'field_pattern': ['seq/{id}.png', 'seq/manifest.json', 'seq/meta.csv'],
        'field_key': ['', '', ''],
        'field_shared': ['0', '1', '0'],
        'field_axis': ['0', '0', '0'],
        'field_pointer': ['', 'frames.{id}.pose', ''],
        'field_column': ['', '', 'score'],
        'field_id_column': ['', '', 'id'],
    }, follow_redirects=False)
    assert r.status_code == 302
    spec = enq['spec']
    assert spec[1]['loader'] == 'json' and spec[1]['pointer'] == 'frames.{id}.pose'
    assert spec[1]['shared'] is True
    assert spec[2]['loader'] == 'csv' and spec[2]['column'] == 'score'
    assert spec[2]['id_column'] == 'id'


def test_inspect_accepts_get_with_repo_id(user_client, monkeypatch):
    """The tabular importer hands off here via GET ?repo_id=."""
    client, _ = user_client
    _stub_hfapi(monkeypatch, ['seq/0.png', 'seq/1.png'])
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_card', lambda r, **k: {})
    r = client.get('/import_from_files/inspect?repo_id=a/b')
    assert r.status_code == 200
    assert 'Map the modalities' in r.data.decode()


def test_tabular_preview_failure_redirects_to_file_tree(user_client, monkeypatch):
    """When no Croissant/info schema exists, the tabular preview hands off
    to the file-tree importer for the same repo."""
    client, _ = user_client
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs
    monkeypatch.setattr(hfc, 'fetch_croissant',
                        lambda r, **k: (_ for _ in ()).throw(hfc.CroissantFetchError('none')))
    monkeypatch.setattr(hfs, 'fetch_dataset_info', lambda r, **k: None)
    r = client.post('/admin/import_from_hf/preview', data={'repo_id': 'x/y'},
                    follow_redirects=False)
    assert r.status_code == 302
    assert '/import_from_files/inspect' in r.headers['Location']
    assert 'repo_id=x' in r.headers['Location']


def test_commit_token_loader_and_filter_parsed(user_client, monkeypatch):
    """The folder→label `token` loader (no pattern) + a single-value
    subset filter survive the form round-trip into the task call."""
    client, u = user_client
    files = [f'Alex_Brush/{i}.png' for i in range(2)] + \
            [f'Cookie/{i}.png' for i in range(2)]
    _stub_hfapi(monkeypatch, files)
    import tasks as _tasks
    enq = {}
    monkeypatch.setattr(_tasks.run_file_tree_import, 'delay',
                        lambda **kw: enq.update(kw) or types.SimpleNamespace(id='t'))

    r = client.post('/import_from_files/commit', data={
        'repo_id': 'Benjy/sig', 'dataset_name': 'sig',
        'filter_token': 'split', 'filter_value': 'test',
        'field_name': ['image', 'font'],
        'field_kind': ['image', 'label'],
        'field_role': ['input', 'gt'],
        'field_loader': ['file', 'token'],
        'field_pattern': ['{cls}/{id}.png', ''],   # token row: blank pattern
        'field_key': ['', ''], 'field_shared': ['0', '0'],
        'field_axis': ['0', '0'], 'field_pointer': ['', ''],
        'field_column': ['', ''], 'field_id_column': ['', ''],
        'field_member': ['', ''], 'field_token': ['', 'cls'],
    }, follow_redirects=False)
    assert r.status_code == 302
    spec = enq['spec']
    assert spec[1]['loader'] == 'token' and spec[1]['token'] == 'cls'
    assert spec[1]['kind'] == 'label'
    assert enq['token_filter'] == {'split': 'test'}


def test_commit_zip_member_parsed(user_client, monkeypatch):
    client, u = user_client
    _stub_hfapi(monkeypatch, ['ids/0.txt', 'imgs.zip'])
    import tasks as _tasks
    enq = {}
    monkeypatch.setattr(_tasks.run_file_tree_import, 'delay',
                        lambda **kw: enq.update(kw) or types.SimpleNamespace(id='t'))
    r = client.post('/import_from_files/commit', data={
        'repo_id': 'a/b', 'dataset_name': 'z',
        'field_name': ['sid', 'image'],
        'field_kind': ['text', 'image'],
        'field_role': ['gt', 'input'],
        'field_loader': ['file', 'zip'],
        'field_pattern': ['ids/{id}.txt', 'imgs.zip'],
        'field_key': ['', ''], 'field_shared': ['0', '0'],
        'field_axis': ['0', '0'], 'field_pointer': ['', ''],
        'field_column': ['', ''], 'field_id_column': ['', ''],
        'field_member': ['', 'pics/{id}.png'], 'field_token': ['', ''],
    }, follow_redirects=False)
    assert r.status_code == 302
    assert enq['spec'][1]['loader'] == 'zip'
    assert enq['spec'][1]['member'] == 'pics/{id}.png'


def test_from_roles_generates_spec(user_client, monkeypatch):
    """The 'describe the structure' endpoint turns level roles into a
    field-row spec (modality fan-out + a label token field)."""
    client, _ = user_client
    files = [f'image/{i}.png' for i in range(2)] + \
            [f'depth/{i}.npz' for i in range(2)]
    _stub_hfapi(monkeypatch, files)
    r = client.post('/import_from_files/from_roles',
                    json={'repo_id': 'a/b', 'roles': ['modality', 'id']})
    assert r.status_code == 200
    spec = r.get_json()['spec']
    by = {f['name']: f for f in spec}
    assert set(by) == {'image', 'depth'}
    assert by['image']['pattern'] == 'image/{id}.png'
    assert by['depth']['loader'] == 'npz'


def test_inspect_renders_structure_panel(user_client, monkeypatch):
    client, _ = user_client
    files = [f'{c}/{i}.png' for c in ('Alex_Brush', 'Cookie') for i in range(2)]
    _stub_hfapi(monkeypatch, files)
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_card', lambda r, **k: {})
    body = client.post('/import_from_files/inspect', data={'repo_id': 'a/b'}).data.decode()
    assert 'Describe the structure' in body
    assert 'level-role' in body
    assert 'Generate fields from structure' in body


def test_sequence_cf_streams_video_and_renders(user_client, db_session, tmp_path, monkeypatch):
    """A sequence CustomField streams a video via /api/viz and the
    dataset_view renders an inline <video>."""
    import os
    import numpy as np
    from app import (app as flask_app, Dataset, Sample, DatasetField,
                     CustomField, db)
    from benchhub.types import Sequence, Image
    monkeypatch.setitem(flask_app.config, 'UPLOAD_FOLDER', str(tmp_path))
    client, u = user_client

    ds = Dataset(name='clipds', owner_user_id=u.id, visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='clip', kind='sequence', role='gt'))
    s = Sample(dataset_id=ds.id, name='c0'); db.session.add(s); db.session.flush()

    seq = Sequence([Image(np.full((8, 8, 3), i * 50, np.uint8)) for i in range(3)],
                   item_kind='image', fps=4)
    rel = f'datasets/{ds.id}/clip/c0.zip'
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'wb') as fh:
        fh.write(seq.encode())
    cf = CustomField(sample_id=s.id, name='clip', data_type='sequence', value_text=rel)
    cf.set_params({'item_kind': 'image', 'fps': 4})
    db.session.add(cf); db.session.commit()

    # /api/viz streams a clip (mp4 via ffmpeg, or GIF fallback).
    r = client.get(f'/api/viz/{cf.id}')
    assert r.status_code == 200
    assert r.content_type in ('video/mp4', 'image/gif')
    assert len(r.data) > 0

    # dataset_view renders an inline <video> for the sequence column.
    body = client.get(f'/dataset/{ds.id}').data.decode()
    assert '<video' in body
    assert f'/api/viz/{cf.id}' in body
