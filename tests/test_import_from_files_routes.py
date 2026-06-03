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
