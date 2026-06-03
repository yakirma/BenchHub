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
