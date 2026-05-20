"""Admin route tests for /admin/import_from_hf — auth + Croissant preview.

The /commit handler talks to `datasets.load_dataset()` and isn't tested
here; its materialiser logic is covered separately."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import Dataset, DatasetField, User, db


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def admin_user(db_session):
    u = User(
        email='hfadmin@bench.local', display_name='hf-admin',
        oauth_provider='github', oauth_sub='hfadmin-1',
        is_admin=True,
    )
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def admin_client(client, admin_user):
    with client.session_transaction() as sess:
        sess['user_id'] = admin_user.id
    return client


def test_get_import_form_admin_only(client, db_session):
    """Unauthenticated → 302 to login; non-admin → 403."""
    r = client.get('/admin/import_from_hf')
    assert r.status_code == 302  # login redirect
    other = User(email='nope@bench.local', display_name='no',
                 oauth_provider='github', oauth_sub='no-1', is_admin=False)
    db.session.add(other); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = other.id
    r = client.get('/admin/import_from_hf')
    assert r.status_code == 403


def test_get_import_form_renders_for_admin(admin_client):
    r = admin_client.get('/admin/import_from_hf')
    assert r.status_code == 200
    assert b'Import from HuggingFace' in r.data
    assert b'repo_id' in r.data


def test_preview_renders_partial_form_from_fixture(admin_client, monkeypatch):
    """Stub fetch_croissant to return a known fixture, then assert the
    preview template renders every field as an editable row with the
    parsed kind pre-selected."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    # Stub the split-count fetch too so the test doesn't touch the network.
    monkeypatch.setattr(hfs, 'fetch_split_row_counts',
                        lambda repo_id, **kw: {"train": 50000, "test": 10000})

    r = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'uoft-cs/cifar10'},
    )
    assert r.status_code == 200
    body = r.data.decode()
    # Repo id surfaces in the form action target.
    assert 'uoft-cs/cifar10' in body
    # Both real fields appear as rows.
    assert 'img' in body
    assert 'label' in body
    # Kind selects exist (one per field × 9 kinds, so plenty of <option> tags).
    assert body.count('name="field_kind"') >= 2
    # Hidden field_source_column tracks the HF column name for the
    # commit step's row-value lookup.
    assert 'name="field_source_column"' in body
    # Splits dropdown — at least one option present.
    assert 'name="split"' in body
    # Per-split row counts render in the option labels + as data-row-count.
    assert 'data-row-count="10000"' in body
    assert '(10,000)' in body
    assert 'data-row-count="50000"' in body
    assert '(50,000)' in body


def test_preview_role_dropdowns_default_to_disabled_placeholder(
    admin_client, monkeypatch,
):
    """The preview form must force an explicit role pick — every
    `name="field_role"` <select> renders a disabled placeholder as
    the default option, no role-name is selected, and the select is
    HTML5-required so the browser refuses to submit until the admin
    chooses one. Pred is NOT in the column dropdown (pred fields
    are a separate section, not a re-labelling of an HF column)."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts',
                        lambda repo_id, **kw: {"train": 50000, "test": 10000})

    r = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'uoft-cs/cifar10'},
    )
    assert r.status_code == 200
    body = r.data.decode()
    # The role select is `required` and carries a disabled+selected
    # placeholder so unmodified submits get blocked.
    assert 'name="field_role" class="form-select form-select-sm" required' in body
    assert 'value="" selected disabled hidden' in body
    # Only input / gt / skip in the per-column dropdown; pred lives
    # in its own section below.
    for role in ('input', 'gt', 'skip'):
        assert f'<option value="{role}" selected>' not in body
        assert f'<option value="{role}">' in body
    assert '<option value="pred"' not in body


def test_preview_renders_separate_pred_fields_section(admin_client, monkeypatch):
    """The preview page renders an empty Prediction-fields table +
    an Add button + the cloneable <template> row carrying name/kind/
    params inputs. Pred fields go into THIS section, never as a
    re-labelled HF column."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})

    r = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'uoft-cs/cifar10'},
    )
    body = r.data.decode()
    # Empty body, the template, and the Add button — all need to be present.
    assert 'id="pred-fields-body"' in body
    assert 'id="pred-row-template"' in body
    assert 'id="pred-add-row"' in body
    assert 'name="pred_field_name"' in body
    assert 'name="pred_field_kind"' in body
    assert 'name="pred_field_params"' in body


def test_commit_route_collects_pred_fields_into_selected(client, db_session, tmp_path, monkeypatch):
    """The commit handler merges the pred_field_* arrays into the
    list it hands to materialize_hf_to_typed_dir, tagged role='pred'.
    Stub the materialiser + importer to capture what flows through
    without touching the network."""
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, 'UPLOAD_FOLDER', str(tmp_path))
    admin = User(email='predsec@bench.local', display_name='ps',
                 oauth_provider='github', oauth_sub='ps-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id

    captured = {}
    import benchhub.hf_materialize as hfm

    def _fake_materialize(repo_id, *, split, sample_cap, staging_dir,
                          dataset_name, fields, **kw):
        captured['fields'] = fields
        # Write a manifest the importer can later read.
        from pathlib import Path
        Path(staging_dir, 'manifest.json').write_text(json.dumps({
            'name': dataset_name, 'version': '1.0',
            'fields': [
                {'name': f['name'], 'kind': f['kind'],
                 'role': f['role'], 'params': f.get('params') or {}}
                for f in fields
            ],
            'samples': ['s0'],
        }))
        return {'samples': 1, 'fields': len(fields),
                'rows_written': 0, 'rows_skipped': 0}

    monkeypatch.setattr(hfm, 'materialize_hf_to_typed_dir', _fake_materialize)

    # Also stub import_typed_dataset since we don't have real files on disk.
    from benchhub import manifest as bh_manifest

    def _fake_import(source_root, *, db_session, Dataset, Sample, CustomField,
                     DatasetField=None, upload_folder, owner_user_id=None,
                     visibility='public'):
        # Read the staged manifest and create the DatasetField rows.
        from pathlib import Path
        m = json.loads(Path(source_root, 'manifest.json').read_text())
        ds = Dataset(name=m['name'], owner_user_id=owner_user_id,
                     visibility=visibility)
        db_session.add(ds); db_session.flush()
        if DatasetField is not None:
            for f in m['fields']:
                df = DatasetField(dataset_id=ds.id, name=f['name'],
                                  kind=f['kind'], role=f.get('role', 'gt'))
                if f.get('params'):
                    df.set_params(f['params'])
                db_session.add(df)
        db_session.flush()
        return ds.id, {
            'dataset_id': ds.id, 'name': m['name'],
            'samples': len(m['samples']), 'fields': len(m['fields']),
            'custom_field_rows': 0, 'files_copied': 0, 'bytes_on_disk': 0,
        }

    monkeypatch.setattr(bh_manifest, 'import_typed_dataset', _fake_import)

    r = client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'uoft-cs/cifar10',
        'dataset_name': 'wp-test',
        'split': 'test',
        'sample_cap': '1',
        'sampling': 'head',
        'sampling_seed': '0',
        # One column kept as input + the pred section adds one role=pred.
        'field_name': ['img'],
        'field_source_column': ['img'],
        'field_kind': ['image'],
        'field_role': ['input'],
        'field_params': [''],
        'pred_field_name': ['depth_pred'],
        'pred_field_kind': ['depth'],
        'pred_field_params': ['{"shape": [16, 24]}'],
    }, follow_redirects=False)
    assert r.status_code == 302

    # The materialiser saw both fields, with the pred tagged role='pred'.
    by_name = {f['name']: f for f in captured['fields']}
    assert by_name['img']['role'] == 'input'
    assert by_name['depth_pred']['role'] == 'pred'
    assert by_name['depth_pred']['params'] == {'shape': [16, 24]}
    # And the resulting Dataset has the DatasetField row.
    ds = Dataset.query.filter_by(name='wp-test').one()
    schema = {f.name: f for f in
              __import__('app').DatasetField.query.filter_by(dataset_id=ds.id).all()}
    assert schema['depth_pred'].role == 'pred'
    assert schema['depth_pred'].get_params() == {'shape': [16, 24]}


def test_commit_route_rejects_pred_field_colliding_with_column(client, db_session, monkeypatch):
    """A pred field name that duplicates an imported column name
    is a contract bug — bounce with a flash."""
    admin = User(email='collide@bench.local', display_name='c',
                 oauth_provider='github', oauth_sub='c-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id

    r = client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'x/y', 'dataset_name': 'd', 'split': 'test', 'sample_cap': '1',
        'sampling': 'head',
        'field_name': ['label'],
        'field_source_column': ['label'],
        'field_kind': ['label'],
        'field_role': ['gt'],
        'field_params': [''],
        'pred_field_name': ['label'],  # ← collision with column
        'pred_field_kind': ['label'],
        'pred_field_params': [''],
    }, follow_redirects=False)
    assert r.status_code == 302
    assert '/admin/import_from_hf' in r.headers['Location']


def test_commit_route_rejects_blank_roles(client, db_session):
    """JS-disabled browser or bookmarked POST can still submit with
    an empty role string. Server-side check bounces with a clear
    flash listing the affected field names."""
    admin = User(email='blank-role@bench.local', display_name='br',
                 oauth_provider='github', oauth_sub='br-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id

    r = client.post(
        '/admin/import_from_hf/commit',
        data={
            'repo_id': 'uoft-cs/cifar10',
            'dataset_name': 'unused',
            'split': 'test',
            'sample_cap': '5',
            'sampling': 'head',
            'field_name': ['img', 'label'],
            'field_source_column': ['img', 'label'],
            'field_kind': ['image', 'scalar'],
            'field_role': ['gt', ''],       # one missing
            'field_params': ['', ''],
        },
        follow_redirects=False,
    )
    assert r.status_code == 302
    # Redirects back to the form (not the preview).
    assert '/admin/import_from_hf' in r.headers['Location']


def test_preview_404_when_croissant_fetch_fails(admin_client, monkeypatch):
    from benchhub import hf_croissant as hfc

    def _boom(repo_id, **kw):
        raise hfc.CroissantFetchError("no such repo")

    monkeypatch.setattr(hfc, 'fetch_croissant', _boom)
    r = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'private/secret'},
        follow_redirects=False,
    )
    # On error we flash + redirect back to the form, not 5xx.
    assert r.status_code == 302
    assert '/admin/import_from_hf' in r.headers['Location']


def test_preview_redirects_when_repo_id_missing(admin_client):
    r = admin_client.post('/admin/import_from_hf/preview', data={})
    assert r.status_code == 302
    assert '/admin/import_from_hf' in r.headers['Location']


# ---------------------------------------------------------------------------
# Suggestion endpoints (search + trending)
# ---------------------------------------------------------------------------

def test_search_route_admin_only(client, db_session):
    """Unauthenticated → 302 to login; non-admin → 403."""
    r = client.get('/admin/import_from_hf/search?q=cifar')
    assert r.status_code == 302
    other = User(email='regular@bench.local', display_name='reg',
                 oauth_provider='github', oauth_sub='reg-search-1', is_admin=False)
    db.session.add(other); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = other.id
    r = client.get('/admin/import_from_hf/search?q=cifar')
    assert r.status_code == 403


def test_search_route_returns_normalised_json(admin_client, monkeypatch):
    """Stub the HF Hub fetch to return two records; the route should
    pass them through `_normalize` and serve as JSON."""
    from benchhub import hf_search

    def _fake_search(q, *, limit=10):
        assert q == 'cifar'
        return [
            {"id": "uoft-cs/cifar10", "downloads": 100, "likes": 5,
             "description": "", "gated": False},
            {"id": "uoft-cs/cifar100", "downloads": 50, "likes": 2,
             "description": "", "gated": False},
        ]
    monkeypatch.setattr(hf_search, 'search_datasets', _fake_search)

    r = admin_client.get('/admin/import_from_hf/search?q=cifar')
    assert r.status_code == 200
    body = r.get_json()
    assert [d['id'] for d in body] == ['uoft-cs/cifar10', 'uoft-cs/cifar100']


def test_search_route_empty_query_returns_empty_array(admin_client):
    """No upstream call should fire — the helper short-circuits on
    empty input and the route just relays."""
    r = admin_client.get('/admin/import_from_hf/search?q=')
    assert r.status_code == 200
    assert r.get_json() == []


def test_trending_route_admin_only(client, db_session):
    r = client.get('/admin/import_from_hf/trending')
    assert r.status_code == 302
    other = User(email='regular2@bench.local', display_name='reg2',
                 oauth_provider='github', oauth_sub='reg-trending-1', is_admin=False)
    db.session.add(other); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = other.id
    r = client.get('/admin/import_from_hf/trending')
    assert r.status_code == 403


def test_trending_route_returns_grouped_json(admin_client, monkeypatch):
    """Stub the trending helper to return a fixed shape; route should
    serialise it as JSON without rearranging keys."""
    from benchhub import hf_search

    def _fake_trending(*, limit_per_domain=5):
        return {
            "Vision": [{"id": "v/x", "downloads": 1, "likes": 0,
                        "description": "", "gated": False}],
            "NLP":    [],
            "Audio":  [],
            "Tabular": [],
        }
    monkeypatch.setattr(hf_search, 'trending_by_domain', _fake_trending)

    r = admin_client.get('/admin/import_from_hf/trending')
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"Vision", "NLP", "Audio", "Tabular"}
    assert body["Vision"][0]["id"] == "v/x"
