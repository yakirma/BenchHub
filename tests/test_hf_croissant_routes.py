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


def test_get_import_form_requires_login_open_to_users(client, db_session):
    """Self-service HF caching: unauthenticated → 302 to login; any
    signed-in user (not just admins) → 200."""
    r = client.get('/admin/import_from_hf')
    assert r.status_code == 302  # login redirect
    other = User(email='nope@bench.local', display_name='no',
                 oauth_provider='github', oauth_sub='no-1', is_admin=False)
    db.session.add(other); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = other.id
    r = client.get('/admin/import_from_hf')
    assert r.status_code == 200


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


def test_preview_renders_use_as_name_badge_on_text_field_rows(
    admin_client, monkeypatch,
):
    """Each text-kind field row gets a small "use as name" badge
    in the field cell. Clicking it sets the hidden
    `sample_name_from` tracker to that column's source name.
    Non-text rows render the same button but `hidden`, so the
    eligibility-tracking JS can flip it visible when the admin
    changes a row's kind to text."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = {
        "@type": "sc:Dataset", "name": "synth",
        "recordSet": [{
            "@id": "rs", "@type": "cr:RecordSet",
            "field": [
                {"@id": "rs/img", "@type": "cr:Field", "dataType": "sc:ImageObject",
                 "source": {"extract": {"column": "img"}}},
                {"@id": "rs/caption", "@type": "cr:Field", "dataType": "sc:Text",
                 "source": {"extract": {"column": "caption"}}},
            ],
        }],
    }
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})

    body = admin_client.post('/admin/import_from_hf/preview',
                             data={'repo_id': 'x/y'}).data.decode()
    # Hidden tracker present, starts empty.
    assert 'id="sample-name-from" name="sample_name_from" value=""' in body
    # Per-row badge on the text column row, eligible (not hidden).
    import re
    text_badge = re.search(
        r'<button[^>]*class="[^"]*sample-name-toggle[^"]*"[^>]*data-source-column="caption"[^>]*>',
        body,
    )
    assert text_badge is not None, 'expected a sample-name-toggle on the caption row'
    assert ' hidden' not in text_badge.group(0)
    # Non-text image row also has a badge in markup, but `hidden`.
    img_badge = re.search(
        r'<button[^>]*class="[^"]*sample-name-toggle[^"]*"[^>]*data-source-column="img"[^>]*>',
        body,
    )
    assert img_badge is not None
    assert ' hidden' in img_badge.group(0)


def test_preview_no_longer_renders_role_dropdown_or_pred_fields(
    admin_client, monkeypatch,
):
    """Roles + pred fields moved to LB creation. The HF import
    preview is now role-neutral and pred-field-free; it captures
    only what the dataset itself owns (name, kind, params)."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts',
                        lambda repo_id, **kw: {"train": 50000, "test": 10000})

    body = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'uoft-cs/cifar10'},
    ).data.decode()
    # No role <select> on any column row.
    assert 'name="field_role"' not in body
    # No "Prediction fields" section header / Add-row button.
    assert 'Prediction fields' not in body
    # No <input name="pred_field_name"> form elements either (the
    # `pred_field_name` string can appear in dead JS branches; check
    # the actual form-element string instead).
    assert '<input type="text" name="pred_field_name"' not in body
    assert 'id="pred-add-row"' not in body


def test_preview_renders_class_label_vocab_as_data_attribute(admin_client, monkeypatch):
    """When the HF datasets-server /info endpoint returns a
    ClassLabel.names vocab for a column, the matching row carries
    `data-label-names="[...]"` so the per-kind params editor can
    pre-fill the textarea on the client without the admin typing it."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})
    monkeypatch.setattr(hfs, 'fetch_class_label_vocabs',
                        lambda repo_id, **kw: {'label': ['airplane', 'automobile', 'bird']})

    body = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'uoft-cs/cifar10'},
    ).data.decode()
    assert 'data-label-names' in body
    # Bracketed JSON list with the class names — escaped HTML attr.
    assert 'airplane' in body
    assert 'automobile' in body


def test_preview_defaults_to_stratified_when_label_field_detected(admin_client, monkeypatch):
    """Classification datasets (any field's kind suggested as label)
    default the sampling strategy to `stratified` so the admin
    doesn't get a class-imbalanced subset by accident."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    # cifar10's `label` column triggers the name-based label upgrade.
    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})

    body = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'uoft-cs/cifar10'},
    ).data.decode()
    assert '<option value="stratified" selected>' in body
    # And uniform is NOT selected on classification preview.
    assert '<option value="uniform" selected>' not in body


def test_preview_keeps_uniform_default_when_no_label_field(admin_client, monkeypatch):
    """A non-classification Croissant doc (e.g. a depth-only dataset)
    leaves the default at uniform."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = {
        "@type": "sc:Dataset",
        "name": "depth_only",
        "recordSet": [{
            "@id": "rs",
            "@type": "cr:RecordSet",
            "field": [
                {"@id": "rs/depth", "@type": "cr:Field", "dataType": "sc:ImageObject",
                 "source": {"extract": {"column": "depth"}}},
            ],
        }],
    }
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})

    body = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'fake/depth'},
    ).data.decode()
    assert '<option value="uniform" selected>' in body
    assert '<option value="stratified" selected>' not in body


def test_commit_route_imports_fields_with_neutral_role(client, db_session, tmp_path, monkeypatch):
    """Dataset is role-neutral now: the commit handler hands every
    field to the materializer with role='gt' as a placeholder (the
    column is NOT NULL on DatasetField). LBs override per-LB via
    `field_roles_json` at creation time. No more pred-fields
    section on the import — preds are LB-level."""
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, 'UPLOAD_FOLDER', str(tmp_path))
    admin = User(email='neutral@bench.local', display_name='n',
                 oauth_provider='github', oauth_sub='n-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id

    captured = {}
    import benchhub.hf_materialize as hfm

    def _fake_materialize(repo_id, *, split, sample_cap, staging_dir,
                          dataset_name, fields, **kw):
        captured['fields'] = fields
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

    from benchhub import manifest as bh_manifest

    def _fake_import(source_root, *, db_session, Dataset, Sample, CustomField,
                     DatasetField=None, upload_folder, owner_user_id=None,
                     visibility='public', existing_dataset=None, **_kw):
        from pathlib import Path
        m = json.loads(Path(source_root, 'manifest.json').read_text())
        if existing_dataset is not None:
            ds = existing_dataset
        else:
            ds = Dataset(name=m['name'], owner_user_id=owner_user_id,
                         visibility=visibility)
            db_session.add(ds)
        db_session.flush()
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
        'dataset_name': 'neutral-test',
        'split': 'test',
        'sample_cap': '1',
        'sampling': 'head',
        'sampling_seed': '0',
        'field_name': ['img', 'label'],
        'field_source_column': ['img', 'label'],
        'field_kind': ['image', 'label'],
        'field_params': ['', ''],
    }, follow_redirects=False)
    assert r.status_code == 302
    # Both fields land as role='gt' placeholders; LB decides at creation.
    by_name = {f['name']: f for f in captured['fields']}
    assert by_name['img']['role'] == 'gt'
    assert by_name['label']['role'] == 'gt'


def test_commit_drops_field_used_as_sample_name_source(client, db_session, tmp_path, monkeypatch):
    """When the admin toggles `use as name` on a text field, that
    field's values become the on-disk sample names; importing it
    ALSO as a regular column would duplicate the same data in
    every Sample row. The commit handler must drop it from the
    field list before materialise."""
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, 'UPLOAD_FOLDER', str(tmp_path))
    admin = User(email='dedupe@bench.local', display_name='d',
                 oauth_provider='github', oauth_sub='d-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id

    captured = {}
    import benchhub.hf_materialize as hfm

    def _fake_materialize(repo_id, *, split, sample_cap, staging_dir,
                          dataset_name, fields, sample_name_from=None, **kw):
        captured['fields'] = fields
        captured['sample_name_from'] = sample_name_from
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

    from benchhub import manifest as bh_manifest

    def _fake_import(source_root, *, db_session, Dataset, Sample, CustomField,
                     DatasetField=None, upload_folder, owner_user_id=None,
                     visibility='public', existing_dataset=None, **_kw):
        from pathlib import Path
        m = json.loads(Path(source_root, 'manifest.json').read_text())
        ds = existing_dataset or Dataset(
            name=m['name'], owner_user_id=owner_user_id, visibility=visibility)
        if existing_dataset is None:
            db_session.add(ds)
        db_session.flush()
        return ds.id, {
            'dataset_id': ds.id, 'name': m['name'],
            'samples': len(m['samples']), 'fields': len(m['fields']),
            'custom_field_rows': 0, 'files_copied': 0, 'bytes_on_disk': 0,
        }

    monkeypatch.setattr(bh_manifest, 'import_typed_dataset', _fake_import)

    r = client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'uoft-cs/cifar10',
        'dataset_name': 'name-dedupe',
        'split': 'test',
        'sample_cap': '1',
        'sampling': 'head',
        'sampling_seed': '0',
        'field_name':           ['filename',  'img',   'label'],
        'field_source_column':  ['filename',  'img',   'label'],
        'field_kind':           ['text',      'image', 'label'],
        'field_params':         ['',          '',      ''],
        'sample_name_from': 'filename',
    }, follow_redirects=False)
    assert r.status_code == 302
    assert captured['sample_name_from'] == 'filename'
    by_name = {f['name']: f for f in captured['fields']}
    assert 'filename' not in by_name, (
        "the column flagged as sample-name source must NOT also "
        "be imported as its own field"
    )
    assert 'img' in by_name and 'label' in by_name


def test_preview_hands_off_to_file_tree_when_no_schema(admin_client, monkeypatch):
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    def _boom(repo_id, **kw):
        raise hfc.CroissantFetchError("no such repo")

    monkeypatch.setattr(hfc, 'fetch_croissant', _boom)
    monkeypatch.setattr(hfs, 'fetch_dataset_info', lambda r, **k: None)
    r = admin_client.post(
        '/admin/import_from_hf/preview',
        data={'repo_id': 'private/secret'},
        follow_redirects=False,
    )
    # No tabular schema → hand off to the file-tree importer, not 5xx.
    assert r.status_code == 302
    assert '/import_from_files/inspect' in r.headers['Location']


def test_preview_redirects_when_repo_id_missing(admin_client):
    r = admin_client.post('/admin/import_from_hf/preview', data={})
    assert r.status_code == 302
    assert '/admin/import_from_hf' in r.headers['Location']


# ---------------------------------------------------------------------------
# Suggestion endpoints (search + trending)
# ---------------------------------------------------------------------------

def test_search_route_requires_login_open_to_users(client, db_session, monkeypatch):
    """Unauthenticated → 302 to login; any signed-in user → 200."""
    from benchhub import hf_search
    monkeypatch.setattr(hf_search, 'search_datasets', lambda q, **kw: [])
    r = client.get('/admin/import_from_hf/search?q=cifar')
    assert r.status_code == 302
    other = User(email='regular@bench.local', display_name='reg',
                 oauth_provider='github', oauth_sub='reg-search-1', is_admin=False)
    db.session.add(other); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = other.id
    r = client.get('/admin/import_from_hf/search?q=cifar')
    assert r.status_code == 200


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


def test_trending_route_requires_login_open_to_users(client, db_session, monkeypatch):
    from benchhub import hf_search
    monkeypatch.setattr(hf_search, 'trending_by_domain',
                        lambda **kw: {"Vision": [], "NLP": [], "Audio": [], "Tabular": []})
    r = client.get('/admin/import_from_hf/trending')
    assert r.status_code == 302
    other = User(email='regular2@bench.local', display_name='reg2',
                 oauth_provider='github', oauth_sub='reg-trending-1', is_admin=False)
    db.session.add(other); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = other.id
    r = client.get('/admin/import_from_hf/trending')
    assert r.status_code == 200


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


def test_card_route_returns_summary(admin_client, monkeypatch):
    """The card route serialises the helper's summary; missing repo_id
    is a 400."""
    from benchhub import hf_search
    monkeypatch.setattr(hf_search, 'card_summary', lambda r, **k: {
        "id": r, "title": "Foo", "description": "A test dataset.",
        "gated": True, "private": False, "downloads": 9, "likes": 1,
        "task_categories": ["image-classification"]})
    r = admin_client.get('/admin/import_from_hf/card?repo_id=me/foo')
    assert r.status_code == 200
    body = r.get_json()
    assert body["title"] == "Foo" and body["gated"] is True
    assert body["description"] == "A test dataset."
    # No repo_id → 400.
    assert admin_client.get('/admin/import_from_hf/card').status_code == 400


def _mock_hf_import(monkeypatch, tmp_path, captured):
    """Patch the heavy HF materialize + typed import so the commit route
    runs end-to-end (eager Celery) without network. Records kwargs."""
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, 'UPLOAD_FOLDER', str(tmp_path))
    import benchhub.hf_materialize as hfm
    from benchhub import manifest as bh_manifest
    from pathlib import Path

    def _fake_materialize(repo_id, *, split, sample_cap, staging_dir,
                          dataset_name, fields, **kw):
        captured['sample_cap'] = sample_cap
        Path(staging_dir, 'manifest.json').write_text(json.dumps({
            'name': dataset_name, 'version': '1.0',
            'fields': [{'name': f['name'], 'kind': f['kind'],
                        'role': f['role'], 'params': f.get('params') or {}}
                       for f in fields],
            'samples': ['s0'],
        }))
        return {'samples': 1, 'fields': len(fields),
                'rows_written': 0, 'rows_skipped': 0}

    def _fake_import(source_root, *, db_session, Dataset, Sample, CustomField,
                     DatasetField=None, upload_folder, owner_user_id=None,
                     visibility='public', existing_dataset=None, **_kw):
        ds = existing_dataset
        db_session.flush()
        return ds.id, {'dataset_id': ds.id, 'name': ds.name, 'samples': 1,
                       'fields': 1, 'custom_field_rows': 0, 'files_copied': 0,
                       'bytes_on_disk': 0}

    monkeypatch.setattr(hfm, 'materialize_hf_to_typed_dir', _fake_materialize)
    monkeypatch.setattr(bh_manifest, 'import_typed_dataset', _fake_import)


def test_user_commit_is_private_and_full_split(client, db_session, tmp_path, monkeypatch):
    """A non-admin import lands private but caches the FULL split (no row
    cap — quota is the only bound, per the no-cap policy)."""
    from app import Dataset
    user = User(email='hfuser@bench.local', display_name='u',
                oauth_provider='github', oauth_sub='hfuser-1', is_admin=False)
    db.session.add(user); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id

    captured = {}
    _mock_hf_import(monkeypatch, tmp_path, captured)

    r = client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'uoft-cs/cifar10', 'dataset_name': 'user-cache',
        'split': 'test', 'sample_cap': '-1', 'sampling': 'head',
        'field_name': ['img'], 'field_source_column': ['img'],
        'field_kind': ['image'], 'field_params': [''],
    }, follow_redirects=False)
    assert r.status_code == 302
    assert captured['sample_cap'] == -1   # full split, not truncated
    ds = Dataset.query.filter_by(name='user-cache').first()
    assert ds is not None and ds.visibility == 'private'
    assert ds.owner_user_id == user.id


def test_user_commit_concurrency_guard(client, db_session, tmp_path, monkeypatch):
    """A non-admin with an in-flight import can't start a second one."""
    from app import Dataset
    user = User(email='busy@bench.local', display_name='b',
                oauth_provider='github', oauth_sub='busy-1', is_admin=False)
    db.session.add(user)
    db.session.add(Dataset(name='inflight', owner_user_id=2,
                           visibility='private', import_status='importing'))
    db.session.flush()
    inflight = Dataset.query.filter_by(name='inflight').first()
    db.session.commit()
    inflight.owner_user_id = user.id
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id

    captured = {}
    _mock_hf_import(monkeypatch, tmp_path, captured)
    before = Dataset.query.count()
    r = client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'uoft-cs/cifar10', 'dataset_name': 'second',
        'split': 'test', 'sample_cap': '10',
        'field_name': ['img'], 'field_source_column': ['img'],
        'field_kind': ['image'], 'field_params': [''],
    }, follow_redirects=False)
    assert r.status_code == 302
    # No new dataset created; materialize never invoked.
    assert Dataset.query.count() == before
    assert 'sample_cap' not in captured


def test_preview_blocks_gated_dataset_without_token(client, db_session, monkeypatch):
    """A gated/private repo with no available HF token is hard-blocked at
    preview with a redirect back to the form."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs
    import app as _app
    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda r, **kw: {})
    monkeypatch.setattr(_app, 'fetch_dataset_card', None, raising=False)
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_card',
                        lambda r, **kw: {'gated': 'manual'})
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_info',
                        lambda r, **kw: {'features': {}})
    monkeypatch.delenv('HF_TOKEN', raising=False)

    u = User(email='notoken@bench.local', display_name='nt',
             oauth_provider='github', oauth_sub='nt-1', is_admin=False)
    db.session.add(u); db.session.commit()
    with client.session_transaction() as s:
        s['user_id'] = u.id

    r = client.post('/admin/import_from_hf/preview',
                    data={'repo_id': 'secret/repo'}, follow_redirects=False)
    assert r.status_code == 302
    assert '/admin/import_from_hf' in r.headers['Location']
    with client.session_transaction() as s:
        flashes = ' '.join(v for _, v in s.get('_flashes', []))
    assert 'gated or private' in flashes


def test_preview_warns_on_json_fallback_fields(admin_client, monkeypatch):
    """Fields that map to raw JSON get an advisory warning banner (not a
    block)."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs
    # A field whose Croissant type isn't a typed kind → json.
    fixture = {
        "@type": "sc:Dataset", "name": "synth",
        "recordSet": [{"@id": "rs", "@type": "cr:RecordSet", "field": [
            {"@id": "rs/img", "@type": "cr:Field", "dataType": "sc:ImageObject",
             "source": {"extract": {"column": "img"}}},
            {"@id": "rs/meta", "@type": "cr:Field", "dataType": "sc:VideoObject",
             "source": {"extract": {"column": "meta"}}},
        ]}],
    }
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda r, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda r, **kw: {})
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_card', lambda r, **kw: {})
    monkeypatch.setattr('benchhub.hf_search.fetch_dataset_info',
                        lambda r, **kw: {'features': {}})

    body = admin_client.post('/admin/import_from_hf/preview',
                             data={'repo_id': 'x/y'}).data.decode()
    assert 'Heads up before you import' in body
    assert 'raw JSON' in body
