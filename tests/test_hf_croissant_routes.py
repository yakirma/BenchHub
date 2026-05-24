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


def test_preview_offers_sample_name_from_dropdown_for_text_cols(
    admin_client, monkeypatch,
):
    """When the upstream schema has any text-kind column, the
    preview form surfaces a 'Sample name from' dropdown listing
    them so the admin can use their values as sample names
    instead of the default `s000000…` enumeration."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    # Synthetic schema: image + a text-kind `caption`.
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
    assert 'name="sample_name_from"' in body
    # Default is the auto-numbered option (selected).
    assert 'auto-numbered' in body
    # The text column appears as a selectable option.
    assert '<option value="caption">' in body


def test_preview_skips_sample_name_dropdown_when_no_text_cols(
    admin_client, monkeypatch,
):
    """No text columns → no point showing the dropdown."""
    from benchhub import hf_croissant as hfc
    from benchhub import hf_search as hfs

    fixture = {
        "@type": "sc:Dataset", "name": "synth",
        "recordSet": [{
            "@id": "rs", "@type": "cr:RecordSet",
            "field": [
                {"@id": "rs/img", "@type": "cr:Field", "dataType": "sc:ImageObject",
                 "source": {"extract": {"column": "img"}}},
                {"@id": "rs/score", "@type": "cr:Field", "dataType": "sc:Float",
                 "source": {"extract": {"column": "score"}}},
            ],
        }],
    }
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)
    monkeypatch.setattr(hfs, 'fetch_split_row_counts', lambda repo_id, **kw: {})
    body = admin_client.post('/admin/import_from_hf/preview',
                             data={'repo_id': 'x/y'}).data.decode()
    assert 'name="sample_name_from"' not in body


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
                     visibility='public'):
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
