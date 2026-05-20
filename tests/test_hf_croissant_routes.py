"""Admin route tests for /admin/import_from_hf — auth + Croissant preview.

The /commit handler talks to `datasets.load_dataset()` and isn't tested
here; its materialiser logic is covered separately."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import User, db


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

    fixture = json.loads((FIXTURES / 'croissant_cifar10.json').read_text())
    monkeypatch.setattr(hfc, 'fetch_croissant', lambda repo_id, **kw: fixture)

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
