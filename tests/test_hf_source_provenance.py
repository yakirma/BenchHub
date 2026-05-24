"""HF imports record provenance on Dataset.source_kind/url/metadata.

The dataset-view page surfaces a "Source" card with a back-link to
the HF repo and a one-line summary of how the BH copy was sampled.
Without this the import is anonymous — admins (and users browsing
the dataset later) have no way to tell where the bytes came from.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app import Dataset, app as flask_app, db
from benchhub import hf_croissant as hfc
from benchhub import hf_search as hfs


def _load_fixture(name):
    p = Path(__file__).parent / 'fixtures' / name
    return json.loads(p.read_text())


class _FakeDataset:
    def __init__(self, rows, features=None):
        self._rows = rows
        if features is not None:
            self.features = features

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


@pytest.fixture
def admin_client(client, db_session):
    from app import User
    admin = User(email='admin@example.com', display_name='admin',
                 oauth_provider='github', oauth_sub='prov-1', is_admin=True)
    db.session.add(admin); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id
    return client


def _install_fake_hf(monkeypatch, rows, features=None):
    fake = types.ModuleType('datasets')
    fake.load_dataset = lambda repo_id, **kw: _FakeDataset(rows, features=features)
    monkeypatch.setitem(sys.modules, 'datasets', fake)


def test_hf_commit_writes_source_url_kind_and_metadata(admin_client, monkeypatch, tmp_path):
    """End-to-end: hitting /admin/import_from_hf/commit must populate
    Dataset.source_kind='hf', .source_url, and .source_metadata with
    enough detail to render the provenance card."""
    monkeypatch.setattr(hfc, 'fetch_croissant',
                        lambda repo_id, **kw: _load_fixture('croissant_cifar10.json'))
    rows = [{'img': np.zeros((4, 4, 3), dtype=np.uint8), 'label': i % 3}
            for i in range(20)]
    _install_fake_hf(monkeypatch, rows)

    r = admin_client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'uoft-cs/cifar10',
        'dataset_name': 'cifar_provenance',
        'split': 'test',
        'sample_cap': '10',
        'sampling': 'uniform',
        'sampling_seed': '7',
        'field_name': ['img', 'label'],
        'field_source_column': ['img', 'label'],
        'field_kind': ['image', 'label'],
        'field_role': ['input', 'gt'],
        'field_params': ['', ''],
    }, follow_redirects=False)
    # Either 302 (redirect to dataset_view) or 200 — both fine for the assertion.
    assert r.status_code in (200, 302)

    ds = Dataset.query.filter_by(name='cifar_provenance').first()
    assert ds is not None
    assert ds.source_kind == 'hf'
    assert ds.source_url == 'https://huggingface.co/datasets/uoft-cs/cifar10'
    meta = ds.source_metadata_parsed
    assert meta['repo_id'] == 'uoft-cs/cifar10'
    assert meta['split'] == 'test'
    assert meta['sampling'] == 'uniform'
    assert meta['sampling_seed'] == 7
    assert meta['samples_imported'] == 10
    assert meta['total_rows_in_split'] == 20


def test_dataset_view_renders_source_card_for_hf_dataset(client, db_session):
    """The provenance card on /dataset/<id> shows the HF link and
    the human-readable sampling summary."""
    import os
    ds = Dataset(
        name='hf_back_link_test',
        visibility='public',
        source_kind='hf',
        source_url='https://huggingface.co/datasets/foo/bar',
        source_metadata=json.dumps({
            'repo_id': 'foo/bar',
            'split': 'validation',
            'sampling': 'stratified',
            'sampling_seed': 42,
            'samples_imported': 500,
            'total_rows_in_split': 50000,
            'rows_skipped': 0,
        }),
    )
    db.session.add(ds); db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)

    body = client.get(f'/dataset/{ds.id}').data.decode('utf-8')
    assert 'https://huggingface.co/datasets/foo/bar' in body
    assert 'foo/bar' in body
    assert 'validation' in body
    assert 'stratified' in body
    # Comma-formatted row count for readability.
    assert '50,000' in body
    assert '500' in body


def test_hf_commit_rejects_over_quota_before_download(client, db_session, monkeypatch):
    """The pre-materialize quota check rejects an over-cap import
    before `datasets.load_dataset` is even invoked, so no bytes are
    pulled. We assert by raising loudly from the fake loader and
    confirming the route never reaches it.

    Admins bypass the quota cap (commit f36fe8f), so this test uses
    a regular user — the quota path only applies to them. The
    route still requires `is_admin` to enter; we elevate the
    user just enough to pass the route gate but keep the email out
    of `BENCHHUB_ADMIN_EMAILS` so `is_admin(user)` returns False
    inside check_quota.
    """
    from app import User
    user = User(email='quota-test@example.com', display_name='q',
                oauth_provider='github', oauth_sub='q-1', is_admin=True)
    db.session.add(user); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    # Bypass admin gating on the route but stub `check_quota` so the
    # dataset_create path returns a rejection — admins normally
    # bypass quota (commit f36fe8f) but the pre-materialize check
    # is the contract under test here.
    import app as _app
    monkeypatch.setattr(_app, 'check_quota',
                        lambda u, *, kind, incoming_bytes=0: (False, 'Over quota (test)'))

    monkeypatch.setattr(hfc, 'fetch_croissant',
                        lambda repo_id, **kw: _load_fixture('croissant_cifar10.json'))
    monkeypatch.setattr(hfs, 'fetch_split_row_counts',
                        lambda repo_id, **kw: {'test': 10000})
    # 5 GB parquet × 1.5x headroom → ~7.5 GB estimate, way over the
    # user's default 50 MB quota.
    monkeypatch.setattr(hfs, 'fetch_split_byte_sizes',
                        lambda repo_id, **kw: {'test': 5_000_000_000})

    def _boom(*a, **kw):
        raise AssertionError(
            'load_dataset must NOT be called when the pre-check rejects')
    fake = types.ModuleType('datasets')
    fake.load_dataset = _boom
    monkeypatch.setitem(sys.modules, 'datasets', fake)

    r = client.post('/admin/import_from_hf/commit', data={
        'repo_id': 'uoft-cs/cifar10',
        'dataset_name': 'too_big',
        'split': 'test',
        'sample_cap': '-1',
        'sampling': 'head',
        'sampling_seed': '7',
        'field_name': ['img', 'label'],
        'field_source_column': ['img', 'label'],
        'field_kind': ['image', 'label'],
        'field_role': ['input', 'gt'],
        'field_params': ['', ''],
    }, follow_redirects=False)
    assert r.status_code == 302
    # The route flashes a danger message and redirects to the
    # import form — the DB has no new dataset row.
    assert Dataset.query.filter_by(name='too_big').first() is None


def test_source_card_hides_sampling_when_all_rows_imported(client, db_session):
    """sample_cap=-1 (or samples_imported == total_rows_in_split)
    means every row in the split is in this BH dataset — the
    sampling method + seed don't constrain anything, so they're
    suppressed from the card."""
    import os
    ds = Dataset(
        name='full_split_dataset',
        visibility='public',
        source_kind='hf',
        source_url='https://huggingface.co/datasets/foo/bar',
        source_metadata=json.dumps({
            'repo_id': 'foo/bar',
            'split': 'test',
            'sample_cap': -1,
            'sampling': 'stratified',
            'sampling_seed': 42,
            'samples_imported': 10000,
            'total_rows_in_split': 10000,
            'rows_skipped': 0,
        }),
    )
    db.session.add(ds); db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    body = client.get(f'/dataset/{ds.id}').data.decode('utf-8')
    assert 'all 10,000' in body
    # Method + seed are suppressed on the full-split path.
    assert 'stratified sampling' not in body
    assert 'seed' not in body


def test_no_source_card_for_local_dataset(client, db_session):
    """ZIP-uploaded datasets (no source_kind) don't show the card."""
    import os
    ds = Dataset(name='local_dataset', visibility='public')
    db.session.add(ds); db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)

    body = client.get(f'/dataset/{ds.id}').data.decode('utf-8')
    assert 'huggingface.co/datasets' not in body
