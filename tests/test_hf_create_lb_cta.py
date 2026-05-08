"""End-to-end: clicking 'Create leaderboard with this dataset' on
/hf/<repo_id> should land the user on the auto-LB preview after the
import completes — NOT the dataset page (which is what the 'Just
import as a dataset' CTA does).

Pin the divergence so the two CTAs don't drift back to being
functionally identical."""
import sys
import types
from unittest.mock import patch

import pytest
from PIL import Image

from app import Dataset, Leaderboard, db


@pytest.fixture
def fake_hf_classlabel(monkeypatch):
    """Tiny ClassLabel-shaped fixture: 2 rows with image + label.
    The label column gives the auto-LB proposer a top-1 metric to
    propose, so the auto-LB preview is non-empty."""
    rows = [
        {'image': Image.new('RGB', (4, 4), (10, 20, 30)), 'label': 0},
        {'image': Image.new('RGB', (4, 4), (40, 50, 60)), 'label': 1},
    ]

    class _ClassLabel:
        names = ['cat', 'dog']

    class _IterableDS:
        features = {'image': object(), 'label': _ClassLabel()}
        def __iter__(self):
            return iter(rows)

    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: _IterableDS()
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)

    class _MetaResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                'tags': ['task_categories:image-classification'],
                'description': 'fixture',
                'cardData': {
                    'dataset_info': [{'features': {
                        'image': {'_type': 'Image'},
                        'label': {'_type': 'ClassLabel',
                                  'names': ['cat', 'dog']},
                    }}],
                },
            }
    monkeypatch.setattr('requests.get', lambda *a, **kw: _MetaResp())


def test_create_lb_cta_lands_on_auto_lb_preview(
    auth_client, logged_in_user, db_session, fake_hf_classlabel,
    tmp_path, monkeypatch,
):
    """Two-step flow: /import_from_hf/preview with auto_create_lb=1
    renders the mapping preview with the flag carried through; the
    follow-up /import_from_hf/auto submit (which the preview's
    'Confirm & import' button does) imports the dataset AND renders
    auto_lb_preview.html instead of redirecting to the dataset page."""
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )

    # Step 1: preview, with the auto_create_lb flag set.
    resp = auth_client.post('/import_from_hf/preview', data={
        'hf_repo_id': 'fake/cls-bench',
        'auto_create_lb': '1',
    }, follow_redirects=False)
    assert resp.status_code == 200
    body = resp.data.decode()
    # The flag is propagated as a hidden field so the next POST keeps it.
    assert 'name="auto_create_lb"' in body
    assert 'value="1"' in body

    # Step 2: confirm & import. follow_redirects=False since the
    # success path RENDERS auto_lb_preview rather than redirecting.
    resp = auth_client.post('/import_from_hf/auto', data={
        'hf_repo_id': 'fake/cls-bench',
        'dataset_name': 'cls_bench_ds',
        'sample_cap': '5',
        'auto_create_lb': '1',
        'mapping_column[]': ['image', 'label'],
        'mapping_target_kind[]': ['image', 'scalar'],
        'mapping_target_field[]': ['image_image', 'label'],
    }, follow_redirects=False)
    assert resp.status_code == 200
    body = resp.data.decode()
    # We landed on the auto-LB preview (not a redirect to dataset).
    assert 'Review auto-proposed' in body
    assert 'cls_bench_ds_leaderboard' in body
    # The dataset DID land — the auto-LB preview is the next step,
    # not a parallel one.
    ds = Dataset.query.filter_by(name='cls_bench_ds').first()
    assert ds is not None
    # No LB created yet — the user has to confirm on the auto-LB
    # preview to actually persist it.
    assert Leaderboard.query.filter_by(name='cls_bench_ds_leaderboard').first() is None


def test_just_import_cta_lands_on_dataset_page_unchanged(
    auth_client, logged_in_user, db_session, fake_hf_classlabel,
    tmp_path, monkeypatch,
):
    """Without auto_create_lb=1 the import_from_hf/auto path
    redirects to the dataset page as before."""
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    resp = auth_client.post('/import_from_hf/auto', data={
        'hf_repo_id': 'fake/cls-plain',
        'dataset_name': 'plain_ds',
        'sample_cap': '5',
        'mapping_column[]': ['image', 'label'],
        'mapping_target_kind[]': ['image', 'scalar'],
        'mapping_target_field[]': ['image_image', 'label'],
    }, follow_redirects=False)
    assert resp.status_code == 302
    assert '/dataset/' in resp.headers['Location']


def test_create_lb_cta_falls_back_when_no_metrics_proposable(
    auth_client, logged_in_user, db_session, monkeypatch, tmp_path,
):
    """If the imported dataset has no scalar / image / depth GT
    columns, the proposer returns nothing. We flash a hint and fall
    through to the dataset page rather than rendering an empty preview."""
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache')
    )
    # Text-only fixture — proposer returns [].
    rows = [{'caption': 'hi'}, {'caption': 'bye'}]

    class _IterableDS:
        features = {'caption': object()}
        def __iter__(self):
            return iter(rows)

    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: _IterableDS()
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)

    class _MetaResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'tags': [], 'description': '', 'cardData': {}}
    monkeypatch.setattr('requests.get', lambda *a, **kw: _MetaResp())

    resp = auth_client.post('/import_from_hf/auto', data={
        'hf_repo_id': 'fake/text-only',
        'dataset_name': 'text_only_ds',
        'sample_cap': '5',
        'auto_create_lb': '1',
        'mapping_column[]': ['caption'],
        'mapping_target_kind[]': ['text'],
        'mapping_target_field[]': ['caption'],
    }, follow_redirects=False)
    # Falls through to the dataset page with a flash.
    assert resp.status_code == 302
    assert '/dataset/' in resp.headers['Location']
