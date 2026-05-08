"""HF live-preview surface: /hf/<repo_id> renders without writing
a Dataset row, captures visits, and feeds the recent + trending
widgets on /datasets.
"""
import io
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app import (
    Dataset, HfDatasetVisit, db,
    _record_hf_visit, _user_recent_hf_visits, _trending_hf_visits,
    _classify_preview_value,
)


# ---------------------------------------------------------------------------
# _classify_preview_value — runtime-shape classification.
# ---------------------------------------------------------------------------


def test_classify_image_pil():
    assert _classify_preview_value(Image.new('RGB', (4, 4))) == 'image'


def test_classify_depth_2d_numpy():
    assert _classify_preview_value(np.zeros((4, 4))) == 'depth'


def test_classify_image_3d_numpy():
    assert _classify_preview_value(np.zeros((4, 4, 3), dtype=np.uint8)) == 'image'


def test_classify_scalar_int():
    assert _classify_preview_value(42) == 'scalar'


def test_classify_scalar_float():
    assert _classify_preview_value(0.5) == 'scalar'


def test_classify_text():
    assert _classify_preview_value('hello world') == 'text'


def test_classify_none():
    assert _classify_preview_value(None) == 'unknown'


# ---------------------------------------------------------------------------
# Visit tracking — upsert + read helpers.
# ---------------------------------------------------------------------------


def test_record_visit_creates_row(client, db_session, logged_in_user):
    _record_hf_visit(logged_in_user.id, 'cifar10')
    v = HfDatasetVisit.query.filter_by(
        user_id=logged_in_user.id, repo_id='cifar10').first()
    assert v is not None
    assert v.visit_count == 1


def test_record_visit_bumps_count_on_revisit(client, db_session, logged_in_user):
    _record_hf_visit(logged_in_user.id, 'cifar10')
    _record_hf_visit(logged_in_user.id, 'cifar10')
    _record_hf_visit(logged_in_user.id, 'cifar10')
    v = HfDatasetVisit.query.filter_by(
        user_id=logged_in_user.id, repo_id='cifar10').one()
    assert v.visit_count == 3


def test_record_visit_silent_on_anonymous(client, db_session):
    """user_id None → no row written. Browsing is allowed for anon
    users (no @login_required on /hf/<repo>); we just don't track."""
    _record_hf_visit(None, 'cifar10')
    assert HfDatasetVisit.query.count() == 0


def test_user_recent_hf_returns_newest_first(client, db_session, logged_in_user):
    older = HfDatasetVisit(
        user_id=logged_in_user.id, repo_id='older/repo',
        last_visited_at=datetime.utcnow() - timedelta(days=2),
    )
    newer = HfDatasetVisit(
        user_id=logged_in_user.id, repo_id='newer/repo',
        last_visited_at=datetime.utcnow() - timedelta(minutes=5),
    )
    db.session.add_all([older, newer]); db.session.commit()
    out = _user_recent_hf_visits(logged_in_user.id)
    assert [v.repo_id for v in out] == ['newer/repo', 'older/repo']


def test_trending_hf_aggregates_across_users(client, db_session):
    from app import User
    u1 = User(email='t1@example.com', display_name='t1',
              oauth_provider='github', oauth_sub='t1')
    u2 = User(email='t2@example.com', display_name='t2',
              oauth_provider='github', oauth_sub='t2')
    db.session.add_all([u1, u2]); db.session.flush()
    now = datetime.utcnow()
    db.session.add_all([
        HfDatasetVisit(user_id=u1.id, repo_id='hot/repo', visit_count=5,
                       last_visited_at=now),
        HfDatasetVisit(user_id=u2.id, repo_id='hot/repo', visit_count=3,
                       last_visited_at=now),
        HfDatasetVisit(user_id=u1.id, repo_id='warm/repo', visit_count=2,
                       last_visited_at=now),
        # Stale: outside the 7-day window.
        HfDatasetVisit(user_id=u1.id, repo_id='cold/repo', visit_count=99,
                       last_visited_at=now - timedelta(days=30)),
    ])
    db.session.commit()
    rows = _trending_hf_visits(days=7, limit=10)
    by_repo = {r['repo_id']: r for r in rows}
    assert by_repo['hot/repo']['visits'] == 8
    assert by_repo['hot/repo']['users'] == 2
    assert by_repo['warm/repo']['visits'] == 2
    assert 'cold/repo' not in by_repo


# ---------------------------------------------------------------------------
# /hf/<repo_id> route — renders without writing a Dataset row.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hf_for_preview(monkeypatch):
    rows = [
        {'image': Image.new('RGB', (4, 4), (10, 20, 30)), 'label': 0,
         'caption': 'a cat'},
        {'image': Image.new('RGB', (4, 4), (40, 50, 60)), 'label': 1,
         'caption': 'a dog'},
    ]

    class _ClassLabel:
        names = ['cat', 'dog']

    class _IterableDS:
        features = {
            'image': object(),
            'label': _ClassLabel(),
            'caption': object(),
        }
        def __iter__(self):
            return iter(rows)

    fake_mod = types.ModuleType('datasets')
    fake_mod.load_dataset = lambda *a, **kw: _IterableDS()
    monkeypatch.setitem(sys.modules, 'datasets', fake_mod)

    class _MetaResp:
        def raise_for_status(self): pass
        status_code = 200
        def json(self):
            return {
                'tags': ['task_categories:image-classification', 'license:mit'],
                'description': 'Tiny preview-fixture dataset.',
                'cardData': {'license': 'mit'},
            }
    monkeypatch.setattr('requests.get', lambda *a, **kw: _MetaResp())


def test_hf_preview_renders_without_creating_dataset_row(
    client, db_session, fake_hf_for_preview, tmp_path, monkeypatch,
):
    """The whole point of this surface: visiting /hf/<repo> doesn't
    materialize a Dataset row, doesn't write to uploads/datasets/.
    Only when the user explicitly clicks 'Create LB' does a row land."""
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    initial = Dataset.query.count()
    resp = client.get('/hf/fake/preview-bench')
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'fake/preview-bench' in body
    assert 'Tiny preview-fixture dataset.' in body
    assert 'image-classification' in body  # tag rendered
    # Schema columns appear.
    assert 'caption' in body and 'label' in body
    # No new Dataset row.
    assert Dataset.query.count() == initial


def test_hf_preview_records_visit_for_logged_in_user(
    auth_client, logged_in_user, db_session, fake_hf_for_preview, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    auth_client.get('/hf/fake/preview-bench')
    v = HfDatasetVisit.query.filter_by(
        user_id=logged_in_user.id, repo_id='fake/preview-bench').first()
    assert v is not None
    assert v.visit_count == 1


def test_hf_preview_does_not_record_for_anon_user(
    client, db_session, fake_hf_for_preview, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    client.get('/hf/fake/preview-bench')
    assert HfDatasetVisit.query.count() == 0


def test_hf_preview_renders_image_thumbnails_via_preview_cell_url(
    client, db_session, fake_hf_for_preview, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    resp = client.get('/hf/fake/preview-bench')
    body = resp.data.decode()
    # Each row's image cell should produce a /hf/.../preview_cell/.../image link.
    assert '/hf/fake/preview-bench/preview_cell/0/image' in body
    assert '/hf/fake/preview-bench/preview_cell/1/image' in body


def test_preview_cell_endpoint_serves_cached_bytes(
    client, db_session, fake_hf_for_preview, tmp_path, monkeypatch,
):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    # Hit the page first to warm the cache.
    client.get('/hf/fake/preview-bench')
    # Now fetch one cell directly.
    resp = client.get('/hf/fake/preview-bench/preview_cell/0/image')
    assert resp.status_code == 200
    assert resp.headers['Content-Type'].startswith('image/')


def test_preview_cell_404s_when_cache_evicted(client, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    resp = client.get('/hf/fake/preview-bench/preview_cell/0/image')
    assert resp.status_code == 404
    body = resp.get_json()
    assert 'evicted' in body['error'].lower() or 'refresh' in body['error'].lower()


# ---------------------------------------------------------------------------
# /datasets renders the recent + trending widgets when data exists.
# ---------------------------------------------------------------------------


def test_datasets_page_renders_recent_widget_for_logged_in_user(
    auth_client, logged_in_user, db_session,
):
    db.session.add(HfDatasetVisit(
        user_id=logged_in_user.id, repo_id='cifar10',
        last_visited_at=datetime.utcnow(), visit_count=2,
    ))
    db.session.commit()
    resp = auth_client.get('/datasets')
    body = resp.data.decode()
    assert 'Your recent HF picks' in body
    assert 'cifar10' in body
    # Each entry links back to /hf/<repo_id>.
    assert 'href="/hf/cifar10"' in body


def test_datasets_page_renders_trending_widget_when_visits_exist(
    auth_client, logged_in_user, db_session,
):
    db.session.add(HfDatasetVisit(
        user_id=logged_in_user.id, repo_id='popular/repo',
        last_visited_at=datetime.utcnow(), visit_count=42,
    ))
    db.session.commit()
    resp = auth_client.get('/datasets')
    body = resp.data.decode()
    assert 'Trending across BenchHub' in body
    assert 'popular/repo' in body


def test_datasets_page_omits_widgets_when_no_data(client, db_session):
    """Anonymous user, no visits anywhere → don't render either widget
    (avoids empty cards on a fresh deploy)."""
    resp = client.get('/datasets')
    body = resp.data.decode()
    assert 'Your recent HF picks' not in body
    assert 'Trending across BenchHub' not in body
