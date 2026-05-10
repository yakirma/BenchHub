"""Phase 15: Papers With Code import. Admin-only flow that creates a
canonical leaderboard backed by an HF dataset, with one mirrored
Submission per PWC result row. Mirrored submissions skip the eval
pipeline; their MetricResult rows are inserted at import time.
"""
from unittest.mock import patch

import pytest

from app import (
    Attachment, GlobalMetric, Leaderboard, LeaderboardMetric,
    MetricResult, Submission, User, db,
    _create_lb_from_pwc_benchmark,
)
import pwc_client


# ---------------------------------------------------------------------------
# pwc_client URL parsing
# ---------------------------------------------------------------------------


def test_hf_repo_extraction():
    f = pwc_client._hf_repo_from_url
    assert f('https://huggingface.co/datasets/imagenet-1k') == 'imagenet-1k'
    assert f('https://huggingface.co/datasets/owner/repo') == 'owner/repo'
    assert f('https://huggingface.co/datasets/owner/repo/tree/main') == 'owner/repo'
    assert f('https://huggingface.co/owner/model') is None  # not a dataset URL
    assert f('https://github.com/foo/bar') is None
    assert f('') is None
    assert f(None) is None


# ---------------------------------------------------------------------------
# Static-archive index (in-process only — actual download is too big
# to exercise in tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_archive_index(tmp_path):
    """Hand-build a tiny SQLite index matching pwc_client's schema, then
    point the module at it. Avoids downloading the real archive (~hundreds
    of MB)."""
    import json as _json
    import sqlite3
    idx_path = str(tmp_path / 'pwc_index.sqlite')
    conn = sqlite3.connect(idx_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE pwc_dataset (id INTEGER PRIMARY KEY, name TEXT, name_lc TEXT, description TEXT, links_json TEXT)")
    cur.execute("CREATE INDEX ix_pwc_dataset_name_lc ON pwc_dataset(name_lc)")
    cur.execute("CREATE TABLE pwc_evaluation (id INTEGER PRIMARY KEY, dataset_id INTEGER, task TEXT, description TEXT, metrics_json TEXT, results_json TEXT)")
    cur.execute("CREATE INDEX ix_pwc_evaluation_ds ON pwc_evaluation(dataset_id)")
    cur.executemany(
        "INSERT INTO pwc_dataset (id, name, name_lc, description, links_json) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 'CIFAR-10', 'cifar-10', 'Tiny image classification dataset.',
             _json.dumps([{'url': 'https://huggingface.co/datasets/cifar10', 'title': 'HF'}])),
            (2, 'ImageNet', 'imagenet', 'Large image classification dataset.',
             _json.dumps([])),
        ],
    )
    cur.executemany(
        "INSERT INTO pwc_evaluation (id, dataset_id, task, description, metrics_json, results_json) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (100, 1, 'Image Classification', 'Class 0..9 prediction.',
             _json.dumps(['Top 1 Accuracy', 'Top 5 Accuracy']),
             _json.dumps([{'model_name': 'BigModel', 'paper_url': 'https://x/y',
                           'paper_title': 'Big Paper',
                           'metrics': {'Top 1 Accuracy': '99.3', 'Top 5 Accuracy': '99.9'}}])),
            (200, 2, 'Image Classification', '',
             _json.dumps(['Top 1 Accuracy']),
             _json.dumps([{'model_name': 'X', 'paper_url': '', 'paper_title': 'X',
                           'metrics': {'Top 1 Accuracy': '78.0'}}])),
            (300, 1, 'ASR', '',
             _json.dumps(['Word Error Rate', 'BLEU-4']),
             _json.dumps([])),
        ],
    )
    conn.commit(); conn.close()
    pwc_client._set_index_path_for_tests(idx_path)
    yield idx_path
    pwc_client._reset_index_path_for_tests()


def test_search_datasets_substring(fake_archive_index):
    rows = pwc_client.search_datasets('cifar')
    names = [r['name'] for r in rows]
    assert names == ['CIFAR-10']
    assert rows[0]['hf_repo'] == 'cifar10'


def test_search_datasets_empty_query(fake_archive_index):
    assert pwc_client.search_datasets('') == []


def test_search_datasets_no_hf_link_returns_none(fake_archive_index):
    rows = pwc_client.search_datasets('imagenet')
    assert rows[0]['name'] == 'ImageNet'
    assert rows[0]['hf_repo'] is None


def test_list_evaluations(fake_archive_index):
    evs = pwc_client.list_evaluations_for_dataset(1)
    tasks = sorted(e['task'] for e in evs)
    assert tasks == ['ASR', 'Image Classification']


def test_get_evaluation_normalizes_metrics(fake_archive_index):
    ev = pwc_client.get_evaluation(100)
    metric_names = [m['name'] for m in ev['metrics']]
    assert metric_names == ['Top 1 Accuracy', 'Top 5 Accuracy']
    assert all(m['sort_direction'] == 'higher_is_better' for m in ev['metrics'])


def test_get_evaluation_loss_metric_sort_direction(fake_archive_index):
    """Loss/error names auto-flip to lower_is_better."""
    ev = pwc_client.get_evaluation(300)
    by_name = {m['name']: m['sort_direction'] for m in ev['metrics']}
    assert by_name['Word Error Rate'] == 'lower_is_better'
    assert by_name['BLEU-4'] == 'higher_is_better'


def test_get_evaluation_unknown_id_raises(fake_archive_index):
    with pytest.raises(pwc_client.PwcError):
        pwc_client.get_evaluation(999_999)


def test_slugify_metric_name():
    f = pwc_client.slugify_metric_name
    assert f('Top 1 Accuracy') == 'top_1_accuracy'
    assert f('BLEU-4') == 'bleu_4'
    assert f('mAP@0.5:0.95') == 'map05095'
    assert f('  Test ') == 'test'
    assert f('1bad-start') == 'metric_1bad_start'


# ---------------------------------------------------------------------------
# Admin route auth
# ---------------------------------------------------------------------------


def _mk_admin(email='admin@bench.local'):
    u = User(email=email, display_name='admin', is_admin=True,
             oauth_provider='github', oauth_sub=email)
    db.session.add(u); db.session.commit()
    return u


def _mk_regular_user(email='user@bench.local'):
    u = User(email=email, display_name='user', is_admin=False,
             oauth_provider='github', oauth_sub=email)
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def login_as(client):
    def _go(user):
        with client.session_transaction() as sess:
            sess['user_id'] = user.id
    return _go


def test_pwc_search_requires_admin(client, db_session, login_as):
    user = _mk_regular_user()
    login_as(user)
    r = client.get('/admin/pwc/import')
    assert r.status_code == 403


def test_pwc_search_renders_for_admin(client, db_session, login_as):
    admin = _mk_admin()
    login_as(admin)
    with patch('pwc_client.index_status', return_value='ready'), \
         patch('pwc_client.search_datasets', return_value=[]):
        r = client.get('/admin/pwc/import?q=cifar')
    assert r.status_code == 200
    assert b'Import from Papers With Code' in r.data


def test_pwc_search_renders_build_cta_when_index_absent(client, db_session, login_as):
    admin = _mk_admin()
    login_as(admin)
    with patch('pwc_client.index_status', return_value='absent'):
        r = client.get('/admin/pwc/import')
    assert b"PWC index isn't built yet" in r.data
    assert b'Build PWC index' in r.data


def test_pwc_search_renders_building_state(client, db_session, login_as):
    admin = _mk_admin()
    login_as(admin)
    with patch('pwc_client.index_status', return_value='building'):
        r = client.get('/admin/pwc/import')
    assert b'Building the PWC index' in r.data


def test_pwc_search_lists_results_when_ready(client, db_session, login_as):
    admin = _mk_admin()
    login_as(admin)
    fake_rows = [
        {'id': 1, 'name': 'cifar10', 'full_name': 'CIFAR-10',
         'description': 'Tiny image classification dataset.',
         'huggingface_url': None,
         'hf_repo': 'cifar10', 'url': None},
        {'id': 2, 'name': 'unknown-source', 'full_name': 'No HF Link',
         'description': '',
         'huggingface_url': None, 'hf_repo': None, 'url': None},
    ]
    with patch('pwc_client.index_status', return_value='ready'), \
         patch('pwc_client.search_datasets', return_value=fake_rows):
        r = client.get('/admin/pwc/import?q=tiny')
    assert b'CIFAR-10' in r.data
    assert b'No HF Link' in r.data
    assert b'HF link in archive' in r.data


def test_pwc_index_build_route_enqueues_task(client, db_session, login_as):
    admin = _mk_admin('build_admin@bench.local')
    login_as(admin)
    with patch('pwc_client.index_status', return_value='absent'), \
         patch('pwc_client.begin_build_marker', return_value=True), \
         patch('tasks.build_pwc_index') as mock_task:
        r = client.post('/admin/pwc/index/build', follow_redirects=False)
    assert r.status_code in (302, 303)
    mock_task.delay.assert_called_once()


def test_pwc_index_build_is_idempotent_when_already_building(
    client, db_session, login_as,
):
    """Clicking Build twice within a minute shouldn't enqueue a second task."""
    admin = _mk_admin('idem_admin@bench.local')
    login_as(admin)
    with patch('pwc_client.index_status', return_value='building'), \
         patch('tasks.build_pwc_index') as mock_task:
        r = client.post('/admin/pwc/index/build', follow_redirects=True)
    mock_task.delay.assert_not_called()
    assert b'already in progress' in r.data


def test_pwc_index_build_is_idempotent_when_ready(client, db_session, login_as):
    admin = _mk_admin('ready_admin@bench.local')
    login_as(admin)
    with patch('pwc_client.index_status', return_value='ready'), \
         patch('tasks.build_pwc_index') as mock_task:
        r = client.post('/admin/pwc/index/build', follow_redirects=True)
    mock_task.delay.assert_not_called()
    assert b'already built' in r.data


def test_pwc_index_build_requires_admin(client, db_session, login_as):
    user = _mk_regular_user('not_admin@bench.local')
    login_as(user)
    r = client.post('/admin/pwc/index/build')
    assert r.status_code == 403


def test_pwc_search_renders_progress_line(client, db_session, login_as):
    admin = _mk_admin('progress_admin@bench.local')
    login_as(admin)
    with patch('pwc_client.index_status', return_value='building'), \
         patch('pwc_client.index_progress_message',
               return_value='Walking shard 2/4: 73 tasks indexed'):
        r = client.get('/admin/pwc/import')
    assert b'Walking shard 2/4: 73 tasks indexed' in r.data


def test_begin_build_marker_skips_when_index_exists(tmp_path, monkeypatch):
    """begin_build_marker must not stomp on a finished build."""
    import pwc_client as pc
    # Point at a tmp cache dir + drop a fake "finished" sqlite file.
    monkeypatch.setattr(pc, '_cache_dir', lambda: str(tmp_path))
    pc._set_index_path_for_tests(str(tmp_path / 'idx.sqlite'))
    try:
        (tmp_path / 'idx.sqlite').write_bytes(b'stub')
        assert pc.begin_build_marker() is False
        assert not (tmp_path / 'index.building.tmp').exists()
    finally:
        pc._reset_index_path_for_tests()


def test_begin_build_marker_creates_when_absent(tmp_path, monkeypatch):
    import pwc_client as pc
    monkeypatch.setattr(pc, '_cache_dir', lambda: str(tmp_path))
    pc._set_index_path_for_tests(str(tmp_path / 'idx.sqlite'))
    try:
        assert pc.begin_build_marker() is True
        assert (tmp_path / 'index.building.tmp').exists()
        # Second call no-ops while marker is fresh.
        assert pc.begin_build_marker() is False
    finally:
        pc._reset_index_path_for_tests()


def test_begin_build_marker_overwrites_stale(tmp_path, monkeypatch):
    """A marker older than the staleness window should be treated as
    a crashed worker and overwritten on the next Build click."""
    import os, time
    import pwc_client as pc
    monkeypatch.setattr(pc, '_cache_dir', lambda: str(tmp_path))
    pc._set_index_path_for_tests(str(tmp_path / 'idx.sqlite'))
    try:
        marker = tmp_path / 'index.building.tmp'
        marker.write_text('queued')
        # Make it 2 hours old.
        old = time.time() - 7200
        os.utime(str(marker), (old, old))
        assert pc.begin_build_marker() is True
        # Marker recreated with fresh mtime.
        assert (time.time() - os.path.getmtime(str(marker))) < 5
    finally:
        pc._reset_index_path_for_tests()


def test_build_index_into_emits_progress(tmp_path, monkeypatch):
    """The streaming build calls progress_cb at every shard boundary
    and at finalize-time, so the web tier always has a fresh line
    to render."""
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    snap = tmp_path / 'snap'; snap.mkdir()
    # One parquet shard with one row that has one (task, dataset).
    table = pa.table({
        'task': pa.array(['Image Classification']),
        'categories': pa.array([[]]),
        'description': pa.array(['']),
        'subtasks': pa.array([[]]),
        'synonyms': pa.array([[]]),
        'source_link': pa.array([None], type=pa.string()),
        'datasets': pa.array([[
            {'dataset': 'CIFAR-10', 'description': '', 'dataset_links': [],
             'subdatasets': [], 'dataset_citations': [],
             'sota': {'metrics': ['Accuracy'],
                      'rows': [{'model_name': 'M', 'paper_title': '', 'paper_url': '',
                                'paper_date': '', 'metrics': {'Accuracy': '99.0'}}]}}
        ]]),
    })
    pq.write_table(table, str(snap / 'shard.parquet'))
    msgs = []
    import pwc_client as pc
    pc._build_index_into(str(tmp_path / 'idx.sqlite'), str(snap),
                         progress_cb=msgs.append)
    assert any('Walking shard 1/1' in m for m in msgs)
    assert any('Done' in m for m in msgs)


# ---------------------------------------------------------------------------
# _create_lb_from_pwc_benchmark
# ---------------------------------------------------------------------------


def test_creates_lb_with_hf_attachment_and_mirrored_subs(client, db_session, login_as):
    admin = _mk_admin('lb_create_admin@bench.local')
    evaluation = {
        'id': 42,
        'task': 'Image Classification',
        'dataset': 'CIFAR-10',
        'description': '',
        'metrics': [
            {'name': 'Top 1 Accuracy', 'description': '', 'sort_direction': 'higher_is_better'},
            {'name': 'Top 5 Accuracy', 'description': '', 'sort_direction': 'higher_is_better'},
        ],
        'results': [
            {'id': 100, 'paper_title': 'BigModel-2024',
             'paper_url': 'https://example.com/paper1',
             'methodology': 'BigModel',
             'metrics': {'Top 1 Accuracy': '99.5', 'Top 5 Accuracy': '99.9'},
             'external_source_url': 'https://paperswithcode.com/r/100'},
            {'id': 101, 'paper_title': 'SmallModel-2023',
             'paper_url': 'https://example.com/paper2',
             'methodology': 'SmallModel',
             'metrics': {'Top 1 Accuracy': '88.0', 'Top 5 Accuracy': '95.0'},
             'external_source_url': None},
        ],
    }
    lb_id = _create_lb_from_pwc_benchmark(
        evaluation, hf_repo='cifar10',
        lb_name='Image Classification on CIFAR-10',
        owner_user_id=admin.id,
    )

    lb = Leaderboard.query.get(lb_id)
    assert lb is not None
    assert lb.canonicality == 'public'
    assert lb.canonical_for_repo == 'cifar10'
    assert lb.visibility == 'public'

    # HF attachment created.
    atts = Attachment.query.filter_by(leaderboard_id=lb.id).all()
    assert len(atts) == 1
    assert atts[0].kind == 'hf'
    assert atts[0].hf_repo_id == 'cifar10'

    # GlobalMetric + LeaderboardMetric per PWC metric.
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    target_names = {lm.target_name for lm in lms}
    assert 'Top 1 Accuracy' in target_names
    assert 'Top 5 Accuracy' in target_names
    # Slugified GlobalMetric names so future verified subs land in the
    # same column the mirrored rows populate.
    gm_names = {gm.name for gm in GlobalMetric.query.all()}
    assert 'top_1_accuracy' in gm_names
    assert 'top_5_accuracy' in gm_names

    # One mirrored Submission per result row.
    subs = Submission.query.filter_by(leaderboard_id=lb.id).all()
    assert len(subs) == 2
    for s in subs:
        assert s.kind == 'mirrored'
        assert s.processing_status == 'Mirrored'
        assert s.source_attribution == 'Papers With Code'
    big = next(s for s in subs if s.name.startswith('BigModel'))
    assert big.source_paper_url == 'https://example.com/paper1'
    assert big.source_external_url == 'https://paperswithcode.com/r/100'

    # MetricResult rows persisted with parsed scores.
    big_results = MetricResult.query.filter_by(submission_id=big.id).all()
    assert len(big_results) == 2
    by_target = {r.leaderboard_metric.target_name: r.value for r in big_results}
    assert by_target['Top 1 Accuracy'] == 99.5
    assert by_target['Top 5 Accuracy'] == 99.9


def test_skips_unparseable_metric_values(client, db_session, login_as):
    admin = _mk_admin('skip_admin@bench.local')
    evaluation = {
        'id': 50, 'task': 'X', 'dataset': 'Y', 'description': '',
        'metrics': [{'name': 'Score', 'description': '',
                     'sort_direction': 'higher_is_better'}],
        'results': [
            {'id': 1, 'paper_title': 'P1', 'paper_url': '',
             'methodology': 'P1', 'metrics': {'Score': 'N/A'}, 'external_source_url': None},
            {'id': 2, 'paper_title': 'P2', 'paper_url': '',
             'methodology': 'P2', 'metrics': {'Score': '0.91'}, 'external_source_url': None},
        ],
    }
    lb_id = _create_lb_from_pwc_benchmark(
        evaluation, hf_repo='owner/repo', lb_name='lb_skip', owner_user_id=admin.id,
    )
    lb = Leaderboard.query.get(lb_id)
    subs = Submission.query.filter_by(leaderboard_id=lb.id).all()
    assert len(subs) == 2
    p1 = next(s for s in subs if s.name.startswith('P1'))
    p2 = next(s for s in subs if s.name.startswith('P2'))
    # Unparseable → no MetricResult.
    assert MetricResult.query.filter_by(submission_id=p1.id).count() == 0
    assert MetricResult.query.filter_by(submission_id=p2.id).count() == 1


# ---------------------------------------------------------------------------
# Mirrored submissions skip the Celery eval pipeline
# ---------------------------------------------------------------------------


def test_mirrored_submission_short_circuits_in_process(
    client, db_session, login_as,
):
    """tasks._process_submission_impl must noop on kind='mirrored' so
    no extraction / metric exec / status mutation happens."""
    from tasks import _process_submission_impl
    ds = _seed_dataset()
    lb = Leaderboard(name='mirror_pipeline_lb', summary_metrics='',
                     visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    sub = Submission(
        name='mirrored_sub', leaderboard_id=lb.id,
        kind='mirrored', processing_status='Mirrored',
    )
    db.session.add(sub); db.session.commit()

    _process_submission_impl(sub.id)

    # Status shouldn't have been mutated (still 'Mirrored').
    db.session.expire_all()
    fresh = Submission.query.get(sub.id)
    assert fresh.processing_status == 'Mirrored'


def _seed_dataset():
    from app import Dataset, Sample
    ds = Dataset(name='mirror_pipeline_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.commit()
    return ds
