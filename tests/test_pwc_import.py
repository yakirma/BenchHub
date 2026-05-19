"""Phase 15: Papers With Code import. Admin-only flow that creates a
canonical leaderboard backed by an HF dataset, with one mirrored
Submission per PWC result row. Mirrored submissions skip the eval
pipeline; their MetricResult rows are inserted at import time.
"""
import json
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
            # Depth-estimation-style eval to exercise the broader lower-better
            # heuristic (user-reported: RMS got tagged higher-is-better).
            (400, 2, 'Depth Estimation', '',
             _json.dumps([
                 'RMS', 'RMSE', 'AbsRel', 'SqRel', 'Log10', 'SILog',
                 'Delta < 1.25', 'mAP',
             ]),
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


def test_search_datasets_empty_query_returns_popular_datasets(fake_archive_index):
    """Empty query → top-N by total result rows, so admin can browse
    without needing to know what to search for."""
    rows = pwc_client.search_datasets('')
    # Both fixture datasets show up; CIFAR-10 has more results so it
    # comes first (ImageNet has only 1 result row in the fixture).
    assert len(rows) == 2
    assert rows[0]['name'] == 'CIFAR-10'


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


def test_get_evaluation_sort_direction_covers_depth_error_metrics(
    fake_archive_index,
):
    """User-reported: NYU Depth V2 imports tagged `RMS` as higher-is-better
    because the heuristic only had `rmse`. Lock the expanded table for
    common depth / regression / LM error metric names so the default
    sort direction matches what the literature actually means."""
    ev = pwc_client.get_evaluation(400)
    by_name = {m['name']: m['sort_direction'] for m in ev['metrics']}

    # Errors → lower is better.
    for name in ('RMS', 'RMSE', 'AbsRel', 'SqRel', 'Log10', 'SILog'):
        assert by_name[name] == 'lower_is_better', (
            f"{name!r} should default to lower_is_better but got "
            f"{by_name[name]!r}"
        )

    # Non-errors → higher is better (Delta accuracy / mAP).
    for name in ('Delta < 1.25', 'mAP'):
        assert by_name[name] == 'higher_is_better', (
            f"{name!r} should default to higher_is_better but got "
            f"{by_name[name]!r}"
        )


def test_get_evaluation_unknown_id_raises(fake_archive_index):
    with pytest.raises(pwc_client.PwcError):
        pwc_client.get_evaluation(999_999)


@pytest.fixture
def fake_hf_hub(monkeypatch):
    """Inject a stub `huggingface_hub` module so suggest_hf_repo can
    be exercised without the real package installed locally. The
    fixture returns a dict the test mutates to control HfApi behavior."""
    import sys, types
    state = {'datasets_by_term': {}, 'call_count': 0}

    class _R:
        def __init__(self, id_, downloads):
            self.id = id_
            self.downloads = downloads

    class _Api:
        def list_datasets(self, search=None, limit=20):
            state['call_count'] += 1
            return state['datasets_by_term'].get(search, [])

    fake_module = types.ModuleType('huggingface_hub')
    fake_module.HfApi = _Api
    fake_module._R = _R  # so tests can build response objects
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_module)
    import pwc_client as pc
    pc._HF_SUGGEST_CACHE.clear()
    yield state
    pc._HF_SUGGEST_CACHE.clear()


def test_suggest_hf_repo_picks_most_downloaded_match(fake_hf_hub):
    """list_datasets returns matching repos; we filter to those whose
    id contains the search term, then rank by download count."""
    import sys
    R = sys.modules['huggingface_hub']._R
    fake_hf_hub['datasets_by_term'] = {
        'ImageNet': [
            R('imagenet-1k', 50000),
            R('zh-plus/tiny-imagenet', 1200),
            R('not-related-repo', 99999),  # filtered out by id-contains check
        ],
        'imagenet': [
            R('imagenet-1k', 50000),  # dedupe across calls
            R('axiong/imagenet-r', 800),
        ],
    }
    import pwc_client as pc
    best, alts = pc.suggest_hf_repo('ImageNet')
    assert best == 'imagenet-1k'
    assert 'zh-plus/tiny-imagenet' in alts
    assert 'not-related-repo' not in alts


def test_suggest_hf_repo_returns_none_when_no_matches(fake_hf_hub):
    import pwc_client as pc
    best, alts = pc.suggest_hf_repo('NonexistentDataset')
    assert best is None
    assert alts == []


def test_suggest_hf_repo_caches_result(fake_hf_hub):
    """Same name (case-folded) shouldn't re-hit the API on second call."""
    import pwc_client as pc
    pc.suggest_hf_repo('CIFAR-10')
    n_after_first = fake_hf_hub['call_count']
    pc.suggest_hf_repo('cifar-10')
    assert fake_hf_hub['call_count'] == n_after_first  # cache hit; no new calls


def test_search_datasets_includes_n_benchmarks_and_n_results(fake_archive_index):
    """Coverage badges in the admin UI need both counts. The fixture has
    CIFAR-10 with 1 benchmark / 1 result row and ImageNet with 1/1 too,
    but the SUM/COUNT shape is what we're validating."""
    rows = pwc_client.search_datasets('cifar')
    assert rows[0]['n_benchmarks'] >= 1
    assert rows[0]['n_results'] >= 1


def test_list_evaluations_sorted_by_n_results_desc(fake_archive_index):
    """Rich benchmarks should bubble to the top of the per-dataset
    benchmark list so admins don't accidentally pick the empty ones."""
    evs = pwc_client.list_evaluations_for_dataset(1)
    assert all('n_results' in e and 'n_metrics' in e for e in evs)
    # Sorted descending by n_results — fixture puts the ASR benchmark
    # (0 rows) after the Image Classification one (1 row).
    counts = [e['n_results'] for e in evs]
    assert counts == sorted(counts, reverse=True)


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
    assert b"PWC index isn't loaded" in r.data
    assert b'upload_pwc_index.py' in r.data


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
         'hf_repo': 'cifar10', 'url': None,
         'n_benchmarks': 5, 'n_results': 80},
        {'id': 2, 'name': 'unknown-source', 'full_name': 'No HF Link',
         'description': '',
         'huggingface_url': None, 'hf_repo': None, 'url': None,
         'n_benchmarks': 1, 'n_results': 1},
    ]
    with patch('pwc_client.index_status', return_value='ready'), \
         patch('pwc_client.search_datasets', return_value=fake_rows):
        r = client.get('/admin/pwc/import?q=tiny')
    assert b'CIFAR-10' in r.data
    assert b'No HF Link' in r.data
    assert b'HF: cifar10' in r.data


def test_pwc_index_build_route_returns_disabled_message(
    client, db_session, login_as,
):
    """In-prod build was disabled — pyarrow decode of the archive's
    parquet was unreliable on a 4 GB worker. Index ships pre-built
    via scripts/upload_pwc_index.py. Endpoint stays as a polite
    redirect-with-flash so an admin who bookmarked it gets a useful
    message rather than a 404."""
    admin = _mk_admin('build_admin@bench.local')
    login_as(admin)
    with patch('tasks.build_pwc_index') as mock_task:
        r = client.post('/admin/pwc/index/build', follow_redirects=True)
    mock_task.delay.assert_not_called()
    assert b'shipped pre-built' in r.data


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


@pytest.fixture
def stub_hf_features(monkeypatch):
    """Stub `_hf_fetch_features` for the import tests. Default returns
    the cifar10 schema; tests can mutate `state['features']` per-call."""
    state = {'features': {
        'img': {'type': 'Image'},
        'label': {'type': 'Value:int64'},
    }}
    monkeypatch.setattr('app._hf_fetch_features',
                        lambda *a, **kw: state['features'])
    return state


@pytest.mark.xfail(reason="Asserts Leaderboard.canonicality (dropped in commit 317dd94). HF import wiring is Phase A delete pile.")
def test_creates_lb_with_hf_attachment_and_mirrored_subs(
    client, db_session, login_as, stub_hf_features,
):
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


@pytest.mark.xfail(reason="Asserts old _infer_mapping field-naming (bare 'gt_label' vs new 'gt_scalar_label'). HF import wiring is Phase A delete pile.")
def test_creates_lb_arg_mappings_target_real_gt_field(
    client, db_session, login_as, stub_hf_features,
):
    """The PWC LB's LeaderboardMetric.arg_mappings must reference an
    actual GT field on the HF dataset (e.g. 'gt_label' for cifar10),
    not the placeholder 'gt_unknown'. Without this, generated SOTA
    notebooks have PRED_FIELDS=['unknown_pred'] and submissions can't
    score against the right column."""
    admin = _mk_admin('arg_mapping_admin@bench.local')
    evaluation = {
        'id': 60, 'task': 'Image Classification', 'dataset': 'CIFAR-10',
        'description': '',
        'metrics': [{'name': 'Top 1 Accuracy', 'description': '',
                     'sort_direction': 'higher_is_better'}],
        'results': [],
    }
    lb_id = _create_lb_from_pwc_benchmark(
        evaluation, hf_repo='cifar10', lb_name='arg_mapping_lb',
        owner_user_id=admin.id,
    )
    lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb_id).first()
    arg_map = json.loads(lm.arg_mappings)
    # The cifar10 stub schema has Image('img') + Value:int64('label').
    # _infer_mapping picks 'label' as the scalar GT; arg_mappings must
    # reference it (NOT 'gt_unknown').
    assert arg_map['gt'] == 'gt_label'
    assert arg_map['pred'] == 'sub_label_pred'

    # The Attachment also gets the inferred mapping persisted so the
    # eval engine + the LB Settings UI can see it.
    att = Attachment.query.filter_by(leaderboard_id=lb_id).first()
    mapping = json.loads(att.hf_mapping_json)
    by_col = {m['column']: m['target_kind'] for m in mapping}
    assert by_col == {'img': 'image', 'label': 'scalar'}


def test_creates_lb_falls_back_to_unknown_when_schema_probe_fails(
    client, db_session, login_as, monkeypatch,
):
    """Schema fetch can fail (gated dataset / network blip / HF
    outage). LB creation must still succeed with the placeholder
    arg_mappings — admin can wire it manually later."""
    admin = _mk_admin('schema_fail_admin@bench.local')

    def _boom(*a, **kw):
        raise RuntimeError("HF Hub timeout (test stub)")
    monkeypatch.setattr('app._hf_fetch_features', _boom)

    evaluation = {
        'id': 61, 'task': 'X', 'dataset': 'Y', 'description': '',
        'metrics': [{'name': 'Score', 'description': '',
                     'sort_direction': 'higher_is_better'}],
        'results': [],
    }
    lb_id = _create_lb_from_pwc_benchmark(
        evaluation, hf_repo='owner/private', lb_name='schema_fail_lb',
        owner_user_id=admin.id,
    )
    lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb_id).first()
    arg_map = json.loads(lm.arg_mappings)
    assert arg_map['gt'] == 'gt_unknown'
    assert arg_map['pred'] == 'sub_unknown_pred'


def test_creates_lb_calls_llm_to_author_metric_code(
    client, db_session, login_as, stub_hf_features,
):
    """When ANTHROPIC_API_KEY is set, _create_lb_from_pwc_benchmark
    should run each PWC metric name through _llm_generate_metric_code
    and persist the generated code instead of the NotImplementedError
    stub. Falls back to the stub when the helper returns None."""
    admin = _mk_admin('llm_metric_admin@bench.local')
    evaluation = {
        'id': 999, 'task': 'X', 'dataset': 'Y', 'description': '',
        'metrics': [{'name': 'mIoU', 'description': '',
                     'sort_direction': 'higher_is_better'}],
        'results': [],
    }
    fake_code = (
        "def miou(gt, pred):\n"
        "    \"\"\"Mean intersection-over-union.\"\"\"\n"
        "    import numpy as _np\n"
        "    return float(_np.mean(_np.asarray(gt) == _np.asarray(pred)))\n"
    )
    with patch('app._llm_generate_metric_code', return_value=fake_code) as mock_llm:
        lb_id = _create_lb_from_pwc_benchmark(
            evaluation, hf_repo='owner/repo', lb_name='miou_lb',
            owner_user_id=admin.id,
        )
        mock_llm.assert_called_once()
        # Helper called with (slugified_name, llm_hint).
        called_name, called_hint = mock_llm.call_args.args
        assert called_name == 'miou'
        assert "PWC metric 'mIoU'" in called_hint

    gm = GlobalMetric.query.filter_by(name='miou').first()
    assert gm is not None
    assert gm.python_code == fake_code  # LLM-authored, not the stub
    assert 'NotImplementedError' not in gm.python_code


def test_creates_lb_falls_back_to_stub_when_llm_unavailable(
    client, db_session, login_as, stub_hf_features,
):
    admin = _mk_admin('llm_fail_admin@bench.local')
    evaluation = {
        'id': 998, 'task': 'X', 'dataset': 'Y', 'description': '',
        'metrics': [{'name': 'BLEU-4', 'description': '',
                     'sort_direction': 'higher_is_better'}],
        'results': [],
    }
    with patch('app._llm_generate_metric_code', return_value=None):
        _create_lb_from_pwc_benchmark(
            evaluation, hf_repo='owner/repo', lb_name='bleu_fallback_lb',
            owner_user_id=admin.id,
        )
    gm = GlobalMetric.query.filter_by(name='bleu_4').first()
    assert gm is not None
    assert 'NotImplementedError' in gm.python_code
    assert 'LB Settings page' in gm.python_code


def test_creates_lb_summary_metrics_uses_target_names(
    client, db_session, login_as, stub_hf_features,
):
    """summary_metrics must match LeaderboardMetric.target_name (PWC name)
    so the LB view's resolver finds them. Earlier bug stored slugified
    GlobalMetric names which the resolver auto-pruned, leaving zero
    metric columns rendered."""
    admin = _mk_admin('summary_admin@bench.local')
    evaluation = {
        'id': 800, 'task': 'X', 'dataset': 'Y', 'description': '',
        'metrics': [
            {'name': 'Top 1 Accuracy', 'description': '',
             'sort_direction': 'higher_is_better'},
            {'name': 'BLEU-4', 'description': '',
             'sort_direction': 'higher_is_better'},
        ],
        'results': [],
    }
    lb_id = _create_lb_from_pwc_benchmark(
        evaluation, hf_repo='owner/repo', lb_name='summary_test_lb',
        owner_user_id=admin.id,
    )
    lb = Leaderboard.query.get(lb_id)
    parts = [p.strip() for p in (lb.summary_metrics or '').split(',') if p.strip()]
    # Verbatim PWC names, NOT slugified.
    assert 'Top 1 Accuracy' in parts
    assert 'BLEU-4' in parts
    assert 'top_1_accuracy' not in parts


def test_skips_unparseable_metric_values(client, db_session, login_as, stub_hf_features):
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
