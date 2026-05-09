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
    with patch('pwc_client.search_datasets', return_value=[]):
        r = client.get('/admin/pwc/import?q=cifar')
    assert r.status_code == 200
    assert b'Import from Papers With Code' in r.data


def test_pwc_search_lists_results(client, db_session, login_as):
    admin = _mk_admin()
    login_as(admin)
    fake_rows = [
        {'id': 1, 'name': 'cifar10', 'full_name': 'CIFAR-10',
         'description': 'Tiny image classification dataset.',
         'paper': None, 'huggingface_url': 'https://huggingface.co/datasets/cifar10',
         'hf_repo': 'cifar10',
         'url': 'https://paperswithcode.com/dataset/cifar-10'},
        {'id': 2, 'name': 'no-hf-dataset', 'full_name': 'No HF Dataset',
         'description': '', 'paper': None,
         'huggingface_url': '', 'hf_repo': None, 'url': None},
    ]
    with patch('pwc_client.search_datasets', return_value=fake_rows):
        r = client.get('/admin/pwc/import?q=tiny')
    assert b'CIFAR-10' in r.data
    assert b'cifar10' in r.data
    # Non-HF datasets are listed but disabled (no "Browse benchmarks" CTA).
    assert b'No HF Dataset' in r.data
    assert b'no HF mirror' in r.data


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
