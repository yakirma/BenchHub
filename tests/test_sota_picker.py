"""Phase 26: admin SOTA notebook picker. Auth-gates the route, infers
the HF task slug from the LB metadata, lists trending HF models for
that task, and generates a Claude-authored notebook for the chosen
model. Falls back gracefully when the LLM is unavailable.
"""
import sys
import types
from unittest.mock import patch

import pytest

import app as app_mod
from app import (
    Dataset, GlobalMetric, Leaderboard, LeaderboardMetric, Sample, User, db,
    _hf_task_for_lb,
)


def _mk_admin(email='sota_admin@bench.local'):
    u = User(email=email, display_name='admin', is_admin=True,
             oauth_provider='github', oauth_sub=email)
    db.session.add(u); db.session.commit()
    return u


def _mk_user(email='regular@bench.local'):
    u = User(email=email, display_name='reg', is_admin=False,
             oauth_provider='github', oauth_sub=email)
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def login_as(client):
    def _go(user):
        with client.session_transaction() as sess:
            sess['user_id'] = user.id
    return _go


def _seed_lb(name='Image Classification on CIFAR-10', metric_target='Top 1 Accuracy'):
    ds = Dataset(name=f'{name}_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    lb = Leaderboard(name=name, summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    gm = GlobalMetric(
        name='top_1_accuracy',
        python_code='def top_1_accuracy(gt, pred): return 0.0',
    )
    db.session.add(gm); db.session.flush()
    db.session.add(LeaderboardMetric(
        leaderboard_id=lb.id, global_metric_id=gm.id,
        target_name=metric_target,
        arg_mappings='{}', sort_direction='higher_is_better',
    ))
    db.session.commit()
    return lb


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------


def test_hf_task_for_lb_image_classification(client, db_session):
    lb = _seed_lb('Image Classification on CIFAR-10', 'top-1 accuracy')
    assert _hf_task_for_lb(lb) == 'image-classification'


def test_hf_task_for_lb_question_answering(client, db_session):
    lb = _seed_lb('Question Answering on SQuAD', 'F1')
    assert _hf_task_for_lb(lb) == 'question-answering'


def test_hf_task_for_lb_unknown_returns_none(client, db_session):
    lb = _seed_lb('Some Made Up Task on Mystery', 'mystery_metric')
    assert _hf_task_for_lb(lb) is None


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------


def test_sota_picker_requires_admin(client, db_session, login_as):
    lb = _seed_lb()
    user = _mk_user()
    login_as(user)
    r = client.get(f'/admin/leaderboard/{lb.id}/sota_picker')
    assert r.status_code == 403


def test_sota_notebook_post_requires_admin(client, db_session, login_as):
    lb = _seed_lb()
    user = _mk_user()
    login_as(user)
    r = client.post(f'/admin/leaderboard/{lb.id}/sota_notebook',
                    data={'model_id': 'microsoft/resnet-50'})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Picker rendering with stub HF Hub
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hf_models(monkeypatch):
    """Inject huggingface_hub with a stubbed list_models so the picker
    page renders without network."""
    state = {'models': []}

    class _Model:
        def __init__(self, id_, downloads=0, likes=0,
                     library_name='transformers', last_modified='2025-01-01'):
            self.id = id_
            self.downloads = downloads
            self.likes = likes
            self.library_name = library_name
            self.last_modified = last_modified

    class _Api:
        def list_models(self, *, sort=None, direction=None, limit=20, filter=None):
            return state['models']

    fake = types.ModuleType('huggingface_hub')
    fake.HfApi = _Api
    fake._Model = _Model
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake)
    app_mod._HF_SOTA_CACHE.clear()
    yield state
    app_mod._HF_SOTA_CACHE.clear()


def test_sota_picker_lists_trending_models(client, db_session, login_as, fake_hf_models):
    admin = _mk_admin()
    login_as(admin)
    lb = _seed_lb()
    Model = sys.modules['huggingface_hub']._Model
    fake_hf_models['models'] = [
        Model('microsoft/resnet-50', downloads=12_000_000, likes=400),
        Model('google/vit-base', downloads=8_000_000, likes=300),
    ]
    r = client.get(f'/admin/leaderboard/{lb.id}/sota_picker')
    assert r.status_code == 200
    assert b'microsoft/resnet-50' in r.data
    assert b'google/vit-base' in r.data
    assert b'image-classification' in r.data  # inferred task


def test_sota_notebook_generates_via_llm(client, db_session, login_as, fake_hf_models):
    admin = _mk_admin('sota_gen_admin@bench.local')
    login_as(admin)
    lb = _seed_lb()
    fake_nb = '{"cells": [{"cell_type": "markdown", "source": "ResNet-50"}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    with patch('app._llm_sota_colab_notebook', return_value=fake_nb) as mock:
        r = client.post(
            f'/admin/leaderboard/{lb.id}/sota_notebook',
            data={'model_id': 'microsoft/resnet-50'},
        )
    assert r.status_code == 200
    assert r.headers['Content-Type'].startswith('application/x-ipynb+json')
    assert b'ResNet-50' in r.data
    mock.assert_called_once()


def test_sota_notebook_falls_back_on_llm_failure(client, db_session, login_as, fake_hf_models):
    admin = _mk_admin('sota_fallback_admin@bench.local')
    login_as(admin)
    lb = _seed_lb()
    with patch('app._llm_sota_colab_notebook', return_value=None):
        r = client.post(
            f'/admin/leaderboard/{lb.id}/sota_notebook',
            data={'model_id': 'broken/model'},
            follow_redirects=True,
        )
    # Redirect back to picker with a flash, no .ipynb download.
    assert r.headers.get('Content-Type', '').startswith('text/html')
    # HTML escapes the apostrophe — match either form.
    assert (b"Couldn&#39;t generate" in r.data
            or b"Couldn't generate" in r.data)


def test_sota_notebook_requires_model_id(client, db_session, login_as, fake_hf_models):
    admin = _mk_admin('sota_empty_admin@bench.local')
    login_as(admin)
    lb = _seed_lb()
    r = client.post(
        f'/admin/leaderboard/{lb.id}/sota_notebook',
        data={'model_id': ''},
        follow_redirects=True,
    )
    assert b'HF model id is required' in r.data
