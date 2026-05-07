"""Auto-create-leaderboard-with-metrics during HF auto-import.

Tests the metric proposer + LB-creation helper:
- ClassLabel-shaped metric_<col> → accuracy proposal (higher_is_better)
- Plain numeric metric_<col> → MAE proposal (lower_is_better)
- Strict name-match dedupe against the GlobalMetric library.
- LLM code generation when the GlobalMetric is missing, with a
  deterministic fallback when the LLM is unavailable.
"""
import json
from unittest.mock import patch

import pytest

from app import (
    CustomField, Dataset, GlobalMetric, Leaderboard, LeaderboardMetric,
    Sample, db,
    _propose_metrics_for_dataset,
    _llm_generate_metric_code,
    _auto_create_lb_with_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_classlabel_shape(db_session):
    """Dataset that mirrors a HF ClassLabel auto-import: a metric_label
    field plus a sibling label_class string field signaling the
    ClassLabel name list."""
    ds = Dataset(name='cls_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='metric_label', field_type='metric',
        value_float=0.0,
    ))
    db.session.add(CustomField(
        sample_id=s.id, name='label_class', field_type='text',
        value_text='cat',
    ))
    db.session.commit()
    return ds


@pytest.fixture
def dataset_numeric_shape(db_session):
    """Dataset with a plain numeric metric_<col> and no class sidecar."""
    ds = Dataset(name='num_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='metric_score', field_type='metric',
        value_float=0.42,
    ))
    db.session.commit()
    return ds


# ---------------------------------------------------------------------------
# _propose_metrics_for_dataset
# ---------------------------------------------------------------------------


def test_propose_classlabel_picks_accuracy(dataset_classlabel_shape):
    proposals = _propose_metrics_for_dataset(dataset_classlabel_shape)
    assert len(proposals) == 1
    p = proposals[0]
    assert p['global_name'] == 'accuracy_label'
    assert p['sort_direction'] == 'higher_is_better'
    assert p['arg_mappings'] == {'gt': 'gt_metric_label', 'pred': 'sub_metric_label'}
    assert 'def accuracy_label' in p['fallback_code']


def test_propose_numeric_picks_mae(dataset_numeric_shape):
    proposals = _propose_metrics_for_dataset(dataset_numeric_shape)
    assert len(proposals) == 1
    p = proposals[0]
    assert p['global_name'] == 'mae_score'
    assert p['sort_direction'] == 'lower_is_better'
    assert 'def mae_score' in p['fallback_code']


def test_propose_skips_non_metric_fields(db_session):
    ds = Dataset(name='img_only', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='image_image', field_type='image',
        value_text='/path/to/img.png',
    ))
    db.session.commit()
    assert _propose_metrics_for_dataset(ds) == []


# ---------------------------------------------------------------------------
# _llm_generate_metric_code
# ---------------------------------------------------------------------------


def test_llm_generate_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert _llm_generate_metric_code('accuracy_label', 'classlabel') is None


def test_llm_generate_returns_python_when_api_key_set(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    fake_code = (
        "def accuracy_label(gt, pred):\n"
        "    return 1.0 if int(gt) == int(pred) else 0.0\n"
    )

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': fake_code}]}

    with patch('requests.post', return_value=_Resp()):
        out = _llm_generate_metric_code('accuracy_label', 'classlabel')
    assert out is not None
    assert 'def accuracy_label(' in out


def test_llm_generate_rejects_response_with_wrong_function_name(monkeypatch):
    """Hallucinated function names trip the safety check so the caller
    falls back to the deterministic template."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': 'def wrong_name(gt, pred): return 0.0'}]}

    with patch('requests.post', return_value=_Resp()):
        out = _llm_generate_metric_code('accuracy_label', 'classlabel')
    assert out is None


# ---------------------------------------------------------------------------
# _auto_create_lb_with_metrics
# ---------------------------------------------------------------------------


def test_auto_create_lb_uses_existing_global_metric_when_named_match(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    """If a GlobalMetric with the proposed name already exists, attach
    it to the new LB instead of creating a duplicate."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    pre_existing = GlobalMetric(
        name='accuracy_label',
        description='Pre-existing — should be reused.',
        python_code='def accuracy_label(gt, pred):\n    return 1.0\n',
        is_aggregated=False,
        owner_user_id=logged_in_user.id,
    )
    db.session.add(pre_existing); db.session.commit()
    pre_id = pre_existing.id

    ok, _msg, lb_id = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'lb_reuse', owner_user_id=logged_in_user.id,
    )
    assert ok and lb_id

    lb = Leaderboard.query.get(lb_id)
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    assert len(lms) == 1
    assert lms[0].global_metric_id == pre_id
    # Still only one GlobalMetric named accuracy_label.
    assert GlobalMetric.query.filter_by(name='accuracy_label').count() == 1


def test_auto_create_lb_falls_back_to_static_code_when_llm_unavailable(
    dataset_numeric_shape, logged_in_user, db_session, monkeypatch,
):
    """No API key → use the proposer's deterministic fallback_code so
    the metric is functional even on local dev."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    ok, _msg, lb_id = _auto_create_lb_with_metrics(
        dataset_numeric_shape, 'lb_static_metric',
        owner_user_id=logged_in_user.id,
    )
    assert ok and lb_id

    gm = GlobalMetric.query.filter_by(name='mae_score').first()
    assert gm is not None
    assert 'def mae_score' in gm.python_code
    assert 'abs(' in gm.python_code


def test_auto_create_lb_uses_llm_code_when_api_key_set(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    fake_code = (
        "def accuracy_label(gt, pred):\n"
        "    # LLM-authored\n"
        "    try:\n"
        "        return 1.0 if int(gt) == int(pred) else 0.0\n"
        "    except Exception:\n"
        "        return 0.0\n"
    )

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': fake_code}]}

    with patch('requests.post', return_value=_Resp()):
        ok, _msg, lb_id = _auto_create_lb_with_metrics(
            dataset_classlabel_shape, 'lb_llm_metric',
            owner_user_id=logged_in_user.id,
        )
    assert ok and lb_id
    gm = GlobalMetric.query.filter_by(name='accuracy_label').first()
    assert gm is not None
    assert '# LLM-authored' in gm.python_code


def test_auto_create_lb_refuses_duplicate_name(
    dataset_classlabel_shape, logged_in_user, db_session,
):
    """Don't silently overwrite an existing leaderboard."""
    db.session.add(Leaderboard(
        name='dup', summary_metrics='', owner_user_id=logged_in_user.id,
    ))
    db.session.commit()
    ok, msg, _ = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'dup', owner_user_id=logged_in_user.id,
    )
    assert not ok
    assert 'already exists' in msg.lower()


def test_auto_create_lb_returns_clear_error_when_no_metric_fields(db_session, logged_in_user):
    """Image-only datasets have nothing to auto-attach; surface that
    instead of creating an empty LB."""
    ds = Dataset(name='img_only_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='image_rgb', field_type='image',
        value_text='/x.png',
    ))
    db.session.commit()
    ok, msg, lb_id = _auto_create_lb_with_metrics(
        ds, 'empty_lb', owner_user_id=logged_in_user.id,
    )
    assert not ok
    assert lb_id is None
    assert 'metric_' in msg.lower() or 'no metric' in msg.lower()


# ---------------------------------------------------------------------------
# /create_leaderboard end-to-end with auto_assign_metrics=1
# ---------------------------------------------------------------------------


def test_create_leaderboard_with_auto_assign_metrics(
    auth_client, logged_in_user, db_session, monkeypatch,
    dataset_classlabel_shape,
):
    """The 'New leaderboard' form on the dataset page can opt into
    auto-assigned metrics. The handler hands off to
    _auto_create_lb_with_metrics, lands the user on the new LB."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    resp = auth_client.post(
        '/create_leaderboard',
        data={
            'leaderboard_name': 'auto_lb_from_form',
            'dataset_ids': str(dataset_classlabel_shape.id),
            'auto_assign_metrics': '1',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert '/leaderboard/' in resp.headers['Location']

    lb = Leaderboard.query.filter_by(name='auto_lb_from_form').first()
    assert lb is not None
    assert len(lb.leaderboard_metrics) == 1
    assert lb.leaderboard_metrics[0].global_metric.name == 'accuracy_label'


def test_create_leaderboard_without_auto_assign_metrics_path_unchanged(
    auth_client, logged_in_user, db_session, dataset_classlabel_shape,
):
    """When the box is unchecked, the legacy create_leaderboard path
    runs unchanged — empty LB, no metrics attached."""
    resp = auth_client.post(
        '/create_leaderboard',
        data={
            'leaderboard_name': 'manual_lb',
            'dataset_ids': str(dataset_classlabel_shape.id),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert '/leaderboard/' in resp.headers['Location']
    lb = Leaderboard.query.filter_by(name='manual_lb').first()
    assert lb is not None
    assert lb.leaderboard_metrics == []


def test_create_leaderboard_auto_assign_requires_a_dataset(
    auth_client, logged_in_user, db_session,
):
    """Auto-assign needs a dataset to inspect — flash + redirect when
    the form arrives without one (defensive UI fallback)."""
    resp = auth_client.post(
        '/create_leaderboard',
        data={
            'leaderboard_name': 'no_ds',
            'auto_assign_metrics': '1',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert Leaderboard.query.filter_by(name='no_ds').first() is None
