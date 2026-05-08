"""Phase 8: auto-LB preview lets the user augment auto-proposals with
library picks + AI-authored metrics. Backend coverage for the new API
endpoints + the merge into create_leaderboard_auto_finalize."""
import json
from unittest.mock import patch

from app import (
    Dataset, GlobalMetric, Leaderboard, LeaderboardMetric, Sample, db,
    _parse_extra_metrics,
)


# ---------------------------------------------------------------------------
# _parse_extra_metrics — defensive normalization
# ---------------------------------------------------------------------------


def test_parse_extra_metrics_rejects_bad_json():
    assert _parse_extra_metrics("not-json") == []
    assert _parse_extra_metrics("") == []
    assert _parse_extra_metrics("{}") == []  # not a list


def test_parse_extra_metrics_skips_invalid_global_name():
    raw = json.dumps([
        {'global_name': '99bad', 'python_code': 'def 99bad(gt, pred): return 0.0'},
        {'global_name': 'with-dash', 'python_code': 'def with_dash(gt, pred): return 0.0'},
        {'global_name': 'good_one', 'python_code': 'def good_one(gt, pred): return 0.0'},
    ])
    out = _parse_extra_metrics(raw)
    assert len(out) == 1
    assert out[0]['global_name'] == 'good_one'


def test_parse_extra_metrics_requires_def_for_named_function():
    """Code that doesn't define the named function is rejected — would
    crash at metric-eval time."""
    raw = json.dumps([
        {'global_name': 'no_def', 'python_code': 'x = 5'},
        {'global_name': 'wrong_def', 'python_code': 'def other(): pass'},
    ])
    assert _parse_extra_metrics(raw) == []


def test_parse_extra_metrics_normalizes_defaults():
    raw = json.dumps([{
        'global_name': 'mae',
        'python_code': 'def mae(gt, pred): return abs(gt - pred)',
        'description': '',
    }])
    out = _parse_extra_metrics(raw)
    assert out[0]['target_name'] == 'mae'
    assert out[0]['sort_direction'] == 'higher_is_better'
    assert out[0]['pooling_type'] == 'mean'
    assert out[0]['arg_mappings'] == {}


# ---------------------------------------------------------------------------
# Library-metrics endpoint
# ---------------------------------------------------------------------------


def test_library_metrics_requires_login(client, db_session):
    r = client.get('/api/lb_preview/library_metrics', follow_redirects=False)
    assert r.status_code == 302


def test_library_metrics_returns_global_metrics(auth_client, logged_in_user, db_session):
    db.session.add(GlobalMetric(
        name='libtest_mae', description='abs error',
        python_code='def libtest_mae(gt, pred): return abs(gt - pred)',
        is_aggregated=False, owner_user_id=logged_in_user.id,
    ))
    db.session.commit()
    body = auth_client.get('/api/lb_preview/library_metrics').get_json()
    names = [m['name'] for m in body['metrics']]
    assert 'libtest_mae' in names
    row = next(m for m in body['metrics'] if m['name'] == 'libtest_mae')
    assert row['python_code'].startswith('def libtest_mae')


# ---------------------------------------------------------------------------
# LLM-metric endpoint
# ---------------------------------------------------------------------------


def test_llm_metric_requires_description(auth_client, db_session):
    r = auth_client.post('/api/lb_preview/llm_metric',
                         json={'name': 'foo'})
    assert r.status_code == 400
    assert 'description' in r.get_json()['error']


def test_llm_metric_returns_503_without_api_key(auth_client, db_session, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    r = auth_client.post('/api/lb_preview/llm_metric',
                         json={'description': 'mean abs error'})
    assert r.status_code == 503


def test_llm_metric_uses_helper_and_sanitizes_name(auth_client, db_session, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    fake_code = 'def custom_thing(gt, pred):\n    return float(gt - pred)\n'
    with patch('app._llm_generate_metric_code', return_value=fake_code) as m:
        r = auth_client.post('/api/lb_preview/llm_metric', json={
            'name': 'Custom Thing!',  # gets sanitized to 'custom_thing'
            'description': 'Pretty please diff gt and pred',
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body['global_name'] == 'custom_thing'
        assert body['python_code'] == fake_code
        # Helper got called with (sanitized_name, description).
        m.assert_called_once_with('custom_thing', 'Pretty please diff gt and pred')


def test_llm_metric_502_when_helper_returns_none(auth_client, db_session, monkeypatch):
    """If Claude's response doesn't define the named function, the
    helper returns None — we surface a 502 with a helpful message."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    with patch('app._llm_generate_metric_code', return_value=None):
        r = auth_client.post('/api/lb_preview/llm_metric', json={
            'description': 'something the LLM cannot author',
        })
        assert r.status_code == 502


# ---------------------------------------------------------------------------
# auto_finalize merges extra metrics into the LB
# ---------------------------------------------------------------------------


def test_auto_finalize_persists_extra_metric_for_bh_dataset(
    auth_client, logged_in_user, db_session,
):
    ds = Dataset(name='efds', visibility='public', owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.commit()

    extra = [{
        'global_name': 'extra_mae',
        'target_name': 'extra mae',
        'description': 'user-added',
        'python_code': 'def extra_mae(gt, pred):\n    return abs(float(gt) - float(pred))',
        'arg_mappings': {'gt': 'gt_score', 'pred': 'sub_score_pred'},
        'sort_direction': 'lower_is_better',
        'code_source': 'llm',
    }]

    r = auth_client.post('/create_leaderboard/auto_finalize', data={
        'leaderboard_name': 'extra_lb',
        'dataset_id': str(ds.id),
        'extra_metrics_json': json.dumps(extra),
    }, follow_redirects=False)
    assert r.status_code in (302, 303)

    lb = Leaderboard.query.filter_by(name='extra_lb').first()
    assert lb is not None
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    assert any(lm.target_name == 'extra mae' for lm in lms)
    gm = GlobalMetric.query.filter_by(name='extra_mae').first()
    assert gm is not None
    assert 'def extra_mae' in gm.python_code
