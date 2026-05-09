"""Phase 13: LLM-driven text-task proposer dispatches different
metric+viz suites for classification vs generation vs QA. Static
top1+F1+confusion is the fallback when no API key / LLM call fails.
"""
import json
from unittest.mock import patch

import pytest

import app as app_mod
from app import (
    Dataset, Sample, CustomField, db,
    _propose_metrics_for_dataset, _propose_visualizations_for_dataset,
    _llm_propose_text_evaluation_suite,
)


@pytest.fixture(autouse=True)
def clear_suite_cache():
    """Cache is keyed by id(ds)+col — different fixtures reuse ids
    after teardown, which can resurface stale suites in adjacent tests.
    Clear between every test."""
    app_mod._TEXT_EVAL_SUITE_CACHE.clear()
    yield
    app_mod._TEXT_EVAL_SUITE_CACHE.clear()


def _seed_text_ds(value, name='label', ds_name=None):
    ds = Dataset(name=ds_name or f'tprop_{name}', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name=name, field_type='text', value_text=value,
    ))
    db.session.commit()
    return ds


# ---------------------------------------------------------------------------
# Falls back to static top1+F1+confusion when no API key
# ---------------------------------------------------------------------------


def test_falls_back_to_static_when_no_api_key(client, db_session, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    ds = _seed_text_ds('pos', name='sentiment')
    metrics = _propose_metrics_for_dataset(ds)
    names = {p['global_name'] for p in metrics}
    assert 'top1_text' in names
    assert 'macro_f1_text' in names
    vizes = _propose_visualizations_for_dataset(ds)
    assert any(v['global_name'] == 'confusion_matrix_text' for v in vizes)


# ---------------------------------------------------------------------------
# LLM suite is honored when the call succeeds
# ---------------------------------------------------------------------------


def _fake_resp(suite_json):
    """Build a mocked Anthropic response shape that returns one text
    block carrying our suite JSON."""
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': json.dumps(suite_json)}]}
    return _R()


def test_llm_classification_suite_passes_through(client, db_session, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    suite = {
        'task_type': 'classification',
        'metrics': [
            {
                'global_name': 'accuracy',
                'target_name': 'accuracy',
                'description': 'Top-1 accuracy on text labels.',
                'sort_direction': 'higher_is_better',
                'is_aggregated': False,
                'python_code': 'def accuracy(gt, pred):\n    return 1.0 if str(gt).strip() == str(pred).strip() else 0.0\n',
            },
        ],
        'visualization': {
            'global_name': 'confusion_matrix_v2',
            'target_name': 'confusion matrix',
            'description': 'Aggregated confusion matrix.',
            'python_code': 'def confusion_matrix_v2(gt, pred):\n    from PIL import Image as _PILImage\n    return _PILImage.new("L", (256, 256), 0)\n',
        },
    }
    with patch('requests.post', return_value=_fake_resp(suite)):
        ds = _seed_text_ds('pos', name='sentiment',
                           ds_name='llm_classification_pass')
        metrics = _propose_metrics_for_dataset(ds)
        names = {p['global_name'] for p in metrics}
        # LLM-authored metric wins; static top1_text/macro_f1_text NOT used.
        assert 'accuracy' in names
        assert 'top1_text' not in names
        assert 'macro_f1_text' not in names

        vizes = _propose_visualizations_for_dataset(ds)
        viz_names = {v['global_name'] for v in vizes}
        assert 'confusion_matrix_v2' in viz_names
        assert 'confusion_matrix_text' not in viz_names


def test_llm_generation_suite_skips_confusion_matrix(client, db_session, monkeypatch):
    """For free-form generation, the LLM might (correctly) decline to
    propose a viz. The static fallback shouldn't kick in — that would
    second-guess the LLM's deliberate omission."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    suite = {
        'task_type': 'generation',
        'metrics': [
            {
                'global_name': 'rouge_l',
                'target_name': 'ROUGE-L',
                'description': 'Aggregated ROUGE-L F-measure.',
                'sort_direction': 'higher_is_better',
                'is_aggregated': True,
                'python_code': 'def rouge_l(gt, pred):\n    return 0.5\n',
            },
        ],
        'visualization': None,
    }
    with patch('requests.post', return_value=_fake_resp(suite)):
        ds = _seed_text_ds('A long-form sentence about something.',
                           name='completion', ds_name='llm_gen_no_viz')
        metrics = _propose_metrics_for_dataset(ds)
        assert {p['global_name'] for p in metrics} == {'rouge_l'}
        assert metrics[0]['is_aggregated'] is True

        vizes = _propose_visualizations_for_dataset(ds)
        # No viz proposed for this column.
        assert all(v.get('arg_mappings', {}).get('gt') != 'gt_completion'
                   for v in vizes)


# ---------------------------------------------------------------------------
# LLM call failure / malformed response → static fallback
# ---------------------------------------------------------------------------


def test_falls_back_when_llm_returns_garbage(client, db_session, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    class _BadR:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': 'not-json {{{'}]}
    with patch('requests.post', return_value=_BadR()):
        ds = _seed_text_ds('pos', name='sentiment',
                           ds_name='llm_garbage_fallback')
        metrics = _propose_metrics_for_dataset(ds)
        names = {p['global_name'] for p in metrics}
        assert 'top1_text' in names
        assert 'macro_f1_text' in names


def test_falls_back_when_llm_metrics_have_invalid_code(client, db_session, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    suite = {
        'task_type': 'classification',
        'metrics': [{
            'global_name': 'broken',
            'target_name': 'broken',
            'description': 'no def line',
            'sort_direction': 'higher_is_better',
            'is_aggregated': False,
            'python_code': 'x = 5',  # missing def
        }],
        'visualization': None,
    }
    with patch('requests.post', return_value=_fake_resp(suite)):
        ds = _seed_text_ds('pos', name='sentiment',
                           ds_name='llm_invalid_code_fallback')
        metrics = _propose_metrics_for_dataset(ds)
        names = {p['global_name'] for p in metrics}
        # All LLM metrics rejected by validation → fall through to static.
        assert 'top1_text' in names


# ---------------------------------------------------------------------------
# Cache: metric + viz proposers share one LLM call per (ds, col)
# ---------------------------------------------------------------------------


def test_metric_and_viz_share_single_llm_call(client, db_session, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    suite = {
        'task_type': 'classification',
        'metrics': [{
            'global_name': 'accuracy',
            'target_name': 'accuracy',
            'description': '',
            'sort_direction': 'higher_is_better',
            'is_aggregated': False,
            'python_code': 'def accuracy(gt, pred):\n    return 1.0\n',
        }],
        'visualization': None,
    }
    ds = _seed_text_ds('pos', name='sentiment', ds_name='cache_one_call')
    with patch('requests.post', return_value=_fake_resp(suite)) as mock:
        _propose_metrics_for_dataset(ds)
        _propose_visualizations_for_dataset(ds)
        assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Direct unit test on the helper
# ---------------------------------------------------------------------------


def test_helper_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert _llm_propose_text_evaluation_suite('label', 'pos') is None


def test_helper_returns_none_when_all_metrics_invalid(monkeypatch):
    """If every LLM-proposed metric fails validation, return None so
    the caller falls back to static — don't return an empty suite that
    suppresses both LLM and static."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    suite = {
        'task_type': 'classification',
        'metrics': [{
            'global_name': '99bad',  # invalid identifier
            'target_name': 'b', 'description': 'b',
            'sort_direction': 'higher_is_better', 'is_aggregated': False,
            'python_code': 'def f(): pass',
        }],
        'visualization': None,
    }
    with patch('requests.post', return_value=_fake_resp(suite)):
        out = _llm_propose_text_evaluation_suite('label', 'pos')
    assert out is None
