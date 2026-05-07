"""Auto-tagging on HF imports: HF-metadata path + LLM path + combined."""
from unittest.mock import patch

import pytest

from app import (
    _normalize_hf_tags,
    _llm_suggest_tags,
    _auto_tags_for_hf,
)


# ---------------------------------------------------------------------------
# _normalize_hf_tags: filter HF's noisy raw tag list down to discovery tags
# ---------------------------------------------------------------------------


def test_normalize_strips_metadata_prefixes():
    raw = [
        'task_categories:image-classification',
        'language:en',                            # drop
        'size_categories:1K<n<10K',               # drop
        'license:mit',                            # drop
        'modality:image',                         # 'image' → drop (catch-all)
        'arxiv:2304.12345',                       # drop
    ]
    out = _normalize_hf_tags(raw)
    assert out == ['image-classification']


def test_normalize_dedupes_and_caps_at_six():
    raw = [
        'task:depth-estimation', 'depth-estimation',
        'task:segmentation', 'task:detection',
        'task:tracking', 'task:pose-estimation',
        'task:reconstruction', 'task:retrieval',  # 7th — gets capped
    ]
    out = _normalize_hf_tags(raw)
    assert len(out) == 6
    assert out[0] == 'depth-estimation'  # dedupe winner
    assert 'retrieval' not in out         # capped


def test_normalize_drops_empty_and_too_short():
    assert _normalize_hf_tags(['', 'a', '  ', 'foo', 'b']) == ['foo']


# ---------------------------------------------------------------------------
# _llm_suggest_tags: opt-in via ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------


def test_llm_suggest_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert _llm_suggest_tags('foo/bar', [], 'description') == []


def test_llm_suggest_parses_json_array(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text',
                'text': '["depth-estimation", "indoor", "rgb-d"]',
            }]}

    with patch('requests.post', return_value=_Ok()):
        tags = _llm_suggest_tags('a/b', ['x', 'y'], 'A great dataset')
    assert tags == ['depth-estimation', 'indoor', 'rgb-d']


def test_llm_suggest_silently_falls_back_on_error(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    with patch('requests.post', side_effect=RuntimeError('upstream down')):
        assert _llm_suggest_tags('a/b', [], '') == []


def test_llm_suggest_handles_fenced_json(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text',
                'text': '```json\n["medical-imaging", "ct"]\n```',
            }]}
    with patch('requests.post', return_value=_Ok()):
        tags = _llm_suggest_tags('a/b', [], '')
    assert tags == ['medical-imaging', 'ct']


# ---------------------------------------------------------------------------
# _auto_tags_for_hf: combine both sources
# ---------------------------------------------------------------------------


def test_auto_tags_combines_llm_first_then_hf(monkeypatch):
    """LLM-suggested tags lead; HF tags fill in any new ones; cap at 6."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {
                'tags': ['task:depth-estimation', 'language:en', 'size:1k'],
                'description': 'NYU depth dataset',
            }

    class _LLMOk:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text',
                'text': '["indoor", "rgb-d"]',
            }]}

    def fake_get(url, *a, **kw):
        return _MetaOk()
    def fake_post(url, *a, **kw):
        return _LLMOk()

    with patch('requests.get', side_effect=fake_get), \
         patch('requests.post', side_effect=fake_post):
        tags = _auto_tags_for_hf('nyu/depth')

    # LLM suggestions come first; HF-derived 'depth-estimation' tacked on after.
    assert tags[0] == 'indoor'
    assert tags[1] == 'rgb-d'
    assert 'depth-estimation' in tags
    # HF noise prefixes filtered out.
    assert 'language:en' not in tags


def test_auto_tags_works_without_llm(monkeypatch):
    """No ANTHROPIC_API_KEY → still returns HF-derived tags."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {'tags': ['task:segmentation'], 'description': ''}
    with patch('requests.get', return_value=_MetaOk()):
        tags = _auto_tags_for_hf('foo/seg')
    assert tags == ['segmentation']


def test_auto_tags_returns_empty_when_both_sources_empty(monkeypatch):
    """No HF tags + no LLM = no tags. Caller leaves the dataset untagged."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {'tags': [], 'description': ''}
    with patch('requests.get', return_value=_MetaOk()):
        assert _auto_tags_for_hf('blank/repo') == []
