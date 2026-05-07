"""Auto-tagging on HF imports.

Contract: every auto-tagged dataset gets at most TWO discovery tags —
a primary task category from a fixed vocabulary plus an optional
qualifier. The previous "up to 6 vague tags from union(HF, LLM)"
behavior is gone; tags are intentionally tiny so /explore stays
scannable.
"""
from unittest.mock import patch

import pytest

from app import (
    _PRIMARY_TASK_TAGS,
    _normalize_hf_tags,
    _heuristic_primary_tag,
    _heuristic_qualifier_tag,
    _llm_suggest_tags,
    _auto_tags_for_hf,
)


# ---------------------------------------------------------------------------
# _normalize_hf_tags: strip noise prefixes, return a flat list (input to the
# heuristic — NOT the dataset's final tag set).
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


def test_normalize_dedupes():
    raw = [
        'task:depth-estimation', 'depth-estimation',  # same after strip
        'task:segmentation',
    ]
    out = _normalize_hf_tags(raw)
    # Dedupe still applies; cap is no longer enforced here (the old
    # 6-tag cap is gone — _auto_tags_for_hf collapses to ≤ 2 instead).
    assert out == ['depth-estimation', 'segmentation']


def test_normalize_drops_empty_and_too_short():
    assert _normalize_hf_tags(['', 'a', '  ', 'foo', 'b']) == ['foo']


# ---------------------------------------------------------------------------
# Heuristic primary + qualifier: deterministic fallback when no API key.
# ---------------------------------------------------------------------------


def test_heuristic_primary_maps_hf_task_tag_onto_vocabulary():
    assert _heuristic_primary_tag(['image-classification']) == 'classification'
    assert _heuristic_primary_tag(['object-detection']) == 'detection'
    assert _heuristic_primary_tag(['depth-estimation']) == 'depth'
    assert _heuristic_primary_tag(['semantic-segmentation']) == 'segmentation'


def test_heuristic_primary_returns_none_when_nothing_matches():
    assert _heuristic_primary_tag(['some-random-tag', 'another']) is None


def test_heuristic_qualifier_picks_known_modifier():
    assert _heuristic_qualifier_tag(['stereo', 'rgb-d'], primary='depth') == 'stereo'
    assert _heuristic_qualifier_tag(['indoor'], primary='depth') == 'indoor'


def test_heuristic_qualifier_skips_unknown_modifiers():
    assert _heuristic_qualifier_tag(['rgb-d', 'kinect'], primary='depth') is None


def test_heuristic_qualifier_doesnt_duplicate_primary():
    assert _heuristic_qualifier_tag(['depth'], primary='depth') is None


# ---------------------------------------------------------------------------
# _llm_suggest_tags: returns [] or [primary] or [primary, qualifier].
# ---------------------------------------------------------------------------


def test_llm_suggest_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert _llm_suggest_tags('foo/bar', [], 'description') == []


def test_llm_suggest_accepts_primary_and_qualifier(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text',
                'text': '["depth", "stereo"]',
            }]}

    with patch('requests.post', return_value=_Ok()):
        tags = _llm_suggest_tags('a/b', ['x', 'y'], 'A stereo depth dataset')
    assert tags == ['depth', 'stereo']


def test_llm_suggest_accepts_primary_only(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text', 'text': '["segmentation"]',
            }]}

    with patch('requests.post', return_value=_Ok()):
        tags = _llm_suggest_tags('a/b', [], '')
    assert tags == ['segmentation']


def test_llm_suggest_rejects_invalid_primary(monkeypatch):
    """Primary tag must be in _PRIMARY_TASK_TAGS; off-vocab → []."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text', 'text': '["chocolate", "vanilla"]',
            }]}

    with patch('requests.post', return_value=_Ok()):
        assert _llm_suggest_tags('a/b', [], '') == []


def test_llm_suggest_caps_at_two_tags(monkeypatch):
    """Even if the model emits 5 tags, only the first two are kept."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text',
                'text': '["depth", "indoor", "rgb-d", "kinect", "scenes"]',
            }]}

    with patch('requests.post', return_value=_Ok()):
        tags = _llm_suggest_tags('a/b', [], '')
    assert len(tags) <= 2
    assert tags[0] == 'depth'


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
                'type': 'text', 'text': '```json\n["classification", "medical"]\n```',
            }]}
    with patch('requests.post', return_value=_Ok()):
        tags = _llm_suggest_tags('a/b', [], '')
    assert tags == ['classification', 'medical']


# ---------------------------------------------------------------------------
# _auto_tags_for_hf: end-to-end, with + without LLM.
# ---------------------------------------------------------------------------


def test_auto_tags_uses_llm_when_available(monkeypatch):
    """LLM picks the primary + qualifier; HF tags are not unioned in."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {
                'tags': ['task:depth-estimation', 'language:en', 'size:1k'],
                'description': 'NYU stereo depth dataset',
            }

    class _LLMOk:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text', 'text': '["depth", "stereo"]',
            }]}

    def fake_get(*a, **kw): return _MetaOk()
    def fake_post(*a, **kw): return _LLMOk()

    with patch('requests.get', side_effect=fake_get), \
         patch('requests.post', side_effect=fake_post):
        tags = _auto_tags_for_hf('nyu/depth')

    assert tags == ['depth', 'stereo']


def test_auto_tags_heuristic_when_no_llm(monkeypatch):
    """No ANTHROPIC_API_KEY → heuristic maps the HF task tag onto the
    primary vocabulary."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {'tags': ['task_categories:image-segmentation'],
                    'description': ''}
    with patch('requests.get', return_value=_MetaOk()):
        tags = _auto_tags_for_hf('foo/seg')
    assert tags == ['segmentation']


def test_auto_tags_heuristic_picks_qualifier_when_compatible(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {
                'tags': ['task:depth-estimation', 'stereo', 'rgb-d'],
                'description': '',
            }
    with patch('requests.get', return_value=_MetaOk()):
        tags = _auto_tags_for_hf('foo/stereo-depth')
    assert tags == ['depth', 'stereo']


def test_auto_tags_returns_empty_when_nothing_matches(monkeypatch):
    """No primary mapping + no LLM → leave dataset untagged."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {'tags': ['weird-tag-not-in-vocab'], 'description': ''}
    with patch('requests.get', return_value=_MetaOk()):
        assert _auto_tags_for_hf('blank/repo') == []


def test_auto_tags_falls_back_to_heuristic_when_llm_returns_empty(monkeypatch):
    """LLM said 'I cannot classify confidently' → heuristic still tries."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {'tags': ['task:depth-estimation'], 'description': ''}

    class _LLMEmpty:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': '[]'}]}

    with patch('requests.get', return_value=_MetaOk()), \
         patch('requests.post', return_value=_LLMEmpty()):
        tags = _auto_tags_for_hf('a/b')
    assert tags == ['depth']


def test_auto_tags_caps_at_two_globally(monkeypatch):
    """Even if internals slip up, the public API never returns > 2."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _MetaOk:
        def raise_for_status(self): pass
        def json(self):
            return {'tags': [], 'description': ''}

    class _LLMTriple:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{
                'type': 'text', 'text': '["classification", "medical", "ct"]',
            }]}

    with patch('requests.get', return_value=_MetaOk()), \
         patch('requests.post', return_value=_LLMTriple()):
        tags = _auto_tags_for_hf('a/b')
    assert len(tags) <= 2


def test_primary_vocabulary_is_locked_down():
    """Defensive: pin the vocabulary so a future refactor can't quietly
    add tags that would re-introduce sprawl."""
    assert 'depth' in _PRIMARY_TASK_TAGS
    assert 'segmentation' in _PRIMARY_TASK_TAGS
    assert 'classification' in _PRIMARY_TASK_TAGS
    assert 'language' in _PRIMARY_TASK_TAGS
    # The old generic 'image' / 'machine-learning' / 'benchmark' tags
    # are explicitly NOT primary categories — they were the bloat
    # source we just got rid of.
    assert 'image' not in _PRIMARY_TASK_TAGS
    assert 'machine-learning' not in _PRIMARY_TASK_TAGS
    assert 'benchmark' not in _PRIMARY_TASK_TAGS
