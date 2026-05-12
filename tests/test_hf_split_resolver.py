"""Tests for `_resolve_hf_split_and_load` — the helper that walks
_HF_SPLIT_PREFERENCE (test → validation → val → dev → train), probes
row 0 to skip GT-less splits, and persists the resolved split back
onto Attachment.hf_split.

These tests use a fake `load_fn` callable instead of touching HF, so
they're fast and offline-safe. We instantiate a minimal Attachment-
shaped object (just `hf_repo_id`, `hf_split`, `hf_mapping_json`) since
that's all the resolver reads — no DB needed."""
import json
import types

import pytest

from app import _resolve_hf_split_and_load, _HF_SPLIT_PREFERENCE


def _make_att(*, hf_split=None, mapping=None):
    """Minimal Attachment stand-in. We don't need a real SQLAlchemy
    row because the resolver only reads four attributes and calls
    _persist_resolved_split() which is a no-op without a committed
    session — we patch it out via the on_log callback chain."""
    return types.SimpleNamespace(
        hf_repo_id='fake/repo',
        hf_split=hf_split,
        hf_revision=None,
        hf_mapping_json=json.dumps(mapping or []),
    )


class _ReIterable:
    """Stand-in for a streaming HF Dataset: `iter(self)` returns a
    fresh iterator each call. The resolver does a `next(iter(ds))`
    probe and then returns `ds`; the caller later does another full
    iteration — a single-use generator would yield empty the second
    time."""
    def __init__(self, rows):
        self._rows = list(rows)
    def __iter__(self):
        return iter(self._rows)


def _fake_load_fn(split_to_rows):
    """Build a load_fn whose return value for each split is a
    re-iterable wrapper over the given rows. Raises ValueError with
    the standard "Bad split: ... Available splits: [...]" message when
    the asked-for split isn't present."""
    available = list(split_to_rows.keys())

    def _load(split):
        if split not in split_to_rows:
            avail_str = ', '.join(repr(s) for s in available)
            raise ValueError(f"Bad split: {split}. Available splits: [{avail_str}]")
        return _ReIterable(split_to_rows[split])

    return _load


def test_picks_test_when_all_splits_available_with_full_gt(monkeypatch):
    """Preference order is test > validation > train. When all three
    load AND all three have GT in the mapped column, test wins."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att(mapping=[
        {'column': 'label', 'target_kind': 'scalar'},
    ])
    load_fn = _fake_load_fn({
        'test':       [{'label': 1}],
        'validation': [{'label': 2}],
        'train':      [{'label': 3}],
    })
    ds = _resolve_hf_split_and_load(att, load_fn)
    rows = list(ds)
    assert rows == [{'label': 1}]  # came from `test`


def test_skips_split_with_null_gt_and_falls_through(monkeypatch):
    """If `test` row 0 has the mapped GT column as None, skip and try
    `validation`. The classic case: a contest test split where labels
    are withheld."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att(mapping=[
        {'column': 'label', 'target_kind': 'scalar'},
    ])
    load_fn = _fake_load_fn({
        'test':       [{'label': None}],      # withheld
        'validation': [{'label': 42}],        # has GT
    })
    ds = _resolve_hf_split_and_load(att, load_fn)
    rows = list(ds)
    assert rows == [{'label': 42}]


def test_falls_back_to_loadable_split_when_no_split_has_gt(monkeypatch):
    """When every preferred split lacks GT, return the first loadable
    one as a last resort — better SOME data than none."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att(mapping=[
        {'column': 'label', 'target_kind': 'scalar'},
    ])
    load_fn = _fake_load_fn({
        'test':       [{'label': None}],
        'validation': [{'label': None}],
        'train':      [{'label': None}],
    })
    ds = _resolve_hf_split_and_load(att, load_fn)
    # Got fallback — test split loaded first, so test's row.
    rows = list(ds)
    assert rows == [{'label': None}]


def test_returns_none_when_no_split_loads(monkeypatch):
    """All splits raise non-retryable errors → return None."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att()
    def _load(split):
        raise ValueError(
            f"Bad split: {split}. Available splits: []"
        )
    assert _resolve_hf_split_and_load(att, _load) is None


def test_explicit_hf_split_hint_promoted_to_front(monkeypatch):
    """When the attachment carries an explicit split hint that's in
    the preference list, try that one FIRST, not test. Honors a real
    user override."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att(hf_split='train', mapping=[
        {'column': 'label', 'target_kind': 'scalar'},
    ])
    load_fn = _fake_load_fn({
        'test':  [{'label': 1}],
        'train': [{'label': 2}],
    })
    ds = _resolve_hf_split_and_load(att, load_fn)
    rows = list(ds)
    assert rows == [{'label': 2}]  # train was tried first because of the hint


def test_exotic_split_name_from_error_gets_added_to_order(monkeypatch):
    """Some repos ship unusual split names (DocRED's `train_annotated`).
    The resolver should learn from the "Available splits:" error and
    try those too."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att(mapping=[
        {'column': 'label', 'target_kind': 'scalar'},
    ])
    load_fn = _fake_load_fn({
        'train_annotated': [{'label': 7}],
    })
    ds = _resolve_hf_split_and_load(att, load_fn)
    rows = list(ds)
    assert rows == [{'label': 7}]


def test_auth_errors_propagate(monkeypatch):
    """401 / gated / restricted errors must NOT be swallowed — they
    drive the LB to surface a 'request HF token' affordance."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att()
    def _load(split):
        raise RuntimeError("401 Unauthorized: dataset is gated")
    with pytest.raises(RuntimeError, match='401'):
        _resolve_hf_split_and_load(att, _load)


def test_no_mapping_means_no_gt_probe(monkeypatch):
    """When the mapping is empty (no GT columns to probe), the
    resolver picks the first loadable split without inspecting row 0."""
    monkeypatch.setattr('app._persist_resolved_split', lambda *a, **kw: None)
    att = _make_att(mapping=[])
    load_fn = _fake_load_fn({
        'test': [{'foo': 'bar'}],
    })
    ds = _resolve_hf_split_and_load(att, load_fn)
    rows = list(ds)
    assert rows == [{'foo': 'bar'}]


def test_preference_order_constant_shape():
    """Anchor the preference order. If you intentionally reorder it,
    update this assertion + the LB-detail badge tooltip + CLAUDE.md."""
    assert _HF_SPLIT_PREFERENCE == ['test', 'validation', 'val', 'dev', 'train']
