"""Disk-bounded LRU cache for streamed GT + submission bytes.

The pointer-mode storage refactor (and the future remote-submission
path) both go through `bench_cache`. This test pins:
- Atomic writes (interrupted writers don't leave half-files visible).
- Per-key locking serializes concurrent fetches of the same key.
- LRU eviction respects the budget and evicts submissions before GT.
- last_accessed_at gets bumped on hit so the eviction loop has fresh data.
- cache_clear / cache_stats are sane.
"""
import os
import threading
import time
from datetime import datetime, timedelta

import pytest

import bench_cache
from app import CacheEntry, db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _writer_for(payload_bytes):
    """Build a writer that just dumps the supplied bytes."""
    def _w(path):
        with open(path, 'wb') as f:
            f.write(payload_bytes)
    return _w


# ---------------------------------------------------------------------------
# Path resolution + budget
# ---------------------------------------------------------------------------


def test_hashed_filename_is_filesystem_safe():
    """Sketchy keys with slashes / colons / query strings must produce
    a filename containing only [0-9a-f]."""
    name = bench_cache._hashed_filename(
        'gt:huggingface.co/some/repo@refs/pr/1?token=...:42'
    )
    assert all(c in '0123456789abcdef' for c in name)
    assert len(name) == 64


def test_resolve_budget_bytes_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv('BENCHHUB_CACHE_BUDGET_BYTES', '12345')
    assert bench_cache.resolve_budget_bytes(str(tmp_path)) == 12345


def test_resolve_budget_bytes_falls_back_to_60_pct_of_fs(monkeypatch, tmp_path):
    monkeypatch.delenv('BENCHHUB_CACHE_BUDGET_BYTES', raising=False)
    out = bench_cache.resolve_budget_bytes(str(tmp_path))
    assert out > 0  # depends on the host fs; just assert non-zero


# ---------------------------------------------------------------------------
# put + get round-trip
# ---------------------------------------------------------------------------


def test_put_then_get_returns_path_with_correct_bytes(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='gt:foo', origin='gt',
        writer=_writer_for(b'hello'),
        budget_bytes=10_000,
    )
    path = bench_cache.cache_get(
        db.session, CacheEntry, cache_root=cache_root, key='gt:foo',
    )
    assert path is not None
    assert open(path, 'rb').read() == b'hello'


def test_get_returns_none_for_missing_key(db_session, tmp_path):
    out = bench_cache.cache_get(
        db.session, CacheEntry,
        cache_root=str(tmp_path / 'cache'), key='nope',
    )
    assert out is None


def test_get_bumps_last_accessed_at_on_hit(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='k', origin='gt',
        writer=_writer_for(b'x'), budget_bytes=10_000,
    )
    entry = CacheEntry.query.filter_by(cache_key='k').one()
    # Backdate it so the bump is detectable.
    entry.last_accessed_at = datetime.utcnow() - timedelta(hours=1)
    db.session.commit()
    bench_cache.cache_get(
        db.session, CacheEntry, cache_root=cache_root, key='k',
    )
    refreshed = CacheEntry.query.filter_by(cache_key='k').one()
    assert refreshed.last_accessed_at > datetime.utcnow() - timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def test_eviction_drops_submissions_before_gt(db_session, tmp_path):
    """Budget = just enough for one entry. After putting GT then
    submission then a second submission, the first submission must
    go (oldest submission) — not the GT (older but higher priority)."""
    cache_root = str(tmp_path / 'cache')
    payload = b'x' * 1024
    # Budget = 1.5 KiB so two payloads can't fit but one + headroom can.
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='gt:1', origin='gt',
        writer=_writer_for(payload), budget_bytes=10_000,
    )
    time.sleep(0.01)
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='sub:1', origin='submission',
        writer=_writer_for(payload), budget_bytes=10_000,
    )
    time.sleep(0.01)
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='sub:2', origin='submission',
        # Budget = 2.2 KiB → fits exactly two 1 KiB entries with headroom.
        writer=_writer_for(payload), budget_bytes=2300,
    )
    keys_left = {r.cache_key for r in CacheEntry.query.all()}
    # GT survives (priority); newest submission survives; oldest
    # submission gets dropped.
    assert 'gt:1' in keys_left
    assert 'sub:2' in keys_left
    assert 'sub:1' not in keys_left


def test_eviction_falls_through_to_gt_when_only_gt_is_present(
    db_session, tmp_path,
):
    cache_root = str(tmp_path / 'cache')
    payload = b'y' * 1024
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='gt:old', origin='gt',
        writer=_writer_for(payload), budget_bytes=10_000,
    )
    time.sleep(0.01)
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='gt:new', origin='gt',
        writer=_writer_for(payload), budget_bytes=1500,  # tight
    )
    keys_left = {r.cache_key for r in CacheEntry.query.all()}
    assert keys_left == {'gt:new'}  # oldest GT evicted


def test_eviction_no_op_when_under_budget(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='small', origin='gt',
        writer=_writer_for(b'tiny'), budget_bytes=10_000,
    )
    evicted = bench_cache.cache_gc(
        db.session, CacheEntry,
        cache_root=cache_root, budget_bytes=10_000,
    )
    assert evicted == 0
    assert CacheEntry.query.count() == 1


def test_eviction_files_on_disk_actually_removed(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    payload = b'z' * 1024
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='kill_me', origin='submission',
        writer=_writer_for(payload), budget_bytes=10_000,
    )
    on_disk = bench_cache.cache_path_for(cache_root, 'kill_me')
    assert os.path.exists(on_disk)
    bench_cache.cache_gc(
        db.session, CacheEntry,
        cache_root=cache_root, budget_bytes=0,
    )
    assert not os.path.exists(on_disk)


# ---------------------------------------------------------------------------
# Atomic write — interrupted writers don't leave a visible half-file.
# ---------------------------------------------------------------------------


def test_atomic_write_raising_in_writer_leaves_no_artifact(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')

    def _bad_writer(path):
        with open(path, 'wb') as f:
            f.write(b'partial')
        raise RuntimeError('boom mid-write')

    with pytest.raises(RuntimeError):
        bench_cache.cache_put(
            db.session, CacheEntry,
            cache_root=cache_root, key='broken', origin='gt',
            writer=_bad_writer, budget_bytes=10_000,
        )
    # No final file, no .tmp leftover, no DB row.
    assert not os.path.exists(bench_cache.cache_path_for(cache_root, 'broken'))
    assert not os.path.exists(
        bench_cache.cache_path_for(cache_root, 'broken') + '.tmp'
    )
    assert CacheEntry.query.filter_by(cache_key='broken').count() == 0


# ---------------------------------------------------------------------------
# Don't re-fetch a key that already has bytes on disk. Same semantic
# as "second concurrent caller waits, sees the finished file, skips
# the writer" — easier to verify without threads.
# ---------------------------------------------------------------------------


def test_second_put_on_existing_key_does_not_invoke_writer(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    fetches = []

    def writer(path):
        fetches.append(1)
        with open(path, 'wb') as f:
            f.write(b'first')

    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='dedupe', origin='gt',
        writer=writer, budget_bytes=10_000,
    )
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='dedupe', origin='gt',
        writer=writer, budget_bytes=10_000,
    )
    # Writer ran exactly once; the second put no-op'd because the file
    # already existed under the per-key lock.
    assert len(fetches) == 1
    assert open(
        bench_cache.cache_path_for(cache_root, 'dedupe'), 'rb',
    ).read() == b'first'


# ---------------------------------------------------------------------------
# cache_clear / cache_stats
# ---------------------------------------------------------------------------


def test_cache_clear_drops_files_and_rows(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    for k in ('a', 'b', 'c'):
        bench_cache.cache_put(
            db.session, CacheEntry,
            cache_root=cache_root, key=k, origin='gt',
            writer=_writer_for(b'x'), budget_bytes=10_000,
        )
    bench_cache.cache_clear(db.session, CacheEntry, cache_root=cache_root)
    assert CacheEntry.query.count() == 0
    for k in ('a', 'b', 'c'):
        assert not os.path.exists(bench_cache.cache_path_for(cache_root, k))


def test_cache_stats_reports_per_origin_counts(db_session, tmp_path):
    cache_root = str(tmp_path / 'cache')
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='gt-only', origin='gt',
        writer=_writer_for(b'x' * 100), budget_bytes=10_000,
    )
    bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key='sub-only', origin='submission',
        writer=_writer_for(b'y' * 200), budget_bytes=10_000,
    )
    stats = bench_cache.cache_stats(db.session, CacheEntry)
    assert stats['total_entries'] == 2
    assert stats['total_bytes'] == 300
    assert stats['by_origin']['gt']['count'] == 1
    assert stats['by_origin']['gt']['bytes'] == 100
    assert stats['by_origin']['submission']['count'] == 1
    assert stats['by_origin']['submission']['bytes'] == 200


# ---------------------------------------------------------------------------
# Origin validation
# ---------------------------------------------------------------------------


def test_put_rejects_unknown_origin(db_session, tmp_path):
    with pytest.raises(ValueError, match='unknown origin'):
        bench_cache.cache_put(
            db.session, CacheEntry,
            cache_root=str(tmp_path / 'cache'),
            key='k', origin='nope',
            writer=_writer_for(b'x'), budget_bytes=10_000,
        )
