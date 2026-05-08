"""Disk-bounded LRU cache for streamed GT + submission bytes.

The pointer-mode storage refactor stops cloning whole HF datasets to
the volume. Instead, individual rows are fetched on demand and stashed
here. This module is the only thing that touches the cache directory;
both the GT-streaming path and the future remote-submission path go
through it.

Key shape: callers pass any string. We hash it for the on-disk file
name so colons, slashes, query-strings etc. are safe. The original
key stays in the DB row for debugging.

Origin tag (`'gt' | 'submission'`) drives eviction priority:
submissions evict first when budget is tight (they're cheap to
re-fetch from the user's external store), GT second.

Concurrency: per-key file locks via `fcntl.flock` keep two metric
jobs from racing on the same fetch. A second caller hitting the same
key while a fetch is in flight blocks until the first one writes the
final file, then reads it.

Budget: `BENCHHUB_CACHE_BUDGET_BYTES` env var, defaults to 60% of the
filesystem the cache lives on. Eviction runs on every successful put
plus a periodic Celery-beat tick.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Budget resolution
# ---------------------------------------------------------------------------


def _default_budget_bytes(cache_root: str) -> int:
    """Default to 60% of the filesystem `cache_root` lives on. Falls
    back to 5 GiB if statvfs is unavailable (e.g. Windows in tests)."""
    try:
        st = os.statvfs(cache_root)
        return int(st.f_blocks * st.f_frsize * 0.6)
    except (OSError, AttributeError):
        return 5 * 1024 * 1024 * 1024


def resolve_budget_bytes(cache_root: str) -> int:
    env = os.environ.get('BENCHHUB_CACHE_BUDGET_BYTES')
    if env:
        try:
            return max(int(env), 0)
        except ValueError:
            pass
    return _default_budget_bytes(cache_root)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _hashed_filename(key: str) -> str:
    """Filesystem-safe filename for a cache key. SHA-256 keeps
    everything ASCII + collision-free across the keyspace we care about."""
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def cache_path_for(cache_root: str, key: str) -> str:
    """Where the bytes for `key` live on disk (whether or not it
    currently exists)."""
    return os.path.join(cache_root, _hashed_filename(key))


# ---------------------------------------------------------------------------
# Per-key locking
# ---------------------------------------------------------------------------


@contextmanager
def _key_lock(cache_root: str, key: str):
    """Exclusive lock per cache key. Two callers hitting the same
    missing key serialize: the first fetches, the rest wait + read."""
    lock_dir = os.path.join(cache_root, '_locks')
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, _hashed_filename(key) + '.lock')
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write_from_callback(dest_path: str,
                                writer: Callable[[str], None]) -> int:
    """Write to `<dest_path>.tmp` via `writer(tmp_path)`, fsync, then
    atomic-rename to `dest_path`. Returns the final size in bytes.
    If `writer` raises, the .tmp file is cleaned up."""
    tmp = dest_path + '.tmp'
    try:
        writer(tmp)
        # fsync the file to catch storage-layer eccentricities before
        # we let other readers see it.
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(tmp, dest_path)
    except Exception:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    return os.path.getsize(dest_path)


# ---------------------------------------------------------------------------
# Public API — cache_get / cache_put / cache_gc
# These take the SQLAlchemy session + CacheEntry model as args so the
# module stays decoupled from app.py's import graph. The Flask app
# wires it up in a one-line shim.
# ---------------------------------------------------------------------------


def cache_get(session, CacheEntry, *, cache_root: str, key: str) -> Optional[str]:
    """Return the on-disk path to the cached bytes for `key`, or None
    if missing / corrupt. Bumps `last_accessed_at` on hit so the LRU
    eviction loop has fresh data."""
    path = cache_path_for(cache_root, key)
    if not os.path.exists(path):
        return None
    entry = session.query(CacheEntry).filter_by(cache_key=key).first()
    if entry is None:
        # Disk file exists but no DB record (migration / manual fiddling).
        # Treat as a hit but skip the recency bump so the eviction loop
        # picks it up cleanly when budget tightens.
        return path
    entry.last_accessed_at = datetime.utcnow()
    try:
        session.commit()
    except Exception:
        session.rollback()
    return path


def cache_put(session, CacheEntry, *, cache_root: str, key: str,
              writer: Callable[[str], None], origin: str,
              budget_bytes: Optional[int] = None) -> str:
    """Materialize bytes for `key` via `writer(tmp_path)`, register the
    cache entry, and run the eviction loop. Returns the on-disk path.

    `writer` is any callable that writes the bytes to a path the
    function passes in (lets the caller stream from HF / a remote
    URL / wherever without intermediate buffers).

    Per-key lock prevents duplicate concurrent fetches.
    """
    if origin not in ('gt', 'submission'):
        raise ValueError(f"unknown origin {origin!r}")
    os.makedirs(cache_root, exist_ok=True)
    dest = cache_path_for(cache_root, key)
    with _key_lock(cache_root, key):
        # Re-check after acquiring the lock — the call we were waiting
        # on may have already produced the file.
        if os.path.exists(dest):
            entry = session.query(CacheEntry).filter_by(cache_key=key).first()
            if entry is not None:
                entry.last_accessed_at = datetime.utcnow()
                try:
                    session.commit()
                except Exception:
                    session.rollback()
            return dest
        size = _atomic_write_from_callback(dest, writer)
    # Register the DB row OUTSIDE the lock so concurrent gets see the
    # finished file regardless of session-commit timing.
    entry = session.query(CacheEntry).filter_by(cache_key=key).first()
    now = datetime.utcnow()
    if entry is None:
        entry = CacheEntry(
            cache_key=key, size_bytes=size, origin=origin,
            last_accessed_at=now, created_at=now,
        )
        session.add(entry)
    else:
        entry.size_bytes = size
        entry.origin = origin
        entry.last_accessed_at = now
    session.commit()

    # Run eviction. Submissions evict first when over budget, GT second.
    if budget_bytes is None:
        budget_bytes = resolve_budget_bytes(cache_root)
    if budget_bytes > 0:
        cache_gc(session, CacheEntry, cache_root=cache_root,
                 budget_bytes=budget_bytes)
    return dest


def cache_gc(session, CacheEntry, *, cache_root: str,
             budget_bytes: int) -> int:
    """Evict oldest entries until total cached bytes are under
    `budget_bytes`. Submissions evicted first (cheaper to re-fetch),
    then GT. Returns the number of entries evicted."""
    rows = session.query(CacheEntry).all()
    total_bytes = sum(r.size_bytes or 0 for r in rows)
    if total_bytes <= budget_bytes:
        return 0
    # Two-pass eviction: kill submissions oldest-first, then GT
    # oldest-first if still over.
    submissions = sorted(
        [r for r in rows if r.origin == 'submission'],
        key=lambda r: r.last_accessed_at or datetime.min,
    )
    gt = sorted(
        [r for r in rows if r.origin == 'gt'],
        key=lambda r: r.last_accessed_at or datetime.min,
    )
    evicted = 0
    for entry in submissions + gt:
        if total_bytes <= budget_bytes:
            break
        path = cache_path_for(cache_root, entry.cache_key)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EBUSY):
                raise
            continue
        total_bytes -= (entry.size_bytes or 0)
        session.delete(entry)
        evicted += 1
    if evicted:
        session.commit()
    return evicted


def cache_clear(session, CacheEntry, *, cache_root: str) -> None:
    """Wipe every cached file + DB row. Used by the wipe script and
    for explicit per-test isolation. Does NOT delete the cache_root
    directory itself or the lock files (so concurrent calls don't
    crash mid-flight)."""
    rows = session.query(CacheEntry).all()
    for r in rows:
        try:
            os.remove(cache_path_for(cache_root, r.cache_key))
        except FileNotFoundError:
            pass
        session.delete(r)
    session.commit()


def cache_stats(session, CacheEntry) -> dict:
    """Cheap introspection. Used by the admin page once we wire it
    up; harmless to call from anywhere."""
    rows = session.query(CacheEntry).all()
    by_origin = {}
    total = 0
    for r in rows:
        by_origin.setdefault(r.origin, [0, 0])
        by_origin[r.origin][0] += 1
        by_origin[r.origin][1] += (r.size_bytes or 0)
        total += (r.size_bytes or 0)
    return {
        'total_entries': len(rows),
        'total_bytes': total,
        'by_origin': {
            o: {'count': c, 'bytes': b} for o, (c, b) in by_origin.items()
        },
    }
