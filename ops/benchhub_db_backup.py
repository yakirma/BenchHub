#!/usr/bin/env python3
"""Daily snapshot of the BenchHub SQLite database.

Uses sqlite3's online-backup API (`Connection.backup`) so the live
gunicorn + celery processes don't have to be stopped — the snapshot
is taken from a read transaction held during the copy, with no
locking impact on writers beyond the duration of the copy.

Output: `<DB_BACKUP_DIR>/database-YYYYMMDD-HHMMSS.db.gz`
Retention: keep the newest `KEEP_LAST` snapshots; older ones are
deleted in the same run.

Run via systemd timer (`benchhub-db-backup.timer`); also safe to
invoke by hand for a one-off snapshot.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

DB_PATH = Path('/home/ymatri/.dtofbenchmarking/database.db')
BACKUP_DIR = Path('/home/ymatri/.dtofbenchmarking/db_backups')
KEEP_LAST = 14  # ~two weeks of daily snapshots
PREFIX = 'database-'
SUFFIX = '.db.gz'


def take_snapshot(src: Path, dest_gz: Path) -> int:
    """Snapshot `src` into a gzipped file at `dest_gz`. Returns the
    snapshot's on-disk byte size."""
    # Stream into a temp .db file first so we can gzip it as a second
    # step. The online-backup API doesn't accept a stream destination.
    with tempfile.NamedTemporaryFile(
        suffix='.db',
        dir=dest_gz.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with sqlite3.connect(str(src)) as live, \
             sqlite3.connect(str(tmp_path)) as snap:
            # pages=-1 → copy in one shot, which is fine for our DB
            # size (single-digit MB). For multi-GB DBs you'd want to
            # chunk + sleep to avoid holding the writer lock too long.
            live.backup(snap, pages=-1)
            snap.commit()
        # gzip into final location, then drop the uncompressed temp.
        with open(tmp_path, 'rb') as fin, gzip.open(dest_gz, 'wb', compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    return dest_gz.stat().st_size


def prune_old(backup_dir: Path, keep_last: int) -> int:
    """Delete all but the newest `keep_last` backups. Returns the
    number removed."""
    snapshots = sorted(
        (p for p in backup_dir.iterdir()
         if p.is_file() and p.name.startswith(PREFIX) and p.name.endswith(SUFFIX)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for stale in snapshots[keep_last:]:
        try:
            stale.unlink()
            removed += 1
        except OSError as e:
            print(f'warn: could not delete {stale}: {e}', file=sys.stderr)
    return removed


def main() -> int:
    if not DB_PATH.exists():
        print(f'fatal: no DB at {DB_PATH}', file=sys.stderr)
        return 1
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    dest = BACKUP_DIR / f'{PREFIX}{ts}{SUFFIX}'
    t0 = time.monotonic()
    size = take_snapshot(DB_PATH, dest)
    elapsed = time.monotonic() - t0
    removed = prune_old(BACKUP_DIR, KEEP_LAST)
    print(
        f'backup ok: {dest.name} ({size:,} bytes, {elapsed:.2f}s); '
        f'pruned {removed} old snapshot(s); kept {KEEP_LAST} newest'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
