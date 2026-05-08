"""Disk-savings closeout: remote submissions stop keeping
`uploads/submissions/<id>/` around after eval. The cached ZIP
under bench_cache is canonical; recalcs re-extract on demand
via `_with_extracted_submission`.
"""
import io
import os
import shutil
import sys
import zipfile
from unittest.mock import patch

import pytest

from app import (
    Dataset, Leaderboard, Submission, db,
    _evict_extracted_submission_folder,
    _with_extracted_submission,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('metric_dummy/s00000.txt', '0.5')
        zf.writestr('README.md', 'sub-bytes')
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def remote_sub_with_folder(db_session, logged_in_user, tmp_path, monkeypatch):
    """A remote Submission with `uploads/submissions/<id>/` already
    populated (initial-eval state — extraction happened, CustomFields
    persisted, folder still on disk awaiting eviction)."""
    monkeypatch.setitem(
        __import__('app').app.config, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'),
    )
    ds = Dataset(name='evict_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='evict_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(
        name='evict_sub', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        storage_mode='remote',
        remote_url='https://example.test/sub.zip',
    )
    db.session.add(sub); db.session.commit()

    folder = os.path.join(tmp_path / 'uploads', 'submissions', str(sub.id))
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, 'metric_dummy_s00000.txt'), 'w').write('0.5')
    return sub, folder


@pytest.fixture
def local_sub_with_folder(db_session, logged_in_user, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__('app').app.config, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'),
    )
    ds = Dataset(name='local_keep_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='local_keep_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(
        name='local_keep', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id, storage_mode='local',
    )
    db.session.add(sub); db.session.commit()

    folder = os.path.join(tmp_path / 'uploads', 'submissions', str(sub.id))
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, 'placeholder.txt'), 'w').write('keep me')
    return sub, folder


# ---------------------------------------------------------------------------
# _evict_extracted_submission_folder — local stays, remote evicts.
# ---------------------------------------------------------------------------


def test_evict_no_op_for_local_submission(local_sub_with_folder):
    sub, folder = local_sub_with_folder
    _evict_extracted_submission_folder(sub)
    assert os.path.isdir(folder)
    assert os.path.exists(os.path.join(folder, 'placeholder.txt'))


def test_evict_removes_folder_for_remote_submission(remote_sub_with_folder):
    sub, folder = remote_sub_with_folder
    assert os.path.isdir(folder)
    _evict_extracted_submission_folder(sub)
    assert not os.path.exists(folder)


def test_evict_silent_when_folder_already_gone(remote_sub_with_folder):
    sub, folder = remote_sub_with_folder
    shutil.rmtree(folder)
    # No raise.
    _evict_extracted_submission_folder(sub)


# ---------------------------------------------------------------------------
# _with_extracted_submission — yields a valid path in all cases.
# ---------------------------------------------------------------------------


def test_with_extracted_yields_local_folder_unchanged(local_sub_with_folder):
    sub, folder = local_sub_with_folder
    with _with_extracted_submission(sub) as path:
        assert path == folder
    # Local folder NEVER cleaned up.
    assert os.path.isdir(folder)


def test_with_extracted_yields_existing_remote_folder_when_present(
    remote_sub_with_folder,
):
    sub, folder = remote_sub_with_folder
    with _with_extracted_submission(sub) as path:
        assert path == folder
    # Initial-eval path: folder NOT cleaned up by the context manager
    # (the post-eval evictor is what tears it down — separate concern).
    assert os.path.isdir(folder)


def test_with_extracted_re_extracts_when_folder_missing(
    db_session, logged_in_user, tmp_path, monkeypatch,
):
    """The interesting case — recalc on a remote sub whose extracted
    folder was already evicted. The context manager fetches the
    cached ZIP, extracts to a tempdir, yields it, cleans up on exit."""
    monkeypatch.setitem(
        __import__('app').app.config, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'),
    )
    monkeypatch.setitem(
        __import__('app').app.config, 'CACHE_FOLDER', str(tmp_path / 'cache'),
    )
    os.makedirs(tmp_path / 'uploads' / 'submissions', exist_ok=True)
    ds = Dataset(name='reextract_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='reextract_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(
        name='reextract', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        storage_mode='remote',
        remote_url='https://example.test/sub.zip',
    )
    db.session.add(sub); db.session.commit()

    payload = _build_zip_bytes()

    class _Resp:
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_content(self, chunk_size=None):
            yield payload

    with patch('requests.get', return_value=_Resp()):
        with _with_extracted_submission(sub) as path:
            assert os.path.isdir(path)
            # The README written by _build_zip_bytes lands inside the
            # extracted form regardless of the ZIP's nested-folder
            # structure (the helper unwraps single-top-level layouts).
            assert os.path.exists(os.path.join(path, 'README.md'))
        # On exit, the transient extraction is gone.
        assert not os.path.exists(path)


def test_with_extracted_remote_without_url_raises(db_session, logged_in_user, tmp_path, monkeypatch):
    """storage_mode='remote' with no URL is corrupt state — the
    context manager surfaces it loudly rather than silently skipping."""
    monkeypatch.setitem(
        __import__('app').app.config, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'),
    )
    os.makedirs(tmp_path / 'uploads' / 'submissions', exist_ok=True)
    ds = Dataset(name='broken_remote_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='broken_remote_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(
        name='broken', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        storage_mode='remote',
        remote_url=None,
    )
    db.session.add(sub); db.session.commit()
    with pytest.raises(RuntimeError, match='no remote_url'):
        with _with_extracted_submission(sub) as _:
            pass
