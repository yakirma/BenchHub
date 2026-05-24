"""Admins (BENCHHUB_ADMIN_EMAILS / is_admin=True) bypass the
50 MB storage cap and the per-user dataset count limit. Regular
users still get the standard checks.
"""
from __future__ import annotations

from app import (
    User,
    app as flask_app,
    check_quota,
    db,
)


def test_admin_bypasses_storage_cap(db_session):
    admin = User(email='a@bench.local', display_name='a',
                 oauth_provider='github', oauth_sub='a-1',
                 is_admin=True,
                 quota_max_storage_bytes=50 * 1024 * 1024,
                 quota_max_datasets=5)
    db.session.add(admin); db.session.commit()
    # 1 TB import — would fail for any non-admin under the 50 MB cap.
    ok, msg = check_quota(admin, kind='dataset_create',
                          incoming_bytes=10**12)
    assert ok is True
    assert msg is None


def test_admin_bypasses_dataset_count_cap(db_session):
    admin = User(email='a2@bench.local', display_name='a2',
                 oauth_provider='github', oauth_sub='a2-1',
                 is_admin=True,
                 quota_max_storage_bytes=50 * 1024 * 1024,
                 quota_max_datasets=0)
    db.session.add(admin); db.session.commit()
    # quota_max_datasets=0 would otherwise reject the very first
    # dataset; admin bypass lets it through.
    ok, _ = check_quota(admin, kind='dataset_create', incoming_bytes=0)
    assert ok is True


def test_non_admin_still_constrained(db_session):
    user = User(email='u@bench.local', display_name='u',
                oauth_provider='github', oauth_sub='u-1',
                is_admin=False,
                quota_max_storage_bytes=50 * 1024 * 1024,
                quota_max_datasets=5)
    db.session.add(user); db.session.commit()
    ok, msg = check_quota(user, kind='dataset_create',
                          incoming_bytes=10**12)
    assert ok is False
    assert 'Storage limit would be exceeded' in msg


def test_admin_bypasses_daily_submission_cap(db_session):
    admin = User(email='a3@bench.local', display_name='a3',
                 oauth_provider='github', oauth_sub='a3-1',
                 is_admin=True,
                 quota_max_submissions_per_day=0)
    db.session.add(admin); db.session.commit()
    ok, _ = check_quota(admin, kind='submission')
    assert ok is True
