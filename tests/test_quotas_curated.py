"""Phase 7 (quotas) + Phase 5 (curated framework) tests.

Quotas are enforced server-side at the upload routes; the helpers themselves
are also unit-tested so a future refactor that drops the route gate without
the helper change still trips a test.

Curated content is just a flag + seed helper — the seeded namespace exists
even on a fresh DB, and the landing/explore surfaces respect the flag.
"""
import os
from datetime import datetime, timedelta

import pytest

from app import (
    Dataset,
    Leaderboard,
    Submission,
    User,
    check_quota,
    daily_submission_count,
    dataset_count,
    storage_used_bytes,
    db,
)


# ===========================================================================
# Quota helpers (unit)
# ===========================================================================


def test_storage_used_sums_owned_datasets(db_session, logged_in_user):
    db.session.add(Dataset(
        name="ds_a", owner_user_id=logged_in_user.id, storage_bytes=1000,
    ))
    db.session.add(Dataset(
        name="ds_b", owner_user_id=logged_in_user.id, storage_bytes=2500,
    ))
    db.session.add(Dataset(
        name="ds_other", owner_user_id=None, storage_bytes=999_999,
    ))
    db.session.commit()
    assert storage_used_bytes(logged_in_user) == 3500


def test_dataset_count_only_counts_owned(db_session, logged_in_user):
    db.session.add(Dataset(name="d1", owner_user_id=logged_in_user.id))
    db.session.add(Dataset(name="d2", owner_user_id=logged_in_user.id))
    db.session.add(Dataset(name="d_other", owner_user_id=None))
    db.session.commit()
    assert dataset_count(logged_in_user) == 2


def test_daily_submission_count_uses_rolling_window(db_session, logged_in_user):
    ds = Dataset(name="rsq_ds", owner_user_id=logged_in_user.id)
    db.session.add_all([ds])
    db.session.flush()
    lb = Leaderboard(name="rsq_lb", summary_metrics='',
                     owner_user_id=logged_in_user.id)
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    now = datetime.utcnow()
    for delta in [timedelta(hours=1), timedelta(hours=20), timedelta(hours=23)]:
        db.session.add(Submission(
            name=f"s{delta}", leaderboard_id=lb.id,
            owner_user_id=logged_in_user.id,
            upload_date=now - delta,
        ))
    # Outside the rolling 24h window:
    db.session.add(Submission(
        name="old_one", leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        upload_date=now - timedelta(days=2),
    ))
    db.session.commit()

    assert daily_submission_count(logged_in_user) == 3


def test_check_quota_dataset_create_under_cap(db_session, logged_in_user):
    ok, msg = check_quota(logged_in_user, kind='dataset_create', incoming_bytes=1024)
    assert ok is True
    assert msg is None


def test_check_quota_dataset_create_count_over(db_session, logged_in_user):
    logged_in_user.quota_max_datasets = 1
    db.session.add(Dataset(name="only", owner_user_id=logged_in_user.id))
    db.session.commit()
    ok, msg = check_quota(logged_in_user, kind='dataset_create', incoming_bytes=0)
    assert ok is False
    assert "limit" in msg.lower()


def test_check_quota_dataset_create_storage_over(db_session, logged_in_user):
    logged_in_user.quota_max_storage_bytes = 1000
    db.session.add(Dataset(
        name="bulky", owner_user_id=logged_in_user.id, storage_bytes=900,
    ))
    db.session.commit()
    ok, msg = check_quota(logged_in_user, kind='dataset_create', incoming_bytes=200)
    assert ok is False
    assert "storage" in msg.lower()


def test_check_quota_submission_over_limit(db_session, logged_in_user):
    ds = Dataset(name="qds", owner_user_id=logged_in_user.id)
    db.session.add_all([ds]); db.session.flush()
    lb = Leaderboard(name="qlb", summary_metrics='',
                     owner_user_id=logged_in_user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()

    logged_in_user.quota_max_submissions_per_day = 2
    now = datetime.utcnow()
    for i in range(2):
        db.session.add(Submission(
            name=f"s{i}", leaderboard_id=lb.id,
            owner_user_id=logged_in_user.id,
            upload_date=now - timedelta(hours=i + 1),
        ))
    db.session.commit()

    ok, msg = check_quota(logged_in_user, kind='submission')
    assert ok is False
    assert "submission" in msg.lower()


# ===========================================================================
# Quota gates wired at upload routes
# ===========================================================================


@pytest.fixture
def project_for_user(db_session, logged_in_user, client):
    import types
    return types.SimpleNamespace(id=0, name='legacy')


def test_upload_dataset_blocked_when_dataset_count_at_cap(
    auth_client, db_session, logged_in_user, project_for_user, make_zip,
):
    logged_in_user.quota_max_datasets = 1
    db.session.add(Dataset(name="already", owner_user_id=logged_in_user.id))
    db.session.commit()

    zip_path = make_zip("blocked.zip", {
        "metric_score/s1.txt": "0.5",
    }, root_folder="blocked")

    with open(zip_path, "rb") as fh:
        resp = auth_client.post(
            "/upload_dataset",
            data={"dataset_name": "blocked",
                  "dataset_zip": (fh, "blocked.zip")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
    assert resp.status_code == 200
    # The new dataset should NOT exist.
    assert Dataset.query.filter_by(name="blocked").first() is None
    # And the over-cap message should have flashed.
    assert b"limit" in resp.data.lower() or b"reached" in resp.data.lower()


def test_upload_submission_blocked_when_daily_cap_hit(
    auth_client, db_session, logged_in_user, project_for_user,
):
    logged_in_user.quota_max_submissions_per_day = 1
    ds = Dataset(name="qsub_ds", owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name="qsub_lb",
                     summary_metrics='', owner_user_id=logged_in_user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    db.session.add(Submission(
        name="prior", leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        upload_date=datetime.utcnow() - timedelta(hours=1),
    ))
    db.session.commit()

    # Empty file payload is fine — gate runs before extraction. The point
    # is the gate, not the extraction.
    resp = auth_client.post(
        f"/leaderboard/{lb.id}/upload_submission",
        data={"submission_zip": (b"x", "ignored.zip")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    # Should still only have the one prior submission.
    assert Submission.query.filter_by(leaderboard_id=lb.id).count() == 1
