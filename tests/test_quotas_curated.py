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
    Project,
    Submission,
    User,
    check_quota,
    daily_submission_count,
    dataset_count,
    ensure_curated_seed,
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
    p = Project(name="rsq_p", owner_user_id=logged_in_user.id)
    ds = Dataset(name="rsq_ds", owner_user_id=logged_in_user.id)
    db.session.add_all([p, ds])
    db.session.flush()
    lb = Leaderboard(name="rsq_lb", project_id=p.id, summary_metrics='',
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
    p = Project(name="qp", owner_user_id=logged_in_user.id)
    ds = Dataset(name="qds", owner_user_id=logged_in_user.id)
    db.session.add_all([p, ds]); db.session.flush()
    lb = Leaderboard(name="qlb", project_id=p.id, summary_metrics='',
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


def test_system_user_bypasses_all_quotas(db_session):
    """The bookkeeping system user owns curated content. It must not be
    rate-limited by free-tier caps that would prevent seeding new
    benchmarks."""
    sys_user = User(
        email="sys@example.com",
        display_name="System",
        oauth_provider="system",
        oauth_sub="sys-1",
        is_system=True,
        quota_max_datasets=0,                    # zero
        quota_max_storage_bytes=0,               # zero
        quota_max_submissions_per_day=0,         # zero
    )
    db.session.add(sys_user); db.session.commit()
    ok_d, _ = check_quota(sys_user, kind='dataset_create', incoming_bytes=10**9)
    ok_s, _ = check_quota(sys_user, kind='submission')
    assert ok_d is True
    assert ok_s is True


# ===========================================================================
# Quota gates wired at upload routes
# ===========================================================================


@pytest.fixture
def project_for_user(db_session, logged_in_user, client):
    """A project owned by the logged-in user, pinned via cookie so the
    middleware resolves /<project_name>/... routes."""
    p = Project(name="quota_proj", owner_user_id=logged_in_user.id)
    db.session.add(p); db.session.commit()
    client.set_cookie("active_project_id", str(p.id))
    return p


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
            f"/{project_for_user.name}/upload_dataset",
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
    lb = Leaderboard(name="qsub_lb", project_id=project_for_user.id,
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
        f"/{project_for_user.name}/leaderboard/{lb.id}/upload_submission",
        data={"submission_zip": (b"x", "ignored.zip")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    # Should still only have the one prior submission.
    assert Submission.query.filter_by(leaderboard_id=lb.id).count() == 1


# ===========================================================================
# Curated framework
# ===========================================================================


def test_ensure_curated_seed_creates_user_and_project(db_session):
    ensure_curated_seed()
    sys_user = User.query.filter_by(email='curated@benchhub.local').first()
    assert sys_user is not None
    assert sys_user.is_system is True
    proj = Project.query.filter_by(name='benchhub-curated').first()
    assert proj is not None
    assert proj.is_curated is True
    assert proj.owner_user_id == sys_user.id


def test_ensure_curated_seed_is_idempotent(db_session):
    ensure_curated_seed()
    ensure_curated_seed()
    ensure_curated_seed()
    assert User.query.filter_by(email='curated@benchhub.local').count() == 1
    assert Project.query.filter_by(name='benchhub-curated').count() == 1


def test_explore_curated_filter_only_shows_curated(client, db_session):
    """?curated=1 narrows results to leaderboards with is_curated=True."""
    ensure_curated_seed()
    curated_proj = Project.query.filter_by(name='benchhub-curated').first()
    other_proj = Project(name='other_p', visibility='public')
    db.session.add(other_proj); db.session.flush()

    ds = Dataset(name='cur_ds')
    db.session.add(ds); db.session.flush()

    cur_lb = Leaderboard(name='curated_only_lb', project_id=curated_proj.id,
                         summary_metrics='', is_curated=True)
    cur_lb.datasets.append(ds)
    other_lb = Leaderboard(name='regular_lb', project_id=other_proj.id,
                           summary_metrics='')
    other_lb.datasets.append(ds)
    db.session.add_all([cur_lb, other_lb])
    db.session.commit()

    # Default explore: both visible.
    body_all = client.get('/explore').data
    assert b'curated_only_lb' in body_all
    assert b'regular_lb' in body_all

    # Curated-only: only the curated leaderboard remains.
    body_cur = client.get('/explore?curated=1').data
    assert b'curated_only_lb' in body_cur
    assert b'regular_lb' not in body_cur


def test_landing_renders_curated_section_when_present(client, db_session):
    ensure_curated_seed()
    curated_proj = Project.query.filter_by(name='benchhub-curated').first()
    ds = Dataset(name='cur_ds_landing')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='hero_curated_lb', project_id=curated_proj.id,
                     summary_metrics='', visibility='public', is_curated=True)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    body = client.get('/').data
    assert b'Curated benchmarks' in body
    assert b'hero_curated_lb' in body


def test_landing_hides_curated_section_when_empty(client, db_session):
    ensure_curated_seed()
    body = client.get('/').data
    # Section header is gated by the truthiness of the rows list.
    assert b'Curated benchmarks' not in body


def test_user_profile_404_for_system_user(client, db_session):
    ensure_curated_seed()
    sys_user = User.query.filter_by(email='curated@benchhub.local').first()
    resp = client.get(f'/u/{sys_user.id}')
    assert resp.status_code == 404
