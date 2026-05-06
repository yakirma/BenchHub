"""Tests for the assorted JSON APIs and the DB migration runner.

- /api/leaderboard/<id>/recalculate_async
- /api/leaderboard/<id>/metrics_status
- /api/user/merge, /api/user/unmerge
- check_and_migrate_db idempotency
"""
import json
from unittest.mock import patch

import pytest

from app import (
    AuthorProfile,
    CustomField,
    Dataset,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    MetricResult,
    Sample,
    Submission,
    check_and_migrate_db,
    db,
)


@pytest.fixture
def lb_with_subs(db_session, client):
    ds = Dataset(name="api_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name="s1"))
    db.session.flush()

    lb = Leaderboard(name="api_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    subs = [
        Submission(name=f"sub{i}", leaderboard_id=lb.id, processing_status="Processed")
        for i in range(2)
    ]
    db.session.add_all(subs)
    db.session.commit()

    return {"lb": lb, "subs": subs}


# ---------------------------------------------------------------------------
# /api/leaderboard/<id>/recalculate_async
# ---------------------------------------------------------------------------


def test_recalculate_async_dispatches_one_task_per_submission(client, lb_with_subs):
    lb_id = lb_with_subs["lb"].id
    sub_ids = [s.id for s in lb_with_subs["subs"]]

    with patch("tasks.process_submission.delay") as task_mock:
        resp = client.post(
            f"/api/leaderboard/{lb_id}/recalculate_async",
            data=json.dumps({"submission_ids": sub_ids}),
            content_type="application/json",
        )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["triggered_count"] == 2
    assert task_mock.call_count == 2


def test_recalculate_async_400_with_no_ids(client, lb_with_subs):
    lb_id = lb_with_subs["lb"].id

    resp = client.post(
        f"/api/leaderboard/{lb_id}/recalculate_async",
        data=json.dumps({"submission_ids": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_recalculate_async_only_dispatches_for_matching_leaderboard(
    client, lb_with_subs
):
    """Submissions on a different leaderboard must not be triggered even if
    their IDs are passed in the payload."""
    lb_id = lb_with_subs["lb"].id

    # Build a second leaderboard with one submission.
    other_ds = Dataset(name="other_api_ds")
    db.session.add(other_ds)
    db.session.flush()
    other_lb = Leaderboard(name="other_lb", summary_metrics="")
    other_lb.datasets.append(other_ds)
    db.session.add(other_lb)
    db.session.flush()
    other_sub = Submission(name="cross", leaderboard_id=other_lb.id, processing_status="Processed")
    db.session.add(other_sub)
    db.session.commit()

    payload_ids = [s.id for s in lb_with_subs["subs"]] + [other_sub.id]

    with patch("tasks.process_submission.delay") as task_mock:
        resp = client.post(
            f"/api/leaderboard/{lb_id}/recalculate_async",
            data=json.dumps({"submission_ids": payload_ids}),
            content_type="application/json",
        )

    body = resp.get_json()
    # Only the 2 submissions actually on lb_id were dispatched.
    assert body["triggered_count"] == 2
    assert task_mock.call_count == 2


# ---------------------------------------------------------------------------
# /api/leaderboard/<id>/metrics_status
# ---------------------------------------------------------------------------


def test_metrics_status_returns_status_and_results(client, lb_with_subs):
    lb_id = lb_with_subs["lb"].id
    sub_ids = [s.id for s in lb_with_subs["subs"]]

    # Add a metric and a result for sub0.
    gm = GlobalMetric(name="m", python_code="def m(): return 1", is_aggregated=False)
    db.session.add(gm)
    db.session.flush()
    lm = LeaderboardMetric(
        leaderboard_id=lb_id, global_metric_id=gm.id, arg_mappings="{}", target_name="M"
    )
    db.session.add(lm)
    db.session.flush()
    db.session.add(
        MetricResult(submission_id=sub_ids[0], leaderboard_metric_id=lm.id, value=0.42)
    )
    db.session.commit()

    resp = client.post(
        f"/api/leaderboard/{lb_id}/metrics_status",
        data=json.dumps({"submission_ids": sub_ids}),
        content_type="application/json",
    )
    assert resp.status_code == 200

    body = resp.get_json()
    # Response shape: {submissions: {<id>: {...}}, directions: {...}, ranges: {...}}
    assert str(sub_ids[0]) in body["submissions"]
    sub_entry = body["submissions"][str(sub_ids[0])]
    assert sub_entry["status"] == "Processed"
    assert sub_entry["metrics"][f"lm_{lm.id}"] == 0.42


def test_metrics_status_400_with_no_ids(client, lb_with_subs):
    lb_id = lb_with_subs["lb"].id

    resp = client.post(
        f"/api/leaderboard/{lb_id}/metrics_status",
        data=json.dumps({"submission_ids": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DB migrations
# ---------------------------------------------------------------------------


def test_check_and_migrate_db_is_idempotent(client, db_session):
    """Running the migration twice in a row must not raise."""
    # First run is implicit (db.create_all() in conftest covers everything).
    # Second run should be a no-op.
    check_and_migrate_db()
    check_and_migrate_db()


def test_check_and_migrate_db_adds_missing_column(app, tmp_path, monkeypatch):
    """Simulate an "old" DB by dropping a recently-added column, then run the
    migration and confirm the column is added back."""
    import sqlite3
    from app import db

    db_uri = app.config["SQLALCHEMY_DATABASE_URI"]
    db_path = db_uri.replace("sqlite:///", "")

    # Drop the connection pool so SQLite isn't holding the DB open.
    db.session.remove()
    db.engine.dispose()

    # Recreate `submission` table without the `git_author` column to simulate
    # an old install.
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS submission")
    cur.execute("""
        CREATE TABLE submission (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            leaderboard_id INTEGER,
            git_commit VARCHAR(100),
            git_branch VARCHAR(100),
            git_message VARCHAR(200),
            upload_date DATETIME,
            is_archived BOOLEAN,
            processing_status VARCHAR(50),
            last_sample_filter TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Run the migration.
    check_and_migrate_db()

    # Verify the column is now present.
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(submission)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()

    assert "git_author" in cols
