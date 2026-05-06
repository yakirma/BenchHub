"""Route tests for submission lifecycle.

Upload via web form, recalculate (single + batch), update tags, batch action
(archive / unarchive / delete / add_tags / compare-redirect), download, and
the eager Celery dispatch for recalculation.
"""
import json
from unittest.mock import patch

import pytest

from app import (
    CustomField,
    Dataset,
    Leaderboard,
    Project,
    Sample,
    Submission,
    Tag,
    db,
)


@pytest.fixture
def project(db_session, client):
    p = Project(name="sub_proj")
    db.session.add(p)
    db.session.commit()
    client.set_cookie("active_project_id", str(p.id))
    return p


@pytest.fixture
def leaderboard_with_samples(db_session, project):
    ds = Dataset(name="sub_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add_all([Sample(dataset_id=ds.id, name=f"s{i}") for i in range(1, 4)])
    db.session.flush()

    lb = Leaderboard(name="sub_lb", project_id=project.id, summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.commit()
    return lb


@pytest.fixture
def submissions(db_session, leaderboard_with_samples):
    """Three submissions on the leaderboard for batch-action tests."""
    subs = []
    for n in ["alpha", "beta", "gamma"]:
        sub = Submission(name=n, leaderboard_id=leaderboard_with_samples.id, processing_status="Processed")
        db.session.add(sub)
        db.session.flush()
        subs.append(sub)
    db.session.commit()
    return subs


# ---------------------------------------------------------------------------
# Web upload (single submission ZIP)
# ---------------------------------------------------------------------------


def test_upload_single_submission_via_web_form(
    auth_client, project, leaderboard_with_samples, make_zip, logged_in_user
):
    proj_name, lb_id = project.name, leaderboard_with_samples.id
    layout = {"metric_acc/s1.txt": "0.95"}
    zip_path = make_zip("upload.zip", layout, root_folder="up")

    with open(zip_path, "rb") as f:
        resp = auth_client.post(
            f"/{proj_name}/leaderboard/{lb_id}/upload_submission",
            data={"submission_name": "via_form", "submission_zip": (f, "upload.zip")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 302

    sub = Submission.query.filter_by(leaderboard_id=lb_id).first()
    assert sub is not None
    # Inner-folder name overrides the form name (existing behavior).
    assert sub.name == "up"
    assert sub.owner_user_id == logged_in_user.id


def test_upload_with_no_files_redirects_back(
    auth_client, project, leaderboard_with_samples
):
    proj_name, lb_id = project.name, leaderboard_with_samples.id
    resp = auth_client.post(
        f"/{proj_name}/leaderboard/{lb_id}/upload_submission",
        data={"submission_name": "x"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    assert Submission.query.count() == 0


# ---------------------------------------------------------------------------
# Single recalculate
# ---------------------------------------------------------------------------


def test_recalculate_single_submission_dispatches_task(
    auth_client, project, leaderboard_with_samples, submissions
):
    proj_name, sub_id = project.name, submissions[0].id

    with patch("tasks.process_submission.delay") as task_mock:
        resp = auth_client.post(f"/{proj_name}/submission/{sub_id}/recalculate")

    assert resp.status_code == 302
    task_mock.assert_called_once()
    assert task_mock.call_args.args[0] == sub_id


def test_recalculate_single_marks_pending_immediately(
    auth_client, project, submissions
):
    proj_name, sub_id = project.name, submissions[0].id

    with patch("tasks.process_submission.delay"):
        auth_client.post(f"/{proj_name}/submission/{sub_id}/recalculate")

    db.session.expire_all()
    assert Submission.query.get(sub_id).processing_status == "Pending"


def test_recalculate_single_404_for_unknown(auth_client, project):
    resp = auth_client.post(f"/{project.name}/submission/9999/recalculate")
    assert resp.status_code == 404


def test_recalculate_forwards_sample_filters(auth_client, project, submissions):
    proj_name, sub_id = project.name, submissions[0].id

    with patch("tasks.process_submission.delay") as task_mock:
        auth_client.post(
            f"/{proj_name}/submission/{sub_id}/recalculate",
            data={
                "enable_sample_include": "true",
                "sample_include_tags": "easy,fast",
            },
        )

    filters = task_mock.call_args.kwargs["sample_filters"]
    assert filters["include"]["enabled"] is True
    assert sorted(filters["include"]["tags"]) == ["easy", "fast"]


# ---------------------------------------------------------------------------
# batch_action (archive / unarchive / delete / add_tags / recalculate / compare)
# ---------------------------------------------------------------------------


def test_batch_action_archive(client, project, leaderboard_with_samples, submissions):
    proj_name = project.name
    sub_ids = [str(s.id) for s in submissions]

    resp = client.post(
        f"/{proj_name}/submissions/batch_action",
        data={
            "action": "archive",
            "submission_ids": sub_ids,
            "leaderboard_id": str(leaderboard_with_samples.id),
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    statuses = {s.id: s.is_archived for s in Submission.query.all()}
    assert all(statuses.values())


def test_batch_action_unarchive(client, project, leaderboard_with_samples, submissions):
    for s in submissions:
        s.is_archived = True
    db.session.commit()

    resp = client.post(
        f"/{project.name}/submissions/batch_action",
        data={
            "action": "unarchive",
            "submission_ids": [str(s.id) for s in submissions],
            "leaderboard_id": str(leaderboard_with_samples.id),
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    assert all(not s.is_archived for s in Submission.query.all())


def test_batch_action_delete(client, project, leaderboard_with_samples, submissions):
    proj_name = project.name
    sub_ids = [str(s.id) for s in submissions[:2]]  # delete first two

    resp = client.post(
        f"/{proj_name}/submissions/batch_action",
        data={
            "action": "delete",
            "submission_ids": sub_ids,
            "leaderboard_id": str(leaderboard_with_samples.id),
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    remaining = {s.name for s in Submission.query.all()}
    assert remaining == {"gamma"}


def test_batch_action_add_tags_creates_and_links(
    client, project, leaderboard_with_samples, submissions
):
    proj_name = project.name
    sub_ids = [str(s.id) for s in submissions]

    resp = client.post(
        f"/{proj_name}/submissions/batch_action",
        data={
            "action": "add_tags",
            "submission_ids": sub_ids,
            "tags": "experiment, run_42",
            "leaderboard_id": str(leaderboard_with_samples.id),
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    for s in Submission.query.all():
        names = {t.name for t in s.tags}
        assert names == {"experiment", "run_42"}


def test_batch_action_recalculate_dispatches_sequential_task(
    client, project, leaderboard_with_samples, submissions
):
    proj_name = project.name
    sub_ids = [str(s.id) for s in submissions]

    with patch("tasks.process_submissions_batch_sequential.delay") as task_mock:
        resp = client.post(
            f"/{proj_name}/submissions/batch_action",
            data={
                "action": "recalculate",
                "submission_ids": sub_ids,
                "leaderboard_id": str(leaderboard_with_samples.id),
            },
        )

    assert resp.status_code == 302
    task_mock.assert_called_once()
    args, kwargs = task_mock.call_args.args, task_mock.call_args.kwargs
    assert sorted(args[0]) == sorted([s.id for s in submissions])
    assert "sample_filters" in kwargs

    # All targeted submissions are flipped back to Pending.
    db.session.expire_all()
    statuses = {s.processing_status for s in Submission.query.all()}
    assert statuses == {"Pending"}


def test_batch_action_compare_redirects_to_comparison_view(
    client, project, leaderboard_with_samples, submissions
):
    proj_name = project.name
    sub_ids = [str(s.id) for s in submissions[:2]]

    resp = client.post(
        f"/{proj_name}/submissions/batch_action",
        data={
            "action": "compare",
            "submission_ids": sub_ids,
            "leaderboard_id": str(leaderboard_with_samples.id),
        },
    )
    assert resp.status_code == 302
    location = resp.headers["Location"]
    assert f"/{proj_name}/comparison/{leaderboard_with_samples.id}" in location
    assert "compare_ids=" in location


def test_batch_action_no_submissions_redirects_to_leaderboard(
    client, project, leaderboard_with_samples
):
    resp = client.post(
        f"/{project.name}/submissions/batch_action",
        data={"action": "archive", "leaderboard_id": str(leaderboard_with_samples.id)},
    )
    assert resp.status_code == 302


# ---------------------------------------------------------------------------
# update_tags (single submission)
# ---------------------------------------------------------------------------


def test_update_submission_tags_replaces_existing(
    auth_client, project, submissions
):
    sub = submissions[0]
    existing_tag = Tag(name="old_tag")
    sub.tags.append(existing_tag)
    db.session.commit()

    resp = auth_client.post(
        f"/{project.name}/submission/{sub.id}/update_tags",
        data={"tags": "new_a,new_b"},
    )
    assert resp.status_code == 302

    db.session.expire_all()
    refreshed = Submission.query.get(sub.id)
    assert {t.name for t in refreshed.tags} == {"new_a", "new_b"}


# ---------------------------------------------------------------------------
# delete_submission
# ---------------------------------------------------------------------------


def test_delete_submission_removes_row(auth_client, project, submissions):
    sub_id = submissions[0].id

    resp = auth_client.post(f"/{project.name}/delete_submission/{sub_id}")
    assert resp.status_code == 302

    db.session.expire_all()
    assert Submission.query.get(sub_id) is None
