"""Phase 1 Slice 2 — multi-tenancy ownership tests.

Confirms:
- Anonymous users can't reach @login_required creation routes.
- @owner_required gates edit/delete to the owning user.
- @owner_required treats NULL owner_user_id as "legacy data" and permits.
- Created rows get owner_user_id stamped from the session.
- Visibility column defaults to 'public' for legacy ALTER-added rows.
"""
import pytest

from app import (
    Dataset,
    Leaderboard,
    Project,
    Sample,
    Submission,
    User,
    db,
)


@pytest.fixture
def other_user(db_session):
    """A second user, used to test the "non-owner" branch of @owner_required."""
    u = User(
        email="other@example.com",
        display_name="Other User",
        oauth_provider="github",
        oauth_sub="other-sub",
    )
    db_session.add(u)
    db_session.commit()
    return u


# ---------------------------------------------------------------------------
# Anonymous → /login redirect
# ---------------------------------------------------------------------------


def test_anon_create_project_redirects_to_login(client):
    resp = client.post("/projects/create", data={"name": "x"}, follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_anon_upload_dataset_redirects_to_login(client, db_session):
    p = Project(name="anon_proj")
    db_session.add(p)
    db_session.commit()
    client.set_cookie("active_project_id", str(p.id))

    resp = client.post(
        f"/{p.name}/upload_dataset",
        data={"dataset_name": "x"},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_anon_upload_submission_redirects_to_login(client, db_session):
    p = Project(name="anon_proj")
    db_session.add(p)
    db_session.flush()
    ds = Dataset(name="anon_ds")
    db_session.add(ds)
    db_session.flush()
    lb = Leaderboard(name="anon_lb", project_id=p.id, summary_metrics="")
    lb.datasets.append(ds)
    db_session.add(lb)
    db_session.commit()
    client.set_cookie("active_project_id", str(p.id))

    resp = client.post(
        f"/{p.name}/leaderboard/{lb.id}/upload_submission",
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# Stamping owner_user_id on creation
# ---------------------------------------------------------------------------


def test_project_create_stamps_owner(auth_client, logged_in_user):
    auth_client.post("/projects/create", data={"name": "owned_proj"})

    p = Project.query.filter_by(name="owned_proj").first()
    assert p is not None
    assert p.owner_user_id == logged_in_user.id


def test_leaderboard_create_stamps_owner(auth_client, logged_in_user, db_session):
    p = Project(name="lb_owner_proj", owner_user_id=logged_in_user.id)
    db_session.add(p)
    db_session.flush()
    ds = Dataset(name="lb_owner_ds")
    db_session.add(ds)
    db_session.commit()

    auth_client.post(
        f"/{p.name}/create_leaderboard",
        data={"leaderboard_name": "owned_lb", "dataset_ids": [str(ds.id)]},
    )

    lb = Leaderboard.query.filter_by(name="owned_lb").first()
    assert lb.owner_user_id == logged_in_user.id


# ---------------------------------------------------------------------------
# @owner_required: non-owner gets 403
# ---------------------------------------------------------------------------


def test_non_owner_cannot_delete_project(auth_client, logged_in_user, other_user, db_session):
    """logged_in_user is the session user. The project belongs to other_user.
    Delete must 403."""
    p = Project(name="not_yours", owner_user_id=other_user.id)
    db_session.add(p)
    db_session.commit()

    resp = auth_client.post(f"/projects/{p.id}/delete")
    assert resp.status_code == 403

    db_session.expire_all()
    assert Project.query.get(p.id) is not None  # not deleted


def test_non_owner_cannot_rename_project(auth_client, logged_in_user, other_user, db_session):
    p = Project(name="not_yours_either", owner_user_id=other_user.id)
    db_session.add(p)
    db_session.commit()

    resp = auth_client.post(f"/projects/{p.id}/rename", data={"name": "renamed"})
    assert resp.status_code == 403

    db_session.expire_all()
    assert Project.query.get(p.id).name == "not_yours_either"


def test_non_owner_cannot_delete_leaderboard(
    auth_client, logged_in_user, other_user, db_session
):
    p = Project(name="lb_proj", owner_user_id=other_user.id)
    db_session.add(p)
    db_session.flush()
    ds = Dataset(name="lb_ds")
    db_session.add(ds)
    db_session.flush()
    lb = Leaderboard(
        name="not_yours_lb", project_id=p.id, summary_metrics="",
        owner_user_id=other_user.id,
    )
    lb.datasets.append(ds)
    db_session.add(lb)
    db_session.commit()

    resp = auth_client.post(f"/{p.name}/delete_leaderboard/{lb.id}")
    assert resp.status_code == 403


def test_non_owner_cannot_delete_submission(
    auth_client, logged_in_user, other_user, db_session
):
    p = Project(name="sp", owner_user_id=other_user.id)
    db_session.add(p)
    db_session.flush()
    ds = Dataset(name="sd")
    db_session.add(ds)
    db_session.flush()
    lb = Leaderboard(name="sl", project_id=p.id, summary_metrics="",
                     owner_user_id=other_user.id)
    lb.datasets.append(ds)
    db_session.add(lb)
    db_session.flush()
    sub = Submission(name="other_sub", leaderboard_id=lb.id, owner_user_id=other_user.id)
    db_session.add(sub)
    db_session.commit()

    resp = auth_client.post(f"/{p.name}/delete_submission/{sub.id}")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Owner CAN edit/delete
# ---------------------------------------------------------------------------


def test_owner_can_delete_own_project(auth_client, logged_in_user, db_session):
    p = Project(name="mine", owner_user_id=logged_in_user.id)
    db_session.add(p)
    db_session.commit()

    resp = auth_client.post(f"/projects/{p.id}/delete")
    assert resp.status_code == 302

    db_session.expire_all()
    assert Project.query.get(p.id) is None


# ---------------------------------------------------------------------------
# Legacy NULL owner — anyone signed in can act on it (current policy)
# ---------------------------------------------------------------------------


def test_legacy_null_owner_project_can_be_renamed_by_anyone_logged_in(
    auth_client, db_session
):
    """Pre-Phase-1 rows have owner_user_id IS NULL. Until a backfill assigns
    them an owner, @owner_required allows any authenticated user to act —
    matching the pre-multi-tenant behavior of the existing app."""
    p = Project(name="legacy", owner_user_id=None)
    db_session.add(p)
    db_session.commit()

    resp = auth_client.post(f"/projects/{p.id}/rename", data={"name": "renamed_legacy"})
    assert resp.status_code == 302

    db_session.expire_all()
    assert Project.query.get(p.id).name == "renamed_legacy"


# ---------------------------------------------------------------------------
# visibility column default
# ---------------------------------------------------------------------------


def test_new_project_has_visibility_public_by_default(db_session):
    p = Project(name="vis_default")
    db_session.add(p)
    db_session.commit()
    assert p.visibility == "public"


def test_new_dataset_has_visibility_public_by_default(db_session):
    ds = Dataset(name="vis_ds")
    db_session.add(ds)
    db_session.commit()
    assert ds.visibility == "public"


def test_new_leaderboard_has_visibility_public_by_default(db_session):
    p = Project(name="vp")
    db_session.add(p)
    db_session.flush()
    ds = Dataset(name="vds")
    db_session.add(ds)
    db_session.flush()
    lb = Leaderboard(name="vlb", project_id=p.id, summary_metrics="")
    lb.datasets.append(ds)
    db_session.add(lb)
    db_session.commit()
    assert lb.visibility == "public"
