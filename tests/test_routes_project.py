"""Route tests for project lifecycle.

Project rows are namespaces. Most other URLs are prefixed with /<project_name>/.
The `load_project_context` middleware redirects to /projects when there's no
project name in the URL or `active_project_id` cookie.
"""
import pytest

from app import (
    CustomField,
    Dataset,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    Project,
    Sample,
    Submission,
    db,
)


# ---------------------------------------------------------------------------
# /projects (list)
# ---------------------------------------------------------------------------


def test_projects_index_renders_with_no_projects(client):
    resp = client.get("/projects")
    assert resp.status_code == 200
    assert b"projects" in resp.data.lower()


def test_projects_index_lists_existing_projects(client):
    db.session.add_all([Project(name="alpha"), Project(name="beta")])
    db.session.commit()

    resp = client.get("/projects")
    assert resp.status_code == 200
    assert b"alpha" in resp.data
    assert b"beta" in resp.data


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_project_inserts_row_and_redirects(auth_client, logged_in_user):
    resp = auth_client.post("/projects/create", data={"name": "shiny", "description": "desc"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/projects")

    p = Project.query.filter_by(name="shiny").first()
    assert p is not None
    assert p.description == "desc"
    # Phase 1 multi-tenancy: owner stamped from session.
    assert p.owner_user_id == logged_in_user.id


def test_create_project_anonymous_redirects_to_login(client):
    resp = client.post("/projects/create", data={"name": "shiny"}, follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    assert Project.query.count() == 0


def test_create_project_rejects_blank_name(auth_client):
    resp = auth_client.post("/projects/create", data={"name": ""})
    # Redirects with flash; no DB row created.
    assert resp.status_code == 302
    assert Project.query.count() == 0


def test_create_project_rejects_duplicate_name(auth_client):
    db.session.add(Project(name="dup"))
    db.session.commit()

    resp = auth_client.post("/projects/create", data={"name": "dup"})
    assert resp.status_code == 302
    # No second row.
    assert Project.query.filter_by(name="dup").count() == 1


# ---------------------------------------------------------------------------
# Select (cookie)
# ---------------------------------------------------------------------------


def test_select_project_sets_cookie(client):
    p = Project(name="picked")
    db.session.add(p)
    db.session.commit()

    resp = client.get(f"/projects/select/{p.id}")
    assert resp.status_code == 302
    cookies = resp.headers.getlist("Set-Cookie")
    assert any(f"active_project_id={p.id}" in c for c in cookies)


def test_select_project_with_dashboard_next_redirects_to_index(client):
    p = Project(name="dash")
    db.session.add(p)
    db.session.commit()

    resp = client.get(f"/projects/select/{p.id}?next=dashboard")
    assert resp.status_code == 302
    assert "/dash/" in resp.headers["Location"]


def test_select_project_404_for_unknown(client):
    resp = client.get("/projects/select/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


def test_rename_project_updates_name(auth_client, project_ctx):
    p = Project(name="old")
    db.session.add(p)
    db.session.commit()

    resp = auth_client.post(f"/projects/{p.id}/rename", data={"name": "new"})
    assert resp.status_code == 302

    db.session.expire_all()
    refreshed = Project.query.get(p.id)
    assert refreshed.name == "new"


def test_rename_project_rejects_blank_name(auth_client, project_ctx):
    p = Project(name="keep")
    db.session.add(p)
    db.session.commit()

    auth_client.post(f"/projects/{p.id}/rename", data={"name": ""})

    db.session.expire_all()
    assert Project.query.get(p.id).name == "keep"


def test_rename_project_rejects_collision(auth_client, project_ctx):
    p1 = Project(name="one")
    p2 = Project(name="two")
    db.session.add_all([p1, p2])
    db.session.commit()

    auth_client.post(f"/projects/{p1.id}/rename", data={"name": "two"})

    db.session.expire_all()
    assert Project.query.get(p1.id).name == "one"  # unchanged


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------


def test_clone_project_duplicates_leaderboards_and_metrics(auth_client, project_ctx):
    src = Project(name="source")
    db.session.add(src)
    db.session.flush()

    ds = Dataset(name="shared_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name="s1"))
    db.session.flush()

    lb = Leaderboard(name="lb1", project_id=src.id, summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    gm = GlobalMetric(name="gm1", python_code="def m(): return 1")
    db.session.add(gm)
    db.session.flush()

    db.session.add(LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        arg_mappings="{}",
        target_name="renamed",
    ))
    db.session.commit()

    resp = auth_client.post(f"/projects/{src.id}/clone", data={"name": "cloned"})
    assert resp.status_code == 302

    cloned = Project.query.filter_by(name="cloned").first()
    assert cloned is not None

    cloned_lbs = Leaderboard.query.filter_by(project_id=cloned.id).all()
    assert len(cloned_lbs) == 1
    assert cloned_lbs[0].name == "lb1"

    cloned_lms = LeaderboardMetric.query.filter_by(leaderboard_id=cloned_lbs[0].id).all()
    assert len(cloned_lms) == 1
    assert cloned_lms[0].target_name == "renamed"


def test_clone_project_rejects_existing_target_name(auth_client, project_ctx):
    src = Project(name="src")
    dst = Project(name="taken")
    db.session.add_all([src, dst])
    db.session.commit()
    before = Project.query.count()

    resp = auth_client.post(f"/projects/{src.id}/clone", data={"name": "taken"})
    assert resp.status_code == 302
    # No new project created.
    assert Project.query.count() == before


def test_clone_project_rejects_blank_name(auth_client, project_ctx):
    src = Project(name="src")
    db.session.add(src)
    db.session.commit()
    before = Project.query.count()

    resp = auth_client.post(f"/projects/{src.id}/clone", data={"name": ""})
    assert resp.status_code == 302
    assert Project.query.count() == before


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_project_removes_row_and_leaderboards(auth_client, project_ctx):
    p = Project(name="doomed")
    db.session.add(p)
    db.session.flush()

    ds = Dataset(name="ds_for_doomed")
    db.session.add(ds)
    db.session.flush()
    lb = Leaderboard(name="lb", project_id=p.id, summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.commit()

    resp = auth_client.post(f"/projects/{p.id}/delete")
    assert resp.status_code == 302

    db.session.expire_all()
    assert Project.query.get(p.id) is None
    # Leaderboards cascaded.
    assert Leaderboard.query.count() == 0
    # Dataset is global → not deleted.
    assert Dataset.query.filter_by(name="ds_for_doomed").count() == 1


# ---------------------------------------------------------------------------
# Project context middleware
# ---------------------------------------------------------------------------


def test_project_route_with_unknown_name_redirects_to_projects(client):
    resp = client.get("/no_such_project/", follow_redirects=False)
    assert resp.status_code == 302
    assert "/projects" in resp.headers["Location"]


def test_project_route_with_known_name_passes_context(client):
    p = Project(name="known_proj")
    db.session.add(p)
    db.session.commit()

    resp = client.get("/known_proj/", follow_redirects=False)
    # The index renders (200) — not redirected away.
    assert resp.status_code == 200
