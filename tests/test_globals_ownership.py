"""Phase 1 Slice 4 — ownership of GlobalMetric / GlobalVisualization.

Same shape as Slice 2/3 (Project, Dataset, Leaderboard, Submission). The
extra wrinkle: GlobalMetric.name is globally UNIQUE (not scoped to owner),
so two users can't pick the same name. Pinned in
test_upload_metric_owned_by_other_user_blocked.
"""
import pytest

from app import (
    GlobalMetric,
    GlobalVisualization,
    Project,
    User,
    db,
)


@pytest.fixture
def stranger(db_session):
    u = User(
        email="ext@example.com",
        display_name="Stranger",
        oauth_provider="github",
        oauth_sub="ext-sub",
    )
    db_session.add(u)
    db_session.commit()
    return u


@pytest.fixture
def proj(db_session, logged_in_user):
    """A project owned by logged_in_user, used so client.set_cookie sets up
    project context for the project-name-routed metric/viz endpoints."""
    p = Project(name="metric_proj", owner_user_id=logged_in_user.id)
    db_session.add(p)
    db_session.commit()
    return p


# ---------------------------------------------------------------------------
# Anonymous → /login redirect
# ---------------------------------------------------------------------------


def test_anon_create_metric_redirects_to_login(client, proj):
    resp = client.post(
        f"/{proj.name}/metrics/create",
        data={"name": "anon_m", "python_code": "def m(): return 1\n"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    assert GlobalMetric.query.count() == 0


def test_anon_create_visualization_redirects_to_login(client, proj):
    resp = client.post(
        f"/{proj.name}/create_visualization",
        data={"name": "anon_v", "python_code": "def v(): pass\n"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# Owner stamping on creation
# ---------------------------------------------------------------------------


def test_create_metric_stamps_owner(auth_client, proj, logged_in_user):
    auth_client.post(
        f"/{proj.name}/metrics/create",
        data={"name": "owned_m", "python_code": "def m(): return 1\n"},
    )
    gm = GlobalMetric.query.filter_by(name="owned_m").first()
    assert gm.owner_user_id == logged_in_user.id


def test_create_visualization_stamps_owner(auth_client, proj, logged_in_user):
    auth_client.post(
        f"/{proj.name}/create_visualization",
        data={"name": "owned_v", "python_code": "def v(): pass\n"},
    )
    gv = GlobalVisualization.query.filter_by(name="owned_v").first()
    assert gv.owner_user_id == logged_in_user.id


# ---------------------------------------------------------------------------
# @owner_required: non-owner gets 403
# ---------------------------------------------------------------------------


def test_non_owner_cannot_edit_metric(auth_client, proj, db_session, stranger):
    gm = GlobalMetric(
        name="not_yours_m",
        python_code="def m(): return 1\n",
        owner_user_id=stranger.id,
    )
    db_session.add(gm)
    db_session.commit()

    resp = auth_client.post(
        f"/{proj.name}/metrics/{gm.id}/edit",
        data={"name": "renamed", "python_code": "def m2(): return 2\n"},
    )
    assert resp.status_code == 403


def test_non_owner_cannot_delete_metric(auth_client, proj, db_session, stranger):
    gm = GlobalMetric(
        name="del_blocked",
        python_code="def m(): return 1\n",
        owner_user_id=stranger.id,
    )
    db_session.add(gm)
    db_session.commit()

    resp = auth_client.post(f"/{proj.name}/metrics/{gm.id}/delete")
    assert resp.status_code == 403

    db_session.expire_all()
    assert GlobalMetric.query.get(gm.id) is not None  # still there


def test_non_owner_cannot_edit_visualization(auth_client, proj, db_session, stranger):
    gv = GlobalVisualization(
        name="not_yours_v",
        python_code="def v(): pass\n",
        owner_user_id=stranger.id,
    )
    db_session.add(gv)
    db_session.commit()

    resp = auth_client.post(
        f"/{proj.name}/visualizations/{gv.id}/edit",
        data={"name": "renamed_v"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Owner can edit/delete their own
# ---------------------------------------------------------------------------


def test_owner_can_delete_own_metric(auth_client, proj, db_session, logged_in_user):
    gm = GlobalMetric(
        name="mine_to_delete",
        python_code="def m(): return 1\n",
        owner_user_id=logged_in_user.id,
    )
    db_session.add(gm)
    db_session.commit()

    resp = auth_client.post(f"/{proj.name}/metrics/{gm.id}/delete")
    assert resp.status_code == 302

    db_session.expire_all()
    assert GlobalMetric.query.get(gm.id) is None


# ---------------------------------------------------------------------------
# upload_metric: name-collision guard (the route does an upsert by name)
# ---------------------------------------------------------------------------


def test_upload_metric_owned_by_other_user_blocked(
    auth_client, proj, db_session, stranger
):
    """upload_metric is an upsert keyed on name. The multi-tenancy guard
    prevents user A from overwriting user B's metric of the same name —
    even though name is still globally unique, only the owner can update
    the row in place."""
    gm = GlobalMetric(
        name="shared_name",
        python_code="def m(): return 1\n",
        owner_user_id=stranger.id,
    )
    db_session.add(gm)
    db_session.commit()
    original_code = gm.python_code

    # Build a tiny fake .txt upload containing different code.
    import io
    fake_file = (io.BytesIO(b"def m(): return 999\n"), "shared_name.txt")

    resp = auth_client.post(
        f"/{proj.name}/metrics/upload",
        data={"metric_name": "shared_name", "metric_file": fake_file},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302  # redirect with flash

    db_session.expire_all()
    refreshed = GlobalMetric.query.filter_by(name="shared_name").first()
    # Untouched.
    assert refreshed.python_code == original_code
    assert refreshed.owner_user_id == stranger.id


def test_upload_metric_overwrite_own(auth_client, proj, db_session, logged_in_user):
    """Same upload_metric route, but the existing metric IS owned by the
    current user — the upsert should succeed."""
    gm = GlobalMetric(
        name="mine_to_overwrite",
        python_code="def m(): return 1\n",
        owner_user_id=logged_in_user.id,
    )
    db_session.add(gm)
    db_session.commit()

    import io
    fake_file = (io.BytesIO(b"def m(): return 42\n"), "mine_to_overwrite.txt")

    resp = auth_client.post(
        f"/{proj.name}/metrics/upload",
        data={"metric_name": "mine_to_overwrite", "metric_file": fake_file},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302

    db_session.expire_all()
    refreshed = GlobalMetric.query.filter_by(name="mine_to_overwrite").first()
    assert "return 42" in refreshed.python_code


# ---------------------------------------------------------------------------
# List filtering on /metrics and /visualizations
# ---------------------------------------------------------------------------


def test_metrics_list_filters_private_other_owner(
    auth_client, proj, db_session, logged_in_user, stranger
):
    db_session.add_all([
        GlobalMetric(name="m_pub", python_code="def m(): return 1",
                     owner_user_id=stranger.id, visibility="public"),
        GlobalMetric(name="m_priv_strangers", python_code="def m(): return 1",
                     owner_user_id=stranger.id, visibility="private"),
        GlobalMetric(name="m_priv_mine", python_code="def m(): return 1",
                     owner_user_id=logged_in_user.id, visibility="private"),
    ])
    db_session.commit()

    resp = auth_client.get(f"/{proj.name}/metrics")
    body = resp.data
    assert b"m_pub" in body
    assert b"m_priv_mine" in body  # I own it
    assert b"m_priv_strangers" not in body


def test_visualizations_list_filters_private_other_owner(
    auth_client, proj, db_session, logged_in_user, stranger
):
    db_session.add_all([
        GlobalVisualization(name="v_pub", python_code="def v(): pass",
                            owner_user_id=stranger.id, visibility="public"),
        GlobalVisualization(name="v_priv_strangers", python_code="def v(): pass",
                            owner_user_id=stranger.id, visibility="private"),
        GlobalVisualization(name="v_priv_mine", python_code="def v(): pass",
                            owner_user_id=logged_in_user.id, visibility="private"),
    ])
    db_session.commit()

    resp = auth_client.get(f"/{proj.name}/visualizations")
    body = resp.data
    assert b"v_pub" in body
    assert b"v_priv_mine" in body
    assert b"v_priv_strangers" not in body
