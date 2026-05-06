"""Route tests for visualization management.

Mirrors the metric routes structure: GlobalVisualization CRUD + the
LeaderboardVisualization link table. Skip the actual image-execution endpoints
(those run user matplotlib code and are integration-heavy — covered in Phase 3
where it matters).
"""
import json

import pytest

from app import (
    Dataset,
    GlobalVisualization,
    Leaderboard,
    LeaderboardVisualization,
    Project,
    db,
)


@pytest.fixture
def project(db_session, client):
    p = Project(name="viz_proj")
    db.session.add(p)
    db.session.commit()
    client.set_cookie("active_project_id", str(p.id))
    return p


@pytest.fixture
def viz(db_session):
    gv = GlobalVisualization(
        name="line_plot",
        description="Plot a line",
        python_code="def line_plot(values):\n    pass\n",
        is_aggregated=False,
    )
    db.session.add(gv)
    db.session.commit()
    return gv


@pytest.fixture
def leaderboard(db_session, project):
    ds = Dataset(name="viz_ds")
    db.session.add(ds)
    db.session.flush()
    lb = Leaderboard(name="viz_lb", project_id=project.id, summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.commit()
    return lb


# ---------------------------------------------------------------------------
# Global visualization CRUD
# ---------------------------------------------------------------------------


def test_visualizations_view_renders(client, project, viz):
    resp = client.get(f"/{project.name}/visualizations")
    assert resp.status_code == 200
    assert b"line_plot" in resp.data


def test_create_visualization_persists(auth_client, project, logged_in_user):
    resp = auth_client.post(
        f"/{project.name}/create_visualization",
        data={
            "name": "scatter",
            "description": "Scatter plot",
            "python_code": "def scatter(x, y):\n    pass\n",
        },
    )
    assert resp.status_code == 302

    gv = GlobalVisualization.query.filter_by(name="scatter").first()
    assert gv is not None
    assert gv.owner_user_id == logged_in_user.id


def test_create_visualization_blocks_blank_code(auth_client, project):
    resp = auth_client.post(
        f"/{project.name}/create_visualization",
        data={"name": "empty", "python_code": ""},
    )
    assert resp.status_code == 302
    assert GlobalVisualization.query.filter_by(name="empty").count() == 0


def test_edit_visualization_updates_fields(auth_client, project, viz):
    resp = auth_client.post(
        f"/{project.name}/visualizations/{viz.id}/edit",
        data={
            "name": "renamed_viz",
            "description": "updated",
            "python_code": "def renamed_viz(): pass\n",
            "is_aggregated": "on",
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    fresh = GlobalVisualization.query.get(viz.id)
    assert fresh.name == "renamed_viz"
    assert fresh.is_aggregated is True


def test_delete_visualization_removes_row(auth_client, project, viz):
    viz_id = viz.id
    resp = auth_client.post(f"/{project.name}/visualizations/{viz_id}/delete")
    assert resp.status_code == 302
    assert GlobalVisualization.query.get(viz_id) is None


def test_download_visualization_returns_python_text(client, project, viz):
    resp = client.get(f"/{project.name}/visualizations/{viz.id}/download")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/plain")
    assert b"def line_plot" in resp.data


# ---------------------------------------------------------------------------
# Leaderboard visualization (link)
# ---------------------------------------------------------------------------


def test_add_leaderboard_visualization_creates_link_with_arg_mappings(
    client, project, viz, leaderboard
):
    proj_name, lb_id = project.name, leaderboard.id
    resp = client.post(
        f"/{proj_name}/leaderboard/{lb_id}/leaderboard_visualization/add",
        data={
            "global_visualization_id": str(viz.id),
            "viz_arg_name[]": ["x", "y"],
            "viz_source[]": ["gt", "sub"],
            "viz_field_name[]": ["x_field", "y_field"],
            "display_name": "MyPlot",
            "display_order": "5",
        },
    )
    assert resp.status_code == 302

    lv = LeaderboardVisualization.query.filter_by(leaderboard_id=lb_id).first()
    assert lv is not None
    assert lv.target_name == "MyPlot"
    assert lv.display_order == 5

    mappings = json.loads(lv.arg_mappings)
    assert mappings == {"x": "gt_x_field", "y": "sub_y_field"}


def test_add_leaderboard_visualization_auto_disambiguates_duplicate_name(
    client, project, viz, leaderboard
):
    """When the requested display_name is already taken on this LB, the route
    appends a counter (_1, _2, ...) to keep names unique."""
    proj_name, lb_id = project.name, leaderboard.id

    # First add — claims name "MyPlot".
    client.post(
        f"/{proj_name}/leaderboard/{lb_id}/leaderboard_visualization/add",
        data={
            "global_visualization_id": str(viz.id),
            "display_name": "MyPlot",
        },
    )
    # Second add with the same name — should land as MyPlot_1.
    client.post(
        f"/{proj_name}/leaderboard/{lb_id}/leaderboard_visualization/add",
        data={
            "global_visualization_id": str(viz.id),
            "display_name": "MyPlot",
        },
    )

    names = sorted(
        v.target_name
        for v in LeaderboardVisualization.query.filter_by(leaderboard_id=lb_id).all()
    )
    assert names == ["MyPlot", "MyPlot_1"]


def test_edit_leaderboard_visualization_updates_mappings(
    client, project, viz, leaderboard
):
    lv = LeaderboardVisualization(
        leaderboard_id=leaderboard.id,
        global_visualization_id=viz.id,
        arg_mappings=json.dumps({"x": "gt_old"}),
        target_name="Original",
        display_order=0,
    )
    db.session.add(lv)
    db.session.commit()

    proj_name, lb_id, lv_id = project.name, leaderboard.id, lv.id
    client.post(
        f"/{proj_name}/leaderboard/{lb_id}/leaderboard_visualization/{lv_id}/edit",
        data={
            "viz_arg_name[]": ["x"],
            "viz_source[]": ["sub"],
            "viz_field_name[]": ["new_field"],
            "display_name": "Updated",
            "display_order": "10",
        },
    )

    db.session.expire_all()
    fresh = LeaderboardVisualization.query.get(lv_id)
    assert fresh.target_name == "Updated"
    assert fresh.display_order == 10
    assert json.loads(fresh.arg_mappings) == {"x": "sub_new_field"}


def test_delete_leaderboard_visualization_removes_row(
    client, project, viz, leaderboard
):
    lv = LeaderboardVisualization(
        leaderboard_id=leaderboard.id,
        global_visualization_id=viz.id,
        arg_mappings="{}",
    )
    db.session.add(lv)
    db.session.commit()

    proj_name, lb_id, lv_id = project.name, leaderboard.id, lv.id
    resp = client.post(
        f"/{proj_name}/leaderboard/{lb_id}/leaderboard_visualization/{lv_id}/delete"
    )
    assert resp.status_code == 302
    assert LeaderboardVisualization.query.get(lv_id) is None
