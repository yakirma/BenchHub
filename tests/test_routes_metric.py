"""Route tests for metric management.

Two layers:
1. Global metrics (project-scoped URL but DB-global): create / edit / delete /
   download / DLP-base64 path.
2. Leaderboard metrics (link table): add / edit / delete with arg-mapping form,
   summary_metrics auto-update, recalculation dispatch.
"""

import pytest

from app import (
    Dataset,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    db,
)


@pytest.fixture
def project(db_session, client):
    import types
    return types.SimpleNamespace(id=0, name='legacy')


@pytest.fixture
def metric(db_session):
    gm = GlobalMetric(
        name="abs_err",
        description="Absolute error",
        python_code="def abs_err(pred, target):\n    return abs(pred - target)\n",
        is_aggregated=False,
    )
    db.session.add(gm)
    db.session.commit()
    return gm


@pytest.fixture
def leaderboard(db_session, project):
    ds = Dataset(name="metric_ds")
    db.session.add(ds)
    db.session.flush()
    lb = Leaderboard(name="metric_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.commit()
    return lb


# ---------------------------------------------------------------------------
# Global metric: list / create / edit / delete / download
# ---------------------------------------------------------------------------


def test_metrics_view_renders(client, project, metric):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"abs_err" in resp.data


def test_create_metric_persists_to_db(auth_client, project, logged_in_user):
    code = "def m(x):\n    return x * 2\n"
    resp = auth_client.post(
        "/metrics/create",
        data={
            "name": "doubler",
            "description": "doubles input",
            "python_code": code,
        },
    )
    assert resp.status_code == 302

    gm = GlobalMetric.query.filter_by(name="doubler").first()
    assert gm is not None
    assert gm.python_code.strip() == code.strip()
    assert gm.owner_user_id == logged_in_user.id


def test_create_metric_blocks_zip_placeholder(auth_client, project):
    resp = auth_client.post(
        "/metrics/create",
        data={
            "name": "bad",
            "python_code": "Implementation will be loaded from ZIP",
        },
    )
    assert resp.status_code == 302
    assert GlobalMetric.query.filter_by(name="bad").count() == 0


def test_create_metric_blocks_blank_code(auth_client, project):
    resp = auth_client.post(
        "/metrics/create",
        data={"name": "empty", "python_code": ""},
    )
    assert resp.status_code == 302
    assert GlobalMetric.query.filter_by(name="empty").count() == 0


def test_edit_metric_updates_fields(auth_client, project, metric):
    resp = auth_client.post(
        f"/metrics/{metric.id}/edit",
        data={
            "name": "renamed",
            "description": "new description",
            "python_code": "def renamed(x):\n    return -x\n",
            "is_aggregated": "on",
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    fresh = GlobalMetric.query.get(metric.id)
    assert fresh.name == "renamed"
    assert fresh.description == "new description"
    assert "return -x" in fresh.python_code
    assert fresh.is_aggregated is True


def test_edit_metric_404_unknown(auth_client, project):
    resp = auth_client.post(
        "/metrics/9999/edit",
        data={"name": "x", "python_code": "def x(): return 1"},
    )
    assert resp.status_code == 404


def test_delete_metric_removes_row(auth_client, project, metric):
    metric_id = metric.id

    resp = auth_client.post(f"/metrics/{metric_id}/delete")
    assert resp.status_code == 302

    assert GlobalMetric.query.get(metric_id) is None


def test_download_metric_returns_python_text(client, project, metric):
    resp = client.get(f"/metrics/{metric.id}/download")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/plain")
    assert b"def abs_err" in resp.data


# ---------------------------------------------------------------------------
# Leaderboard metric: add / delete
# ---------------------------------------------------------------------------


def test_add_leaderboard_metric_creates_link_with_arg_mappings(
    client, project, metric, leaderboard
):
    proj_name, lb_id = project.name, leaderboard.id
    resp = client.post(
        f"/leaderboard/{lb_id}/leaderboard_metric/add",
        data={
            "global_metric_id": str(metric.id),
            "arg_name[]": ["pred", "target"],
            "source[]": ["sub", "gt"],
            "field_name[]": ["pred_value", "gt_value"],
            "display_name": "AbsErr_Pred_GT",
            "sort_direction": "lower_is_better",
        },
    )
    assert resp.status_code == 302

    lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb_id).first()
    assert lm is not None
    assert lm.target_name == "AbsErr_Pred_GT"
    assert lm.sort_direction == "lower_is_better"

    import json
    mappings = json.loads(lm.arg_mappings)
    # source=sub adds "sub_" prefix; source=gt adds "gt_" prefix.
    assert mappings == {"pred": "sub_pred_value", "target": "gt_gt_value"}


def test_add_leaderboard_metric_appends_lm_id_to_summary_metrics(
    client, project, metric, leaderboard
):
    proj_name, lb_id = project.name, leaderboard.id
    client.post(
        f"/leaderboard/{lb_id}/leaderboard_metric/add",
        data={
            "global_metric_id": str(metric.id),
            "arg_name[]": ["x"],
            "source[]": ["sub"],
            "field_name[]": ["pred"],
            "display_name": "M",
        },
    )

    db.session.expire_all()
    lb = Leaderboard.query.get(lb_id)
    lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb_id).first()
    # summary_metrics auto-extended with the new lm_<id>.
    assert f"lm_{lm.id}" in (lb.summary_metrics or "")


def test_add_second_metric_keeps_first_in_summary_metrics(
    client, project, metric, leaderboard
):
    """Adding a 2nd metric must NOT drop the 1st from summary_metrics —
    every bound metric stays visible (regression: a metric not in
    summary_metrics only showed via the empty-list 'show all' fallback,
    so populating the list silently hid it)."""
    lb_id = leaderboard.id
    # Simulate a pre-existing bound metric that was never added to
    # summary_metrics (older creation path): bind directly, leave
    # summary_metrics empty.
    import json
    pre = LeaderboardMetric(
        leaderboard_id=lb_id, global_metric_id=metric.id,
        target_name="pre", arg_mappings=json.dumps({"x": "sub_pred"}),
        pooling_type="mean",
    )
    db.session.add(pre)
    lb = Leaderboard.query.get(lb_id)
    lb.summary_metrics = ""  # the "show all" fallback state
    db.session.commit()
    pre_id = pre.id

    # Now add a second metric through the route.
    client.post(
        f"/leaderboard/{lb_id}/leaderboard_metric/add",
        data={
            "global_metric_id": str(metric.id),
            "arg_name[]": ["x"],
            "source[]": ["sub"],
            "field_name[]": ["pred"],
            "display_name": "M2",
        },
    )

    db.session.expire_all()
    lb = Leaderboard.query.get(lb_id)
    summary = lb.summary_metrics or ""
    # BOTH the pre-existing metric and the new one are present.
    assert f"lm_{pre_id}" in summary
    new_lm = LeaderboardMetric.query.filter_by(
        leaderboard_id=lb_id, target_name="M2",
    ).first()
    assert f"lm_{new_lm.id}" in summary


def test_add_leaderboard_metric_supports_scalar_literal_argument(
    client, project, metric, leaderboard
):
    """A SCALAR: prefix in arg_mappings lets a metric receive a constant
    rather than a context lookup. Pin the wiring."""
    proj_name, lb_id = project.name, leaderboard.id
    client.post(
        f"/leaderboard/{lb_id}/leaderboard_metric/add",
        data={
            "global_metric_id": str(metric.id),
            "arg_name[]": ["pred", "threshold"],
            "source[]": ["sub", "scalar"],
            "field_name[]": ["pred_value", "0.5"],
            "display_name": "T",
        },
    )

    import json
    lm = LeaderboardMetric.query.first()
    mappings = json.loads(lm.arg_mappings)
    assert mappings["threshold"] == "SCALAR:0.5"


def test_delete_leaderboard_metric_removes_link_and_prunes_summary(
    client, project, metric, leaderboard
):
    # Pre-create a leaderboard metric via direct DB insert.
    import json
    lm = LeaderboardMetric(
        leaderboard_id=leaderboard.id,
        global_metric_id=metric.id,
        arg_mappings=json.dumps({"x": "sub_pred"}),
        target_name="DeleteMe",
    )
    db.session.add(lm)
    db.session.flush()
    leaderboard.summary_metrics = f"lm_{lm.id}"
    db.session.commit()

    proj_name, lb_id, lm_id = project.name, leaderboard.id, lm.id

    resp = client.post(
        f"/leaderboard/{lb_id}/leaderboard_metric/{lm_id}/delete"
    )
    assert resp.status_code == 302

    db.session.expire_all()
    assert LeaderboardMetric.query.get(lm_id) is None
    # summary_metrics CSV was pruned of the lm_<id> reference.
    lb_fresh = Leaderboard.query.get(lb_id)
    assert f"lm_{lm_id}" not in (lb_fresh.summary_metrics or "")


def test_delete_leaderboard_metric_403_for_wrong_leaderboard_id(
    client, project, metric, leaderboard
):
    """The route guards against deleting a metric belonging to a different
    leaderboard via path-id mismatch."""
    other_ds = Dataset(name="other_ds")
    db.session.add(other_ds)
    db.session.flush()
    other_lb = Leaderboard(name="other_lb", summary_metrics="")
    other_lb.datasets.append(other_ds)
    db.session.add(other_lb)
    db.session.flush()

    import json
    lm = LeaderboardMetric(
        leaderboard_id=leaderboard.id,
        global_metric_id=metric.id,
        arg_mappings=json.dumps({"x": "sub_pred"}),
    )
    db.session.add(lm)
    db.session.commit()

    proj_name, wrong_lb_id, lm_id = project.name, other_lb.id, lm.id
    resp = client.post(
        f"/leaderboard/{wrong_lb_id}/leaderboard_metric/{lm_id}/delete"
    )
    assert resp.status_code == 403
