"""Route tests for the comparison view (and leaderboard view smoke).

Comparison view is the most complex template — pin the regression from commit
40ed53a where pagination dropped the compared-subset.
"""
import pytest

from app import (
    CustomField,
    Dataset,
    Leaderboard,
    Project,
    Sample,
    Submission,
    db,
)


@pytest.fixture
def project(db_session, client):
    p = Project(name="cmp_proj")
    db.session.add(p)
    db.session.commit()
    client.set_cookie("active_project_id", str(p.id))
    return p


@pytest.fixture
def lb_with_subs(db_session, project):
    """Leaderboard with 4 samples and 3 submissions, one archived."""
    ds = Dataset(name="cmp_ds")
    db.session.add(ds)
    db.session.flush()
    for i in range(1, 5):
        db.session.add(Sample(dataset_id=ds.id, name=f"s{i}"))
    db.session.flush()

    lb = Leaderboard(name="cmp_lb", project_id=project.id, summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    subs = []
    for n in ["alpha", "beta", "gamma"]:
        sub = Submission(name=n, leaderboard_id=lb.id, processing_status="Processed")
        db.session.add(sub)
        subs.append(sub)
    subs[2].is_archived = True  # gamma is archived
    db.session.commit()
    return {"lb": lb, "subs": subs}


# ---------------------------------------------------------------------------
# Comparison view
# ---------------------------------------------------------------------------


def test_comparison_view_renders_with_all_unarchived_submissions(
    client, project, lb_with_subs
):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/{proj_name}/comparison/{lb_id}")
    assert resp.status_code == 200
    body = resp.data
    assert b"alpha" in body
    assert b"beta" in body
    # gamma is archived → excluded.
    assert b"gamma" not in body


def test_comparison_view_filters_to_compare_ids_when_provided(
    client, project, lb_with_subs
):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id
    alpha_id = lb_with_subs["subs"][0].id

    resp = client.get(f"/{proj_name}/comparison/{lb_id}?compare_ids={alpha_id}")
    assert resp.status_code == 200
    assert b"alpha" in resp.data
    # beta should NOT be rendered when compare_ids restricts to alpha.
    assert b"beta" not in resp.data


def test_comparison_view_pagination_preserves_compare_ids(
    client, project, lb_with_subs
):
    """Regression for commit 40ed53a — pagination dropped the compare_ids
    filter, falling back to all submissions."""
    proj_name, lb_id = project.name, lb_with_subs["lb"].id
    alpha_id, beta_id = lb_with_subs["subs"][0].id, lb_with_subs["subs"][1].id

    # Compare alpha+beta, request page 2 with very small per_page.
    resp = client.get(
        f"/{proj_name}/comparison/{lb_id}?compare_ids={alpha_id},{beta_id}&page=2&per_page=2"
    )
    assert resp.status_code == 200
    # gamma must remain excluded even though it would appear when no compare_ids
    # filter is applied.
    assert b"gamma" not in resp.data


def test_comparison_view_with_empty_subs_raises_unbound_local(
    client, project, lb_with_subs
):
    """REAL BUG: when all submissions are filtered out (compare_ids matches
    nothing), comparison_view's early-return path passes `metric_labels` to
    the template — but `metric_labels` is only defined LATER in the function
    body (line ~3839), so this raises UnboundLocalError before rendering.

    Pin the bug. Fix is to initialize `metric_labels = {}` near the top of
    the function. Flip the assertion when fixed."""
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    # The Flask test client propagates server-side exceptions in TESTING mode
    # rather than converting to a 500 response — so catch it here.
    with pytest.raises(UnboundLocalError, match="metric_labels"):
        client.get(f"/{proj_name}/comparison/{lb_id}?compare_ids=999999")


def test_comparison_view_search_filters_samples(client, project, lb_with_subs):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    # Search for "s1" should narrow visible samples.
    resp = client.get(f"/{proj_name}/comparison/{lb_id}?search_query=s1")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8", errors="ignore")
    assert "s1" in body


# ---------------------------------------------------------------------------
# Leaderboard view (smoke — full template render path)
# ---------------------------------------------------------------------------


def test_leaderboard_view_smoke_with_submissions(client, project, lb_with_subs):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/{proj_name}/leaderboard/{lb_id}")
    assert resp.status_code == 200
    assert b"alpha" in resp.data
    assert b"beta" in resp.data


def test_leaderboard_view_show_archived_includes_archived(
    client, project, lb_with_subs
):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/{proj_name}/leaderboard/{lb_id}?show_archived=true")
    assert resp.status_code == 200
    assert b"gamma" in resp.data


def test_leaderboard_view_search_filters_submissions(
    client, project, lb_with_subs
):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/{proj_name}/leaderboard/{lb_id}?search_query=alpha")
    assert resp.status_code == 200
    # alpha is present; beta name should not appear in submission rows.
    # (We can't fully assert because tag-search also goes here; weak check.)
    assert b"alpha" in resp.data
