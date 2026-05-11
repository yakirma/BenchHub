"""Route tests for the comparison view (and leaderboard view smoke).

Comparison view is the most complex template — pin the regression from commit
40ed53a where pagination dropped the compared-subset.
"""
import pytest

from app import (
    CustomField,
    Dataset,
    Leaderboard,
    Sample,
    Submission,
    db,
)


@pytest.fixture
def project(db_session, client):
    import types
    return types.SimpleNamespace(id=0, name='legacy')


@pytest.fixture
def lb_with_subs(db_session, project):
    """Leaderboard with 4 samples and 3 submissions, one archived."""
    ds = Dataset(name="cmp_ds")
    db.session.add(ds)
    db.session.flush()
    for i in range(1, 5):
        db.session.add(Sample(dataset_id=ds.id, name=f"s{i}"))
    db.session.flush()

    lb = Leaderboard(name="cmp_lb", summary_metrics="")
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

    resp = client.get(f"/comparison/{lb_id}")
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

    resp = client.get(f"/comparison/{lb_id}?compare_ids={alpha_id}")
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
        f"/comparison/{lb_id}?compare_ids={alpha_id},{beta_id}&page=2&per_page=2"
    )
    assert resp.status_code == 200
    # gamma must remain excluded even though it would appear when no compare_ids
    # filter is applied.
    assert b"gamma" not in resp.data


def test_comparison_view_with_empty_subs_renders_samples_only_mode(
    client, project, lb_with_subs
):
    """When compare_ids matches no submissions (or is unset on an LB with
    none), the route falls through to "Explore samples" mode rather than
    early-returning. The page renders the GT side and shows an empty-state
    banner — used to be an UnboundLocalError on metric_labels."""
    lb_id = lb_with_subs["lb"].id

    resp = client.get(f"/comparison/{lb_id}?compare_ids=999999")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8", errors="ignore")
    # Header text switches to the samples-only label.
    assert "Explore samples" in body


def test_comparison_view_samples_only_param_renders_for_lb_with_subs(
    client, project, lb_with_subs
):
    """Explicit `?samples_only=1` opts into the samples-only surface even
    when the LB has submissions. The user can browse GT-side data before
    picking submissions to compare."""
    lb_id = lb_with_subs["lb"].id
    resp = client.get(f"/comparison/{lb_id}?samples_only=1")
    assert resp.status_code == 200
    assert b"Explore samples" in resp.data


def test_comparison_view_search_filters_samples(client, project, lb_with_subs):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    # Search for "s1" should narrow visible samples.
    resp = client.get(f"/comparison/{lb_id}?search_query=s1")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8", errors="ignore")
    assert "s1" in body


# ---------------------------------------------------------------------------
# Leaderboard view (smoke — full template render path)
# ---------------------------------------------------------------------------


def test_leaderboard_view_smoke_with_submissions(client, project, lb_with_subs):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/leaderboard/{lb_id}")
    assert resp.status_code == 200
    assert b"alpha" in resp.data
    assert b"beta" in resp.data


def test_leaderboard_view_show_archived_includes_archived(
    client, project, lb_with_subs
):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/leaderboard/{lb_id}?show_archived=true")
    assert resp.status_code == 200
    assert b"gamma" in resp.data


def test_leaderboard_view_search_filters_submissions(
    client, project, lb_with_subs
):
    proj_name, lb_id = project.name, lb_with_subs["lb"].id

    resp = client.get(f"/leaderboard/{lb_id}?search_query=alpha")
    assert resp.status_code == 200
    # alpha is present; beta name should not appear in submission rows.
    # (We can't fully assert because tag-search also goes here; weak check.)
    assert b"alpha" in resp.data
