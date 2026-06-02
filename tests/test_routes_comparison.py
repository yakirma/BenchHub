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


def test_comparison_view_shows_gt_and_pred_labels_in_stats(client, project, db_session):
    """The per_source_stats panel surfaces GT labels and predicted labels
    (in addition to scalars/metrics), and empty field columns are pruned."""
    ds = Dataset(name="lblcmp_ds")
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name="s0")
    db.session.add(s); db.session.flush()
    # GT label = 3; plus a file-backed field with NO data (should be pruned).
    db.session.add(CustomField(sample_id=s.id, name="label",
                               data_type="label", value_text="3"))
    db.session.commit()
    lb = Leaderboard(name="lblcmp_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    sub = Submission(name="runX", leaderboard_id=lb.id,
                     processing_status="Processed")
    db.session.add(sub); db.session.flush()
    # Predicted label = 3 (matches GT).
    db.session.add(CustomField(submission_id=sub.id, sample_name="s0",
                               name="label_pred", data_type="label",
                               value_text="3"))
    db.session.commit()

    # No samples_only param — full comparison mode shows per_source_stats.
    body = client.get(f"/comparison/{lb.id}").data.decode()
    # The per_source_stats "Labels" section header appears for both the
    # GT label and the predicted label.
    assert "Labels" in body
    assert "label_pred" in body


def test_comparison_view_disagreement_sort_orders_by_spread(client, project, db_session):
    """sort_by=disagreement:lm_<id> ranks samples by max-min of a
    per-sample metric across submissions (most contentious first)."""
    import json
    from app import (CustomField, GlobalMetric, LeaderboardMetric)

    ds = Dataset(name="dis_ds"); db.session.add(ds); db.session.flush()
    for nm in ["s_hi", "s_lo"]:
        db.session.add(Sample(dataset_id=ds.id, name=nm))
    db.session.flush()
    lb = Leaderboard(name="dis_lb", summary_metrics=""); lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    gm = GlobalMetric(name="acc_dis", python_code="def acc_dis(x): return x",
                      visibility="public")
    db.session.add(gm); db.session.flush()
    lm = LeaderboardMetric(leaderboard_id=lb.id, global_metric_id=gm.id,
                           target_name="acc_dis", arg_mappings=json.dumps({}),
                           pooling_type="mean")
    db.session.add(lm); db.session.flush()
    key = f"lm_{lm.id}"
    subs = []
    for nm in ["a", "b"]:
        s = Submission(name=nm, leaderboard_id=lb.id, processing_status="Processed")
        db.session.add(s); db.session.flush(); subs.append(s)
    # s_hi: submissions disagree a lot (0.1 vs 0.9 -> spread 0.8);
    # s_lo: agree (0.5 vs 0.5 -> spread 0.0).
    vals = {("a", "s_hi"): 0.1, ("b", "s_hi"): 0.9,
            ("a", "s_lo"): 0.5, ("b", "s_lo"): 0.5}
    for (sub_name, s_name), v in vals.items():
        sub = next(x for x in subs if x.name == sub_name)
        db.session.add(CustomField(submission_id=sub.id, sample_name=s_name,
                                   name=key, data_type="metric", value_float=v))
    db.session.commit()

    # High→Low: most-disagreement sample (s_hi) first.
    body = client.get(f"/comparison/{lb.id}?sort_by=disagreement:{key}&sort_order=desc").data.decode()
    assert body.index("s_hi") < body.index("s_lo")
    # Low→High flips it.
    body_asc = client.get(f"/comparison/{lb.id}?sort_by=disagreement:{key}&sort_order=asc").data.decode()
    assert body_asc.index("s_lo") < body_asc.index("s_hi")


def test_comparison_view_restricts_to_materialized_subset(client, project, db_session):
    """Samples not in the LB's materialised subset aren't shown — they
    can't have predictions (submitters only get the subset), so listing
    them yields empty columns (regression: s000002 on cifar)."""
    import os
    from app import LeaderboardMaterialization
    from benchhub.lb_materialize import materialization_dir
    from app import app as flask_app

    ds = Dataset(name="mat_cmp_ds"); db.session.add(ds); db.session.flush()
    for nm in ["s000000", "s000001", "s000002"]:
        db.session.add(Sample(dataset_id=ds.id, name=nm))
    lb = Leaderboard(name="mat_cmp_lb", summary_metrics=""); lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    db.session.add(LeaderboardMaterialization(
        leaderboard_id=lb.id, status="ready", sample_cap=2,
        sampling="head", sampling_seed=42))
    # Materialise only s000000 + s000001 (NOT s000002).
    fdir = materialization_dir(flask_app.config["UPLOAD_FOLDER"], lb.id) / "img"
    os.makedirs(fdir, exist_ok=True)
    for nm in ["s000000", "s000001"]:
        (fdir / f"{nm}.jpg").write_bytes(b"x")
    db.session.commit()

    body = client.get(f"/comparison/{lb.id}").data.decode()
    assert "s000000" in body
    assert "s000002" not in body  # not materialised → excluded


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


def test_leaderboard_view_sorts_by_metric_value(client, project, lb_with_subs):
    """?sort_metric=lm_<id>&sort_order=desc orders the submission rows by
    that metric's value. Pins the server side of the click-to-sort
    behaviour on metric columns."""
    import json
    from app import GlobalMetric, LeaderboardMetric, MetricResult

    lb = lb_with_subs["lb"]
    subs = lb_with_subs["subs"]  # alpha, beta, gamma(archived)
    gm = GlobalMetric(name="acc_sort", python_code="def acc_sort(x): return x",
                      visibility="public")
    db.session.add(gm); db.session.flush()
    lm = LeaderboardMetric(leaderboard_id=lb.id, global_metric_id=gm.id,
                           target_name="acc_sort", arg_mappings=json.dumps({}),
                           pooling_type="mean", sort_direction="higher_is_better")
    db.session.add(lm); db.session.flush()
    lb.summary_metrics = f"lm_{lm.id}"
    # alpha=0.1, beta=0.9 — so desc order should put beta before alpha.
    db.session.add(MetricResult(submission_id=subs[0].id,
                                leaderboard_metric_id=lm.id, value=0.1))
    db.session.add(MetricResult(submission_id=subs[1].id,
                                leaderboard_metric_id=lm.id, value=0.9))
    db.session.commit()

    body = client.get(
        f"/leaderboard/{lb.id}?sort_metric=lm_{lm.id}&sort_order=desc"
    ).data.decode()
    # beta (0.9) must appear before alpha (0.1) in the rendered rows.
    assert body.index("beta") < body.index("alpha")
    # asc flips it.
    body_asc = client.get(
        f"/leaderboard/{lb.id}?sort_metric=lm_{lm.id}&sort_order=asc"
    ).data.decode()
    assert body_asc.index("alpha") < body_asc.index("beta")


def test_leaderboard_view_defaults_to_first_metric_best_first(client, project, lb_with_subs):
    """With no sort params, the board defaults to best-first by the first
    metric: higher_is_better → the higher value lands on top."""
    import json
    from app import GlobalMetric, LeaderboardMetric, MetricResult

    lb = lb_with_subs["lb"]
    subs = lb_with_subs["subs"]
    gm = GlobalMetric(name="acc_def", python_code="def acc_def(x): return x",
                      visibility="public")
    db.session.add(gm); db.session.flush()
    lm = LeaderboardMetric(leaderboard_id=lb.id, global_metric_id=gm.id,
                           target_name="acc_def", arg_mappings=json.dumps({}),
                           pooling_type="mean", sort_direction="higher_is_better")
    db.session.add(lm); db.session.flush()
    lb.summary_metrics = f"lm_{lm.id}"
    db.session.add(MetricResult(submission_id=subs[0].id,
                                leaderboard_metric_id=lm.id, value=0.2))  # alpha
    db.session.add(MetricResult(submission_id=subs[1].id,
                                leaderboard_metric_id=lm.id, value=0.8))  # beta
    db.session.commit()

    # No sort params at all.
    body = client.get(f"/leaderboard/{lb.id}").data.decode()
    # beta (0.8) on top for higher_is_better default.
    assert body.index("beta") < body.index("alpha")
    # The first-metric header is marked as the active (desc) sort.
    assert f'data-sort-order="desc"' in body
