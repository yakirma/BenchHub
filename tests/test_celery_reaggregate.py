"""Tests for the Celery task `reaggregate_submission_metrics`.

Recomputes MetricResult.value from existing per-sample CustomField rows
(name=lm_<id>) without re-running the user's metric code. Used when only
the pooling settings change.

Asymmetry pinned here: reaggregate supports pooling_type=min/max, but
process_submission does NOT — so a fresh run of a min/max-pooled metric
would silently fall through to mean. Worth flagging.
"""
import json
from unittest.mock import patch

import pytest

from app import (
    CustomField,
    Dataset,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    MetricResult,
    Sample,
    Submission,
    db,
)
from tasks import reaggregate_submission_metrics


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded(client):
    """Pre-populated submission with per-sample CustomField values
    already on disk (i.e. process_submission has previously run)."""
    ds = Dataset(name="reagg_ds")
    db.session.add(ds)
    db.session.flush()
    for n in ["s1", "s2", "s3", "s4", "s5"]:
        db.session.add(Sample(dataset_id=ds.id, name=n))
    db.session.flush()

    lb = Leaderboard(name="reagg_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    gm = GlobalMetric(
        name="passthrough",
        python_code="def passthrough(pred):\n    return pred\n",
        is_aggregated=False,
    )
    db.session.add(gm)
    db.session.flush()

    lm = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        arg_mappings=json.dumps({"pred": "sub_pred"}),
        target_name="thing",
        pooling_type="mean",
    )
    db.session.add(lm)
    db.session.flush()

    sub = Submission(name="reagg_sub", leaderboard_id=lb.id)
    db.session.add(sub)
    db.session.flush()

    # Per-sample values (what process_submission would have written).
    sample_values = {"s1": 10.0, "s2": 20.0, "s3": 30.0, "s4": 40.0, "s5": 50.0}
    for s_name, val in sample_values.items():
        db.session.add(
            CustomField(
                submission_id=sub.id,
                sample_name=s_name,
                name=f"lm_{lm.id}",
                data_type="scalar",
                value_float=val,
            )
        )

    # Existing MetricResult from a prior run (mean=30.0).
    db.session.add(MetricResult(submission_id=sub.id, leaderboard_metric_id=lm.id, value=30.0))
    db.session.commit()

    return {"lb": lb, "lm": lm, "sub": sub}


# ---------------------------------------------------------------------------
# Pooling modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pooling_type,kwargs,expected",
    [
        ("mean",       {},                     30.0),
        ("median",     {},                     30.0),
        ("percentile", {"pooling_percentile": 90}, 46.0),  # np.percentile([10..50], 90) == 46
        ("min",        {},                     10.0),
        ("max",        {},                     50.0),
    ],
)
def test_reaggregate_pooling_modes(seeded, pooling_type, kwargs, expected):
    lm = seeded["lm"]
    lm.pooling_type = pooling_type
    for k, v in kwargs.items():
        setattr(lm, k, v)
    db.session.commit()

    reaggregate_submission_metrics.delay(seeded["sub"].id)

    db.session.expire_all()
    result = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).first()
    assert result.value == pytest.approx(expected)


def test_reaggregate_does_not_invoke_user_metric_code(seeded):
    """Skipping execution is the whole point. Mock `evaluate_dynamic_metric`
    and confirm reaggregate never calls it."""
    seeded["lm"].pooling_type = "max"
    db.session.commit()

    with patch("metric_engine.evaluate_dynamic_metric") as eval_mock, \
         patch("tasks.evaluate_dynamic_metric", side_effect=eval_mock):
        reaggregate_submission_metrics.delay(seeded["sub"].id)

    eval_mock.assert_not_called()


def test_reaggregate_marks_status_processed(seeded):
    reaggregate_submission_metrics.delay(seeded["sub"].id)

    db.session.expire_all()
    sub = Submission.query.get(seeded["sub"].id)
    assert sub.processing_status == "Processed"


# ---------------------------------------------------------------------------
# Dirty-filter fallback
# ---------------------------------------------------------------------------


def test_dirty_filter_falls_back_to_full_process_submission(seeded):
    """When last_sample_filter is set, the submission was previously calculated
    on a subset, so per-sample values would be incomplete. The task must defer
    to process_submission for a full recalculation."""
    seeded["sub"].last_sample_filter = json.dumps({"include": {"enabled": True, "tags": ["foo"]}})
    db.session.commit()

    with patch("tasks.process_submission") as ps_mock:
        reaggregate_submission_metrics.delay(seeded["sub"].id)

    ps_mock.assert_called_once_with(seeded["sub"].id)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_per_sample_values_skips_metric(client):
    """If no CustomField rows exist for a metric (e.g. the metric was added
    after process_submission ran), reaggregate has nothing to pool — silently
    skip without overwriting the (possibly nonexistent) MetricResult."""
    ds = Dataset(name="empty_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name="s1"))
    db.session.flush()

    lb = Leaderboard(name="empty_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)

    gm = GlobalMetric(name="m", python_code="def m(): return 1", is_aggregated=False)
    db.session.add(gm)
    db.session.flush()

    lm = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        arg_mappings="{}",
        pooling_type="mean",
    )
    db.session.add(lm)

    sub = Submission(name="s", leaderboard_id=lb.id)
    db.session.add(sub)
    db.session.commit()

    reaggregate_submission_metrics.delay(sub.id)

    db.session.expire_all()
    # No MetricResult was created (no values to pool).
    assert MetricResult.query.filter_by(leaderboard_metric_id=lm.id).count() == 0


def test_aggregated_metrics_skipped_by_reaggregate(client):
    """Aggregated metrics don't have per-sample CustomField values to pool
    from, so reaggregate must skip them. Verify the existing MetricResult
    for an aggregated metric is left alone."""
    ds = Dataset(name="agg_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name="s1"))
    db.session.flush()

    lb = Leaderboard(name="agg_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)

    gm = GlobalMetric(
        name="agg_m",
        python_code="def agg_m(): return 1",
        is_aggregated=True,
    )
    db.session.add(gm)
    db.session.flush()

    lm = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        arg_mappings="{}",
        pooling_type="mean",
    )
    db.session.add(lm)

    sub = Submission(name="s", leaderboard_id=lb.id)
    db.session.add(sub)
    db.session.flush()

    pre = MetricResult(submission_id=sub.id, leaderboard_metric_id=lm.id, value=42.0)
    db.session.add(pre)
    db.session.commit()

    reaggregate_submission_metrics.delay(sub.id)

    db.session.expire_all()
    result = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).first()
    # Untouched.
    assert result.value == 42.0
