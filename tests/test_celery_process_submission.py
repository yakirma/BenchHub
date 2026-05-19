"""End-to-end tests for the Celery task `process_submission`.

Eager mode is enabled in conftest, so `.delay(...)` runs synchronously. Each
test wires up a real dataset + leaderboard + metric(s) and asserts the task's
DB side-effects: MetricResult rows, per-sample CustomField persistence,
processing_status transitions.
"""
import json

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
from tasks import process_submission


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_dataset(samples_with_gt):
    """samples_with_gt: list of (sample_name, {gt_field: value, ...})."""
    ds = Dataset(name="task_ds")
    db.session.add(ds)
    db.session.flush()
    for s_name, gt_fields in samples_with_gt:
        sample = Sample(dataset_id=ds.id, name=s_name)
        db.session.add(sample)
        db.session.flush()
        for k, v in gt_fields.items():
            db.session.add(
                CustomField(
                    sample_id=sample.id, name=k, data_type="scalar", value_float=v
                )
            )
    db.session.commit()
    return ds


def _make_leaderboard(ds):
    lb = Leaderboard(name="task_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.commit()
    return lb


def _make_submission(lb, predictions_per_sample):
    """predictions_per_sample: {sample_name: {field_name: value, ...}}."""
    sub = Submission(name="task_sub", leaderboard_id=lb.id)
    db.session.add(sub)
    db.session.flush()
    for s_name, fields in predictions_per_sample.items():
        for k, v in fields.items():
            db.session.add(
                CustomField(
                    submission_id=sub.id,
                    sample_name=s_name,
                    name=k,
                    data_type="scalar",
                    value_float=v,
                )
            )
    db.session.commit()
    return sub


def _make_metric(name, code, *, is_aggregated=False):
    gm = GlobalMetric(name=name, python_code=code, is_aggregated=is_aggregated)
    db.session.add(gm)
    db.session.commit()
    return gm


def _attach_metric(lb, gm, arg_mappings, *, target_name=None, pooling_type="mean", pooling_percentile=None, tag_filter=None):
    lm = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        arg_mappings=json.dumps(arg_mappings),
        target_name=target_name,
        pooling_type=pooling_type,
        pooling_percentile=pooling_percentile,
        tag_filter=tag_filter,
    )
    db.session.add(lm)
    db.session.commit()
    return lm


# ---------------------------------------------------------------------------
# Per-sample metric
# ---------------------------------------------------------------------------


def test_per_sample_metric_aggregates_to_mean(client):
    ds = _make_dataset([
        ("s1", {"gt": 10.0}),
        ("s2", {"gt": 20.0}),
        ("s3", {"gt": 30.0}),
    ])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {
        "s1": {"pred": 11.0},  # error = 1
        "s2": {"pred": 18.0},  # error = 2
        "s3": {"pred": 33.0},  # error = 3
    })

    gm = _make_metric(
        "abs_err",
        "def abs_err(pred, target):\n    return abs(pred - target)\n",
        is_aggregated=False,
    )
    lm = _attach_metric(lb, gm, {"pred": "sub_pred", "target": "gt"}, pooling_type="mean")

    process_submission.delay(sub.id)

    db.session.expire_all()
    sub = Submission.query.get(sub.id)
    assert sub.processing_status == "Processed"

    result = MetricResult.query.filter_by(submission_id=sub.id, leaderboard_metric_id=lm.id).first()
    assert result.value == pytest.approx((1 + 2 + 3) / 3)
    assert result.error_message is None

    # Per-sample CustomFields persisted as name=lm_<id>.
    rows = CustomField.query.filter_by(submission_id=sub.id, name=f"lm_{lm.id}").all()
    by_sample = {cf.sample_name: cf.value_float for cf in rows}
    assert by_sample == {"s1": 1.0, "s2": 2.0, "s3": 3.0}


@pytest.mark.parametrize(
    "pooling_type,kwargs,values,expected",
    [
        ("mean",       {},                     [1, 2, 3, 4], 2.5),
        ("median",     {},                     [1, 2, 3, 4], 2.5),
        ("percentile", {"pooling_percentile": 50}, [1, 2, 3, 4, 5], 3.0),
        ("percentile", {"pooling_percentile": 95}, [1, 2, 3, 4, 5], 4.8),
    ],
)
def test_pooling_modes(client, pooling_type, kwargs, values, expected):
    ds = _make_dataset([(f"s{i}", {"gt": 0.0}) for i in range(len(values))])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {f"s{i}": {"pred": float(v)} for i, v in enumerate(values)})

    gm = _make_metric(
        "passthrough",
        "def passthrough(pred, target):\n    return pred\n",
        is_aggregated=False,
    )
    lm = _attach_metric(lb, gm, {"pred": "sub_pred", "target": "gt"}, pooling_type=pooling_type, **kwargs)

    process_submission.delay(sub.id)

    db.session.expire_all()
    result = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).first()
    assert result.value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Aggregated metric
# ---------------------------------------------------------------------------


def test_aggregated_metric_receives_list_of_values(client):
    ds = _make_dataset([("s1", {}), ("s2", {}), ("s3", {})])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {
        "s1": {"pred": 1.0},
        "s2": {"pred": 2.0},
        "s3": {"pred": 4.0},
    })

    gm = _make_metric(
        "sum_total",
        "def sum_total(values):\n    return float(sum(v for v in values if v is not None))\n",
        is_aggregated=True,
    )
    lm = _attach_metric(lb, gm, {"values": "sub_pred"})

    process_submission.delay(sub.id)

    db.session.expire_all()
    result = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).first()
    assert result.value == pytest.approx(7.0)


def test_aggregated_consumes_per_sample_output_via_lm_id(client):
    """Pin the working dependency path: aggregated metric B consumes per-sample
    metric A's output via the literal `lm_<id>` key (which is the actual stash
    key used at runtime — see the bug-pin test below)."""
    ds = _make_dataset([
        ("s1", {"gt": 10.0}),
        ("s2", {"gt": 20.0}),
        ("s3", {"gt": 30.0}),
    ])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {
        "s1": {"pred": 12.0},  # err=2
        "s2": {"pred": 25.0},  # err=5
        "s3": {"pred": 31.0},  # err=1
    })

    gm_a = _make_metric(
        "per_sample_err",
        "def per_sample_err(pred, target):\n    return abs(pred - target)\n",
        is_aggregated=False,
    )
    lm_a = _attach_metric(lb, gm_a, {"pred": "sub_pred", "target": "gt"}, target_name="A")

    gm_b = _make_metric(
        "max_err",
        "def max_err(values):\n    return float(max(v for v in values if v is not None))\n",
        is_aggregated=True,
    )
    # NOTE: must reference A's *stash key* (lm_<id>), not its target_name "A".
    lm_b = _attach_metric(lb, gm_b, {"values": f"lm_{lm_a.id}"}, target_name="B")

    process_submission.delay(sub.id)

    db.session.expire_all()
    result_b = MetricResult.query.filter_by(leaderboard_metric_id=lm_b.id).first()
    assert result_b.value == pytest.approx(5.0)


def test_dependency_via_target_name_does_not_resolve_at_runtime(client):
    """REAL BUG: sort_metrics_by_dependency builds the topo graph on friendly
    names (target_name or global_metric.name), so it correctly orders A before
    B. But at runtime, per-sample outputs are stashed in each context dict
    under the key `lm_<id>` — NOT the friendly name. As a result, an
    aggregated metric whose arg_mappings reference the friendly name reads
    `[None, None, ...]` and falls through to error or 0.

    Pin the broken behavior — flip when the runtime stash uses friendly names
    (or arg_mappings translates them at lookup time)."""
    ds = _make_dataset([("s1", {"gt": 10.0}), ("s2", {"gt": 20.0})])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {"s1": {"pred": 11.0}, "s2": {"pred": 22.0}})

    gm_a = _make_metric(
        "per_sample_err",
        "def per_sample_err(pred, target):\n    return abs(pred - target)\n",
        is_aggregated=False,
    )
    _attach_metric(lb, gm_a, {"pred": "sub_pred", "target": "gt"}, target_name="A")

    # B references A by friendly name — the natural way a user might wire it.
    gm_b = _make_metric(
        "max_err",
        "def max_err(values):\n    return float(max(v for v in values if v is not None))\n",
        is_aggregated=True,
    )
    lm_b = _attach_metric(lb, gm_b, {"values": "A"}, target_name="B")

    process_submission.delay(sub.id)

    db.session.expire_all()
    result_b = MetricResult.query.filter_by(leaderboard_metric_id=lm_b.id).first()
    # Bug: B receives [None, None] → max() raises → value is None and error_message is set.
    assert result_b.value is None
    assert result_b.error_message is not None


# ---------------------------------------------------------------------------
# tag_filter (per-metric subsetting)
# ---------------------------------------------------------------------------


def test_tag_filter_restricts_metric_to_subset(client):
    ds = _make_dataset([("s1", {}), ("s2", {}), ("s3", {})])
    # Tag s1 and s3 with "easy"; s2 with "hard".
    samples = {s.name: s for s in Sample.query.filter_by(dataset_id=ds.id).all()}
    samples["s1"].tags = "easy"
    samples["s2"].tags = "hard"
    samples["s3"].tags = "easy,extra"
    db.session.commit()

    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {
        "s1": {"pred": 1.0},
        "s2": {"pred": 100.0},   # would dominate if not filtered out
        "s3": {"pred": 3.0},
    })

    gm = _make_metric(
        "passthrough",
        "def passthrough(pred):\n    return pred\n",
        is_aggregated=False,
    )
    lm = _attach_metric(lb, gm, {"pred": "sub_pred"}, tag_filter="easy", pooling_type="mean")

    process_submission.delay(sub.id)

    db.session.expire_all()
    result = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).first()
    # Only s1 and s3 → mean(1, 3) = 2.
    assert result.value == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_metric_runtime_error_recorded_in_metric_result(client):
    ds = _make_dataset([("s1", {})])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {"s1": {"pred": 1.0}})

    gm = _make_metric(
        "boom",
        "def boom(pred):\n    raise ValueError('intentional')\n",
        is_aggregated=False,
    )
    lm = _attach_metric(lb, gm, {"pred": "sub_pred"})

    process_submission.delay(sub.id)

    db.session.expire_all()
    sub = Submission.query.get(sub.id)
    # The whole submission still ends up "Processed" — only this metric's row records the error.
    assert sub.processing_status == "Processed"

    result = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).first()
    assert result.value is None
    assert result.error_message is not None
    assert "ValueError" in result.error_message


# ---------------------------------------------------------------------------
# sample_filters (passed through from batch routes)
# ---------------------------------------------------------------------------


def test_sample_filters_persisted_on_submission(client):
    ds = _make_dataset([("s1", {}), ("s2", {})])
    # Tag samples to enable include filtering.
    samples = {s.name: s for s in Sample.query.filter_by(dataset_id=ds.id).all()}
    samples["s1"].tags = "good"
    samples["s2"].tags = "good"
    db.session.commit()

    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {"s1": {"pred": 1.0}, "s2": {"pred": 2.0}})
    gm = _make_metric(
        "passthrough",
        "def passthrough(pred):\n    return pred\n",
        is_aggregated=False,
    )
    _attach_metric(lb, gm, {"pred": "sub_pred"})

    filters = {"include": {"enabled": True, "tags": ["good"]}}
    process_submission.delay(sub.id, sample_filters=filters)

    db.session.expire_all()
    sub = Submission.query.get(sub.id)
    assert sub.last_sample_filter is not None
    assert json.loads(sub.last_sample_filter) == filters


def test_no_filters_clears_last_sample_filter(client):
    ds = _make_dataset([("s1", {})])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {"s1": {"pred": 1.0}})
    sub.last_sample_filter = '{"stale": true}'
    db.session.commit()

    gm = _make_metric(
        "passthrough",
        "def passthrough(pred):\n    return pred\n",
        is_aggregated=False,
    )
    _attach_metric(lb, gm, {"pred": "sub_pred"})

    process_submission.delay(sub.id)

    db.session.expire_all()
    sub = Submission.query.get(sub.id)
    assert sub.last_sample_filter is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_metrics_defined_just_marks_processed(client):
    ds = _make_dataset([("s1", {})])
    lb = _make_leaderboard(ds)  # zero metrics attached
    sub = _make_submission(lb, {"s1": {"pred": 1.0}})

    process_submission.delay(sub.id)

    db.session.expire_all()
    sub = Submission.query.get(sub.id)
    assert sub.processing_status == "Processed"
    assert MetricResult.query.count() == 0


def test_recalculation_replaces_prior_metric_result(client):
    ds = _make_dataset([("s1", {})])
    lb = _make_leaderboard(ds)
    sub = _make_submission(lb, {"s1": {"pred": 5.0}})
    gm = _make_metric(
        "passthrough",
        "def passthrough(pred):\n    return pred\n",
        is_aggregated=False,
    )
    lm = _attach_metric(lb, gm, {"pred": "sub_pred"})

    # First run.
    process_submission.delay(sub.id)
    # Tweak the prediction.
    cf = CustomField.query.filter_by(submission_id=sub.id, name="pred").first()
    cf.value_float = 9.0
    db.session.commit()
    # Re-run.
    process_submission.delay(sub.id)

    db.session.expire_all()
    # Exactly one result row remains (existing was deleted before insert).
    results = MetricResult.query.filter_by(leaderboard_metric_id=lm.id).all()
    assert len(results) == 1
    assert results[0].value == pytest.approx(9.0)


