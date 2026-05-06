"""Tests for the env-flag dispatch between in-process exec and sandboxed
metric execution in tasks._eval_metric_batch and end-to-end through the
process_submission task.

Both backends already have their own unit tests (test_metric_engine,
test_sandbox_harness, test_sandbox_wrapper). These tests are about the
*wiring*: the right one fires based on BENCHHUB_SANDBOX_METRICS, and the
existing in-process default is preserved when the flag is unset.
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
import tasks


# ---------------------------------------------------------------------------
# Fixture: minimal LB with one per-sample metric and one submission ready
# to be processed end-to-end.
# ---------------------------------------------------------------------------


@pytest.fixture
def submission_with_metric(client):
    ds = Dataset(name="wiring_ds")
    db.session.add(ds)
    db.session.flush()
    for n in ["s1", "s2", "s3"]:
        db.session.add(Sample(dataset_id=ds.id, name=n))
    db.session.flush()

    lb = Leaderboard(name="wiring_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    sub = Submission(name="wiring_sub", leaderboard_id=lb.id)
    db.session.add(sub)
    db.session.flush()

    # Add per-sample predictions so a real eval has something to work with.
    for n, v in [("s1", 1.0), ("s2", 2.0), ("s3", 4.0)]:
        db.session.add(CustomField(
            submission_id=sub.id, sample_name=n, name="pred",
            field_type="scalar", value_float=v,
        ))

    gm = GlobalMetric(
        name="passthrough",
        python_code="def passthrough(x):\n    return x\n",
        is_aggregated=False,
    )
    db.session.add(gm)
    db.session.flush()

    lm = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        arg_mappings=json.dumps({"x": "sub_pred"}),
        target_name="pt",
    )
    db.session.add(lm)
    db.session.commit()
    return sub


# ---------------------------------------------------------------------------
# _eval_metric_batch dispatch
# ---------------------------------------------------------------------------


def test_default_uses_in_process_exec(monkeypatch):
    """Without the env flag, _eval_metric_batch must call evaluate_dynamic_metric
    once per context. evaluate_in_sandbox must NOT be called."""
    monkeypatch.delenv("BENCHHUB_SANDBOX_METRICS", raising=False)

    gm = GlobalMetric(
        name="d1", python_code="def f(): return 1\n", is_aggregated=False,
    )
    contexts = [{}, {}, {}]

    with patch("tasks.evaluate_in_sandbox") as sandbox_mock, \
         patch("tasks.evaluate_dynamic_metric", return_value=(1.0, None)) as exec_mock:
        results = tasks._eval_metric_batch(gm, contexts, "{}")

    sandbox_mock.assert_not_called()
    assert exec_mock.call_count == 3
    assert results == [(1.0, None), (1.0, None), (1.0, None)]


def test_flag_routes_to_sandbox(monkeypatch):
    """With BENCHHUB_SANDBOX_METRICS=1, _eval_metric_batch must call
    evaluate_in_sandbox ONCE with the full context list. The in-process
    path must NOT fire (any call would be wasted exec'ing untrusted code)."""
    monkeypatch.setenv("BENCHHUB_SANDBOX_METRICS", "1")

    gm = GlobalMetric(name="d2", python_code="def f(x): return x\n", is_aggregated=False)
    contexts = [{"x": 1}, {"x": 2}, {"x": 3}]
    sandbox_results = [(1.0, None), (2.0, None), (3.0, None)]

    with patch("tasks.evaluate_in_sandbox", return_value=sandbox_results) as sandbox_mock, \
         patch("tasks.evaluate_dynamic_metric") as exec_mock:
        results = tasks._eval_metric_batch(gm, contexts, '{"x": "x"}')

    exec_mock.assert_not_called()
    sandbox_mock.assert_called_once()
    args = sandbox_mock.call_args.args
    # First arg is the metric, second is the contexts list (passed wholesale,
    # NOT one-at-a-time — the whole point of the batched path).
    assert args[0] is gm
    assert args[1] == contexts
    assert results == sandbox_results


def test_flag_only_active_when_value_is_exactly_one(monkeypatch):
    """The check is `os.environ.get(...) == '1'`; any other truthy-looking
    value (e.g. 'true', 'yes') should NOT enable the sandbox. Pin the
    contract so a misread env never silently flips the backend."""
    gm = GlobalMetric(name="d3", python_code="def f(): return 1\n", is_aggregated=False)

    for raw in ['true', 'yes', '0', '', 'on']:
        monkeypatch.setenv("BENCHHUB_SANDBOX_METRICS", raw)
        with patch("tasks.evaluate_in_sandbox") as sandbox_mock, \
             patch("tasks.evaluate_dynamic_metric", return_value=(0.0, None)):
            tasks._eval_metric_batch(gm, [{}], "{}")
        sandbox_mock.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end through the process_submission Celery task (eager mode)
# ---------------------------------------------------------------------------


def test_e2e_default_path_runs_metric_in_process(submission_with_metric):
    """Sanity check: the existing in-process behavior keeps working —
    the metric value matches what evaluate_dynamic_metric would produce."""
    from tasks import process_submission

    # No env flag set → in-process path.
    process_submission.delay(submission_with_metric.id)

    db.session.expire_all()
    sub = Submission.query.get(submission_with_metric.id)
    assert sub.processing_status == "Processed"

    result = MetricResult.query.filter_by(submission_id=sub.id).first()
    assert result is not None
    assert result.value == pytest.approx((1.0 + 2.0 + 4.0) / 3)


def test_e2e_with_flag_calls_sandbox_backend(submission_with_metric, monkeypatch):
    """End-to-end through process_submission with the flag on. We mock
    evaluate_in_sandbox to return predetermined per-context values; verify
    the task picked them up, computed the mean, and never touched the
    in-process exec path."""
    from tasks import process_submission

    monkeypatch.setenv("BENCHHUB_SANDBOX_METRICS", "1")

    sandbox_returns = [(10.0, None), (20.0, None), (30.0, None)]

    with patch("tasks.evaluate_in_sandbox", return_value=sandbox_returns) as sandbox_mock, \
         patch("tasks.evaluate_dynamic_metric") as exec_mock:
        process_submission.delay(submission_with_metric.id)

    sandbox_mock.assert_called_once()  # batched: ONE call for all 3 samples
    exec_mock.assert_not_called()

    db.session.expire_all()
    sub = Submission.query.get(submission_with_metric.id)
    assert sub.processing_status == "Processed"

    result = MetricResult.query.filter_by(submission_id=sub.id).first()
    assert result.value == pytest.approx((10.0 + 20.0 + 30.0) / 3)


def test_e2e_sandbox_fatal_marks_metric_with_error(submission_with_metric, monkeypatch):
    """When the sandbox returns the same fatal error for every context (e.g.
    image missing, container OOM), the per-sample loop sees no values and
    the MetricResult ends up with the error message rather than a number."""
    from tasks import process_submission

    monkeypatch.setenv("BENCHHUB_SANDBOX_METRICS", "1")

    fatal_returns = [(None, "sandbox timed out after 60s")] * 3

    with patch("tasks.evaluate_in_sandbox", return_value=fatal_returns):
        process_submission.delay(submission_with_metric.id)

    db.session.expire_all()
    result = MetricResult.query.filter_by(submission_id=submission_with_metric.id).first()
    assert result.value is None
    assert "timed out" in (result.error_message or "")
