"""Tests for `process_submissions_batch_sequential`.

Commit 8a77b48 explicitly rolled back concurrent batch processing in favor of
sequential — pin that contract so a future "let's parallelize" change has to
delete this test on purpose.
"""
import json
from unittest.mock import patch

import pytest

from app import (
    Dataset,
    Leaderboard,
    Sample,
    Submission,
    db,
)
from tasks import process_submissions_batch_sequential


@pytest.fixture
def lb_with_subs(client):
    ds = Dataset(name="batch_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name="s1"))
    db.session.flush()

    lb = Leaderboard(name="batch_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()

    sub_ids = []
    for n in ["one", "two", "three"]:
        sub = Submission(name=n, leaderboard_id=lb.id)
        db.session.add(sub)
        db.session.flush()
        sub_ids.append(sub.id)

    db.session.commit()
    return {"lb": lb, "sub_ids": sub_ids}


def test_batch_processes_submissions_in_input_order(lb_with_subs):
    """Order of `submission_ids` must be preserved — no reshuffling."""
    sub_ids = lb_with_subs["sub_ids"]

    with patch("tasks._process_submission_impl") as impl_mock:
        process_submissions_batch_sequential.delay(sub_ids)

    actual_order = [call.args[0] for call in impl_mock.call_args_list]
    assert actual_order == sub_ids


def test_batch_processes_in_reverse_when_caller_supplied_reversed(lb_with_subs):
    sub_ids = list(reversed(lb_with_subs["sub_ids"]))

    with patch("tasks._process_submission_impl") as impl_mock:
        process_submissions_batch_sequential.delay(sub_ids)

    actual_order = [call.args[0] for call in impl_mock.call_args_list]
    assert actual_order == sub_ids


def test_batch_forwards_sample_filters_to_each_call(lb_with_subs):
    filters = {"include": {"enabled": True, "tags": ["x"]}}

    with patch("tasks._process_submission_impl") as impl_mock:
        process_submissions_batch_sequential.delay(
            lb_with_subs["sub_ids"], sample_filters=filters
        )

    for call in impl_mock.call_args_list:
        # _process_submission_impl(submission_id, sample_filters, task_instance=None)
        assert call.args[1] == filters


def test_batch_calls_impl_serially_not_via_delay(lb_with_subs):
    """The internal task calls `_process_submission_impl` directly (not
    `process_submission.delay`). That's what enforces "one at a time" within
    a single Celery worker — pin it."""
    with patch("tasks._process_submission_impl") as impl_mock, \
         patch("tasks.process_submission.delay") as delay_mock:
        process_submissions_batch_sequential.delay(lb_with_subs["sub_ids"])

    # Direct calls only; no fan-out via .delay.
    assert impl_mock.call_count == len(lb_with_subs["sub_ids"])
    delay_mock.assert_not_called()


def test_batch_end_to_end_marks_every_submission_processed(lb_with_subs):
    """End-to-end: with no metrics on the leaderboard, each submission should
    transition to status='Processed' after the batch task completes."""
    process_submissions_batch_sequential.delay(lb_with_subs["sub_ids"])

    db.session.expire_all()
    statuses = {
        sub.id: sub.processing_status
        for sub in Submission.query.filter(Submission.id.in_(lb_with_subs["sub_ids"])).all()
    }
    assert all(s == "Processed" for s in statuses.values()), statuses


def test_batch_with_empty_id_list_is_noop(lb_with_subs):
    with patch("tasks._process_submission_impl") as impl_mock:
        process_submissions_batch_sequential.delay([])

    impl_mock.assert_not_called()
