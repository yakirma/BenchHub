"""Tests for `_compute_explorable_lb_ids` — the helper that decides
which LB cards get the green Explorable pill vs the yellow "No
samples" hourglass.

The rule: an LB is explorable iff
  (a) it has at least one attached Dataset with at least one Sample
  OR
  (b) it has at least one LB-scoped CustomField (sample_id IS NULL AND
      submission_id IS NULL, i.e. the HF-stub-mode marker rows written
      by `_persist_hf_eval_snapshots` / `populate_lb_samples`)."""
import pytest

from app import (
    db, _compute_explorable_lb_ids,
    Leaderboard, Dataset, Sample, CustomField, leaderboard_datasets,
)


@pytest.fixture
def empty_lb(db_session, logged_in_user):
    lb = Leaderboard(
        name='Empty LB', summary_metrics='',
        owner_user_id=logged_in_user.id, visibility='public',
    )
    db.session.add(lb); db.session.commit()
    return lb


@pytest.fixture
def bh_lb_with_sample(db_session, logged_in_user):
    ds = Dataset(
        name='ds',
        owner_user_id=logged_in_user.id, visibility='public',
    )
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s_000000'))
    lb = Leaderboard(
        name='BH LB', summary_metrics='',
        owner_user_id=logged_in_user.id, visibility='public',
    )
    db.session.add(lb); db.session.flush()
    db.session.execute(leaderboard_datasets.insert().values(
        leaderboard_id=lb.id, dataset_id=ds.id, role='primary',
    ))
    db.session.commit()
    return lb


@pytest.fixture
def hf_lb_with_marker(db_session, logged_in_user):
    """HF-stub-mode LB: no attached Dataset, but a LB-scoped CF row
    written by populate_lb_samples / _persist_hf_eval_snapshots."""
    lb = Leaderboard(
        name='HF LB', summary_metrics='',
        owner_user_id=logged_in_user.id, visibility='public',
    )
    db.session.add(lb); db.session.flush()
    db.session.add(CustomField(
        leaderboard_id=lb.id,
        sample_name='s_000000',
        name='label',
        data_type='scalar',
        value_float=1.0,
    ))
    db.session.commit()
    return lb


def test_empty_lb_is_not_explorable(empty_lb):
    assert _compute_explorable_lb_ids([empty_lb.id]) == set()


def test_bh_lb_with_samples_is_explorable(bh_lb_with_sample):
    assert _compute_explorable_lb_ids([bh_lb_with_sample.id]) == {bh_lb_with_sample.id}


def test_hf_lb_with_marker_cf_is_explorable(hf_lb_with_marker):
    assert _compute_explorable_lb_ids([hf_lb_with_marker.id]) == {hf_lb_with_marker.id}


def test_mixed_batch_returns_only_explorable_ones(
    empty_lb, bh_lb_with_sample, hf_lb_with_marker,
):
    """Pass three LBs in, get two back (the two with cached GT)."""
    out = _compute_explorable_lb_ids(
        [empty_lb.id, bh_lb_with_sample.id, hf_lb_with_marker.id]
    )
    assert out == {bh_lb_with_sample.id, hf_lb_with_marker.id}


def test_empty_input_returns_empty_set():
    """Don't run pointless joins when there's nothing to check."""
    assert _compute_explorable_lb_ids([]) == set()
    assert _compute_explorable_lb_ids(None) == set()


def test_sample_scoped_cf_does_not_count_as_marker(db_session, logged_in_user):
    """A regular per-sample CustomField (sample_id NOT NULL) belongs to
    a real BH Dataset Sample, not to the HF stub mode. The first half
    of the OR clause covers it. Make sure a sample_id-bearing CF on a
    LB without any Sample row does NOT light up the marker branch."""
    lb = Leaderboard(
        name='LB', summary_metrics='',
        owner_user_id=logged_in_user.id, visibility='public',
    )
    db.session.add(lb); db.session.commit()
    db.session.add(CustomField(
        leaderboard_id=lb.id,
        sample_id=999,  # NON-NULL → not a stub-mode marker
        name='label',
        data_type='scalar',
        value_float=1.0,
    ))
    db.session.commit()
    assert _compute_explorable_lb_ids([lb.id]) == set()


def test_submission_scoped_cf_does_not_count_as_marker(db_session, logged_in_user):
    """A submission's per-sample prediction is not GT; it shouldn't
    flip the explorability pill."""
    lb = Leaderboard(
        name='LB', summary_metrics='',
        owner_user_id=logged_in_user.id, visibility='public',
    )
    db.session.add(lb); db.session.commit()
    db.session.add(CustomField(
        leaderboard_id=lb.id,
        submission_id=999,  # NON-NULL → prediction, not GT
        sample_name='s_000000',
        name='label_pred',
        data_type='scalar',
        value_float=1.0,
    ))
    db.session.commit()
    assert _compute_explorable_lb_ids([lb.id]) == set()
