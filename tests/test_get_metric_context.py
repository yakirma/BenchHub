"""Tests for metric_engine.get_metric_context.

Builds the per-sample kwargs dict that's fed to user metric code. Pulls from:
  - GT (Sample): histogram → entropy, scalar CustomFields
  - Submission: scalar/metric CustomFields (filtered by sample_name)
  - Submission folder on disk: per-sample histogram .npz → sub_entropy_<folder>

Also exercises both paths of the dual-source `Sample.histogram_data` @property
(legacy `HistogramData` row vs. new `CustomField` fallback) — the CLAUDE.md
warns that the property shadows the relationship, so we cover both.
"""
import json
import math

import numpy as np
import pytest

from app import (
    CustomField,
    Dataset,
    HistogramData,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    Sample,
    Submission,
    db,
)
from metric_engine import get_metric_context


# ---------------------------------------------------------------------------
# Fixtures (reusable for Phase 3 Celery tests too)
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(db_session):
    ds = Dataset(name="ctx_ds")
    db_session.add(ds)
    db_session.flush()
    return ds


@pytest.fixture
def sample(db_session, dataset):
    s = Sample(dataset_id=dataset.id, name="s1")
    db_session.add(s)
    db_session.commit()
    return s


@pytest.fixture
def leaderboard(db_session, dataset):
    lb = Leaderboard(name="lb", summary_metrics="")
    lb.datasets.append(dataset)
    db_session.add(lb)
    db_session.commit()
    return lb


@pytest.fixture
def submission(db_session, leaderboard):
    sub = Submission(name="sub1", leaderboard_id=leaderboard.id)
    db_session.add(sub)
    db_session.commit()
    return sub


def _add_histogram_legacy(sample, bins, counts):
    """Use the legacy HistogramData table."""
    db.session.add(
        HistogramData(
            sample_id=sample.id,
            bins=json.dumps(bins),
            counts=json.dumps(counts),
        )
    )
    db.session.commit()


def _add_histogram_via_custom_field(sample, bins, counts):
    """Use the new CustomField fallback path."""
    db.session.add(
        CustomField(
            sample_id=sample.id,
            name="hist",
            field_type="histogram",
            value_text=json.dumps({"bins": bins, "counts": counts}),
        )
    )
    db.session.commit()


# ---------------------------------------------------------------------------
# Empty / minimal contexts
# ---------------------------------------------------------------------------


def test_empty_sample_yields_empty_context(sample):
    assert get_metric_context(sample) == {}


def test_no_submission_no_submission_folder_yields_only_gt(sample):
    db.session.add(
        CustomField(sample_id=sample.id, name="peak", field_type="scalar", value_float=12.5)
    )
    db.session.commit()

    ctx = get_metric_context(sample)
    assert ctx == {"gt_peak": 12.5, "peak": 12.5}


# ---------------------------------------------------------------------------
# GT entropy (both histogram source paths)
# ---------------------------------------------------------------------------


def _expected_entropy(counts):
    arr = np.array([c for c in counts if c > 0], dtype=float)
    if arr.sum() == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-np.sum(p * np.log2(p)))


def test_gt_entropy_via_legacy_histogram_table(sample):
    counts = [10, 20, 30, 40]
    _add_histogram_legacy(sample, bins=[0, 1, 2, 3], counts=counts)

    ctx = get_metric_context(sample)
    assert ctx["gt_entropy"] == pytest.approx(_expected_entropy(counts))


def test_gt_entropy_via_custom_field_fallback(sample):
    # CRITICAL: The Sample.histogram_data @property shadows the relationship and
    # falls back to a CustomField with name='hist'. CLAUDE.md flags this as a
    # bug-prone area — pin both paths.
    counts = [5, 5, 5, 5]
    _add_histogram_via_custom_field(sample, bins=[0, 1, 2, 3], counts=counts)

    ctx = get_metric_context(sample)
    assert ctx["gt_entropy"] == pytest.approx(_expected_entropy(counts))


def test_gt_entropy_zero_counts_returns_zero(sample):
    _add_histogram_legacy(sample, bins=[0, 1, 2], counts=[0, 0, 0])
    ctx = get_metric_context(sample)
    assert ctx["gt_entropy"] == 0.0


def test_gt_entropy_invalid_json_swallowed_to_zero(db_session, sample):
    # Bad JSON in counts → bare except catches → entropy = 0.0.
    db_session.add(
        HistogramData(sample_id=sample.id, bins="[]", counts="not-json")
    )
    db_session.commit()

    ctx = get_metric_context(sample)
    assert ctx["gt_entropy"] == 0.0


# ---------------------------------------------------------------------------
# GT custom fields
# ---------------------------------------------------------------------------


def test_gt_scalars_exposed_with_and_without_prefix(db_session, sample):
    db_session.add_all(
        [
            CustomField(sample_id=sample.id, name="peak", field_type="scalar", value_float=1.5),
            CustomField(sample_id=sample.id, name="snr", field_type="scalar", value_float=42.0),
        ]
    )
    db_session.commit()

    ctx = get_metric_context(sample)
    assert ctx["gt_peak"] == 1.5
    assert ctx["peak"] == 1.5
    assert ctx["gt_snr"] == 42.0
    assert ctx["snr"] == 42.0


def test_non_scalar_gt_fields_loaded_lazily(db_session, sample):
    """`image` and `depth` GT custom fields used to be excluded from
    the context. After Option B they're loaded lazily as numpy arrays
    (or None on failure) so structured-GT metrics like RMSE / PSNR
    can consume them. JSON / text GT fields stay excluded — they're
    not metric inputs."""
    db_session.add_all(
        [
            CustomField(sample_id=sample.id, name="thumbnail", field_type="image", value_text="path/x.png"),
            CustomField(sample_id=sample.id, name="meta", field_type="json", value_text="path/x.json"),
        ]
    )
    db_session.commit()

    ctx = get_metric_context(sample)
    # Image GT now appears in the context; the path doesn't exist on
    # disk so the loader returns None, but the key IS present so the
    # metric can detect "missing GT" via context.get().
    assert "thumbnail" in ctx and ctx["thumbnail"] is None
    assert "gt_thumbnail" in ctx and ctx["gt_thumbnail"] is None
    # JSON fields still not exposed (they're not metric inputs).
    assert "meta" not in ctx


# ---------------------------------------------------------------------------
# Submission custom fields
# ---------------------------------------------------------------------------


def test_submission_scalar_field_exposed(db_session, sample, submission):
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name="accuracy",
            field_type="scalar",
            value_float=0.91,
        )
    )
    db_session.commit()

    ctx = get_metric_context(sample, sub=submission)
    assert ctx["sub_accuracy"] == 0.91
    assert ctx["accuracy"] == 0.91


def test_submission_metric_field_exposed(db_session, sample, submission):
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name="l1",
            field_type="metric",
            value_float=0.05,
        )
    )
    db_session.commit()

    ctx = get_metric_context(sample, sub=submission)
    assert ctx["sub_l1"] == 0.05
    assert ctx["l1"] == 0.05


def test_submission_field_for_other_sample_ignored(db_session, dataset, submission):
    s1 = Sample(dataset_id=dataset.id, name="s1")
    s2 = Sample(dataset_id=dataset.id, name="s2")
    db_session.add_all([s1, s2])
    db_session.flush()

    # Field belongs to s2, but we're building context for s1.
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name="s2",
            name="accuracy",
            field_type="scalar",
            value_float=0.99,
        )
    )
    db_session.commit()

    ctx = get_metric_context(s1, sub=submission)
    assert "accuracy" not in ctx
    assert "sub_accuracy" not in ctx


def test_submission_non_scalar_metric_fields_skipped(db_session, sample, submission):
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name="viz",
            field_type="image",
            value_text="path.png",
        )
    )
    db_session.commit()

    ctx = get_metric_context(sample, sub=submission)
    assert "viz" not in ctx
    assert "sub_viz" not in ctx


# ---------------------------------------------------------------------------
# lm_{id} friendly-name aliasing
# ---------------------------------------------------------------------------


def test_lm_id_field_aliased_to_friendly_name(db_session, dataset, sample, leaderboard, submission):
    # Build a LeaderboardMetric whose target_name is "L1_Loss".
    gm = GlobalMetric(name="l1_global", python_code="def m(): return 1", is_aggregated=False)
    db_session.add(gm)
    db_session.flush()

    lm = LeaderboardMetric(
        leaderboard_id=leaderboard.id,
        global_metric_id=gm.id,
        arg_mappings="{}",
        target_name="L1_Loss",
    )
    db_session.add(lm)
    db_session.flush()

    # A persisted per-sample metric value uses the lm_{id} naming convention.
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name=f"lm_{lm.id}",
            field_type="metric",
            value_float=0.123,
        )
    )
    db_session.commit()

    ctx = get_metric_context(sample, sub=submission)
    # Both raw (lm_<id>) and friendly forms should be present.
    assert ctx[f"lm_{lm.id}"] == 0.123
    assert ctx[f"sub_lm_{lm.id}"] == 0.123
    assert ctx["L1_Loss"] == 0.123
    assert ctx["sub_L1_Loss"] == 0.123


def test_lm_id_field_falls_back_to_global_metric_name_without_target_name(
    db_session, dataset, sample, leaderboard, submission
):
    gm = GlobalMetric(name="bare_metric", python_code="def m(): return 1", is_aggregated=False)
    db_session.add(gm)
    db_session.flush()

    lm = LeaderboardMetric(
        leaderboard_id=leaderboard.id,
        global_metric_id=gm.id,
        arg_mappings="{}",
        target_name=None,
    )
    db_session.add(lm)
    db_session.flush()

    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name=f"lm_{lm.id}",
            field_type="metric",
            value_float=7.0,
        )
    )
    db_session.commit()

    ctx = get_metric_context(sample, sub=submission)
    assert ctx["bare_metric"] == 7.0
    assert ctx["sub_bare_metric"] == 7.0


def test_lm_id_with_unknown_id_silently_ignored(db_session, sample, submission):
    # lm_99999 has no matching LeaderboardMetric → exception path swallows.
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name="lm_99999",
            field_type="metric",
            value_float=1.0,
        )
    )
    db_session.commit()

    ctx = get_metric_context(sample, sub=submission)
    # Raw lm_<id> still present; friendly name is not (no matching LM).
    assert ctx["lm_99999"] == 1.0


# ---------------------------------------------------------------------------
# Submission folder histogram entropy
# ---------------------------------------------------------------------------


def test_submission_folder_hist_entropy_computed(tmp_path, sample, submission):
    folder_name = "hist_filtered"
    folder = tmp_path / folder_name
    folder.mkdir()
    counts = np.array([1, 2, 3, 4])
    np.savez(folder / f"{sample.name}.npz", bins=np.array([0, 1, 2, 3]), counts=counts)

    ctx = get_metric_context(sample, sub=submission, submission_folder=str(tmp_path))
    assert ctx[f"sub_entropy_{folder_name}"] == pytest.approx(_expected_entropy(counts.tolist()))


def test_submission_folder_raw_histogram_alias_supported(tmp_path, sample, submission):
    folder = tmp_path / "raw_histogram"
    folder.mkdir()
    np.savez(folder / f"{sample.name}.npz", bins=np.array([0, 1]), counts=np.array([5, 5]))

    ctx = get_metric_context(sample, sub=submission, submission_folder=str(tmp_path))
    assert "sub_entropy_raw_histogram" in ctx
    assert ctx["sub_entropy_raw_histogram"] == pytest.approx(1.0)  # log2(2) = 1 for 50/50


def test_submission_folder_zero_counts_returns_zero_entropy(tmp_path, sample, submission):
    folder = tmp_path / "hist_zero"
    folder.mkdir()
    np.savez(folder / f"{sample.name}.npz", bins=np.array([0, 1]), counts=np.array([0, 0]))

    ctx = get_metric_context(sample, sub=submission, submission_folder=str(tmp_path))
    assert ctx["sub_entropy_hist_zero"] == 0.0


def test_submission_folder_non_hist_folders_ignored(tmp_path, sample, submission):
    # Folder doesn't start with hist_ and isn't raw_histogram → not scanned.
    folder = tmp_path / "metric_acc"
    folder.mkdir()
    np.savez(folder / f"{sample.name}.npz", bins=np.array([0]), counts=np.array([1]))

    ctx = get_metric_context(sample, sub=submission, submission_folder=str(tmp_path))
    assert all(not k.startswith("sub_entropy_") for k in ctx)


def test_submission_folder_missing_sample_npz_skipped(tmp_path, dataset, submission):
    # Two samples, but only s_other has a .npz in the folder.
    s_target = Sample(dataset_id=dataset.id, name="s_target")
    db.session.add(s_target)
    db.session.commit()

    folder = tmp_path / "hist_a"
    folder.mkdir()
    np.savez(folder / "s_other.npz", bins=np.array([0]), counts=np.array([1]))

    ctx = get_metric_context(s_target, sub=submission, submission_folder=str(tmp_path))
    assert "sub_entropy_hist_a" not in ctx


def test_submission_folder_does_not_exist_does_not_raise(sample, submission):
    # Defensive: function wraps the os.listdir call; missing dir is logged, not raised.
    ctx = get_metric_context(
        sample, sub=submission, submission_folder="/tmp/definitely-not-a-real-path-12345"
    )
    assert isinstance(ctx, dict)


# ---------------------------------------------------------------------------
# Combined integration: GT + sub + folder
# ---------------------------------------------------------------------------


def test_full_context_combination(tmp_path, db_session, sample, submission):
    # GT histogram (legacy) + GT scalar + sub scalar + sub-folder histogram.
    _add_histogram_legacy(sample, bins=[0, 1, 2], counts=[1, 2, 3])
    db_session.add(
        CustomField(sample_id=sample.id, name="peak", field_type="scalar", value_float=1.0)
    )
    db_session.add(
        CustomField(
            submission_id=submission.id,
            sample_name=sample.name,
            name="pred_peak",
            field_type="scalar",
            value_float=1.1,
        )
    )
    db_session.commit()

    folder = tmp_path / "hist_pred"
    folder.mkdir()
    np.savez(folder / f"{sample.name}.npz", bins=np.array([0, 1]), counts=np.array([3, 1]))

    ctx = get_metric_context(sample, sub=submission, submission_folder=str(tmp_path))

    assert "gt_entropy" in ctx
    assert ctx["gt_peak"] == 1.0
    assert ctx["peak"] == 1.0
    assert ctx["sub_pred_peak"] == 1.1
    assert ctx["pred_peak"] == 1.1
    assert "sub_entropy_hist_pred" in ctx
    assert math.isfinite(ctx["sub_entropy_hist_pred"])
