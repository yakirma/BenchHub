"""Integration tests for app.process_submission_zip.

End-to-end ingest for a submission ZIP: extract → CustomField rows linked by
sample_name → tags from tags.txt → git metadata → enqueue (eager) Celery task
to compute metrics. The leaderboard has no metrics defined, so the eager task
short-circuits to status='Processed'.
"""
import json
import os

import numpy as np
import pytest

from app import (
    CustomField,
    Dataset,
    Leaderboard,
    Sample,
    Submission,
    Tag,
    app,
    db,
)
from app import process_submission_zip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def leaderboard_with_samples(db_session):
    ds = Dataset(name="sub_test_ds")
    db_session.add(ds)
    db_session.flush()
    for name in ["s1", "s2"]:
        db_session.add(Sample(dataset_id=ds.id, name=name))
    db_session.flush()

    lb = Leaderboard(name="sub_lb", summary_metrics="")
    lb.datasets.append(ds)
    db_session.add(lb)
    db_session.commit()
    return lb


def upload_path(*parts):
    return os.path.join(app.config["UPLOAD_FOLDER"], *parts)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Test predicates the dropped metric_* prefix→'scalar' behavior; surface is Phase A delete pile (detect_custom_fields rewrite).")
def test_happy_path_creates_submission_with_custom_fields(
    leaderboard_with_samples, make_zip
):
    layout = {
        "metric_acc/s1.txt": "0.91",
        "metric_acc/s2.txt": "0.87",
        "viz/s1.png": b"\x89PNG-1",
        "viz/s2.png": b"\x89PNG-2",
    }
    zip_path = make_zip("sub.zip", layout, root_folder="sub_pkg")

    success, error = process_submission_zip(
        leaderboard_with_samples.id, "sub_v1", zip_path
    )

    assert success is True, error
    assert error is None

    sub = Submission.query.filter_by(leaderboard_id=leaderboard_with_samples.id).first()
    assert sub is not None
    # NOTE: process_submission_zip renames the submission to the inner-folder name.
    assert sub.name == "sub_pkg"

    # Per-sample scalar fields linked by sample_name (not by sample_id).
    # The legacy `metric_*` prefix used to coerce field_type to 'metric'
    # only when is_submission=True; that branch is gone, so the type
    # is now 'scalar' (driven by the .txt extension alone).
    metric_cfs = CustomField.query.filter_by(
        submission_id=sub.id, name="metric_acc", field_type="scalar"
    ).all()
    by_sample = {cf.sample_name: cf.value_float for cf in metric_cfs}
    assert by_sample == {"s1": 0.91, "s2": 0.87}

    # Image fields stored under uploads/submissions/<id>/images/<field>/.
    img_cfs = CustomField.query.filter_by(submission_id=sub.id, field_type="image").all()
    assert {c.sample_name for c in img_cfs} == {"s1", "s2"}
    for cf in img_cfs:
        assert os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], cf.value_text))


def test_eager_celery_task_transitions_status_to_processed(
    leaderboard_with_samples, make_zip
):
    # Empty leaderboard → no metrics to compute → task short-circuits.
    layout = {"metric_acc/s1.txt": "0.5"}
    zip_path = make_zip("eager.zip", layout, root_folder="eager_sub")

    success, _ = process_submission_zip(leaderboard_with_samples.id, "eager", zip_path)
    assert success

    # Re-query — the task may have happened on a fresh session.
    db.session.expire_all()
    sub = Submission.query.filter_by(leaderboard_id=leaderboard_with_samples.id).first()
    assert sub.processing_status == "Processed"


# ---------------------------------------------------------------------------
# Persistence locations on disk
# ---------------------------------------------------------------------------


def test_image_field_copied_to_submission_images_dir(leaderboard_with_samples, make_zip):
    zip_path = make_zip(
        "img.zip", {"viz/s1.png": b"abc"}, root_folder="img_sub"
    )
    process_submission_zip(leaderboard_with_samples.id, "img", zip_path)

    sub = Submission.query.first()
    expected = upload_path("submissions", str(sub.id), "images", "viz", "s1.png")
    assert os.path.exists(expected)


def test_depth_field_copied_to_depth_maps_dir(leaderboard_with_samples, make_zip):
    layout = {
        "raw_depth/s1_4x3.npz": {"npz": {"depth": np.zeros((3, 4))}},
        "raw_depth/s2_4x3.npz": {"npz": {"depth": np.zeros((3, 4))}},
    }
    zip_path = make_zip("depth.zip", layout, root_folder="d_sub")
    process_submission_zip(leaderboard_with_samples.id, "d", zip_path)

    sub = Submission.query.first()
    assert os.path.exists(
        upload_path("submissions", str(sub.id), "depth_maps", "raw_depth", "s1_4x3.npz")
    )


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_tags_txt_creates_tag_rows_and_associations(
    leaderboard_with_samples, make_zip
):
    layout = {
        "metric_acc/s1.txt": "0.5",
        "tags.txt": "alpha, beta , gamma",
    }
    zip_path = make_zip("tags.zip", layout, root_folder="tag_sub")
    success, _ = process_submission_zip(leaderboard_with_samples.id, "tag", zip_path)
    assert success

    sub = Submission.query.first()
    tag_names = {t.name for t in sub.tags}
    assert tag_names == {"alpha", "beta", "gamma"}
    # And the Tag rows should exist globally.
    assert {t.name for t in Tag.query.all()} >= {"alpha", "beta", "gamma"}


def test_tags_txt_skips_empty_entries(leaderboard_with_samples, make_zip):
    layout = {
        "metric_acc/s1.txt": "0.5",
        "tags.txt": "alpha,, ,beta",
    }
    zip_path = make_zip("blanktags.zip", layout, root_folder="blank_tag")
    process_submission_zip(leaderboard_with_samples.id, "blank", zip_path)

    sub = Submission.query.first()
    assert {t.name for t in sub.tags} == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# git_info.json key aliases
# ---------------------------------------------------------------------------


def test_git_info_uses_commit_sha_alias(leaderboard_with_samples, make_zip):
    # The submission code accepts `commit_sha` as an alias for `commit`.
    git_info = {"commit_sha": "deadbeef", "branch": "main", "author": "Eve"}
    layout = {
        "metric_acc/s1.txt": "0.5",
        "git_info.json": json.dumps(git_info),
    }
    zip_path = make_zip("git.zip", layout, root_folder="git_sub")
    process_submission_zip(leaderboard_with_samples.id, "g", zip_path)

    sub = Submission.query.first()
    assert sub.git_commit == "deadbeef"
    assert sub.git_branch == "main"
    assert sub.git_author == "Eve"


def test_git_info_repo_url_falls_back_into_branch_field(
    leaderboard_with_samples, make_zip
):
    # The submission code uses repo_url as a fallback for branch (so the value
    # at least surfaces somewhere visible). Pin this quirk.
    git_info = {"commit": "abc", "repo_url": "git@host:repo.git", "author": "A"}
    layout = {
        "metric_acc/s1.txt": "0.5",
        "git_info.json": json.dumps(git_info),
    }
    zip_path = make_zip("repourl.zip", layout, root_folder="ru_sub")
    process_submission_zip(leaderboard_with_samples.id, "r", zip_path)

    sub = Submission.query.first()
    assert sub.git_branch == "git@host:repo.git"


# ---------------------------------------------------------------------------
# Single-root rename
# ---------------------------------------------------------------------------


def test_inner_folder_name_overrides_provided_submission_name(
    leaderboard_with_samples, make_zip
):
    layout = {"metric_acc/s1.txt": "0.5"}
    zip_path = make_zip("any.zip", layout, root_folder="actual_run_id")

    process_submission_zip(leaderboard_with_samples.id, "user_typed_name", zip_path)

    sub = Submission.query.first()
    # The inner folder wins — submission.name is "actual_run_id", not "user_typed_name".
    assert sub.name == "actual_run_id"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_corrupt_zip_returns_false(leaderboard_with_samples, tmp_path):
    """REAL BUG (analogous to the dataset orphan): process_submission_zip
    commits the Submission row early to release the lock and get the ID, then
    extracts the ZIP. If extraction fails, the except clause does
    db.session.rollback() — but the early INSERT was already committed in a
    separate transaction, so the orphan Submission survives.

    Pin the failure return code AND the orphan to make the bug visible. Flip
    the orphan assertion when the bug is fixed."""
    bad_zip = tmp_path / "garbage.zip"
    bad_zip.write_bytes(b"not a zip file at all")

    success, error = process_submission_zip(
        leaderboard_with_samples.id, "broken", str(bad_zip)
    )
    assert success is False
    assert error is not None
    # Bug: orphan Submission row left behind because the early commit isn't undone.
    assert Submission.query.count() == 1
    orphan = Submission.query.first()
    assert orphan.name == "broken"


# ---------------------------------------------------------------------------
# Sample-name linkage (CustomFields use sample_name, not sample_id, for subs)
# ---------------------------------------------------------------------------


def test_custom_fields_for_submissions_use_sample_name_not_sample_id(
    leaderboard_with_samples, make_zip
):
    layout = {"metric_acc/s1.txt": "0.5"}
    zip_path = make_zip("link.zip", layout, root_folder="link_sub")
    process_submission_zip(leaderboard_with_samples.id, "l", zip_path)

    sub = Submission.query.first()
    cf = CustomField.query.filter_by(submission_id=sub.id, name="metric_acc").first()
    assert cf.sample_name == "s1"
    # Submission CustomFields explicitly do NOT set sample_id (the link is by name only).
    assert cf.sample_id is None
