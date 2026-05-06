"""Integration tests for app.process_dataset_zip.

The full ingest pipeline: ZIP → temp extract → Dataset/Sample/CustomField rows
+ files copied to uploads/datasets/<name>/.... High-bug-yield because the
file-on-disk semantics drive most user-visible bugs.
"""
import json
import os

import numpy as np
import pytest

from app import CustomField, Dataset, Sample, app, db
from app import process_dataset_zip


def upload_path(*parts):
    return os.path.join(app.config["UPLOAD_FOLDER"], *parts)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_creates_dataset_samples_and_custom_fields(client, make_zip):
    layout = {
        "config/s1.json": '{"key": 1}',
        "config/s2.json": '{"key": 2}',
        "metric_acc/s1.txt": "0.9",  # only-submission prefix; here scanned as scalar
        "metric_acc/s2.txt": "0.8",
        "thumbnails/s1.png": b"\x89PNG-fake-1",
        "thumbnails/s2.png": b"\x89PNG-fake-2",
    }
    zip_path = make_zip("happy.zip", layout)

    success, message, ds_id = process_dataset_zip(zip_path, "happy_ds")

    assert success is True, message
    assert ds_id is not None
    assert "2 samples" in message

    ds = Dataset.query.get(ds_id)
    assert ds.name == "happy_ds"

    samples = {s.name: s for s in Sample.query.filter_by(dataset_id=ds_id).all()}
    assert set(samples) == {"s1", "s2"}

    fields_by_name = {(cf.name, cf.sample_id) for cf in CustomField.query.all()}
    assert ("config", samples["s1"].id) in fields_by_name
    assert ("metric_acc", samples["s1"].id) in fields_by_name
    assert ("thumbnails", samples["s1"].id) in fields_by_name


def test_raw_depth_does_not_create_phantom_samples(client, make_zip):
    """Regression: `raw_<field>/<sample>_<W>x<H>.npz` files used to land
    sample-name=`s0_640x480` because os.path.splitext only strips the
    extension. The discovery code now strips the dimension suffix when
    the folder is a raw_ depth folder."""
    layout = {
        "image_rgb/s0.png": b"\x89PNG-img",
        "image_rgb/s1.png": b"\x89PNG-img",
        "raw_depth/s0_8x6.npz": {"npz": {"depth": np.zeros((6, 8))}},
        "raw_depth/s1_8x6.npz": {"npz": {"depth": np.zeros((6, 8))}},
    }
    zip_path = make_zip("nodupes.zip", layout, root_folder="nodupes_ds")
    success, msg, ds_id = process_dataset_zip(zip_path, "nodupes_ds")
    assert success
    assert "(2 samples)" in msg, msg
    sample_names = {s.name for s in Sample.query.filter_by(dataset_id=ds_id).all()}
    assert sample_names == {"s0", "s1"}


def test_persists_files_to_uploads_directory(client, make_zip):
    layout = {
        "thumbnails/s1.png": b"\x89PNG-img",
        "raw_depth/s1_8x6.npz": {"npz": {"depth": np.zeros((6, 8))}},
        "config/s1.json": '{"a":1}',
    }
    zip_path = make_zip("files.zip", layout)
    success, _, ds_id = process_dataset_zip(zip_path, "files_ds")
    assert success

    # Original ZIP archived alongside extracted content
    assert os.path.exists(upload_path("datasets", "files_ds", "files_ds.zip"))
    # Image goes under datasets/<name>/images/<field>/
    assert os.path.exists(upload_path("datasets", "files_ds", "images", "thumbnails", "s1.png"))
    # Depth under datasets/<name>/depth_maps/<field>/
    assert os.path.exists(upload_path("datasets", "files_ds", "depth_maps", "raw_depth", "s1_8x6.npz"))
    # JSON preserved under datasets/<name>/<field>/
    assert os.path.exists(upload_path("datasets", "files_ds", "config", "s1.json"))


def test_image_custom_field_value_text_is_relative_path(client, make_zip):
    # NOTE: Use root_folder so the inner "viz/" isn't treated as the dataset wrapper
    # by the single-top-level-folder detection logic.
    zip_path = make_zip("img.zip", {"viz/s1.png": b"img"}, root_folder="img_pkg")
    success, _, ds_id = process_dataset_zip(zip_path, "img_ds")
    assert success

    cf = CustomField.query.filter_by(name="viz", field_type="image").first()
    assert cf is not None
    # Stored as relative path from UPLOAD_FOLDER, so prefixing UPLOAD_FOLDER reaches the file.
    assert os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], cf.value_text))


# ---------------------------------------------------------------------------
# Histogram NPZ → JSON conversion
# ---------------------------------------------------------------------------


def test_histogram_npz_loaded_into_value_text_json(client, make_zip):
    bins = [0.0, 0.5, 1.0, 1.5]
    counts = [10, 20, 30, 40]
    layout = {
        "hist/s1.npz": {"npz": {"bins": np.array(bins), "counts": np.array(counts)}},
    }
    zip_path = make_zip("hist.zip", layout, root_folder="hist_pkg")

    success, _, ds_id = process_dataset_zip(zip_path, "hist_ds")
    assert success

    cf = CustomField.query.filter_by(name="hist", field_type="histogram").first()
    assert cf is not None
    parsed = json.loads(cf.value_text)
    assert parsed["bins"] == bins
    assert parsed["counts"] == counts


# ---------------------------------------------------------------------------
# Single-root-folder rename
# ---------------------------------------------------------------------------


def test_single_root_folder_in_zip_renames_dataset(client, make_zip):
    # ZIP has a single top-level folder "wrapped_name/" containing the data.
    layout = {"config/s1.json": '{"x":1}'}
    zip_path = make_zip("wrapped.zip", layout, root_folder="wrapped_name")

    success, message, ds_id = process_dataset_zip(zip_path, "originally_called_this")
    assert success, message

    ds = Dataset.query.get(ds_id)
    # Rename happens because the inner folder is the "real" dataset.
    assert ds.name == "wrapped_name"


# ---------------------------------------------------------------------------
# Override / collision behavior
# ---------------------------------------------------------------------------


def test_collision_without_override_returns_false(client, make_zip):
    layout = {"config/s1.json": '{"a":1}', "tags/s1.txt": "foo"}
    zip_path = make_zip("first.zip", layout, root_folder="dup_ds")
    success1, _, _ = process_dataset_zip(zip_path, "dup_ds")
    assert success1

    zip_path2 = make_zip("second.zip", layout, root_folder="dup_ds")
    success, message, ds_id = process_dataset_zip(zip_path2, "dup_ds")

    assert success is False
    assert ds_id is None
    assert "already exists" in message


def test_override_replaces_existing_dataset(client, make_zip):
    # Seed an existing dataset with one sample.
    zip_path1 = make_zip(
        "v1.zip",
        {"config/s1.json": '{"v":1}', "tags/s1.txt": "x"},
        root_folder="ver_ds",
    )
    success1, _, ds_id1 = process_dataset_zip(zip_path1, "ver_ds")
    assert success1

    # Re-upload same name with different content + override=True.
    zip_path2 = make_zip(
        "v2.zip",
        {
            "config/sample_a.json": '{"v":2}',
            "config/sample_b.json": '{"v":3}',
            "tags/sample_a.txt": "x",
            "tags/sample_b.txt": "x",
        },
        root_folder="ver_ds",
    )
    success2, _, ds_id2 = process_dataset_zip(zip_path2, "ver_ds", override=True)
    assert success2

    # Old samples are gone; the new ones replaced them.
    # (Don't compare ds_id1 vs ds_id2 — SQLite recycles deleted INTEGER PKs.)
    assert Dataset.query.filter_by(name="ver_ds").count() == 1
    new_ds = Dataset.query.filter_by(name="ver_ds").first()
    sample_names = {s.name for s in Sample.query.filter_by(dataset_id=new_ds.id)}
    assert sample_names == {"sample_a", "sample_b"}
    # And the previously-existing s1 sample is gone.
    assert Sample.query.filter_by(name="s1").count() == 0


# ---------------------------------------------------------------------------
# Empty / malformed
# ---------------------------------------------------------------------------


def test_empty_zip_returns_false(client, make_zip):
    zip_path = make_zip("empty.zip", {"git_info.json": '{"commit":"abc"}'})
    success, message, ds_id = process_dataset_zip(zip_path, "empty_ds")
    assert success is False
    assert ds_id is None
    assert "No valid samples" in message


def test_failure_path_leaves_orphan_dataset_row(client, make_zip):
    """REAL BUG: process_dataset_zip commits a Dataset row early (to release
    the lock and get the ID). When sample discovery later fails ("No valid
    samples..."), the function returns False but does NOT delete the orphan
    row. Subsequent uploads with the same name will hit a "collision" error
    against this ghost dataset.

    Pin this so a future fix (e.g. wrapping in try/except with rollback) is
    a positive — flip the assertion when fixed."""
    # Single-folder layout that triggers the bug: the inner folder gets treated
    # as the "wrapper" and dataset_content_path ends up at a leaf with no subdirs.
    zip_path = make_zip("orphan.zip", {"config/s1.json": '{"x":1}'})
    success, message, ds_id = process_dataset_zip(zip_path, "should_fail")
    assert success is False
    assert "No valid samples" in message

    # Bug: the function still leaves a Dataset(name="config") row behind.
    orphan = Dataset.query.filter_by(name="config").first()
    assert orphan is not None  # remove this line when the bug is fixed


# ---------------------------------------------------------------------------
# git_info.json parsing
# ---------------------------------------------------------------------------


def test_git_info_json_populates_dataset_metadata(client, make_zip):
    git_info = {
        "commit": "abc1234",
        "branch": "feat/bench",
        "message": "Add benchmark v2",
        "author": "Alice",
    }
    layout = {
        "config/s1.json": '{"k":1}',
        "git_info.json": json.dumps(git_info),
    }
    zip_path = make_zip("git.zip", layout)

    success, _, ds_id = process_dataset_zip(zip_path, "git_ds")
    assert success

    ds = Dataset.query.get(ds_id)
    assert ds.git_commit == "abc1234"
    assert ds.git_branch == "feat/bench"
    assert ds.git_message == "Add benchmark v2"
    assert ds.git_author == "Alice"


def test_git_info_alternate_filename_supported(client, make_zip):
    # Legacy: `git.info` is also accepted.
    git_info = {"commit": "deadbeef", "author": "Bob"}
    zip_path = make_zip(
        "alt.zip",
        {"config/s1.json": '{"k":1}', "git.info": json.dumps(git_info)},
    )

    success, _, ds_id = process_dataset_zip(zip_path, "alt_ds")
    assert success

    ds = Dataset.query.get(ds_id)
    assert ds.git_commit == "deadbeef"
    assert ds.git_author == "Bob"


# ---------------------------------------------------------------------------
# tags/ folder special case (text field also populates Sample.tags)
# ---------------------------------------------------------------------------


def test_tags_folder_populates_sample_tags_column(client, make_zip):
    layout = {
        "config/s1.json": '{"k":1}',
        "tags/s1.txt": "alpha,beta,gamma",
    }
    zip_path = make_zip("tags.zip", layout)
    success, _, ds_id = process_dataset_zip(zip_path, "tags_ds")
    assert success

    sample = Sample.query.filter_by(dataset_id=ds_id, name="s1").first()
    # The function normalizes newlines to commas; assert the tag string is set.
    assert sample.tags == "alpha,beta,gamma"

    # The CustomField row also exists with field_type='text'.
    cf = CustomField.query.filter_by(sample_id=sample.id, name="tags", field_type="text").first()
    assert cf is not None


def test_tags_folder_normalizes_newlines_to_commas(client, make_zip):
    zip_path = make_zip(
        "tagn.zip",
        {"config/s1.json": '{"k":1}', "tags/s1.txt": "alpha\nbeta\r\ngamma"},
    )
    success, _, ds_id = process_dataset_zip(zip_path, "tagn_ds")
    assert success

    sample = Sample.query.filter_by(dataset_id=ds_id, name="s1").first()
    # Newlines + carriage returns become a single comma run.
    assert "alpha" in sample.tags and "beta" in sample.tags and "gamma" in sample.tags
    assert "\n" not in sample.tags
