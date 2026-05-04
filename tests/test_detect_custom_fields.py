"""Tests for app.detect_custom_fields.

Pure function (no DB) that classifies folder/file conventions inside dataset
and submission ZIPs into typed CustomField records. The folder-prefix +
extension precedence is subtle, so tests pin the actual current behavior —
including a couple of quirks called out inline.
"""
import os

import numpy as np
import pytest

from app import detect_custom_fields


def _write(folder, **files):
    """Create files inside `folder`. Values are strings (text) or bytes."""
    os.makedirs(folder, exist_ok=True)
    for name, content in files.items():
        path = os.path.join(folder, name)
        if isinstance(content, str):
            with open(path, "w") as f:
                f.write(content)
        else:
            with open(path, "wb") as f:
                f.write(content)


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


def test_returns_empty_when_base_path_missing(tmp_path):
    out = detect_custom_fields(str(tmp_path / "does_not_exist"), ["s1"], set())
    assert out == {}


def test_returns_empty_when_no_folders(tmp_path):
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert out == {}


def test_known_folders_excluded(tmp_path):
    _write(tmp_path / "skipme", **{"s1.txt": "1.0"})
    _write(tmp_path / "keepme", **{"s1.txt": "2.0"})
    out = detect_custom_fields(str(tmp_path), ["s1"], known_folders={"skipme"})
    assert "skipme" not in out
    assert "keepme" in out


def test_folder_with_no_matching_samples_is_dropped(tmp_path):
    # Folder has files but none match sample names → field_type stays None → folder skipped.
    _write(tmp_path / "stranger", **{"unrelated.txt": "1"})
    out = detect_custom_fields(str(tmp_path), ["s1", "s2"], set())
    assert out == {}


# ---------------------------------------------------------------------------
# Per-type detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
def test_image_extension_detected(tmp_path, ext):
    folder = tmp_path / "viz"
    _write(folder, **{f"s1{ext}": b"\x89PNG-fake"})
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert out["viz"]["type"] == "image"
    assert out["viz"]["data"]["s1"].endswith(f"s1{ext}")


def test_scalar_txt_parsed_as_float(tmp_path):
    _write(tmp_path / "accuracy", **{"s1.txt": "0.87", "s2.txt": "1.5"})
    out = detect_custom_fields(str(tmp_path), ["s1", "s2"], set())
    assert out["accuracy"]["type"] == "scalar"
    assert out["accuracy"]["data"] == {"s1": 0.87, "s2": 1.5}


def test_non_floatable_txt_classified_as_text(tmp_path):
    _write(tmp_path / "labels", **{"s1.txt": "good", "s2.txt": "bad"})
    out = detect_custom_fields(str(tmp_path), ["s1", "s2"], set())
    assert out["labels"]["type"] == "text"
    assert out["labels"]["data"] == {"s1": "good", "s2": "bad"}


def test_json_files_detected(tmp_path):
    _write(tmp_path / "config", **{"s1.json": '{"k": 1}'})
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert out["config"]["type"] == "json"
    assert out["config"]["data"]["s1"].endswith("s1.json")


@pytest.mark.parametrize("folder_name", ["hist", "hist_filtered", "raw_histogram"])
def test_histogram_npz_detected(tmp_path, folder_name):
    folder = tmp_path / folder_name
    os.makedirs(folder, exist_ok=True)
    np.savez(folder / "s1.npz", bins=np.array([0, 1, 2]), counts=np.array([10, 20, 30]))
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert out[folder_name]["type"] == "histogram"
    assert out[folder_name]["data"]["s1"].endswith("s1.npz")


def test_depth_npz_detected_with_dimensions_in_filename(tmp_path):
    folder = tmp_path / "raw_depth"
    os.makedirs(folder, exist_ok=True)
    np.savez(folder / "s1_64x48.npz", depth=np.zeros((48, 64)))
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert out["raw_depth"]["type"] == "depth"
    assert out["raw_depth"]["data"]["s1"].endswith("s1_64x48.npz")


def test_raw_folder_without_dimensions_not_classified_as_depth(tmp_path):
    folder = tmp_path / "raw_other"
    os.makedirs(folder, exist_ok=True)
    np.savez(folder / "s1.npz", x=np.zeros(3))  # No WxH suffix
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    # Regex `\d+x\d+\.npz$` fails → no type assigned → folder dropped.
    assert "raw_other" not in out


# ---------------------------------------------------------------------------
# is_submission gating for `metric_` prefix
# ---------------------------------------------------------------------------


def test_metric_prefix_typed_as_metric_when_is_submission(tmp_path):
    _write(tmp_path / "metric_acc", **{"s1.txt": "0.9"})
    out = detect_custom_fields(str(tmp_path), ["s1"], set(), is_submission=True)
    assert out["metric_acc"]["type"] == "metric"
    assert out["metric_acc"]["data"] == {"s1": 0.9}


def test_metric_prefix_treated_as_scalar_when_not_submission(tmp_path):
    # Same folder name, but is_submission=False → no special metric handling.
    _write(tmp_path / "metric_acc", **{"s1.txt": "0.9"})
    out = detect_custom_fields(str(tmp_path), ["s1"], set(), is_submission=False)
    assert out["metric_acc"]["type"] == "scalar"


# ---------------------------------------------------------------------------
# Precedence quirks (current behavior — pinned, not endorsed)
# ---------------------------------------------------------------------------


def test_image_and_txt_in_same_folder_image_wins_type_but_txt_wins_value(tmp_path):
    """When a sample has BOTH s1.png and s1.txt(=float):
    - field_type is set to 'image' on the first hit (the image check runs first).
    - The txt branch only re-types when type is None or 'metric', so type stays 'image'.
    - But field_data[s1] is overwritten by the txt value.
    Result: type='image', value=float. This is a known quirk worth pinning."""
    folder = tmp_path / "mixed"
    _write(folder, **{"s1.png": b"x", "s1.txt": "0.5"})
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert out["mixed"]["type"] == "image"
    assert out["mixed"]["data"]["s1"] == 0.5


def test_metric_folder_with_image_sample_keeps_metric_type(tmp_path):
    """Same precedence rule: metric folders start as 'metric' type; an image file
    won't downgrade it. The image data gets overwritten by any later txt/json hit
    if present, but here we only have a png."""
    _write(tmp_path / "metric_foo", **{"s1.png": b"x"})
    out = detect_custom_fields(str(tmp_path), ["s1"], set(), is_submission=True)
    assert out["metric_foo"]["type"] == "metric"
    assert out["metric_foo"]["data"]["s1"].endswith("s1.png")


def test_partial_sample_coverage(tmp_path):
    # Only s1 has a file; s2 is silently absent from data.
    _write(tmp_path / "scores", **{"s1.txt": "1.0"})
    out = detect_custom_fields(str(tmp_path), ["s1", "s2"], set())
    assert out["scores"]["type"] == "scalar"
    assert out["scores"]["data"] == {"s1": 1.0}


def test_files_at_base_path_root_ignored(tmp_path):
    # Stray file at root (not in any subfolder) must not crash the walker.
    (tmp_path / "loose.txt").write_text("ignore me")
    _write(tmp_path / "real", **{"s1.txt": "1.0"})
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    assert list(out.keys()) == ["real"]


def test_unreadable_txt_file_silently_skipped(tmp_path, monkeypatch):
    # The txt branch wraps the open() in try/except Exception — verify it
    # doesn't propagate. Simulate by making open() raise for one specific path.
    folder = tmp_path / "readme"
    _write(folder, **{"s1.txt": "1.0"})

    real_open = open
    target = str(folder / "s1.txt")

    def fake_open(path, *a, **kw):
        if str(path) == target:
            raise PermissionError("denied")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    out = detect_custom_fields(str(tmp_path), ["s1"], set())
    # Folder dropped entirely because no file successfully classified the sample.
    assert "readme" not in out
