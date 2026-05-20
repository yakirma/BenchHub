"""Tests for benchhub.hf_materialize — HF rows → typed manifest layout.

`datasets.load_dataset()` is monkeypatched throughout so these tests
don't touch the network or require the real HF library to be installed.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import benchhub as bh
from benchhub.hf_materialize import _row_value_to_typed, materialize_hf_to_typed_dir


# ---------------------------------------------------------------------------
# Per-kind value coercion
# ---------------------------------------------------------------------------

def test_scalar_value_wraps_to_bh_Scalar():
    inst = _row_value_to_typed(3.5, "scalar", {})
    assert isinstance(inst, bh.Scalar)
    assert inst.value == 3.5


def test_label_int_value_wraps_to_bh_Label():
    inst = _row_value_to_typed(7, "label", {})
    assert isinstance(inst, bh.Label)
    assert inst.value == 7


def test_text_value_stringifies():
    inst = _row_value_to_typed(42, "text", {})
    assert isinstance(inst, bh.Text)
    assert inst.text == "42"


def test_image_numpy_array_wraps_to_bh_Image():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    inst = _row_value_to_typed(arr, "image", {})
    assert isinstance(inst, bh.Image)
    assert inst.array.shape == (4, 4, 3)


def test_depth_picks_up_unit_param():
    arr = np.ones((4, 4), dtype=np.float32)
    inst = _row_value_to_typed(arr, "depth", {"unit": "millimeters"})
    assert isinstance(inst, bh.Depth)
    assert inst.unit == "millimeters"


def test_none_value_returns_none():
    assert _row_value_to_typed(None, "scalar", {}) is None


def test_audio_dict_value_wraps_to_bh_Audio():
    wav = np.zeros(16000, dtype=np.float32)
    inst = _row_value_to_typed(
        {"array": wav, "sampling_rate": 16000}, "audio", {},
    )
    assert isinstance(inst, bh.Audio)
    assert inst.sample_rate == 16000


def test_json_value_wraps_dict():
    inst = _row_value_to_typed({"a": 1}, "json", {})
    assert isinstance(inst, bh.Json)
    assert inst.data == {"a": 1}


def test_bad_value_for_kind_returns_none_not_raises():
    # A string for kind=scalar can't be float()'d → None, no crash.
    assert _row_value_to_typed("not-a-float", "scalar", {}) is None


# ---------------------------------------------------------------------------
# materialize_hf_to_typed_dir — uses a stubbed `datasets` module
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Stand-in for an HF Dataset object. Indexable, has __len__."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def _install_fake_datasets(monkeypatch, rows: list[dict]):
    """Inject a fake `datasets` module whose `load_dataset` returns the
    given rows. Lets us materialise without the real HF library."""
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda repo_id, **kw: _FakeDataset(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_materialize_rejects_empty_field_list(tmp_path):
    with pytest.raises(ValueError, match="no fields"):
        materialize_hf_to_typed_dir(
            "x/y", split="test", sample_cap=1,
            staging_dir=str(tmp_path), dataset_name="d", fields=[],
        )


def test_materialize_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        materialize_hf_to_typed_dir(
            "x/y", split="test", sample_cap=1,
            staging_dir=str(tmp_path), dataset_name="d",
            fields=[{"name": "x", "source_column": "x", "kind": "WRONG",
                     "role": "gt", "params": {}}],
        )


def test_materialize_writes_manifest_and_files(tmp_path, monkeypatch):
    rows = [
        {"img": np.zeros((4, 4, 3), dtype=np.uint8), "label": 0},
        {"img": np.ones((4, 4, 3), dtype=np.uint8) * 255, "label": 1},
    ]
    _install_fake_datasets(monkeypatch, rows)

    summary = materialize_hf_to_typed_dir(
        "uoft-cs/cifar10",
        split="test",
        sample_cap=10,
        staging_dir=str(tmp_path),
        dataset_name="cifar10-small",
        fields=[
            {"name": "image", "source_column": "img", "kind": "image",
             "role": "input", "params": {}},
            {"name": "label", "source_column": "label", "kind": "label",
             "role": "gt", "params": {}},
        ],
    )
    assert summary["samples"] == 2
    assert summary["rows_written"] == 4  # 2 samples × 2 fields
    assert summary["rows_skipped"] == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["name"] == "cifar10-small"
    assert manifest["samples"] == ["s000000", "s000001"]
    assert {f["name"] for f in manifest["fields"]} == {"image", "label"}

    # File-backed kind: .png written under image/.
    assert (tmp_path / "image" / "s000000.png").exists()
    assert (tmp_path / "image" / "s000001.png").exists()
    # Inline kind: .txt under label/.
    assert (tmp_path / "label" / "s000000.txt").read_text() == "0"
    assert (tmp_path / "label" / "s000001.txt").read_text() == "1"


def test_materialize_respects_sample_cap(tmp_path, monkeypatch):
    rows = [{"x": i} for i in range(50)]
    _install_fake_datasets(monkeypatch, rows)

    summary = materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=5,
        staging_dir=str(tmp_path), dataset_name="capped",
        fields=[{"name": "x", "source_column": "x", "kind": "scalar",
                 "role": "gt", "params": {}}],
    )
    assert summary["samples"] == 5
    assert len(list((tmp_path / "x").iterdir())) == 5


def test_materialize_skips_when_value_is_none(tmp_path, monkeypatch):
    rows = [
        {"img": np.zeros((4, 4, 3), dtype=np.uint8), "tag": "first"},
        {"img": np.ones((4, 4, 3), dtype=np.uint8) * 255, "tag": None},
    ]
    _install_fake_datasets(monkeypatch, rows)

    summary = materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="sparse",
        fields=[
            {"name": "img", "source_column": "img", "kind": "image",
             "role": "input", "params": {}},
            {"name": "tag", "source_column": "tag", "kind": "text",
             "role": "gt", "params": {}},
        ],
    )
    assert summary["samples"] == 2
    assert summary["rows_written"] == 3  # img×2 + tag×1; tag for second row is None
    assert summary["rows_skipped"] == 1
    assert "s000001/tag" in summary["skipped_sample_field_pairs"]
