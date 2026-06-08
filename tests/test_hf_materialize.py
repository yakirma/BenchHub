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


def test_materialize_sample_name_from_text_column_uses_its_values(tmp_path, monkeypatch):
    """Admin picks a text column → its sanitized value becomes the
    sample name on disk + in the manifest."""
    rows = [
        {"id": "alpha", "label": 0},
        {"id": "beta",  "label": 1},
        {"id": "gamma/three", "label": 2},  # gets sanitised
    ]
    _install_fake_datasets(monkeypatch, rows)
    summary = materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="named",
        fields=[
            {"name": "id",    "source_column": "id",    "kind": "text",
             "role": "input", "params": {}},
            {"name": "label", "source_column": "label", "kind": "label",
             "role": "gt", "params": {}},
        ],
        sample_name_from="id",
    )
    assert summary["samples"] == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["samples"] == ["alpha", "beta", "gamma_three"]
    # Files land under the chosen names.
    assert (tmp_path / "label" / "alpha.txt").exists()
    assert (tmp_path / "label" / "beta.txt").exists()
    assert (tmp_path / "label" / "gamma_three.txt").exists()


def test_materialize_sample_name_dedupes_collisions(tmp_path, monkeypatch):
    """Two rows with the same text value get distinct names — the
    second gets a `__<row_idx>` suffix so the on-disk layout is
    always unique."""
    rows = [
        {"id": "dup", "label": 0},
        {"id": "dup", "label": 1},
    ]
    _install_fake_datasets(monkeypatch, rows)
    materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="dedupe",
        fields=[
            {"name": "id",    "source_column": "id",    "kind": "text",
             "role": "input", "params": {}},
            {"name": "label", "source_column": "label", "kind": "label",
             "role": "gt", "params": {}},
        ],
        sample_name_from="id",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["samples"][0] == "dup"
    # Second occurrence carries a stable suffix tied to source-row index.
    assert manifest["samples"][1].startswith("dup__")
    assert manifest["samples"][0] != manifest["samples"][1]


def test_materialize_sample_name_falls_back_when_value_empty(tmp_path, monkeypatch):
    """An empty / whitespace text value uses the numbered default
    for that row instead of dropping the sample."""
    rows = [
        {"id": "first", "x": 1},
        {"id": "",      "x": 2},
        {"id": None,    "x": 3},
    ]
    _install_fake_datasets(monkeypatch, rows)
    materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="empty_names",
        fields=[
            {"name": "id", "source_column": "id", "kind": "text",
             "role": "input", "params": {}},
            {"name": "x",  "source_column": "x",  "kind": "scalar",
             "role": "gt", "params": {}},
        ],
        sample_name_from="id",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["samples"][0] == "first"
    # Empty and None fall back to enumerated names.
    assert manifest["samples"][1].startswith("s00000")
    assert manifest["samples"][2].startswith("s00000")


def test_materialize_sample_cap_minus_one_imports_every_row(tmp_path, monkeypatch):
    """sample_cap = -1 (or 0 / None) means "no cap" — the materializer
    pulls every row in the split. Quota gating lives in the calling
    route, not here."""
    rows = [{"x": i} for i in range(7)]
    _install_fake_datasets(monkeypatch, rows)
    summary = materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=-1,
        staging_dir=str(tmp_path), dataset_name="full_split",
        fields=[{"name": "x", "source_column": "x", "kind": "scalar",
                 "role": "gt", "params": {}}],
    )
    assert summary["samples"] == 7
    assert summary["total_rows_in_split"] == 7


def test_materialize_lifts_classlabel_names_into_params(tmp_path, monkeypatch):
    """For label fields, the HF `ClassLabel.names` vocab must land in
    the manifest's per-field params so the downstream import writes
    it to DatasetField.data_params. Drives the legend + int→name
    rendering on the dataset-view page."""
    rows = [{"label": 0}, {"label": 1}]

    class _FakeClassLabel:
        def __init__(self, names):
            self.names = names

    class _FakeDatasetWithFeatures(_FakeDataset):
        def __init__(self, rows, features):
            super().__init__(rows)
            self.features = features

    fake = types.ModuleType("datasets")
    ds = _FakeDatasetWithFeatures(
        rows,
        features={"label": _FakeClassLabel(["airplane", "automobile", "bird"])},
    )
    fake.load_dataset = lambda repo_id, **kw: ds
    monkeypatch.setitem(sys.modules, "datasets", fake)

    materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="d",
        fields=[{"name": "label", "source_column": "label", "kind": "label",
                 "role": "gt", "params": {}}],
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    label_field = next(f for f in manifest["fields"] if f["name"] == "label")
    assert label_field["params"]["names"] == ["airplane", "automobile", "bird"]


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


# ---------------------------------------------------------------------------
# _pick_indices — sampling strategies
# ---------------------------------------------------------------------------

from benchhub.hf_materialize import _pick_indices


def test_pick_indices_head_returns_prefix():
    ds = _FakeDataset([{"label": i % 3} for i in range(20)])
    out = _pick_indices(ds, 5, fields=[], strategy="head", seed=0)
    assert out == [0, 1, 2, 3, 4]


def test_pick_indices_uniform_is_seeded_and_sorted():
    """Same seed → same indices, deterministically; ascending order."""
    ds = _FakeDataset([{"x": i} for i in range(100)])
    a = _pick_indices(ds, 10, fields=[], strategy="uniform", seed=7)
    b = _pick_indices(ds, 10, fields=[], strategy="uniform", seed=7)
    c = _pick_indices(ds, 10, fields=[], strategy="uniform", seed=8)
    assert a == b
    assert a != c          # different seeds reshuffle
    assert a == sorted(a)  # ascending
    assert len(set(a)) == 10  # no dupes


def test_pick_indices_returns_full_range_when_n_exceeds_dataset():
    ds = _FakeDataset([{"x": i} for i in range(5)])
    assert _pick_indices(ds, 100, fields=[], strategy="uniform", seed=0) \
        == list(range(5))


def test_full_split_import_walks_rows_in_order_no_randomisation():
    """When the caller asks for every row (sample_cap=-1 → n=total),
    the picker must walk the split in row order regardless of the
    sampling strategy passed in. We assert this for all strategies
    so a future caller that forwards a stale strategy can't
    silently reshuffle a full import."""
    rows = [{"label": "a"}] * 5 + [{"label": "b"}] * 5
    ds = _FakeDataset(rows)
    fields = [{"name": "label", "kind": "label", "role": "gt"}]
    full = list(range(len(rows)))
    for strat in ("head", "uniform", "stratified"):
        assert _pick_indices(ds, len(rows), fields=fields,
                             strategy=strat, seed=0) == full, (
            f"strategy={strat} must NOT shuffle when n == len(ds)"
        )


def test_pick_indices_stratified_balances_classes():
    """All rows of class A come first, then class B. Stratified should
    pick equal counts of each, not just the head."""
    rows = (
        [{"label": "a"}] * 60
        + [{"label": "b"}] * 40
    )
    ds = _FakeDataset(rows)
    fields = [{"name": "label", "kind": "label", "role": "gt"}]
    out = _pick_indices(ds, 20, fields=fields, strategy="stratified", seed=0)
    labels_picked = [rows[i]["label"] for i in out]
    # Two classes, n=20 → 10 each.
    assert labels_picked.count("a") == 10
    assert labels_picked.count("b") == 10


def test_pick_indices_stratified_handles_remainder():
    """n=21 over 4 classes → 5+5+5+6 (extra goes to first-sorted class)."""
    rows = (
        [{"label": "a"}] * 10
        + [{"label": "b"}] * 10
        + [{"label": "c"}] * 10
        + [{"label": "d"}] * 10
    )
    ds = _FakeDataset(rows)
    fields = [{"name": "label", "kind": "label", "role": "gt"}]
    out = _pick_indices(ds, 21, fields=fields, strategy="stratified", seed=0)
    labels = [rows[i]["label"] for i in out]
    counts = {c: labels.count(c) for c in "abcd"}
    assert sum(counts.values()) == 21
    # The earliest label (sorted by str) gets the extra.
    assert counts["a"] == 6
    assert counts["b"] == counts["c"] == counts["d"] == 5


def test_pick_indices_stratified_falls_back_when_a_class_is_too_small():
    """One tiny class shouldn't break the request — we use what's
    available and top up the shortfall with uniform random."""
    rows = (
        [{"label": "rare"}] * 2
        + [{"label": "common"}] * 100
    )
    ds = _FakeDataset(rows)
    fields = [{"name": "label", "kind": "label", "role": "gt"}]
    out = _pick_indices(ds, 20, fields=fields, strategy="stratified", seed=0)
    labels = [rows[i]["label"] for i in out]
    # Took both rare samples + topped up with commons.
    assert labels.count("rare") == 2
    assert labels.count("common") == 18
    assert len(out) == 20


def test_pick_indices_stratified_falls_back_to_uniform_without_label_field():
    """No role=gt + kind=label among the fields → fall through to
    uniform without raising."""
    rows = [{"x": i} for i in range(50)]
    ds = _FakeDataset(rows)
    fields = [{"name": "x", "kind": "scalar", "role": "gt"}]
    out = _pick_indices(ds, 10, fields=fields, strategy="stratified", seed=0)
    expected = _pick_indices(ds, 10, fields=[], strategy="uniform", seed=0)
    assert out == expected


def test_pick_indices_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        _pick_indices(_FakeDataset([{"x": 1}]), 1, fields=[],
                      strategy="WAT", seed=0)


# ---------------------------------------------------------------------------
# materialize_hf_to_typed_dir × sampling — uses _pick_indices internally
# ---------------------------------------------------------------------------

def test_materialize_uniform_writes_seeded_subset(tmp_path, monkeypatch):
    rows = [{"v": i} for i in range(100)]
    _install_fake_datasets(monkeypatch, rows)
    summary = materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=8,
        staging_dir=str(tmp_path), dataset_name="ufm",
        fields=[{"name": "v", "source_column": "v", "kind": "scalar",
                 "role": "gt", "params": {}}],
        sampling="uniform", seed=123,
    )
    assert summary["samples"] == 8
    assert summary["sampling"] == "uniform"
    # The manifest carries the chosen source rows for traceability.
    import json
    mf = json.loads((tmp_path / "manifest.json").read_text())
    indices = mf["source"]["row_indices"]
    assert len(indices) == 8
    assert sorted(indices) == indices
    assert max(indices) <= 99


def test_materialize_stratified_balances_classes(tmp_path, monkeypatch):
    rows = [{"label": "cat", "v": i} for i in range(50)] \
        + [{"label": "dog", "v": i} for i in range(50)]
    _install_fake_datasets(monkeypatch, rows)
    materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="bal",
        fields=[
            {"name": "label", "source_column": "label", "kind": "label",
             "role": "gt", "params": {}},
        ],
        sampling="stratified", seed=0,
    )
    # Inspect what landed on disk — 5 cats + 5 dogs.
    label_files = sorted((tmp_path / "label").iterdir())
    classes = [p.read_text().strip('"') for p in label_files]
    assert classes.count("cat") == 5
    assert classes.count("dog") == 5


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

    # The sparse field (present for some-but-not-all rows) is flagged
    # `optional` so the importer tolerates the gap instead of failing
    # its missing-file pre-flight; the dense `img` field is not.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    by_name = {f["name"]: f for f in manifest["fields"]}
    assert by_name["tag"].get("optional") is True
    assert "optional" not in by_name["img"]


def test_materialize_all_null_field_stays_required(tmp_path, monkeypatch):
    """A field that's null for EVERY row is almost always a column/kind
    mis-map, not a legitimately-sparse column. Leave it un-flagged so
    the importer's strict pre-flight surfaces it rather than silently
    importing an empty field."""
    rows = [
        {"img": np.zeros((4, 4, 3), dtype=np.uint8), "tag": None},
        {"img": np.ones((4, 4, 3), dtype=np.uint8) * 255, "tag": None},
    ]
    _install_fake_datasets(monkeypatch, rows)

    materialize_hf_to_typed_dir(
        "x/y", split="test", sample_cap=10,
        staging_dir=str(tmp_path), dataset_name="allnull",
        fields=[
            {"name": "img", "source_column": "img", "kind": "image",
             "role": "input", "params": {}},
            {"name": "tag", "source_column": "tag", "kind": "text",
             "role": "gt", "params": {}},
        ],
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    by_name = {f["name"]: f for f in manifest["fields"]}
    assert "optional" not in by_name["tag"]
