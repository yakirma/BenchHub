"""Engine tests for the user-declared file-tree importer.

Builds a synthetic repo on disk mirroring the ECCV shape (paired
RGB .png + per-sample event .npz in `low/`, plus a shared stacked
depth.npz in `normal/`), then drives `materialize_file_tree` with a
local-dir fetch (no network)."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from benchhub.file_tree_import import (
    inspect_repo, match_files, resolve_samples, materialize_file_tree,
)


@pytest.fixture
def fake_repo(tmp_path):
    """Two sequences, each with `normal/` (pngs + one stacked depth.npz)
    and `low/` (pngs + per-sample event npz). Returns (root, files)."""
    root = tmp_path / "repo"
    files = []

    def _w(rel, writer):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        writer(str(p))
        files.append(rel)

    for seq in ("i_0", "i_1"):
        ids = [f"{seq}_t{j:02d}" for j in range(3)]
        # normal/: 3 pngs + one stacked depth.npz of shape (3, 4, 5, 1)
        depth = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5, 1) + (
            0 if seq == "i_0" else 1000)
        for k, sid in enumerate(ids):
            _w(f"train/{seq}/normal/{sid}.png",
               lambda p, k=k: Image.fromarray(
                   np.full((4, 5, 3), k * 10, np.uint8)).save(p))
        _w(f"train/{seq}/normal/depth.npz",
           lambda p, depth=depth: np.savez_compressed(p, depth=depth))
        # low/: per-sample event npz (arr_0, shape (N,4))
        for k, sid in enumerate(ids):
            _w(f"train/{seq}/low/{sid}.npz",
               lambda p, k=k: np.savez_compressed(
                   p, arr_0=np.arange((k + 1) * 4, dtype=np.int64).reshape(-1, 4)))
            _w(f"train/{seq}/low/{sid}.png",
               lambda p: Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(p))
    return str(root), files


def _local_fetch(root):
    def fetch(rel):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(rel)
        return p
    return fetch


def test_match_files_captures_tokens(fake_repo):
    _root, files = fake_repo
    matches, names = match_files("train/{seq}/normal/{id}.png", files)
    assert names == ["seq", "id"]
    assert len(matches) == 6  # 2 seqs × 3
    assert all("seq" in m and "id" in m for m in matches)


def test_inspect_suggests_patterns(fake_repo):
    _root, files = fake_repo
    info = inspect_repo(files)
    assert info["ext_histogram"].get("png", 0) >= 6
    assert any(s["ext"] == "png" for s in info["suggested_patterns"])


def test_resolve_samples_sorted_and_named(fake_repo):
    _root, files = fake_repo
    spec = [{"name": "image", "kind": "image", "role": "input",
             "loader": "file", "pattern": "train/{seq}/normal/{id}.png"}]
    samples, index = resolve_samples(spec, files)
    assert len(samples) == 6
    # sorted by path → i_0 group before i_1
    assert samples[0]["name"].startswith("i_0")
    assert samples[-1]["name"].startswith("i_1")


def test_materialize_file_image_and_shared_depth_and_per_sample_events(fake_repo, tmp_path):
    root, files = fake_repo
    staging = tmp_path / "stage"
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "train/{seq}/normal/{id}.png"},
        {"name": "depth", "kind": "depth", "role": "gt",
         "loader": "npz", "pattern": "train/{seq}/normal/depth.npz",
         "key": "depth", "shared": True, "axis": 0},
        {"name": "events", "kind": "json", "role": "input",
         "loader": "npz", "pattern": "train/{seq}/low/{id}.npz", "key": "arr_0"},
    ]
    summary = materialize_file_tree(spec, files, _local_fetch(root), str(staging),
                                    dataset_name="eccv-test")
    assert summary["samples"] == 6
    assert summary["rows_written"]["image"] == 6
    assert summary["rows_written"]["depth"] == 6
    assert summary["rows_written"]["events"] == 6

    manifest = json.loads((staging / "manifest.json").read_text())
    assert {f["name"] for f in manifest["fields"]} == {"image", "depth", "events"}
    assert len(manifest["samples"]) == 6

    # Shared depth correctly split + aligned: sample 0 (i_0, t00) → frame 0.
    s0 = manifest["samples"][0]
    dz = np.load(staging / "depth" / f"{s0}.npz")["depth"]
    assert dz.shape == (4, 5)              # (4,5,1) squeezed to 2D
    assert float(dz[0, 0]) == 0.0          # frame 0 of i_0 starts at 0
    # Last i_0 sample → frame 2 of i_0's depth (value base 2*20=40 at [0,0]).
    s2 = manifest["samples"][2]
    dz2 = np.load(staging / "depth" / f"{s2}.npz")["depth"]
    assert float(dz2[0, 0]) == 40.0
    # i_1's first sample → its own archive, base 1000.
    s3 = manifest["samples"][3]
    dz3 = np.load(staging / "depth" / f"{s3}.npz")["depth"]
    assert float(dz3[0, 0]) == 1000.0

    # Per-sample events decoded to JSON arrays.
    ev0 = json.loads((staging / "events" / f"{s0}.json").read_text())
    assert isinstance(ev0, list) and len(ev0[0]) == 4

    # Image copied through.
    assert (staging / "image" / f"{s0}.png").exists()


def test_resolve_raises_without_index_field(fake_repo):
    _root, files = fake_repo
    spec = [{"name": "depth", "kind": "depth", "loader": "npz",
             "pattern": "x/depth.npz", "key": "depth", "shared": True}]
    with pytest.raises(ValueError, match="index modality"):
        resolve_samples(spec, files)


def test_resolve_raises_on_no_match(fake_repo):
    _root, files = fake_repo
    spec = [{"name": "image", "kind": "image", "loader": "file",
             "pattern": "nope/{id}.png"}]
    with pytest.raises(ValueError, match="matched no files"):
        resolve_samples(spec, files)
