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
    distinct_token_values, _resolve_json_pointer, _SafeFmt,
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


# --- Phase 2: json / csv loaders + variant automation ---

@pytest.fixture
def fake_repo_v2(tmp_path):
    """One sequence: pngs + a shared manifest.json (per-frame poses keyed
    by id, and a list aligned by order) + a meta.csv (one row per frame)."""
    root = tmp_path / "repo2"
    files = []

    def _w(rel, writer):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        writer(str(p)); files.append(rel)

    ids = [f"t{j:02d}" for j in range(3)]
    for k, sid in enumerate(ids):
        _w(f"seq/{sid}.png",
           lambda p, k=k: Image.fromarray(np.full((4, 5, 3), k, np.uint8)).save(p))
    manifest = {"frames": {sid: {"pose": [k, k, k]} for k, sid in enumerate(ids)},
                "ordered": [{"q": k * 10} for k in range(3)]}
    _w("seq/manifest.json", lambda p: open(p, "w").write(json.dumps(manifest)))
    csv_text = "id,score\n" + "\n".join(f"{sid},{k*0.5}" for k, sid in enumerate(ids))
    _w("seq/meta.csv", lambda p: open(p, "w").write(csv_text))
    return str(root), files


def _fetch(root):
    def f(rel):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(rel)
        return p
    return f


def test_json_pointer_resolver():
    obj = {"frames": {"t01": {"pose": [1, 2, 3]}}, "list": [{"v": 9}]}
    assert _resolve_json_pointer(obj, "frames.{id}.pose", {"id": "t01"}) == [1, 2, 3]
    assert _resolve_json_pointer(obj, "list.{ordinal}.v", {"ordinal": 0}) == 9
    assert _resolve_json_pointer(obj, "", {}) is obj


def test_json_loader_keyed_and_ordered(fake_repo_v2, tmp_path):
    root, files = fake_repo_v2
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "seq/{id}.png"},
        {"name": "pose", "kind": "json", "role": "gt", "loader": "json",
         "pattern": "seq/manifest.json", "pointer": "frames.{id}.pose", "shared": True},
        {"name": "q", "kind": "scalar", "role": "gt", "loader": "json",
         "pattern": "seq/manifest.json", "pointer": "ordered.{ordinal}.q", "shared": True},
    ]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(root), str(st), dataset_name="v2")
    assert summ["rows_written"]["pose"] == 3 and summ["rows_written"]["q"] == 3
    man = json.loads((st / "manifest.json").read_text())
    s0 = man["samples"][0]
    assert json.loads((st / "pose" / f"{s0}.json").read_text()) == [0, 0, 0]
    assert (st / "q" / f"{s0}.txt").read_text() == "0"   # ordered[0].q == 0
    s2 = man["samples"][2]
    assert (st / "q" / f"{s2}.txt").read_text() == "20"  # ordered[2].q == 20


def test_csv_loader_by_id_column(fake_repo_v2, tmp_path):
    root, files = fake_repo_v2
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "seq/{id}.png"},
        {"name": "score", "kind": "scalar", "role": "gt", "loader": "csv",
         "pattern": "seq/meta.csv", "column": "score", "id_column": "id"},
    ]
    st = tmp_path / "s"
    materialize_file_tree(spec, files, _fetch(root), str(st), dataset_name="v2")
    man = json.loads((st / "manifest.json").read_text())
    s1 = man["samples"][1]
    assert (st / "score" / f"{s1}.txt").read_text() == "0.5"


def test_distinct_token_values_for_variants(fake_repo):
    _root, files = fake_repo
    spec = [{"name": "image", "kind": "image", "loader": "file",
             "pattern": "train/{seq}/{quality}/{id}.png"}]
    assert sorted(distinct_token_values(spec, files, "quality")) == ["low", "normal"]


def test_token_filter_restricts_samples(fake_repo, tmp_path):
    _root, files = fake_repo
    root = _root
    spec = [{"name": "image", "kind": "image", "role": "input", "loader": "file",
             "pattern": "train/{seq}/{quality}/{id}.png"}]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(root), str(st),
                                 token_filter={"quality": "normal"}, dataset_name="v")
    # Only the 6 normal-quality pngs (2 seqs × 3), not the low ones.
    assert summ["samples"] == 6


# --- Phase 3: parquet + hdf5 loaders ---

def test_parquet_loader_by_order(fake_repo_v2, tmp_path):
    pytest.importorskip("pandas")
    root, files = fake_repo_v2
    import pandas as pd
    pq = os.path.join(root, "seq", "meta.parquet")
    pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_parquet(pq)
    files = files + ["seq/meta.parquet"]
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "seq/{id}.png"},
        {"name": "score", "kind": "scalar", "role": "gt", "loader": "parquet",
         "pattern": "seq/meta.parquet", "column": "score"},  # row order
    ]
    st = tmp_path / "s"
    materialize_file_tree(spec, files, _fetch(root), str(st), dataset_name="pq")
    man = json.loads((st / "manifest.json").read_text())
    assert (st / "score" / f"{man['samples'][1]}.txt").read_text() == "2.0"


def test_hdf5_shared_stacked_and_per_sample(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo3"
    files = []

    def _w(rel, writer):
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True)
        writer(str(p)); files.append(rel)

    ids = [f"t{j}" for j in range(3)]
    for k, sid in enumerate(ids):
        _w(f"seq/{sid}.png",
           lambda p, k=k: Image.fromarray(np.full((4, 5, 3), k, np.uint8)).save(p))
    # shared depth.h5: dataset 'depth' shape (3,4,5)
    _w("seq/depth.h5", lambda p: h5py.File(p, 'w').create_dataset(
        'depth', data=np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)))

    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "seq/{id}.png"},
        {"name": "depth", "kind": "depth", "role": "gt", "loader": "hdf5",
         "pattern": "seq/depth.h5", "key": "depth", "shared": True, "axis": 0},
    ]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(str(root)), str(st),
                                 dataset_name="h5")
    assert summ["rows_written"]["depth"] == 3
    man = json.loads((st / "manifest.json").read_text())
    d0 = np.load(st / "depth" / f"{man['samples'][0]}.npz")["depth"]
    assert d0.shape == (4, 5) and float(d0[0, 0]) == 0.0
    d2 = np.load(st / "depth" / f"{man['samples'][2]}.npz")["depth"]
    assert float(d2[0, 0]) == 40.0   # frame 2 starts at 2*20


# --- token loader (folder → label) + zip/tar/gz containers ---

def test_token_loader_folder_to_label(tmp_path):
    """`<class>/<id>.png` → image field + a label field from the {class}
    folder token, stored as an int index with a names vocab."""
    root = tmp_path / "sig"
    files = []

    def _w(rel):
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((3, 3, 3), np.uint8)).save(str(p)); files.append(rel)

    for cls in ("Alex_Brush", "Cookie"):
        for j in range(2):
            _w(f"{cls}/{j:03d}.png")

    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "{cls}/{id}.png"},
        {"name": "font", "kind": "label", "role": "gt",
         "loader": "token", "token": "cls"},
    ]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(str(root)), str(st),
                                 dataset_name="sig")
    assert summ["samples"] == 4 and summ["rows_written"]["font"] == 4
    man = json.loads((st / "manifest.json").read_text())
    font = next(f for f in man["fields"] if f["name"] == "font")
    assert font["params"]["names"] == ["Alex_Brush", "Cookie"]
    # sample 0 (Alex_Brush) → index 0; a Cookie sample → index 1.
    s0 = man["samples"][0]
    assert (st / "font" / f"{s0}.txt").read_text() == "0"
    cookie = next(n for n in man["samples"] if n.startswith("Cookie"))
    assert (st / "font" / f"{cookie}.txt").read_text() == "1"


def test_zip_container_member_loader(tmp_path):
    """A zip holds the images; index is a loose id list file, image read
    from the zip member by {id}."""
    import zipfile
    root = tmp_path / "z"; root.mkdir()
    # loose index files: ids 0,1,2 as tiny .txt so resolve enumerates them
    files = []
    for j in range(3):
        p = root / "ids" / f"{j}.txt"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x"); files.append(f"ids/{j}.txt")
    # zip of pngs
    zp = root / "imgs.zip"
    with zipfile.ZipFile(zp, 'w') as z:
        for j in range(3):
            import io as _io
            b = _io.BytesIO(); Image.fromarray(np.full((3, 3, 3), j, np.uint8)).save(b, 'PNG')
            z.writestr(f"pics/{j}.png", b.getvalue())
    files.append("imgs.zip")

    spec = [
        {"name": "sid", "kind": "text", "role": "gt",
         "loader": "file", "pattern": "ids/{id}.txt"},
        {"name": "image", "kind": "image", "role": "input",
         "loader": "zip", "pattern": "imgs.zip", "member": "pics/{id}.png"},
    ]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(str(root)), str(st),
                                 dataset_name="z")
    assert summ["rows_written"]["image"] == 3
    man = json.loads((st / "manifest.json").read_text())
    assert (st / "image" / f"{man['samples'][0]}.png").exists()


def test_zip_as_index_modality(tmp_path):
    """No loose files — the zip itself is the index; members enumerate
    the samples."""
    import zipfile
    root = tmp_path / "zi"; root.mkdir()
    zp = root / "data.zip"
    with zipfile.ZipFile(zp, 'w') as z:
        for j in range(3):
            import io as _io
            b = _io.BytesIO(); Image.fromarray(np.zeros((3, 3, 3), np.uint8)).save(b, 'PNG')
            z.writestr(f"frames/{j:02d}.png", b.getvalue())
    files = ["data.zip"]
    spec = [{"name": "image", "kind": "image", "role": "input",
             "loader": "zip", "pattern": "data.zip", "member": "frames/{id}.png"}]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(str(root)), str(st),
                                 dataset_name="zi")
    assert summ["samples"] == 3 and summ["rows_written"]["image"] == 3


def test_gz_single_file_loader(tmp_path):
    """A per-sample .json.gz decoded to a json field."""
    import gzip
    root = tmp_path / "g"
    files = []
    for j in range(2):
        # index pngs
        p = root / f"{j}.png"; p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((3, 3, 3), np.uint8)).save(str(p)); files.append(f"{j}.png")
        gp = root / f"{j}.json.gz"
        with gzip.open(str(gp), 'wb') as gf:
            gf.write(json.dumps({"v": j}).encode())
        files.append(f"{j}.json.gz")
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "{id}.png"},
        {"name": "meta", "kind": "json", "role": "gt",
         "loader": "gz", "pattern": "{id}.json.gz"},
    ]
    st = tmp_path / "s"
    materialize_file_tree(spec, files, _fetch(str(root)), str(st), dataset_name="g")
    man = json.loads((st / "manifest.json").read_text())
    assert json.loads((st / "meta" / f"{man['samples'][0]}.json").read_text()) == {"v": 0}


def test_materialize_sample_offset_picks_nth_sample(tmp_path):
    """(sample_offset=i, sample_cap=1) materializes exactly the i-th
    resolved sample — the decode preview's prev/next cycling contract."""
    root = tmp_path / "repo"
    (root / "txt").mkdir(parents=True)
    for i in range(3):
        (root / "txt" / f"s{i}.txt").write_text(f"value-{i}")
    files = [f"txt/s{i}.txt" for i in range(3)]
    spec = [{"name": "note", "kind": "text", "role": "gt",
             "loader": "file", "pattern": "txt/{id}.txt"}]
    st = tmp_path / "staging"
    materialize_file_tree(spec, files, _fetch(str(root)), str(st),
                          sample_cap=1, sample_offset=1, dataset_name="o")
    man = json.loads((st / "manifest.json").read_text())
    assert man["samples"] == ["s1"]
    assert (st / "note" / "s1.txt").read_text() == "value-1"


def test_inspect_suggests_label_folder_pattern():
    """`<class>/<id>.png` with multiple class folders → a
    {label}/{id}.png suggestion (folder = label), preferred + first; the
    per-class literals are still emitted but tagged group='folder' so the
    UI's toggle hides them by default."""
    files = [f"{cls}/{i:03d}.png" for cls in ("Alex_Brush", "Cookie", "Lobster")
             for i in range(3)]
    info = inspect_repo(files)
    pats = [s["pattern"] for s in info["suggested_patterns"]]
    assert "{label}/{id}.png" in pats
    assert info["folder_toggle"] == "label"
    # the label suggestion is flagged + comes first
    first = info["suggested_patterns"][0]
    assert first["pattern"] == "{label}/{id}.png" and first.get("label_folder")
    assert first["group"] == "label"
    # per-class literals exist only as the toggled-off 'folder' reading
    per_class = [s for s in info["suggested_patterns"]
                 if s["pattern"].startswith("Alex_Brush/")]
    assert per_class and all(s.get("group") == "folder" for s in per_class)


def test_inspect_no_label_suggestion_for_single_folder():
    """A single folder of files (parent doesn't vary) → no {label}
    suggestion, no toggle, just the normal `<dir>/{id}` one."""
    files = [f"images/{i}.png" for i in range(4)]
    info = inspect_repo(files)
    pats = [s["pattern"] for s in info["suggested_patterns"]]
    assert "{label}/{id}.png" not in pats
    assert "images/{id}.png" in pats
    assert info["folder_toggle"] is None


def test_inspect_keeps_modality_folders_label_as_alternate():
    """Sibling MODALITY folders (image/ + mask/, both png) keep their
    individual `folder/{id}` suggestions as the preferred reading; the
    {label}/{id} alternate is still offered (tagged group='label', after
    them) so the toggle can flip a misclassified class-folder dataset."""
    files = [f"image/{i}.png" for i in range(4)] + \
            [f"mask/{i}.png" for i in range(4)]
    info = inspect_repo(files)
    pats = [s["pattern"] for s in info["suggested_patterns"]]
    assert info["folder_toggle"] == "modality"
    assert "image/{id}.png" in pats                # both modality lines kept
    assert "mask/{id}.png" in pats
    # the {label} reading is available but non-preferred (after the others)
    assert "{label}/{id}.png" in pats
    lab = next(s for s in info["suggested_patterns"]
               if s["pattern"] == "{label}/{id}.png")
    assert lab["group"] == "label"
    assert pats.index("{label}/{id}.png") > pats.index("mask/{id}.png")


def test_inspect_label_alternate_for_modality_looking_class_names():
    """Class folders whose names collide with modality words (e.g. a
    fonts dataset with a class literally called 'normal') used to lose the
    {label} suggestion entirely when ≥50% looked modality-like. Now the
    heuristic only picks the default; {label}/{id} is always offered."""
    files = [f"{cls}/{i}.jpg" for cls in ("normal", "thermal")  # all modality words
             for i in range(3)]
    info = inspect_repo(files)
    pats = [s["pattern"] for s in info["suggested_patterns"]]
    assert info["folder_toggle"] == "modality"     # heuristic default
    assert "{label}/{id}.jpg" in pats              # …but the toggle has both


def test_inspect_mixed_modalities_different_exts():
    """image/<id>.png + depth/<id>.npz (the user's example) → each ext has
    one folder, so no {label}; both individual patterns are suggested."""
    files = [f"image/{i}.png" for i in range(4)] + \
            [f"depth/{i}.npz" for i in range(4)]
    info = inspect_repo(files)
    pats = [s["pattern"] for s in info["suggested_patterns"]]
    assert "{label}/{id}.png" not in pats and "{label}/{id}.npz" not in pats
    assert "image/{id}.png" in pats
    assert "depth/{id}.npz" in pats


# --- Phase: path-level role analysis → auto-generated spec ---

from benchhub.file_tree_import import analyze_levels, generate_spec_from_roles


def test_analyze_levels_benjy_class_folders():
    files = [f"{c}/{i:03d}.png" for c in ("Alex_Brush", "Cookie", "Lobster")
             for i in range(3)]
    info = analyze_levels(files)
    assert info["depth"] == 2
    assert info["levels"][0]["default_role"] == "property"   # arbitrary folder names
    assert info["levels"][1]["default_role"] == "id"
    assert info["levels"][1]["exts"] == {"png": 9}


def test_analyze_levels_modality_folders():
    files = [f"image/{i}.png" for i in range(3)] + [f"depth/{i}.npz" for i in range(3)]
    info = analyze_levels(files)
    assert info["levels"][0]["default_role"] == "modality"


def test_generate_spec_property_folder_to_image_plus_label():
    files = [f"{c}/{i:03d}.png" for c in ("Alex_Brush", "Cookie") for i in range(2)]
    spec = generate_spec_from_roles(files, ["property", "id"])
    by = {f["name"]: f for f in spec}
    assert by["image"]["loader"] == "file"
    assert by["image"]["pattern"] == "{label}/{id}.png"
    assert by["label"]["loader"] == "token" and by["label"]["kind"] == "label"
    assert by["label"]["token"] == "label"


def test_generate_spec_modality_fans_out_fields():
    files = [f"image/{i}.png" for i in range(2)] + [f"depth/{i}.npz" for i in range(2)]
    spec = generate_spec_from_roles(files, ["modality", "id"])
    by = {f["name"]: f for f in spec}
    assert set(by) == {"image", "depth"}
    assert by["image"]["pattern"] == "image/{id}.png" and by["image"]["kind"] == "image"
    assert by["depth"]["pattern"] == "depth/{id}.npz"
    assert by["depth"]["loader"] == "npz"


def test_generate_spec_fixed_group_split_tokens():
    files = [f"train/i_{s}/{q}/{i:03d}.png"
             for s in range(2) for q in ("normal", "low") for i in range(2)]
    spec = generate_spec_from_roles(files, ["fixed", "group", "split", "id"])
    # one image field; train literal, i_* → {seq}, quality → {split}
    img = [f for f in spec if f["kind"] == "image"][0]
    assert img["pattern"] == "train/{seq}/{split}/{id}.png"


def test_generate_spec_file_level_multi_ext_fans_out():
    files = [f"{i}.png" for i in range(2)] + [f"{i}.npz" for i in range(2)]
    spec = generate_spec_from_roles(files, ["id"])
    kinds = {f["name"]: f for f in spec}
    assert kinds["image"]["pattern"] == "{id}.png"
    assert kinds["depth"]["pattern"] == "{id}.npz"


# --- sequence loader: group frames into a clip ---

def test_sequence_loader_groups_frames_into_clip(tmp_path):
    from benchhub.types import Sequence
    import zipfile
    root = tmp_path / "r"
    files = []
    def _w(rel):
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(str(p)); files.append(rel)
    for clip in ("c0", "c1"):
        for fr in range(3):
            _w(f"clips/{clip}/{fr:02d}.png")

    spec = [{"name": "video", "kind": "sequence", "role": "input",
             "loader": "sequence", "pattern": "clips/{id}/{frame}.png"}]
    st = tmp_path / "s"
    summ = materialize_file_tree(spec, files, _fetch(str(root)), str(st),
                                 dataset_name="v")
    assert summ["samples"] == 2                       # 2 clips, not 6 frames
    man = json.loads((st / "manifest.json").read_text())
    vf = next(f for f in man["fields"] if f["name"] == "video")
    assert vf["kind"] == "sequence" and vf["params"]["item_kind"] == "image"
    s0 = man["samples"][0]
    z = zipfile.ZipFile(st / "video" / f"{s0}.zip")
    assert len(z.namelist()) == 3                     # 3 frames in the clip
    seq = Sequence.decode((st / "video" / f"{s0}.zip").read_bytes(), vf["params"])
    assert len(seq.frames) == 3


# --- Canonical-ext transcoding at the staging boundary -------------------
# The typed-manifest contract is ONE canonical ext per kind (image/mask →
# .png, audio → .wav). Non-canonical sources (jpg, webp, jxl, flac …) must
# transcode at staging — verbatim copy used to stage `image/<s>.jpg` and
# the importer's existence check then failed on the expected `<s>.png`.

def _validate_staging(staging):
    """The importer's missing-file check, standalone (no DB)."""
    from pathlib import Path
    from benchhub.manifest import _DATA_BEARING_ROLES, expected_file_path
    manifest = json.loads((Path(staging) / "manifest.json").read_text())
    missing = []
    for f in manifest["fields"]:
        if f.get("role", "gt") not in _DATA_BEARING_ROLES:
            continue
        for s in manifest["samples"]:
            if not expected_file_path(staging, f, s).exists():
                missing.append(f"{f['name']}/{s}")
    return missing


def test_jpg_image_transcodes_to_canonical_png(tmp_path):
    from PIL import Image as PILImage
    root = tmp_path / "repo"; (root / "img").mkdir(parents=True)
    PILImage.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(root / "img" / "a.jpg")
    spec = [{"name": "image", "kind": "image", "role": "input",
             "loader": "file", "pattern": "img/{id}.jpg"}]
    st = tmp_path / "st"
    materialize_file_tree(spec, ["img/a.jpg"], _fetch(str(root)), str(st),
                          dataset_name="t")
    assert (st / "image" / "a.png").exists()       # canonical, not a.jpg
    assert _validate_staging(st) == []             # importer check passes
    arr = np.asarray(PILImage.open(st / "image" / "a.png").convert("RGB"))
    assert abs(int(arr[0, 0, 0]) - 120) <= 3       # jpg-lossy tolerance


def test_jxl_image_transcodes_to_canonical_png(tmp_path):
    pytest.importorskip("pillow_jxl")
    from PIL import Image as PILImage
    root = tmp_path / "repo"; (root / "img").mkdir(parents=True)
    PILImage.fromarray(np.full((8, 8, 3), 64, np.uint8)).save(root / "img" / "a.jxl")
    spec = [{"name": "image", "kind": "image", "role": "input",
             "loader": "file", "pattern": "img/{id}.jxl"}]
    st = tmp_path / "st"
    materialize_file_tree(spec, ["img/a.jxl"], _fetch(str(root)), str(st),
                          dataset_name="t")
    assert (st / "image" / "a.png").exists()
    assert _validate_staging(st) == []
    arr = np.asarray(PILImage.open(st / "image" / "a.png").convert("RGB"))
    assert abs(int(arr[0, 0, 0]) - 64) <= 3


def test_flac_audio_transcodes_to_canonical_wav(tmp_path):
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "repo"; (root / "au").mkdir(parents=True)
    wave = np.sin(np.linspace(0, 200, 1600)).astype(np.float32)
    sf.write(root / "au" / "a.flac", wave, 16000, format="FLAC")
    spec = [{"name": "audio", "kind": "audio", "role": "input",
             "loader": "file", "pattern": "au/{id}.flac"}]
    st = tmp_path / "st"
    materialize_file_tree(spec, ["au/a.flac"], _fetch(str(root)), str(st),
                          dataset_name="t")
    assert (st / "audio" / "a.wav").exists()
    assert _validate_staging(st) == []
    data, sr = sf.read(st / "audio" / "a.wav")
    assert sr == 16000 and len(data) == 1600


def test_canonical_png_still_copies_verbatim(tmp_path):
    from PIL import Image as PILImage
    root = tmp_path / "repo"; (root / "img").mkdir(parents=True)
    PILImage.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "img" / "a.png")
    src_bytes = (root / "img" / "a.png").read_bytes()
    spec = [{"name": "image", "kind": "image", "role": "input",
             "loader": "file", "pattern": "img/{id}.png"}]
    st = tmp_path / "st"
    materialize_file_tree(spec, ["img/a.png"], _fetch(str(root)), str(st),
                          dataset_name="t")
    assert (st / "image" / "a.png").read_bytes() == src_bytes  # untouched


def test_undecodable_bytes_fail_with_readable_error(tmp_path):
    root = tmp_path / "repo"; (root / "img").mkdir(parents=True)
    (root / "img" / "a.jpg").write_bytes(b"definitely not an image")
    spec = [{"name": "image", "kind": "image", "role": "input",
             "loader": "file", "pattern": "img/{id}.jpg"}]
    with pytest.raises(ValueError, match="can't decode a.jpg as image"):
        materialize_file_tree(spec, ["img/a.jpg"], _fetch(str(root)),
                              str(tmp_path / "st"), dataset_name="t")


def test_jxl_ext_maps_to_image_kind():
    from benchhub.file_tree_import import _kind_loader_for_ext
    assert _kind_loader_for_ext("jxl") == ("image", "file")
