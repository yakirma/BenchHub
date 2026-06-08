"""End-to-end (offline) tests for the Kaggle conversion loaders wired into
benchhub.file_tree_import: rle / coco / voc / yolo loaders + the shared
CSV/parquet row index. Builds a real staging dir with a local-dir fetch and
asserts the decoded staged outputs."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from benchhub.file_tree_import import materialize_file_tree
from benchhub.kaggle_detect import detect_shape


@pytest.fixture
def repo(tmp_path):
    """A tiny on-disk repo + a fetch(relpath)->localpath callable."""
    root = tmp_path / "repo"
    root.mkdir()

    def write(rel, data):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            p.write_bytes(data)
        else:
            p.write_text(data)

    def write_img(rel, w, h, color=(10, 20, 30)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (w, h), color).save(p)

    def fetch(rel):
        return str(root / rel)

    repo = type("Repo", (), {})()
    repo.root, repo.write, repo.write_img, repo.fetch = root, write, write_img, fetch
    return repo


def _read_json(staging, field, name):
    with open(os.path.join(staging, field, name + ".json")) as fh:
        return json.load(fh)


def _read_mask(staging, field, name):
    return np.asarray(Image.open(os.path.join(staging, field, name + ".png")))


# --------------------------------------------------------------------------
# RLE-in-CSV → Mask (shape I), paired with an images dir
# --------------------------------------------------------------------------

def test_rle_csv_to_mask(repo, tmp_path):
    # Two 3x2 (W=3,H=2) images; col-major RLE "1 2" sets column 0.
    repo.write_img("train_images/a.png", 3, 2)
    repo.write_img("train_images/b.png", 3, 2)
    repo.write("train.csv",
               "ImageId,EncodedPixels\na.png,1 2\nb.png,3 2\n")
    files = ["train.csv", "train_images/a.png", "train_images/b.png"]
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "train_images/{id}.png"},
        {"name": "mask", "kind": "mask", "role": "gt", "loader": "rle",
         "pattern": "train.csv", "id_column": "ImageId", "id_token": "id",
         "value_column": "EncodedPixels", "class_column": None,
         "order": "F", "one_indexed": True},
    ]
    staging = str(tmp_path / "stage")
    summary = materialize_file_tree(spec, files, repo.fetch, staging)
    assert summary["samples"] == 2
    m = _read_mask(staging, "mask", "a")
    assert m.shape == (2, 3)
    assert (m[:, 0] == 1).all()        # column 0 set (col-major decode)
    assert (m[:, 1:] == 0).all()
    # 'b' → RLE "3 2" sets column 1.
    mb = _read_mask(staging, "mask", "b")
    assert (mb[:, 1] == 1).all() and (mb[:, 0] == 0).all()


def test_rle_csv_with_classid(repo, tmp_path):
    repo.write_img("imgs/x.png", 2, 2)
    repo.write("masks.csv",
               "ImageId,ClassId,EncodedPixels\nx.png,3,1 2\nx.png,7,3 2\n")
    files = ["masks.csv", "imgs/x.png"]
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "imgs/{id}.png"},
        {"name": "mask", "kind": "mask", "role": "gt", "loader": "rle",
         "pattern": "masks.csv", "id_column": "ImageId", "id_token": "id",
         "value_column": "EncodedPixels", "class_column": "ClassId",
         "order": "F", "one_indexed": True},
    ]
    staging = str(tmp_path / "stage")
    materialize_file_tree(spec, files, repo.fetch, staging)
    m = _read_mask(staging, "mask", "x")
    assert set(np.unique(m)) == {3, 7}     # the two ClassId values, no bg here


# --------------------------------------------------------------------------
# COCO JSON → CocoDetections (shape E)
# --------------------------------------------------------------------------

def test_coco_json_to_detections(repo, tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"},
                   {"id": 2, "file_name": "b.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 5, "bbox": [1, 2, 3, 4], "area": 12},
            {"image_id": 1, "category_id": 6, "bbox": [5, 6, 7, 8]},
            {"image_id": 2, "category_id": 5, "bbox": [0, 0, 2, 2]},
        ],
        "categories": [{"id": 5, "name": "cat"}, {"id": 6, "name": "dog"}],
    }
    repo.write("annotations/inst.json", json.dumps(coco))
    repo.write_img("images/a.jpg", 8, 8)
    repo.write_img("images/b.jpg", 8, 8)
    files = ["annotations/inst.json", "images/a.jpg", "images/b.jpg"]
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "images/{id}.jpg"},
        {"name": "detections", "kind": "coco_detections", "role": "gt",
         "loader": "coco", "pattern": "annotations/inst.json"},
    ]
    staging = str(tmp_path / "stage")
    materialize_file_tree(spec, files, repo.fetch, staging)
    dets = _read_json(staging, "detections", "a")
    assert len(dets) == 2
    assert dets[0]["category_id"] == 5 and dets[0]["category_name"] == "cat"
    assert dets[0]["bbox"] == [1, 2, 3, 4]      # COCO xywh preserved
    assert _read_json(staging, "detections", "b")[0]["category_id"] == 5


# --------------------------------------------------------------------------
# VOC XML → BBoxes (shape F)
# --------------------------------------------------------------------------

def test_voc_xml_to_bboxes(repo, tmp_path):
    xml = ("<annotation><object><name>cat</name>"
           "<bndbox><xmin>1</xmin><ymin>1</ymin>"
           "<xmax>3</xmax><ymax>5</ymax></bndbox></object></annotation>")
    repo.write("Annotations/a.xml", xml)
    repo.write("Annotations/b.xml", xml)
    repo.write_img("JPEGImages/a.jpg", 8, 8)
    repo.write_img("JPEGImages/b.jpg", 8, 8)
    files = ["Annotations/a.xml", "Annotations/b.xml",
             "JPEGImages/a.jpg", "JPEGImages/b.jpg"]
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "JPEGImages/{id}.jpg"},
        {"name": "boxes", "kind": "bboxes", "role": "gt", "loader": "voc",
         "pattern": "Annotations/{id}.xml", "one_indexed": True},
    ]
    staging = str(tmp_path / "stage")
    materialize_file_tree(spec, files, repo.fetch, staging)
    b = _read_json(staging, "boxes", "a")
    assert b["format"] == "xyxy" and b["labels"] == ["cat"]
    # VOC 1-indexed inclusive → shifted to 0-indexed.
    assert b["boxes"] == [[0, 0, 2, 4]]


# --------------------------------------------------------------------------
# YOLO TXT → BBoxes (shape G)
# --------------------------------------------------------------------------

def test_yolo_txt_to_bboxes(repo, tmp_path):
    repo.write("labels/a.txt", "0 0.5 0.5 0.5 0.5\n")
    repo.write_img("images/a.jpg", 10, 10)
    files = ["labels/a.txt", "images/a.jpg"]
    spec = [
        {"name": "image", "kind": "image", "role": "input",
         "loader": "file", "pattern": "images/{id}.jpg"},
        {"name": "boxes", "kind": "bboxes", "role": "gt", "loader": "yolo",
         "pattern": "labels/{id}.txt"},
    ]
    staging = str(tmp_path / "stage")
    materialize_file_tree(spec, files, repo.fetch, staging)
    b = _read_json(staging, "boxes", "a")
    # cx=cy=0.5 w=h=0.5 on 10x10 → centre (5,5), 5x5 → (2.5,2.5,7.5,7.5).
    assert b["boxes"] == [[2.5, 2.5, 7.5, 7.5]] and b["labels"] == [0]


# --------------------------------------------------------------------------
# Shared CSV row index (shape A — pure tabular)
# --------------------------------------------------------------------------

def test_tabular_csv_row_index(repo, tmp_path):
    repo.write("train.csv", "text,label\nhello world,a\nfoo bar,b\n")
    files = ["train.csv"]
    spec = [
        {"name": "text", "kind": "text", "role": "input", "loader": "csv",
         "pattern": "train.csv", "column": "text", "index": True},
        {"name": "label", "kind": "label", "role": "gt", "loader": "csv",
         "pattern": "train.csv", "column": "label"},
    ]
    staging = str(tmp_path / "stage")
    summary = materialize_file_tree(spec, files, repo.fetch, staging)
    assert summary["samples"] == 2
    # rows map in order to row_000000 / row_000001
    with open(os.path.join(staging, "text", "row_000000.txt")) as fh:
        assert fh.read() == "hello world"
    with open(os.path.join(staging, "label", "row_000001.txt")) as fh:
        assert fh.read() == "b"


def test_tabular_offset_aligns_rows(repo, tmp_path):
    # sample_offset must not desync row picking (the _row token guards it).
    repo.write("t.csv", "x,y\n0,a\n1,b\n2,c\n")
    spec = [
        {"name": "x", "kind": "scalar", "role": "input", "loader": "csv",
         "pattern": "t.csv", "column": "x", "index": True},
        {"name": "y", "kind": "label", "role": "gt", "loader": "csv",
         "pattern": "t.csv", "column": "y"},
    ]
    staging = str(tmp_path / "stage")
    materialize_file_tree(spec, ["t.csv"], repo.fetch, staging,
                          sample_offset=1, sample_cap=1)
    # offset=1 → only row 1 ('b'); its x must be 1, not 0.
    files = os.listdir(os.path.join(staging, "y"))
    assert len(files) == 1
    with open(os.path.join(staging, "y", files[0])) as fh:
        assert fh.read() == "b"
    with open(os.path.join(staging, "x", files[0])) as fh:
        assert fh.read() == "1"


# --------------------------------------------------------------------------
# detect_shape → materialize integration
# --------------------------------------------------------------------------

def test_detect_then_materialize_rle(repo, tmp_path):
    repo.write_img("train_images/a.png", 3, 2)
    repo.write_img("train_images/b.png", 3, 2)
    repo.write("train.csv", "ImageId,EncodedPixels\na.png,1 2\nb.png,3 2\n")
    files = ["train.csv", "train_images/a.png", "train_images/b.png"]
    peek = {"train.csv": {"columns": ["ImageId", "EncodedPixels"]}}
    res = detect_shape(files, peek=peek)
    assert res["shape"] == "I" and res["benchmarkable"]
    staging = str(tmp_path / "stage")
    summary = materialize_file_tree(res["spec"], files, repo.fetch, staging)
    assert summary["samples"] == 2
    m = _read_mask(staging, "mask", "a")
    assert (m[:, 0] == 1).all()
