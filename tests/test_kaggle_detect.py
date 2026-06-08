"""Offline tests for benchhub.kaggle_detect — synthetic file lists + peeks,
no network or disk. Verifies the hidden-GT guard and that each Kaggle shape
emits a materialize_file_tree-compatible spec."""
import pytest

from benchhub.kaggle_detect import (
    detect_shape, partition_hidden_gt, looks_like_rle_columns,
)


# --------------------------------------------------------------------------
# Hidden-GT guard
# --------------------------------------------------------------------------

def test_partition_competition_hides_test_and_submission():
    files = ["train.csv", "test.csv", "sample_submission.csv",
             "train/0.png", "test/9.png"]
    usable, hidden = partition_hidden_gt(files)
    assert "train.csv" in usable and "train/0.png" in usable
    assert "test.csv" in hidden and "sample_submission.csv" in hidden
    assert "test/9.png" in hidden
    assert "test.csv" not in usable


def test_partition_no_competition_keeps_test():
    # No sample_submission and no train signal → 'test.csv' is a labelled split.
    files = ["data/test.csv", "data/extra.csv"]
    usable, hidden = partition_hidden_gt(files)
    assert hidden == []
    assert "data/test.csv" in usable


def test_partition_drops_junk():
    files = ["train.csv", ".DS_Store", "readme.md", "train/0.png"]
    usable, _ = partition_hidden_gt(files)
    assert ".DS_Store" not in usable and "readme.md" not in usable


def test_detect_rejects_all_hidden():
    res = detect_shape(["sample_submission.csv", "test.csv", "test/0.png"])
    assert res["benchmarkable"] is False
    assert "withheld" in res["reason"] or "test" in res["reason"].lower()


# --------------------------------------------------------------------------
# RLE column sniff
# --------------------------------------------------------------------------

def test_rle_columns_basic():
    hit = looks_like_rle_columns(["ImageId", "EncodedPixels"])
    assert hit == ("EncodedPixels", "ImageId", None)


def test_rle_columns_with_class():
    rle, idc, cls = looks_like_rle_columns(["ImageId", "ClassId", "EncodedPixels"])
    assert rle == "EncodedPixels" and idc == "ImageId" and cls == "ClassId"


def test_rle_columns_none():
    assert looks_like_rle_columns(["id", "feature", "label"]) is None


# --------------------------------------------------------------------------
# Shape I — RLE in CSV
# --------------------------------------------------------------------------

def test_detect_rle_csv():
    files = ["train.csv"] + [f"train_images/{i}.png" for i in range(5)]
    peek = {"train.csv": {"columns": ["ImageId", "EncodedPixels"]}}
    res = detect_shape(files, peek=peek)
    assert res["shape"] == "I"
    assert "rle" in res["needs_conversion"]
    mask = next(f for f in res["spec"] if f["kind"] == "mask")
    assert mask["loader"] == "rle" and mask["value_column"] == "EncodedPixels"
    assert mask["id_column"] == "ImageId"
    assert any(f["kind"] == "image" and f["role"] == "input" for f in res["spec"])


# --------------------------------------------------------------------------
# Shape E — COCO JSON
# --------------------------------------------------------------------------

def test_detect_coco():
    files = ["annotations/instances.json"] + \
            [f"images/{i}.jpg" for i in range(4)]
    peek = {"annotations/instances.json":
            {"keys": ["images", "annotations", "categories"]}}
    res = detect_shape(files, peek=peek)
    assert res["shape"] == "E"
    det = next(f for f in res["spec"] if f["loader"] == "coco")
    assert det["kind"] == "coco_detections" and det["role"] == "gt"


# --------------------------------------------------------------------------
# Shape F — VOC XML
# --------------------------------------------------------------------------

def test_detect_voc():
    files = [f"Annotations/{i}.xml" for i in range(3)] + \
            [f"JPEGImages/{i}.jpg" for i in range(3)]
    res = detect_shape(files)
    assert res["shape"] == "F"
    box = next(f for f in res["spec"] if f["loader"] == "voc")
    assert box["kind"] == "bboxes" and box["one_indexed"] is True


# --------------------------------------------------------------------------
# Shape G — YOLO TXT
# --------------------------------------------------------------------------

def test_detect_yolo():
    files = [f"labels/{i}.txt" for i in range(3)] + \
            [f"images/{i}.jpg" for i in range(3)] + ["classes.txt"]
    res = detect_shape(files)
    assert res["shape"] == "G"
    box = next(f for f in res["spec"] if f["loader"] == "yolo")
    assert box["names_file"] == "classes.txt"


# --------------------------------------------------------------------------
# Shape H — paired image+mask dirs
# --------------------------------------------------------------------------

def test_detect_paired_mask():
    files = [f"images/{i}.png" for i in range(5)] + \
            [f"masks/{i}.png" for i in range(5)]
    res = detect_shape(files)
    assert res["shape"] == "H"
    mask = next(f for f in res["spec"] if f["kind"] == "mask")
    assert mask["role"] == "gt" and mask["loader"] == "file"
    assert mask["pattern"] == "masks/{id}.png"
    img = next(f for f in res["spec"] if f["kind"] == "image")
    assert img["pattern"] == "images/{id}.png" and img["role"] == "input"


# --------------------------------------------------------------------------
# Shape N — restoration pairs
# --------------------------------------------------------------------------

def test_detect_restoration_pair():
    files = [f"noisy/{i}.png" for i in range(5)] + \
            [f"clean/{i}.png" for i in range(5)]
    res = detect_shape(files)
    assert res["shape"] == "N"
    # both image kind; the 'clean' side is gt.
    gt = next(f for f in res["spec"] if f["role"] == "gt")
    assert gt["kind"] == "image" and gt["pattern"] == "clean/{id}.png"
    inp = next(f for f in res["spec"] if f["role"] == "input")
    assert inp["pattern"] == "noisy/{id}.png"


# --------------------------------------------------------------------------
# Shape C — ImageFolder
# --------------------------------------------------------------------------

def test_detect_image_folder():
    files = [f"cats/{i}.jpg" for i in range(4)] + \
            [f"dogs/{i}.jpg" for i in range(4)]
    res = detect_shape(files)
    assert res["shape"] == "C"
    lbl = next(f for f in res["spec"] if f["kind"] == "label")
    assert lbl["loader"] == "token" and lbl["token"] == "label"
    img = next(f for f in res["spec"] if f["kind"] == "image")
    assert img["pattern"] == "{label}/{id}.jpg" and img["role"] == "input"


def test_image_folder_not_confused_with_modality_dirs():
    # image/ + mask/ are modality folders, not class labels.
    files = [f"image/{i}.png" for i in range(3)] + \
            [f"mask/{i}.png" for i in range(3)]
    res = detect_shape(files)
    assert res["shape"] != "C"          # should be H (paired)
    assert res["shape"] == "H"


# --------------------------------------------------------------------------
# Shape D — image + CSV
# --------------------------------------------------------------------------

def test_detect_image_plus_csv():
    files = ["labels.csv"] + [f"images/{i}.jpg" for i in range(5)]
    peek = {"labels.csv": {"columns": ["filename", "label"],
                           "rows": [{"filename": "0.jpg", "label": "cat"},
                                    {"filename": "1.jpg", "label": "dog"}]}}
    res = detect_shape(files, peek=peek)
    assert res["shape"] == "D"
    gt = next(f for f in res["spec"] if f["role"] == "gt")
    assert gt["loader"] == "csv" and gt["column"] == "label"
    assert gt["id_column"] == "filename" and gt["kind"] == "label"


# --------------------------------------------------------------------------
# Shapes A / A′ / A″ — tabular
# --------------------------------------------------------------------------

def test_detect_tabular_classification():
    peek = {"train.csv": {"columns": ["id", "age", "income", "label"],
                          "rows": [{"id": "1", "age": "30", "income": "50000",
                                    "label": "yes"},
                                   {"id": "2", "age": "40", "income": "60000",
                                    "label": "no"}]}}
    res = detect_shape(["train.csv"], peek=peek)
    assert res["shape"] == "A"
    gt = next(f for f in res["spec"] if f["role"] == "gt")
    assert gt["column"] == "label" and gt["kind"] == "label"
    # an input field carries the row index; id column is dropped.
    inputs = [f for f in res["spec"] if f["role"] == "input"]
    assert inputs and any(f.get("index") for f in res["spec"])
    assert all(f["column"] != "id" for f in res["spec"])


def test_detect_tabular_regression():
    peek = {"train.csv": {"columns": ["sqft", "price"],
                          "rows": [{"sqft": "1000", "price": "250000.5"},
                                   {"sqft": "2000", "price": "480000.25"}]}}
    res = detect_shape(["train.csv"], peek=peek)
    assert res["shape"] == "A_prime"
    gt = next(f for f in res["spec"] if f["role"] == "gt")
    assert gt["kind"] == "scalar"


def test_detect_text_classification():
    peek = {"train.csv": {"columns": ["review", "sentiment"],
                          "rows": [{"review": "x" * 80, "sentiment": "pos"},
                                   {"review": "y" * 80, "sentiment": "neg"}]}}
    res = detect_shape(["train.csv"], peek=peek)
    assert res["shape"] == "A_dprime"
    txt = next(f for f in res["spec"] if f["name"] == "review")
    assert txt["kind"] == "text" and txt["role"] == "input"


def test_tabular_single_table_gets_index():
    peek = {"data.csv": {"columns": ["text", "label"],
                         "rows": [{"text": "hi", "label": "a"}]}}
    res = detect_shape(["data.csv"], peek=peek)
    assert any(f.get("index") for f in res["spec"])


# --------------------------------------------------------------------------
# Shape O — DICOM/NIfTI not benchmarkable yet
# --------------------------------------------------------------------------

def test_detect_dicom_not_benchmarkable():
    files = [f"scans/{i}.dcm" for i in range(4)]
    res = detect_shape(files)
    assert res["shape"] == "O"
    assert res["benchmarkable"] is False


# --------------------------------------------------------------------------
# Unrecognised
# --------------------------------------------------------------------------

def test_unrecognised():
    res = detect_shape(["weird.bin", "other.xyz"])
    assert res["benchmarkable"] is False
    assert res["shape"] is None
