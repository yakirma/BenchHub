"""Offline unit tests for benchhub.kaggle_convert — the Kaggle CV
conversion primitives. No network, no kaggle package, pure numpy/PIL."""
import numpy as np
import pytest

from benchhub.kaggle_convert import (
    parse_rle, decode_kaggle_rle, encode_kaggle_rle, rle_rows_to_labelmap,
    coco_uncompressed_rle_to_mask, bbox_to_xyxy, yolo_line_to_xyxy,
    voc_box_to_xyxy, palette_to_labelmap, composite_instance_masks,
    depth16_to_float, downcast_labelmap,
)


# --------------------------------------------------------------------------
# parse_rle
# --------------------------------------------------------------------------

def test_parse_rle_string():
    assert parse_rle("1 3 10 5") == [(1, 3), (10, 5)]


def test_parse_rle_list_and_array():
    assert parse_rle([1, 3, 10, 5]) == [(1, 3), (10, 5)]
    assert parse_rle(np.array([1, 3, 10, 5])) == [(1, 3), (10, 5)]


@pytest.mark.parametrize("empty", ["", "   ", None, float("nan"), "-1", [], np.array([])])
def test_parse_rle_empty(empty):
    assert parse_rle(empty) == []


def test_parse_rle_odd_tokens_raises():
    with pytest.raises(ValueError):
        parse_rle("1 3 10")


# --------------------------------------------------------------------------
# decode_kaggle_rle — the column-major gotcha
# --------------------------------------------------------------------------

def test_decode_single_column_run_is_column_major():
    # H=3, W=4. RLE "1 3" (1-indexed) fills flat[0:3] → the first COLUMN.
    m = decode_kaggle_rle("1 3", 3, 4)
    expected = np.zeros((3, 4), dtype=np.uint8)
    expected[:, 0] = 1
    np.testing.assert_array_equal(m, expected)


def test_decode_column_major_differs_from_c_order():
    # The same RLE under C-order is the transposed (wrong) interpretation.
    f = decode_kaggle_rle("1 3", 3, 4, order="F")
    c = decode_kaggle_rle("1 3", 3, 4, order="C")
    assert not np.array_equal(f, c), "F and C order must differ on a non-square run"
    # C-order "1 3" fills the first ROW of the (3,4)... actually flat[0:3]
    # in C-order lands in row 0 cols 0,1,2.
    expected_c = np.zeros((3, 4), dtype=np.uint8)
    expected_c[0, 0:3] = 1
    np.testing.assert_array_equal(c, expected_c)


def test_decode_one_indexed_vs_zero_indexed():
    one = decode_kaggle_rle("1 2", 2, 2, one_indexed=True)   # flat[0:2]
    zero = decode_kaggle_rle("1 2", 2, 2, one_indexed=False)  # flat[1:3]
    assert one[0, 0] == 1 and one[1, 0] == 1
    assert zero[1, 0] == 1 and zero[0, 1] == 1


def test_decode_empty_is_all_zero():
    np.testing.assert_array_equal(decode_kaggle_rle("", 5, 7), np.zeros((5, 7)))
    np.testing.assert_array_equal(decode_kaggle_rle(float("nan"), 5, 7),
                                  np.zeros((5, 7)))


# --------------------------------------------------------------------------
# round-trip encode <-> decode (the real correctness proof)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3, 4), (1, 10), (10, 1), (16, 9), (101, 101)])
def test_rle_roundtrip_random_masks(shape):
    rng = np.random.default_rng(0)
    h, w = shape
    mask = (rng.random((h, w)) > 0.6).astype(np.uint8)
    rle = encode_kaggle_rle(mask)
    back = decode_kaggle_rle(rle, h, w)
    np.testing.assert_array_equal(mask, back)


def test_rle_roundtrip_known_fixture():
    # A hand-built mask; verify the exact RLE string is column-major.
    mask = np.zeros((4, 3), dtype=np.uint8)
    mask[1, 0] = 1            # col-major flat index 1
    mask[2, 0] = 1            # flat index 2
    mask[0, 2] = 1            # col 2 starts at flat index 8
    # column-major flatten: col0=[0,1,1,0], col1=[0,0,0,0], col2=[1,0,0,0]
    # runs (1-indexed): start 2 len 2, start 9 len 1
    assert encode_kaggle_rle(mask) == "2 2 9 1"
    np.testing.assert_array_equal(decode_kaggle_rle("2 2 9 1", 4, 3), mask)


def test_rle_full_mask():
    mask = np.ones((5, 5), dtype=np.uint8)
    assert encode_kaggle_rle(mask) == "1 25"


# --------------------------------------------------------------------------
# rle_rows_to_labelmap
# --------------------------------------------------------------------------

def test_rle_rows_instance_ids_by_order():
    rows = [{"EncodedPixels": "1 2"}, {"EncodedPixels": "5 2"}]
    lm = rle_rows_to_labelmap(rows, 2, 4)
    # row0 → id 1 at flat 0..1 (col0), row1 → id 2 at flat 4..5 (col2)
    assert lm[0, 0] == 1 and lm[1, 0] == 1
    assert lm[0, 2] == 2 and lm[1, 2] == 2


def test_rle_rows_class_key():
    rows = [{"EncodedPixels": "1 2", "ClassId": 3},
            {"EncodedPixels": "5 2", "ClassId": 7}]
    lm = rle_rows_to_labelmap(rows, 2, 4, class_key="ClassId")
    assert set(np.unique(lm)) == {0, 3, 7}


def test_rle_rows_overlap_last_vs_first():
    rows = [{"EncodedPixels": "1 4"}, {"EncodedPixels": "1 4"}]
    last = rle_rows_to_labelmap(rows, 2, 2, overlap="last")
    first = rle_rows_to_labelmap(rows, 2, 2, overlap="first")
    assert last.max() == 2 and (last == 2).all()
    assert first.max() == 1 and (first == 1).all()


def test_rle_rows_skips_empty():
    rows = [{"EncodedPixels": ""}, {"EncodedPixels": "1 2"}]
    lm = rle_rows_to_labelmap(rows, 2, 2)
    # empty row contributes nothing; second row still gets its ordinal id 2
    assert set(np.unique(lm)) == {0, 2}


def test_rle_rows_tuple_form():
    # 3x2 (6 px) so the last 2 px stay background: row0 → id 5 at flat 0..1,
    # row1 → id 9 at flat 2..3, flat 4..5 untouched.
    rows = [("1 2", 5), ("3 2", 9)]
    lm = rle_rows_to_labelmap(rows, 3, 2, class_key=1)
    assert set(np.unique(lm)) == {0, 5, 9}


# --------------------------------------------------------------------------
# COCO uncompressed RLE
# --------------------------------------------------------------------------

def test_coco_uncompressed_rle():
    # 2x2, counts [1,2,1] col-major: bg 1, fg 2, bg 1 → flat [0,1,1,0]
    m = coco_uncompressed_rle_to_mask([1, 2, 1], 2, 2)
    expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # col-major fill
    np.testing.assert_array_equal(m, expected)


def test_coco_starts_background():
    # counts [0,4] → no bg, all fg
    m = coco_uncompressed_rle_to_mask([0, 4], 2, 2)
    assert m.sum() == 4


# --------------------------------------------------------------------------
# bbox conversions
# --------------------------------------------------------------------------

def test_bbox_xywh():
    assert bbox_to_xyxy([10, 20, 30, 40], "xywh") == (10, 20, 40, 60)


def test_bbox_xyxy_passthrough():
    assert bbox_to_xyxy([1, 2, 3, 4], "xyxy") == (1, 2, 3, 4)


def test_bbox_voc_one_indexed():
    assert bbox_to_xyxy([1, 1, 100, 200], "xyxy_voc") == (0, 0, 99, 199)


def test_bbox_cxcywh():
    assert bbox_to_xyxy([50, 50, 20, 10], "cxcywh") == (40, 45, 60, 55)


def test_bbox_cxcywh_norm_yolo():
    # center (0.5,0.5), wh (0.5,0.5) on a 100x200 image → x 25..75, y 50..150
    x1, y1, x2, y2 = bbox_to_xyxy([0.5, 0.5, 0.5, 0.5], "cxcywh_norm",
                                  img_w=100, img_h=200)
    assert (x1, y1, x2, y2) == (25.0, 50.0, 75.0, 150.0)


def test_bbox_norm_requires_dims():
    with pytest.raises(ValueError):
        bbox_to_xyxy([0.5, 0.5, 0.5, 0.5], "cxcywh_norm")


def test_bbox_unknown_format():
    with pytest.raises(ValueError):
        bbox_to_xyxy([1, 2, 3, 4], "nonsense")


def test_yolo_line():
    cls, x1, y1, x2, y2 = yolo_line_to_xyxy("2 0.5 0.5 0.5 0.5", 100, 200)
    assert cls == 2
    assert (x1, y1, x2, y2) == (25.0, 50.0, 75.0, 150.0)


def test_yolo_line_bad_field_count():
    with pytest.raises(ValueError):
        yolo_line_to_xyxy("2 0.5 0.5", 100, 100)


def test_voc_box():
    assert voc_box_to_xyxy(1, 1, 10, 20) == (0, 0, 9, 19)
    assert voc_box_to_xyxy(1, 1, 10, 20, one_indexed=False) == (1, 1, 10, 20)


# --------------------------------------------------------------------------
# palette / legend masks
# --------------------------------------------------------------------------

def test_palette_mode_p():
    from PIL import Image
    arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    img = Image.fromarray(arr, mode="P")
    np.testing.assert_array_equal(palette_to_labelmap(img), arr)


def test_palette_grayscale_array_passthrough():
    arr = np.array([[0, 5], [9, 2]], dtype=np.uint8)
    np.testing.assert_array_equal(palette_to_labelmap(arr), arr)


def test_palette_rgb_no_legend_lexicographic():
    # two colors → ids 0 and 1 in lexicographic order (black < red)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, :] = (255, 0, 0)
    lm = palette_to_labelmap(rgb)
    assert lm.shape == (2, 2)
    assert set(np.unique(lm)) == {0, 1}
    # black (0,0,0) is lexicographically smallest → id 0
    assert lm[1, 0] == 0 and lm[0, 0] == 1


def test_palette_rgb_with_legend():
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (0, 255, 0)
    rgb[0, 2] = (0, 0, 0)
    legend = {(0, 0, 0): 0, (255, 0, 0): 7, (0, 255, 0): 9}
    lm = palette_to_labelmap(rgb, legend=legend)
    np.testing.assert_array_equal(lm[0], np.array([7, 9, 0]))


def test_palette_too_many_colors_raises():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        palette_to_labelmap(rgb, max_colors=32)


# --------------------------------------------------------------------------
# instance compositing
# --------------------------------------------------------------------------

def test_composite_instance_masks_basic():
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[0, 0], [1, 1]])
    lm = composite_instance_masks([a, b])
    expected = np.array([[1, 0], [2, 2]], dtype=np.uint8)
    np.testing.assert_array_equal(lm, expected)


def test_composite_overlap_last_and_first():
    a = np.ones((2, 2))
    b = np.ones((2, 2))
    assert (composite_instance_masks([a, b], overlap="last") == 2).all()
    assert (composite_instance_masks([a, b], overlap="first") == 1).all()


def test_composite_from_stack():
    stack = np.zeros((3, 2, 2))
    stack[0, 0, 0] = 1
    stack[1, 1, 1] = 1
    lm = composite_instance_masks(stack)
    assert lm[0, 0] == 1 and lm[1, 1] == 2


def test_composite_widens_to_uint16():
    masks = [np.zeros((1, 300), dtype=np.uint8) for _ in range(300)]
    for i, m in enumerate(masks):
        m[0, i] = 1
    lm = composite_instance_masks(masks)
    assert lm.dtype == np.uint16
    assert lm.max() == 300


def test_composite_shape_mismatch_raises():
    with pytest.raises(ValueError):
        composite_instance_masks([np.zeros((2, 2)), np.zeros((3, 3))])


def test_composite_empty_raises():
    with pytest.raises(ValueError):
        composite_instance_masks([])


# --------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------

def test_depth16_scale():
    raw = np.array([[1000, 2000], [3000, 0]], dtype=np.uint16)
    d = depth16_to_float(raw, scale=1 / 1000)
    assert d.dtype == np.float32
    np.testing.assert_allclose(d, [[1.0, 2.0], [3.0, 0.0]])


def test_depth16_squeeze_channel():
    raw = np.zeros((4, 5, 1), dtype=np.uint16)
    assert depth16_to_float(raw).shape == (4, 5)


def test_depth16_bad_ndim_raises():
    with pytest.raises(ValueError):
        depth16_to_float(np.zeros((2, 2, 3)))


# --------------------------------------------------------------------------
# downcast
# --------------------------------------------------------------------------

def test_downcast_uint8_vs_uint16():
    assert downcast_labelmap(np.array([0, 255])).dtype == np.uint8
    assert downcast_labelmap(np.array([0, 256])).dtype == np.uint16
