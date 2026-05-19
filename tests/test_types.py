"""Round-trip + validation tests for benchhub.types.

Each `test_<kind>_roundtrip` builds an instance, encodes it to bytes,
decodes the bytes back into a new instance, and asserts the recovered
data matches. Validation tests assert that bad shapes/dtypes raise.
"""

from __future__ import annotations

import numpy as np
import pytest

import benchhub as bh
from benchhub.types import DTYPES, get_type


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lists_all_mvp_types():
    expected = {"image", "mask", "depth", "audio", "text", "bboxes", "label", "scalar", "json"}
    assert set(DTYPES) == expected


def test_get_type_returns_class():
    assert get_type("depth") is bh.Depth
    assert get_type("image") is bh.Image


def test_get_type_unknown_raises():
    with pytest.raises(KeyError):
        get_type("nope")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def test_image_rgb_roundtrip():
    arr = (np.random.default_rng(0).integers(0, 256, (32, 48, 3))).astype(np.uint8)
    img = bh.Image(arr)
    img.validate()
    recovered = bh.Image.decode(img.encode())
    np.testing.assert_array_equal(recovered.array, arr)


def test_image_grayscale_roundtrip():
    arr = (np.random.default_rng(0).integers(0, 256, (16, 24))).astype(np.uint8)
    img = bh.Image(arr)
    img.validate()
    recovered = bh.Image.decode(img.encode())
    np.testing.assert_array_equal(recovered.array, arr)


def test_image_rgba_roundtrip():
    arr = (np.random.default_rng(0).integers(0, 256, (8, 8, 4))).astype(np.uint8)
    img = bh.Image(arr)
    img.validate()
    recovered = bh.Image.decode(img.encode())
    np.testing.assert_array_equal(recovered.array, arr)


def test_image_validate_rejects_float():
    with pytest.raises(ValueError, match="uint8"):
        bh.Image(np.zeros((4, 4, 3), dtype=np.float32)).validate()


def test_image_validate_rejects_weird_shape():
    with pytest.raises(ValueError, match="must be"):
        bh.Image(np.zeros((4, 4, 5), dtype=np.uint8)).validate()


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def test_mask_uint8_roundtrip():
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    m = bh.Mask(arr, num_classes=6, ignore_index=255)
    m.validate()
    recovered = bh.Mask.decode(m.encode(), m.params)
    np.testing.assert_array_equal(recovered.array, arr)
    assert recovered.num_classes == 6
    assert recovered.ignore_index == 255


def test_mask_uint16_roundtrip():
    arr = np.array([[0, 300, 600], [1000, 2000, 3000]], dtype=np.uint16)
    m = bh.Mask(arr, num_classes=3001)
    recovered = bh.Mask.decode(m.encode(), m.params)
    np.testing.assert_array_equal(recovered.array, arr)


def test_mask_validate_rejects_3d():
    with pytest.raises(ValueError, match="\\(H,W\\)"):
        bh.Mask(np.zeros((4, 4, 3), dtype=np.uint8)).validate()


def test_mask_validate_rejects_float():
    with pytest.raises(ValueError, match="integer"):
        bh.Mask(np.zeros((4, 4), dtype=np.float32)).validate()


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------

def test_depth_roundtrip_meters():
    arr = np.array([[1.0, 2.0, np.nan], [3.5, 0.0, 5.25]], dtype=np.float32)
    d = bh.Depth(arr, unit="meters")
    d.validate()
    recovered = bh.Depth.decode(d.encode(), d.params)
    np.testing.assert_array_equal(recovered.array, arr)
    assert recovered.unit == "meters"


def test_depth_roundtrip_millimeters():
    arr = np.ones((4, 5), dtype=np.float32) * 1500
    d = bh.Depth(arr, unit="millimeters")
    recovered = bh.Depth.decode(d.encode(), d.params)
    assert recovered.unit == "millimeters"


def test_depth_unit_validation():
    with pytest.raises(ValueError, match="unit"):
        bh.Depth(np.zeros((4, 4), dtype=np.float32), unit="furlongs")


def test_depth_array_must_be_2d():
    with pytest.raises(ValueError, match="\\(H,W\\)"):
        bh.Depth(np.zeros((4, 4, 1), dtype=np.float32)).validate()


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def test_audio_mono_roundtrip():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a = bh.Audio(wav, sr)
    a.validate()
    recovered = bh.Audio.decode(a.encode(), a.params)
    assert recovered.sample_rate == sr
    np.testing.assert_allclose(recovered.waveform, wav, atol=1e-6)


def test_audio_validate_rejects_3d():
    with pytest.raises(ValueError, match="\\(T,"):
        bh.Audio(np.zeros((4, 4, 2), dtype=np.float32), 16000).validate()


def test_audio_validate_rejects_bad_sr():
    with pytest.raises(ValueError, match="sample_rate"):
        bh.Audio(np.zeros(100, dtype=np.float32), 0).validate()


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def test_text_roundtrip_ascii():
    t = bh.Text("hello world")
    assert bh.Text.decode(t.encode()).text == "hello world"


def test_text_roundtrip_unicode():
    s = "café — résumé 日本語"
    t = bh.Text(s)
    assert bh.Text.decode(t.encode()).text == s


# ---------------------------------------------------------------------------
# BBoxes
# ---------------------------------------------------------------------------

def test_bboxes_roundtrip_minimal():
    b = bh.BBoxes([[0.0, 1.0, 10.0, 20.0], [5.0, 5.0, 8.0, 9.0]])
    b.validate()
    r = bh.BBoxes.decode(b.encode(), b.params)
    np.testing.assert_allclose(r.boxes, b.boxes)
    assert r.format == "xyxy"
    assert r.labels is None
    assert r.scores is None


def test_bboxes_roundtrip_full():
    b = bh.BBoxes(
        [[0, 0, 1, 1], [2, 2, 3, 3]],
        labels=["cat", "dog"],
        scores=[0.9, 0.6],
        format="xywh",
    )
    r = bh.BBoxes.decode(b.encode(), b.params)
    assert r.format == "xywh"
    assert r.labels == ["cat", "dog"]
    np.testing.assert_allclose(r.scores, [0.9, 0.6])


def test_bboxes_empty_roundtrip():
    b = bh.BBoxes([])
    b.validate()
    r = bh.BBoxes.decode(b.encode(), b.params)
    assert r.boxes.shape == (0, 4)


def test_bboxes_bad_format():
    with pytest.raises(ValueError, match="format"):
        bh.BBoxes([[0, 0, 1, 1]], format="weird")


def test_bboxes_labels_length_mismatch():
    with pytest.raises(ValueError, match="labels length"):
        bh.BBoxes([[0, 0, 1, 1]], labels=["a", "b"]).validate()


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def test_label_int_roundtrip():
    assert bh.Label.decode(bh.Label(7).encode()).value == 7


def test_label_str_roundtrip():
    assert bh.Label.decode(bh.Label("cat").encode()).value == "cat"


def test_label_rejects_float():
    with pytest.raises(ValueError, match="int or str"):
        bh.Label(3.14)


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------

def test_scalar_roundtrip():
    s = bh.Scalar(3.14159)
    assert bh.Scalar.decode(s.encode()).value == pytest.approx(3.14159)


def test_scalar_accepts_int():
    s = bh.Scalar(42)
    assert s.value == 42.0
    assert isinstance(s.value, float)


# ---------------------------------------------------------------------------
# Json
# ---------------------------------------------------------------------------

def test_json_dict_roundtrip():
    payload = {"relations": [{"head": 0, "tail": 1, "type": "lives_in"}]}
    j = bh.Json(payload)
    j.validate()
    assert bh.Json.decode(j.encode()).data == payload


def test_json_list_roundtrip():
    j = bh.Json([1, "two", {"three": 3.0}])
    assert bh.Json.decode(j.encode()).data == [1, "two", {"three": 3.0}]


def test_json_rejects_non_serializable():
    with pytest.raises(ValueError, match="JSON-serializable"):
        bh.Json({"k": {1, 2, 3}}).validate()  # sets aren't JSON


# ---------------------------------------------------------------------------
# Type registry consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", list(DTYPES.values()))
def test_every_class_has_kind_string(cls):
    assert isinstance(cls.kind, str) and cls.kind


@pytest.mark.parametrize("cls", list(DTYPES.values()))
def test_every_class_declares_file_ext(cls):
    # Either a string starting with "." or None (inline).
    assert cls.file_ext is None or (
        isinstance(cls.file_ext, str) and cls.file_ext.startswith(".")
    )
