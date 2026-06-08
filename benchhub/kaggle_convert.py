"""Kaggle computer-vision conversion primitives.

Pure converters that turn Kaggle annotation encodings (RLE-in-CSV masks,
palette/legend masks, YOLO/VOC bboxes, per-instance mask stacks, 16-bit
depth PNGs) into the canonical in-memory form the staging layer
(`file_tree_import._stage_value`) writes:

  - Mask   → INTEGER label map (H,W), dtype uint8/uint16 (NOT RGB).
  - BBoxes → list of [x1,y1,x2,y2], abs pixels, 0-indexed (BBoxes format="xyxy").
  - Depth  → float32 (H,W) (caller saves .npz keyed `depth`).

numpy + PIL only, side-effect free → exhaustively unit-testable offline
(tests/test_kaggle_convert.py).

⚠️ #1 GOTCHA — the dominant Kaggle RLE convention (Airbus Ship, Carvana,
SIIM-ACR, TGS Salt, Severstal, Sartorius, HuBMAP) is COLUMN-MAJOR (Fortran
order), 1-indexed. Decoding C-order silently TRANSPOSES every mask — looks
plausible, IoU is garbage. decode_kaggle_rle defaults to order='F',
one_indexed=True for exactly this reason.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "parse_rle", "decode_kaggle_rle", "encode_kaggle_rle",
    "rle_rows_to_labelmap", "coco_uncompressed_rle_to_mask",
    "bbox_to_xyxy", "yolo_line_to_xyxy", "voc_box_to_xyxy",
    "palette_to_labelmap", "composite_instance_masks", "depth16_to_float",
    "downcast_labelmap",
]


def _is_empty_rle(v) -> bool:
    """True for the many ways Kaggle spells 'no mask': empty string, None,
    NaN, an empty/whitespace token, or a literal '-1'."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) == 0
    s = str(v).strip()
    return s == "" or s == "-1"


def parse_rle(rle) -> list[tuple[int, int]]:
    """RLE value → list of (start, length) int pairs. Accepts the CSV string
    form ("1 3 10 5"), a split list/ndarray, or an empty marker (→ []).
    Raises on an odd token count — a malformed RLE should fail loudly."""
    if _is_empty_rle(rle):
        return []
    if isinstance(rle, str):
        nums = [int(t) for t in rle.split()]
    else:
        nums = [int(t) for t in np.asarray(rle).ravel().tolist()]
    if len(nums) % 2 != 0:
        raise ValueError(
            f"RLE has an odd number of tokens ({len(nums)}); expected "
            f"alternating start/length pairs.")
    return list(zip(nums[0::2], nums[1::2]))


def decode_kaggle_rle(rle, height, width, *, order="F", one_indexed=True,
                      fill=1, dtype=np.uint8) -> np.ndarray:
    """Kaggle RLE → (H,W) binary mask (column-major, 1-indexed by default —
    see the module gotcha note)."""
    height, width = int(height), int(width)
    flat = np.zeros(height * width, dtype=dtype)
    for start, length in parse_rle(rle):
        s = (start - 1) if one_indexed else start
        if s < 0:
            length += s
            s = 0
        if length <= 0:
            continue
        flat[s:s + length] = fill
    return flat.reshape((height, width), order=order)


def encode_kaggle_rle(mask, *, order="F", one_indexed=True) -> str:
    """Inverse of decode_kaggle_rle: an (H,W) mask → 'start length …'. Any
    non-zero pixel is foreground; flatten order must match the decoder's."""
    flat = (np.asarray(mask).reshape(-1, order=order) != 0).astype(np.int8)
    padded = np.concatenate(([0], flat, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts
    out: list[int] = []
    for s, length in zip(starts, lengths):
        out.append(int(s) + (1 if one_indexed else 0))
        out.append(int(length))
    return " ".join(map(str, out))


def downcast_labelmap(arr) -> np.ndarray:
    """Smallest int dtype holding a label map: uint8 (≤255) else uint16.
    Mirrors benchhub.types.Mask.encode() so the staged PNG mode is consistent."""
    arr = np.asarray(arr)
    hi = int(arr.max(initial=0))
    return arr.astype(np.uint8) if hi <= 255 else arr.astype(np.uint16)


def rle_rows_to_labelmap(rows, height, width, *, value_key="EncodedPixels",
                         class_key=None, order="F", one_indexed=True,
                         overlap="last") -> np.ndarray:
    """Composite several RLE rows for ONE image into one integer label map
    (H,W). `rows` is a list of dicts (or (rle, class) tuples). Class id is
    int(row[class_key]) when class_key is given (semantic seg), else the
    row's 1-based ordinal (instance seg). Empty rows contribute nothing.
    `overlap`: 'last' (later row wins) or 'first' (keep earlier)."""
    height, width = int(height), int(width)
    out = np.zeros((height, width), dtype=np.int32)
    for idx, row in enumerate(rows):
        if isinstance(row, dict):
            rle = row.get(value_key)
            cls = int(row[class_key]) if class_key is not None else idx + 1
        else:
            rle = row[0]
            cls = int(row[1]) if class_key is not None else idx + 1
        if _is_empty_rle(rle):
            continue
        on = decode_kaggle_rle(rle, height, width, order=order,
                               one_indexed=one_indexed).astype(bool)
        if overlap == "first":
            on = on & (out == 0)
        out[on] = cls
    return downcast_labelmap(out)


def coco_uncompressed_rle_to_mask(counts, height, width, *,
                                  order="F") -> np.ndarray:
    """COCO uncompressed RLE (alternating run lengths, starting on a
    background run, column-major) → (H,W) binary mask. Compressed RLE
    (counts as a string) needs pycocotools and is out of scope."""
    height, width = int(height), int(width)
    counts = [int(c) for c in np.asarray(counts).ravel().tolist()]
    flat = np.zeros(height * width, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((height, width), order=order)


def bbox_to_xyxy(box, fmt, *, img_w=None, img_h=None) -> tuple:
    """One box in `fmt` → canonical (x1,y1,x2,y2), abs pixels, 0-indexed,
    matching benchhub.types.BBoxes(format='xyxy').

    fmt ∈ {xywh (COCO top-left+wh), xyxy, xyxy_voc (1-indexed inclusive →
    shift -1), cxcywh (abs center), cxcywh_norm (YOLO normalised center —
    needs img_w/img_h)}."""
    x, y, c, d = (float(v) for v in box)
    if fmt == "xywh":
        return x, y, x + c, y + d
    if fmt == "xyxy":
        return x, y, c, d
    if fmt == "xyxy_voc":
        return x - 1, y - 1, c - 1, d - 1
    if fmt == "cxcywh":
        return x - c / 2, y - d / 2, x + c / 2, y + d / 2
    if fmt == "cxcywh_norm":
        if img_w is None or img_h is None:
            raise ValueError("cxcywh_norm needs img_w and img_h to denormalise")
        cx, cy, w, h = x * img_w, y * img_h, c * img_w, d * img_h
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    raise ValueError(f"unknown bbox format {fmt!r}")


def yolo_line_to_xyxy(parts, img_w, img_h) -> tuple:
    """YOLO label line ("<class> <cx> <cy> <w> <h>", box values normalised)
    → (class_id, x1, y1, x2, y2) abs 0-indexed pixels. YOLO classes are
    already 0-indexed."""
    if isinstance(parts, str):
        parts = parts.split()
    if len(parts) != 5:
        raise ValueError(f"YOLO line needs 5 fields; got {len(parts)}: {parts!r}")
    cls = int(float(parts[0]))
    x1, y1, x2, y2 = bbox_to_xyxy(parts[1:], "cxcywh_norm",
                                  img_w=img_w, img_h=img_h)
    return cls, x1, y1, x2, y2


def voc_box_to_xyxy(xmin, ymin, xmax, ymax, *, one_indexed=True) -> tuple:
    """Pascal VOC <bndbox> corners → canonical 0-indexed xyxy. VOC corners
    are conventionally 1-indexed; pass one_indexed=False for 0-indexed sources."""
    shift = 1 if one_indexed else 0
    return (float(xmin) - shift, float(ymin) - shift,
            float(xmax) - shift, float(ymax) - shift)


def palette_to_labelmap(img, *, legend=None, max_colors=256) -> np.ndarray:
    """Palette/color-encoded mask → integer label map (H,W).

    - PIL mode 'P': pixel values ARE class indices → returned directly.
    - mode L/I or a 2-D array: already a label map → returned as-is.
    - RGB(A) ≤ max_colors colors: each color → a class id (via `legend`
      {(r,g,b): id} or [color, …]; else lexicographic color order).
    Raises if an RGB mask has > max_colors colors (likely a photo)."""
    mode = getattr(img, "mode", None)
    if mode == "P":
        return np.asarray(img, dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.int32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"palette mask must be 2-D or H×W×3/4; got {arr.shape}")
    rgb = arr[..., :3]
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)  # np<2 returns (N,1); flatten for indexing
    if legend is not None:
        lut = {}
        if isinstance(legend, dict):
            for color, cid in legend.items():
                lut[tuple(int(v) for v in color)] = int(cid)
        else:
            for cid, color in enumerate(legend):
                lut[tuple(int(v) for v in color)] = int(cid)
        ids = np.array([lut.get(tuple(int(v) for v in c), 0) for c in colors],
                       dtype=np.int32)
        out = ids[inverse].reshape(h, w)
    else:
        if len(colors) > max_colors:
            raise ValueError(
                f"RGB mask has {len(colors)} distinct colors (> {max_colors}); "
                f"this looks like a photo, not a segmentation mask.")
        out = inverse.reshape(h, w).astype(np.int32)
    return downcast_labelmap(out)


def composite_instance_masks(masks, *, overlap="last") -> np.ndarray:
    """Stack/list of per-instance BINARY masks → one instance-id label map
    (H,W): instance k (0-based in input order) → id k+1; 0 is background.
    `masks` may be a list of 2-D arrays or one (N,H,W) stack. `overlap`:
    'last' (later wins) or 'first' (earlier kept). Widens to uint16 >255."""
    stack = [np.asarray(m) for m in masks]
    if not stack:
        raise ValueError("composite_instance_masks needs at least one mask")
    if len(stack) == 1 and stack[0].ndim == 3:
        stack = list(stack[0])
    h, w = stack[0].shape[:2]
    out = np.zeros((h, w), dtype=np.int32)
    for k, m in enumerate(stack):
        on = np.asarray(m).astype(bool)
        if on.shape != (h, w):
            raise ValueError(f"instance {k} shape {on.shape} != base {(h, w)}")
        if overlap == "first":
            on = on & (out == 0)
        out[on] = k + 1
    return downcast_labelmap(out)


def depth16_to_float(img, *, scale=1.0) -> np.ndarray:
    """A 16-bit (or any-int) depth raster → float32 (H,W) in the declared
    unit. `scale` converts stored counts to the unit (e.g. NYU 16-bit PNGs
    use scale=1/1000 for mm→m). The caller declares the unit on the field."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"depth raster must be (H,W); got {arr.shape}")
    return (arr.astype(np.float32) * float(scale)).astype(np.float32)
