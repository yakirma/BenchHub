"""Strict-typed data classes for BenchHub.

One class per kind of data (image, depth, mask, audio, text, bboxes,
label, scalar, json). Each subclass of `DataType` knows how to:

- hold its data in memory (the `.array` / `.text` / `.value` attribute);
- serialize itself to bytes via `encode()` (PNG, NPZ, WAV, JSON, TXT);
- deserialize from bytes + per-instance params via `decode()`;
- validate its own shape / dtype in `validate()`.

Per-instance metadata that the LB needs to know about (depth unit, bbox
format, mask `ignore_index`) lives in `.params` — a small dict that
travels alongside the blob through storage and re-emerges when the file
is decoded.

The registry `DTYPES` maps wire-kind strings ("depth", "image", ...) to
the concrete class. `get_type("depth")` is the public lookup.
"""

from __future__ import annotations

import io
import json
from typing import Any, ClassVar

import numpy as np
from PIL import Image as PILImage


DTYPES: dict[str, type["DataType"]] = {}


class DataType:
    """Abstract base. Subclasses register themselves in `DTYPES`."""

    kind: ClassVar[str]
    file_ext: ClassVar[str | None]  # None ⇒ inline storage (SQLite), not a file
    # MIME type for the default `visualize()` output. Subclasses that
    # override visualize() should set this so the dispatch route's
    # Content-Type header matches the bytes.
    viz_mime: ClassVar[str] = "application/octet-stream"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "kind", None):
            DTYPES[cls.kind] = cls

    @property
    def params(self) -> dict:
        """Per-instance metadata that travels with the data. Override."""
        return {}

    def encode(self) -> bytes:
        """Serialize to bytes for on-disk (or inline blob) storage."""
        raise NotImplementedError

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "DataType":
        """Deserialize from bytes + per-instance params."""
        raise NotImplementedError

    def validate(self) -> None:
        """Raise ValueError on bad shape/dtype/range."""
        return None

    def visualize(self, **opts: Any) -> tuple[bytes, str]:
        """Render this instance for display. Returns `(bytes, mime_type)`.

        The default implementation returns `encode()` + `viz_mime`,
        which works for kinds whose on-disk format is already viewer-
        friendly (PNG for Image/Mask, WAV for Audio, JSON for Json,
        plain text for Scalar/Text/Label). Subclasses override when
        the storage form isn't directly renderable — `Depth.visualize`
        applies a colormap, `BBoxes.visualize` returns SVG.

        `opts` are renderer params from the dispatch route's query
        string (`?cmap=turbo`, `?width=512`, …); subclasses that
        accept them declare each as a keyword arg.
        """
        return self.encode(), self.viz_mime

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params!r})"


# ---------------------------------------------------------------------------
# Image — RGB(A) photos or grayscale, uint8.
# ---------------------------------------------------------------------------

class Image(DataType):
    kind = "image"
    file_ext = ".png"
    viz_mime = "image/png"

    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)

    def encode(self) -> bytes:
        buf = io.BytesIO()
        PILImage.fromarray(self.array).save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Image":
        img = PILImage.open(io.BytesIO(blob))
        # Force a deterministic mode: P (palettized) -> RGB to keep submissions
        # consistent; Mask is what should be palettized.
        if img.mode == "P":
            img = img.convert("RGB")
        return cls(np.asarray(img))

    def validate(self) -> None:
        a = self.array
        if a.dtype != np.uint8:
            raise ValueError(f"Image must be uint8; got {a.dtype}")
        if a.ndim == 2:
            return
        if a.ndim == 3 and a.shape[2] in (3, 4):
            return
        raise ValueError(f"Image must be (H,W), (H,W,3), or (H,W,4); got {a.shape}")


# ---------------------------------------------------------------------------
# Mask — integer label map, (H,W).
# ---------------------------------------------------------------------------

class Mask(DataType):
    kind = "mask"
    file_ext = ".png"
    viz_mime = "image/png"

    def __init__(
        self,
        array: np.ndarray,
        *,
        num_classes: int | None = None,
        ignore_index: int = 255,
    ):
        self.array = np.asarray(array)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    @property
    def params(self) -> dict:
        return {"num_classes": self.num_classes, "ignore_index": self.ignore_index}

    def encode(self) -> bytes:
        a = self.array
        buf = io.BytesIO()
        if np.issubdtype(a.dtype, np.integer) and int(a.max(initial=0)) <= 255 and int(a.min(initial=0)) >= 0:
            PILImage.fromarray(a.astype(np.uint8)).save(buf, format="PNG")
        else:
            PILImage.fromarray(a.astype(np.uint16)).save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Mask":
        params = params or {}
        img = PILImage.open(io.BytesIO(blob))
        if img.mode == "I;16":
            array = np.asarray(img, dtype=np.uint16)
        elif img.mode in ("L", "P"):
            array = np.asarray(img, dtype=np.uint8)
        else:
            raise ValueError(f"Mask PNG must be mode L/P/I;16; got {img.mode!r}")
        return cls(
            array,
            num_classes=params.get("num_classes"),
            ignore_index=params.get("ignore_index", 255),
        )

    def validate(self) -> None:
        a = self.array
        if a.ndim != 2:
            raise ValueError(f"Mask must be (H,W); got {a.shape}")
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"Mask must be integer dtype; got {a.dtype}")


# ---------------------------------------------------------------------------
# Depth — float32 (H,W), with a unit declared at the LB level.
# ---------------------------------------------------------------------------

class Depth(DataType):
    kind = "depth"
    file_ext = ".npz"
    viz_mime = "image/png"
    UNITS: ClassVar[set[str]] = {"meters", "millimeters", "unitless"}

    def __init__(self, array: np.ndarray, *, unit: str = "meters"):
        if unit not in self.UNITS:
            raise ValueError(f"unit must be one of {sorted(self.UNITS)}; got {unit!r}")
        self.array = np.asarray(array, dtype=np.float32)
        self.unit = unit

    @property
    def params(self) -> dict:
        return {"unit": self.unit}

    def encode(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, depth=self.array)
        return buf.getvalue()

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Depth":
        params = params or {}
        with np.load(io.BytesIO(blob)) as data:
            array = np.asarray(data["depth"], dtype=np.float32)
        return cls(array, unit=params.get("unit", "meters"))

    def validate(self) -> None:
        a = self.array
        if a.ndim != 2:
            raise ValueError(f"Depth must be (H,W); got {a.shape}")
        if a.dtype != np.float32:
            raise ValueError(f"Depth must be float32; got {a.dtype}")

    def visualize(self, *, cmap: str = "turbo", **_) -> tuple[bytes, str]:
        """Normalize finite depths to [0,1], apply a colormap, return
        PNG bytes. `cmap` accepts any matplotlib colormap name; when
        matplotlib isn't importable we fall back to plain grayscale
        so the client lib still has a renderable representation."""
        arr = self.array
        finite = np.isfinite(arr)
        if not finite.any():
            buf = io.BytesIO()
            PILImage.new("RGB", (16, 16), (0, 0, 0)).save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        lo = float(arr[finite].min())
        hi = float(arr[finite].max())
        norm = np.zeros_like(arr, dtype=np.float32)
        if hi > lo:
            norm[finite] = (arr[finite] - lo) / (hi - lo)
        try:
            import matplotlib
            # matplotlib.cm.get_cmap is deprecated and slated for 3.11
            # removal; the supported API is `matplotlib.colormaps[name]`,
            # which raises KeyError on unknown names. We catch via the
            # outer Exception to fall back to grayscale either way.
            cm = matplotlib.colormaps[cmap]
            rgba = cm(norm)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
        except Exception:
            gray = (norm * 255).astype(np.uint8)
            rgb = np.stack([gray] * 3, axis=-1)
        # Where the original was non-finite, fill with black so NaN
        # regions are visually distinct.
        rgb[~finite] = 0
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue(), "image/png"


# ---------------------------------------------------------------------------
# Audio — 1D or (T, channels) float32 waveform + sample rate.
# ---------------------------------------------------------------------------

class Audio(DataType):
    kind = "audio"
    file_ext = ".wav"
    viz_mime = "audio/wav"

    def __init__(self, waveform: np.ndarray, sample_rate: int):
        self.waveform = np.asarray(waveform, dtype=np.float32)
        self.sample_rate = int(sample_rate)

    @property
    def params(self) -> dict:
        return {"sample_rate": self.sample_rate}

    def encode(self) -> bytes:
        import soundfile as sf  # heavy dep; lazy import keeps cold-start cheap
        buf = io.BytesIO()
        sf.write(buf, self.waveform, self.sample_rate, format="WAV", subtype="FLOAT")
        return buf.getvalue()

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Audio":
        import soundfile as sf
        waveform, sample_rate = sf.read(io.BytesIO(blob), dtype="float32")
        return cls(waveform, sample_rate)

    def validate(self) -> None:
        if self.waveform.ndim not in (1, 2):
            raise ValueError(f"Audio waveform must be (T,) or (T,C); got {self.waveform.shape}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive; got {self.sample_rate}")


# ---------------------------------------------------------------------------
# Text — UTF-8 string.
# ---------------------------------------------------------------------------

class Text(DataType):
    kind = "text"
    file_ext = ".txt"
    viz_mime = "text/plain; charset=utf-8"

    def __init__(self, text: str):
        self.text = str(text)

    def encode(self) -> bytes:
        return self.text.encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Text":
        return cls(blob.decode("utf-8"))

    def validate(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError(f"Text must be a string; got {type(self.text)}")


# ---------------------------------------------------------------------------
# BBoxes — list of (x1,y1,x2,y2) (or alternative format) + optional labels/scores.
# ---------------------------------------------------------------------------

class BBoxes(DataType):
    kind = "bboxes"
    file_ext = ".json"
    viz_mime = "image/svg+xml"
    FORMATS: ClassVar[set[str]] = {"xyxy", "xywh", "cxcywh"}

    def __init__(
        self,
        boxes: np.ndarray | list,
        *,
        labels: list | None = None,
        scores: np.ndarray | list | None = None,
        format: str = "xyxy",
    ):
        if format not in self.FORMATS:
            raise ValueError(f"format must be one of {sorted(self.FORMATS)}; got {format!r}")
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            arr = arr.reshape(0, 4)
        self.boxes = arr
        self.labels = list(labels) if labels is not None else None
        self.scores = np.asarray(scores, dtype=np.float32) if scores is not None else None
        self.format = format

    @property
    def params(self) -> dict:
        return {"format": self.format}

    def encode(self) -> bytes:
        payload: dict[str, Any] = {
            "boxes": self.boxes.tolist(),
            "format": self.format,
        }
        if self.labels is not None:
            payload["labels"] = self.labels
        if self.scores is not None:
            payload["scores"] = self.scores.tolist()
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "BBoxes":
        data = json.loads(blob.decode("utf-8"))
        boxes = data.get("boxes", [])
        scores = data.get("scores")
        return cls(
            boxes,
            labels=data.get("labels"),
            scores=scores,
            format=data.get("format", "xyxy"),
        )

    def validate(self) -> None:
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValueError(f"boxes must be (N,4); got {self.boxes.shape}")
        if self.labels is not None and len(self.labels) != len(self.boxes):
            raise ValueError(
                f"labels length {len(self.labels)} != boxes count {len(self.boxes)}"
            )
        if self.scores is not None and len(self.scores) != len(self.boxes):
            raise ValueError(
                f"scores length {len(self.scores)} != boxes count {len(self.boxes)}"
            )

    def _to_xyxy(self, box) -> tuple[float, float, float, float]:
        x, y, c, d = (float(v) for v in box)
        if self.format == "xywh":
            return x, y, x + c, y + d
        if self.format == "cxcywh":
            return x - c / 2, y - d / 2, x + c / 2, y + d / 2
        return x, y, c, d

    def visualize(self, *, width: int = 256, height: int = 256, **_) -> tuple[bytes, str]:
        """Render the boxes as SVG. No background image — callers that
        want boxes-on-image should layer the SVG over the Image
        visualization at render time. Each box gets a distinct hue so
        crowded scenes are still readable; an optional `labels` list
        is dropped as a small text label next to each box."""
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError):
            width, height = 256, 256
        parts: list[str] = []
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._to_xyxy(box)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            hue = (i * 47) % 360  # pseudo-random spread; deterministic
            stroke = f"hsl({hue},80%,50%)"
            parts.append(
                f'<rect x="{x1:.2f}" y="{y1:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" '
                f'fill="none" stroke="{stroke}" stroke-width="1.5"/>'
            )
            if self.labels is not None and i < len(self.labels):
                tag = str(self.labels[i]).replace("<", "&lt;").replace(">", "&gt;")
                parts.append(
                    f'<text x="{x1:.2f}" y="{(y1 - 2):.2f}" '
                    f'fill="{stroke}" font-size="10" '
                    f'font-family="ui-sans-serif, system-ui">{tag}</text>'
                )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
            + "".join(parts)
            + "</svg>"
        )
        return svg.encode("utf-8"), self.viz_mime


# ---------------------------------------------------------------------------
# Label — single class (int or string); vocab declared at LB level.
# ---------------------------------------------------------------------------

class Label(DataType):
    kind = "label"
    file_ext = None  # stored inline
    viz_mime = "text/plain; charset=utf-8"

    def __init__(self, value: int | str, *, names: list[str] | None = None):
        if not isinstance(value, (int, str)):
            raise ValueError(f"Label value must be int or str; got {type(value).__name__}")
        self.value = value
        # Optional class vocab carried on the field's params (same as
        # LabelList). Comparison metrics use `.value`; `names` is just
        # for display / human-readable rendering. Accepting it here is
        # what lets a label field with a `names` vocab round-trip through
        # the typed-instance builder (`cls(value, **params)`).
        self.names = list(names) if names else None

    @property
    def params(self) -> dict:
        return {"names": self.names} if self.names else {}

    def encode(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Label":
        params = params or {}
        return cls(json.loads(blob.decode("utf-8")), names=params.get("names"))

    def visualize(self, **_: Any) -> tuple[bytes, str]:
        """Render the label as the bare value (not JSON-quoted) so the
        comparison view shows `cat` rather than `"cat"`."""
        return str(self.value).encode("utf-8"), self.viz_mime


# ---------------------------------------------------------------------------
# LabelList — ranked list of class predictions (top-K). Same vocab
# semantics as Label, but the .values list holds K entries in
# descending-confidence order. The dataset declares the expected K
# via params['k']; the LB binding's metric (top_1 / top_5) chooses
# how deep to slice. Stored inline as a JSON-encoded list.
# ---------------------------------------------------------------------------

class LabelList(DataType):
    kind = "label_list"
    file_ext = None  # stored inline
    viz_mime = "text/plain; charset=utf-8"

    def __init__(self, values: list[int | str], *, k: int,
                 names: list[str] | None = None):
        """`k` is REQUIRED: every LabelList instance carries its
        declared top-K size. The contract is exact — the values
        list must be exactly k long (no padding, no truncation).
        That keeps top-1 vs top-5 unambiguous and lets metrics
        slice `values[:k]` with confidence."""
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            raise ValueError(f"LabelList requires an integer k; got {k!r}")
        if k_int < 1:
            raise ValueError(f"LabelList k must be >= 1; got {k_int}")
        for v in values:
            if not isinstance(v, (int, str)):
                raise ValueError(
                    f"LabelList values must be int or str; got {type(v).__name__}"
                )
        self.values = list(values)
        self.names = list(names) if names else None
        self.k = k_int

    @property
    def params(self) -> dict:
        out: dict[str, Any] = {"k": self.k}
        if self.names:
            out["names"] = self.names
        return out

    def encode(self) -> bytes:
        return json.dumps(self.values).encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "LabelList":
        params = params or {}
        if "k" not in params:
            raise ValueError(
                "LabelList.decode requires `k` in params — the dataset "
                "field must declare top-K size before submissions can "
                "be evaluated against it."
            )
        text = blob.decode("utf-8") if blob else "[]"
        values = json.loads(text) if text.strip() else []
        if not isinstance(values, list):
            raise ValueError(
                f"LabelList blob must decode to a JSON list; got {type(values).__name__}"
            )
        return cls(values, k=params["k"], names=params.get("names"))

    def validate(self) -> None:
        # Exact-length contract: `pred.values` must hold exactly k
        # entries. Sub- or over-length submissions fail the manifest
        # check before any metric runs.
        if len(self.values) != self.k:
            raise ValueError(
                f"LabelList has {len(self.values)} values; declared k={self.k} (exact match required)"
            )

    def visualize(self, **_: Any) -> tuple[bytes, str]:
        """Render as `<idx> <name>, <idx> <name>, …` if a vocab is
        attached; otherwise just the comma-joined values."""
        if (self.names
                and all(isinstance(v, int) and 0 <= v < len(self.names)
                        for v in self.values)):
            labeled = [f"{v} {self.names[v]}" for v in self.values]
            return ", ".join(labeled).encode("utf-8"), self.viz_mime
        return ", ".join(str(v) for v in self.values).encode("utf-8"), self.viz_mime


# ---------------------------------------------------------------------------
# Scalar — single float.
# ---------------------------------------------------------------------------

class Scalar(DataType):
    kind = "scalar"
    file_ext = None  # stored inline
    viz_mime = "text/plain; charset=utf-8"

    def __init__(self, value: float):
        self.value = float(value)

    def encode(self) -> bytes:
        return repr(self.value).encode("ascii")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Scalar":
        return cls(float(blob.decode("ascii")))


# ---------------------------------------------------------------------------
# Json — escape hatch for structured GT/preds that don't fit other kinds.
# ---------------------------------------------------------------------------

class Json(DataType):
    kind = "json"
    file_ext = ".json"
    viz_mime = "application/json"

    def __init__(self, data: dict | list):
        self.data = data

    def encode(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Json":
        return cls(json.loads(blob.decode("utf-8")))

    def validate(self) -> None:
        try:
            json.dumps(self.data)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Json data must be JSON-serializable: {e}") from e


class CocoDetections(DataType):
    """COCO-style detection annotations for ONE image.

    The wire JSON shape is a list of detection records:

        [
          {
            "category_id": int,
            "category_name": str | None,     # optional, for human reading
            "bbox": [x, y, w, h],            # COCO standard xywh:
                                             # top-left corner + WIDTH + HEIGHT
                                             # NOT xyxy. Importers that get xyxy
                                             # from upstream MUST convert.
            "segmentation": [                # zero or more polygons
                [x0, y0, x1, y1, ...],       # flat (x,y) pairs in image space
                ...
            ],
            "area": float | None,            # mask/region area (NOT bbox area)
            "iscrowd": 0 | 1                 # optional
          },
          ...
        ]

    All fields except `category_id` and either `bbox` or `segmentation`
    are optional. The visualisation requires the image as a separate
    input — DataType.visualize() can't reach it, so the actual overlay
    rendering lives in the dataset-view route (see `/api/coco_overlay`
    in app.py)."""
    kind = "coco_detections"
    file_ext = ".json"
    viz_mime = "image/png"

    def __init__(self, detections: list[dict]):
        if not isinstance(detections, list):
            raise ValueError("CocoDetections data must be a list of detection dicts")
        self.detections = detections

    def encode(self) -> bytes:
        return json.dumps(self.detections).encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "CocoDetections":
        return cls(json.loads(blob.decode("utf-8")))

    def validate(self) -> None:
        for i, det in enumerate(self.detections):
            if not isinstance(det, dict):
                raise ValueError(f"detection {i} must be a dict, got {type(det).__name__}")
            if "category_id" not in det:
                raise ValueError(f"detection {i} missing required 'category_id'")
            has_bbox = "bbox" in det and len(det["bbox"]) == 4
            has_seg = "segmentation" in det and bool(det["segmentation"])
            if not (has_bbox or has_seg):
                raise ValueError(
                    f"detection {i} must have either 'bbox' (length 4) or "
                    f"'segmentation' (non-empty)"
                )

    def visualize(self, **_) -> tuple[bytes, str]:
        """Standalone visualisation — text summary. Use the
        image-overlay endpoint when the source Image is available."""
        n = len(self.detections)
        cats = {d.get("category_name") or d["category_id"] for d in self.detections}
        body = (
            f"COCO detections: {n} object{'' if n == 1 else 's'} across "
            f"{len(cats)} categor{'y' if len(cats) == 1 else 'ies'}.\n"
            f"Categories: {sorted(map(str, cats))[:12]}"
        )
        return body.encode("utf-8"), "text/plain"


# ---------------------------------------------------------------------------
# Registry helper.
# ---------------------------------------------------------------------------

def get_type(kind: str) -> type[DataType]:
    """Look up a DataType class by its wire kind."""
    if kind not in DTYPES:
        raise KeyError(f"Unknown data type {kind!r}; known: {sorted(DTYPES)}")
    return DTYPES[kind]


__all__ = [
    "DataType",
    "DTYPES",
    "Image",
    "Mask",
    "Depth",
    "Audio",
    "Text",
    "BBoxes",
    "Label",
    "LabelList",
    "Scalar",
    "Json",
    "CocoDetections",
    "get_type",
]
