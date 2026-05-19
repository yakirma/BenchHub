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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params!r})"


# ---------------------------------------------------------------------------
# Image — RGB(A) photos or grayscale, uint8.
# ---------------------------------------------------------------------------

class Image(DataType):
    kind = "image"
    file_ext = ".png"

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


# ---------------------------------------------------------------------------
# Audio — 1D or (T, channels) float32 waveform + sample rate.
# ---------------------------------------------------------------------------

class Audio(DataType):
    kind = "audio"
    file_ext = ".wav"

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


# ---------------------------------------------------------------------------
# Label — single class (int or string); vocab declared at LB level.
# ---------------------------------------------------------------------------

class Label(DataType):
    kind = "label"
    file_ext = None  # stored inline

    def __init__(self, value: int | str):
        if not isinstance(value, (int, str)):
            raise ValueError(f"Label value must be int or str; got {type(value).__name__}")
        self.value = value

    def encode(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")

    @classmethod
    def decode(cls, blob: bytes, params: dict | None = None) -> "Label":
        return cls(json.loads(blob.decode("utf-8")))


# ---------------------------------------------------------------------------
# Scalar — single float.
# ---------------------------------------------------------------------------

class Scalar(DataType):
    kind = "scalar"
    file_ext = None  # stored inline

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
    "Scalar",
    "Json",
    "get_type",
]
