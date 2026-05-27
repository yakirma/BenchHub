"""Preview-tier renderers for the hybrid storage model.

A "preview-only" dataset stores a lightweight rendered thumbnail
per per-sample modality, NOT the original bytes:

  image / mask / depth / bboxes  → JPG (downscaled, colormapped where
                                         relevant) — ~10-30 KB each
  audio                          → PNG waveform thumbnail
  text / json / scalar / label   → stored full (already tiny)

The catalog browse + dataset_view samples table renders the preview;
the original bytes are re-materialised per-LB on demand (a
LeaderboardMaterialization fetches the relevant subset from
upstream HF and stores them at full resolution).

This file holds the pure rendering helpers. They take bytes/arrays
and return bytes — no DB, no filesystem dependencies, so they're
unit-testable and reusable from any import pipeline.
"""
from __future__ import annotations
import io
from typing import Any

import numpy as np
from PIL import Image as PILImage


# Max edge length for preview thumbnails. 512 keeps file size ~30 KB
# for typical content while still being readable in the samples
# table modal (which displays at 160-300 px).
PREVIEW_MAX_EDGE = 512
PREVIEW_JPEG_Q = 85


def _resize_max_edge(im: PILImage.Image, max_edge: int = PREVIEW_MAX_EDGE) -> PILImage.Image:
    """Resize so the longer edge == max_edge, preserving aspect ratio.
    No-op when the image is already smaller."""
    w, h = im.size
    if max(w, h) <= max_edge:
        return im
    if w >= h:
        new_w = max_edge
        new_h = max(1, int(round(h * max_edge / w)))
    else:
        new_h = max_edge
        new_w = max(1, int(round(w * max_edge / h)))
    return im.resize((new_w, new_h), PILImage.LANCZOS)


def image_preview(src: bytes | PILImage.Image) -> bytes:
    """RGB JPG preview of an image. Handles all PIL-readable formats."""
    if isinstance(src, (bytes, bytearray)):
        im = PILImage.open(io.BytesIO(src))
    else:
        im = src
    if im.mode not in ('RGB', 'L'):
        im = im.convert('RGB')
    im = _resize_max_edge(im)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=PREVIEW_JPEG_Q, optimize=True)
    return buf.getvalue()


def _turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB via the turbo colormap. Numpy-only
    so we don't need matplotlib at preview time. Coefficient table
    from Google AI's turbo polynomial approximation."""
    v = np.clip(values, 0.0, 1.0)
    r = (34.61 + v * (1172.33 + v * (-10793.56 + v * (33300.12 + v * (-38394.49 + v * 14825.05))))).clip(0, 255)
    g = (23.31 + v * (557.33 + v * (1225.33 + v * (-3574.96 + v * (1073.77 + v * 707.56))))).clip(0, 255)
    b = (27.2 + v * (3211.1 + v * (-15327.97 + v * (27814.0 + v * (-22569.18 + v * 6838.66))))).clip(0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


_TURBO_LUT_CACHE = None
def _turbo_lut() -> np.ndarray:
    """256-entry RGB LUT for turbo. Cached because the polynomial
    only needs to run once per process."""
    global _TURBO_LUT_CACHE
    if _TURBO_LUT_CACHE is None:
        _TURBO_LUT_CACHE = _turbo_colormap(np.linspace(0, 1, 256))
    return _TURBO_LUT_CACHE


def reverse_turbo(rgb: np.ndarray) -> np.ndarray:
    """Approximate inverse of the turbo colormap. Given an (H, W, 3)
    uint8 RGB array, return an (H, W) float32 array in [0, 1] where
    each pixel is the LUT index (normalised) whose RGB is closest to
    the input by L2 distance.
    Lossy by JPEG compression + LUT quantisation, but monotonic — the
    hover number tracks the original depth ordering."""
    lut = _turbo_lut().astype(np.float32)
    flat = rgb.reshape(-1, 3).astype(np.float32)
    # L2 distance from each pixel to all 256 LUT entries; argmin per pixel.
    d = ((flat[:, None, :] - lut[None, :, :]) ** 2).sum(axis=-1)
    idx = d.argmin(axis=1)
    return (idx.astype(np.float32) / 255.0).reshape(rgb.shape[:2])


def depth_meta(arr: np.ndarray, *, vmin: float | None = None,
               vmax: float | None = None) -> dict:
    """Return the (min, max, shape) that depth_preview would normalise
    against. Saved as a sidecar .meta.json next to preview JPGs so
    callers can recover real metric values from the reverse-turbo
    lookup."""
    a = arr
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[-1] == 3:
        chan_max = float(np.nanmax(a)) if a.size else 0.0
        if chan_max <= 255.5:
            r = a[..., 0].astype(np.float64)
            g = a[..., 1].astype(np.float64)
            b = a[..., 2].astype(np.float64)
            a = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
    a = np.asarray(a, dtype=np.float32)
    mask = np.isfinite(a) & (a > 0)
    if mask.any():
        lo = float(vmin) if vmin is not None else float(a[mask].min())
        hi = float(vmax) if vmax is not None else float(a[mask].max())
    else:
        lo, hi = 0.0, 0.0
    return {'min': lo, 'max': hi, 'shape': list(a.shape)}


def depth_preview(arr: np.ndarray, *, vmin: float | None = None,
                  vmax: float | None = None) -> bytes:
    """Colormapped JPG preview of a depth map.
    Accepts (H,W) or (H,W,1) float arrays in any unit. The visual
    range is the array's own min/max unless overridden — preview is
    qualitative, not metric.
    Also accepts (H,W,3) byte-range arrays as RGB-packed depth
    (CARLA convention: depth = R + G*256 + B*65536, normalised to
    the 24-bit range)."""
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        chan_max = float(np.nanmax(arr)) if arr.size else 0.0
        if chan_max <= 255.5:
            r = arr[..., 0].astype(np.float64)
            g = arr[..., 1].astype(np.float64)
            b = arr[..., 2].astype(np.float64)
            arr = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
    if arr.ndim != 2:
        raise ValueError(f'depth array must be 2D, got shape {arr.shape}')
    a = arr.astype(np.float32)
    # Treat zero / negative as "no data" — common in depth datasets
    # (KITTI uses 0 = unknown, NYU uses 0/Inf, ToF cameras use NaN).
    mask = np.isfinite(a) & (a > 0)
    if not mask.any():
        # Render a black image of the same shape.
        out = np.zeros((*a.shape, 3), dtype=np.uint8)
    else:
        lo = float(vmin) if vmin is not None else float(a[mask].min())
        hi = float(vmax) if vmax is not None else float(a[mask].max())
        if hi <= lo:
            hi = lo + 1e-6
        normed = np.zeros_like(a)
        normed[mask] = (a[mask] - lo) / (hi - lo)
        out = _turbo_colormap(normed)
        # Mark no-data pixels as solid black.
        out[~mask] = 0
    im = _resize_max_edge(PILImage.fromarray(out, mode='RGB'))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=PREVIEW_JPEG_Q, optimize=True)
    return buf.getvalue()


def _deterministic_palette(n_classes: int) -> np.ndarray:
    """Distinct-hue RGB palette indexed 0..n_classes-1; index 0 stays
    black (background). Golden-ratio rotation gives well-separated
    hues without a tabulated lookup."""
    pal = np.zeros((n_classes, 3), dtype=np.uint8)
    for k in range(1, n_classes):
        hue = ((k * 137) % 360) / 360.0
        # HSV→RGB at full saturation/value
        i = int(hue * 6)
        f = hue * 6 - i
        v = 1.0
        s = 0.85
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = [(v, t, p), (q, v, p), (p, v, t),
                   (p, q, v), (t, p, v), (v, p, q)][i % 6]
        pal[k] = (int(r * 255), int(g * 255), int(b * 255))
    return pal


def mask_preview(arr: np.ndarray) -> bytes:
    """Palette-colored JPG preview of a class-id segmentation mask.
    Index 0 → black background, indices 1..N → distinct hues."""
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # Already a colored mask — keep visual, just downscale.
        im = PILImage.fromarray(arr[..., :3].astype(np.uint8), 'RGB')
    elif arr.ndim != 2:
        raise ValueError(f'mask must be (H,W) or (H,W,1)/(H,W,3), got {arr.shape}')
    else:
        ids = arr.astype(np.int32)
        n = int(ids.max()) + 1 if ids.size else 1
        pal = _deterministic_palette(max(n, 16))
        rgb = pal[ids.clip(0, n - 1)]
        im = PILImage.fromarray(rgb, 'RGB')
    im = _resize_max_edge(im)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=PREVIEW_JPEG_Q, optimize=True)
    return buf.getvalue()


def audio_preview(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Waveform-thumbnail PNG. Renders amplitude min/max envelopes
    over time so the result reads as a recognisable waveform."""
    if samples.ndim > 1:
        samples = samples.mean(axis=-1) if samples.shape[-1] in (2, 6) else samples[..., 0]
    a = np.asarray(samples, dtype=np.float32)
    if a.size == 0:
        a = np.zeros(1, dtype=np.float32)
    # Bin the samples into ~256 horizontal pixels.
    nbins = 256
    if a.size < nbins:
        bins = a
    else:
        edges = np.linspace(0, a.size, nbins + 1, dtype=int)
        env_max = np.array([a[edges[i]:edges[i+1]].max() for i in range(nbins)])
        env_min = np.array([a[edges[i]:edges[i+1]].min() for i in range(nbins)])
        bins = (env_max, env_min)
    h, w = 128, nbins
    img = np.zeros((h, w, 3), dtype=np.uint8) + 30  # near-black bg
    if isinstance(bins, tuple):
        env_max, env_min = bins
        # Scale to [-1, 1]
        m = max(float(np.abs(env_max).max()), float(np.abs(env_min).max()), 1e-6)
        env_max = (env_max / m * (h / 2 - 2)).astype(int) + h // 2
        env_min = (env_min / m * (h / 2 - 2)).astype(int) + h // 2
        for x in range(w):
            y0, y1 = sorted((int(env_max[x]), int(env_min[x])))
            img[max(0, y0):min(h, y1 + 1), x] = (124, 58, 237)  # primary purple
    im = PILImage.fromarray(img, 'RGB')
    buf = io.BytesIO()
    im.save(buf, 'PNG', optimize=True)
    return buf.getvalue()


# Dispatch by typed-kind. Used by the import pipeline when
# preview_only=True: the staged file gets fed through this and
# the resulting bytes are written with the appropriate preview ext.
def render_preview(kind: str, payload: Any) -> tuple[bytes, str]:
    """Return (bytes, file_ext) for a preview rendering of `payload`.

    `payload` accepts whatever's most natural per kind:
      - image / mask: raw bytes (file content) OR PIL.Image OR np.ndarray
      - depth: np.ndarray (HxW float)
      - audio: np.ndarray (samples) — sr can be passed via wrapper
    For inline kinds (scalar / label / text / json) the caller should
    NOT call this — those store their value_text inline, no file.
    """
    if kind == 'image':
        if isinstance(payload, np.ndarray):
            payload = PILImage.fromarray(payload.astype(np.uint8))
        return image_preview(payload), '.jpg'
    if kind == 'mask':
        if isinstance(payload, (bytes, bytearray)):
            im = PILImage.open(io.BytesIO(payload))
            arr = np.asarray(im)
        elif isinstance(payload, PILImage.Image):
            arr = np.asarray(payload)
        else:
            arr = np.asarray(payload)
        return mask_preview(arr), '.jpg'
    if kind == 'depth':
        if isinstance(payload, (bytes, bytearray)):
            # Probably an .npz; let the caller pre-decode.
            raise ValueError('depth preview needs an np.ndarray, not bytes')
        return depth_preview(np.asarray(payload, dtype=np.float32)), '.jpg'
    if kind == 'audio':
        return audio_preview(np.asarray(payload)), '.png'
    if kind == 'bboxes':
        # No standalone preview — bboxes draw on top of an image.
        # Inline as JSON like before; the caller should pass the data
        # straight to value_text as a json kind would.
        raise ValueError('bboxes have no standalone preview; render via overlay')
    raise ValueError(f'no preview renderer for kind={kind!r}')
