#!/usr/bin/env python3
"""Agent-mode HF importer: works directly from the file tree.

The Croissant-based importer only sees columns that the dataset
uploader explicitly exposed in `dataset_info`. Many real benchmark
datasets ship as a folder of paired files (RGB + depth + mask, etc.)
with no Croissant doc at all. This script peeks at the repo's tree,
groups files by a shared sample-id stem, and imports each modality
as a typed field — no Croissant required.

Supported file layouts (auto-detected per repo):

  A. `<modality>/<sample_id>.<ext>`          (KITTI eval, NYUv2, …)
  B. `<modality>_<sample_id>.<ext>` at root  (the modality-prefix shape)
  C. `<split>/<modality>/<sample_id>.<ext>`  (test/depth/foo.png, …)

For each modality, the file extension picks the BH kind:
   .png/.jpg/.jpeg/.bmp/.tiff   → image (or mask if name hints at one)
   .npz/.npy/.exr/.tif/.tiff    → depth (heuristic — see _kind_for_ext)
   .json                        → json
   .txt                         → text
   .wav/.mp3/.flac              → audio

Single-dataset usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/import_hf_agent.py \\
        --repo Kai-Yin-UoA/Monocular_Depth_Essentials \\
        --i-know-what-im-doing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_HF_API_BASE = "https://huggingface.co/api/datasets"
_HF_RAW_BASE = "https://huggingface.co/datasets"


# ---------------------------------------------------------------------------
# HF tree walking
# ---------------------------------------------------------------------------


def _walk_tree(repo_id: str, *, timeout: int = 60,
               max_files: int = 0, max_pages: int = 100) -> list[dict]:
    """Return file entries under <repo>/main, paginating through HF's
    `?cursor=` continuation tokens (1000/page).

    `max_files`: stop once we have at least that many file entries.
    Used by the layout probe — paired-modality structure shows up in
    the first few thousand entries because the API returns paths in
    a stable order, so we don't need to walk a 100k-entry tree just
    to decide "no, this one doesn't pair".
    """
    base = f"{_HF_API_BASE}/{urllib.parse.quote(repo_id, safe='/')}/tree/main?recursive=1"
    out: list[dict] = []
    cursor = None
    pages = 0
    while pages < max_pages:
        url = base + (f"&cursor={cursor}" if cursor else "")
        req = urllib.request.Request(url, headers={"User-Agent": "benchhub-agent/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                rows = json.loads(resp.read())
                link = resp.headers.get("Link") or ""
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  tree fetch error: {e}", file=sys.stderr)
            break
        out.extend(r for r in rows if isinstance(r, dict) and r.get("type") == "file")
        pages += 1
        if max_files and len(out) >= max_files:
            break
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                m = re.search(r"[?&]cursor=([^&>]+)", part)
                if m:
                    nxt = urllib.parse.unquote(m.group(1))
        if not nxt:
            break
        cursor = nxt
    return out


def _fetch_dataset_info(repo_id: str, *, timeout: int = 20) -> dict | None:
    url = f"{_HF_API_BASE}/{urllib.parse.quote(repo_id, safe='/')}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            doc = json.loads(r.read())
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _list_top_for_task(task: str, *, limit: int = 30) -> list[dict]:
    params = urllib.parse.urlencode({
        "filter": f"task_categories:{task}",
        "sort": "downloads", "direction": "-1", "limit": str(limit),
    })
    url = f"{_HF_API_BASE}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return [d for d in json.loads(r.read()) if isinstance(d, dict)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


_FILE_RE = re.compile(r"^(?P<stem>.+)\.(?P<ext>[A-Za-z0-9]+)$")
_NUMERIC_STEM_RE = re.compile(r"^\d{3,}$")          # eg "00042", "00000123"
# Locate the numeric sample-id segment of a filename stem, possibly
# followed by a sub-modality suffix:
#   `cam_1_10002434`           → id=10002434, before=`cam_1`, after=
#   `panorama_0000`            → id=0000,     before=`panorama`, after=
#   `panorama_0000_depth`      → id=0000,     before=`panorama`, after=`depth`
#   `scene12_0123_rgb`         → id=0123,     before=`scene12`,  after=`rgb`
# The id pattern requires ≥3 digits so 2-digit camera indices
# (`cam_1`, `image_02`) aren't mistaken for ids. The suffix is
# optional; modality buckets are then `before` (no suffix) or
# `before/after` (with suffix).
_ID_RE = re.compile(
    r"_(?P<id>\d{3,})(?:_(?P<suffix>[A-Za-z][A-Za-z0-9]*))?$"
)


def _parse_stem(stem: str) -> tuple[str, str] | None:
    """Return `(modality, sample_id)` parsed from a filename stem, or
    None when no `_<digits>` segment exists. The `modality` includes
    a `/` separator between the pre-id and post-id portions, so e.g.
    `panorama_0000_depth` → ('panorama/depth', '0000'). Different
    sub-modalities end up in different buckets in the layout-B
    detector."""
    m = _ID_RE.search(stem)
    if not m:
        return None
    before = stem[:m.start()]
    suffix = m.group("suffix") or ""
    if not before:
        return None
    modality = f"{before}/{suffix}" if suffix else before
    return (modality, m.group("id"))


@dataclass
class DetectedLayout:
    """How a repo's files pair up into (sample, modality)."""
    kind: str                   # 'A_<modality>/<id>' | 'B_<modality>_<id>' | 'C_<split>/<modality>/<id>'
    modalities: dict[str, list[tuple[str, str]]]   # modality → [(sample_id, file_path)]
    note: str = ""

    @property
    def sample_ids(self) -> set[str]:
        s: set[str] = set()
        for items in self.modalities.values():
            for sid, _ in items:
                s.add(sid)
        return s


_IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}
_DEPTH_EXTS = {"npz", "npy", "exr", "pfm"}
_AUDIO_EXTS = {"wav", "mp3", "flac", "ogg"}
_TEXT_EXTS = {"txt"}
_JSON_EXTS = {"json"}
_ALL_DATA_EXTS = _IMAGE_EXTS | _DEPTH_EXTS | _AUDIO_EXTS | _TEXT_EXTS | _JSON_EXTS


def _is_junk_filename(leaf: str) -> bool:
    """Skip hidden / metadata files that pollute layout detection.

    `._<name>` is AppleDouble metadata that macOS Finder inserts when
    zipping; `.DS_Store` is the Finder index. Both show up in HF
    repos that were prepared on Macs and end up surfacing as fake
    modalities (`._panorama`, `._pose`) that pair with nothing.
    Also skip `Thumbs.db` and any plain `.<name>` dotfile.
    """
    return (
        leaf.startswith("._")
        or leaf == ".DS_Store"
        or leaf == "Thumbs.db"
        or leaf.startswith(".")
    )


def _detect_layout(files: list[dict]) -> DetectedLayout | None:
    """Try layouts A → C in order, return the first that pairs up.

    Heuristic: a layout 'pairs up' when at least 2 modalities share
    ≥ 20 sample ids, ie the dataset actually has multi-modality
    structure rather than one folder of unrelated images."""
    # --- Layout A: <modality>/<id>.<ext> (possibly with deeper subdirs;
    # we treat the FIRST path component as the modality and the last
    # segment's stem as the id). Drops everything not in a recognised
    # ext set so README.md / .gitattributes / configs don't pollute.
    a_mods: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in files:
        path = f.get("path", "")
        parts = path.split("/")
        if len(parts) < 2:
            continue
        modality = parts[0]
        leaf = parts[-1]
        if _is_junk_filename(leaf):
            continue
        m = _FILE_RE.match(leaf)
        if not m:
            continue
        ext = m.group("ext").lower()
        if ext not in _ALL_DATA_EXTS:
            continue
        stem = m.group("stem")
        if not _NUMERIC_STEM_RE.match(stem):
            continue
        a_mods[modality].append((stem, path))

    if _has_paired_modalities(a_mods):
        collapsed = _collapse_modalities(dict(a_mods))
        # Collapse can fold N variants of "the same modality" into 1,
        # which would invalidate the pairing. Re-check on the
        # collapsed dict so we only return a layout the import
        # pipeline can actually act on.
        if _has_paired_modalities(collapsed):
            return DetectedLayout(kind="A", modalities=collapsed,
                                   note="<modality>/<id>.<ext>")

    # --- Layout B: <modality>_<id>.<ext> at ANY depth. The full
    # modality name picks up the parent path so different deeply-
    # nested branches (e.g. `germany_batch3/Ingol1/agg_depth/cam_1`
    # vs `…/rgb/cam_1`) get separate field names instead of
    # collapsing into one bucket.
    b_mods: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in files:
        path = f.get("path", "")
        parts = path.split("/")
        leaf = parts[-1]
        if _is_junk_filename(leaf):
            continue
        m = _FILE_RE.match(leaf)
        if not m:
            continue
        ext = m.group("ext").lower()
        if ext not in _ALL_DATA_EXTS:
            continue
        parsed = _parse_stem(m.group("stem"))
        if not parsed:
            continue
        mod_prefix, sid = parsed
        parent = "/".join(parts[:-1])
        full_mod = f"{parent}/{mod_prefix}" if parent else mod_prefix
        b_mods[full_mod].append((sid, path))
    if _has_paired_modalities(b_mods):
        collapsed = _collapse_modalities(dict(b_mods))
        if _has_paired_modalities(collapsed):
            return DetectedLayout(kind="B", modalities=collapsed,
                                   note="<...>/<modality>_<id>.<ext>")

    # --- Layout C: <split>/<modality>/<id>.<ext>. Walks one level
    # deeper than A and prefers a `test`/`val` split branch.
    c_mods: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for f in files:
        path = f.get("path", "")
        parts = path.split("/")
        if len(parts) < 3:
            continue
        split, modality = parts[0], parts[1]
        leaf = parts[-1]
        if _is_junk_filename(leaf):
            continue
        m = _FILE_RE.match(leaf)
        if not m:
            continue
        ext = m.group("ext").lower()
        if ext not in _ALL_DATA_EXTS:
            continue
        stem = m.group("stem")
        if not _NUMERIC_STEM_RE.match(stem):
            continue
        c_mods[(split, modality)].append((stem, path))

    # Prefer a non-train split
    splits = {s for (s, _) in c_mods}
    chosen_split = next(
        (s for s in ("test", "validation", "val") if s in splits),
        None,
    )
    if chosen_split:
        chosen = {mod: rows for (s, mod), rows in c_mods.items() if s == chosen_split}
        if _has_paired_modalities(chosen):
            collapsed = _collapse_modalities(chosen)
            if _has_paired_modalities(collapsed):
                return DetectedLayout(kind="C", modalities=collapsed,
                                       note=f"{chosen_split}/<modality>/<id>.<ext>")
    return None


def _collapse_modalities(
    mods: dict[str, list[tuple[str, str]]],
) -> dict[str, list[tuple[str, str]]]:
    """Collapse modalities whose names share a common path suffix.

    Datasets like PanoCity ship one `<city>/<block>/pano_images/pano`
    + `<city>/<block>/panodepth_images/pano_depth` PAIR per city
    block, giving us 100+ modalities that are really just `pano` and
    `pano_depth` repeated. Field-explosion makes the LB unusable.

    This pass groups modalities by their FINAL path component
    (which is the actual modality token after the prefix-stem regex
    strips `_<id>`) and dedupes sample ids: first occurrence wins.
    The reason for the dedupe rather than namespacing-by-parent is
    that across pano / pano_depth we WANT the same `0001` id to
    line up (so the modalities can be intersected and paired). If
    block1 and block2 both expose a sample called `0001`, we keep
    block1's — losing data, but preserving the cross-modality
    pairing that makes the dataset useful at all.

    Returns the collapsed dict, or the original if collapsing
    wouldn't reduce the modality count.
    """
    by_suffix: dict[str, dict[str, str]] = defaultdict(dict)
    for full_name, items in mods.items():
        suffix = full_name.rsplit("/", 1)[-1]
        for sid, path in items:
            # `setdefault` semantics: first occurrence wins. Stable
            # because dict insertion order is preserved.
            by_suffix[suffix].setdefault(sid, path)
    if len(by_suffix) >= len(mods):
        return mods
    return {name: list(d.items()) for name, d in by_suffix.items()}


_SEMANTIC_CATEGORY_TOKENS = {
    # token (case-insensitive substring of any modality path
    # component) → coarse semantic category. Catches the case
    # where multiple modalities ship as PNGs (so file-ext-based
    # `_kind_for` collapses them all to 'image') but the NAMES
    # clearly mean different things: PanoCity's pano + pano_depth
    # are both PNG but obviously a paired modality dataset.
    "depth":         "depth-like",
    "disparity":     "depth-like",
    "normal":        "normal-like",
    "normals":       "normal-like",
    "mask":          "mask-like",
    "masks":         "mask-like",
    "seg":           "mask-like",
    "segmentation":  "mask-like",
    "panopt":        "mask-like",
    "panoptic":      "mask-like",
    "semantic":      "mask-like",
    "annotation":    "mask-like",
    "annotations":   "mask-like",
    "rgb":           "image-like",
    "rgba":          "image-like",
    "color":         "image-like",
    "image":         "image-like",
    "images":        "image-like",
    "photo":         "image-like",
    "photos":        "image-like",
    "pano":          "image-like",
    "panorama":      "image-like",
    "panoramas":     "image-like",
    "pose":          "pose-like",
    "poses":         "pose-like",
    "keypoint":      "pose-like",
    "keypoints":     "pose-like",
    "flow":          "flow-like",
    "optical_flow":  "flow-like",
    "forward_flow":  "flow-like",
    "bbox":          "bbox-like",
    "bboxes":        "bbox-like",
    "boxes":         "bbox-like",
    "caption":       "text-like",
    "captions":      "text-like",
    "text":          "text-like",
    "audio":         "audio-like",
}


def _semantic_category(modality: str) -> str | None:
    """Coarse semantic bucket for a modality name.

    Splits on `/`, `_`, `-` and checks each segment against
    `_SEMANTIC_CATEGORY_TOKENS`. Returns the LAST match, because
    modality names tend to be `<container>_<actual-thing>`
    (`pano_depth`, `front_normal`, `cam_1_segmentation`); the
    rightmost token is usually the modality itself, the earlier
    ones are containers / camera indices / etc.
    """
    parts = re.split(r"[/_\-]+", modality.lower())
    last: str | None = None
    for p in parts:
        cat = _SEMANTIC_CATEGORY_TOKENS.get(p)
        if cat:
            last = cat
    return last


def _has_paired_modalities(mods: dict[str, list[tuple[str, str]]]) -> bool:
    """A layout pairs up when:
      - ≥ 2 modalities exist
      - the two biggest share ≥ 20 sample ids
      - those modalities span ≥ 2 *distinct semantic categories*

    The semantic-category check is what kills the false-positive
    case of a flat classification tree (LFW: one folder per person,
    all PNGs) — every "modality" name is a person name with no
    `depth`/`mask`/`pose`/etc. token in it, so they all map to the
    same `None` bucket and fail the diversity check.

    We use semantic name buckets rather than the file-extension-
    derived BH `kind` so that legit multi-modality datasets where
    every modality ships as PNG (PanoCity: `pano` + `pano_depth`
    are both PNG → both kind=image, but `pano` and `pano_depth`
    are obviously different semantic categories) still match.
    """
    if len(mods) < 2:
        return False
    pop = sorted(mods.items(),
                 key=lambda kv: len({sid for sid, _ in kv[1]}),
                 reverse=True)
    big_sets = [{sid for sid, _ in v} for _, v in pop[:2]]
    if len(big_sets[0] & big_sets[1]) < 20:
        return False
    cats_seen = {_semantic_category(name) for name in mods}
    cats_seen.discard(None)
    return len(cats_seen) >= 2


# ---------------------------------------------------------------------------
# Kind inference per modality
# ---------------------------------------------------------------------------


# Word-boundary anchors so we don't false-match `seg` inside
# `Bernardo_Segura` or `depth` inside some person name. Modality
# segments are typically separated by `/`, `_`, or `-`; treat all
# three as word boundaries so `pano_depth` and `sem-seg` still match.
_MASK_NAME_TOKENS = re.compile(
    r"(?:^|[/_\-])"
    r"(mask|masks|seg|segmentation|semantic|panopt|panoptic|"
    r"annotation|annotations|label_map|sem_seg|instance_seg)"
    r"(?:$|[/_\-])",
    re.I,
)
_DEPTH_NAME_TOKENS = re.compile(
    r"(?:^|[/_\-])(depth|disparity|normal|normals)(?:$|[/_\-])",
    re.I,
)


def _kind_for(modality: str, paths: list[str]) -> str:
    """Pick a BH `kind` for a modality, given its name + file ext."""
    if not paths:
        return "json"
    exts = Counter(p.rsplit(".", 1)[-1].lower() for p in paths if "." in p)
    ext = exts.most_common(1)[0][0]
    if ext in _DEPTH_EXTS:
        return "depth"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _JSON_EXTS:
        return "json"
    if ext in _TEXT_EXTS:
        return "text"
    if ext in _IMAGE_EXTS:
        # Mask promotion is safe because `bh.Mask.file_ext` is also
        # `.png`, so the typed-manifest importer accepts our PNG
        # bytes for a mask field.
        if _MASK_NAME_TOKENS.search(modality):
            return "mask"
        # Depth-named PNGs/TIFFs (PanoCity's `pano_depth`,
        # IntuitivePhysics's depth tiff) get promoted to depth so
        # the catalog applies the colormap palette + colorbar.
        # `bh.Depth.file_ext` is `.npz`, so `_stage_dataset` runs
        # a PIL → npz conversion before the typed-manifest importer
        # sees the staged files.
        if _DEPTH_NAME_TOKENS.search(modality):
            return "depth"
        return "image"
    return "json"


# ---------------------------------------------------------------------------
# Download + import
# ---------------------------------------------------------------------------


def _raw_url(repo_id: str, path: str) -> str:
    return f"{_HF_RAW_BASE}/{urllib.parse.quote(repo_id, safe='/')}/resolve/main/{urllib.parse.quote(path)}"


def _download(url: str, dest: Path, *, timeout: int = 120):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "benchhub-agent/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out)


def _png_depth_to_npz(src: Path, dst: Path) -> None:
    """Decode a depth map shipped as PNG/TIFF/etc. and save it as
    `.npz` with key `depth`, the shape `bh.Depth.decode` expects.

    Handles the common encodings:
      - 16-bit grayscale (PIL modes I, I;16, I;16L, I;16B) — direct uint16
      - 32-bit float (mode F) — direct float32
      - 8-bit grayscale (mode L, P) — promote to float32 for headroom
      - RGB / RGBA with packed depth — take the first channel.
        (More exotic packings — 24-bit packed across R/G/B — would
        need per-dataset code; this baseline at least produces a
        decodable depth that renders sensibly via the colormap.)
    """
    from PIL import Image
    import numpy as np
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        mode = im.mode
        if mode in ("I", "I;16", "I;16L", "I;16B"):
            arr = np.array(im, dtype=np.uint16)
        elif mode == "F":
            arr = np.array(im, dtype=np.float32)
        elif mode in ("L", "P"):
            arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32)
        else:
            # RGB / RGBA / other — pull the first channel.
            arr = np.array(im.convert("RGB"), dtype=np.uint8)[..., 0].astype(np.float32)
    np.savez_compressed(dst, depth=arr)


def _build_manifest(repo_id: str, layout: DetectedLayout, *,
                    max_samples: int) -> tuple[dict, list[tuple[str, str, Path]]]:
    """Return (manifest_dict, [(modality, sample_id, source_path_in_repo)])
    capped at `max_samples` shared sample ids."""
    # Order modalities by coverage (biggest first) so we don't lose
    # the most-populated one to capping.
    by_cov = sorted(layout.modalities.items(),
                    key=lambda kv: len({s for s, _ in kv[1]}),
                    reverse=True)
    # Build the shared-sample set: intersect the two biggest modalities
    big_sets = [{sid for sid, _ in v} for _, v in by_cov[:2]]
    shared = big_sets[0] & big_sets[1] if len(big_sets) > 1 else set()
    if not shared:
        return ({}, [])
    sample_ids = sorted(shared)
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]
    sample_set = set(sample_ids)

    fields = []
    rows = []
    seen_modality_names: dict[str, str] = {}
    field_kinds: dict[str, str] = {}
    for modality, items in by_cov:
        # Pick a canonical kind from the modality's actual files.
        paths_in_use = [p for s, p in items if s in sample_set]
        if not paths_in_use:
            continue
        kind = _kind_for(modality, paths_in_use)
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", modality).strip("_") or "field"
        # Avoid name collisions across modalities that sanitise to the same string
        n = safe_name; i = 2
        while n in seen_modality_names:
            n = f"{safe_name}_{i}"; i += 1
        seen_modality_names[n] = modality
        field_kinds[n] = kind
        fields.append({"name": n, "kind": kind, "role": "gt", "params": {}})
        for sid, path in items:
            if sid in sample_set:
                rows.append((n, sid, Path(path)))

    return ({
        "name": repo_id.replace("/", "__"),
        "version": "1.0",
        "fields": fields,
        "samples": sample_ids,
    }, rows)


def _stage_dataset(repo_id: str, layout: DetectedLayout, *,
                   max_samples: int, staging: Path,
                   progress=None) -> dict | None:
    manifest, rows = _build_manifest(repo_id, layout, max_samples=max_samples)
    if not manifest:
        return None
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Stage each field's files under <staging>/<field>/<sample_id>.<ext>
    # — the layout `import_typed_dataset` expects. The on-disk
    # extension MUST match the canonical extension for the field's
    # kind (e.g. `bh.Image.file_ext == ".png"`), otherwise the
    # importer's per-sample existence check fails. PIL sniffs magic
    # bytes regardless of filename, so renaming a `.tiff` to `.png`
    # for an image-kind field still decodes correctly. For binary
    # formats with their own header (`depth → .npz`,
    # `audio → .wav`), we can't lie — if source ext mismatches we
    # skip the file (and any sample that becomes incomplete drops
    # out of the manifest later).
    from benchhub.types import DTYPES as _DTYPES
    field_canonical: dict[str, str] = {}
    field_kinds: dict[str, str] = {}
    for f in manifest.get("fields", []):
        k = f["kind"]
        field_kinds[f["name"]] = k
        cls = _DTYPES.get(k)
        field_canonical[f["name"]] = cls.file_ext if cls and cls.file_ext else ""
    sniffable_kinds = {"image", "mask"}

    staged: set[tuple[str, str]] = set()   # (field, sample_id) actually on disk
    total = len(rows)
    for i, (field, sid, src_in_repo) in enumerate(rows):
        src_ext = src_in_repo.suffix
        canonical = field_canonical.get(field, "")
        kind = field_kinds.get(field, "")

        # Decide destination extension + whether the bytes need a
        # format conversion at staging time.
        convert_to_depth_npz = False
        if kind in sniffable_kinds and canonical:
            # image/mask: rename to canonical .png; PIL sniffs magic.
            dest_ext = canonical
        elif kind == "depth" and src_ext.lower() in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
        }:
            # PNG / TIFF depth maps → decode + repack as .npz so the
            # typed-manifest pipeline (and the catalog's depth
            # render path) receive the array shape they expect.
            dest_ext = canonical or ".npz"
            convert_to_depth_npz = True
        elif canonical and src_ext.lower() != canonical.lower():
            if progress and (i % 25 == 0 or i == total - 1):
                progress(i + 1, total)
            continue
        else:
            dest_ext = src_ext or canonical

        dest = staging / field / f"{sid}{dest_ext}"
        try:
            if convert_to_depth_npz:
                # Download to a temp file, decode, re-pack as npz.
                with tempfile.NamedTemporaryFile(suffix=src_ext, delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    _download(_raw_url(repo_id, str(src_in_repo)), Path(tmp_path))
                    _png_depth_to_npz(Path(tmp_path), dest)
                    staged.add((field, sid))
                finally:
                    try: os.unlink(tmp_path)
                    except OSError: pass
            else:
                _download(_raw_url(repo_id, str(src_in_repo)), dest)
                staged.add((field, sid))
        except Exception as e:
            print(f"    [warn] download failed {src_in_repo}: {e}",
                  file=sys.stderr)
        if progress and (i % 25 == 0 or i == total - 1):
            progress(i + 1, total)

    # Prune manifest samples to only those with every field staged.
    # Otherwise the importer bails on the first missing file.
    if staged:
        per_sample_fields: dict[str, set[str]] = {}
        for fld, sid in staged:
            per_sample_fields.setdefault(sid, set()).add(fld)
        required_fields = {f["name"] for f in manifest["fields"]}
        kept = [sid for sid in manifest["samples"]
                if per_sample_fields.get(sid) == required_fields]
        if len(kept) != len(manifest["samples"]):
            print(f"    pruning manifest: {len(manifest['samples'])} → "
                  f"{len(kept)} samples after field-completeness check")
            manifest["samples"] = kept
        if not kept:
            return None
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _import_staged(repo_id: str, staging: Path, *, owner_user_id: int,
                   task_label: str | None, dataset_card: dict | None) -> dict:
    from app import (
        Dataset, DatasetField, Sample, CustomField, app, db,
        _hf_tags_to_category,
    )
    from benchhub.manifest import import_typed_dataset

    with app.app_context():
        # Pre-create the row so it shows on /datasets with status='importing'
        ds_row = Dataset(
            name=repo_id.replace("/", "__"),
            owner_user_id=owner_user_id,
            visibility="public",
            import_status="importing",
            import_progress_json=json.dumps({
                "phase": "agent-import", "message": f"agent-mode: {repo_id}",
            }),
        )
        db.session.add(ds_row); db.session.commit()
        ds_id = ds_row.id

        existing = Dataset.query.get(ds_id)
        try:
            _, summary = import_typed_dataset(
                str(staging),
                db_session=db.session,
                Dataset=Dataset, Sample=Sample,
                CustomField=CustomField, DatasetField=DatasetField,
                upload_folder=app.config["UPLOAD_FOLDER"],
                existing_dataset=existing,
            )
            existing.source_kind = "hf"
            existing.source_url = f"https://huggingface.co/datasets/{repo_id}"
            existing.source_metadata = json.dumps({
                "repo_id": repo_id,
                "split": None,
                "agent_mode": True,
                "samples_imported": summary.get("samples"),
            })
            existing.import_status = "ready"
            existing.import_error = None
            existing.import_progress_json = None
            tags = (dataset_card or {}).get("tags") or []
            cat = _hf_tags_to_category(tags)
            if cat:
                existing.category = cat
            db.session.commit()
            return {"dataset_id": ds_id, **summary}
        except Exception as e:
            db.session.rollback()
            row = Dataset.query.get(ds_id)
            if row is not None:
                row.import_status = "failed"
                row.import_error = str(e)
                db.session.commit()
            raise


# ---------------------------------------------------------------------------
# Top-level: import one repo
# ---------------------------------------------------------------------------


def _admin_user_id() -> int | None:
    from app import User
    u = User.query.filter_by(is_admin=True).order_by(User.id).first()
    return u.id if u else None


def import_one(repo_id: str, *, max_samples: int = 200, dry_run: bool = False,
               task_label: str | None = None) -> dict:
    print(f"  → walking tree of {repo_id}…", flush=True)
    files = _walk_tree(repo_id)
    print(f"    {len(files)} files")
    if not files:
        return {"status": "skipped", "reason": "empty tree"}
    layout = _detect_layout(files)
    if not layout:
        return {"status": "skipped", "reason": "no recognised paired layout"}
    print(f"    layout {layout.kind}: {layout.note}")
    print(f"    modalities: { {m: len({sid for sid,_ in v}) for m, v in layout.modalities.items()} }")
    shared = layout.sample_ids
    print(f"    shared samples (intersection of biggest two): {len(shared)}")
    if dry_run:
        return {"status": "would-import",
                "layout": layout.kind, "modalities": list(layout.modalities)}
    from app import app
    with app.app_context():
        owner = _admin_user_id()
    if owner is None:
        return {"status": "failed", "reason": "no admin user"}
    card = _fetch_dataset_info(repo_id)
    with tempfile.TemporaryDirectory(prefix="bh_agent_") as staging:
        manifest = _stage_dataset(
            repo_id, layout, max_samples=max_samples, staging=Path(staging),
            progress=lambda i, total: print(f"    download {i}/{total}", flush=True),
        )
        if not manifest:
            return {"status": "skipped", "reason": "no shared samples after capping"}
        result = _import_staged(
            repo_id, Path(staging), owner_user_id=owner,
            task_label=task_label, dataset_card=card,
        )
    return {"status": "imported", **result}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=str, help="single repo_id to import")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-know-what-im-doing", action="store_true",
                   help="confirm writes to the production data dir")
    args = p.parse_args(argv)

    dd = os.environ.get("BENCHHUB_DATA_DIR") or os.path.expanduser("~/.dtofbenchmarking")
    if dd == os.path.expanduser("~/.dtofbenchmarking") and not args.i_know_what_im_doing:
        print("fatal: production data dir. pass --i-know-what-im-doing",
              file=sys.stderr)
        return 2
    os.environ.setdefault("BENCHHUB_DATA_DIR", dd)

    if not args.repo:
        print("--repo required for single-import mode", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    r = import_one(args.repo, max_samples=args.max_samples, dry_run=args.dry_run)
    print(f"\n[{time.monotonic()-t0:.1f}s] {args.repo}: {r}")
    return 0 if r.get("status") in ("imported", "would-import") else 1


if __name__ == "__main__":
    raise SystemExit(main())
