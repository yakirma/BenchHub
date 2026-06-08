"""Kaggle dataset shape detection → `materialize_file_tree` spec.

Fingerprints the *extracted* file tree of a Kaggle dataset (a flat list of
repo-relative path strings, optionally a `peek` into tabular headers/rows)
and emits the same `spec` structure `benchhub.file_tree_import` consumes —
i.e. it *extends* `inspect_repo`/`analyze_levels` with the Kaggle-specific
shapes those generic helpers can't infer (RLE-in-CSV masks, COCO/VOC/YOLO
detection, palette masks, DICOM/NIfTI), and enforces the **hidden-GT guard**.

It is pure (no network, no disk beyond the optional `peek` callable) so it
is exhaustively unit-testable offline (tests/test_kaggle_detect.py).

Return shape (`detect_shape`):
    {
      "shape": "C",                     # taxonomy code (plan §4), or None
      "shape_label": "ImageFolder (class dirs)",
      "spec": [ <field dict>, ... ],    # ready for materialize_file_tree
      "benchmarkable": True,            # has usable ground truth?
      "reason": "",                     # why not, when benchmarkable is False
      "hidden_gt": [ "test.csv", ... ], # files refused as GT (competition)
      "warnings": [ ... ],
      "needs_conversion": ["rle"],      # new loaders the spec relies on
    }

The new loaders a detected spec may reference — implemented in
`file_tree_import` (the conversion hooks):
  - `rle`  : a shared CSV of run-length rows → integer-label-map Mask.
  - `coco` : a COCO detection JSON → CocoDetections / BBoxes per image.
  - `voc`  : per-image Pascal VOC XML → BBoxes.
  - `yolo` : per-image YOLO TXT (+ names file) → BBoxes.
  - `csv` / `parquet` with `index: true` : the table's ROWS enumerate the
    samples (pure-tabular datasets have no per-file index modality).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .file_tree_import import (
    _data_files, _kind_loader_for_ext, _MASK_NAME_WORDS, _MODALITY_WORDS,
)

__all__ = [
    "detect_shape", "partition_hidden_gt", "looks_like_rle_columns",
    "TARGET_COL_NAMES", "TEXT_COL_NAMES",
]

# --------------------------------------------------------------------------
# Hidden-GT guard (plan §5)
# --------------------------------------------------------------------------

# Competition artifacts whose labels are withheld — never a GT source.
_HIDDEN_BASENAMES = {"sample_submission.csv", "samplesubmission.csv",
                     "sample_submission.csv.gz"}
_TEST_DIR_RE = re.compile(r"(^|/)(test|testing|test_images|test_set|holdout)(/|$)",
                          re.IGNORECASE)
_TEST_CSV_RE = re.compile(r"(^|/)test\.csv(\.gz)?$", re.IGNORECASE)


def partition_hidden_gt(files):
    """Split a file list into (usable, hidden_gt). The hidden bucket holds
    competition files whose ground truth is withheld — `sample_submission`
    always, and (only when a competition signal is present) `test.csv` and
    `test*/` image dirs. Without a competition signal, `test` data is left
    usable (it may be a labelled eval split)."""
    data = _data_files(files)
    has_sample_sub = any(f.rsplit("/", 1)[-1].lower() in _HIDDEN_BASENAMES
                         for f in data)
    has_train = any(re.search(r"(^|/)train", f, re.IGNORECASE) for f in data)
    competition = has_sample_sub or has_train
    usable, hidden = [], []
    for f in data:
        base = f.rsplit("/", 1)[-1].lower()
        if base in _HIDDEN_BASENAMES:
            hidden.append(f)
        elif competition and (_TEST_CSV_RE.search(f) or _TEST_DIR_RE.search(f)):
            hidden.append(f)
        else:
            usable.append(f)
    return usable, hidden


# --------------------------------------------------------------------------
# Tabular column heuristics
# --------------------------------------------------------------------------

# Column names that conventionally hold the prediction target.
TARGET_COL_NAMES = {
    "label", "labels", "target", "class", "classes", "category", "categories",
    "y", "output", "outcome", "sentiment", "diagnosis", "species", "type",
    "is_fraud", "fraud", "survived", "rating", "score", "price", "result",
}
# Columns that read as free-text inputs (→ Text), not categorical/numeric.
TEXT_COL_NAMES = {
    "text", "review", "reviews", "comment", "comments", "tweet", "sentence",
    "sentences", "content", "body", "title", "question", "answer", "abstract",
    "description", "message", "headline", "summary", "caption",
}
# RLE columns (segmentation masks encoded in a CSV cell).
_RLE_COL_RE = re.compile(
    r"(encoded[\s_-]?pixels|^rle$|_rle$|run[\s_-]?length|segmentation_rle)",
    re.IGNORECASE)
_ID_COL_RE = re.compile(r"(^id$|image[\s_-]?id|imageid|file[\s_-]?name|"
                        r"filename|fname|^image$|^img$|^name$)", re.IGNORECASE)
_CLASS_COL_RE = re.compile(r"(class[\s_-]?id|classid|category[\s_-]?id|"
                           r"^class$|^label$)", re.IGNORECASE)


def looks_like_rle_columns(columns):
    """(rle_col, id_col, class_col) for a header that encodes RLE masks, or
    None. `class_col` is None for binary/single-class RLE."""
    rle = next((c for c in columns if _RLE_COL_RE.search(str(c))), None)
    if not rle:
        return None
    idc = next((c for c in columns if _ID_COL_RE.search(str(c))), None)
    cls = next((c for c in columns
                if _CLASS_COL_RE.search(str(c)) and c != rle), None)
    return rle, idc, cls


def _is_floatish(v):
    try:
        f = float(str(v))
        return not (f == int(f) and abs(f) < 50)   # small ints read as classes
    except (ValueError, TypeError):
        return False


def _classify_target(values):
    """A target column's sampled values → 'label' (categorical) or 'scalar'
    (continuous regression)."""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return "label"
    floatish = sum(1 for v in vals if _is_floatish(v))
    if floatish >= max(1, int(0.6 * len(vals))):
        return "scalar"
    return "label"


def _classify_feature(name, values):
    """A feature column → kind. Long free text → text; mostly-numeric →
    scalar; else label/text by name."""
    nm = str(name).lower()
    if nm in TEXT_COL_NAMES:
        return "text"
    vals = [v for v in (values or []) if v not in (None, "")]
    if vals:
        if sum(1 for v in vals if _is_floatish(v)) >= max(1, int(0.6 * len(vals))):
            return "scalar"
        avg_len = sum(len(str(v)) for v in vals) / len(vals)
        if avg_len >= 40:                 # free text, not a category
            return "text"
        if len({str(v) for v in vals}) <= max(2, len(vals) // 2):
            return "label"
        return "text"
    return "scalar"


# --------------------------------------------------------------------------
# Tree structure helpers
# --------------------------------------------------------------------------

_IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"}
_AUDIO_EXTS = {"wav", "mp3", "flac", "ogg", "m4a"}


def _ext(path):
    base = path.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def _top_dirs(files):
    """First-segment directory → list of files under it (only multi-segment
    paths)."""
    out = defaultdict(list)
    for f in files:
        if "/" in f:
            out[f.split("/", 1)[0]].append(f)
    return out


def _all_images(files):
    return files and all(_ext(f) in _IMAGE_EXTS for f in files)


def _spec_image(name, role, pattern):
    return {"name": name, "kind": "image", "role": role,
            "loader": "file", "pattern": pattern}


# --------------------------------------------------------------------------
# Individual shape detectors. Each returns a result dict or None.
# --------------------------------------------------------------------------

def _detect_rle_csv(files, peek):
    """Shape I — RLE-in-CSV segmentation. A CSV whose header has an
    EncodedPixels-like column, paired with an images dir → Mask(gt) +
    Image(input)."""
    csvs = [f for f in files if _ext(f) == "csv"]
    img_dirs = _image_dirs(files)
    for csv in csvs:
        cols = _columns(peek, csv)
        if not cols:
            continue
        hit = looks_like_rle_columns(cols)
        if not hit:
            continue
        rle_col, id_col, class_col = hit
        spec = []
        # Image input: the largest image dir; join by the CSV id column.
        if img_dirs:
            idir, ifiles = max(img_dirs.items(), key=lambda kv: len(kv[1]))
            iext = Counter(_ext(f) for f in ifiles).most_common(1)[0][0]
            spec.append(_spec_image("image", "input", f"{idir}/{{id}}.{iext}"))
        mask = {"name": "mask", "kind": "mask", "role": "gt", "loader": "rle",
                "pattern": csv, "index": not img_dirs,
                "id_column": id_col, "id_token": "id",
                "value_column": rle_col, "class_column": class_col,
                "order": "F", "one_indexed": True}
        spec.append(mask)
        return _result("I", "Segmentation — RLE in CSV", spec,
                       needs_conversion=["rle"],
                       warnings=([] if img_dirs else
                                 ["No images dir found to pair with the RLE "
                                  "masks; mask H/W must be supplied."]))
    return None


def _detect_coco(files, peek):
    """Shape E — COCO detection JSON (images+annotations+categories)."""
    for f in files:
        if _ext(f) != "json":
            continue
        keys = _json_keys(peek, f)
        if keys and {"images", "annotations"} <= set(keys):
            img_dirs = _image_dirs(files)
            spec = []
            if img_dirs:
                idir, ifiles = max(img_dirs.items(), key=lambda kv: len(kv[1]))
                iext = Counter(_ext(x) for x in ifiles).most_common(1)[0][0]
                spec.append(_spec_image("image", "input", f"{idir}/{{id}}.{iext}"))
            spec.append({"name": "detections", "kind": "coco_detections",
                         "role": "gt", "loader": "coco", "pattern": f,
                         "index": not img_dirs, "image_field": "file_name"})
            return _result("E", "Detection — COCO JSON", spec,
                           needs_conversion=["coco"])
    return None


def _detect_voc(files, peek):
    """Shape F — Pascal VOC per-image XML annotations."""
    xmls = [f for f in files if _ext(f) == "xml"]
    img_dirs = _image_dirs(files)
    if len(xmls) >= 2 and img_dirs:
        xdir = _common_dir(xmls)
        idir, ifiles = max(img_dirs.items(), key=lambda kv: len(kv[1]))
        iext = Counter(_ext(x) for x in ifiles).most_common(1)[0][0]
        spec = [
            _spec_image("image", "input", f"{idir}/{{id}}.{iext}"),
            {"name": "boxes", "kind": "bboxes", "role": "gt", "loader": "voc",
             "pattern": f"{xdir}/{{id}}.xml", "one_indexed": True},
        ]
        return _result("F", "Detection — Pascal VOC XML", spec,
                       needs_conversion=["voc"])
    return None


def _detect_yolo(files, peek):
    """Shape G — YOLO per-image TXT label files (5 floats) + image dir."""
    txts = [f for f in files if _ext(f) == "txt"
            and "label" in f.lower()]
    img_dirs = _image_dirs(files)
    if len(txts) >= 2 and img_dirs:
        tdir = _common_dir(txts)
        idir, ifiles = max(img_dirs.items(), key=lambda kv: len(kv[1]))
        iext = Counter(_ext(x) for x in ifiles).most_common(1)[0][0]
        names_file = next((f for f in files
                           if f.rsplit("/", 1)[-1].lower()
                           in ("classes.txt", "obj.names", "data.yaml")), None)
        spec = [
            _spec_image("image", "input", f"{idir}/{{id}}.{iext}"),
            {"name": "boxes", "kind": "bboxes", "role": "gt", "loader": "yolo",
             "pattern": f"{tdir}/{{id}}.txt", "names_file": names_file},
        ]
        return _result("G", "Detection — YOLO TXT", spec,
                       needs_conversion=["yolo"])
    return None


# Dir-name tokens that mark the GT/target side of a parallel pair.
_PAIR_GT_WORDS = (_MASK_NAME_WORDS | {
    "gt", "groundtruth", "ground_truth", "target", "clean", "label", "labels",
    "output", "sharp", "hr", "high", "denoised", "original"})
# The full vocabulary that makes a parallel-dir pair *meaningful* (vs two
# arbitrary class folders): either side being a modality/target word.
_PAIR_VOCAB = (_MODALITY_WORDS | _PAIR_GT_WORDS | {
    "noisy", "noise", "dirty", "blur", "blurry", "blurred", "lr", "low",
    "source", "before", "after", "in", "out"})


def _dir_tokens(d):
    return set(re.split(r"[\s_/-]+", d.lower()))


def _detect_paired_dirs(files, peek):
    """Shapes H & N — two parallel single-level dirs of files joined by id.
    One side image input; the other → Mask (segmentation, H) when its name
    reads as a mask, else Image gt (restoration pair, N). Only fires when at
    least one dir name is a modality/target word — otherwise two arbitrary
    sibling dirs are class folders (shape C), not a paired modality."""
    img_dirs = _image_dirs(files)
    if len(img_dirs) != 2:
        return None
    (da, fa), (db, fb) = sorted(img_dirs.items(), key=lambda kv: kv[0].lower())
    if not (_dir_tokens(da) & _PAIR_VOCAB or _dir_tokens(db) & _PAIR_VOCAB):
        return None                       # arbitrary names → class folders
    # Names that look like the GT/target side.
    def _is_target(d):
        return bool(_dir_tokens(d) & _PAIR_GT_WORDS)
    if _is_target(da) and not _is_target(db):
        in_dir, gt_dir = db, da
    else:
        in_dir, gt_dir = da, db
    gt_is_mask = bool(_dir_tokens(gt_dir) & _MASK_NAME_WORDS)
    iext = Counter(_ext(x) for x in img_dirs[in_dir]).most_common(1)[0][0]
    gext = Counter(_ext(x) for x in img_dirs[gt_dir]).most_common(1)[0][0]
    if gt_is_mask:
        spec = [
            _spec_image("image", "input", f"{in_dir}/{{id}}.{iext}"),
            {"name": "mask", "kind": "mask", "role": "gt", "loader": "file",
             "pattern": f"{gt_dir}/{{id}}.{gext}"},
        ]
        return _result("H", "Segmentation — paired image+mask dirs", spec)
    spec = [
        _spec_image("image", "input", f"{in_dir}/{{id}}.{iext}"),
        _spec_image("target", "gt", f"{gt_dir}/{{id}}.{gext}"),
    ]
    return _result("N", "Restoration — paired image dirs (dirty→clean)", spec)


def _detect_image_folder(files, peek):
    """Shape C — ImageFolder: `<class>/<img>` (one folder level, leaf dirs
    are class labels, contents only images). → Image(input) + Label(gt)."""
    two_seg = [f for f in files if f.count("/") == 1]
    if len(two_seg) < 2:
        return None
    classes = {f.split("/")[0] for f in two_seg}
    if len(classes) < 2:
        return None
    if not _all_images(two_seg):
        return None
    # Reject the modality reading (image/ + mask/ …): class folders are
    # arbitrary names, not modality words.
    modality_like = sum(1 for c in classes if c.lower() in _MODALITY_WORDS)
    if modality_like / len(classes) >= 0.5:
        return None
    iext = Counter(_ext(f) for f in two_seg).most_common(1)[0][0]
    spec = [
        _spec_image("image", "input", f"{{label}}/{{id}}.{iext}"),
        {"name": "label", "kind": "label", "role": "gt",
         "loader": "token", "token": "label"},
    ]
    return _result("C", "ImageFolder (class dirs)", spec)


def _detect_image_plus_csv(files, peek):
    """Shape D — an images dir + a CSV joining filename → label/score."""
    img_dirs = _image_dirs(files)
    csvs = [f for f in files if _ext(f) == "csv"]
    if not img_dirs or not csvs:
        return None
    idir, ifiles = max(img_dirs.items(), key=lambda kv: len(kv[1]))
    # Pick the CSV that has both an id-like column and a target column.
    for csv in csvs:
        cols = _columns(peek, csv)
        if not cols:
            continue
        idc = next((c for c in cols if _ID_COL_RE.search(str(c))), None)
        if not idc:
            continue
        targets = [c for c in cols
                   if c != idc and str(c).lower() in TARGET_COL_NAMES]
        target = targets[0] if targets else (
            [c for c in cols if c != idc][-1] if len(cols) > 1 else None)
        if not target:
            continue
        rows = _rows(peek, csv)
        tvals = [r.get(target) for r in rows] if rows else []
        tkind = _classify_target(tvals)
        iext = Counter(_ext(f) for f in ifiles).most_common(1)[0][0]
        spec = [
            _spec_image("image", "input", f"{idir}/{{id}}.{iext}"),
            {"name": _safe(target), "kind": tkind, "role": "gt",
             "loader": "csv", "pattern": csv, "column": target,
             "id_column": idc, "id_token": "id"},
        ]
        return _result("D", "Image + CSV label map", spec)
    return None


def _detect_tabular(files, peek):
    """Shapes A/A′/A″ — a single tabular file (CSV/parquet); rows are the
    samples. Feature cols → input, target col → Label/Scalar gt."""
    tables = [f for f in files if _ext(f) in ("csv", "parquet")]
    # Prefer a train.csv; else the single table.
    train = [f for f in tables if "train" in f.rsplit("/", 1)[-1].lower()]
    cand = (train or tables)
    if not cand:
        return None
    table = cand[0]
    # Only treat as pure-tabular when there's no images/audio dir to pair.
    if _image_dirs(files) or _audio_dirs(files):
        return None
    cols = _columns(peek, table)
    if not cols:
        return None
    rows = _rows(peek, table)
    by_col = defaultdict(list)
    for r in (rows or []):
        for c in cols:
            by_col[c].append(r.get(c))
    # Target = first name-matched target col, else last column.
    target = next((c for c in cols if str(c).lower() in TARGET_COL_NAMES),
                  cols[-1])
    idc = next((c for c in cols if _ID_COL_RE.search(str(c))), None)
    tkind = _classify_target(by_col.get(target, []))
    loader = _ext(table)
    spec = []
    first = True
    for c in cols:
        if c == target or c == idc:
            continue
        fkind = _classify_feature(c, by_col.get(c, []))
        field = {"name": _safe(c), "kind": fkind, "role": "input",
                 "loader": loader, "pattern": table, "column": c}
        if first:
            field["index"] = True          # rows enumerate samples
            first = False
        spec.append(field)
    tfield = {"name": _safe(target), "kind": tkind, "role": "gt",
              "loader": loader, "pattern": table, "column": target}
    if first:                              # target was the only usable column
        tfield["index"] = True
    spec.append(tfield)
    label = {"label": "Tabular classification", "scalar": "Tabular regression"}
    code = "A" if tkind == "label" else "A_prime"
    text_in = any(fld["kind"] == "text" for fld in spec if fld["role"] == "input")
    if text_in and tkind == "label":
        code = "A_dprime"
        label_str = "Text classification"
    else:
        label_str = label[tkind]
    return _result(code, label_str, spec)


def _detect_dicom_nifti(files, peek):
    """Shape O — medical DICOM/NIfTI. Out of Phase-1 scope (needs a
    DataTypeDef or a transcode pass); flagged not-benchmarkable here."""
    med = [f for f in files if _ext(f) == "dcm"
           or f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")]
    if len(med) >= 2:
        return _result("O", "Medical DICOM/NIfTI", [], benchmarkable=False,
                       reason="DICOM/NIfTI needs a registered data type or a "
                              "transcode pass (Phase 4); not auto-importable yet.")
    return None


# Priority order: specific annotation shapes first, generic last.
_DETECTORS = [
    _detect_rle_csv, _detect_coco, _detect_voc, _detect_yolo,
    _detect_paired_dirs, _detect_image_folder, _detect_image_plus_csv,
    _detect_tabular, _detect_dicom_nifti,
]


def detect_shape(files, *, peek=None):
    """Fingerprint an extracted Kaggle tree → a detection result (see module
    docstring). `files` is a flat list of repo-relative path strings; `peek`
    is an optional callable `peek(path) -> {'columns': [...], 'rows': [...]}`
    (or a dict mapping path → that) used for tabular/JSON shape inference.
    Returns a not-benchmarkable result rather than raising when nothing
    matches."""
    usable, hidden = partition_hidden_gt(files)
    if not usable:
        return _result(None, "Unrecognised", [], benchmarkable=False,
                       reason="No labelled data found — only withheld "
                              "competition test files.",
                       hidden_gt=hidden)
    for det in _DETECTORS:
        res = det(usable, peek)
        if res:
            res["hidden_gt"] = hidden
            if hidden:
                res["warnings"].append(
                    f"{len(hidden)} competition test file(s) excluded from "
                    f"ground truth (hidden labels).")
            return res
    return _result(None, "Unrecognised", [], benchmarkable=False,
                   reason="Could not match a known Kaggle dataset shape; use "
                          "the manual file-tree mapping.",
                   hidden_gt=hidden)


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def _result(shape, label, spec, *, benchmarkable=True, reason="",
            warnings=None, needs_conversion=None, hidden_gt=None):
    return {
        "shape": shape, "shape_label": label, "spec": spec,
        "benchmarkable": benchmarkable and bool(spec),
        "reason": reason,
        "warnings": list(warnings or []),
        "needs_conversion": list(needs_conversion or []),
        "hidden_gt": list(hidden_gt or []),
    }


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name)).strip("_") or "field"


def _image_dirs(files):
    """Directory → its image files, for dirs that hold ≥2 images (the
    leaf directory of each image path)."""
    out = defaultdict(list)
    for f in files:
        if "/" in f and _ext(f) in _IMAGE_EXTS:
            out[f.rsplit("/", 1)[0]].append(f)
    return {d: fs for d, fs in out.items() if len(fs) >= 2}


def _audio_dirs(files):
    out = defaultdict(list)
    for f in files:
        if "/" in f and _ext(f) in _AUDIO_EXTS:
            out[f.rsplit("/", 1)[0]].append(f)
    return {d: fs for d, fs in out.items() if len(fs) >= 2}


def _common_dir(paths):
    dirs = {p.rsplit("/", 1)[0] if "/" in p else "" for p in paths}
    return max(dirs, key=lambda d: sum(1 for p in paths
                                       if p.rsplit("/", 1)[0] == d))


def _peek_get(peek, path):
    if peek is None:
        return None
    if callable(peek):
        try:
            return peek(path)
        except Exception:
            return None
    return peek.get(path)


def _columns(peek, path):
    info = _peek_get(peek, path) or {}
    return list(info.get("columns") or [])


def _rows(peek, path):
    info = _peek_get(peek, path) or {}
    return list(info.get("rows") or [])


def _json_keys(peek, path):
    info = _peek_get(peek, path) or {}
    return list(info.get("keys") or info.get("columns") or [])
