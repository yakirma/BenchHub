#!/usr/bin/env python
"""Seed reference typed metrics + visualizations into BenchHub.

These are the curated metrics the Phase A/B typed contract was built
around — each declares `input_kinds` so the metric engine hands it
fully-decoded `DataType` instances (not raw numpy arrays). Idempotent:
re-running upserts each metric by `name` on the system admin's user.

Usage:
    python scripts/seed_reference_metrics.py
"""
from __future__ import annotations

import json
import sys

from app import GlobalMetric, User, app, db


# Metric source code — each receives typed instances per its input_kinds
# declaration. Bodies are small enough that we inline them as string
# literals here; the seeder writes them to GlobalMetric.python_code.

_ACCURACY = """\
import benchhub as bh

def accuracy(gt: bh.Label, pred: bh.Label | bh.LabelList):
    \"\"\"Per-sample classification accuracy. Accepts either a single
    bh.Label pred (exact-match against gt) OR a bh.LabelList top-K
    pred (degraded to its first entry — same shape as top-1). One
    metric works for both pred contracts.\"\"\"
    if gt is None or pred is None:
        return 0.0
    assert isinstance(gt, bh.Label), f"gt must be bh.Label, got {type(gt).__name__}"
    assert isinstance(pred, (bh.Label, bh.LabelList)), \\
        f"pred must be bh.Label or bh.LabelList, got {type(pred).__name__}"
    if isinstance(pred, bh.LabelList):
        if not pred.values:
            return 0.0
        pred_value = pred.values[0]
    else:
        pred_value = pred.value
    return 1.0 if gt.value == pred_value else 0.0
"""

_RMSE_DEPTH = """\
import numpy as np
import benchhub as bh

def rmse_depth(gt: bh.Depth, pred: bh.Depth):
    \"\"\"Plain depth RMSE in METERS (no scale/affine alignment). Both GT and
    prediction are put in DEPTH space (inverting whichever is flagged
    is_inverse), GT converted to meters, pred resized to the GT grid.
    Rewards true metric-depth models; penalises relative/inverse-depth
    output left unaligned.\"\"\"
    def arr(x):
        if x is None:
            return None
        a = x.array if hasattr(x, "array") else x
        a = np.asarray(a, dtype=np.float64)
        return a[..., 0] if a.ndim == 3 else a
    def to_m(a, unit):
        if unit == "millimeters":
            return a / 1000.0
        if unit == "centimeters":
            return a / 100.0
        return a
    def resize_nn(a, shape):
        if a.shape == shape:
            return a
        H, W = shape
        yi = np.clip(np.round(np.linspace(0, a.shape[0]-1, H)).astype(int), 0, a.shape[0]-1)
        xi = np.clip(np.round(np.linspace(0, a.shape[1]-1, W)).astype(int), 0, a.shape[1]-1)
        return a[yi][:, xi]
    g = arr(gt); p = arr(pred)
    if g is None or p is None:
        return float("nan")
    g = to_m(g, getattr(gt, "unit", None))
    gd = 1.0 / np.clip(g, 1e-9, None) if getattr(gt, "is_inverse", False) else g
    pd = 1.0 / np.clip(p, 1e-9, None) if getattr(pred, "is_inverse", False) else p
    pd = resize_nn(pd, gd.shape)
    m = np.isfinite(gd) & np.isfinite(pd) & (gd > 1e-3)
    if int(m.sum()) < 10:
        return float("nan")
    return float(np.sqrt(np.mean((pd[m] - gd[m]) ** 2)))
"""

_MAE_DEPTH = """\
import numpy as np
import benchhub as bh

def mae(gt: bh.Depth, pred: bh.Depth):
    \"\"\"Per-sample mean absolute error for Depth, in meters.\"\"\"
    if gt is None or pred is None:
        return float('nan')
    assert isinstance(gt, bh.Depth),   f"gt must be bh.Depth, got {type(gt).__name__}"
    assert isinstance(pred, bh.Depth), f"pred must be bh.Depth, got {type(pred).__name__}"
    g = gt.array.astype(np.float32)
    p = pred.array.astype(np.float32)
    def to_meters(arr, unit):
        return arr / 1000.0 if unit == 'millimeters' else arr
    g = to_meters(g, gt.unit); p = to_meters(p, pred.unit)
    mask = np.isfinite(g) & np.isfinite(p)
    if not mask.any():
        return float('nan')
    return float(np.abs(g[mask] - p[mask]).mean())
"""

_IOU_MASK = """\
import numpy as np
import benchhub as bh

def iou(gt: bh.Mask, pred: bh.Mask):
    \"\"\"Per-sample mean IoU across all class IDs for a segmentation
    Mask. ignore_index is honored when set on the gt Mask.\"\"\"
    if gt is None or pred is None:
        return float('nan')
    assert isinstance(gt, bh.Mask),   f"gt must be bh.Mask, got {type(gt).__name__}"
    assert isinstance(pred, bh.Mask), f"pred must be bh.Mask, got {type(pred).__name__}"
    g = gt.array
    p = pred.array
    if g.shape != p.shape:
        return float('nan')
    ignore = getattr(gt, 'ignore_index', 255)
    valid = g != ignore
    if not valid.any():
        return float('nan')
    g = g[valid]
    p = p[valid]
    classes = np.unique(np.concatenate([np.unique(g), np.unique(p)]))
    ious = []
    for c in classes:
        if c == ignore:
            continue
        inter = ((g == c) & (p == c)).sum()
        union = ((g == c) | (p == c)).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else float('nan')
"""

_TOP_1_ACC = """\
import benchhub as bh

def top_1_accuracy(gt: bh.Label, pred: bh.LabelList):
    \"\"\"Per-sample top-1 accuracy for classification with a ranked
    pred list. Checks gt.value against the FIRST entry of pred.values
    (i.e. the model's top guess).\"\"\"
    if gt is None or pred is None or not pred.values:
        return 0.0
    assert isinstance(gt, bh.Label),     f"gt must be bh.Label, got {type(gt).__name__}"
    assert isinstance(pred, bh.LabelList), f"pred must be bh.LabelList, got {type(pred).__name__}"
    return 1.0 if gt.value == pred.values[0] else 0.0
"""

_TOP_5_ACC = """\
import benchhub as bh

def top_5_accuracy(gt: bh.Label, pred: bh.LabelList):
    \"\"\"Per-sample top-5 accuracy. 1.0 if gt.value appears anywhere
    in the first 5 entries of pred.values, else 0.0.\"\"\"
    if gt is None or pred is None:
        return 0.0
    assert isinstance(gt, bh.Label),     f"gt must be bh.Label, got {type(gt).__name__}"
    assert isinstance(pred, bh.LabelList), f"pred must be bh.LabelList, got {type(pred).__name__}"
    return 1.0 if gt.value in pred.values[:5] else 0.0
"""

_TEXT_EM = """\
import benchhub as bh

def exact_match(gt: bh.Text, pred: bh.Text):
    \"\"\"Per-sample exact-match for Text predictions. Strips whitespace
    and lowercases for a forgiving compare; tighter variants can layer
    tokenisation on top.\"\"\"
    if gt is None or pred is None:
        return 0.0
    assert isinstance(gt, bh.Text),   f"gt must be bh.Text, got {type(gt).__name__}"
    assert isinstance(pred, bh.Text), f"pred must be bh.Text, got {type(pred).__name__}"
    g = gt.text.strip().lower()
    p = pred.text.strip().lower()
    return 1.0 if g == p else 0.0
"""


# NOTE: `sort_direction` lives on the per-LB binding (LeaderboardMetric),
# not on the global metric definition — each LB chooses what "better"
# means in its context. The hints below are just documentation for the
# admin wiring up a binding.
_SEED = [
    {
        "name": "accuracy",
        "description": "Per-sample classification accuracy (higher is better). Accepts a Label pred or a LabelList top-K pred (uses first entry).",
        "input_kinds": ["label", "label|label_list"],
        "input_roles": ["gt", "pred"],
        "python_code": _ACCURACY,
    },
    {
        "name": "rmse_depth",
        "description": "Root-mean-squared error for Depth predictions in meters (lower is better).",
        "input_kinds": ["depth", "depth"],
        "input_roles": ["gt", "pred"],
        "python_code": _RMSE_DEPTH,
    },
    {
        "name": "mae_depth",
        "description": "Mean absolute error for Depth predictions in meters (lower is better).",
        "input_kinds": ["depth", "depth"],
        "input_roles": ["gt", "pred"],
        "python_code": _MAE_DEPTH,
    },
    {
        "name": "iou_mask",
        "description": "Per-sample mean IoU across class IDs for segmentation Masks (higher is better).",
        "input_kinds": ["mask", "mask"],
        "input_roles": ["gt", "pred"],
        "python_code": _IOU_MASK,
    },
    {
        "name": "exact_match_text",
        "description": "Per-sample exact-match for Text predictions, whitespace + case insensitive (higher is better).",
        "input_kinds": ["text", "text"],
        "input_roles": ["gt", "pred"],
        "python_code": _TEXT_EM,
    },
    {
        "name": "top_1_accuracy",
        "description": "Per-sample top-1 accuracy: gt matches the FIRST entry of a ranked LabelList pred (higher is better).",
        "input_kinds": ["label", "label_list"],
        "input_roles": ["gt", "pred"],
        "python_code": _TOP_1_ACC,
    },
    {
        "name": "top_5_accuracy",
        "description": "Per-sample top-5 accuracy: gt appears in the first 5 entries of a ranked LabelList pred (higher is better).",
        "input_kinds": ["label", "label_list"],
        "input_roles": ["gt", "pred"],
        "python_code": _TOP_5_ACC,
    },
]


def seed_reference_metrics(owner_email: str | None = None) -> dict:
    """Upsert the reference metrics. Returns a {name: id} map."""
    with app.app_context():
        if owner_email:
            owner = User.query.filter_by(email=owner_email).first()
        else:
            owner = User.query.filter_by(is_admin=True).first()

        result: dict[str, int] = {}
        for entry in _SEED:
            existing = GlobalMetric.query.filter_by(name=entry["name"]).first()
            payload = dict(entry)
            payload["input_kinds"] = json.dumps(payload["input_kinds"])
            payload["input_roles"] = json.dumps(payload["input_roles"])
            if existing:
                for k, v in payload.items():
                    setattr(existing, k, v)
                row = existing
            else:
                row = GlobalMetric(
                    owner_user_id=owner.id if owner else None,
                    visibility="public",
                    **payload,
                )
                db.session.add(row)
            db.session.flush()
            result[row.name] = row.id
        db.session.commit()
        return result


if __name__ == "__main__":
    out = seed_reference_metrics()
    print(json.dumps(out, indent=2))
