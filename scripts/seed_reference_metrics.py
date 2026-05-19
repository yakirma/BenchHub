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
def accuracy(gt, pred):
    \"\"\"Per-sample classification accuracy. gt + pred are bh.Label
    instances; the .value attribute holds the int / str class id.
    Pool with mean across the LB for overall accuracy.\"\"\"
    if gt is None or pred is None:
        return 0.0
    return 1.0 if gt.value == pred.value else 0.0
"""

_RMSE_DEPTH = """\
def rmse(gt, pred):
    \"\"\"Per-sample root mean squared error for Depth predictions.
    Handles unit mismatch by normalising both to meters.\"\"\"
    import numpy as np
    if gt is None or pred is None:
        return float('nan')
    g = gt.array.astype(np.float32)
    p = pred.array.astype(np.float32)
    # Unit normalisation — Depth carries unit in .unit.
    def to_meters(arr, unit):
        if unit == 'millimeters':
            return arr / 1000.0
        return arr  # 'meters' or 'unitless'
    g = to_meters(g, gt.unit)
    p = to_meters(p, pred.unit)
    mask = np.isfinite(g) & np.isfinite(p)
    if not mask.any():
        return float('nan')
    diff = (g[mask] - p[mask]) ** 2
    return float(np.sqrt(diff.mean()))
"""

_MAE_DEPTH = """\
def mae(gt, pred):
    \"\"\"Per-sample mean absolute error for Depth, in meters.\"\"\"
    import numpy as np
    if gt is None or pred is None:
        return float('nan')
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
def iou(gt, pred):
    \"\"\"Per-sample mean IoU across all class IDs for a segmentation
    Mask. ignore_index is honored when set on the gt Mask.\"\"\"
    import numpy as np
    if gt is None or pred is None:
        return float('nan')
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

_TEXT_EM = """\
def exact_match(gt, pred):
    \"\"\"Per-sample exact-match for Text predictions. Strips whitespace
    and lowercases for a forgiving compare; tighter variants can layer
    tokenisation on top.\"\"\"
    if gt is None or pred is None:
        return 0.0
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
        "description": "Per-sample classification accuracy (higher is better). Pool with mean.",
        "input_kinds": ["label", "label"],
        "python_code": _ACCURACY,
    },
    {
        "name": "rmse_depth",
        "description": "Root-mean-squared error for Depth predictions in meters (lower is better).",
        "input_kinds": ["depth", "depth"],
        "python_code": _RMSE_DEPTH,
    },
    {
        "name": "mae_depth",
        "description": "Mean absolute error for Depth predictions in meters (lower is better).",
        "input_kinds": ["depth", "depth"],
        "python_code": _MAE_DEPTH,
    },
    {
        "name": "iou_mask",
        "description": "Per-sample mean IoU across class IDs for segmentation Masks (higher is better).",
        "input_kinds": ["mask", "mask"],
        "python_code": _IOU_MASK,
    },
    {
        "name": "exact_match_text",
        "description": "Per-sample exact-match for Text predictions, whitespace + case insensitive (higher is better).",
        "input_kinds": ["text", "text"],
        "python_code": _TEXT_EM,
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
