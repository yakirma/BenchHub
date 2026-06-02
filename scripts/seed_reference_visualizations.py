"""Seed the curated reference visualizations (idempotent upsert).

Run standalone:
    python scripts/seed_reference_visualizations.py

Also called on boot from app.run_migrations() so the curated set is
always present + up to date (same pattern as seed_reference_metrics).
"""
from __future__ import annotations

import json

from app import GlobalVisualization, User, app, db


# Confusion matrix — aggregated. Receives gt + pred as LISTS of inline
# label values across every sample (each value is whatever
# CustomField.get_value() returns for a label field: usually a
# JSON-encoded int like "3", or a top-K JSON list "[3, 5, ...]" for
# label_list). Returns a PIL.Image heatmap.
_CONFUSION_MATRIX_CODE = '''
def confusion_matrix(gt, pred):
    import json
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    def _idx(v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                pass
        if isinstance(v, (list, tuple)):
            v = v[0] if len(v) else None  # top-1 of a top-K list
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    pairs = [(_idx(g), _idx(p)) for g, p in zip(gt or [], pred or [])]
    pairs = [(g, p) for g, p in pairs if g is not None and p is not None]

    if not pairs:
        img = Image.new("RGB", (360, 80), (255, 255, 255))
        return img

    n = max(max(g, p) for g, p in pairs) + 1
    cm = np.zeros((n, n), dtype=int)
    for g, p in pairs:
        cm[g, p] += 1

    side = min(1.2 + 0.55 * n, 12)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(range(n), fontsize=7)
    thresh = cm.max() / 2.0 if cm.max() else 0
    if n <= 25:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
'''.strip()


_SEED = [
    {
        "name": "confusion_matrix",
        "description": (
            "Aggregated confusion matrix for classification: counts of "
            "(ground-truth class, predicted class) across all samples. "
            "Map gt → the label GT field and pred → the predicted label "
            "field (top-1 is used for top-K predictions)."
        ),
        "python_code": _CONFUSION_MATRIX_CODE,
        "is_aggregated": True,
        "accepts_aggregated_inputs": True,
        "input_kinds": ["label", "label"],
    },
]


def seed_reference_visualizations(owner_email: str | None = None) -> dict:
    """Upsert the reference visualizations. Returns a {name: id} map."""
    with app.app_context():
        if owner_email:
            owner = User.query.filter_by(email=owner_email).first()
        else:
            owner = User.query.filter_by(is_admin=True).first()

        result: dict[str, int] = {}
        for entry in _SEED:
            existing = GlobalVisualization.query.filter_by(name=entry["name"]).first()
            payload = dict(entry)
            payload["input_kinds"] = json.dumps(payload["input_kinds"])
            if existing:
                for k, v in payload.items():
                    setattr(existing, k, v)
                row = existing
            else:
                row = GlobalVisualization(
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
    out = seed_reference_visualizations()
    print(json.dumps(out, indent=2))
