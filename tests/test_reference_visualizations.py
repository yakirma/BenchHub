"""The seeded reference visualizations (confusion_matrix)."""
from __future__ import annotations

from app import GlobalVisualization, User, db
from scripts.seed_reference_visualizations import (
    _CONFUSION_MATRIX_CODE,
    seed_reference_visualizations,
)


def test_seed_creates_confusion_matrix(db_session):
    db.session.add(User(email="adm@bench.local", display_name="adm",
                        oauth_provider="github", oauth_sub="adm-1",
                        is_admin=True))
    db.session.commit()
    out = seed_reference_visualizations()
    assert "confusion_matrix" in out
    gv = GlobalVisualization.query.filter_by(name="confusion_matrix").one()
    assert gv.is_aggregated is True
    assert gv.visibility == "public"


def test_seed_is_idempotent(db_session):
    seed_reference_visualizations()
    first = GlobalVisualization.query.filter_by(name="confusion_matrix").one().id
    seed_reference_visualizations()
    rows = GlobalVisualization.query.filter_by(name="confusion_matrix").all()
    assert len(rows) == 1
    assert rows[0].id == first  # upsert, not duplicate


def test_confusion_matrix_code_returns_image():
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    scope = {"np": np, "plt": plt, "Image": Image}
    exec(_CONFUSION_MATRIX_CODE, scope)
    fn = scope["confusion_matrix"]

    # JSON-encoded single labels.
    img = fn(["0", "1", "2", "1"], ["0", "1", "1", "1"])
    assert isinstance(img, Image.Image)
    # Top-K list predictions (uses top-1).
    img2 = fn(["3", "8"], ["[3, 5, 9]", "[1, 8, 0]"])
    assert isinstance(img2, Image.Image)
    # Empty / all-None inputs still return an image (no crash).
    img3 = fn([], [])
    assert isinstance(img3, Image.Image)
