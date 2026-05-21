"""Regression tests for prune_incomplete_datasets.

The typed-dataset importer writes per-dataset bytes under
`uploads/datasets/<id>/`. The prune helper had been looking at the
legacy `uploads/datasets/<secure_filename(name)>/` layout for so
long that every reboot decided HF-imported datasets were incomplete
and cascade-deleted them. The layout is now id-only; nothing in the
codebase still writes the name-based layout, so the prune treats it
as absent.
"""
import os

import pytest

from app import (
    Dataset,
    Sample,
    app as flask_app,
    db,
    prune_incomplete_datasets,
)


def _make_dataset(name, *, samples=1, storage_mode="local"):
    ds = Dataset(name=name, storage_mode=storage_mode)
    db.session.add(ds)
    db.session.flush()
    for i in range(samples):
        db.session.add(Sample(dataset_id=ds.id, name=f"s{i}"))
    db.session.commit()
    return ds


def test_prune_keeps_dataset_with_id_based_folder(db_session):
    """The typed importer writes to uploads/datasets/<id>/. The prune
    helper must accept that as a sign of completion."""
    ds = _make_dataset("cifar10_id_layout", samples=3)
    folder = os.path.join(flask_app.config["UPLOAD_FOLDER"], "datasets", str(ds.id))
    os.makedirs(folder, exist_ok=True)

    removed = prune_incomplete_datasets()
    assert removed == 0
    assert Dataset.query.get(ds.id) is not None


def test_prune_removes_dataset_with_only_legacy_name_based_folder(db_session):
    """The legacy uploads/datasets/<safe-name>/ layout is no longer
    produced by anything in the codebase, so a row whose only bytes
    live there is treated as incomplete (and the prune removes the
    orphan)."""
    ds = _make_dataset("legacy_named", samples=2)
    ds_id = ds.id
    folder = os.path.join(flask_app.config["UPLOAD_FOLDER"], "datasets", "legacy_named")
    os.makedirs(folder, exist_ok=True)

    removed = prune_incomplete_datasets()
    assert removed == 1
    assert Dataset.query.get(ds_id) is None


def test_prune_removes_dataset_with_no_folder_at_all(db_session):
    """Sample rows exist but no on-disk bytes → still incomplete."""
    ds = _make_dataset("orphaned", samples=1)
    ds_id = ds.id

    removed = prune_incomplete_datasets()
    assert removed == 1
    assert Dataset.query.get(ds_id) is None


def test_prune_removes_dataset_with_zero_samples(db_session):
    """Folder present but zero Sample rows → still incomplete."""
    ds = _make_dataset("no_samples", samples=0)
    ds_id = ds.id
    folder = os.path.join(flask_app.config["UPLOAD_FOLDER"], "datasets", str(ds_id))
    os.makedirs(folder, exist_ok=True)

    removed = prune_incomplete_datasets()
    assert removed == 1
    assert Dataset.query.get(ds_id) is None
