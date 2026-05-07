"""prune_incomplete_datasets() — boot-time cleanup of half-uploaded datasets."""
import os

import pytest

from app import (
    CustomField,
    Dataset,
    Sample,
    app,
    db,
    process_dataset_zip,
    prune_incomplete_datasets,
)


def _datasets_root():
    return os.path.join(app.config['UPLOAD_FOLDER'], 'datasets')


def test_prune_keeps_fully_uploaded_dataset(client, db_session, make_zip):
    """A real ingest should leave the dataset alone."""
    zip_path = make_zip(
        "complete.zip",
        {"config/s1.json": '{"k": 1}', "tags/s1.txt": "x"},
        root_folder="complete_ds",
    )
    success, _, ds_id = process_dataset_zip(zip_path, "complete_ds")
    assert success

    removed = prune_incomplete_datasets()
    assert removed == 0
    assert Dataset.query.get(ds_id) is not None


def test_prune_removes_dataset_with_zero_samples(client, db_session):
    """Direct DB row insert (no samples, no folder) → looks like a crash
    mid-upload. Prune deletes it."""
    ds = Dataset(name="zombie_ds")
    db.session.add(ds)
    db.session.commit()
    ds_id = ds.id

    removed = prune_incomplete_datasets()
    assert removed == 1
    assert Dataset.query.get(ds_id) is None


def test_prune_removes_dataset_when_folder_missing(client, db_session, make_zip):
    """A dataset row with samples but no on-disk folder is also broken
    (someone wiped the volume). Drop it."""
    zip_path = make_zip(
        "vanished.zip",
        {"config/s1.json": '{"k": 1}'},
        root_folder="vanished_ds",
    )
    success, _, ds_id = process_dataset_zip(zip_path, "vanished_ds")
    assert success

    # Simulate the volume-loss scenario.
    import shutil as _sh
    folder = os.path.join(_datasets_root(), "vanished_ds")
    _sh.rmtree(folder, ignore_errors=True)
    assert not os.path.isdir(folder)

    removed = prune_incomplete_datasets()
    assert removed == 1
    assert Dataset.query.get(ds_id) is None


def test_prune_returns_zero_when_nothing_to_clean(client, db_session):
    """No datasets at all → no-op, returns 0."""
    assert prune_incomplete_datasets() == 0


def test_prune_cascades_samples_and_custom_fields(client, db_session):
    """Deleting an incomplete dataset still cleans its child rows (the
    SQLAlchemy cascade on Sample / CustomField). Otherwise we'd accumulate
    orphans even as we tidy datasets."""
    ds = Dataset(name="leaky_ds")
    db.session.add(ds); db.session.flush()
    sample = Sample(dataset_id=ds.id, name="s1")
    db.session.add(sample); db.session.flush()
    cf = CustomField(name="metric_x", field_type="scalar",
                     value_float=0.5, sample_id=sample.id)
    db.session.add(cf); db.session.commit()
    sample_id, cf_id = sample.id, cf.id

    # Force the "incomplete" condition by *not* creating the on-disk folder.
    # (The cascade delete should still fire because Dataset → Sample +
    # Sample → CustomField are both delete-cascade.)
    folder = os.path.join(_datasets_root(), "leaky_ds")
    if os.path.isdir(folder):
        import shutil as _sh; _sh.rmtree(folder)

    removed = prune_incomplete_datasets()
    assert removed == 1
    assert Sample.query.get(sample_id) is None
    assert CustomField.query.get(cf_id) is None
