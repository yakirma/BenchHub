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


def test_prune_keeps_pointer_mode_dataset_with_no_folder(client, db_session):
    """Pointer-mode datasets INTENTIONALLY have no on-disk folder —
    bytes live on HF, samples carry source_ref_json. The prune routine
    must not treat the missing folder as "incomplete" or it will
    silently delete every successful HF auto-import."""
    ds = Dataset(
        name='pointer_keepalive_ds', visibility='public',
        storage_mode='hf-pointer', source_kind='hf-pointer',
    )
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(
        dataset_id=ds.id, name='s00000',
        source_ref_json='{"repo_id": "fake/x", "row_idx": 0}',
    ))
    db.session.commit()
    assert prune_incomplete_datasets() == 0
    assert Dataset.query.get(ds.id) is not None


def test_prune_still_drops_pointer_dataset_with_zero_samples(client, db_session):
    """Even pointer-mode rows aren't immune to the zero-sample sweep —
    that signals a half-finished import where the metadata stream
    yielded nothing usable."""
    ds = Dataset(
        name='pointer_empty_ds', visibility='public',
        storage_mode='hf-pointer', source_kind='hf-pointer',
    )
    db.session.add(ds); db.session.commit()
    assert prune_incomplete_datasets() == 1
    assert Dataset.query.get(ds.id) is None


def test_prune_returns_zero_when_nothing_to_clean(client, db_session):
    """No datasets at all → no-op, returns 0."""
    assert prune_incomplete_datasets() == 0


def test_process_dataset_zip_self_cleans_on_failure(client, db_session, make_zip):
    """A ZIP with no recognizable samples fails inside ingest. The
    failure path must leave NO orphan Dataset row and NO leftover folder."""
    # Only __MACOSX/ entries — excluded by the sample-discovery loop, so
    # sample_names ends up empty and the function takes the "No valid
    # samples" failure path.
    zip_path = make_zip(
        "broken.zip",
        {"__MACOSX/garbage.txt": "junk"},
        root_folder="broken_ds",
    )
    success, msg, ds_id = process_dataset_zip(zip_path, "broken_ds")
    assert success is False
    assert "No valid samples" in msg
    assert ds_id is None

    # No orphan row left behind.
    assert Dataset.query.filter_by(name="broken_ds").count() == 0

    # No leftover folder either.
    folder = os.path.join(_datasets_root(), "broken_ds")
    assert not os.path.isdir(folder)


def test_datasets_list_prunes_orphans_on_view(client, db_session):
    """The /datasets route runs the prune routine inline so users don't
    wait for the next deploy to see a clean catalog."""
    # Plant an orphan directly in the DB.
    db.session.add(Dataset(name="orphan_visible_to_view"))
    db.session.commit()

    resp = client.get('/datasets')
    assert resp.status_code == 200
    # Orphan is gone after the view runs.
    assert Dataset.query.filter_by(name="orphan_visible_to_view").count() == 0


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
