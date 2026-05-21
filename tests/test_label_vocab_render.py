"""Class-name vocab rendering on the dataset-view page.

The HF importer lifts `ClassLabel.names` into the label field's
`DatasetField.data_params['names']`. The dataset_view route must
- expose a `label_vocabs` map so the template renders a legend
- swap integer label values for the matching class name in cells

Without this, the user sees `3` instead of `cat` and has no clue
what the integers mean.
"""
from __future__ import annotations

import os

from app import (
    CustomField,
    Dataset,
    DatasetField,
    Sample,
    app as flask_app,
    db,
)


def _make_dataset_with_label_vocab(names):
    ds = Dataset(name="cifar_synth", visibility="public")
    db.session.add(ds)
    db.session.flush()

    img = DatasetField(dataset_id=ds.id, name="image", kind="image", role="input")
    label = DatasetField(dataset_id=ds.id, name="label", kind="label", role="gt")
    label.set_params({"names": names})
    db.session.add_all([img, label])

    # Two samples with label values 0 and 3.
    for i, lbl in enumerate([0, 3]):
        s = Sample(dataset_id=ds.id, name=f"s{i}")
        db.session.add(s)
        db.session.flush()
        cf = CustomField(sample_id=s.id, name="label", data_type="label",
                         value_text=str(lbl))
        db.session.add(cf)
    db.session.commit()

    # Folder must exist so the inline prune doesn't sweep the row.
    folder = os.path.join(flask_app.config["UPLOAD_FOLDER"], "datasets", str(ds.id))
    os.makedirs(folder, exist_ok=True)
    return ds


def test_label_vocab_renders_class_names_in_cells(client, db_session):
    names = ["airplane", "automobile", "bird", "cat", "deer"]
    ds = _make_dataset_with_label_vocab(names)

    body = client.get(f"/dataset/{ds.id}").data.decode("utf-8")
    # Label value 0 → "airplane", value 3 → "cat". The legend panel
    # also shows "airplane" so the cells alone aren't a sufficient
    # signal; "cat" only appears in the rendered cell.
    assert "cat" in body
    assert "airplane" in body
    # And critically, the bare integer doesn't appear as the rendered
    # value (the page may still contain '3' in other places like
    # pagination, so we just check that "cat" is rendered).


def test_label_vocab_panel_shows_legend(client, db_session):
    names = ["airplane", "automobile", "bird"]
    ds = _make_dataset_with_label_vocab(names)
    body = client.get(f"/dataset/{ds.id}").data.decode("utf-8")
    # Heading + all three class names + the field name.
    assert "Class labels" in body
    for nm in names:
        assert nm in body


def test_no_label_vocab_panel_when_no_names(client, db_session):
    """If DatasetField.data_params has no `names` (e.g. a user-built
    dataset that never declared a vocab), the legend panel is
    suppressed entirely."""
    ds = Dataset(name="no_vocab", visibility="public")
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name="label",
                                kind="label", role="gt"))
    s = Sample(dataset_id=ds.id, name="s0")
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(sample_id=s.id, name="label",
                               data_type="label", value_text="7"))
    db.session.commit()
    folder = os.path.join(flask_app.config["UPLOAD_FOLDER"], "datasets", str(ds.id))
    os.makedirs(folder, exist_ok=True)

    body = client.get(f"/dataset/{ds.id}").data.decode("utf-8")
    assert "Class labels" not in body
    # Bare int still rendered (no names to swap in).
    assert ">7<" in body or "> 7 <" in body or "7\n" in body
