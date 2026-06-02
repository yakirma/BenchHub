"""Add a composite `instance` mask field to the cached ADE20K dataset (id 92).

The HF source `1aurent/ADE20K` ships `instances`: a variable-length stack
of per-object binary masks. The original import kept only `segmentations`
(-> `mask`) and dropped `instances`. This backfills a single `instance`
field where each object instance gets a unique id, mirroring how `mask`
is stored: a 16-bit `.classid.png` (instance ids) + a palette-colored
`.jpg` preview, with one CustomField per sample.

Idempotent: drops any existing `instance` field/CFs first.
Run:  ~/benchhub/.venv/bin/python scripts/add_ade20k_instance_field.py
"""
import os
import numpy as np
from PIL import Image

import app
from app import db, Dataset, DatasetField, CustomField, Sample
from benchhub.preview import mask_preview

DS_ID = 92
REPO = "1aurent/ADE20K"
SPLIT = "validation"
FIELD = "instance"


def composite_instance_map(instances):
    """Overlay per-object binary layers into one instance-id map.
    Pixels where a layer == 255 are assigned that layer's 1-based id;
    later layers win on overlap."""
    if not instances:
        return None
    h, w = np.array(instances[0]).shape[:2]
    out = np.zeros((h, w), dtype=np.uint16)
    for k, layer in enumerate(instances, start=1):
        a = np.array(layer)
        if a.ndim == 3:
            a = a[..., 0]
        out[a == 255] = k
    return out


def main():
    with app.app.app_context():
        ds = Dataset.query.get(DS_ID)
        assert ds and "ADE20K" in ds.name, f"unexpected dataset {ds}"
        upload_root = app.app.config["UPLOAD_FOLDER"]
        out_dir = os.path.join(upload_root, "datasets", str(DS_ID), FIELD)
        os.makedirs(out_dir, exist_ok=True)

        # filename -> sample.id (robust to ordering)
        rows = (
            db.session.query(Sample.id, CustomField.value_text)
            .join(CustomField, CustomField.sample_id == Sample.id)
            .filter(Sample.dataset_id == DS_ID, CustomField.name == "filename")
            .all()
        )
        fn_to_sid = {vt: sid for sid, vt in rows}
        name_to_sid = {
            s.name: s.id for s in Sample.query.filter_by(dataset_id=DS_ID).all()
        }
        print(f"{len(fn_to_sid)} filename->sample, {len(name_to_sid)} samples")

        # Clean slate (idempotent re-run).
        sids = list(name_to_sid.values())
        CustomField.query.filter(
            CustomField.sample_id.in_(sids), CustomField.name == FIELD
        ).delete(synchronize_session=False)
        DatasetField.query.filter_by(dataset_id=DS_ID, name=FIELD).delete(
            synchronize_session=False
        )
        db.session.commit()

        db.session.add(
            DatasetField(dataset_id=DS_ID, name=FIELD, kind="mask", role="gt")
        )
        db.session.commit()

        from datasets import load_dataset

        stream = load_dataset(REPO, split=SPLIT, streaming=True)
        done = miss = empty = 0
        for i, row in enumerate(stream):
            sid = fn_to_sid.get(row["filename"]) or name_to_sid.get(f"s_{i:05d}")
            if sid is None:
                miss += 1
                continue
            inst = composite_instance_map(row.get("instances"))
            if inst is None:
                empty += 1
                continue
            base = f"s_{i:05d}"
            # 16-bit instance-id PNG (canonical) + palette preview JPG.
            Image.fromarray(inst.astype(np.uint16), mode="I;16").save(
                os.path.join(out_dir, base + ".classid.png")
            )
            with open(os.path.join(out_dir, base + ".jpg"), "wb") as f:
                f.write(mask_preview(inst))
            db.session.add(
                CustomField(
                    sample_id=sid,
                    name=FIELD,
                    data_type="mask",
                    value_text=f"datasets/{DS_ID}/{FIELD}/{base}.jpg",
                )
            )
            done += 1
            if done % 250 == 0:
                db.session.commit()
                print(f"  ...{done} done")
        db.session.commit()
        print(f"DONE done={done} miss={miss} empty={empty}")


if __name__ == "__main__":
    main()
