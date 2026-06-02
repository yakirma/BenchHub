"""Validate + remove redundant path/folder/filename bookkeeping columns
from ALL datasets, once the real data has been extracted into typed fields.

A column is a removal candidate when its name is a pure file-system
locator: `filename`, `file_name`, `folder`, `directory`, `filepath`,
`file_path`, `path`, or anything ending `_path`. Identifier columns
(`id`, `image_id`), provenance (`source`), and dimensions (`width`,
`height`, `date_captured`) are NOT touched — they're not paths.

Validation: a `*_path`/`filepath`/`path` pointer is only dropped when the
dataset already has a file-backed data field (image/mask/depth/audio) —
proof the pointed-at bytes were extracted. `filename`/`folder`/`directory`
are pure metadata and always safe.

Dry-run by default; pass `--apply` to actually delete.
Run:  PYTHONPATH=<repo> ~/benchhub/.venv/bin/python scripts/prune_path_columns.py [--apply]
"""
import sys

import app
from app import db, Dataset, DatasetField, CustomField, Sample

NAME_EXACT = {"filename", "file_name", "folder", "directory", "dir",
              "filepath", "file_path", "path"}
FILE_BACKED = {"image", "mask", "depth", "audio"}


def is_candidate(name):
    n = (name or "").strip().lower()
    return n in NAME_EXACT or n.endswith("_path")


def is_path_pointer(name):
    n = (name or "").strip().lower()
    return n in {"filepath", "file_path", "path"} or n.endswith("_path")


def main(apply=False):
    with app.app.app_context():
        removed = 0
        for ds in Dataset.query.order_by(Dataset.id).all():
            fields = list(ds.fields)
            kinds = {f.kind for f in fields}
            for f in fields:
                if not is_candidate(f.name):
                    continue
                # A path pointer is only safe to drop if the real bytes
                # were extracted into a file-backed field.
                if is_path_pointer(f.name) and not (kinds & FILE_BACKED):
                    print(f"  SKIP ds {ds.id} '{f.name}': path pointer but no "
                          f"file-backed field extracted (kinds={sorted(kinds)})")
                    continue
                sids = [r[0] for r in db.session.query(Sample.id)
                        .filter(Sample.dataset_id == ds.id)]
                ncf = (CustomField.query
                       .filter(CustomField.sample_id.in_(sids),
                               CustomField.name == f.name).count()) if sids else 0
                print(f"  {'DROP' if apply else 'WOULD DROP'} ds {ds.id} "
                      f"({ds.name}) field '{f.name}' [{f.kind}] — {ncf} CFs")
                if apply:
                    if sids:
                        CustomField.query.filter(
                            CustomField.sample_id.in_(sids),
                            CustomField.name == f.name,
                        ).delete(synchronize_session=False)
                    DatasetField.query.filter_by(
                        dataset_id=ds.id, name=f.name).delete(
                        synchronize_session=False)
                    removed += 1
        if apply:
            db.session.commit()
        print(f"\n{'Removed' if apply else 'Would remove'} {removed} field(s).")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
