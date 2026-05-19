#!/usr/bin/env python
"""Import a typed dataset from a local path directly into the BenchHub DB.

Use this on the prod box when you have a directory laid out per the
manifest spec (`benchhub/manifest.py`). Bypasses the HTTP route — runs
inside the Flask app context so it can write rows + copy files in one
transaction.

    python scripts/import_typed_dataset.py /abs/path/to/dataset_root
                                           [--owner-email you@example.com]
                                           [--visibility public|private|unlisted]
"""
import argparse
import sys

from app import CustomField, Dataset, Sample, User, app, db
from benchhub.manifest import import_typed_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="Path to a dataset directory containing manifest.json")
    p.add_argument(
        "--owner-email", default=None,
        help="Set Dataset.owner_user_id from this user's email. Defaults to "
             "the first admin user.",
    )
    p.add_argument("--visibility", default="public",
                   choices=["public", "private", "unlisted"])
    args = p.parse_args()

    with app.app_context():
        owner = None
        if args.owner_email:
            owner = User.query.filter_by(email=args.owner_email).first()
            if owner is None:
                print(f"no user found for email {args.owner_email!r}", file=sys.stderr)
                return 2
        else:
            owner = User.query.filter_by(is_admin=True).first()

        ds_id, summary = import_typed_dataset(
            args.source,
            db_session=db.session,
            Dataset=Dataset, Sample=Sample, CustomField=CustomField,
            upload_folder=app.config["UPLOAD_FOLDER"],
            owner_user_id=owner.id if owner else None,
            visibility=args.visibility,
        )
        db.session.commit()
        print(f"imported dataset id={ds_id} summary={summary}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
