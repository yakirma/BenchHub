"""One-shot migration: convert pre-refactor `Dataset.storage_mode='hf-pointer'`
rows into HF-ref `Attachment` rows on whichever LBs they were attached
to, then delete the Dataset / Sample / CustomField bookkeeping.

Run on the fly machine:
    python scripts/migrate_pointer_to_attachment.py [--yes]

Without --yes the script prints what it WOULD do and exits.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    app, db,
    Attachment, CustomField, Dataset, Leaderboard, Sample,
    leaderboard_datasets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes', action='store_true',
                        help='Actually delete + insert. Without this we just print.')
    args = parser.parse_args()
    with app.app_context():
        ptr = Dataset.query.filter_by(storage_mode='hf-pointer').all()
        print(f"Found {len(ptr)} pointer-mode datasets.")
        if not ptr:
            return
        for ds in ptr:
            try:
                meta = json.loads(ds.source_metadata or '{}')
            except (TypeError, ValueError):
                meta = {}
            repo_id = meta.get('repo_id')
            if not repo_id:
                print(f"  [skip] dataset {ds.id} ({ds.name}) has no repo_id "
                      f"in source_metadata — orphaned pointer row, will delete.")
                continue
            attached_lbs = db.session.execute(
                leaderboard_datasets.select().where(
                    leaderboard_datasets.c.dataset_id == ds.id
                )
            ).all()
            print(f"  dataset {ds.id} ({ds.name}) → repo_id={repo_id}, "
                  f"attached to {len(attached_lbs)} leaderboards")
            for row in attached_lbs:
                lb_id = row.leaderboard_id
                role = getattr(row, 'role', 'primary') or 'primary'
                already = (Attachment.query
                           .filter_by(leaderboard_id=lb_id,
                                      hf_repo_id=repo_id)
                           .first())
                if already:
                    print(f"    [keep] LB {lb_id}: HF-ref attachment already exists")
                    continue
                print(f"    [add] LB {lb_id}: HF-ref attachment for {repo_id} (role={role})")
                if args.yes:
                    db.session.add(Attachment(
                        leaderboard_id=lb_id,
                        hf_repo_id=repo_id,
                        hf_revision=meta.get('revision'),
                        hf_split=meta.get('split') or 'train',
                        hf_mapping_json=json.dumps(meta.get('mapping') or []),
                        role=role,
                    ))

        if args.yes:
            db.session.commit()
            # Delete the pointer-mode bookkeeping (cascade handles
            # Sample + CustomField via the dataset relationships).
            for ds in ptr:
                CustomField.query.filter(
                    CustomField.sample_id.in_(
                        db.session.query(Sample.id).filter_by(dataset_id=ds.id)
                    )
                ).delete(synchronize_session=False)
                Sample.query.filter_by(dataset_id=ds.id).delete()
                # Also drop legacy m2m rows pointing at this dataset
                # (the Attachment we just added is canonical now).
                db.session.execute(
                    leaderboard_datasets.delete().where(
                        leaderboard_datasets.c.dataset_id == ds.id
                    )
                )
                db.session.delete(ds)
            db.session.commit()
            print(f"\nDone. Removed {len(ptr)} pointer-mode datasets + "
                  f"their samples + custom_fields.")
        else:
            print("\nDry-run only. Re-run with --yes to apply.")


if __name__ == '__main__':
    main()
