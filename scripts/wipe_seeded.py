"""Nuclear-option wipe for seeded datasets / leaderboards / metrics.

Run on the fly machine when you want a clean slate before re-seeding:

    python scripts/wipe_seeded.py --yes

Without --yes the script prints the counts and exits — handy as a
dry-run before committing to the delete.

What gets removed (in order, to satisfy FK constraints):

    1. Submission rows + their on-disk submission folders.
    2. LeaderboardMetric / LeaderboardVisualization link rows.
    3. MetricResult rows.
    4. Leaderboard rows.
    5. CustomField rows for samples + submissions.
    6. Sample rows.
    7. Dataset rows + their on-disk dataset folders.
    8. GlobalMetric + GlobalVisualization rows.
    9. UserColabGist rows + Tag rows that are no longer referenced.

Users, AuthorProfile, and OAuth state are NOT touched. Same for
GlobalSettings and api_tokens.
"""
import argparse
import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    app, db,
    Dataset, Sample, Submission, Leaderboard,
    LeaderboardMetric, LeaderboardVisualization,
    GlobalMetric, GlobalVisualization,
    MetricResult, CustomField, Tag, UserColabGist,
)


def _counts():
    return {
        'datasets':          Dataset.query.count(),
        'samples':           Sample.query.count(),
        'leaderboards':      Leaderboard.query.count(),
        'submissions':       Submission.query.count(),
        'lb_metrics':        LeaderboardMetric.query.count(),
        'lb_visualizations': LeaderboardVisualization.query.count(),
        'metric_results':    MetricResult.query.count(),
        'global_metrics':    GlobalMetric.query.count(),
        'global_vizs':       GlobalVisualization.query.count(),
        'custom_fields':     CustomField.query.count(),
        'tags':              Tag.query.count(),
        'user_colab_gists':  UserColabGist.query.count(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes', action='store_true',
                        help='Actually delete. Without this flag we just print counts.')
    args = parser.parse_args()

    upload_root = app.config.get('UPLOAD_FOLDER')

    with app.app_context():
        before = _counts()
        print("Before wipe:")
        for k, v in before.items():
            print(f"  {k:18s} {v}")
        if not args.yes:
            print("\nDry-run only. Re-run with --yes to actually delete.")
            return

        # 1. Drop on-disk submission + dataset folders. Doing this BEFORE
        # the DB cleanup so a partial failure leaves orphan rows (which
        # are easier to clean up later) rather than orphan disk content
        # (which silently consumes the volume).
        for sub_dir in ('submissions', 'datasets'):
            full = os.path.join(upload_root, sub_dir) if upload_root else None
            if full and os.path.isdir(full):
                print(f"removing {full}/ ...")
                shutil.rmtree(full)
                os.makedirs(full, exist_ok=True)

        # 2. DB cleanup. SQLAlchemy cascades handle most dependents but
        # delete in FK-safe order to be explicit.
        print("deleting MetricResult ...")
        MetricResult.query.delete()
        print("deleting LeaderboardMetric ...")
        LeaderboardMetric.query.delete()
        print("deleting LeaderboardVisualization ...")
        LeaderboardVisualization.query.delete()
        print("deleting Submission ...")
        Submission.query.delete()
        print("deleting CustomField ...")
        CustomField.query.delete()
        print("deleting Sample ...")
        Sample.query.delete()
        print("deleting UserColabGist ...")
        UserColabGist.query.delete()
        print("deleting Leaderboard ...")
        Leaderboard.query.delete()
        print("deleting Dataset ...")
        Dataset.query.delete()
        print("deleting GlobalMetric ...")
        GlobalMetric.query.delete()
        print("deleting GlobalVisualization ...")
        GlobalVisualization.query.delete()
        # Drop tags that nothing references anymore. Easiest: delete all
        # tags; the importer recreates the ones it needs.
        print("deleting Tag ...")
        Tag.query.delete()
        db.session.commit()

        after = _counts()
        print("\nAfter wipe:")
        for k, v in after.items():
            print(f"  {k:18s} {v}")


if __name__ == '__main__':
    main()
