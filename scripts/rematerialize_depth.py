#!/usr/bin/env python
"""Re-run depth LB materialisations SYNCHRONOUSLY, one at a time.

The Celery-queued materialisations for these LBs were lost (worker child
crashed mid-run, row stuck 'running'). Running them in-process via
.apply() can't be lost and surfaces any error. Sequential to avoid the
concurrent-load issue that lost them the first time.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/rematerialize_depth.py <lb_id> [...]
"""
import os
import sys

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
# Don't run check_and_migrate_db on import (we only need the app context).
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'


def main():
    ids = [int(x) for x in sys.argv[1:] if x.strip()]
    if not ids:
        print('usage: rematerialize_depth.py <lb_id> [...]'); return 2
    import app as A
    from app import db, LeaderboardMaterialization
    import tasks as _tasks
    for lb_id in ids:
        with A.app.app_context():
            mat = LeaderboardMaterialization.query.filter_by(leaderboard_id=lb_id).first()
            if mat is None:
                print(f'REMAT_SKIP {lb_id} no materialization row'); continue
            mat.status = 'pending'
            mat.error_message = None
            db.session.commit()
            _cap, _samp = mat.sample_cap, mat.sampling
        print(f'REMAT_START {lb_id} (cap={_cap}, sampling={_samp})', flush=True)
        try:
            _tasks.materialize_leaderboard.apply(args=[lb_id])
        except Exception as e:
            print(f'REMAT_EXC {lb_id} {type(e).__name__}: {e}', flush=True)
        with A.app.app_context():
            mat = LeaderboardMaterialization.query.filter_by(leaderboard_id=lb_id).first()
            n = db.session.execute(db.text(
                "SELECT COUNT(DISTINCT sample_name) FROM custom_field "
                "WHERE leaderboard_id=:l AND sample_id IS NULL AND submission_id IS NULL"),
                {'l': lb_id}).scalar()
            print(f'REMAT_DONE {lb_id} status={mat.status} gt_samples={n}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
