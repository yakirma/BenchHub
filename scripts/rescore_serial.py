#!/usr/bin/env python
"""Re-score a leaderboard's non-Processed submissions ONE AT A TIME.

Mass-triggering re-scores overwhelms the SQLite write lock (the sandboxed
depth scoring holds it long enough that concurrent status UPDATEs 500). This
triggers each submission, waits for it to reach 'Processed' (re-triggering if
a worker drops the task and it goes stale), then moves on — so only one
scoring task runs at a time and there is no contention.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/rescore_serial.py <lb_id> [<lb_id> ...]
"""
import os
import sys
import time
import sqlite3

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'
DB = os.path.expanduser('~/.dtofbenchmarking/database.db')


def status(sub_id):
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    try:
        r = con.execute("SELECT processing_status FROM submission WHERE id=?", (sub_id,)).fetchone()
        return r[0] if r else None
    finally:
        con.close()


def main():
    lbs = [int(x) for x in sys.argv[1:] if x.strip()]
    if not lbs:
        print('usage: rescore_serial.py <lb_id> [...]'); return 2
    import app as A
    import tasks
    from app import Submission
    with A.app.app_context():
        ids = [s.id for lb in lbs
               for s in Submission.query.filter_by(leaderboard_id=lb).all()
               if s.processing_status != 'Processed']
    print(f'serial re-score of {len(ids)} submissions: {ids}', flush=True)
    for sub_id in ids:
        ok = False
        for attempt in range(4):
            tasks.process_submission.delay(sub_id)
            t0 = time.time()
            last = None
            while time.time() - t0 < 600:
                st = status(sub_id)
                if st == 'Processed':
                    ok = True; break
                # If it went Error (e.g. a lock blip), break to re-trigger.
                if st and st.startswith('Error') and time.time() - t0 > 30:
                    break
                if st != last:
                    last = st
                time.sleep(6)
            if ok:
                break
        print(f'  sub {sub_id}: {status(sub_id)[:48]!r} (ok={ok})', flush=True)
    print('SERIAL_RESCORE_DONE', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
