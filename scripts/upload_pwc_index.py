"""Upload a locally-built PWC SQLite index to the prod fly volume.

Why this exists: the in-prod build (via Celery + pwc_client._build_index_into)
proved unreliable on a 4GB shared-cpu fly machine — pyarrow's decode of
the nested-struct parquet from pwc-archive/evaluation-tables stuck in
the first batch regardless of the optimization angle (recursion cap,
column projection, batch size, sqlite PRAGMA tuning). Iteration ate up
real time without finishing.

Building the SQLite locally on a developer box where pyarrow has more
RAM and faster CPU completes in ~30-45 minutes and produces a small
(~5-50 MB) artifact that just needs to land at the right path on prod.
This script does that upload via flyctl ssh sftp.

Usage
-----

1. Build the index locally:

    python -m venv .venv && source .venv/bin/activate
    pip install pyarrow huggingface_hub
    BENCHHUB_DATA_DIR=/tmp/pwc_offline python -c "
    import os, pwc_client as pc
    pc._build_index_into(
        pc._index_path(),
        os.path.join(os.environ['BENCHHUB_DATA_DIR'], '_cache', 'pwc_archive', 'snapshot'),
        progress_cb=print,
    )
    "

2. Upload to prod:

    python scripts/upload_pwc_index.py /tmp/pwc_offline/_cache/pwc_archive/index.v1.sqlite

The script flyctl-ssh-sftp's it to /data/_cache/pwc_archive/index.v1.sqlite,
clears any stale build markers, and prints the new file size as confirmation.
"""
import argparse
import os
import subprocess
import sys

PROD_PATH = '/data/_cache/pwc_archive/index.v1.sqlite'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('local_path',
                        help='Path to the locally-built SQLite (e.g. '
                             '/tmp/pwc_offline/_cache/pwc_archive/index.v1.sqlite)')
    parser.add_argument('--app', default='benchhub',
                        help='Fly app name (default: benchhub)')
    args = parser.parse_args()

    if not os.path.isfile(args.local_path):
        print(f"ERROR: {args.local_path} not found", file=sys.stderr)
        return 1
    size = os.path.getsize(args.local_path)
    print(f"Uploading {size / 1024 / 1024:.1f} MB to {args.app}:{PROD_PATH}…")

    # Clear any stale build markers first so the web tier doesn't think a
    # build is in progress when it sees the SQLite show up.
    subprocess.run(
        ['flyctl', 'ssh', 'console', '-a', args.app, '-C',
         'sh -c "mkdir -p /data/_cache/pwc_archive && '
         'rm -f /data/_cache/pwc_archive/index.building.tmp '
         '/data/_cache/pwc_archive/index.progress.txt '
         '/data/_cache/pwc_archive/index.error.txt '
         '/data/_cache/pwc_archive/index.v1.sqlite.tmp"'],
        check=True,
    )

    # Push the file. Use sftp put with a -- separator so flyctl doesn't
    # interpret the destination path as a flag.
    subprocess.run(
        ['flyctl', 'ssh', 'sftp', 'put', '-a', args.app,
         args.local_path, PROD_PATH],
        check=True,
    )

    # Verify size on the remote side.
    subprocess.run(
        ['flyctl', 'ssh', 'console', '-a', args.app, '-C',
         f'sh -c "ls -la {PROD_PATH}"'],
        check=True,
    )
    print("Done. /admin/pwc/import should now show 'ready' and search lights up.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
