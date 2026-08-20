#!/usr/bin/env python
"""One-off: email all BenchHub users about the 0.1.11 client fixes.

Loads prod creds from ~/benchhub/.env (the app has no dotenv autoload; systemd
supplies them via EnvironmentFile), then sends via the app's existing Resend/
SMTP helpers. Dry-run by default — pass --send to actually deliver.

    python scripts/announce_release.py           # dry run: list recipients + preview
    python scripts/announce_release.py --send     # actually send
"""
import os
import sys
import time

ENV_PATH = os.path.expanduser('~/benchhub/.env')


def _load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(ENV_PATH)

# The prod .env sets BENCHHUB_AUTO_MIGRATE=1, and app.py runs schema migrations
# at import time when that's set. We only read users + send mail here — never
# migrate — so force it off so importing app can't mutate the live DB from this
# out-of-band process (it runs concurrently with gunicorn/celery).
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402  (env + path must be set first)

SUBJECT = "BenchHub update: submission-script fix + benchhub-client 0.1.11"

BODY_TMPL = """\
{greeting}

A few fixes and improvements just landed on BenchHub:

- Submission scripts fixed. If a downloaded submission script failed with
  "AttributeError: 'Client' object has no attribute 'leaderboard'", it's
  resolved. Update the client and re-run:

      pip install -U benchhub-client

  Scripts you download now pin the client version they need, so this won't
  recur.

- API-token link corrected. Submission scripts now point you to the right
  place for a token: https://runbenchhub.com/settings/api_tokens

- Release notes. See what changed in each client version:
  https://runbenchhub.com/releases

- Also in 0.1.11: reference a board by its <owner>/<slug> handle
  (client.leaderboard("owner/slug")), and raw sample-data downloads now
  require a signed-in session or an API token.

Happy benchmarking,
The BenchHub team
https://runbenchhub.com

You're receiving this because you have a BenchHub account. Prefer not to get
occasional product updates? Just reply and we'll leave you off.
"""


def _valid(email):
    if not email or '@' not in email:
        return False
    domain = email.rsplit('@', 1)[-1]
    return '.' in domain  # drops obvious test addresses like "foo@x"


def body_for(display_name):
    name = (display_name or '').strip()
    greeting = f"Hi {name}," if name else "Hi there,"
    return BODY_TMPL.format(greeting=greeting)


def main():
    send = '--send' in sys.argv
    with A.app.app_context():
        users = A.User.query.order_by(A.User.id).all()
        recipients, skipped = [], []
        for u in users:
            (recipients if _valid(u.email) else skipped).append(
                (u.email, u.display_name))
        print(f"users={len(users)}  valid_recipients={len(recipients)}  "
              f"skipped={len(skipped)}  mode={'SEND' if send else 'DRY-RUN'}")
        if skipped:
            print("  skipped (invalid):", [e for e, _ in skipped])
        print("-" * 60)
        if not send:
            print("SUBJECT:", SUBJECT)
            print()
            print(body_for(recipients[0][1] if recipients else None))
            print("-" * 60)
            print("Recipients:")
            for e, n in recipients:
                print(f"  {e}  ({n or '-'})")
            print("\n(dry run — no mail sent. re-run with --send)")
            return

        sent, failed = 0, []
        for email, name in recipients:
            body = body_for(name)
            ok = False
            try:
                eid = A._resend_send(email, SUBJECT, body)
                ok = eid is not None or A._send_email(email, SUBJECT, body)
            except Exception as exc:  # noqa: BLE001
                print("  EXC", email, exc)
                ok = False
            print(f"  {'OK  ' if ok else 'FAIL'} {email}")
            sent += 1 if ok else 0
            if not ok:
                failed.append(email)
            time.sleep(0.6)  # gentle pacing for the Resend API
        print("-" * 60)
        print(f"done: sent={sent}  failed={len(failed)}  {failed}")


if __name__ == '__main__':
    main()
