# Self-host runbook: the home box

Operational notes for the BenchHub deployment running on the home box.
For the historical *plan* see `docs/SELFHOST_MIGRATION.md`; this file is
what actually got built and how to operate it.

## TL;DR

```bash
# Push code from laptop
git push origin main

# SSH to box
ssh -p 2222 ymatri@runbenchhub.com

# On box: pull + reload
cd ~/benchhub && git pull
sudo systemctl reload benchhub-web        # code-only change (gunicorn HUP)
# OR (env vars / new deps / DB migration / startup-time hooks):
sudo systemctl restart benchhub-web benchhub-celery
```

`reload` is graceful (gunicorn HUP, no dropped requests) and reloads
Python source. It does **not** re-read `.env` — env-var changes need a
full `restart`. Same for adding a column to a model (the `check_and_migrate_db`
call only runs at process boot).

## What lives where

### Host

| Thing | Value |
|---|---|
| Public hostname | `runbenchhub.com` (also `www.runbenchhub.com` → 301) |
| Public IP source | Home WAN, kept fresh in Cloudflare DNS by `ddclient` |
| DNS provider | Cloudflare (DNS-only mode, no proxy) |
| Domain registrar | GoDaddy (nameservers delegated to Cloudflare) |
| SSH | TCP 2222 (router-forwarded) → port 22 on the box |
| Box LAN IP | `192.168.1.214` (DHCP-reserved on the router) |
| Box hostname | `ymatri-System-Product-Name` |
| OS | Ubuntu 24.04 LTS |
| User | `ymatri` |
| GPU | RTX 5090 (Blackwell sm_120) |
| RAM / disk | 128 GB DDR5 / 8 TB |

### On-box paths

| Path | What |
|---|---|
| `~/benchhub/` | Repo clone (origin = `git@github.com:yakirma/BenchHub.git`, branch `main`) |
| `~/benchhub/.venv/` | Python 3.12 venv (`python3.12 -m venv`) |
| `~/benchhub/.env` | Runtime secrets + flags. Loaded via systemd `EnvironmentFile=` — never committed |
| `~/.dtofbenchmarking/database.db` | SQLite DB (WAL mode, 120 s busy timeout) |
| `~/.dtofbenchmarking/uploads/` | Submission + dataset ZIPs and extracted files |
| `~/.dtofbenchmarking/_cache/` | `bench_cache` (rendered GT thumbs, viz cache) |
| `/etc/systemd/system/benchhub-web.service` | gunicorn unit |
| `/etc/systemd/system/benchhub-celery.service` | celery worker unit |
| `/etc/nginx/sites-available/benchhub` | nginx reverse proxy + certbot-managed TLS |
| `/etc/ddclient.conf` | Cloudflare DDNS config |

### Services

| Unit | Notes |
|---|---|
| `redis-server.service` | Distro package. Broker + result backend on `127.0.0.1:6379`. |
| `benchhub-celery.service` | `Requires=redis-server`. `celery -A app.celery worker --concurrency=4`. |
| `benchhub-web.service` | `Requires=benchhub-celery`. `gunicorn -w 4 --threads 2 --timeout 300 -b 127.0.0.1:6060 app:app`. |
| `nginx.service` | 443 → 127.0.0.1:6060. `client_max_body_size 0` (depth-prediction ZIPs can be >100 MB). `proxy_read_timeout 600s`. |
| `ddclient.service` | Pushes current WAN IP to Cloudflare A record for `runbenchhub.com`. |
| `certbot.timer` | Auto-renews Let's Encrypt certs. |

## `.env` keys

These must be present for the app to come up cleanly. Never commit
this file. Reset via `fly secrets list` is no longer possible (Fly app
was destroyed); current values live only on the box.

| Key | What |
|---|---|
| `SECRET_KEY` | Flask session signing key. Rotated off `'supersecretkey'` during migration. |
| `BENCHHUB_DATA_DIR` | `/home/ymatri/.dtofbenchmarking` |
| `BENCHHUB_BASE_URL` | `https://runbenchhub.com` — used by `_personalize_notebook_for_user` to rewrite `benchhub.fly.dev` hard-codes in cached SOTA notebooks. |
| `BENCHHUB_ADMIN_EMAILS` | Comma-separated admin emails (your email is on this list). |
| `BENCHHUB_AUTO_MIGRATE` | `1` — lets `app.py` run `check_and_migrate_db()` at boot so model-level ALTER blocks apply on `systemctl restart`. |
| `GITHUB_CLIENT_ID` / `_SECRET` | GitHub OAuth app. Callback URL `https://runbenchhub.com/oauth/callback/github`. |
| `GOOGLE_CLIENT_ID` / `_SECRET` | Google OAuth client. Redirect URI `https://runbenchhub.com/oauth/callback/google`. |
| `BENCHHUB_GITHUB_GIST_TOKEN` | Used to push SOTA notebooks as gists. Scope: `gist`. |
| `ANTHROPIC_API_KEY` | LLM-authored metric + SOTA-notebook generation (`_llm_generate_metric_code`, `admin_lb_sota_notebook`). |
| `HF_TOKEN` | Optional; used by `datasets.load_dataset` for gated repos. |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` / `MAIL_FROM` | SMTP relay for passwordless email sign-in (the 6-digit verification code). **Required for email login to actually deliver** — without `SMTP_HOST` the code is only logged (and shown on-page in dev/test); in prod the user gets a "couldn't send" error and must fall back to GitHub/Google. Port 465 → implicit SSL; anything else → STARTTLS. `MAIL_FROM` defaults to `no-reply@runbenchhub.com`. |

## Deploy procedure

### Code-only change (most common)

```bash
# laptop
git add -p && git commit -m "..."
git push origin main

# box
ssh -p 2222 ymatri@runbenchhub.com
cd ~/benchhub
git pull
sudo systemctl reload benchhub-web        # HUP gunicorn → graceful
sudo systemctl restart benchhub-celery    # celery has no SIGHUP equivalent for code reload
```

`reload` (HUP) is preferred for the web tier — it drops zero requests
and gunicorn re-execs workers with the new code. Celery doesn't pick
up code changes on HUP, so it gets a full restart.

### `.env` / env-var change

`EnvironmentFile=` is read **only** at process start. HUP does nothing
for env vars.

```bash
# edit ~/benchhub/.env on the box
sudo systemctl restart benchhub-web benchhub-celery
```

### Adding a model column or other startup-time migration

`check_and_migrate_db()` runs in `if __name__ == '__main__'` and at
gunicorn boot when `BENCHHUB_AUTO_MIGRATE=1`. A `reload` (HUP) does
not re-run it because worker forks already inherit the migrated DB
handle of the master process — but to be safe on new schema, do a full
restart:

```bash
sudo systemctl restart benchhub-web
```

### New Python dep in `requirements.txt`

```bash
ssh -p 2222 ymatri@runbenchhub.com
cd ~/benchhub
source .venv/bin/activate
pip install -r requirements.txt
deactivate
sudo systemctl restart benchhub-web benchhub-celery
```

## Health checks + log tailing

```bash
# Are the services up?
systemctl is-active benchhub-web benchhub-celery redis-server nginx ddclient

# Recent logs
journalctl -u benchhub-web -n 100 --no-pager
journalctl -u benchhub-celery -n 100 --no-pager
journalctl -u benchhub-web -f                 # tail
journalctl -u benchhub-celery -f -p err       # only errors

# Nginx access / error
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Is the public DNS pointing where we expect?
dig +short runbenchhub.com                    # should match `curl -s https://api.ipify.org` from the box
```

## DNS / DDNS

Cloudflare is the authoritative DNS for `runbenchhub.com` (GoDaddy
nameservers delegated). `ddclient` runs on the box and PATCHes the A
record via Cloudflare API whenever the home WAN IP changes.

```bash
# Force a ddclient run + print outcome
sudo ddclient -daemon=0 -debug -verbose -noquiet
# Quick sanity:
dig +short runbenchhub.com
curl -s https://api.ipify.org && echo
```

If those two diverge, check `sudo systemctl status ddclient` and
`sudo journalctl -u ddclient -n 50 --no-pager`. The Cloudflare API
token lives in `/etc/ddclient.conf` (root-only).

**No `whatismyip.com` lookups in Cloudflare's dashboard** — DNS is in
"DNS only" mode (grey cloud), not proxied. That's deliberate: proxied
mode would put Cloudflare in the middle of TLS and impose a body-size
cap. We chose direct-hosting.

## TLS

Certbot + nginx plugin. Cert lives in
`/etc/letsencrypt/live/runbenchhub.com/{fullchain,privkey}.pem`. The
`certbot.timer` systemd timer renews automatically; nginx is
reloaded via the renewal hook.

```bash
sudo certbot certificates                     # list + show expiry
sudo certbot renew --dry-run                  # rehearse renewal
```

If you change the public hostname (or add a subdomain), re-issue with:

```bash
sudo certbot --nginx -d runbenchhub.com -d www.runbenchhub.com -d <new>.runbenchhub.com
```

## Rollback

```bash
# laptop
git log --oneline -10
git revert <bad-sha>                         # creates a revert commit
git push origin main

# box
ssh -p 2222 ymatri@runbenchhub.com
cd ~/benchhub && git pull && sudo systemctl reload benchhub-web
```

For a faster local rollback without going through GitHub:

```bash
cd ~/benchhub
git log --oneline -10
git checkout <good-sha> -- <file>            # surgically revert one file
sudo systemctl reload benchhub-web
# follow up with a real revert commit from the laptop afterward
```

DB rollback is **not** automated. SQLite snapshots:

```bash
cp ~/.dtofbenchmarking/database.db ~/db-backup-$(date +%F).db
```

Take one before any migration you're nervous about.

## When the box itself reboots

systemd brings everything back in order:

`redis-server` → `benchhub-celery` → `benchhub-web` → `nginx`.

`ddclient` runs as a daemon and pushes the (possibly new) WAN IP within
a few minutes. If the home network came up with a fresh IP and the box
beat DDNS to start, the site will be unreachable for ~5 min — that's
expected.

If the box's LAN IP shifts (DHCP), the router's port-forward (TCP 2222
→ port 22) needs updating; reserve the IP in the router admin to
prevent recurrence.

## DB backups

A user-level systemd timer takes a daily SQLite snapshot of
`~/.dtofbenchmarking/database.db` and keeps the last 14. The
canonical copies live in `ops/` in the repo; the deployed copies
sit on the box at the paths in the table.

| Repo source (canonical) | Deployed location | Role |
|---|---|---|
| `ops/benchhub_db_backup.py` | `~/bin/benchhub_db_backup.py` | The snapshot script — uses sqlite3's online-backup API so gunicorn / celery don't need to stop; gzips the result. |
| `ops/benchhub-db-backup.service` | `~/.config/systemd/user/benchhub-db-backup.service` | One-shot unit that runs the script. |
| `ops/benchhub-db-backup.timer` | `~/.config/systemd/user/benchhub-db-backup.timer` | Daily at 03:00 local (with up to 10 min random delay); `Persistent=true` so a missed run catches up after a reboot. |
| (output only) | `~/.dtofbenchmarking/db_backups/` | Snapshot output dir. `database-YYYYMMDD-HHMMSS.db.gz`. |

First-time install on a fresh box:

```bash
mkdir -p ~/bin ~/.config/systemd/user ~/.dtofbenchmarking/db_backups
cp ops/benchhub_db_backup.py ~/bin/
chmod +x ~/bin/benchhub_db_backup.py
cp ops/benchhub-db-backup.service ops/benchhub-db-backup.timer \
    ~/.config/systemd/user/
loginctl enable-linger "$USER"   # so user timer survives logout
systemctl --user daemon-reload
systemctl --user enable --now benchhub-db-backup.timer
```

`loginctl enable-linger ymatri` is already set, so the user
systemd instance — and therefore the timer — survives logout.

Useful commands:

```bash
# Take a snapshot right now.
systemctl --user start benchhub-db-backup.service

# When did it last run? When's the next?
systemctl --user list-timers benchhub-db-backup.timer

# Tail snapshot history.
journalctl --user -u benchhub-db-backup -n 30

# Restore from a snapshot (stop the services first or the WAL writer
# will fight you):
sudo systemctl stop benchhub-web
sudo systemctl stop benchhub-celery
zcat ~/.dtofbenchmarking/db_backups/database-YYYYMMDD-HHMMSS.db.gz \
  > ~/.dtofbenchmarking/database.db
# Wipe any stale WAL/-shm — the snapshot is a self-contained DB.
rm -f ~/.dtofbenchmarking/database.db-wal ~/.dtofbenchmarking/database.db-shm
sudo systemctl start benchhub-celery
sudo systemctl start benchhub-web
```

## Common breakages we've already hit

- **`sudo systemctl restart` doesn't trigger code reload for a hot
  patch you applied straight on the box** — that's by design (we ship
  via `git pull`). If you edited `app.py` directly on the box and now
  can't reproduce it locally, copy it back: `scp -P 2222
  ymatri@runbenchhub.com:~/benchhub/app.py /tmp/app.py.box` and diff
  against your laptop tree.
- **`reload` after editing `.env`** — does nothing. Use `restart`.
- **`SIGTERM` to gunicorn while `Restart=on-failure`** leaves the unit
  in `inactive (dead)` because clean exit isn't a failure. The unit
  files use `Restart=on-failure`; switch to `Restart=always` if you
  prefer auto-resurrection from any exit.
- **Big bulk HF imports** can pin both Flask and Celery (single box,
  shared resources). Rate-limit any `populate_lb_samples` bulk job;
  see CLAUDE.md "HF dataset attachment patterns" for the 5-min soft
  timeout that exists to protect this exact box.
- **Cached SOTA notebooks pre-dating `BENCHHUB_BASE_URL`** still say
  `benchhub.fly.dev` until the user clicks "Generate notebook" again.
  `_personalize_notebook_for_user` rewrites the URL at serve time, but
  the cache is keyed on `(lb_id, hf_repo, model_id)` and never
  invalidates on its own.

## Things explicitly NOT used

- DuckDNS / no-ip / dynv6 (planned in the migration doc; replaced by
  Cloudflare + ddclient because we bought a real domain).
- Cloudflare Tunnel / Tailscale Funnel (rejected — TLS terminates at
  the provider, plus 100 MB body cap breaks depth submissions).
- Fly.io (app destroyed post-cutover).
- Docker / containers on the box (gunicorn + celery + redis run
  directly under systemd).
- Alembic (DB migrations are raw `ALTER TABLE` blocks in
  `check_and_migrate_db()`).
