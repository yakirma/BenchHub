# Deploying BenchHub to Fly.io

End-to-end runbook for putting BenchHub behind a Cloudflare Access policy on
Fly.io. The repo already contains the `Dockerfile`, `fly.toml`, and
`.dockerignore` — your job is to provision the cloud resources and wire them
together.

> **Security note**: The metric engine `exec()`s user-supplied Python code
> (see `metric_engine.py`). The Cloudflare Access policy in step 6 is what
> keeps anonymous strangers off the public URL. If you ever loosen that
> policy (or skip step 6), you've shipped an RCE-as-a-service. Don't.

## Prerequisites

- A Fly.io account with `flyctl` installed (`brew install flyctl` /
  `curl -L https://fly.io/install.sh | sh`).
- A Cloudflare account with at least the Free Zero Trust plan (the team
  email login is free for up to 50 users).
- A domain you can manage in Cloudflare (or you can put Cloudflare Access
  in front of the bare `*.fly.dev` hostname — see step 6).

## 1. Log in and create the app

```bash
fly auth login
cd /path/to/BenchHub
fly launch --no-deploy
```

When prompted:

- **App name**: pick something stable (you'll use it as the `.fly.dev`
  hostname). Edit `app = "benchhub"` in `fly.toml` to match.
- **Region**: pick whichever is closest. Update `primary_region` in
  `fly.toml` to match.
- **Postgres**: **No** — we use SQLite on a volume.
- **Redis**: **No** — we'll provision Upstash separately in the next step
  (Fly's old "Redis" was Upstash under the hood; we want explicit control
  of the Upstash instance).
- **Deploy now?**: **No** — we still need volume + Redis + secrets first.

`fly launch --no-deploy` may overwrite `fly.toml`. If it does, copy the
`[processes]`, `[mounts]`, `[deploy]`, and `[[services.http_checks]]`
sections back from the version checked into git (they're easy to lose).

## 2. Create the persistent volume

```bash
fly volumes create benchhub_data \
  --region <same region as primary_region> \
  --size 10
```

10 GB is plenty for a starter. SQLite + the uploaded ZIPs all live here.
Bump it later with `fly volumes extend`. Keep the name `benchhub_data` —
that's what `fly.toml`'s `[mounts]` block references.

## 3. Provision Redis (Upstash)

Easiest path is Upstash directly:

1. Sign up at https://upstash.com (free tier: 10k commands/day, 256MB).
2. Create a Redis database in a region near your Fly region.
3. Copy the **connection string** with TLS, looks like:
   `rediss://default:<token>@<host>.upstash.io:6379`

Alternative: `fly redis create` provisions Upstash through the Fly CLI and
prints the URL, if you'd rather keep billing in one place.

## 4. Set Fly secrets

```bash
# 32 random bytes — Flask uses it for session signing.
fly secrets set SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"

# Redis URL from step 3 (note the rediss:// for TLS).
fly secrets set REDIS_URL="rediss://default:<token>@<host>.upstash.io:6379"
```

Optional secrets:

```bash
# If you use the git-author-from-commit feature, point at the local mirror
# inside the container (see local_config.py for the dev equivalent).
# fly secrets set GIT_REPO_PATH=/data/dtof_sim_mirror
```

You don't need to set `BENCHHUB_DATA_DIR`, `PORT`, or
`CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND` — `[env]` in `fly.toml` covers
the first two, and `app.py` falls back from the celery vars to `REDIS_URL`.

## 5. Deploy

```bash
fly deploy
```

What happens:

- Image builds from `Dockerfile`.
- `release_command = "python app.py migrate"` runs once: applies any
  `check_and_migrate_db()` ALTER TABLEs against the SQLite file on the
  mounted volume, then exits.
- Web (`gunicorn`) and worker (`celery`) processes start. Both mount
  `/data` from the same volume.

Watch logs with `fly logs`. Test the bare hostname:

```bash
curl -I https://<your-app>.fly.dev/projects
# 200 OK → app is up. Skip to step 6 to lock it down.
```

> **Don't link a custom domain or share the `.fly.dev` URL until step 6
> is done.** Right now it's a public, unauthenticated, RCE-able endpoint.

## 6. Cloudflare Access in front of it

We'll put Cloudflare's Zero Trust gateway in front of the `.fly.dev`
hostname so every request has to authenticate before it reaches your app.
Two routing options — pick one:

### Option A — Cloudflare Tunnel (preferred; hides Fly entirely)

This is the right answer if you control a Cloudflare-managed domain.

1. **Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create a
   tunnel** (Cloudflared). Name it `benchhub`.
2. Pick **"Public hostname"**:
   - **Subdomain**: `benchhub`
   - **Domain**: pick one of your CF-managed domains
   - **Service**: `https://<your-app>.fly.dev` (HTTPS, with "No TLS Verify"
     OFF — Fly's cert is valid)
3. Don't run `cloudflared` on a host — for this routing pattern Cloudflare
   proxies directly. (If you need `cloudflared`, run it as a third process
   inside the Fly app or on a separate VM.)
4. **Zero Trust → Access → Applications → Add application → Self-hosted**:
   - **Name**: BenchHub
   - **Application domain**: `benchhub.<yourdomain>`
   - **Identity providers**: enable at least one (Google / GitHub / one-time PIN over email all work; OTP-by-email needs zero setup).
   - **Policy**: `Action = Allow`, `Include = Emails ending in @<your domain>` or `Include = Specific emails: you@…, teammate@…`.
5. Hit the new domain → you should be bounced to Cloudflare's login → after
   login, BenchHub appears.

### Option B — Cloudflare Access on the bare `.fly.dev` hostname

Cloudflare Access can also gate hostnames it doesn't own, via the
`cloudflared access` CLI on the client side. **Skip this.** It's clunky for
casual users and there's no real reason not to use a domain you control.

### Belt-and-suspenders: shut off direct Fly access

Once Access is enforcing on the CF hostname, Fly will still happily serve
`https://<your-app>.fly.dev` to anyone. To force traffic through CF only,
add a request handler in front of the WSGI app that rejects requests
without the Cloudflare-issued JWT (`Cf-Access-Jwt-Assertion` header) — or,
simpler, configure the Cloudflare Tunnel from option A so Fly only accepts
inbound from Cloudflare's IP ranges. This is a hardening step; do it once
the basic deploy works.

## 7. Smoke test

```bash
# Browser: visit your CF-Access-protected URL → log in → land on /projects.
# Through CLI:
fly ssh console -C "python -c 'from app import app; print(app.config[\"SQLALCHEMY_DATABASE_URI\"])'"
# Should print sqlite:////data/database.db.

fly ssh console -C "ls /data"
# Should show database.db and uploads/ once you've done anything.
```

Upload a small dataset ZIP through the UI and verify the `Submission` row
ends at status `Processed` (the worker is doing its job).

## 8. Day-2 operations

| Need | Command |
|---|---|
| Redeploy after a code change | `fly deploy` |
| Tail logs (web + worker) | `fly logs` |
| Open a shell on the web VM | `fly ssh console` |
| Open a shell on a worker VM | `fly ssh console --process=worker` |
| Scale workers | `fly scale count worker=2` (sequential batch task means: don't, see commit `8a77b48`) |
| Resize the VM if jobs OOM | `fly scale memory 2048` |
| Grow the volume | `fly volumes extend <vol_id> --size 25` |
| Back up the SQLite DB | `fly ssh console -C "sqlite3 /data/database.db .backup /data/snapshots/$(date +%F).db"` then `fly sftp shell` to pull |
| Rotate `SECRET_KEY` | `fly secrets set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')` (logs everyone out) |

## What this deployment **doesn't** do

- **No Postgres.** SQLite on a volume is fine for one web instance; if you
  ever need horizontal scaling, switch to Fly Postgres (the SQLAlchemy URI
  is the only meaningful change in `app.py`).
- **No object storage.** Uploads live on the Fly volume. If a single
  volume's 50GB cap becomes a problem, plumb `process_dataset_zip` /
  `process_submission_zip` to S3-compatible storage and update the
  `download_*` routes — touched code paths are in `app.py` around the
  `app.config["UPLOAD_FOLDER"]` references.
- **No real metric sandboxing.** Cloudflare Access keeps anonymous users
  out, but anyone with an email matching your Access policy can still
  upload Python that runs server-side. If you ever open the policy beyond
  trusted teammates, do the Docker-per-job sandbox refactor first.
- **No CI deploys.** `fly deploy` is run manually. The `.github/workflows/test.yml` pipeline runs tests on push but doesn't deploy on green. Add a
  `fly-deploy.yml` workflow with `flyctl deploy --remote-only` and a
  `FLY_API_TOKEN` GitHub secret if you want push-to-deploy.
