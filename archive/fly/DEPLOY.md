# Deploying BenchHub to Fly.io

> **⚠️ DEPRECATED — historical reference only.**
> Production no longer runs on Fly.io. The app was migrated to a
> self-hosted Ubuntu box (`runbenchhub.com`) and the Fly app was
> destroyed post-cutover. The live operational runbook is
> **[`docs/SELFHOST_RUNBOOK.md`](../../docs/SELFHOST_RUNBOOK.md)**.
> The Fly artifacts (`fly.toml`, `Dockerfile`, `.dockerignore`,
> `start.sh`, `entrypoint.sh`, `runner/fly.toml`) live in this directory
> alongside this doc, so a future Fly deploy can be rebooted from here.
> Do not follow the steps below unless you're standing up a fresh Fly
> deploy from scratch.

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

## 9. Sandboxed metric execution (optional, required before opening to the public)

`metric_engine.py` `exec()`s user-supplied Python in-process by default.
That's fine while Cloudflare Access keeps strangers out. If you want to
*remove* the Access gate (open public sign-up), you must run user code in
a hardened container instead. The repo includes that container at
`runner/`, behind an env flag.

### Building the runner image locally

Verify it builds + the integration tests pass on your machine before
worrying about deploy:

```bash
make runner-image          # builds benchhub-runner:local
make test-docker           # runs tests/test_sandbox_docker_integration.py
```

The integration tests exercise the real container: happy path, numpy
injection, `--network=none` blocking outbound traffic, `--read-only`
blocking writes outside `/tmp`. They're slow (~30s first run) and skipped
by `make test`; only `make test-docker` runs them.

### Toggling the sandbox at runtime

```bash
fly secrets set BENCHHUB_SANDBOX_METRICS=1
```

`tasks._eval_metric_batch` reads this on every metric job and routes to
`evaluate_in_sandbox` (one container spawn per metric × submission, with
all sample contexts batched into a single JSON job — see Slice 2 commit).
Unset or any value other than the literal `1` keeps the in-process path.

### The Fly-machines-don't-have-a-Docker-socket problem

Here's the one open design decision before this can ship: Fly machines
don't expose `/var/run/docker.sock`, so `subprocess.run(['docker', 'run', ...])`
inside the BenchHub web/worker container fails. Four ways to solve it:

| Option | Shape | Pros | Cons | Recommendation |
|---|---|---|---|---|
| **A: Runner as a separate Fly app** | Deploy `runner/` as its own Fly app exposing an HTTP endpoint that wraps the harness. The web app POSTs the JSON job over `*.internal` Fly DNS. | Cleanest separation. Runner can run with no volume, no DB access, on a smaller VM. Easy to scale separately. The "container" is one persistent process, not per-job — no startup cost. | Adds an HTTP layer. The runner is a long-running process with all jobs sharing a Python interpreter — you lose the per-job isolation that was the whole point. Can be partially fixed by running the runner with low concurrency + pre-fork worker recycling, but it's a softer boundary than fresh containers. | **Pick this** for the cheap public launch. Tighten later. |
| **B: Fly Machines API per job** | Web app calls `flyctl machines run` (or the HTTP API equivalent) per metric to spin up an ephemeral machine, pipe the job in, read the result, destroy the machine. | Strongest isolation: real fresh VM per metric. Matches the "Docker-per-job" semantics this code was designed for. | Slow (~5-10s per spawn), nontrivial cost, complex orchestration code (auth, polling, error handling). | Defer. Worth it only if option A's shared-process risk turns out to matter. |
| **C: bubblewrap / landlock on the BenchHub VM** | Replace `docker run` with `bwrap` or a `landlock`-restricted `subprocess.Popen` in the same container. Linux-only. | No external service. Per-job isolation. | Different code path than your local Docker tests — you'd lose the integration tests' value. Linux-syscall-level config is fiddly to get right; one wrong flag and you've left a hole. | Skip. Too easy to misconfigure. |
| **D: gVisor or Firecracker** | Run user code under a user-space kernel. | Strongest practical isolation short of separate VMs. | Major engineering. Distribution / packaging is hairy. | Skip unless this is your job. |

**Option A is now the default deploy path.** The runner repo path
includes:

- `runner/server.py` — Flask wrapper: `POST /run` for jobs, `GET /health`
  for Fly. Calls `harness.run_job` in-process; gunicorn `--max-requests=100`
  recycles the worker periodically so leaked state can't accumulate.
- `runner/Dockerfile` — same image as the local CLI tests; default CMD is
  the gunicorn server, but the image still supports `python /app/harness.py`
  for the Docker-subprocess path.
- `runner/fly.toml` — single internal-only TCP service on port 8080. No
  volume, no DB, no public ports. Auto-stops when idle.

### 9.1 Deploy the runner

From the repo root:

```bash
cd runner
fly launch --no-deploy        # name it benchhub-runner; same primary_region as the main app
fly deploy
cd ..
```

Confirm the runner is up via Fly's internal DNS from the main app:

```bash
fly ssh console -a benchhub
# inside:
curl http://benchhub-runner.internal:8080/health
# {"ok": true}
```

### 9.2 Point the main app at the runner

```bash
fly secrets set \
    BENCHHUB_SANDBOX_URL=http://benchhub-runner.internal:8080/run \
  -a benchhub
fly deploy -a benchhub      # picks up the new env var
```

`metric_engine.evaluate_in_sandbox` checks `BENCHHUB_SANDBOX_URL` first; when set, it POSTs the job there and skips the docker-subprocess fallback. `tasks._eval_metric_batch` is gated by the older `BENCHHUB_SANDBOX_METRICS=1` flag — set both, or fold the URL flag's presence into the dispatch (current code requires both, so set them together):

```bash
fly secrets set BENCHHUB_SANDBOX_METRICS=1 -a benchhub
```

### 9.3 Verify with a real submission

Upload any submission with at least one per-sample metric. In the runner's logs (`fly logs -a benchhub-runner`) you should see one POST per metric — *not* one per sample. The main app batches every sample's context into a single job. If you see N requests for an N-sample submission, something's reverted to the per-call path.

### 9.4 Drop Cloudflare Access

Once 9.3 is clean and you've sat with it for a day or two:

1. In Cloudflare → Zero Trust → Access → Applications, **delete** (or set Action: Bypass) the application that fronts BenchHub.
2. Update DNS so the public hostname routes directly to Fly.
3. Hard-refresh the site logged-out — confirm you can reach `/login` without an Access prompt.

The metric engine's `exec()` now runs in the runner VM, not the web/worker VM. The "soft boundary" cost remains: a leaky metric can affect *other metrics' jobs* in the same gunicorn worker until it recycles. If that turns out to matter — i.e. you find a metric that legitimately needs to be isolated from sibling jobs — switch to Option B (machine-per-job via the Fly Machines API).

## What this deployment **doesn't** do

- **No Postgres.** SQLite on a volume is fine for one web instance; if you
  ever need horizontal scaling, switch to Fly Postgres (the SQLAlchemy URI
  is the only meaningful change in `app.py`).
- **No object storage.** Uploads live on the Fly volume. If a single
  volume's 50GB cap becomes a problem, plumb `process_dataset_zip` /
  `process_submission_zip` to S3-compatible storage and update the
  `download_*` routes — touched code paths are in `app.py` around the
  `app.config["UPLOAD_FOLDER"]` references.
- **No real metric sandboxing in production yet.** Cloudflare Access keeps
  anonymous users out, but anyone with an email matching your Access
  policy can still upload Python that runs server-side. The container in
  `runner/` is built and tested; section 9 above is the runbook for
  switching it on.
- **No CI deploys.** `fly deploy` is run manually. The `.github/workflows/test.yml` pipeline runs tests on push but doesn't deploy on green. Add a
  `fly-deploy.yml` workflow with `flyctl deploy --remote-only` and a
  `FLY_API_TOKEN` GitHub secret if you want push-to-deploy.
