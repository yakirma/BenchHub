# Self-host migration: Fly → home box

Plan for moving BenchHub off Fly.io onto a home machine
(RTX 5090 / 128 GB DDR5 / 8 TB / Core Ultra 9 285).

The current Fly machine runs Flask + Celery + Redis on a single
shared VM with no GPU — see CLAUDE.md "HF dataset attachment
patterns" for the constraint that informed several stubs
(`FID/IS` over raw pixel features, no torch, 5-min soft timeout
on `populate_lb_samples`, etc.). Home box has ~50× headroom and a
GPU, so a lot of those stubs become removable.

**Hosting choice: direct.** We rejected Cloudflare Tunnel and
Tailscale Funnel because:

1. Both terminate TLS at the provider — Cloudflare/Tailscale can
   read all request bodies, including OAuth flows and private LB
   content.
2. Cloudflare Tunnel free has a 100 MB request-body cap; depth-
   prediction submissions for datasets like NYU-Depth V2 routinely
   exceed that (~654 samples × 640×480 float32 ≈ 100–260 MB).

Direct hosting (DuckDNS + port-forward + Let's Encrypt) avoids
both problems and is free.

## Phase 0 — Decide a few things

| Question | Why it matters |
|---|---|
| **Linux or macOS on the box?** | Linux is the easier path: native CUDA for the 5090, systemd for service supervision, standard nginx/redis packages. macOS has no CUDA — you'd lose the InceptionV3-FID win. Recommended: Ubuntu 24.04 LTS or Pop!_OS. |
| **Pick a hostname.** | Free options: `<name>.duckdns.org`, `<name>.no-ip.com`, `<name>.dynv6.net`. Or buy a real domain (~$10/yr) and point an A record at the DuckDNS-updated value. The dynamic-IP path works either way — the static-IP question is irrelevant. |
| **Run Fly + home in parallel for a week, or hard cutover?** | Parallel-run recommended: snapshot data over, smoke-test the home box on a `staging.` hostname, then swap DNS + OAuth callbacks. |

### Check first: am I behind CGNAT?

Some residential ISPs put customers behind Carrier-Grade NAT and
no port-forward is possible. Quickest test:

1. From your home network, visit `whatismyip.com` — note the IP.
2. Log into your router admin and look at its WAN IP.
3. If they match → not behind CGNAT, you're good.
4. If they don't match (router shows a `10.x` / `100.64.x` / private
   range) → you're behind CGNAT; call your ISP and ask for a public
   IP (often free or a small one-time fee), or fall back to one of
   the tunnel options at the bottom of this doc.

## Phase 1 — Stand up the home box

1. **OS + drivers**: install NVIDIA driver (570+ branch for Blackwell/5090), CUDA 12.6 toolkit, cuDNN.
2. **System packages**: `redis-server`, `python3.12`, `ffmpeg`, `libsndfile1` (soundfile dep), `git`, `nginx`, `certbot`, `python3-certbot-nginx`.
3. **Repo**: `git clone` into `/srv/benchhub`, `python -m venv .venv`, `pip install -r requirements.txt`.
4. **Add torch with CUDA**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` — unlocks real InceptionV3-FID/IS later.
5. **Data dir**: `mkdir -p ~/.dtofbenchmarking/{uploads,bench_cache}` (or set `BENCHHUB_DATA_DIR`).

## Phase 2 — Copy data off Fly

```bash
# DB + uploads + bench_cache (the live volume)
fly ssh console -C "tar czf /tmp/bh.tgz -C /root .dtofbenchmarking"
fly ssh sftp get /tmp/bh.tgz ./bh.tgz
tar xzf bh.tgz -C ~/.dtofbenchmarking --strip-components=1
```

SQLite WAL files (`-wal`, `-shm`) come along automatically with the tarball.

## Phase 3 — Process supervision

Three systemd units under `/etc/systemd/system/`:

- `benchhub-redis.service` — or just enable the distro's `redis-server`.
- `benchhub-celery.service`
  ```
  ExecStart=/srv/benchhub/.venv/bin/celery -A app.celery worker --loglevel=info --concurrency=4
  ```
- `benchhub-web.service`
  ```
  ExecStart=/srv/benchhub/.venv/bin/gunicorn -w 4 -b 127.0.0.1:6060 app:app
  ```

(Gunicorn instead of `python app.py` for production.)

## Phase 4 — Public hostname + TLS (direct hosting)

### 4a. Dynamic DNS via DuckDNS

1. Sign in at `duckdns.org` with GitHub/Google → claim a subdomain,
   note your token.
2. On the home box, install a tiny cron:
   ```bash
   # /etc/cron.d/duckdns
   */5 * * * * root curl -fsSL "https://www.duckdns.org/update?domains=<sub>&token=<TOKEN>&ip=" > /dev/null
   ```
   The empty `ip=` tells DuckDNS to detect your current public IP
   from the request — it updates the A record every 5 minutes.

If you bought a real domain instead, point its DNS to DuckDNS as a
CNAME — DuckDNS keeps the IP fresh; your domain follows.

### 4b. Router port-forward

Forward TCP **80** and **443** from the router WAN side to the home
box LAN IP (e.g. `192.168.1.10`). DHCP-reservation that IP in the
router so it doesn't churn. Most routers call this "Port Forwarding"
or "Virtual Server".

### 4c. nginx + Let's Encrypt

```nginx
# /etc/nginx/sites-available/benchhub
server {
    listen 80;
    server_name <sub>.duckdns.org;
    # Phase 4d's certbot rewrites this block to add TLS automatically.
    location / {
        proxy_pass http://127.0.0.1:6060;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Submissions can be hundreds of MB (depth prediction etc.) —
        # no body-size cap.
        client_max_body_size 0;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/benchhub /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

### 4d. TLS

```bash
certbot --nginx -d <sub>.duckdns.org
# Cert auto-renews via the certbot systemd timer that ships with the package.
```

Result: `https://<sub>.duckdns.org` → nginx (port 443) → gunicorn (port 6060). End-to-end encrypted from browser to your home box. No third party reads any traffic.

## Phase 5 — Secrets + OAuth

1. Copy Fly secrets to a `.env` file on the box (loaded by your systemd unit's `EnvironmentFile=`):
   ```bash
   fly secrets list   # names only — values aren't readable, re-set from your records
   ```
   Don't forget: `GITHUB_CLIENT_ID/SECRET`, `GOOGLE_CLIENT_ID/SECRET`, `HF_TOKEN` if any, GitHub gist token if the Colab path is enabled.
2. **Rotate `secret_key`** — currently hardcoded to `'supersecretkey'` (per CLAUDE.md). New box is a good time to switch to `os.environ['SECRET_KEY']` and set a fresh random.
3. **Update OAuth redirect URIs**:
   - GitHub OAuth App → callback `https://<sub>.duckdns.org/oauth/callback/github`
   - Google Cloud Console → OAuth client → authorized redirect URI `https://<sub>.duckdns.org/oauth/callback/google`

## Phase 6 — Cutover

1. Bring up home box on a `staging.<sub>.duckdns.org` hostname (claim a separate DuckDNS subdomain for staging).
2. Smoke test: login, upload a submission (esp. a large depth-prediction ZIP to verify the `client_max_body_size 0` works), run a metric, check `/leaderboards`.
3. Flip the production hostname to point at the home tunnel.
4. `fly scale count 0` — keep the app around for a week before `fly apps destroy` in case you need to roll back.

## Phase 7 — Reap the GPU win

Once stable:

- Rewrite the FID/IS metrics on `GlobalMetric.python_code` to use real InceptionV3 features (the `.fix_fid_metrics.py` workaround can be deleted).
- Bump `_default_hf_cap` test/eval to "no cap" comfortably — 8 TB makes lossless GT viable (option (d) from the sample-cap design discussion).
- Lift the `populate_lb_samples` 5-min soft timeout if you want to bulk-import big PWC LBs in one go.

## Fallback paths (only if direct hosting is blocked)

If you're behind CGNAT and the ISP won't give you a public IP, the
tunnel options come back into play — with their trade-offs:

| Option | Setup | Privacy | Caps |
|---|---|---|---|
| **Cloudflare Tunnel** (free) | Cloudflare account + cloudflared daemon | TLS terminated at Cloudflare (they see all bodies) | 100 MB request body — breaks depth-prediction uploads |
| **Cloudflare Tunnel Pro** ($25/mo) | Same | Same | 500 MB request body |
| **Tailscale Funnel** | Tailscale account + funnel command | TLS terminated at Tailscale DERP | Bandwidth quotas on free tier |
| **Self-hosted WireGuard + $5 VPS rendezvous** | VPS + WireGuard config | Full (VPS only sees ciphertext) | None |

The CGNAT workaround that preserves direct-hosting privacy is the
WireGuard-via-VPS option: a small VPS holds a public IP, your home
box dials out to it over WireGuard, the VPS reverse-proxies HTTPS
to your home box through the VPN. Cheap ($3–5/mo) and the VPS can't
decrypt the TLS traffic.

## Open questions before kicking off

- **Linux vs macOS** (Phase 0 row 1) → determines whether CUDA is on the table.
- **DuckDNS sub vs your own domain** (Phase 0 row 2) → cosmetic, both work the same.
- **CGNAT test passed?** (Phase 0 box) → if not, drop down to a fallback path.

Once those are decided, the systemd units and nginx config can be filled in concretely.
