---
name: sandbox-typed-state
description: "Metric/visualization sandbox is LIVE in prod (BENCHHUB_SANDBOX_METRICS=1) — all user metric + viz code runs only in the hardened docker container. Records how it works + how it was verified. Read before touching metric_engine sandbox code, runner/, the harness JSON contract, or the flag."
metadata:
  type: project
  originSessionId: 5c7cb6c2-c6d2-4a9f-9ba4-8a84e62b77b0
---

Pre-launch security: user metric/viz code is `exec()`'d untrusted Python →
RCE unless sandboxed. The docker sandbox (`runner/`, `--network=none
--read-only --memory=512m --cpus=1`) existed but **only handled primitive
metrics** — typed args (`bh.Depth` etc.) aren't JSON-serialisable and
`benchhub` wasn't in the image, so flipping `BENCHHUB_SANDBOX_METRICS=1`
would have broken the 5 reference metrics. Visualizations had no sandbox
path at all (3 in-process `exec()` sites in app.py: ~4450/4512/4627).

**Docker** is installed on the box (v29.x). The shell/service user `ymatri`
is in the `docker` group, but a pre-existing shell session won't have it
active — run docker via `sg docker -c "..."`, or rely on a service restart
(systemd resolves supplementary groups at start, so benchhub-celery picks
up docker access on restart). Image `benchhub-runner:latest` is built.

**DONE (commit dbaf25c, steps 1-3, flag still OFF → no prod change):**
- Image vendors the `benchhub` package (so `import benchhub as bh` works in
  the container). Build context moved to the REPO ROOT:
  `docker build -f runner/Dockerfile -t benchhub-runner .` + a repo-root
  `.dockerignore` whitelisting only `benchhub/`+`runner/`. (client.py is
  stdlib-only at import time, so no extra image deps.)
- `runner/harness.py` rewritten: injects bh/np/Image; decodes typed args
  from the portable form `{"__bh__": kind, "params": {...}, "b64":
  base64(encode())}` via `DTYPES[kind].decode` (walks lists/dicts); adds a
  `kind: "visualization"` job that returns the PIL.Image as base64 PNG
  (`png_b64`).
- `metric_engine.py`: `_jsonify_kwarg` serialises typed `bh.*` args (the fix
  — bh.Depth isn't JSON-serialisable); `_build_job` sets `kind:'metric'` +
  `include_benchhub:True`; new shared `_dispatch` + `evaluate_viz_in_sandbox`
  (returns `(png_bytes, error)`).
- Verified end-to-end through the real container: typed Depth-rmse, primitive
  metric, viz→PNG, network-egress blocked. Integration test
  (`tests/test_sandbox_docker_integration.py`, now builds repo-root context)
  passes 7/7 via `sg docker`; 8 new in-process tests in
  `tests/test_sandbox_typed.py`.

**DONE (steps 4-5, commit 595f0f9 + flag flip — sandbox is LIVE):**
- All 3 app.py viz `exec()` sites (execute_dataset_visualization,
  execute_visualization, generate_and_cache_agg_viz) route through
  `_render_viz_in_sandbox` → `evaluate_viz_in_sandbox` when
  `_sandbox_enabled()`; in-process exec stays as the off-fallback. Harness
  added `_filter_kwargs_for` (drops `<arg>_names` for funcs that don't
  declare it) + injects `plt` for viz jobs.
- `BENCHHUB_SANDBOX_METRICS=1` is set in `~/benchhub/.env`; both services
  restarted (flag confirmed in each process's environ). Image rebuilt from
  `~/benchhub/runner`.
- **Celery + docker:** benchhub-celery runs as User=ymatri with empty
  SupplementaryGroups, but systemd's initgroups picks up the `docker` group
  (gid 126) on restart — verified the restarted worker PID has 126. So the
  worker can `docker run`. (If a future change loses docker access, add a
  systemd drop-in `SupplementaryGroups=docker`.)
- **Verified in prod:** re-scored real submission id=25 (LB 1) through the
  live worker → status 'Processed', metric values identical to before
  ({2:0.967, 3:1.0}). The 5 reference metric kinds + an aggregated
  confusion-matrix viz were also proven through the real container.
- Rollback if ever needed: remove the BENCHHUB_SANDBOX_METRICS line from
  `~/benchhub/.env` + restart → instant return to in-process exec.

**BUILT ON TOP (shipped):** the bh-client **dev kit** and **dynamic dtypes**
both ride on this sandbox — see [devkit-and-dynamic-dtypes](devkit_and_dynamic_dtypes.md).
