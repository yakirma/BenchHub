# runner/ — sandboxed metric/viz execution

The hardened Docker sandbox that runs ALL user-supplied metric + visualization code. **Live in prod** (`BENCHHUB_SANDBOX_METRICS=1`).

- **Execution model**: user metric/viz code runs inside the container, never in the web/worker process. Typed args and raw bytes cross the boundary as JSON. The sandbox image vendors `benchhub` so `bh.<Kind>` is importable inside.
- `harness.py` — entrypoint inside the container. `_decode_arg` runs a registered kind's `decode(blob, params)` **inside the metric's own container** (no extra spawn) or returns raw bytes when there's no decode hook. Kwargs arrive jsonified as `{"__dtype__","decode","params","b64"}` (built server-side by `metric_engine._jsonify_kwarg`). See `benchhub/CLAUDE.md` for the full `DataTypeDef` decode-hook contract.
- `server.py` — the in-container server loop.
- `Dockerfile` — the sandbox image. NOTE: this is `runner/Dockerfile`, distinct from the **dead** Fly artifacts in `archive/fly/`. `runner/{Dockerfile,harness.py,server.py}` stay in place even though Fly is dead — local sandbox tests `tests/test_sandbox_*` reference them.
- **Both paths must keep working**: when the flag is off (and in some tests) `metric_engine.evaluate_dynamic_metric` exec's user code in-process instead of in the container. Don't break the in-process fallback when changing the sandbox.
