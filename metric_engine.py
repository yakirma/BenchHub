import base64
import json
import os
import shlex
import subprocess
import traceback

import numpy as np

# `requests` is in requirements.txt for the OAuth flow; reuse it here
# rather than pulling stdlib urllib for the HTTP sandbox path.
try:
    import requests as _requests
except ImportError:  # pragma: no cover — dependency-pin guarantees it's there
    _requests = None

# Phase A typed contract: wrap each CustomField value in its `DataType`
# subclass and stash under a `__typed__<key>` parallel key. Metrics that
# have declared `GlobalMetric.input_kinds` receive these typed instances;
# legacy metrics (input_kinds NULL/empty) keep getting the raw primitive
# at the bare key, so nothing pre-Phase-A breaks.
from benchhub.types import DTYPES, DataType


class RegisteredBlob:
    """Carrier for a user-registered kind's stored bytes inside a metric
    context. Built-in kinds wrap as `bh.*` instances (see `_typed_for_cf`);
    a registered kind has no DataType class in this package, so we carry its
    raw bytes + the kind's optional sandboxed `decode(blob, params)` source.

    The two eval backends resolve it differently:
      * Sandbox  — `_jsonify_kwarg` emits `{"__dtype__", "decode", "params",
        "b64"}`; the harness runs `decode` *inside the metric's own
        container* (no extra spawn) or hands over raw bytes when there's no
        decode hook.
      * In-process — `_resolve_registered_blob` runs `decode` locally (the
        non-sandbox path already exec()s untrusted metric code in-process).
    """
    __slots__ = ('kind', 'blob', 'params', 'decode_code')

    def __init__(self, kind, blob, params=None, decode_code=None):
        self.kind = kind
        self.blob = bytes(blob or b'')
        self.params = params or {}
        self.decode_code = decode_code or None


def _registered_blob_for_cf(cf, upload_folder):
    """Build a `RegisteredBlob` for a CustomField whose `data_type` is a
    server-registered kind (not in the built-in DTYPES). Reads the stored
    bytes (file-backed → from disk; inline → value_text) and attaches the
    kind's `decode_code`. Returns None when the kind isn't registered or the
    bytes can't be read — caller then just skips the field (same as today)."""
    try:
        from app import DataTypeDef  # lazy: keeps this module DB-decoupled
    except Exception:
        return None
    try:
        dt = DataTypeDef.query.filter_by(name=cf.data_type).first()
    except Exception:
        return None
    if dt is None:
        return None
    params = cf.get_params() if hasattr(cf, 'get_params') else {}
    if dt.file_ext:
        rel = cf.value_text or ''
        if not rel:
            return None
        path = os.path.join(upload_folder, rel)
        try:
            with open(path, 'rb') as fh:
                blob = fh.read()
        except OSError:
            return None
    else:
        blob = (cf.value_text or '').encode('utf-8')
    return RegisteredBlob(cf.data_type, blob, params, dt.decode_code)


def _resolve_registered_blob(rb):
    """In-process decode for a `RegisteredBlob`. Runs the kind's
    `decode(blob, params)` locally and returns its result; falls back to the
    raw bytes when there's no decode hook or it errors. Only used on the
    non-sandbox path (which already exec()s metric code in-process)."""
    if not isinstance(rb, RegisteredBlob):
        return rb
    if not rb.decode_code:
        return rb.blob
    ns = {'np': np, 'numpy': np}
    try:
        from PIL import Image as _Image
        ns['Image'] = _Image
    except ImportError:
        pass
    try:
        exec(rb.decode_code, ns)
        fn = ns.get('decode')
        if callable(fn):
            return fn(rb.blob, rb.params)
    except Exception as e:
        print(f"DEBUG: in-process decode for {rb.kind!r} failed: {e}")
    return rb.blob


def _typed_for_cf(cf, value):
    """Wrap a CustomField's primitive value in its DataType class.

    `value` is whatever the existing context-builder already loaded —
    a numpy array (image/depth/mask/audio), a float (scalar), a str
    (text), a dict/list (json). Returns the typed instance, or None
    when the kind isn't in the registry or construction fails."""
    if value is None:
        return None
    kind = cf.data_type
    if kind not in DTYPES:
        return None
    cls = DTYPES[kind]
    params = cf.get_params() if hasattr(cf, 'get_params') else {}
    try:
        if kind == 'audio':
            # Audio takes sample_rate positionally; rest of the kinds
            # accept their per-instance params as keyword arguments.
            return cls(value, params.get('sample_rate', 16000))
        # Pass ONLY the params this constructor actually accepts. A field
        # can carry params a given kind doesn't take (e.g. a `label` field
        # with a `names` vocab predating Label accepting `names`); blindly
        # splatting **params used to raise TypeError here, get swallowed,
        # and silently drop the typed instance — which made typed metrics
        # fall back to the raw primitive and misbehave (e.g. accuracy
        # asserting isinstance(gt, bh.Label) on a bare int → 0.0).
        accepted = _accepted_kwargs(cls)
        kwargs = ({k: v for k, v in params.items() if k in accepted}
                  if accepted is not None else params)
        return cls(value, **kwargs)
    except Exception as e:
        print(f"DEBUG: _typed_for_cf({cf.name}, kind={kind}) failed: {e}")
        return None


def _accepted_kwargs(cls):
    """Set of keyword-arg names `cls.__init__` accepts (excluding self/
    value). Returns None when the signature takes **kwargs (accept all).
    Cached per class."""
    cache = _accepted_kwargs._cache
    if cls in cache:
        return cache[cls]
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        cache[cls] = None
        return None
    names = set()
    for p in sig.parameters.values():
        if p.name == 'self':
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            cache[cls] = None  # **kwargs → accept anything
            return None
        if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            names.add(p.name)
    cache[cls] = names
    return names


_accepted_kwargs._cache = {}


def _stash_typed(context, key, cf, value):
    """Mirror `context[key] = value` with `context['__typed__'+key]`
    holding the DataType instance. Call this everywhere we set a
    primitive on the context from a CustomField row."""
    context[key] = value
    inst = _typed_for_cf(cf, value)
    if inst is not None:
        context[f'__typed__{key}'] = inst

def _metric_wants_typed(global_metric) -> bool:
    """A metric opts into typed-instance kwargs by declaring
    `GlobalMetric.input_kinds` as a non-empty JSON array. NULL / empty
    list keeps the legacy primitive shape — that's the back-compat
    pin for every metric written before Phase A."""
    raw = getattr(global_metric, 'input_kinds', None)
    if not raw:
        return False
    try:
        kinds = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return False
    return bool(kinds)


def _resolve_context_value(context, mapping_key, typed):
    """Look up a mapping key in the per-sample context, optionally
    swapping in the `__typed__<key>` parallel entry built by
    `_stash_typed`. Falls back to the primitive when no typed instance
    was stashed (e.g. an opt-in metric reading a legacy CustomField
    row that hasn't been re-typed)."""
    if typed:
        typed_val = context.get(f'__typed__{mapping_key}')
        if typed_val is not None:
            return typed_val
    return context.get(mapping_key)


def evaluate_dynamic_metric(global_metric, context, arg_mappings_json):
    """
    Evaluates a GlobalMetric's python_code against the provided context.
    """
    try:
        # Parse mappings
        try:
            arg_mappings = json.loads(arg_mappings_json)
        except:
            arg_mappings = {}

        wants_typed = _metric_wants_typed(global_metric)

        # Prepare arguments
        call_kwargs = {}
        for arg, mapping_key in arg_mappings.items():
            if mapping_key.startswith('SCALAR:'):
                val_str = mapping_key[7:] # len('SCALAR:') == 7
                try:
                    # Try float first
                    val = float(val_str)
                    # Optional: if int, convert to int?
                    if val.is_integer():
                        val = int(val)
                    call_kwargs[arg] = val
                except ValueError:
                    # Fallback to string if not numeric
                    call_kwargs[arg] = val_str
            elif mapping_key not in context and f'__typed__{mapping_key}' not in context:
                 # Not present in either shape — pass None so the metric
                 # body can detect "missing GT/pred" via context.get().
                 call_kwargs[arg] = None
            else:
                 call_kwargs[arg] = _resolve_context_value(context, mapping_key, wants_typed)
        
        # Execute code. Inject `bh` / `benchhub` aliases so user
        # metrics can use `bh.Label` typed annotations + isinstance
        # asserts without an explicit `import benchhub as bh` at the
        # top of every metric file (still works if they include it).
        import benchhub as _bh
        local_scope = {'np': np, 'bh': _bh, 'benchhub': _bh}
        exec(global_metric.python_code, local_scope)
        
        # Find the function (assuming name matches or it's the only one?)
        # Convention: The code defines a function. We can find it by name if we knew it,
        # or just take the last defined function.
        # But 'global_metric.name' might not match function name if user edited code freely.
        # Let's inspect local_scope for callables.
        
        # Ideally, we should enforce function name = metric name or something.
        # Or look for a callable that isn't 'np'.
        func = None
        for k, v in local_scope.items():
            if callable(v) and k != 'np':
                func = v
                break
        
        if not func:
            return None, "No callable function found in code."

        # Registered-kind args arrive as RegisteredBlob carriers; run their
        # decode() hook in-process (this path already exec()s metric code
        # locally) so the metric sees the decoded object, not the bytes.
        for _k, _v in list(call_kwargs.items()):
            if isinstance(_v, RegisteredBlob):
                call_kwargs[_k] = _resolve_registered_blob(_v)
            elif isinstance(_v, (list, tuple)) and any(isinstance(_x, RegisteredBlob) for _x in _v):
                # Aggregated metrics receive a list of per-sample RegisteredBlobs
                # (one carrier per scan) — decode each so the metric sees arrays.
                call_kwargs[_k] = [_resolve_registered_blob(_x) if isinstance(_x, RegisteredBlob) else _x
                                   for _x in _v]

        # Call it
        result = func(**call_kwargs)
        
        # Handle nan/inf
        if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
             return None, "Result is NaN or Inf"

        return float(result), None

    except Exception as e:
        print(f"DEBUG: Error evaluating metric {global_metric.name}: {e}")
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Sandboxed metric execution (Phase 2)
# ---------------------------------------------------------------------------
#
# evaluate_dynamic_metric (above) exec()s untrusted Python in-process. That's
# fine for trusted-LAN deployments but a remote-code-execution invitation on
# any public site. The path below shells out to a locked-down docker
# container instead. It's gated by the BENCHHUB_SANDBOX_METRICS env var and
# defaults OFF — production keeps the in-process path until we flip the
# switch and prove the container is healthy in a staging deploy.
#
# Batched on purpose: container start-up is ~1-2s, so per-call invocation
# would dominate runtime. Instead we send N kwargs dicts in one job and the
# container returns N results.

_DEFAULT_SANDBOX_IMAGE = 'benchhub-runner'


def _sandbox_enabled():
    """One central toggle so callers don't have to re-check the env var."""
    return os.environ.get('BENCHHUB_SANDBOX_METRICS') == '1'


def _build_kwargs(arg_mappings, context, *, typed=False):
    """Turn the leaderboard-metric arg_mappings + a sample context into the
    actual function kwargs. Same logic as the in-process path so behavior
    matches when callers swap one for the other.

    When `typed=True`, `__typed__<key>` is preferred over the bare key —
    callers in evaluate_dynamic_metric set this from the metric's
    `input_kinds` declaration."""
    call_kwargs = {}
    for arg, mapping_key in arg_mappings.items():
        if mapping_key.startswith('SCALAR:'):
            val_str = mapping_key[7:]
            try:
                v = float(val_str)
                if v.is_integer():
                    v = int(v)
                call_kwargs[arg] = v
            except ValueError:
                call_kwargs[arg] = val_str
        elif mapping_key not in context and f'__typed__{mapping_key}' not in context:
            call_kwargs[arg] = None
        else:
            call_kwargs[arg] = _resolve_context_value(context, mapping_key, typed)
    return call_kwargs


def _jsonify_kwarg(v):
    """Make a metric/viz kwarg JSON-serialisable for the sandbox job.

    Typed `bh.*` instances can't cross JSON, so encode each to the portable
    form the harness rebuilds: `{"__bh__": kind, "params": ..., "b64": ...}`.
    Lists/dicts are walked (label_list, aggregated-viz value lists); numpy
    scalars/arrays degrade to python; everything else passes through (and
    json.dumps will surface anything still unserialisable as a fatal)."""
    if isinstance(v, DataType):
        return {'__bh__': v.kind, 'params': v.params or {},
                'b64': base64.b64encode(v.encode()).decode('ascii')}
    if isinstance(v, RegisteredBlob):
        # Registered kind: ship the raw bytes + the optional decode source.
        # The harness runs decode() in-container (or hands over the bytes).
        return {'__dtype__': v.kind, 'decode': v.decode_code,
                'params': v.params or {},
                'b64': base64.b64encode(v.blob).decode('ascii')}
    if isinstance(v, (bytes, bytearray)):
        return {'__bytes__': base64.b64encode(bytes(v)).decode('ascii')}
    if isinstance(v, dict):
        return {k: _jsonify_kwarg(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify_kwarg(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _build_job(global_metric, contexts, arg_mappings_json):
    """Shared by both backends: turn (metric, contexts, mapping_json) into
    the JSON job dict that runner/harness.run_job consumes."""
    try:
        arg_mappings = json.loads(arg_mappings_json) if arg_mappings_json else {}
    except (TypeError, json.JSONDecodeError):
        arg_mappings = {}

    typed = _metric_wants_typed(global_metric)
    kwargs_list = [
        {k: _jsonify_kwarg(v)
         for k, v in _build_kwargs(arg_mappings, ctx or {}, typed=typed).items()}
        for ctx in contexts
    ]
    return {
        'kind': 'metric',
        'code': global_metric.python_code,
        'kwargs_list': kwargs_list,
        'include_numpy': True,
        # Typed metrics need `bh` in the container + their args decoded;
        # harmless for primitive metrics (just an extra import).
        'include_benchhub': True,
    }


def _shape_results(payload_or_fatal, n_contexts):
    """Turn the harness payload (or a fatal-string sentinel) into the
    per-context tuple list callers expect.

    payload_or_fatal:
        - dict (harness output) → use payload['results'], pad if short
        - str → fatal; every context gets (None, str)
    """
    if isinstance(payload_or_fatal, str):
        return [(None, payload_or_fatal)] * n_contexts

    payload = payload_or_fatal
    if payload.get('fatal'):
        return [(None, payload['fatal'])] * n_contexts

    results = payload.get('results') or []
    out = []
    for i in range(n_contexts):
        r = results[i] if i < len(results) else {"value": None, "error": "missing result"}
        out.append((r.get('value'), r.get('error')))
    return out


def _run_via_docker(job, *, image, timeout_seconds, memory, cpus, docker_path):
    """Subprocess path: spawn a one-shot container, pipe job in, parse stdout.
    Returns either the harness payload dict or a fatal-string sentinel."""
    cmd = [
        docker_path, 'run', '--rm', '-i',
        '--network=none',
        '--read-only',
        '--tmpfs', '/tmp:size=64m,exec',
        f'--memory={memory}',
        f'--cpus={cpus}',
        '--security-opt', 'no-new-privileges',
        image,
        # Override the image's default CMD (gunicorn server) and run the
        # CLI harness directly. The HTTP server path uses _run_via_http;
        # this branch is for local dev / docker-on-host deploys.
        'python', '/app/harness.py',
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return f"docker not found (looked for {docker_path!r})"
    except subprocess.TimeoutExpired:
        return f"sandbox timed out after {timeout_seconds}s"

    if proc.returncode != 0 and not proc.stdout:
        stderr_excerpt = (proc.stderr or '').strip().splitlines()
        tail = ' / '.join(stderr_excerpt[-3:]) if stderr_excerpt else 'no stderr'
        return f"sandbox exited rc={proc.returncode}: {tail}"

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return f"sandbox returned non-JSON: {e}"


def _run_via_http(job, *, url, timeout_seconds):
    """HTTP path: POST the job to a runner-as-separate-Fly-app instance.
    Returns either the harness payload dict or a fatal-string sentinel."""
    if _requests is None:
        return "requests library not available"

    try:
        resp = _requests.post(
            url,
            json=job,
            timeout=timeout_seconds,
            headers={'Content-Type': 'application/json'},
        )
    except _requests.exceptions.Timeout:
        return f"sandbox timed out after {timeout_seconds}s"
    except _requests.exceptions.ConnectionError as e:
        return f"sandbox unreachable at {url}: {e}"
    except Exception as e:  # pragma: no cover — defensive
        return f"sandbox HTTP error: {e}"

    # Server returns 4xx/5xx with JSON body when it can; surface the body
    # if present, the status otherwise.
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except ValueError:
            return f"sandbox HTTP {resp.status_code}: {resp.text[:200]}"
        if isinstance(body, dict) and body.get('fatal'):
            return body['fatal']
        return f"sandbox HTTP {resp.status_code}"

    try:
        return resp.json()
    except ValueError as e:
        return f"sandbox returned non-JSON: {e}"


def _metric_is_trusted(global_metric):
    """True when a metric may run in-process (bypassing the docker sandbox): its
    owner is staff. The sandbox exists to contain UNTRUSTED user code — admin-
    authored metrics aren't that. Escape hatch for aggregated metrics whose
    payload (a whole sequence of point clouds) can't fit a sandbox container."""
    try:
        from app import User, is_admin            # lazy: keep module DB-decoupled
        owner_id = getattr(global_metric, 'owner_user_id', None)
        return bool(owner_id is not None and is_admin(User.query.get(owner_id)))
    except Exception:
        return False


def evaluate_in_sandbox(
    global_metric,
    contexts,
    arg_mappings_json,
    *,
    url=None,
    image=None,
    timeout_seconds=60,
    memory='512m',
    cpus='1',
    docker_path='docker',
):
    """Run a metric across many sample contexts inside a hardened sandbox.

    Backends, in priority order:

    1. **HTTP** — when ``url`` is passed or ``BENCHHUB_SANDBOX_URL`` is set,
       POST the job to a long-running runner service (originally a separate
       Fly app; archived under archive/fly/ since the migration to the home
       box — see archive/fly/DEPLOY.md § 9 if you ever revive that path).

    2. **Docker subprocess** — fall back to spawning a one-shot container
       locally. Useful for development, the integration test suite, and
       single-VM deploys where the runner image lives next to the app.

    Returns a list of (value, error) tuples — one per context — matching
    the shape of repeated calls to evaluate_dynamic_metric().

    Fatal failures (image missing, container crash, network error,
    malformed response) propagate the same error to every context. That
    matches tasks.py's expectation: a per-metric fatal should mark every
    sample, not silently drop them.
    """
    if not contexts:
        return []

    # Oversized aggregated metrics (e.g. point-cloud PQ / LSTQ over a whole
    # sequence) ship ONE context holding every sample's arrays — hundreds of MB
    # that can't fit a sandbox container's body + 512m memory cap, so the
    # sandbox returns a fatal. The sandbox exists to contain UNTRUSTED user
    # code; when the metric is admin-authored (trusted) we evaluate it
    # in-process instead. Cheap per-sample metrics never hit this.
    try:
        _one = max(1, len(json.dumps(_build_job(global_metric, contexts[:1], arg_mappings_json))))
    except Exception:
        _one = 0
    if _one > 80_000_000 and _metric_is_trusted(global_metric):
        return [evaluate_dynamic_metric(global_metric, c, arg_mappings_json) for c in contexts]

    # All contexts ship into ONE container, which has a hard memory cap
    # (`memory`, default 512m). Tiny payloads (labels/text/scalars) fit by the
    # thousands, but file-backed arrays (depth .npz, masks) can OOM-kill the
    # container (rc 137) at a few dozen samples. Adaptively chunk so each
    # container's JSON payload stays under MAX_CHUNK_BYTES: probe one context's
    # job size, then size the chunk. Small-payload metrics (the common case)
    # stay single-container, so classification/NLP scoring isn't slowed.
    MAX_CHUNK_BYTES = 80_000_000
    try:
        per_ctx = max(1, len(json.dumps(_build_job(global_metric, contexts[:1], arg_mappings_json))))
    except Exception:
        per_ctx = MAX_CHUNK_BYTES  # un-measurable → don't chunk
    chunk = len(contexts) if per_ctx <= 0 else max(1, MAX_CHUNK_BYTES // per_ctx)

    def _run(ctxs):
        job = _build_job(global_metric, ctxs, arg_mappings_json)
        payload = _dispatch(job, url=url, image=image, timeout_seconds=timeout_seconds,
                            memory=memory, cpus=cpus, docker_path=docker_path)
        return _shape_results(payload, len(ctxs))

    if chunk >= len(contexts):
        return _run(contexts)
    results = []
    for i in range(0, len(contexts), chunk):
        results.extend(_run(contexts[i:i + chunk]))
    return results


def _dispatch(job, *, url=None, image=None, timeout_seconds=60,
              memory='512m', cpus='1', docker_path='docker'):
    """Send a harness job to whichever backend is configured (HTTP runner
    service if a URL is set, else a one-shot docker container) and return
    the raw payload dict or a fatal-error string. Shared by the metric and
    visualization entrypoints."""
    url = url or os.environ.get('BENCHHUB_SANDBOX_URL')
    if url:
        return _run_via_http(job, url=url, timeout_seconds=timeout_seconds)
    image = image or os.environ.get('BENCHHUB_SANDBOX_IMAGE') or _DEFAULT_SANDBOX_IMAGE
    return _run_via_docker(
        job, image=image, timeout_seconds=timeout_seconds,
        memory=memory, cpus=cpus, docker_path=docker_path)


def evaluate_viz_in_sandbox(code, kwargs, *, function_name=None, **opts):
    """Render a visualization inside the sandbox. `code` is the viz source
    (a function returning a PIL.Image); `kwargs` are its decoded arguments.
    Returns `(png_bytes, error)` — `png_bytes` is None on any failure.

    `opts` forwards backend knobs (url / image / timeout_seconds / memory /
    cpus / docker_path) to `_dispatch`."""
    job = {
        'kind': 'visualization',
        'code': code,
        'kwargs_list': [{k: _jsonify_kwarg(v) for k, v in (kwargs or {}).items()}],
        'function_name': function_name,
        'include_numpy': True,
        'include_benchhub': True,
    }
    payload = _dispatch(job, **opts)
    if isinstance(payload, str):
        return None, payload
    if payload.get('fatal'):
        return None, payload['fatal']
    results = payload.get('results') or []
    if not results:
        return None, "sandbox returned no result"
    r = results[0]
    if r.get('error'):
        return None, r['error']
    b64 = r.get('png_b64')
    if not b64:
        return None, "sandbox returned no image"
    try:
        return base64.b64decode(b64), None
    except Exception as e:
        return None, f"sandbox returned bad base64: {e}"


def visualize_dtype_in_sandbox(visualize_code, blob, params=None, **opts):
    """Render a user-registered data type's stored bytes via its sandboxed
    `visualize(blob, params) -> PIL.Image`. Returns `(png_bytes, error)`.
    Reuses the visualization job path (blob crosses JSON as base64)."""
    return evaluate_viz_in_sandbox(
        visualize_code, {'blob': blob, 'params': params or {}},
        function_name='visualize', **opts)


def sandbox_evaluate_one(global_metric, context, arg_mappings_json, **kwargs):
    """Single-context convenience wrapper, matches evaluate_dynamic_metric's
    return shape (value, error)."""
    results = evaluate_in_sandbox(
        global_metric, [context], arg_mappings_json, **kwargs
    )
    return results[0] if results else (None, "sandbox returned no result")


def _load_gt_array(cf, upload_folder):
    """Load a depth/image GT custom field into a numpy array. Returns
    None on any failure so the metric just gets a missing key it can
    treat as "skip this sample".

    `cf.value_text` is a path relative to the BenchHub upload folder;
    the actual file is `<upload_folder>/<value_text>`.
    """
    import os
    rel = cf.value_text
    if not rel:
        return None
    full = os.path.join(upload_folder, rel)
    if not os.path.exists(full):
        return None
    try:
        if cf.data_type == 'depth':
            with np.load(full) as data:
                # The HF auto-importer stores depth under key 'depth';
                # legacy uploads may use the first array key.
                if 'depth' in data:
                    return np.asarray(data['depth'])
                first = next(iter(data.keys()))
                return np.asarray(data[first])
        if cf.data_type == 'image':
            from PIL import Image as _PILImage
            return np.asarray(_PILImage.open(full).convert('RGB'))
        if cf.data_type == 'mask':
            # Segmentation masks are single-channel class-id maps — must NOT
            # be read as RGB (that gives an (H,W,3) array that breaks bh.Mask
            # + every IoU comparison). Mirror bh.Mask.decode's mode handling.
            from PIL import Image as _PILImage
            img = _PILImage.open(full)
            if img.mode == 'I;16':
                return np.asarray(img, dtype=np.uint16)
            if img.mode in ('L', 'P', 'I'):
                return np.asarray(img)
            a = np.asarray(img)
            return a[..., 0] if a.ndim == 3 else a
    except Exception as e:
        print(f"DEBUG: _load_gt_array failed for {cf.name}: {e}")
    return None


def _load_sub_pred_for_sample(submission_folder, folder_name, sample_name):
    """Load a submission's prediction for one sample under one
    `<folder_name>/` directory. Recognizes:
      - `<sample>.txt`              → float (scalar prediction)
      - `<sample>.png`/.jpg/.jpeg   → numpy RGB array (image)
      - `<sample>_<W>x<H>.npz`      → numpy array (depth)
      - `<sample>.npz`              → numpy array (depth, dim-suffix-less)
    Returns None when no matching file exists for this sample."""
    import os
    folder = os.path.join(submission_folder, folder_name)
    if not os.path.isdir(folder):
        return None
    # Image
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
        p = os.path.join(folder, f"{sample_name}{ext}")
        if os.path.exists(p):
            try:
                from PIL import Image as _PILImage
                return np.asarray(_PILImage.open(p).convert('RGB'))
            except Exception as e:
                print(f"DEBUG: load image pred failed for {p}: {e}")
                return None
    # Depth (with or without dim suffix)
    candidates = []
    try:
        for fname in os.listdir(folder):
            if fname == f"{sample_name}.npz":
                candidates.append(fname)
            elif (fname.startswith(f"{sample_name}_") and fname.endswith('.npz')):
                tail = fname[len(sample_name) + 1:-len('.npz')]
                if 'x' in tail and all(p.isdigit() for p in tail.split('x', 1)):
                    candidates.append(fname)
    except OSError:
        return None
    if candidates:
        try:
            with np.load(os.path.join(folder, candidates[0])) as data:
                if 'depth' in data:
                    return np.asarray(data['depth'])
                first = next(iter(data.keys()))
                return np.asarray(data[first])
        except Exception as e:
            print(f"DEBUG: load npz pred failed for {candidates[0]}: {e}")
    # Scalar — default to numeric. If the contents aren't a number
    # (e.g. text-label submission like 'neg' / 'pos'), surface the raw
    # stripped string so exact-match metrics can still consume it.
    p = os.path.join(folder, f"{sample_name}.txt")
    if os.path.exists(p):
        try:
            raw = open(p).read().strip()
        except OSError:
            return None
        try:
            return float(raw)
        except ValueError:
            return raw or None
    return None


def get_metric_context(sample, sub=None, submission_folder=None,
                       upload_folder=None, pointer_resolver=None,
                       paired_gt_provider=None):
    """
    Builds a context dictionary for metric evaluation.
    Includes GT fields and optionally Submission fields for a specific sample.

    submission_folder: Path to the root of the submission contents.
    upload_folder: Root of the BenchHub upload tree (resolves the
    relative paths stored in CustomField.value_text into absolute paths).
    Falls back to the env var BENCHHUB_UPLOAD_FOLDER when not provided.

    pointer_resolver: optional `(sample, cf) -> numpy_array_or_None`
    callback for pointer-mode datasets where the CustomField has no
    on-disk file. Caller (app.py) injects the bench_cache-backed
    resolver; the engine stays decoupled from the cache layer so this
    module is testable in isolation.

    paired_gt_provider: optional `(primary_sample) ->
    iterable[(CustomField, sample)]` callback that yields additional
    GT fields from sibling 'gt_source' datasets attached to the same
    leaderboard (paired-dataset shape — e.g. dirty-docs noisy/clean
    split across two HF repos). The yielded fields land in the
    context as `gt_<name>` like the primary sample's own fields,
    overriding any same-named primary field (gt_source wins on
    conflict).
    """
    import os
    context = {}
    if upload_folder is None:
        upload_folder = (
            os.environ.get('BENCHHUB_UPLOAD_FOLDER')
            or os.path.expanduser('~/.dtofbenchmarking/uploads')
        )

    # GT entropy used to come from the legacy HistogramData table; with
    # that gone, entropy is only available via the histogram CustomField
    # path below. Leave the variable unset so the helper below sets it.
    if False:
        try:
            counts = np.array([])
            counts = counts[counts > 0]
            if counts.sum() > 0:
                p = counts / counts.sum()
                context['gt_entropy'] = float(-np.sum(p * np.log2(p)))
            else:
                context['gt_entropy'] = 0.0
        except:
             context['gt_entropy'] = 0.0

    # GT Custom Fields. Scalars stay as floats; image/depth load lazily
    # as numpy arrays so a metric like rmse_<col>(gt, pred) can consume
    # them directly. Failures (missing file, corrupt NPZ) just skip
    # the field — the metric will see context.get('gt_<name>') is None
    # and can fall through to None / NaN rather than crash the eval.
    for cf in sample.custom_fields:
        if cf.data_type == 'scalar':
            _stash_typed(context, f"gt_{cf.name}", cf, cf.value_float)
            _stash_typed(context, cf.name, cf, cf.value_float)
        elif cf.data_type == 'text':
            _stash_typed(context, f"gt_{cf.name}", cf, cf.value_text)
            _stash_typed(context, cf.name, cf, cf.value_text)
        elif cf.data_type == 'label':
            # Inline-stored Label value (int or str) lives on value_text
            # as the JSON-encoded primitive. Parse back so typed metrics
            # receive a bh.Label(value) and legacy ones see the raw int/str.
            try:
                parsed = json.loads(cf.value_text) if cf.value_text else None
            except Exception:
                parsed = cf.value_text
            _stash_typed(context, f"gt_{cf.name}", cf, parsed)
            _stash_typed(context, cf.name, cf, parsed)
        elif cf.data_type == 'label_list':
            # Ranked top-K list stored as a JSON array on value_text.
            # Parse so the typed wrap builds a bh.LabelList; without this
            # branch the field is dropped from the context entirely and
            # metrics see pred=None (→ a misleading 0.0).
            try:
                parsed = json.loads(cf.value_text) if cf.value_text else None
            except Exception:
                parsed = cf.value_text
            _stash_typed(context, f"gt_{cf.name}", cf, parsed)
            _stash_typed(context, cf.name, cf, parsed)
        elif cf.data_type == 'json':
            try:
                parsed = json.loads(cf.value_text) if cf.value_text else None
            except Exception:
                parsed = cf.value_text
            _stash_typed(context, f"gt_{cf.name}", cf, parsed)
            _stash_typed(context, cf.name, cf, parsed)
        elif cf.data_type in ('image', 'depth', 'mask'):
            arr = _load_gt_array(cf, upload_folder)
            if arr is None and pointer_resolver is not None and getattr(cf, 'source_column', None):
                # Pointer-mode CustomField: no on-disk file, just a
                # reference to a row + column in an HF repo. Hand off
                # to the caller-supplied resolver, which is wired to
                # bench_cache (lazy fetch + LRU cache + eviction).
                try:
                    arr = pointer_resolver(sample, cf)
                except Exception as e:
                    print(f"DEBUG: pointer_resolver raised for {cf.name}: {e}")
                    arr = None
            _stash_typed(context, f"gt_{cf.name}", cf, arr)
            _stash_typed(context, cf.name, cf, arr)
        elif cf.data_type not in DTYPES:
            # User-registered kind: carry the raw bytes + decode hook.
            rb = _registered_blob_for_cf(cf, upload_folder)
            if rb is not None:
                context[f"gt_{cf.name}"] = rb
                context[cf.name] = rb

    # Paired-dataset GT: fold in CustomFields from any 'gt_source'
    # datasets attached to the LB whose sample name matches.
    # gt_source wins on conflict (it's the explicit GT; primary's
    # same-named field would usually be the input bytes).
    if paired_gt_provider is not None:
        try:
            paired_iter = list(paired_gt_provider(sample))
        except Exception as e:
            print(f"DEBUG: paired_gt_provider raised: {e}")
            paired_iter = []
        for partner_cf, partner_sample in paired_iter:
            if partner_cf.data_type == 'scalar':
                _stash_typed(context, f"gt_{partner_cf.name}", partner_cf, partner_cf.value_float)
                _stash_typed(context, partner_cf.name, partner_cf, partner_cf.value_float)
            elif partner_cf.data_type in ('image', 'depth'):
                arr = _load_gt_array(partner_cf, upload_folder)
                if (arr is None and pointer_resolver is not None
                        and getattr(partner_cf, 'source_column', None)):
                    try:
                        arr = pointer_resolver(partner_sample, partner_cf)
                    except Exception as e:
                        print(f"DEBUG: paired pointer_resolver raised for {partner_cf.name}: {e}")
                        arr = None
                _stash_typed(context, f"gt_{partner_cf.name}", partner_cf, arr)
                _stash_typed(context, partner_cf.name, partner_cf, arr)
            elif partner_cf.data_type == 'text' and partner_cf.value_text is not None:
                _stash_typed(context, f"gt_{partner_cf.name}", partner_cf, partner_cf.value_text)
                _stash_typed(context, partner_cf.name, partner_cf, partner_cf.value_text)

    if sub:
        # Submission fields (using CustomField)
        # We need efficient access. Assuming 'sub.custom_fields' is loaded.
        # This might be slow if we iterate list every time.
        # But for single sample context it's okay.
        
        # Get scalar/metric fields for this sample
        for cf in sub.custom_fields:
            if cf.sample_name == sample.name:
                if cf.data_type == 'metric' or cf.data_type == 'scalar':
                    _stash_typed(context, f"sub_{cf.name}", cf, cf.value_float)
                    _stash_typed(context, cf.name, cf, cf.value_float)

                    # [FIX] Fallback for lm_{id} naming: also provide friendly name in context
                    if cf.name.startswith('lm_'):
                        try:
                            lm_id = int(cf.name[3:])
                            from app import LeaderboardMetric
                            lm = LeaderboardMetric.query.get(lm_id)
                            if lm:
                                friendly_name = lm.target_name or lm.global_metric.name
                                _stash_typed(context, friendly_name, cf, cf.value_float)
                                _stash_typed(context, f"sub_{friendly_name}", cf, cf.value_float)
                        except: pass
                elif cf.data_type == 'text':
                    # Text predictions surfaced verbatim for string-
                    # comparison metrics (BLEU/EM/F1 over QA / translation).
                    _stash_typed(context, f"sub_{cf.name}", cf, cf.value_text)
                    _stash_typed(context, cf.name, cf, cf.value_text)
                elif cf.data_type == 'label':
                    # Label predictions stored as JSON-encoded primitive
                    # on value_text. Parse so typed metrics receive a
                    # bh.Label and legacy ones see the int/str.
                    try:
                        parsed = json.loads(cf.value_text) if cf.value_text else None
                    except Exception:
                        parsed = cf.value_text
                    _stash_typed(context, f"sub_{cf.name}", cf, parsed)
                    _stash_typed(context, cf.name, cf, parsed)
                elif cf.data_type == 'label_list':
                    # Ranked top-K predictions (Hits@K / top-K accuracy /
                    # MRR) stored as a JSON array. Parse so the typed wrap
                    # builds a bh.LabelList — otherwise the field is
                    # dropped and the metric sees pred=None → 0.0.
                    try:
                        parsed = json.loads(cf.value_text) if cf.value_text else None
                    except Exception:
                        parsed = cf.value_text
                    _stash_typed(context, f"sub_{cf.name}", cf, parsed)
                    _stash_typed(context, cf.name, cf, parsed)
                elif cf.data_type == 'json':
                    # JSON predictions (bboxes, span offsets, structured
                    # outputs). Deserialise so metric code sees the dict
                    # / list shape, not the raw string.
                    raw = cf.value_text or ''
                    try:
                        parsed = json.loads(raw) if raw else None
                    except Exception:
                        parsed = cf.value_text
                    _stash_typed(context, f"sub_{cf.name}", cf, parsed)
                    _stash_typed(context, cf.name, cf, parsed)
                elif cf.data_type == 'mask':
                    # Segmentation-mask predictions: load the single-channel
                    # class-id array + typed-wrap so iou_mask gets a real
                    # bh.Mask instead of the untyped (H,W,3) RGB array the
                    # folder-loader fallback produced (which broke scoring).
                    # Scoped to 'mask' only — depth/image preds keep their
                    # existing (folder-loader) path to avoid changing their
                    # already-working scores.
                    arr = _load_gt_array(cf, upload_folder)
                    _stash_typed(context, f"sub_{cf.name}", cf, arr)
                    _stash_typed(context, cf.name, cf, arr)
                elif cf.data_type not in DTYPES and cf.data_type != 'metric':
                    # Registered-kind prediction: carry raw bytes + decode hook
                    # (same RegisteredBlob the GT side emits).
                    rb = _registered_blob_for_cf(cf, upload_folder)
                    if rb is not None:
                        context[f"sub_{cf.name}"] = rb
                        context[cf.name] = rb

        # Add 'sub_peak' convenience if it exists
        # 'detect_custom_fields' creates CustomFields so it should be in there.
        
        # Load standard file-based metrics if submission_folder is provided
        if submission_folder:
            try:
                for folder_name in os.listdir(submission_folder):
                    folder_path = os.path.join(submission_folder, folder_name)
                    if not os.path.isdir(folder_path):
                        continue
                    # Histogram entropy convenience: any folder whose
                    # per-sample .npz carries `counts` (regardless of
                    # folder name — the legacy hist_/raw_histogram
                    # prefix requirement was dropped) gets its
                    # Shannon entropy surfaced as
                    # sub_entropy_<folder_name>. Bench-Hub still passes
                    # the raw .npz through _load_sub_pred_for_sample
                    # below, so dependency-chained metrics can grab
                    # bins/counts themselves if they need to.
                    hist_file = os.path.join(folder_path, f'{sample.name}.npz')
                    if os.path.exists(hist_file):
                        try:
                            with np.load(hist_file) as data:
                                if 'counts' in data:
                                    counts = data['counts']
                                    counts = counts[counts > 0]
                                    if counts.sum() > 0:
                                        p = counts / counts.sum()
                                        val = float(-np.sum(p * np.log2(p)))
                                        context[f'sub_entropy_{folder_name}'] = val
                                    else:
                                        context[f'sub_entropy_{folder_name}'] = 0.0
                        except Exception as e:
                            print(f"DEBUG: Error reading npz {folder_name} for {sample.name}: {e}")
                    # Bare-name prediction folder (the auto-LB convention:
                    # `<col>_pred/`). Load the per-sample file as scalar /
                    # image / depth depending on the file extension and
                    # surface it as `sub_<folder_name>` in the context.
                    arr = _load_sub_pred_for_sample(
                        submission_folder, folder_name, sample.name,
                    )
                    if arr is not None:
                        context[f'sub_{folder_name}'] = arr
                        # Also expose under bare name so dependency-chained
                        # metrics can reference the prediction directly,
                        # matching the scalar-side convention.
                        context.setdefault(folder_name, arr)
            except Exception as e:
                print(f"DEBUG: Error scanning submission folder: {e}")

    return context
def sort_metrics_by_dependency(metrics):
    """
    Topologically sorts a list of LeaderboardMetric objects based on dependencies defined in arg_mappings.
    """
    from collections import defaultdict, deque

    # Build graph
    adj = defaultdict(list)
    in_degree = {m.id: 0 for m in metrics}
    metric_map = {m.id: m for m in metrics}
    
    # Map output names to metric IDs for dependency resolution
    # Metric B depends on A if B uses A's name as input.
    # Name is target_name if set, else global_metric.name.
    name_to_id = {}
    for m in metrics:
        name = m.target_name if m.target_name else m.global_metric.name
        name_to_id[name] = m.id

    for m in metrics:
        try:
            mappings = json.loads(m.arg_mappings)
            # dependency is a value in mappings
            for dep_name in mappings.values():
                if dep_name in name_to_id:
                    dep_id = name_to_id[dep_name]
                    # If dependency is in the list we are sorting (self-reference or cycle check?)
                    # If dependency is another metric in this list, add edge.
                    if dep_id != m.id:
                        adj[dep_id].append(m.id)
                        in_degree[m.id] += 1
        except:
            pass

    # Kahn's Algorithm
    queue = deque([mid for mid, deg in in_degree.items() if deg == 0])
    sorted_metrics = []

    while queue:
        u_id = queue.popleft()
        sorted_metrics.append(metric_map[u_id])

        for v_id in adj[u_id]:
            in_degree[v_id] -= 1
            if in_degree[v_id] == 0:
                queue.append(v_id)

    # Check for cycles (if len mismatch, cycle exists or disconnected)
    # If cycle, just append remaining arbitrarily? Or prioritize valid ones?
    if len(sorted_metrics) != len(metrics):
        # Add remaining metrics (cycles)
        seen = set(m.id for m in sorted_metrics)
        for m in metrics:
            if m.id not in seen:
                sorted_metrics.append(m)
    
    return sorted_metrics
