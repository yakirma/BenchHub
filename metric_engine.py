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
            elif mapping_key not in context:
                 # If usage is optional, maybe None? But for now it's required
                 # unless we handle defaults. 
                 # Let's check if the metric function has default values (complex to introspection)
                 # For now, pass None if missing
                 call_kwargs[arg] = None
            else:
                 call_kwargs[arg] = context[mapping_key]
        
        # Execute code
        local_scope = {'np': np} # Inject common libs
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


def _build_kwargs(arg_mappings, context):
    """Turn the leaderboard-metric arg_mappings + a sample context into the
    actual function kwargs. Same logic as the in-process path so behavior
    matches when callers swap one for the other."""
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
        elif mapping_key not in context:
            call_kwargs[arg] = None
        else:
            call_kwargs[arg] = context[mapping_key]
    return call_kwargs


def _build_job(global_metric, contexts, arg_mappings_json):
    """Shared by both backends: turn (metric, contexts, mapping_json) into
    the JSON job dict that runner/harness.run_job consumes."""
    try:
        arg_mappings = json.loads(arg_mappings_json) if arg_mappings_json else {}
    except (TypeError, json.JSONDecodeError):
        arg_mappings = {}

    kwargs_list = [_build_kwargs(arg_mappings, ctx or {}) for ctx in contexts]
    return {
        'code': global_metric.python_code,
        'kwargs_list': kwargs_list,
        'include_numpy': True,
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
       POST the job to a runner-as-separate-Fly-app instance. This is the
       production path once the runner has been deployed (see
       runner/fly.toml + DEPLOY.md § 9).

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
    job = _build_job(global_metric, contexts, arg_mappings_json)
    url = url or os.environ.get('BENCHHUB_SANDBOX_URL')

    if url:
        payload_or_fatal = _run_via_http(
            job, url=url, timeout_seconds=timeout_seconds,
        )
    else:
        image = image or os.environ.get('BENCHHUB_SANDBOX_IMAGE') or _DEFAULT_SANDBOX_IMAGE
        payload_or_fatal = _run_via_docker(
            job,
            image=image,
            timeout_seconds=timeout_seconds,
            memory=memory,
            cpus=cpus,
            docker_path=docker_path,
        )

    return _shape_results(payload_or_fatal, len(contexts))


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
        if cf.field_type == 'depth':
            with np.load(full) as data:
                # The HF auto-importer stores depth under key 'depth';
                # legacy uploads may use the first array key.
                if 'depth' in data:
                    return np.asarray(data['depth'])
                first = next(iter(data.keys()))
                return np.asarray(data[first])
        if cf.field_type == 'image':
            from PIL import Image as _PILImage
            return np.asarray(_PILImage.open(full).convert('RGB'))
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
    # Scalar
    p = os.path.join(folder, f"{sample_name}.txt")
    if os.path.exists(p):
        try:
            return float(open(p).read().strip())
        except (ValueError, OSError):
            return None
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

    # GT Entropy calculation
    if sample.histogram_data:
        try:
            counts = np.array(json.loads(sample.histogram_data.counts))
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
        if cf.field_type == 'scalar':
             context[f"gt_{cf.name}"] = cf.value_float
             context[cf.name] = cf.value_float
        elif cf.field_type in ('image', 'depth'):
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
            context[f"gt_{cf.name}"] = arr
            context[cf.name] = arr

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
            if partner_cf.field_type == 'scalar':
                context[f"gt_{partner_cf.name}"] = partner_cf.value_float
                context[partner_cf.name] = partner_cf.value_float
            elif partner_cf.field_type in ('image', 'depth'):
                arr = _load_gt_array(partner_cf, upload_folder)
                if (arr is None and pointer_resolver is not None
                        and getattr(partner_cf, 'source_column', None)):
                    try:
                        arr = pointer_resolver(partner_sample, partner_cf)
                    except Exception as e:
                        print(f"DEBUG: paired pointer_resolver raised for {partner_cf.name}: {e}")
                        arr = None
                context[f"gt_{partner_cf.name}"] = arr
                context[partner_cf.name] = arr
            elif partner_cf.field_type == 'text' and partner_cf.value_text is not None:
                context[f"gt_{partner_cf.name}"] = partner_cf.value_text
                context[partner_cf.name] = partner_cf.value_text

    if sub:
        # Submission fields (using CustomField)
        # We need efficient access. Assuming 'sub.custom_fields' is loaded.
        # This might be slow if we iterate list every time.
        # But for single sample context it's okay.
        
        # Get scalar/metric fields for this sample
        for cf in sub.custom_fields:
            if cf.sample_name == sample.name:
                if cf.field_type == 'metric' or cf.field_type == 'scalar':
                     context[f"sub_{cf.name}"] = cf.value_float
                     context[cf.name] = cf.value_float # Also store without prefix for direct access
                     
                     # [FIX] Fallback for lm_{id} naming: also provide friendly name in context
                     if cf.name.startswith('lm_'):
                         try:
                             lm_id = int(cf.name[3:])
                             from app import LeaderboardMetric
                             lm = LeaderboardMetric.query.get(lm_id)
                             if lm:
                                 friendly_name = lm.target_name or lm.global_metric.name
                                 context[friendly_name] = cf.value_float
                                 context[f"sub_{friendly_name}"] = cf.value_float
                         except: pass
        
        # Add 'sub_peak' convenience if it exists
        # 'detect_custom_fields' creates CustomFields so it should be in there.
        
        # Load standard file-based metrics if submission_folder is provided
        if submission_folder:
            try:
                for folder_name in os.listdir(submission_folder):
                    folder_path = os.path.join(submission_folder, folder_name)
                    if not os.path.isdir(folder_path):
                        continue
                    # Histogram entropy convenience.
                    if folder_name.startswith('hist_') or folder_name == 'raw_histogram':
                        hist_file = os.path.join(folder_path, f'{sample.name}.npz')
                        if os.path.exists(hist_file):
                            try:
                                with np.load(hist_file) as data:
                                    counts = data['counts']
                                    counts = counts[counts > 0]
                                    if counts.sum() > 0:
                                        p = counts / counts.sum()
                                        val = float(-np.sum(p * np.log2(p)))
                                        context[f'sub_entropy_{folder_name}'] = val
                                    else:
                                        context[f'sub_entropy_{folder_name}'] = 0.0
                            except Exception as e:
                                print(f"DEBUG: Error reading histogram {folder_name} for {sample.name}: {e}")
                        continue
                    # `metric_*` folders are picked up via Submission.custom_fields above.
                    if folder_name.startswith('metric_'):
                        continue
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
