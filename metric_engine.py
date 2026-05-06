import json
import os
import shlex
import subprocess
import traceback

import numpy as np

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


def evaluate_in_sandbox(
    global_metric,
    contexts,
    arg_mappings_json,
    *,
    image=None,
    timeout_seconds=60,
    memory='512m',
    cpus='1',
    docker_path='docker',
):
    """Run a metric across many sample contexts inside a docker container.

    Returns a list of (value, error) tuples — one per context — matching the
    shape of repeated calls to evaluate_dynamic_metric().

    On fatal failures (image missing, json parse error, container crash)
    every context gets the same fatal error. That's intentional: tasks.py
    treats per-sample errors per-row, and a fatal here is a per-metric
    failure that should propagate to every row.
    """
    image = image or os.environ.get('BENCHHUB_SANDBOX_IMAGE') or _DEFAULT_SANDBOX_IMAGE

    try:
        arg_mappings = json.loads(arg_mappings_json) if arg_mappings_json else {}
    except (TypeError, json.JSONDecodeError):
        arg_mappings = {}

    kwargs_list = [_build_kwargs(arg_mappings, ctx or {}) for ctx in contexts]

    job = {
        'code': global_metric.python_code,
        'kwargs_list': kwargs_list,
        'include_numpy': True,
    }

    cmd = [
        docker_path, 'run', '--rm', '-i',
        '--network=none',
        '--read-only',
        '--tmpfs', '/tmp:size=64m,exec',
        f'--memory={memory}',
        f'--cpus={cpus}',
        '--security-opt', 'no-new-privileges',
        image,
    ]

    fatal = None
    raw_stdout = ''
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
        fatal = f"docker not found (looked for {docker_path!r})"
    except subprocess.TimeoutExpired:
        fatal = f"sandbox timed out after {timeout_seconds}s"
    else:
        if proc.returncode != 0 and not proc.stdout:
            # Container exited non-zero AND wrote nothing — surface stderr so
            # operators can see why (likely image missing or OOM-killed).
            stderr_excerpt = (proc.stderr or '').strip().splitlines()
            tail = ' / '.join(stderr_excerpt[-3:]) if stderr_excerpt else 'no stderr'
            fatal = f"sandbox exited rc={proc.returncode}: {tail}"
        raw_stdout = proc.stdout

    if fatal is None:
        try:
            payload = json.loads(raw_stdout)
        except json.JSONDecodeError as e:
            fatal = f"sandbox returned non-JSON: {e}"
        else:
            if payload.get('fatal'):
                fatal = payload['fatal']
            else:
                results = payload.get('results') or []
                # Trust the harness ordering: results[i] corresponds to
                # contexts[i]. If lengths mismatch (shouldn't happen unless
                # the harness misbehaves), pad/truncate so callers don't
                # crash on a zip.
                out = []
                for i in range(len(contexts)):
                    r = results[i] if i < len(results) else {"value": None, "error": "missing result"}
                    out.append((r.get('value'), r.get('error')))
                return out

    # Fatal path: every context inherits the same error message.
    return [(None, fatal) for _ in contexts]


def sandbox_evaluate_one(global_metric, context, arg_mappings_json, **kwargs):
    """Single-context convenience wrapper, matches evaluate_dynamic_metric's
    return shape (value, error)."""
    results = evaluate_in_sandbox(
        global_metric, [context], arg_mappings_json, **kwargs
    )
    return results[0] if results else (None, "sandbox returned no result")


def get_metric_context(sample, sub=None, submission_folder=None):
    """
    Builds a context dictionary for metric evaluation.
    Includes GT fields and optionally Submission fields for a specific sample.
    
    submission_folder: Path to the root of the submission contents.
    """
    context = {}
    

    
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

    # GT Custom Fields (Scalars)
    for cf in sample.custom_fields:
        if cf.field_type == 'scalar':
             context[f"gt_{cf.name}"] = cf.value_float
             context[cf.name] = cf.value_float

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
                # Dynamic Histograms
                # Scan for folders starting with 'hist_' or 'raw_histogram'
            try:
                for folder_name in os.listdir(submission_folder):
                    folder_path = os.path.join(submission_folder, folder_name)
                    if os.path.isdir(folder_path) and (folder_name.startswith('hist_') or folder_name == 'raw_histogram'):
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
            except Exception as e:
                print(f"DEBUG: Error scanning submission folder for histograms: {e}")

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
