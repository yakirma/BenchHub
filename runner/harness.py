"""In-container metric worker.

Reads a JSON job from stdin and writes a JSON result to stdout. Designed to
run inside a locked-down container (no network, read-only fs, memory/cpu
caps, short timeout) so untrusted user-supplied Python can't escape.

Job schema (stdin):
    {
        "code": "<python source defining one callable>",
        "kwargs_list": [{<arg>: <value>, ...}, ...],
        "function_name": "<optional: pick this callable>",
        "include_numpy": true   // inject `np` into the exec scope
    }

Result schema (stdout, single line of JSON):
    {
        "results": [
            {"value": <float|null>, "error": <string|null>},
            ...
        ],
        "fatal": <string|null>     // set when the *code* couldn't be loaded
                                    // at all; per-call errors land in results[i].error
    }

This module is intentionally dependency-light: stdlib only at import time.
The exec'd user code may import numpy / scipy / matplotlib — those are
pre-installed in the runner image; missing imports become per-call errors.
"""
import json
import math
import sys
import traceback


def _load_callable(code, function_name=None, include_numpy=True):
    """exec the user code in a fresh namespace and return (callable, fatal_error)."""
    namespace = {}
    if include_numpy:
        # NumPy is the only injection. Anything else the user wants must be
        # imported in their code (and must be in the runner image).
        try:
            import numpy as _np
            namespace['np'] = _np
        except ImportError:
            pass

    try:
        compile(code, '<user_metric>', 'exec')
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"

    try:
        exec(code, namespace)
    except Exception:
        return None, traceback.format_exc()

    # Pick the callable. Explicit name wins if supplied; otherwise pick the
    # first top-level callable that wasn't injected by us.
    injected = {'np', '__builtins__'}
    if function_name:
        candidate = namespace.get(function_name)
        if not callable(candidate):
            return None, f"function {function_name!r} not found or not callable"
        return candidate, None

    for k, v in namespace.items():
        if k in injected:
            continue
        if callable(v):
            return v, None
    return None, "No callable function found in code."


def _coerce_result(value):
    """Match the existing evaluate_dynamic_metric behavior: NaN/Inf → error."""
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        return None, f"Result is not a number: {e}"
    if math.isnan(f) or math.isinf(f):
        return None, "Result is NaN or Inf"
    return f, None


def run_job(job):
    """Pure-Python entrypoint. Takes a parsed dict, returns a result dict."""
    code = job.get('code', '')
    kwargs_list = job.get('kwargs_list', [])
    function_name = job.get('function_name')
    include_numpy = job.get('include_numpy', True)

    if not isinstance(kwargs_list, list):
        return {"results": [], "fatal": "kwargs_list must be a list"}

    func, fatal = _load_callable(code, function_name=function_name, include_numpy=include_numpy)
    if fatal is not None:
        return {"results": [], "fatal": fatal}

    results = []
    for kwargs in kwargs_list:
        if not isinstance(kwargs, dict):
            results.append({"value": None, "error": "kwargs must be a dict"})
            continue
        try:
            value = func(**kwargs)
        except Exception:
            results.append({"value": None, "error": traceback.format_exc()})
            continue
        coerced, err = _coerce_result(value)
        if err is not None:
            results.append({"value": None, "error": err})
        else:
            results.append({"value": coerced, "error": None})

    return {"results": results, "fatal": None}


def main(stdin=None, stdout=None):
    """CLI entrypoint. stdin/stdout overridable for tests."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    raw = stdin.read()
    try:
        job = json.loads(raw)
    except json.JSONDecodeError as e:
        out = {"results": [], "fatal": f"invalid JSON job: {e}"}
        stdout.write(json.dumps(out))
        return 2

    result = run_job(job)
    stdout.write(json.dumps(result))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
