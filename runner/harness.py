"""In-container metric / visualization worker.

Reads a JSON job from stdin and writes a JSON result to stdout. Designed to
run inside a locked-down container (no network, read-only fs, memory/cpu
caps, short timeout) so untrusted user-supplied Python can't escape.

Job schema (stdin):
    {
        "kind": "metric" | "visualization",   // default "metric"
        "code": "<python source defining one callable>",
        "kwargs_list": [{<arg>: <value>, ...}, ...],
        "function_name": "<optional: pick this callable>",
        "include_numpy": true,                  // inject `np`
        "include_benchhub": false               // inject `bh` + decode typed args
    }

Typed args (bh.Image / bh.Depth / …) can't cross JSON directly, so the
server encodes each as {"__bh__": "<kind>", "params": {...}, "b64": "<base64
of instance.encode()>"} and we rebuild it here via benchhub.DTYPES. Lists
and dicts are walked recursively (label_list, aggregated-viz value lists).

Result schema (stdout, single line of JSON):
    metric        → {"results": [{"value": <float|null>, "error": <str|null>}, ...]}
    visualization → {"results": [{"png_b64": <str|null>, "error": <str|null>}, ...]}
    fatal         → set when the *code* couldn't be loaded at all.

Dependency-light: stdlib only at import time. numpy / benchhub / Pillow are
pre-installed in the runner image; a missing import becomes a per-call error.
"""
import base64
import json
import math
import sys
import traceback


def _decode_arg(v):
    """Rebuild typed bh.* instances from their portable JSON form; walk
    lists/dicts so nested typed values (label_list, aggregated lists) are
    reconstructed too. Everything else passes through unchanged."""
    if isinstance(v, dict):
        if '__bh__' in v:
            import benchhub as _bh
            cls = _bh.DTYPES[v['__bh__']]
            return cls.decode(base64.b64decode(v.get('b64') or b''),
                              v.get('params') or {})
        return {k: _decode_arg(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode_arg(x) for x in v]
    return v


def _exec_namespace(include_numpy=True, include_benchhub=False):
    """Fresh exec scope with the same injections as the in-process paths:
    `np` for metrics/viz, `bh`/`benchhub` for typed code, `Image`/`PIL` for
    visualizations. Returns (namespace, injected_names)."""
    ns = {}
    if include_numpy:
        try:
            import numpy as _np
            ns['np'] = _np
            ns['numpy'] = _np
        except ImportError:
            pass
    if include_benchhub:
        try:
            import benchhub as _bh
            ns['bh'] = _bh
            ns['benchhub'] = _bh
        except ImportError:
            pass
    try:
        from PIL import Image as _Image
        import PIL as _PIL
        ns['Image'] = _Image
        ns['PIL'] = _PIL
    except ImportError:
        pass
    return ns, set(ns) | {'__builtins__'}


def _load_callable(code, function_name=None, *, include_numpy=True,
                   include_benchhub=False):
    """exec the user code in a fresh namespace and return
    (callable, namespace, fatal_error)."""
    namespace, injected = _exec_namespace(include_numpy, include_benchhub)

    try:
        compile(code, '<user_code>', 'exec')
    except SyntaxError as e:
        return None, namespace, f"SyntaxError: {e}"

    try:
        exec(code, namespace)
    except Exception:
        return None, namespace, traceback.format_exc()

    if function_name:
        candidate = namespace.get(function_name)
        if not callable(candidate):
            return None, namespace, f"function {function_name!r} not found or not callable"
        return candidate, namespace, None

    for k, v in namespace.items():
        if k in injected:
            continue
        if callable(v):
            return v, namespace, None
    return None, namespace, "No callable function found in code."


def _coerce_result(value):
    """Match evaluate_dynamic_metric: NaN/Inf → error, non-number → error."""
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        return None, f"Result is not a number: {e}"
    if math.isnan(f) or math.isinf(f):
        return None, "Result is NaN or Inf"
    return f, None


def _image_to_png_b64(result, namespace):
    """Turn a visualization's return value (a PIL.Image) into base64 PNG.
    Returns (png_b64, error)."""
    Image = namespace.get('Image')
    if Image is None:
        from PIL import Image  # always present in the runner image
    if not isinstance(result, Image.Image):
        return None, (f"visualization must return a PIL.Image, got "
                      f"{type(result).__name__}")
    import io
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii'), None


def run_job(job):
    """Pure-Python entrypoint. Takes a parsed dict, returns a result dict."""
    kind = job.get('kind', 'metric')
    code = job.get('code', '')
    kwargs_list = job.get('kwargs_list', [])
    function_name = job.get('function_name')
    include_numpy = job.get('include_numpy', True)
    include_benchhub = job.get('include_benchhub', False)

    if not isinstance(kwargs_list, list):
        return {"results": [], "fatal": "kwargs_list must be a list"}

    func, namespace, fatal = _load_callable(
        code, function_name=function_name,
        include_numpy=include_numpy, include_benchhub=include_benchhub)
    if fatal is not None:
        return {"results": [], "fatal": fatal}

    results = []
    for kwargs in kwargs_list:
        if not isinstance(kwargs, dict):
            results.append({"value": None, "error": "kwargs must be a dict"})
            continue
        try:
            decoded = {k: _decode_arg(v) for k, v in kwargs.items()}
            value = func(**decoded)
        except Exception:
            err = traceback.format_exc()
            results.append({"png_b64": None, "error": err}
                           if kind == 'visualization'
                           else {"value": None, "error": err})
            continue
        if kind == 'visualization':
            png_b64, err = _image_to_png_b64(value, namespace)
            results.append({"png_b64": png_b64, "error": err})
        else:
            coerced, err = _coerce_result(value)
            results.append({"value": coerced, "error": err})

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
