"""Papers With Code REST API client.

Thin wrapper around https://paperswithcode.com/api/v1/ — only the
endpoints BenchHub's import flow needs, with caching so the admin
search page doesn't burn the public rate limit.

PWC's data model: tasks own datasets, datasets are evaluated by
"evaluations" (a.k.a. benchmarks), each evaluation has metrics +
result rows. A dataset may have a `huggingface_url` field; BenchHub
only supports HF-mirrored ones at import time.

Public surface:
- search_datasets(q) → list[dict]
- list_evaluations_for_dataset(dataset_id) → list[dict]
- get_evaluation(evaluation_id) → dict (with metrics + results inlined)
"""
import os
import re
import time
from urllib.parse import urlparse

import requests

PWC_BASE = "https://paperswithcode.com/api/v1"
DEFAULT_TIMEOUT = 15
USER_AGENT = "BenchHub-PWC-Importer/1.0 (+https://benchhub.fly.dev)"

# Tiny in-process TTL cache. Key: (endpoint, frozenset(params)).
_CACHE = {}
_CACHE_TTL_SECONDS = 600  # 10 min — stable enough for an admin browse session


class PwcError(Exception):
    """Raised when a PWC call returns an unrecoverable error or the
    response shape doesn't match what we expect."""


def _get(path, **params):
    """Cached GET against the PWC API. `path` is relative to PWC_BASE."""
    key = (path, frozenset(params.items()))
    now = time.time()
    cached = _CACHE.get(key)
    if cached is not None:
        ts, body = cached
        if now - ts < _CACHE_TTL_SECONDS:
            return body
    url = f"{PWC_BASE}{path}"
    try:
        resp = requests.get(
            url, params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise PwcError(f"Network error reaching {url}: {e}") from e
    if resp.status_code != 200:
        raise PwcError(
            f"PWC returned HTTP {resp.status_code} for {url}: "
            f"{resp.text[:200]}"
        )
    try:
        body = resp.json()
    except ValueError as e:
        raise PwcError(f"PWC response wasn't JSON: {e}") from e
    _CACHE[(path, frozenset(params.items()))] = (now, body)
    return body


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _hf_repo_from_url(url):
    """Extract `owner/repo` from a huggingface.co dataset URL. Returns
    None if the URL isn't a recognizable HF dataset link.

    Accepted shapes:
    - https://huggingface.co/datasets/imagenet-1k
    - https://huggingface.co/datasets/owner/repo
    - https://huggingface.co/datasets/owner/repo/tree/main
    """
    if not url:
        return None
    try:
        u = urlparse(url)
    except ValueError:
        return None
    if u.netloc.lower() not in ('huggingface.co', 'www.huggingface.co'):
        return None
    parts = [p for p in u.path.split('/') if p]
    # Expect ['datasets', '<repo>'] or ['datasets', '<owner>', '<repo>'] (+ extras).
    if not parts or parts[0] != 'datasets':
        return None
    if len(parts) == 2:
        return parts[1]
    if len(parts) >= 3:
        return f"{parts[1]}/{parts[2]}"
    return None


def search_datasets(q, *, limit=30):
    """Search PWC datasets by name. Returns each row's id, name,
    full_name, paper, hf_repo (when discoverable), so callers can
    filter to the HF-linked subset."""
    body = _get("/datasets/", q=q or '', items_per_page=limit)
    results = body.get('results') if isinstance(body, dict) else body
    rows = []
    for row in results or []:
        if not isinstance(row, dict):
            continue
        hf_url = row.get('huggingface_url') or ''
        hf_repo = _hf_repo_from_url(hf_url)
        rows.append({
            'id': row.get('id'),
            'name': row.get('name'),
            'full_name': row.get('full_name') or row.get('name'),
            'description': (row.get('description') or '').strip(),
            'paper': row.get('paper'),
            'huggingface_url': hf_url,
            'hf_repo': hf_repo,
            'url': row.get('url'),
        })
    return rows


# ---------------------------------------------------------------------------
# Evaluations (benchmarks)
# ---------------------------------------------------------------------------


def list_evaluations_for_dataset(dataset_id):
    """All benchmarks PWC tracks for a dataset. Returns each row's
    id, task, dataset (the parent), description."""
    body = _get(f"/datasets/{dataset_id}/evaluations/", items_per_page=100)
    results = body.get('results') if isinstance(body, dict) else body
    out = []
    for row in results or []:
        if not isinstance(row, dict):
            continue
        out.append({
            'id': row.get('id'),
            'task': row.get('task'),
            'description': (row.get('description') or '').strip(),
        })
    return out


def get_evaluation_metrics(evaluation_id):
    """Metric definitions on a benchmark — name + sort direction +
    (when present) the formal metric reference."""
    body = _get(f"/evaluations/{evaluation_id}/metrics/", items_per_page=50)
    results = body.get('results') if isinstance(body, dict) else body
    out = []
    for row in results or []:
        if not isinstance(row, dict):
            continue
        # PWC uses `is_loss` as the inverse of higher-is-better.
        is_loss = bool(row.get('is_loss'))
        out.append({
            'id': row.get('id'),
            'name': row.get('name'),
            'description': (row.get('description') or '').strip(),
            'sort_direction': 'lower_is_better' if is_loss else 'higher_is_better',
        })
    return out


def get_evaluation_results(evaluation_id, *, max_pages=5):
    """Result rows on a benchmark. PWC paginates; we walk up to
    max_pages so we don't blow the cache on an LB with 800 rows."""
    out = []
    page = 1
    while page <= max_pages:
        body = _get(
            f"/evaluations/{evaluation_id}/results/",
            page=page, items_per_page=100,
        )
        if not isinstance(body, dict):
            break
        for row in (body.get('results') or []):
            if not isinstance(row, dict):
                continue
            out.append({
                'id': row.get('id'),
                'best_metric': row.get('best_metric'),
                'metrics': row.get('metrics') or {},
                'methodology': row.get('methodology') or '',
                'paper': row.get('paper') or '',
                'paper_title': row.get('paper_title') or '',
                'paper_url': row.get('paper_url') or '',
                'evaluated_on': row.get('evaluated_on'),
                'external_source_url': row.get('external_source_url'),
                'uses_additional_data': bool(row.get('uses_additional_data')),
            })
        if not body.get('next'):
            break
        page += 1
    return out


def get_evaluation(evaluation_id):
    """Single benchmark with metrics + results inlined. The shape the
    BenchHub admin import preview consumes."""
    body = _get(f"/evaluations/{evaluation_id}/")
    if not isinstance(body, dict):
        raise PwcError(f"Unexpected shape for evaluation {evaluation_id}")
    metrics = get_evaluation_metrics(evaluation_id)
    results = get_evaluation_results(evaluation_id)
    return {
        'id': body.get('id'),
        'task': body.get('task'),
        'dataset': body.get('dataset'),
        'description': (body.get('description') or '').strip(),
        'mirror_url': body.get('mirror_url') or body.get('url'),
        'metrics': metrics,
        'results': results,
    }


# ---------------------------------------------------------------------------
# Helpers exposed to app.py at import time
# ---------------------------------------------------------------------------


def slugify_metric_name(name):
    """PWC metric names ('Top 1 Accuracy', 'BLEU-4') → BH global_name
    (snake_case identifier). Used so the BH metric column matches the
    PWC column for cross-source comparison."""
    s = (name or '').strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s-]+', '_', s).strip('_')
    if not s or not re.match(r'^[a-z_]', s):
        s = 'metric_' + s
    return s
