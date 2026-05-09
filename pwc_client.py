"""Papers With Code data client.

The original paperswithcode.com REST API is gone — the entire domain
302s to huggingface.co/papers/trending after HuggingFace's acquisition.
The historical data lives at `pwc-archive/evaluation-tables` on HF
Datasets, with the same nested sota-extractor shape:

    [{
        task, categories, description, source_link, subtasks, synonyms,
        datasets: [{
            dataset, dataset_links, description, subdatasets,
            sota: {metrics: [...], rows: [{model_name, paper_url, metrics: {name: value}, ...}]}
        }]
    }, ...]

The archive is frozen at acquisition (~early 2024) so live-tracking is
out, but for bootstrapping a fresh BenchHub it's exactly the right
shape: one HF dataset → many benchmarks → many result rows.

This module downloads the parquet shards once via `huggingface_hub`,
caches them under `BENCHHUB_DATA_DIR/_cache/pwc_archive/`, and exposes
`search_datasets`, `list_evaluations_for_dataset`, `get_evaluation` —
the same interface app.py was written against, so the import flow's
routes don't need to change.
"""
import json
import os
import re
import threading
from urllib.parse import urlparse

# In-process index. Built lazily on first use; survives for the
# gunicorn worker's lifetime.
_INDEX_LOCK = threading.Lock()
_INDEX = None  # dict: {datasets: [{...}], by_lc_name: {...}, by_id: {...}, evals_by_id: {...}}

ARCHIVE_REPO = "pwc-archive/evaluation-tables"


class PwcError(Exception):
    """Raised when archive load / lookup fails in a way the caller
    should surface to the admin user."""


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------


def _cache_dir():
    """Where the downloaded parquet shards live. Lazy import of the
    Flask config so this module is testable in isolation."""
    base = os.environ.get('BENCHHUB_DATA_DIR') or os.path.expanduser('~/.dtofbenchmarking')
    path = os.path.join(base, '_cache', 'pwc_archive')
    os.makedirs(path, exist_ok=True)
    return path


def _download_archive():
    """Pull the pwc-archive/evaluation-tables shards via
    huggingface_hub. Returns the local snapshot directory.
    Re-uses the existing snapshot when present (snapshot_download
    is content-addressed)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=ARCHIVE_REPO,
        repo_type='dataset',
        local_dir=_cache_dir(),
        local_dir_use_symlinks=False,
    )


def _iter_archive_rows(snap_dir):
    """Yield decoded row dicts from every parquet shard. Defensive
    parsing: rows missing `task` or `datasets` get skipped instead of
    raising."""
    import pyarrow.parquet as pq
    parquet_files = []
    for root, _, files in os.walk(snap_dir):
        for f in files:
            if f.endswith('.parquet'):
                parquet_files.append(os.path.join(root, f))
    if not parquet_files:
        raise PwcError(f"No parquet shards in {snap_dir}")
    for pf in sorted(parquet_files):
        table = pq.read_table(pf)
        for row in table.to_pylist():
            if not isinstance(row, dict):
                continue
            yield row


def _build_index():
    """Walk every parquet row, flatten task→datasets→benchmarks into
    a flat dataset list + an evaluations table keyed by synthetic int
    ids (PWC's archive doesn't carry stable ids, so we mint our own
    deterministic from the dataset name + task)."""
    snap_dir = _download_archive()
    datasets = {}      # name_lc → dataset dict (deduplicated; same dataset shows up under multiple tasks)
    evals = {}         # synthetic_id → eval dict
    next_dataset_id = 1
    next_eval_id = 1

    def _walk_task(t, dataset_assignments):
        """Recurse into a task entry + its subtasks. Each `datasets`
        entry on a task represents a (task, dataset) benchmark."""
        if not isinstance(t, dict):
            return
        task_name = (t.get('task') or '').strip()
        for ds in (t.get('datasets') or []):
            if not isinstance(ds, dict):
                continue
            ds_name = (ds.get('dataset') or '').strip()
            if not ds_name:
                continue
            ds_name_lc = ds_name.lower()
            existing = datasets.get(ds_name_lc)
            if existing is None:
                nonlocal_ids = next_dataset_id
                existing = {
                    'id': nonlocal_ids,
                    'name': ds_name,
                    'name_lc': ds_name_lc,
                    'description': (ds.get('description') or '').strip(),
                    'links': ds.get('dataset_links') or [],
                }
                datasets[ds_name_lc] = existing
            else:
                nonlocal_ids = existing['id']
            sota = ds.get('sota') or {}
            metrics = sota.get('metrics') or []
            rows = sota.get('rows') or []
            if not (metrics or rows):
                continue
            ev = {
                'id': len(evals) + 1,
                'task': task_name,
                'dataset_id': existing['id'],
                'dataset_name': ds_name,
                'description': (ds.get('description') or '').strip(),
                'metrics': [str(m) for m in metrics if m is not None],
                'rows': rows,
            }
            evals[ev['id']] = ev
            dataset_assignments.setdefault(existing['id'], []).append(ev['id'])
        for sub in (t.get('subtasks') or []):
            _walk_task(sub, dataset_assignments)

    # First pass: assign synthetic ids to datasets as we discover them.
    # Have to do it in two passes because next_dataset_id is closed-over
    # and Python forbids reassigning closure scalars from a nested function.
    flat_dataset_id = 0
    dataset_assignments = {}
    for row in _iter_archive_rows(_download_archive()):
        # pre-assign ids so the closure can read them
        for ds in (row.get('datasets') or []):
            ds_name = (ds.get('dataset') or '').strip()
            if not ds_name:
                continue
            ds_lc = ds_name.lower()
            if ds_lc not in datasets:
                flat_dataset_id += 1
                datasets[ds_lc] = {
                    'id': flat_dataset_id,
                    'name': ds_name,
                    'name_lc': ds_lc,
                    'description': (ds.get('description') or '').strip(),
                    'links': ds.get('dataset_links') or [],
                }
        _walk_task(row, dataset_assignments)

    return {
        'datasets': sorted(datasets.values(), key=lambda d: d['name'].lower()),
        'by_lc_name': datasets,
        'by_id': {d['id']: d for d in datasets.values()},
        'evals_by_id': evals,
        'evals_by_dataset_id': dataset_assignments,
    }


def _index():
    """Return the lazily-built archive index. Single call locks
    everything else out so we don't kick off two parallel downloads
    on a cold start."""
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = _build_index()
    return _INDEX


# ---------------------------------------------------------------------------
# HF repo discovery (best-effort guess from dataset_links)
# ---------------------------------------------------------------------------


def _hf_repo_from_url(url):
    """Extract `owner/repo` from a huggingface.co dataset URL. Returns
    None for non-HF URLs."""
    if not url:
        return None
    try:
        u = urlparse(url)
    except ValueError:
        return None
    if u.netloc.lower() not in ('huggingface.co', 'www.huggingface.co'):
        return None
    parts = [p for p in u.path.split('/') if p]
    if not parts or parts[0] != 'datasets':
        return None
    if len(parts) == 2:
        return parts[1]
    if len(parts) >= 3:
        return f"{parts[1]}/{parts[2]}"
    return None


def _hf_repo_for_dataset(ds_entry):
    """The archive's `dataset_links` is sometimes a list of dicts with
    `url` keys, sometimes a list of plain strings. Walk it looking for
    the first huggingface.co/datasets/... link."""
    for link in (ds_entry.get('links') or []):
        if isinstance(link, dict):
            url = link.get('url') or ''
        else:
            url = str(link or '')
        repo = _hf_repo_from_url(url)
        if repo:
            return repo
    return None


# ---------------------------------------------------------------------------
# Public API (matches the shape app.py was written against)
# ---------------------------------------------------------------------------


def search_datasets(q, *, limit=30):
    """Search PWC archive by dataset name (case-insensitive substring).
    Returns dicts with id, name, description, hf_repo (None unless an
    HF link was discoverable in the archive's dataset_links), url.

    Admins can override the hf_repo at preview time when the archive
    doesn't carry one — most rows don't, since the archive predates
    the HF acquisition's cross-linking.
    """
    q_lc = (q or '').strip().lower()
    if not q_lc:
        return []
    try:
        idx = _index()
    except Exception as e:
        raise PwcError(f"Failed to load PWC archive: {e}") from e
    out = []
    for ds in idx['datasets']:
        if q_lc not in ds['name_lc']:
            continue
        out.append({
            'id': ds['id'],
            'name': ds['name'],
            'full_name': ds['name'],
            'description': ds['description'][:400],
            'hf_repo': _hf_repo_for_dataset(ds),
            'url': None,  # no PWC URL anymore
            'huggingface_url': None,
        })
        if len(out) >= limit:
            break
    return out


def list_evaluations_for_dataset(dataset_id):
    """All benchmarks (task × dataset entries) tracked for one
    dataset id."""
    idx = _index()
    eval_ids = idx['evals_by_dataset_id'].get(dataset_id) or []
    out = []
    for eid in eval_ids:
        ev = idx['evals_by_id'].get(eid)
        if ev is None:
            continue
        out.append({
            'id': eid,
            'task': ev['task'],
            'description': ev['description'][:400],
        })
    return out


def get_evaluation(evaluation_id):
    """Single benchmark with metrics + result rows. Mirrors the old
    PWC API helper so the route doesn't care about the source change."""
    idx = _index()
    ev = idx['evals_by_id'].get(evaluation_id)
    if ev is None:
        raise PwcError(f"Unknown evaluation id {evaluation_id}")
    metrics = []
    for m in ev['metrics']:
        is_loss = bool(re.search(r'\b(loss|error|mae|rmse|mse|wer|cer)\b', m, re.IGNORECASE))
        metrics.append({
            'id': None,
            'name': m,
            'description': '',
            'sort_direction': 'lower_is_better' if is_loss else 'higher_is_better',
        })
    results = []
    for r in ev['rows']:
        if not isinstance(r, dict):
            continue
        # The archive uses model_name + paper_title interchangeably across
        # rows; read both and prefer the more specific.
        results.append({
            'id': None,
            'methodology': r.get('model_name') or '',
            'paper_title': r.get('paper_title') or r.get('paper') or '',
            'paper_url': r.get('paper_url') or r.get('paper') or '',
            'external_source_url': None,
            'metrics': r.get('metrics') or {},
            'evaluated_on': r.get('paper_date'),
        })
    return {
        'id': evaluation_id,
        'task': ev['task'],
        'dataset': idx['by_id'].get(ev['dataset_id'], {}).get('name'),
        'description': ev['description'],
        'metrics': metrics,
        'results': results,
    }


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


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------


def _reset_index_for_tests():
    """Drop the cached index so a test fixture can re-seed."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None


def _set_index_for_tests(index_dict):
    """Inject a hand-built index for unit tests so they don't have to
    download the actual ~hundreds-of-MB archive."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = index_dict
