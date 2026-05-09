"""Papers With Code data client (static archive).

The original paperswithcode.com REST API is gone — the entire domain
302s to huggingface.co/papers/trending after HF's acquisition. The
historical data lives at `pwc-archive/evaluation-tables` on HF
Datasets in the same nested sota-extractor shape.

This module downloads the parquet shards once, then streams them
row-by-row into a local SQLite index — keeps peak memory bounded
even on small fly machines (the previous in-memory dict approach
SIGKILL'd the worker). Subsequent reads are sqlite queries.

Public surface (matches the shape app.py was written against):
- search_datasets(q) → list[dict]
- list_evaluations_for_dataset(dataset_id) → list[dict]
- get_evaluation(evaluation_id) → dict (with metrics + results)
- slugify_metric_name(name) → str
"""
import json
import os
import re
import sqlite3
import threading
from urllib.parse import urlparse

ARCHIVE_REPO = "pwc-archive/evaluation-tables"
SCHEMA_VERSION = 1
_BUILD_LOCK = threading.Lock()


class PwcError(Exception):
    """Raised when archive load / lookup fails in a way the caller
    should surface to the admin user."""


# ---------------------------------------------------------------------------
# Cache + index location
# ---------------------------------------------------------------------------


def _cache_dir():
    base = os.environ.get('BENCHHUB_DATA_DIR') or os.path.expanduser('~/.dtofbenchmarking')
    path = os.path.join(base, '_cache', 'pwc_archive')
    os.makedirs(path, exist_ok=True)
    return path


def _index_path():
    return os.path.join(_cache_dir(), f'index.v{SCHEMA_VERSION}.sqlite')


def _snapshot_dir():
    return os.path.join(_cache_dir(), 'snapshot')


# ---------------------------------------------------------------------------
# Index build (streaming parquet → SQLite)
# ---------------------------------------------------------------------------


def _ensure_snapshot():
    """Pull the parquet shards once. snapshot_download is content-
    addressed so re-runs are cheap when the snapshot is already on
    disk."""
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=ARCHIVE_REPO,
        repo_type='dataset',
        local_dir=_snapshot_dir(),
        local_dir_use_symlinks=False,
    )


def _walk_task_into(cur, t, dataset_ids):
    """Insert benchmarks for one task entry + its subtasks. `cur` is
    a sqlite cursor; `dataset_ids` is a name→id map mutated in-place
    so the same dataset (showing up under multiple tasks) gets one
    row only."""
    if not isinstance(t, dict):
        return
    task_name = (t.get('task') or '').strip()
    for ds in (t.get('datasets') or []):
        if not isinstance(ds, dict):
            continue
        ds_name = (ds.get('dataset') or '').strip()
        if not ds_name:
            continue
        ds_lc = ds_name.lower()
        ds_id = dataset_ids.get(ds_lc)
        if ds_id is None:
            cur.execute(
                "INSERT INTO pwc_dataset (name, name_lc, description, links_json) "
                "VALUES (?, ?, ?, ?)",
                (ds_name, ds_lc,
                 (ds.get('description') or '').strip()[:2000],
                 json.dumps(ds.get('dataset_links') or [])[:8000]),
            )
            ds_id = cur.lastrowid
            dataset_ids[ds_lc] = ds_id
        sota = ds.get('sota') or {}
        metrics = [str(m) for m in (sota.get('metrics') or []) if m is not None]
        rows = sota.get('rows') or []
        if not (metrics or rows):
            continue
        # Cap result rows + only keep the columns we actually render.
        # Some PWC benchmarks (LLM leaderboards) carry hundreds of
        # rows × MB-scale per-result blobs we don't need.
        slim_rows = []
        for r in rows[:200]:
            if not isinstance(r, dict):
                continue
            slim_rows.append({
                'model_name': (r.get('model_name') or '')[:200],
                'paper_title': (r.get('paper_title') or '')[:300],
                'paper_url': (r.get('paper_url') or r.get('paper') or '')[:500],
                'paper_date': str(r.get('paper_date') or '')[:30],
                'metrics': {str(k): str(v)[:60] for k, v in (r.get('metrics') or {}).items()
                            if v is not None and v != ''},
            })
        cur.execute(
            "INSERT INTO pwc_evaluation (dataset_id, task, description, metrics_json, results_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (ds_id, task_name,
             (ds.get('description') or '').strip()[:2000],
             json.dumps(metrics),
             json.dumps(slim_rows)),
        )
    for sub in (t.get('subtasks') or []):
        _walk_task_into(cur, sub, dataset_ids)


def _build_index_into(idx_path, snap_dir):
    """Walk every parquet shard row-by-row in batches, write to a
    fresh SQLite at idx_path. Bounded peak memory: one row group +
    SQLite's commit buffer."""
    import pyarrow.parquet as pq
    parquet_files = []
    for root, _, files in os.walk(snap_dir):
        for f in files:
            if f.endswith('.parquet'):
                parquet_files.append(os.path.join(root, f))
    if not parquet_files:
        raise PwcError(f"No parquet shards in {snap_dir}")
    parquet_files.sort()

    tmp_path = idx_path + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    conn = sqlite3.connect(tmp_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE pwc_dataset ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  name_lc TEXT NOT NULL,"
            "  description TEXT,"
            "  links_json TEXT"
            ")"
        )
        cur.execute("CREATE INDEX ix_pwc_dataset_name_lc ON pwc_dataset(name_lc)")
        cur.execute(
            "CREATE TABLE pwc_evaluation ("
            "  id INTEGER PRIMARY KEY,"
            "  dataset_id INTEGER NOT NULL,"
            "  task TEXT,"
            "  description TEXT,"
            "  metrics_json TEXT,"
            "  results_json TEXT"
            ")"
        )
        cur.execute("CREATE INDEX ix_pwc_evaluation_ds ON pwc_evaluation(dataset_id)")
        conn.commit()

        dataset_ids = {}
        # batch_size=1 keeps peak memory bounded — each row in this
        # parquet is a whole task with all its datasets + SOTA tables,
        # which can be tens of MB on its own.
        import gc as _gc
        for pf in parquet_files:
            pq_file = pq.ParquetFile(pf)
            for batch in pq_file.iter_batches(batch_size=1):
                rows = batch.to_pylist()
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    _walk_task_into(cur, row, dataset_ids)
                conn.commit()
                # Drop the batch + collect early so the next iteration
                # doesn't sit on top of the previous batch's reference graph.
                del rows, batch
                _gc.collect()
        cur.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp_path, idx_path)


def index_status():
    """Lightweight check usable from the web tier without ever touching
    the parquet files. Returns one of:
      'ready'    — SQLite index is on disk; queries are fast.
      'building' — the Celery `build_pwc_index` task is in progress.
      'error'    — a previous build failed; the message is readable
                   from `index_error_message()`.
      'absent'   — no index, no build in progress.
    """
    cache = _cache_dir()
    if os.path.exists(_index_path()):
        return 'ready'
    if os.path.exists(os.path.join(cache, 'index.building.tmp')):
        return 'building'
    if os.path.exists(os.path.join(cache, 'index.error.txt')):
        return 'error'
    return 'absent'


def index_error_message():
    """Read the last failure message left by the Celery build task."""
    try:
        with open(os.path.join(_cache_dir(), 'index.error.txt')) as f:
            return f.read().strip()
    except OSError:
        return ''


def _conn():
    """Read-only sqlite connection to the index. Raises PwcError when
    the index isn't ready — the web tier should call index_status()
    first to render an appropriate UX, NOT trigger a build inline."""
    idx_path = _index_path()
    if not os.path.exists(idx_path):
        raise PwcError(
            "PWC index not built yet. Trigger the build task and try again."
        )
    return sqlite3.connect(idx_path)


# ---------------------------------------------------------------------------
# HF repo discovery (best-effort guess from dataset_links)
# ---------------------------------------------------------------------------


def _hf_repo_from_url(url):
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


def _hf_repo_from_links(links_json):
    """Walk the archive's `dataset_links` (sometimes dicts, sometimes
    strings) for the first huggingface.co/datasets/... link."""
    try:
        links = json.loads(links_json or '[]')
    except (TypeError, ValueError):
        return None
    for link in (links or []):
        if isinstance(link, dict):
            url = link.get('url') or ''
        else:
            url = str(link or '')
        repo = _hf_repo_from_url(url)
        if repo:
            return repo
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_datasets(q, *, limit=30):
    """Search PWC archive by dataset name (case-insensitive substring)."""
    q_lc = (q or '').strip().lower()
    if not q_lc:
        return []
    try:
        conn = _conn()
    except Exception as e:
        raise PwcError(f"Failed to load PWC archive: {e}") from e
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, description, links_json FROM pwc_dataset "
            "WHERE name_lc LIKE ? ORDER BY name LIMIT ?",
            (f"%{q_lc}%", limit),
        )
        out = []
        for ds_id, name, desc, links_json in cur.fetchall():
            out.append({
                'id': ds_id,
                'name': name,
                'full_name': name,
                'description': (desc or '')[:400],
                'hf_repo': _hf_repo_from_links(links_json),
                'url': None,
                'huggingface_url': None,
            })
        return out
    finally:
        conn.close()


def list_evaluations_for_dataset(dataset_id):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, task, description FROM pwc_evaluation "
            "WHERE dataset_id = ? ORDER BY id",
            (dataset_id,),
        )
        return [
            {'id': eid, 'task': task, 'description': (desc or '')[:400]}
            for eid, task, desc in cur.fetchall()
        ]
    finally:
        conn.close()


def get_evaluation(evaluation_id):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT e.id, e.task, e.description, e.metrics_json, e.results_json, d.name "
            "FROM pwc_evaluation e JOIN pwc_dataset d ON d.id = e.dataset_id "
            "WHERE e.id = ?",
            (evaluation_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise PwcError(f"Unknown evaluation id {evaluation_id}")
    eid, task, desc, metrics_json, results_json, ds_name = row
    try:
        metric_names = json.loads(metrics_json or '[]')
    except (TypeError, ValueError):
        metric_names = []
    try:
        rows = json.loads(results_json or '[]')
    except (TypeError, ValueError):
        rows = []
    metrics = []
    for m in metric_names:
        is_loss = bool(re.search(r'\b(loss|error|mae|rmse|mse|wer|cer)\b', str(m), re.IGNORECASE))
        metrics.append({
            'id': None, 'name': m, 'description': '',
            'sort_direction': 'lower_is_better' if is_loss else 'higher_is_better',
        })
    results = []
    for r in rows:
        if not isinstance(r, dict):
            continue
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
        'id': eid, 'task': task,
        'dataset': ds_name, 'description': desc,
        'metrics': metrics, 'results': results,
    }


def slugify_metric_name(name):
    s = (name or '').strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s-]+', '_', s).strip('_')
    if not s or not re.match(r'^[a-z_]', s):
        s = 'metric_' + s
    return s


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------


def _set_index_path_for_tests(path):
    """Override _index_path so unit tests can point at a hand-built
    SQLite without touching the cache dir."""
    global _index_path
    _index_path = lambda: path  # noqa: E731


def _reset_index_path_for_tests():
    """Reset the override; called by the test fixture's teardown."""
    global _index_path
    _index_path = _index_path_default


_index_path_default = _index_path
