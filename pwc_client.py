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


_MAX_RECURSION_DEPTH = 0  # 0 = top-level only. Subtasks add 10x decode + insert cost
                          # for niche benchmarks that aren't bootstrap-relevant.


def _walk_task_into(cur, t, dataset_ids, counters=None, depth=0):
    """Insert benchmarks for one task entry + its (limited) subtasks.

    The PWC archive's subtask tree is deep — some top-level tasks
    have 100+ nested subtasks each with their own datasets, and the
    full recursion was so slow it looked like a hang. We cap at
    _MAX_RECURSION_DEPTH so the build finishes in minutes for the
    bootstrap use case. The deeper niche benchmarks can be added
    later by raising the cap or by indexing them on demand.
    """
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
        for r in rows[:50]:
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
        if counters is not None:
            counters['evals'] = counters.get('evals', 0) + 1
            cb = counters.get('progress_cb')
            if cb is not None and counters['evals'] % 100 == 0:
                cb(
                    f"Walking shard {counters.get('shard_idx', '?')}/"
                    f"{counters.get('n_shards', '?')}: "
                    f"{counters['evals']} benchmarks indexed, "
                    f"{len(dataset_ids)} unique datasets"
                )
    if depth < _MAX_RECURSION_DEPTH:
        for sub in (t.get('subtasks') or []):
            _walk_task_into(cur, sub, dataset_ids,
                            counters=counters, depth=depth + 1)


def _build_index_into(idx_path, snap_dir, progress_cb=None):
    """Walk every parquet shard row-by-row in batches, write to a
    fresh SQLite at idx_path. Bounded peak memory: one row group +
    SQLite's commit buffer.

    progress_cb: optional `(msg) -> None` that the build invokes at
    every significant step so the web tier can show progress. The
    Celery wrapper passes pwc_client.update_progress; tests pass a
    list-append so they can assert on the sequence of stages.
    """
    import pyarrow.parquet as pq
    if progress_cb is None:
        progress_cb = lambda _msg: None
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
        # Aggressive write tuning for the one-shot build. journal_mode=MEMORY
        # since we don't care about crash-recovery for a build artifact —
        # if it dies mid-build we re-run from scratch anyway. Larger cache
        # so the schema/indexes stay in memory; synchronous=OFF so each
        # insert doesn't fsync the volume.
        cur.execute("PRAGMA journal_mode=MEMORY")
        cur.execute("PRAGMA synchronous=OFF")
        cur.execute("PRAGMA cache_size=-65536")  # 64 MB
        cur.execute("PRAGMA temp_store=MEMORY")
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
        # batch_size=1 was too slow (per-batch decode tax). Bulk-read
        # was too hungry (full shard in Python = OOM on a 2 GB box).
        # Middle ground: iter_batches(batch_size=50) with column
        # projection. ~1/20th decode passes vs batch_size=1, and only
        # 50 rows worth of nested Python at any time (~50 MB peak).
        import gc as _gc
        # Column projection: at depth=0 we don't recurse into subtasks,
        # so dropping it from the projection skips ~half the decode +
        # to_pylist work (the subtask tree is the bulky nested column).
        WANTED_COLUMNS = ['task', 'description', 'datasets']
        if _MAX_RECURSION_DEPTH > 0:
            WANTED_COLUMNS.append('subtasks')
        BATCH_SIZE = 50
        n_shards = len(parquet_files)
        counters = {'evals': 0, 'shard_idx': 0, 'n_shards': n_shards,
                    'progress_cb': progress_cb}
        for shard_idx, pf in enumerate(parquet_files, 1):
            counters['shard_idx'] = shard_idx
            pq_file = pq.ParquetFile(pf)
            available = set(pq_file.schema_arrow.names)
            cols = [c for c in WANTED_COLUMNS if c in available]
            n_rows_total = pq_file.metadata.num_rows or 0
            progress_cb(
                f"Walking shard {shard_idx}/{n_shards}: "
                f"{n_rows_total} top-level tasks, "
                f"{counters['evals']} benchmarks indexed cumulative"
            )
            tasks_in_shard = 0
            for batch in pq_file.iter_batches(batch_size=BATCH_SIZE,
                                               columns=cols):
                rows = batch.to_pylist()
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    _walk_task_into(cur, row, dataset_ids, counters=counters)
                    tasks_in_shard += 1
                conn.commit()
                # Heartbeat per batch so the user sees progress even
                # when the recursive walker doesn't hit its own
                # 100-evals threshold (some tasks have few subtasks).
                progress_cb(
                    f"Walking shard {shard_idx}/{n_shards}: "
                    f"{tasks_in_shard}/{n_rows_total} tasks, "
                    f"{counters['evals']} benchmarks, "
                    f"{len(dataset_ids)} datasets"
                )
                del rows, batch
                _gc.collect()
            progress_cb(
                f"Shard {shard_idx}/{n_shards} done — "
                f"{counters['evals']} benchmarks, {len(dataset_ids)} datasets"
            )
        progress_cb("Finalizing index (ANALYZE)…")
        cur.execute("ANALYZE")
        conn.commit()
        progress_cb(
            f"Done — {len(dataset_ids)} datasets, "
            f"{counters['evals']} benchmarks indexed"
        )
    finally:
        conn.close()
    os.replace(tmp_path, idx_path)


_BUILD_MARKER = 'index.building.tmp'
_BUILD_PROGRESS = 'index.progress.txt'
_BUILD_ERROR = 'index.error.txt'
_BUILD_STALE_AFTER_SECONDS = 30 * 60  # 30 min — longer than any realistic build


def _build_marker_path():
    return os.path.join(_cache_dir(), _BUILD_MARKER)


def _build_progress_path():
    return os.path.join(_cache_dir(), _BUILD_PROGRESS)


def _build_error_path():
    return os.path.join(_cache_dir(), _BUILD_ERROR)


def _build_marker_is_fresh():
    """Marker file's mtime is within the freshness window. A crashed
    Celery worker can leave a stale marker; treat anything older than
    the build's normal upper bound as no longer building."""
    p = _build_marker_path()
    try:
        import time
        return (time.time() - os.path.getmtime(p)) < _BUILD_STALE_AFTER_SECONDS
    except OSError:
        return False


def index_status():
    """Lightweight stat-only status — never touches parquet/sqlite.
    Returns one of {ready, building, error, absent}. Stale markers
    (older than _BUILD_STALE_AFTER_SECONDS) are treated as no longer
    building, so a crashed worker doesn't leave the UI stuck.
    """
    cache = _cache_dir()
    if os.path.exists(_index_path()):
        return 'ready'
    if os.path.exists(_build_marker_path()) and _build_marker_is_fresh():
        return 'building'
    if os.path.exists(os.path.join(cache, _BUILD_ERROR)):
        return 'error'
    return 'absent'


def index_error_message():
    """Read the last failure message left by the Celery build task."""
    try:
        with open(_build_error_path()) as f:
            return f.read().strip()
    except OSError:
        return ''


def index_progress_message():
    """Latest progress line written by the build task (if any).
    Empty string when nothing has been reported yet — used by the
    /admin/pwc/import status banner so admins can see what stage
    the build is on."""
    try:
        with open(_build_progress_path()) as f:
            return f.read().strip()[:300]
    except OSError:
        return ''


def begin_build_marker():
    """Mark a build as in-progress from the web tier, BEFORE enqueueing
    Celery. Race-window with celery picking up the task otherwise lets
    the redirect render 'absent' while a build is actually queued.

    Returns True when a fresh marker was created (caller should enqueue
    the task), False when one was already there (caller should treat
    this as 'already building' and skip enqueueing)."""
    marker = _build_marker_path()
    cache = _cache_dir()
    if os.path.exists(_index_path()):
        return False
    if os.path.exists(marker) and _build_marker_is_fresh():
        return False
    # Stale or absent: (re)create.
    try:
        os.remove(_build_error_path())
    except OSError:
        pass
    try:
        os.remove(_build_progress_path())
    except OSError:
        pass
    os.makedirs(cache, exist_ok=True)
    with open(marker, 'w') as f:
        f.write('queued')
    return True


def update_progress(msg):
    """Build task writes its current stage here. Best-effort — failures
    are silent (progress reporting must never block the build)."""
    try:
        with open(_build_progress_path(), 'w') as f:
            f.write(msg)
    except OSError:
        pass
    # Touch the marker so the staleness check stays happy on long builds.
    try:
        os.utime(_build_marker_path(), None)
    except OSError:
        pass


def clear_build_marker():
    """Called by the build task on success or failure to free the marker
    + progress files. The error.txt is left in place when the build
    failed so index_status() can return 'error'."""
    for p in (_build_marker_path(), _build_progress_path()):
        try:
            os.remove(p)
        except OSError:
            pass


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
    """Search PWC archive by dataset name (case-insensitive substring).
    Returns each row with `n_benchmarks` + `n_results` counts so the
    admin search UI can rank/badge by coverage — the archive's tail
    has many datasets with 0-1 result rows each, easy to mistake for
    rich content if all you see is the name.

    Empty query returns the top-N most-populated datasets so the admin
    can browse without having to know what they're looking for.
    """
    q_lc = (q or '').strip().lower()
    try:
        conn = _conn()
    except Exception as e:
        raise PwcError(f"Failed to load PWC archive: {e}") from e
    try:
        cur = conn.cursor()
        # Empty query: skip the LIKE so the planner uses the GROUP BY
        # path against the full table. Same shape, just no name filter.
        if q_lc:
            cur.execute(
                "SELECT d.id, d.name, d.description, d.links_json, "
                "       COUNT(e.id) AS n_benchmarks, "
                "       COALESCE(SUM(json_array_length(e.results_json)), 0) AS n_results "
                "FROM pwc_dataset d "
                "LEFT JOIN pwc_evaluation e ON e.dataset_id = d.id "
                "WHERE d.name_lc LIKE ? "
                "GROUP BY d.id "
                "ORDER BY n_results DESC, d.name "
                "LIMIT ?",
                (f"%{q_lc}%", limit),
            )
        else:
            cur.execute(
                "SELECT d.id, d.name, d.description, d.links_json, "
                "       COUNT(e.id) AS n_benchmarks, "
                "       COALESCE(SUM(json_array_length(e.results_json)), 0) AS n_results "
                "FROM pwc_dataset d "
                "LEFT JOIN pwc_evaluation e ON e.dataset_id = d.id "
                "GROUP BY d.id "
                "ORDER BY n_results DESC, d.name "
                "LIMIT ?",
                (limit,),
            )
        out = []
        for ds_id, name, desc, links_json, n_benchmarks, n_results in cur.fetchall():
            out.append({
                'id': ds_id,
                'name': name,
                'full_name': name,
                'description': (desc or '')[:400],
                'hf_repo': _hf_repo_from_links(links_json),
                'n_benchmarks': int(n_benchmarks or 0),
                'n_results': int(n_results or 0),
                'url': None,
                'huggingface_url': None,
            })
        return out
    finally:
        conn.close()


def list_evaluations_for_dataset(dataset_id):
    """Benchmarks tracked on a single dataset, sorted by result-row
    count descending so the admin sees rich benchmarks first. PWC's
    tail of niche benchmarks has 0-1 rows each — useful to flag, but
    not what you want to import for bootstrapping.
    """
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, task, description, "
            "       json_array_length(results_json) AS n_results, "
            "       json_array_length(metrics_json) AS n_metrics "
            "FROM pwc_evaluation "
            "WHERE dataset_id = ? "
            "ORDER BY n_results DESC, id",
            (dataset_id,),
        )
        return [
            {'id': eid, 'task': task, 'description': (desc or '')[:400],
             'n_results': int(n_results or 0),
             'n_metrics': int(n_metrics or 0)}
            for eid, task, desc, n_results, n_metrics in cur.fetchall()
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
        # Heuristic: tag a metric as "lower-is-better" when its name
        # contains an error/loss-like token. Word boundaries keep e.g.
        # `\brms\b` from matching inside `rmse` (which is matched on
        # its own line). User-reported: NYU Depth V2 imports tagged
        # `RMS` as higher-is-better because the original list only had
        # `rmse`. Depth-estimation also commonly reports absrel /
        # sqrel / log10 / silog as errors; LM tasks use perplexity;
        # trajectory pred uses ade/fde; generative work uses fid/lpips
        # — all unambiguously lower-better.
        is_loss = bool(re.search(
            r'\b('
            r'loss|error|'
            r'mae|rmse|mse|rms|nmse|mape|'
            r'absrel|sqrel|log10|silog|'
            r'wer|cer|'
            r'fid|lpips|chamfer|emd|'
            r'ade|fde|'
            r'perplexity|ppl'
            r')\b',
            str(m), re.IGNORECASE,
        ))
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


_HF_SUGGEST_CACHE = {}  # pwc_name → (best_repo_id, [alt_repo_ids])


def suggest_hf_repo(pwc_name):
    """Best-effort HF Hub lookup for a PWC dataset name. Returns
    (best_repo, alternatives) where best_repo is the most-downloaded
    matching dataset (or None) and alternatives is up to 4 other
    plausible candidates the admin might want to pick instead.

    Heuristic: search HF Hub for the PWC name, then for a normalized
    version (lowercase, hyphens for spaces). Filter results to ones
    whose name contains the search term as a token. Rank by HF
    downloads — the canonical mirror is almost always the most
    downloaded match.

    Cached per-process so a search-page refresh doesn't re-hit HF
    for every row. Keys are case-folded names.
    """
    key = (pwc_name or '').strip().lower()
    if not key:
        return None, []
    if key in _HF_SUGGEST_CACHE:
        return _HF_SUGGEST_CACHE[key]
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try exact name first, then normalized.
        seen = {}  # repo_id → downloads (for dedupe + ranking)
        for term in (pwc_name, pwc_name.lower().replace(' ', '-')):
            try:
                hits = list(api.list_datasets(search=term, limit=20))
            except Exception:
                hits = []
            for h in hits:
                rid = getattr(h, 'id', None)
                if not rid:
                    continue
                # Filter: the search term must appear in the repo id
                # as a delimited token. HF's `search` is permissive
                # and otherwise returns weakly-related results.
                rid_lc = rid.lower()
                norm = key.replace(' ', '-')
                if (key not in rid_lc and norm not in rid_lc
                        and key.replace(' ', '_') not in rid_lc):
                    continue
                downloads = getattr(h, 'downloads', 0) or 0
                # Keep the highest download count if the same repo
                # appeared in multiple searches.
                seen[rid] = max(seen.get(rid, 0), downloads)
        if not seen:
            _HF_SUGGEST_CACHE[key] = (None, [])
            return None, []
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])
        best = ranked[0][0]
        alts = [rid for rid, _ in ranked[1:5]]
        _HF_SUGGEST_CACHE[key] = (best, alts)
        return best, alts
    except Exception as e:
        # Importing huggingface_hub or hitting the network can fail —
        # fall back gracefully to "no suggestion."
        print(f"suggest_hf_repo({pwc_name!r}) failed: {e}")
        _HF_SUGGEST_CACHE[key] = (None, [])
        return None, []


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
