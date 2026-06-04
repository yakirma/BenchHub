"""Thin wrappers around HuggingFace's public /api/datasets endpoint.

Two access patterns:

  - `search_datasets(query, limit=10)` — prefix / full-text search used
    by the import page's autocomplete dropdown. Live, no caching.
  - `trending_by_domain(limit_per_domain=5)` — top-downloaded datasets
    grouped by ML domain (Vision / NLP / Audio / Tabular). Memoised
    with a 1-hour TTL so we don't hammer HF on every page load.

No auth required — both endpoints hit the unauthenticated HF Hub API.
Gated / private repos won't show up; the admin can still paste a
gated repo's id by hand and proceed through the existing per-user
`hf_token` flow.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request


_HF_API_BASE = "https://huggingface.co/api/datasets"
_REQUEST_TIMEOUT = 10


def _fetch_json(url: str) -> list[dict]:
    """GET a JSON-array endpoint; return [] on any failure so callers
    don't have to wrap with try/except. The HF Hub API is best-effort
    here — slow or down means an empty dropdown, not a 5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": "benchhub/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _normalize(d: dict) -> dict:
    """Project the HF API's verbose dataset record down to the fields
    the suggestion dropdown actually shows."""
    return {
        "id": d.get("id") or "",
        "downloads": int(d.get("downloads") or 0),
        "likes": int(d.get("likes") or 0),
        # HF descriptions are sometimes a multi-line dataset-card dump
        # — truncate so the dropdown doesn't blow up on JSON.
        "description": ((d.get("description") or "")
                        .strip().replace("\n", " "))[:200],
        "gated": bool(d.get("gated") or False),
    }


def search_datasets(query: str, *, limit: int = 10) -> list[dict]:
    """Free-text dataset search. Returns a list (possibly empty) of
    normalised `{id, downloads, likes, description, gated}` dicts.

    Empty / whitespace-only queries short-circuit to `[]` so the
    dropdown can stay quiet until the user starts typing."""
    q = (query or "").strip()
    if not q:
        return []
    params = urllib.parse.urlencode({"search": q, "limit": int(limit)})
    return [_normalize(d) for d in _fetch_json(f"{_HF_API_BASE}?{params}")
            if isinstance(d, dict)]


# Map BH-side domain labels → the HF `task_categories:*` filter the
# Hub uses on the public search. Picked to cover the most common
# benchmark families a new admin would want at a glance.
_DOMAIN_FILTERS = {
    "Vision": "task_categories:image-classification",
    "NLP":    "task_categories:text-classification",
    "Audio":  "task_categories:automatic-speech-recognition",
    "Tabular": "task_categories:tabular-classification",
}

# In-memory TTL cache; one entry per domain.
_TRENDING_CACHE: dict[str, tuple[float, list[dict]]] = {}
_TRENDING_TTL_SECONDS = 60 * 60  # 1h is plenty — trending barely shifts hour-to-hour

# Per-repo card-summary cache, so the same dataset shown across several
# preview stages isn't re-fetched from HF each time.
_CARD_CACHE: dict[str, tuple[float, dict]] = {}
_CARD_TTL_SECONDS = 60 * 60


def trending_by_domain(*, limit_per_domain: int = 5) -> dict[str, list[dict]]:
    """Return top-downloaded HF datasets per ML domain, cached.

    Cache key is the domain name; TTL is one hour. Misses fan out one
    HF API call per domain (4 today). Returns `{domain_name: [items]}`
    in `_DOMAIN_FILTERS` insertion order.
    """
    out: dict[str, list[dict]] = {}
    now = time.time()
    limit = max(1, int(limit_per_domain))
    for domain, hf_filter in _DOMAIN_FILTERS.items():
        # Cache key includes the limit so a later larger request ("show
        # more") doesn't get served a smaller cached page.
        key = f"{domain}:{limit}"
        cached = _TRENDING_CACHE.get(key)
        if cached and (now - cached[0]) < _TRENDING_TTL_SECONDS:
            out[domain] = cached[1]
            continue
        params = urllib.parse.urlencode({
            "filter": hf_filter,
            "sort": "downloads",
            "direction": "-1",
            "limit": limit,
        })
        items = [_normalize(d) for d in _fetch_json(f"{_HF_API_BASE}?{params}")
                 if isinstance(d, dict)]
        _TRENDING_CACHE[key] = (now, items)
        out[domain] = items
    return out


def _clear_cache() -> None:
    """Test-only: drop the trending + card caches so monkeypatched
    fetches produce fresh results."""
    _TRENDING_CACHE.clear()
    _CARD_CACHE.clear()


_HF_DATASETS_SERVER = "https://datasets-server.huggingface.co/size"
_HF_DATASETS_SERVER_INFO = "https://datasets-server.huggingface.co/info"


def fetch_class_label_vocabs(repo_id: str, *, timeout: int = 10) -> dict[str, list[str]]:
    """Per-column ClassLabel vocab for an HF dataset.

    Hits HF's `datasets-server` `/info` endpoint, walks each config's
    `features` dict, and collects `{column_name: names}` for every
    feature whose `_type` is `ClassLabel` (which exposes the
    classification class names as a list).

    First config wins on name collisions — the import flow only
    materialises one config anyway. Returns `{}` on network failure,
    non-JSON, or any feature-walk surprise; the preview UI degrades
    cleanly to a blank textarea the admin can fill in by hand.
    """
    if not repo_id:
        return {}
    params = urllib.parse.urlencode({"dataset": repo_id})
    req = urllib.request.Request(
        f"{_HF_DATASETS_SERVER_INFO}?{params}",
        headers={"User-Agent": "benchhub/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return {}
    try:
        doc = json.loads(body)
    except (TypeError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    configs = doc.get("dataset_info") or {}
    if not isinstance(configs, dict):
        return {}
    out: dict[str, list[str]] = {}
    for _cfg_name, cfg in configs.items():
        if not isinstance(cfg, dict):
            continue
        features = cfg.get("features") or {}
        if not isinstance(features, dict):
            continue
        for col, spec in features.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("_type") != "ClassLabel":
                continue
            names = spec.get("names")
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                out.setdefault(col, list(names))
    return out


def fetch_dataset_info(repo_id: str, *, timeout: int = 10) -> dict | None:
    """Fall-back schema source when Croissant isn't available.

    HF's `datasets-server /info` returns the same `features` dict
    HF's own viewer uses — broader coverage than Croissant (it
    indexes anything HF can stream, not just YAML-conformant
    repos). Returns the first config's `{features, splits}`, or
    None on any failure / shape surprise.

    Shape returned:
        {"features": {col: {"_type": "...", ...}},
         "splits": ["train", "test", ...]}

    Callers convert this to the same `CroissantSchema` the preview
    flow consumes — see benchhub.hf_croissant.schema_from_hf_features.
    """
    if not repo_id:
        return None
    params = urllib.parse.urlencode({"dataset": repo_id})
    req = urllib.request.Request(
        f"{_HF_DATASETS_SERVER_INFO}?{params}",
        headers={"User-Agent": "benchhub/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        doc = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    configs = doc.get("dataset_info") or {}
    if not isinstance(configs, dict) or not configs:
        return None
    # First config wins. Multi-config datasets pick one config at
    # materialize time anyway; the user can switch via the form's
    # split dropdown if they need a different config.
    _first_name, cfg = next(iter(configs.items()))
    if not isinstance(cfg, dict):
        return None
    features = cfg.get("features")
    if not isinstance(features, dict) or not features:
        return None
    splits_raw = cfg.get("splits") or {}
    splits: list[str] = []
    if isinstance(splits_raw, dict):
        splits = [s for s in splits_raw.keys() if isinstance(s, str)]
    elif isinstance(splits_raw, list):
        for s in splits_raw:
            if isinstance(s, str):
                splits.append(s)
            elif isinstance(s, dict) and isinstance(s.get("name"), str):
                splits.append(s["name"])
    return {"features": features, "splits": splits}


def fetch_hf_card_description(repo_id: str, *, timeout: int = 10) -> str | None:
    """Return the markdown body of an HF dataset's README.md with the
    YAML front-matter stripped off. Used to seed
    `Dataset.card_description` so the per-dataset page can surface the
    upstream card text under a Details section.
    Returns None on any HTTP / decode failure."""
    if not repo_id:
        return None
    url = (f"https://huggingface.co/datasets/"
           f"{urllib.parse.quote(repo_id, safe='/')}/raw/main/README.md")
    req = urllib.request.Request(
        url, headers={"User-Agent": "benchhub/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError):
        return None
    body = raw
    # Strip YAML frontmatter: starts at the first `---\n` and ends at
    # the next `---\n`. HF README templates always emit it; older
    # cards may not. Only strip the leading delimited block.
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            body = body[end + 4 :].lstrip("\n")
    return body.strip() or None


def fetch_dataset_card(repo_id: str, *, timeout: int = 10) -> dict | None:
    """Return the HF Hub's per-dataset metadata document — the same
    JSON the dataset card page is rendered from. Carries `tags`
    (list of `task_categories:*`, `task_ids:*`, `language:*`, …),
    `cardData` (the YAML front-matter the uploader declared), and
    similar discovery signals. Used by the importer to lift
    `task_categories` into BH's Area/Task taxonomy at materialise
    time so the imported dataset isn't dropped into Uncategorized.

    Returns the parsed dict or None on any HTTP / parse failure.
    """
    if not repo_id:
        return None
    url = f"{_HF_API_BASE}/{urllib.parse.quote(repo_id, safe='/')}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "benchhub/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        doc = json.loads(body)
    except (TypeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def card_summary(repo_id: str, *, timeout: int = 10) -> dict | None:
    """A compact, human-readable summary of a dataset's HF card — shown
    before the user commits to a full preview. Returns
    `{id, title, description, gated, private, downloads, likes,
    task_categories}` or None if the repo can't be read.

    The HF detail API's `description` is the card text with markdown
    structure flattened (lots of stray tabs/newlines), so we collapse
    whitespace and, when present, jump past the boilerplate
    'Dataset Card for X … Dataset Summary' header to the real summary.
    Successful results are memoised for an hour so the same dataset shown
    across several preview stages isn't re-fetched each time."""
    if not repo_id:
        return None
    now = time.time()
    cached = _CARD_CACHE.get(repo_id)
    if cached and (now - cached[0]) < _CARD_TTL_SECONDS:
        return cached[1]
    doc = fetch_dataset_card(repo_id, timeout=timeout)
    if not doc:
        return None
    cd = doc.get('cardData') or {}
    desc = re.sub(r'\s+', ' ', (doc.get('description') or '')).strip()
    i = desc.lower().find('dataset summary')
    if i != -1:
        desc = desc[i + len('dataset summary'):].strip(' :-–—')
    if len(desc) > 600:
        desc = desc[:600].rsplit(' ', 1)[0] + '…'
    tags = [t.split(':', 1)[1] for t in (doc.get('tags') or [])
            if isinstance(t, str) and t.startswith('task_categories:')]
    title = cd.get('pretty_name') or cd.get('title') or repo_id
    summary = {
        'id': repo_id,
        'title': str(title)[:140],
        'description': desc,
        'gated': bool(doc.get('gated') or False),
        'private': bool(doc.get('private') or False),
        'downloads': int(doc.get('downloads') or 0),
        'likes': int(doc.get('likes') or 0),
        'task_categories': tags[:6],
    }
    _CARD_CACHE[repo_id] = (now, summary)
    return summary


def _fetch_size_doc(repo_id: str, *, timeout: int) -> dict | None:
    """Internal: GET the datasets-server `/size` doc as a dict, or
    None on any failure. Shared by row-count and byte-size lookups
    so we don't double the HTTP round-trips."""
    if not repo_id:
        return None
    params = urllib.parse.urlencode({"dataset": repo_id})
    req = urllib.request.Request(
        f"{_HF_DATASETS_SERVER}?{params}",
        headers={"User-Agent": "benchhub/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        doc = json.loads(body)
    except (TypeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def fetch_split_row_counts(repo_id: str, *, timeout: int = 10) -> dict[str, int]:
    """Per-split row count for an HF dataset.

    Hits HF's `datasets-server` `/size` endpoint, which returns one
    entry per (config, split) with a `num_rows` field. We collapse
    that to `{split_name: num_rows}` — for multi-config datasets the
    first config's count wins, since the import flow downloads from
    one config at a time anyway.

    Returns `{}` on network failure / non-JSON / unknown shape so
    the preview UI degrades to "no count available" rather than 500.
    """
    doc = _fetch_size_doc(repo_id, timeout=timeout)
    if doc is None:
        return {}
    splits = ((doc.get("size") or {}).get("splits")) or []
    out: dict[str, int] = {}
    for s in splits:
        if not isinstance(s, dict):
            continue
        name = s.get("split")
        n = s.get("num_rows")
        if isinstance(name, str) and isinstance(n, int):
            out.setdefault(name, n)
    return out


def fetch_split_byte_sizes(repo_id: str, *, timeout: int = 10) -> dict[str, int]:
    """Per-split storage size in bytes for an HF dataset.

    Pulled from the same `/size` endpoint as `fetch_split_row_counts`.
    Uses `num_bytes_parquet_files` (HF's on-disk parquet size) as the
    proxy for how big a full materialization will be; for image /
    audio datasets the BH PNG/NPZ layout typically lands within an
    order of magnitude of parquet, which is plenty of fidelity for a
    pre-import quota guard. Callers should multiply by a headroom
    factor (e.g. 1.5x) before deciding to reject.

    Returns `{}` on any failure → quota check degrades to the
    post-materialization safety net.
    """
    doc = _fetch_size_doc(repo_id, timeout=timeout)
    if doc is None:
        return {}
    splits = ((doc.get("size") or {}).get("splits")) or []
    out: dict[str, int] = {}
    for s in splits:
        if not isinstance(s, dict):
            continue
        name = s.get("split")
        b = s.get("num_bytes_parquet_files")
        if isinstance(name, str) and isinstance(b, int) and b > 0:
            out.setdefault(name, b)
    return out
