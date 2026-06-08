"""Discovery helpers over the Kaggle dataset API — the Kaggle analogue of
benchhub.hf_search. Unlike HF, Kaggle requires auth even to search, so every
function takes a `KaggleClient` (the route builds one from the service
account). Search is live; card summaries and trending are memoised with a
1-hour TTL so browsing doesn't hammer the API or hit rate limits.

Each normalised record carries the **license redistribution verdict**
(`benchhub.kaggle_client.classify_license`) so the UI can badge a dataset and
the import flow can gate public visibility before any bytes are downloaded.
"""
from __future__ import annotations

import time

from .kaggle_client import classify_license, license_name_from_view


def _normalize(d):
    """Project a Kaggle dataset dict down to the fields the UI shows + the
    license verdict."""
    ref = d.get("ref") or ""
    if not ref and d.get("ownerName") and d.get("slug"):
        ref = f"{d['ownerName']}/{d['slug']}"
    lic = (d.get("licenseName") or d.get("licenseShortName") or "")
    verdict = classify_license(lic)
    return {
        "ref": ref,
        "title": (d.get("title") or ref or "").strip()[:140],
        "subtitle": (d.get("subtitle") or "").strip()[:200],
        "total_bytes": int(d.get("totalBytes") or 0),
        "votes": int(d.get("voteCount") or 0),
        "downloads": int(d.get("downloadCount") or d.get("downloads") or 0),
        "usability": float(d.get("usabilityRating") or 0.0),
        "license_name": lic,
        "redistributable": verdict["redistributable"],
        "license_category": verdict["category"],
        "last_updated": d.get("lastUpdated") or "",
        "version": d.get("currentVersionNumber"),
    }


def search_datasets(client, query, *, limit=20, **filters):
    """Live free-text dataset search → normalised records. Empty query →
    []. Network failures bubble up as KaggleAPIError/KaggleAuthError so the
    route can show a real message (unlike the HF dropdown's silent [])."""
    q = (query or "").strip()
    if not q and not filters.get("sort_by"):
        return []
    rows = client.search(q, page_size=int(limit), **filters)
    return [_normalize(d) for d in rows]


# In-memory TTL caches (mirror hf_search): trending per-domain, card per-ref.
_TRENDING_CACHE = {}
_TRENDING_TTL = 60 * 60
_CARD_CACHE = {}
_CARD_TTL = 60 * 60

# BH domain → a (sort, file_type, tag) hint set for the trending grid.
_DOMAIN_QUERY = {
    "Vision": {"sort_by": "votes", "tags": "13207"},      # computer-vision tag
    "NLP": {"sort_by": "votes", "tags": "13204"},          # nlp tag
    "Audio": {"sort_by": "votes", "tags": "13206"},        # audio tag
    "Tabular": {"sort_by": "votes", "file_type": "csv"},
}


def trending_by_domain(client, *, limit_per_domain=5):
    """Top datasets per ML domain, cached 1h. Returns {domain: [records]}
    in insertion order. A failing domain degrades to [] rather than 500."""
    out = {}
    now = time.time()
    limit = max(1, int(limit_per_domain))
    for domain, q in _DOMAIN_QUERY.items():
        key = f"{domain}:{limit}"
        cached = _TRENDING_CACHE.get(key)
        if cached and (now - cached[0]) < _TRENDING_TTL:
            out[domain] = cached[1]
            continue
        try:
            rows = client.search("", page_size=limit, **q)
            items = [_normalize(d) for d in rows]
        except Exception:
            items = []
        _TRENDING_CACHE[key] = (now, items)
        out[domain] = items
    return out


def card_summary(client, ref, *, timeout=None):
    """Compact, human-readable card for one dataset — title, subtitle,
    description, size, usability, votes, the file list size, and the license
    verdict. Memoised 1h. Returns None if the dataset can't be read."""
    if not ref:
        return None
    now = time.time()
    cached = _CARD_CACHE.get(ref)
    if cached and (now - cached[0]) < _CARD_TTL:
        return cached[1]
    try:
        view = client.view(ref)
    except Exception:
        return None
    if not isinstance(view, dict):
        return None
    lic = license_name_from_view(view)
    verdict = classify_license(lic)
    desc = (view.get("description") or view.get("subtitle") or "").strip()
    if len(desc) > 600:
        desc = desc[:600].rsplit(" ", 1)[0] + "…"
    summary = {
        "ref": ref,
        "title": (view.get("title") or ref).strip()[:140],
        "subtitle": (view.get("subtitle") or "").strip()[:200],
        "description": desc,
        "total_bytes": int(view.get("totalBytes") or 0),
        "usability": float(view.get("usabilityRating") or 0.0),
        "votes": int(view.get("voteCount") or 0),
        "downloads": int(view.get("downloadCount") or 0),
        "version": view.get("currentVersionNumber"),
        "license_name": lic,
        "redistributable": verdict["redistributable"],
        "license_category": verdict["category"],
    }
    _CARD_CACHE[ref] = (now, summary)
    return summary


def _clear_cache():
    """Test-only: drop the trending + card caches."""
    _TRENDING_CACHE.clear()
    _CARD_CACHE.clear()
