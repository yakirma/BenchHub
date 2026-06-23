"""Build moat-safe public-leaderboard standings for the read-only HF mirror.

Pure builder: given the ORM models, produce the JSON the HF Space consumes
(see docs/HF_SPACE_MIRROR_PLAN.md). Exports ONLY aggregate public standings —
already-pooled MetricResult scalars + LB/metric/submission DISPLAY metadata.

NEVER exports (the required fixes, plan §3):
  #1  metric error tracebacks (MetricResult.error_message) — a cell with no
      numeric value becomes null, never the exception string.
  #2  raw submitter identity — owner.display_name only when it isn't the
      email local-part; git_author is never exported; fallback 'Anonymous'.
And structurally never: per-sample GT, dataset bytes, prediction files,
arg_mappings/tag_filter internals, or private/unlisted LBs (the caller gates
enumeration on visibility).

`assert_no_leak(payload)` is the belt-and-suspenders scan run before any push.
"""
from __future__ import annotations

import hashlib
import json
import re

RUNBENCHHUB = "https://runbenchhub.com"

# Trailing parenthetical qualifier(s) on a submission name — "RAFT-Stereo (ETH3D)",
# "CREStereo (combined, iter10)" — that distinguish fine-tune / config variants of
# the SAME base model. Stripped for ranking identity so variants group together.
_VARIANT_SUFFIX_RE = re.compile(r"(?:\s*\([^()]*\))+\s*$")

# Substrings that must never appear in an exported string value — a tripwire
# for the moat fixes above (tracebacks, on-disk paths, serialized arrays).
_LEAK_MARKERS = (
    "Traceback (most recent call last)",
    "/home/", "uploads/", "datasets/", "submissions/",
    ".npz", ".npy",
)


def _short(text, limit: int = 280):
    """Collapse whitespace + truncate to a catalog-sized blurb (dataset
    cards can be multi-KB of markdown/TOC; the full text lives on BH/HF)."""
    if not text:
        return None
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[:limit].rstrip() + "…"


def _safe_author(sub) -> str:
    """Submitter identity for a public, third-party mirror (fix #2).

    Prefer owner.display_name, but NOT when it equals the email local-part
    (BenchHub defaults display_name to email.split('@')[0], app.py:2268), and
    never export the raw git_author commit string. Fallback 'Anonymous'."""
    owner = getattr(sub, "owner", None)
    if owner is not None:
        dn = (getattr(owner, "display_name", None) or "").strip()
        email = getattr(owner, "email", None) or ""
        email_local = email.split("@", 1)[0] if "@" in email else ""
        if dn and dn != email_local:
            return dn
    return "Anonymous"


_AUTHOR_HOST_LABELS = {
    "docs.opencv.org": "OpenCV", "opencv.org": "OpenCV",
    "github.com": "GitHub", "gitlab.com": "GitLab",
    "huggingface.co": "Hugging Face",   # generic HF link with no /<owner>/<model>
    "pytorch.org": "PyTorch", "tensorflow.org": "TensorFlow",
}


def _author_fields(sub):
    """(author, author_url) crediting the MODEL's SOURCE — derived from the
    submission link, never the BenchHub user who ran the eval. In priority:

      huggingface.co/<owner>/<model> → the HF owner (links to their HF page)
      github.com/<owner>/<repo>      → the GitHub owner (links to their profile)
      any other http(s) link         → the source domain (e.g. "OpenCV"), link out

    A submission may be a third-party model someone curated onto the board (e.g.
    the stereo boards: RAFT-Stereo, HITNet, … submitted by an admin), so crediting
    the submitter would falsely attribute the model. When there's no usable link,
    show '—' (unknown) rather than the submitter."""
    link = (getattr(sub, "link", None) or "").strip()
    low = link.lower()
    for host, base in (("huggingface.co/", "https://huggingface.co"),
                       ("github.com/",      "https://github.com")):
        if host in low:
            rest = link[low.index(host) + len(host):].strip("/")
            parts = [p for p in rest.split("/") if p]
            # HF needs <owner>/<model>; GitHub credits the owner from <owner>/<repo>.
            need = 2 if "huggingface" in host else 1
            if len(parts) >= need and parts[0]:
                return parts[0], f"{base}/{parts[0]}"
    if low.startswith(("http://", "https://")):
        netloc = low.split("//", 1)[1].split("/", 1)[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return _AUTHOR_HOST_LABELS.get(netloc, netloc), link
    return "—", None   # em-dash: unknown author (never the submitter)


def _score(mr) -> float | None:
    """Cell value (fix #1): the already-pooled scalar, or None. NEVER the
    error_message — it can embed GT/sample/prediction values."""
    if mr is None:
        return None
    v = mr.value
    return v if isinstance(v, (int, float)) else None


def _columns(lb) -> list[dict]:
    """LB columns mirroring leaderboard_view: summary_metrics order (resolving
    both `lm_<id>` and display-name tokens), fallback to all metrics by id.
    Internal pooling/arg_mappings/tag_filter are intentionally omitted."""
    lms = {f"lm_{m.id}": m for m in lb.leaderboard_metrics}

    def label_of(m):
        return m.target_name or (m.global_metric.name if m.global_metric else f"metric_{m.id}")

    name_to_ids: dict[str, list[str]] = {}
    for k, m in lms.items():
        name_to_ids.setdefault(label_of(m), []).append(k)

    order = [t.strip() for t in (lb.summary_metrics or "").split(",") if t.strip()]
    col_keys: list[str] = []
    for tok in order:
        if tok in lms:
            col_keys.append(tok)
        elif tok in name_to_ids:
            col_keys.extend(sorted(name_to_ids[tok]))
    if not col_keys:
        col_keys = sorted(lms.keys(), key=lambda k: lms[k].id)

    cols, seen = [], set()
    for k in col_keys:
        if k in lms and k not in seen:
            seen.add(k)
            m = lms[k]
            cols.append({
                "metric_id": m.id,
                "label": label_of(m),
                "global_metric": m.global_metric.name if m.global_metric else None,
                "sort_direction": m.sort_direction or "higher_is_better",
            })
    return cols


def _submit_url(lb_id: int) -> str:
    return f"{RUNBENCHHUB}/leaderboard/{lb_id}?utm_source=hf_space&utm_medium=submit_btn"


def build_lb_standings(lb, *, MetricResult) -> dict:
    """Per-LB standings payload (aggregate-only). Caller must have already
    confirmed `lb` is public."""
    cols = _columns(lb)
    subs = [s for s in lb.submissions if not getattr(s, "is_archived", False)]
    sub_ids = [s.id for s in subs]

    cells: dict[tuple, object] = {}
    if sub_ids:
        for mr in MetricResult.query.filter(MetricResult.submission_id.in_(sub_ids)).all():
            cells[(mr.submission_id, mr.leaderboard_metric_id)] = mr

    def row(s) -> dict:
        author, author_url = _author_fields(s)
        return {
            "name": s.name,
            "author": author,
            "author_url": author_url,
            "created": s.upload_date.isoformat() if s.upload_date else None,
            "description": (s.description or None),
            "link": (s.link or None),
            "scores": {str(c["metric_id"]): _score(cells.get((s.id, c["metric_id"]))) for c in cols},
        }

    verified = [row(s) for s in subs
                if (s.kind or "verified") != "mirrored" and s.processing_status == "Processed"]
    mirrored = []
    for s in subs:
        if (s.kind or "verified") == "mirrored":
            r = row(s)
            r.update({
                "source_attribution": s.source_attribution or None,
                "source_paper_url": s.source_paper_url or None,
                "source_external_url": s.source_external_url or None,
            })
            mirrored.append(r)

    # Rank verified best-first by the first column's sort_direction; missing /
    # non-numeric scores sink to the bottom (matches leaderboard_view).
    if cols:
        mid0 = str(cols[0]["metric_id"])
        rev = cols[0]["sort_direction"] != "lower_is_better"
        sink = float("-inf") if rev else float("inf")

        def keyf(r):
            v = r["scores"].get(mid0)
            return v if isinstance(v, (int, float)) else sink

        verified.sort(key=keyf, reverse=rev)
    for i, r in enumerate(verified, 1):
        r["rank"] = i

    return {
        "id": lb.id,
        "name": lb.name,
        "category": lb.category,
        "url": f"{RUNBENCHHUB}/leaderboard/{lb.id}",
        "submit_url": _submit_url(lb.id),
        "columns": cols,
        "verified": verified,
        "mirrored": mirrored,
    }


def build_index_entry(lb, payload: dict) -> dict:
    """One catalog entry for index.json. Descriptive text comes from the
    linked dataset (Leaderboard has no description column)."""
    datasets = []
    for d in (lb.datasets or []):
        datasets.append({
            "name": d.name,
            "description": _short(getattr(d, "card_description", None)),
            "source_url": (getattr(d, "source_url", None) or None),
        })
    # Leader (rank-1 verified row) + its score on the primary metric, so the
    # mirror's catalog cards can show "🥇 <model> · <score>" without fetching
    # every per-board file. Cheap: the payload is already ranked best-first.
    top = None
    _v = payload.get("verified") or []
    _cols = payload.get("columns") or []
    if _v and _cols:
        _r0, _mid0 = _v[0], str(_cols[0]["metric_id"])
        top = {
            "name": _r0.get("name"),
            "score": (_r0.get("scores") or {}).get(_mid0),
            "metric": _cols[0].get("label"),
        }
    return {
        "id": lb.id,
        "name": lb.name,
        "category": lb.category,
        "url": f"{RUNBENCHHUB}/leaderboard/{lb.id}",
        "submit_url": _submit_url(lb.id),
        "datasets": datasets,
        "n_verified": len(payload["verified"]),
        "n_mirrored": len(payload["mirrored"]),
        "n_metrics": len(payload.get("columns") or []),
        "top": top,
        "updated_at": lb.upload_date.isoformat() if getattr(lb, "upload_date", None) else None,
    }


# ---------------------------------------------------------------------------
# Category model-rankings ("meta-leaderboards") — aggregate a model's results
# across all the boards in a category/sub-category into one normalized score.
# Pure functions, shared by the site route and the HF-mirror export so both
# agree. Identity is the model's HF id (from its submission link/name); no
# model-registry table needed — submissions are grouped on the fly.
# ---------------------------------------------------------------------------
def model_identity(name, link):
    """(key, display) for a submission's MODEL, stable across boards. Prefer the
    HF id from the link (huggingface.co/<owner>/<model>), else the submission
    name with any trailing parenthetical variant qualifier stripped, so a base
    model's fine-tune / config variants — "RAFT-Stereo (ETH3D)", "(fast)",
    "(iter10)" — group as ONE model in the rankings (best variant per board).
    Key is lowercased for matching; display keeps original casing."""
    link = (link or "").strip()
    low = link.lower()
    if "huggingface.co/" in low:
        rest = link[low.index("huggingface.co/") + len("huggingface.co/"):].strip("/")
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 2:
            disp = f"{parts[0]}/{parts[1]}"
            return disp.lower(), disp
    nm = (name or "").strip()
    base = _VARIANT_SUFFIX_RE.sub("", nm).strip()
    if base:                       # don't let an all-parenthetical name collapse to ""
        nm = base
    return nm.lower(), nm


def _normalize(rows, higher_is_better):
    """rows: [{key, score}]. Min-max scale scores to [0,1] (best=1, worst=0,
    flipping when lower-is-better). Returns {key: norm}, taking a model's BEST
    submission if it appears more than once on the board. All-equal/one-row →
    1.0 (indistinguishable = treated as the board's best)."""
    vals = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    out = {}
    for r in rows:
        s = r.get("score")
        if not isinstance(s, (int, float)):
            continue
        norm = 1.0 if rng == 0 else (s - lo) / rng
        if not higher_is_better:
            norm = 1.0 - norm
        k = r["key"]
        out[k] = max(out.get(k, -1.0), norm)
    return out


def compute_aggregates(boards):
    """Build the meta-leaderboards. `boards` is a list of:
        {id, name, category, higher_is_better, rows: [{key, model, link, score}]}
    Produces one ranking per SUB-CATEGORY (a category containing "/") that has
    >=2 boards. Top-level categories are intentionally NOT ranked — only the
    narrower, apples-to-apples sub-category scopes. Each model is scored by the
    MEAN of its normalized per-board scores. A model is only ranked if it
    appears on >=50% of the sub-category's boards (coverage gate), and a scope
    is only emitted if >=3 such models remain. Each model's `per_board` maps a
    board name -> {lb_id, norm} so callers can link to the model's standing on
    each board. Returns a list of scope dicts sorted by breadth then name."""
    from collections import defaultdict
    # scope_key -> (level, list of boards). Only sub-category scopes: a board
    # whose category has no "/" (top-level only) gets no meta-ranking.
    scopes = defaultdict(lambda: {"level": None, "boards": []})
    for b in boards:
        cat = (b.get("category") or "Uncategorized").strip()
        if "/" not in cat:                  # top-level only → no meta-ranking
            continue
        scopes[cat]["level"] = "subcategory"
        scopes[cat]["boards"].append(b)

    result = []
    for scope, info in scopes.items():
        bds = info["boards"]
        if len(bds) < 2:                      # need >=2 boards in the sub-category
            continue
        n_boards = len(bds)
        # A model must appear on >=50% of the sub-category's boards to be ranked
        # (ceil, so 2 boards needs 1, 3 needs 2, 4 needs 2, 5 needs 3).
        min_cov = (n_boards + 1) // 2
        per_model_norms = defaultdict(dict)   # key -> {board_name: {lb_id, norm}}
        display = {}                          # key -> (model, link)
        for b in bds:
            norms = _normalize(b["rows"], b.get("higher_is_better", True))
            for r in b["rows"]:
                display.setdefault(r["key"], (r.get("model") or r["key"], r.get("link")))
            for k, nv in norms.items():
                per_model_norms[k][b["name"]] = {"lb_id": b.get("id"), "norm": nv}
        models = []
        for k, bmap in per_model_norms.items():
            if len(bmap) < min_cov:           # coverage gate: drop <50% models
                continue
            mdisp, mlink = display.get(k, (k, None))
            models.append({
                "model": mdisp,
                "link": mlink,
                "score": round(sum(v["norm"] for v in bmap.values()) / len(bmap), 4),
                "coverage": len(bmap),
                "per_board": bmap,
            })
        if len(models) < 3:                   # need >=3 valid models to be worth a table
            continue
        models.sort(key=lambda m: (-m["score"], -m["coverage"], m["model"].lower()))
        result.append({
            "scope": scope,
            "level": info["level"],
            "n_boards": n_boards,
            "boards": [b["name"] for b in bds],
            "n_models": len(models),
            "models": models,
        })
    # broadest sub-categories first, then alphabetical
    result.sort(key=lambda s: (-s["n_boards"], s["scope"]))
    return result


def payload_hash(obj) -> str:
    """Stable content hash for idempotent skip-if-unchanged."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def assert_no_leak(payload: dict) -> None:
    """Tripwire: raise if any exported string smells like a traceback, an
    on-disk path, or a serialized array. Run before every push."""
    blob = json.dumps(payload, ensure_ascii=False)
    for marker in _LEAK_MARKERS:
        if marker in blob:
            raise ValueError(
                f"moat-leak tripwire: exported standings contain {marker!r} "
                f"(LB {payload.get('id')}) — refusing to publish."
            )


def _readme(generated_at: str, n_lbs: int) -> str:
    return (
        "---\n"
        "license: other\n"
        "tags:\n  - leaderboard\n  - benchmark\n  - mirror\n"
        "pretty_name: BenchHub leaderboard standings (mirror)\n"
        "---\n\n"
        "# BenchHub leaderboard standings — read-only mirror\n\n"
        f"Auto-generated aggregate standings for {n_lbs} public BenchHub leaderboard(s).\n\n"
        "**Source of truth: <https://runbenchhub.com>.** Submissions happen on BenchHub; "
        "this dataset is a derived, read-only mirror of public leaderboard *standings* "
        "(already-pooled scores + display metadata). It contains **no** ground-truth "
        "samples, **no** predictions, and **no** private data.\n\n"
        f"Last synced: {generated_at}\n\n"
        "- `index.json` — catalog of mirrored leaderboards.\n"
        "- `leaderboards/<id>.json` — per-leaderboard ranked standings.\n"
        "- `aggregates.json` — per-category model rankings (mean normalized "
        "score across the boards in each category/sub-category).\n"
    )


def build_repo_files(lbs, *, MetricResult, generated_at: str):
    """Build the full set of HF-repo files `{relpath: text}` for `lbs`
    (which the caller must already have gated to public). Runs the leak
    tripwire on every per-LB payload. Returns (files, manifest)."""
    files: dict[str, str] = {}
    manifest = {"schema_version": 1, "generated_at": generated_at, "leaderboards": {}}
    index = {"generated_at": generated_at, "source": RUNBENCHHUB, "leaderboards": []}
    agg_boards = []
    for lb in lbs:
        p = build_lb_standings(lb, MetricResult=MetricResult)
        assert_no_leak(p)
        # Skip empty boards — a leaderboard with no verified submissions
        # is noise in the read-only standings mirror (nothing to rank), so
        # don't publish its file or list it in the index. The push task's
        # manifest diff then de-publishes any that previously had subs.
        if not p.get("verified"):
            continue
        files[f"leaderboards/{lb.id}.json"] = json.dumps(p, indent=2, ensure_ascii=False)
        manifest["leaderboards"][str(lb.id)] = payload_hash(p)
        index["leaderboards"].append(build_index_entry(lb, p))
        # Collect (model -> primary-metric score) rows for the category rankings.
        cols = p.get("columns") or []
        if cols:
            mid0 = str(cols[0]["metric_id"])
            hib = (cols[0].get("sort_direction") or "higher_is_better") != "lower_is_better"
            rows = []
            for r in p["verified"]:
                k, disp = model_identity(r.get("name"), r.get("link"))
                rows.append({"key": k, "model": disp, "link": r.get("link"),
                             "score": (r.get("scores") or {}).get(mid0)})
            agg_boards.append({"id": lb.id, "name": p.get("name"),
                               "category": p.get("category"),
                               "higher_is_better": hib, "rows": rows})
    index["leaderboards"].sort(key=lambda e: ((e["category"] or "~"), (e["name"] or "").lower()))
    files["index.json"] = json.dumps(index, indent=2, ensure_ascii=False)
    files["aggregates.json"] = json.dumps(
        {"generated_at": generated_at, "scopes": compute_aggregates(agg_boards)},
        indent=2, ensure_ascii=False)
    files["_manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False)
    files["README.md"] = _readme(generated_at, len(index["leaderboards"]))
    return files, manifest


def hf_source_repos(lbs):
    """Sorted unique HuggingFace dataset repo ids backing the given LBs
    (parsed from each linked Dataset.source_url). Feeds the Space's
    `datasets:` card metadata so HF cross-links the mirror to its sources."""
    import re
    repos = set()
    for lb in lbs:
        for d in (getattr(lb, "datasets", None) or []):
            m = re.search(r"huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)",
                          getattr(d, "source_url", None) or "")
            if m:
                repos.add(m.group(1).rstrip("/"))
    return sorted(repos)


def set_card_datasets(readme_text, repos):
    """Return `readme_text` with its YAML-frontmatter `datasets:` block
    replaced by `repos` (line-based; preserves every other frontmatter key)."""
    import re
    lines = readme_text.split("\n")
    if not lines or lines[0].strip() != "---":
        return readme_text
    try:
        end = lines.index("---", 1)
    except ValueError:
        return readme_text
    fm, body = lines[1:end], lines[end + 1:]
    out, i = [], 0
    while i < len(fm):
        if fm[i].rstrip() == "datasets:":
            i += 1
            while i < len(fm) and re.match(r"\s+-\s", fm[i]):
                i += 1
            continue
        out.append(fm[i]); i += 1
    if repos:
        out.append("datasets:")
        out.extend(f"  - {r}" for r in repos)
    return "\n".join(["---"] + out + ["---"] + body)
