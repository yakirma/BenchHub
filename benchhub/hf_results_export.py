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

RUNBENCHHUB = "https://runbenchhub.com"

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


def _author_fields(sub):
    """(author, author_url) crediting the MODEL's owner for HF-model
    submissions — the namespace in the submission link
    (huggingface.co/<owner>/<model>), linking to that owner's HF page — not
    the BenchHub user who ran the eval. Falls back to the scrubbed submitter
    when the link isn't an HF model URL."""
    link = getattr(sub, "link", None) or ""
    if "huggingface.co/" in link:
        rest = link.split("huggingface.co/", 1)[1].strip("/")
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 2:                      # <owner>/<model>
            owner = parts[0]
            return owner, f"https://huggingface.co/{owner}"
    return _safe_author(sub), None


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
    return {
        "id": lb.id,
        "name": lb.name,
        "category": lb.category,
        "url": f"{RUNBENCHHUB}/leaderboard/{lb.id}",
        "submit_url": _submit_url(lb.id),
        "datasets": datasets,
        "n_verified": len(payload["verified"]),
        "n_mirrored": len(payload["mirrored"]),
        "updated_at": lb.upload_date.isoformat() if getattr(lb, "upload_date", None) else None,
    }


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
    )


def build_repo_files(lbs, *, MetricResult, generated_at: str):
    """Build the full set of HF-repo files `{relpath: text}` for `lbs`
    (which the caller must already have gated to public). Runs the leak
    tripwire on every per-LB payload. Returns (files, manifest)."""
    files: dict[str, str] = {}
    manifest = {"schema_version": 1, "generated_at": generated_at, "leaderboards": {}}
    index = {"generated_at": generated_at, "source": RUNBENCHHUB, "leaderboards": []}
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
    index["leaderboards"].sort(key=lambda e: ((e["category"] or "~"), (e["name"] or "").lower()))
    files["index.json"] = json.dumps(index, indent=2, ensure_ascii=False)
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
