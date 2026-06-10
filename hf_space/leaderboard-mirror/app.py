"""BenchHub leaderboards — read-only HuggingFace Space mirror.

Reads the standings dataset repo BenchHub pushes (see
docs/HF_SPACE_MIRROR_PLAN.md) and renders sortable per-leaderboard standings,
with domain (category) filtering + search. NO submission UI by design — every
"Submit" affordance is an outbound link to runbenchhub.com.
"""
import json
import os

import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = os.environ.get("HF_RESULTS_REPO", "runbenchhub/leaderboards")
SITE = "https://runbenchhub.com"
HEAD = '<meta name="robots" content="noindex,follow">'
ALL = "All domains"


def _load(filename, force=False):
    # hf_hub_download is etag-cached: fast when unchanged, re-pulls when the
    # daily sync updates a file — so the Space stays fresh without a restart.
    path = hf_hub_download(repo_id=DATASET_REPO, filename=filename,
                           repo_type="dataset", force_download=force)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _index():
    try:
        return _load("index.json")
    except Exception:
        return {"generated_at": None, "leaderboards": []}


def _label(e):
    return f"{e['name']} · {e.get('n_verified', 0)} subs"


def _standings(lb_id):
    if lb_id is None:
        return gr.update(value=pd.DataFrame()), "_No leaderboards match this domain/search._"
    data = _load(f"leaderboards/{lb_id}.json")
    cols = data.get("columns", [])
    verified = data.get("verified", [])
    metric_labels = [c["label"] for c in cols]
    headers = ["Rank", "Submission", "Author"] + metric_labels + ["Date"]

    # Native cell values: metric columns stay numeric (right-aligned + green-
    # styled below); only the Submission column is a markdown link to the model.
    rows = []
    for r in verified:
        name = r.get("name", "")
        link = r.get("link")
        sub = f"[{name}]({link})" if link else name
        author = r.get("author") or ""
        a_url = r.get("author_url")
        author_cell = f"[{author}]({a_url})" if a_url else author
        scores = [r["scores"].get(str(c["metric_id"])) for c in cols]
        rows.append([r.get("rank"), sub, author_cell, *scores,
                     (r.get("created") or "")[:10]])
    df = pd.DataFrame(rows, columns=headers)

    # BenchHub green heatmap on metric cells (full-cell, via pandas Styler):
    # best in column = vivid green, worst = pale, respecting sort_direction.
    dir_by = {c["label"]: (c.get("sort_direction") or "higher_is_better") for c in cols}

    def _green(series):
        nums = pd.to_numeric(series, errors="coerce")
        lo, hi = nums.min(), nums.max()
        rng = (hi - lo) or 1.0
        higher = dir_by.get(series.name, "higher_is_better") != "lower_is_better"
        out = []
        for v in nums:
            if pd.isna(v):
                out.append("")
                continue
            norm = (v - lo) / rng
            if not higher:
                norm = 1.0 - norm
            out.append(f"background-color: hsl(120, 70%, {90 - norm * 47:.0f}%); color: #0b3d0b;")
        return out

    value = df
    if metric_labels and not df.empty:
        try:
            value = df.style.apply(_green, subset=metric_labels, axis=0)
        except Exception:
            value = df

    # Per-column datatype so numbers render as numbers (the original look) while
    # the Submission column renders its markdown link.
    datatype = ["number", "markdown", "markdown"] + ["number"] * len(metric_labels) + ["str"]

    submit = data.get("submit_url", f"{SITE}/leaderboard/{lb_id}")
    view = data.get("url", f"{SITE}/leaderboard/{lb_id}")
    links = (
        f"### {data.get('name', '')}\n"
        f"<span style='color:#888'>{data.get('category') or 'Uncategorized'}</span>\n\n"
        f"<a href='{submit}' target='_blank' rel='noopener'>"
        f"🚀 Submit your model on BenchHub →</a>  ·  "
        f"<a href='{view}' target='_blank' rel='noopener'>View on BenchHub</a>\n\n"
        f"<sub>Read-only mirror — submissions run on BenchHub. No upload here by design.</sub>"
    )
    return gr.update(value=value, datatype=datatype), links


def _entries():
    """Fetch the LB index FRESH (etag-cached) so new leaderboards/scores show
    up without a Space restart."""
    # Hide benchmarks with no verified submissions — an empty board is
    # noise in a read-only standings mirror (nothing to rank).
    es = [e for e in _index().get("leaderboards", [])
          if (e.get("n_verified") or 0) > 0]
    for e in es:
        cat = (e.get("category") or "Uncategorized")
        parts = cat.split("/", 1)
        e["_area"] = parts[0]                                   # e.g. "Vision"
        e["_subarea"] = parts[1].strip() if len(parts) > 1 else ""  # e.g. "Image Segmentation"
    return es


def _subareas(entries, area):
    """Sub-domains available under the chosen domain (the part after the
    first `/` in the category). `ALL` first; empty when none exist."""
    subs = sorted({e["_subarea"] for e in entries
                   if (not area or area == ALL or e["_area"] == area) and e["_subarea"]})
    return [ALL] + subs


def _choices(entries, area, sub, query):
    q = (query or "").strip().lower()
    out = []
    for e in entries:
        if area and area != ALL and e["_area"] != area:
            continue
        if sub and sub != ALL and e["_subarea"] != sub:
            continue
        if q and q not in f"{e['name']} {e.get('category') or ''}".lower():
            continue
        out.append((_label(e), e["id"]))   # (display, value=lb_id)
    return out


def build():
    with gr.Blocks(title="BenchHub Leaderboards (mirror)", head=HEAD,
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏆 BenchHub Leaderboards — read-only mirror")
        gr.Markdown(
            f"Live boards & model submission at **[runbenchhub.com]({SITE})**. "
            f"Mirrors *public leaderboard standings only* — no ground-truth data, "
            f"no submissions."
        )
        synced = gr.Markdown()
        with gr.Row():
            area_dd = gr.Dropdown(choices=[ALL], value=ALL, label="Domain", scale=2)
            sub_dd = gr.Dropdown(choices=[ALL], value=ALL, label="Sub-domain", scale=2)
            search = gr.Textbox(label="Search", scale=3,
                                placeholder="filter by leaderboard name or category, then Enter…")
            refresh_btn = gr.Button("🔄 Refresh", scale=0)
        lb_dd = gr.Dropdown(choices=[], label="Leaderboard")
        links = gr.Markdown()
        table = gr.Dataframe(interactive=False, wrap=True)

        def on_area(area, query):
            # Domain changed: repopulate the sub-domain choices for it
            # (reset to All) and re-filter the leaderboard list.
            es = _entries()
            subs = _subareas(es, area)
            ch = _choices(es, area, ALL, query)
            first = ch[0][1] if ch else None
            tbl, md = _standings(first)
            return (gr.update(choices=subs, value=ALL),
                    gr.update(choices=ch, value=first), tbl, md)

        def on_filter(area, sub, query):
            ch = _choices(_entries(), area, sub, query)
            first = ch[0][1] if ch else None
            tbl, md = _standings(first)
            return gr.update(choices=ch, value=first), tbl, md

        def on_pick(lb_id):
            return _standings(lb_id)

        def refresh():
            # Re-read the index on every page load / click: picks up new
            # leaderboards, domains, and scores with no Space restart.
            es = _entries()
            areas = [ALL] + sorted({e["_area"] for e in es})
            subs = _subareas(es, ALL)
            ch = _choices(es, ALL, ALL, "")
            first = ch[0][1] if ch else None
            tbl, md = _standings(first)
            gen = _index().get("generated_at") or "n/a"
            note = (f"<sub>Last synced: {gen} · {len(es)} leaderboard(s) across "
                    f"{len(areas) - 1} domain(s)</sub>")
            return (gr.update(choices=areas, value=ALL),
                    gr.update(choices=subs, value=ALL), "",
                    gr.update(choices=ch, value=first), tbl, md, note)

        area_dd.change(on_area, [area_dd, search], [sub_dd, lb_dd, table, links])
        sub_dd.change(on_filter, [area_dd, sub_dd, search], [lb_dd, table, links])
        search.submit(on_filter, [area_dd, sub_dd, search], [lb_dd, table, links])
        lb_dd.change(on_pick, lb_dd, [table, links])
        refresh_btn.click(refresh, None, [area_dd, sub_dd, search, lb_dd, table, links, synced])
        demo.load(refresh, None, [area_dd, sub_dd, search, lb_dd, table, links, synced])
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
