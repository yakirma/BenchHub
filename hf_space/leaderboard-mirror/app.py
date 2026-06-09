"""BenchHub leaderboards — read-only HuggingFace Space mirror.

Reads the standings dataset repo BenchHub pushes (see
docs/HF_SPACE_MIRROR_PLAN.md) and renders sortable per-leaderboard standings,
with domain (category) filtering + search. NO submission UI by design — every
"Submit" affordance is an outbound link to runbenchhub.com.
"""
import html as _html
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
        return pd.DataFrame(), "_No leaderboards match this domain/search._"
    data = _load(f"leaderboards/{lb_id}.json")
    cols = data.get("columns", [])
    verified = data.get("verified", [])
    headers = ["Rank", "Submission", "Author"] + [c["label"] for c in cols] + ["Date"]

    # Per-metric min/max + direction for the green heatmap.
    stats = {}
    for c in cols:
        mid = str(c["metric_id"])
        nums = [r["scores"].get(mid) for r in verified]
        nums = [v for v in nums if isinstance(v, (int, float))]
        lo, hi = (min(nums), max(nums)) if nums else (0.0, 1.0)
        higher = (c.get("sort_direction") or "higher_is_better") != "lower_is_better"
        stats[mid] = (lo, hi, (hi - lo) or 1.0, higher)

    # Cells rendered as HTML (gr.Dataframe datatype='html'): Submission links to
    # the model, metric cells carry BenchHub's green hsl(120,70%,L%) heatmap
    # (best in column = vivid, worst = pale; respects each metric's direction).
    rows = []
    for r in verified:
        name = _html.escape(str(r.get("name", "")))
        link = r.get("link")
        sub = (f"<a href='{_html.escape(link)}' target='_blank' rel='noopener'>{name}</a>"
               if link else name)
        cells = [str(r.get("rank", "")), sub, _html.escape(str(r.get("author") or ""))]
        for c in cols:
            mid = str(c["metric_id"])
            v = r["scores"].get(mid)
            if isinstance(v, (int, float)):
                lo, hi, rng, higher = stats[mid]
                norm = (v - lo) / rng
                if not higher:
                    norm = 1.0 - norm
                light = 90 - norm * 47
                cells.append(
                    f"<div style='background-color:hsl(120,70%,{light:.0f}%);color:#0b3d0b;"
                    f"padding:2px 6px;border-radius:3px;text-align:center'>{v:g}</div>")
            else:
                cells.append("")
        cells.append((r.get("created") or "")[:10])
        rows.append(cells)
    df = pd.DataFrame(rows, columns=headers)

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
    return df, links


def build():
    idx = _index()
    entries = idx.get("leaderboards", [])
    for e in entries:
        e["_area"] = (e.get("category") or "Uncategorized").split("/", 1)[0]
    areas = [ALL] + sorted({e["_area"] for e in entries})
    gen = idx.get("generated_at") or "n/a"

    def choices(area, query):
        q = (query or "").strip().lower()
        out = []
        for e in entries:
            if area and area != ALL and e["_area"] != area:
                continue
            if q and q not in f"{e['name']} {e.get('category') or ''}".lower():
                continue
            out.append((_label(e), e["id"]))   # (display, value=lb_id)
        return out

    def on_filter(area, query):
        ch = choices(area, query)
        first = ch[0][1] if ch else None
        df, links = _standings(first)
        return gr.update(choices=ch, value=first), df, links

    def on_pick(lb_id):
        return _standings(lb_id)

    with gr.Blocks(title="BenchHub Leaderboards (mirror)", head=HEAD,
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏆 BenchHub Leaderboards — read-only mirror")
        gr.Markdown(
            f"Live boards & model submission at **[runbenchhub.com]({SITE})**. "
            f"Mirrors *public leaderboard standings only* — no ground-truth data, "
            f"no submissions.\n\n<sub>Last synced: {gen} · {len(entries)} "
            f"leaderboard(s) across {len(areas) - 1} domain(s)</sub>"
        )
        with gr.Row():
            area_dd = gr.Dropdown(choices=areas, value=ALL, label="Domain", scale=1)
            search = gr.Textbox(label="Search", scale=2,
                                placeholder="filter by leaderboard name or category, then Enter…")
        init = choices(ALL, "")
        lb_dd = gr.Dropdown(choices=init, value=(init[0][1] if init else None),
                            label="Leaderboard")
        links = gr.Markdown()
        table = gr.Dataframe(interactive=False, wrap=True, datatype="html")

        area_dd.change(on_filter, [area_dd, search], [lb_dd, table, links])
        search.submit(on_filter, [area_dd, search], [lb_dd, table, links])
        lb_dd.change(on_pick, lb_dd, [table, links])
        demo.load(on_filter, [area_dd, search], [lb_dd, table, links])
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
