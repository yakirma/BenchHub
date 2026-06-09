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
        return pd.DataFrame(), "_No leaderboards match this domain/search._"
    data = _load(f"leaderboards/{lb_id}.json")
    cols = data.get("columns", [])
    headers = ["Rank", "Submission", "Author"] + [c["label"] for c in cols] + ["Date"]
    rows = []
    for r in data.get("verified", []):
        scores = [r["scores"].get(str(c["metric_id"])) for c in cols]
        rows.append([r.get("rank"), r["name"], r.get("author"), *scores,
                     (r.get("created") or "")[:10]])
    df = pd.DataFrame(rows, columns=headers)

    # BenchHub green heatmap: color each metric cell hsl(120,70%,L%) where the
    # best value in the column is vivid green (low L) and the worst is pale,
    # respecting the metric's sort_direction (gradio renders a pandas Styler).
    metric_labels = [c["label"] for c in cols]
    dir_by = {c["label"]: (c.get("sort_direction") or "higher_is_better") for c in cols}

    def _green(series):
        nums = pd.to_numeric(series, errors="coerce")
        lo, hi = nums.min(), nums.max()
        rng = (hi - lo) or 1.0
        higher_better = dir_by.get(series.name, "higher_is_better") != "lower_is_better"
        styles = []
        for v in nums:
            if pd.isna(v):
                styles.append("")
                continue
            norm = (v - lo) / rng
            if not higher_better:
                norm = 1.0 - norm                  # 1.0 == best
            light = 90 - norm * 47                 # best ~43% (vivid), worst ~90% (pale)
            styles.append(f"background-color: hsl(120, 70%, {light:.0f}%); color: #0b3d0b;")
        return styles

    value = df
    if metric_labels and not df.empty:
        try:
            value = df.style.apply(_green, subset=metric_labels, axis=0)
        except Exception:
            value = df

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
    return value, links


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
        table = gr.Dataframe(interactive=False, wrap=True)

        area_dd.change(on_filter, [area_dd, search], [lb_dd, table, links])
        search.submit(on_filter, [area_dd, search], [lb_dd, table, links])
        lb_dd.change(on_pick, lb_dd, [table, links])
        demo.load(on_filter, [area_dd, search], [lb_dd, table, links])
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
