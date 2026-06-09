"""BenchHub leaderboards — read-only HuggingFace Space mirror.

Reads the standings dataset repo BenchHub pushes (see
docs/HF_SPACE_MIRROR_PLAN.md in the BenchHub repo) and renders sortable
per-leaderboard standings. There is NO submission UI here by design — every
"Submit" affordance is an outbound link to runbenchhub.com (submissions are a
token-auth action that stays on BenchHub). The Space ships no token and no
upload form, so submitting here is impossible by construction.

Deploy: copy this dir into a HuggingFace Space (SDK: gradio). It reads the
PUBLIC dataset repo anonymously — no token needed.
"""
import json
import os

import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = os.environ.get("HF_RESULTS_REPO", "benchhub/leaderboards")
SITE = "https://runbenchhub.com"
# Keep the mirror out of search indices so it can't outrank / duplicate-content
# the source of truth on runbenchhub.com.
HEAD = '<meta name="robots" content="noindex,follow">'


def _load(filename, force=False):
    path = hf_hub_download(
        repo_id=DATASET_REPO, filename=filename, repo_type="dataset",
        force_download=force,
    )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _index(force=False):
    try:
        return _load("index.json", force=force)
    except Exception:
        return {"generated_at": None, "leaderboards": []}


def _label(entry):
    cat = entry.get("category") or "Uncategorized"
    return f"[{cat}] {entry['name']} · {entry.get('n_verified', 0)} subs"


def _standings(lb_id):
    data = _load(f"leaderboards/{lb_id}.json")
    cols = data.get("columns", [])
    headers = ["Rank", "Submission", "Author"] + [c["label"] for c in cols] + ["Date"]
    rows = []
    for r in data.get("verified", []):
        scores = [r["scores"].get(str(c["metric_id"])) for c in cols]
        rows.append([r.get("rank"), r["name"], r.get("author"), *scores,
                     (r.get("created") or "")[:10]])
    df = pd.DataFrame(rows, columns=headers)
    submit = data.get("submit_url", f"{SITE}/leaderboard/{lb_id}")
    view = data.get("url", f"{SITE}/leaderboard/{lb_id}")
    links = (
        f"### {data.get('name', '')}\n\n"
        f"<a href='{submit}' target='_blank' rel='noopener'>"
        f"🚀 Submit your model on BenchHub →</a>  ·  "
        f"<a href='{view}' target='_blank' rel='noopener'>View on BenchHub</a>\n\n"
        f"<sub>Read-only mirror — submissions run on BenchHub "
        f"(sign-in + benchhub-client). No upload here by design.</sub>"
    )
    return df, links


def build():
    idx = _index()
    entries = idx.get("leaderboards", [])
    label_to_id = {_label(e): e["id"] for e in entries}
    labels = list(label_to_id)
    gen = idx.get("generated_at") or "n/a"

    with gr.Blocks(title="BenchHub Leaderboards (mirror)", head=HEAD) as demo:
        gr.Markdown("# 🏆 BenchHub Leaderboards — read-only mirror")
        gr.Markdown(
            f"Live boards & model submission at **[runbenchhub.com]({SITE})**. "
            f"This Space mirrors *public leaderboard standings only* — no "
            f"ground-truth data, no submissions.\n\n"
            f"<sub>Last synced: {gen} · {len(labels)} leaderboard(s)</sub>"
        )
        dd = gr.Dropdown(choices=labels, value=(labels[0] if labels else None),
                         label="Leaderboard")
        links = gr.Markdown()
        table = gr.Dataframe(interactive=False, wrap=True)

        def on_select(label):
            if not label or label not in label_to_id:
                return gr.update(), gr.update()
            return _standings(label_to_id[label])

        dd.change(on_select, dd, [table, links])
        if labels:
            demo.load(lambda: on_select(labels[0]), None, [table, links])
    return demo


if __name__ == "__main__":
    build().launch()
