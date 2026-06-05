# BenchHub architecture

A one-page map of how data moves through BenchHub. The single organizing
idea is the **typed contract**: every field — in a dataset, a leaderboard
input, or a prediction — declares a **kind** (`image`, `mask`, `depth`,
`audio`, `label`, `label_list`, `bboxes`, `scalar`, `text`, `json`,
`sequence`, `coco_detections`, plus any user-**registered** kinds). Because
the dataset, the `benchhub-client`, and the metric engine all agree on
kinds, data is decoded, scored, and rendered without guessing.

![BenchHub pipeline](diagrams/architecture.svg)

> The diagram is editable: open
> [`diagrams/architecture.drawio`](diagrams/architecture.drawio) in
> [diagrams.net](https://app.diagrams.net) (or the VS Code Draw.io
> extension). The `.svg` above is the rendered copy embedded in the docs.

## The flow

1. **Import** — paste a HuggingFace repo id. One button tries the
   **tabular** importer (Croissant / parquet schema); if the repo is a tree
   of files it falls back to the **file-tree mapper** (paired files, packed
   archives, video clips). The dataset is cached as a cheap **preview tier**.
2. **Dataset** — typed fields stored preview-resolution. Cheap to keep around.
3. **Leaderboard** — bind one or more datasets, mark fields `input` vs `gt`,
   declare the prediction contract, and **materialize** a chosen sample
   subset at full resolution (head / random / stratified). Materialized
   bytes are charged to the LB owner's public quota.
4. **Client** — `benchhub-client` pulls the LB's inputs
   (`iter_samples`, decoded typed instances), runs your model, stages typed
   predictions (`predict`), and `submit`s a ZIP the server validates against
   the contract.
5. **Scoring** — the server enqueues a Celery task; the **metric engine**
   builds each sample's typed context, topologically sorts the bound metrics
   (so one can chain on another), pools per-sample values into the
   leaderboard score, and caches visualizations.
6. **Sandbox** — **all** user-supplied code (metrics, visualizations, and a
   registered data type's `visualize(blob, params)`) runs in a hardened,
   short-lived container — `--network=none`, `--read-only`, memory/CPU
   capped — never in-process on the server. Typed args cross the boundary as
   a portable JSON form and are rebuilt inside the container from the
   vendored `benchhub` package.
7. **Ranking + comparison** — results land as `MetricResult` rows; the
   leaderboard table ranks submissions and the comparison view shows
   predictions vs ground truth sample-by-sample.

## Where the code lives

| Concern | Module |
|---|---|
| Typed kinds (source of truth) | `benchhub/types.py` (`DTYPES`) + DB `DataTypeDef` for registered kinds |
| Client + dev kit | `benchhub/client.py`, `benchhub/author.py` |
| Importers | `benchhub/manifest.py` (typed), `benchhub/file_tree_import.py`, HF helpers |
| Storage tiers | `benchhub/preview.py`, `benchhub/lb_materialize.py` |
| Metric/viz execution + sandbox | `metric_engine.py`, `runner/` (image + harness) |
| Web app, routes, models, migrations | `app.py` |
| Async tasks | `tasks.py` |

For deeper, change-by-change dev notes see [`../CLAUDE.md`](../CLAUDE.md)
and the dated session notes in this folder.
