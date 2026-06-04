# BenchHub user guide

> The canonical, always-current user documentation is **in-app at
> [`/docs`](https://runbenchhub.com/docs)** (source under `templates/docs/`).
> This file is a short orientation that mirrors it.

BenchHub is a benchmarking platform: cache a dataset, declare what a
prediction looks like, submit your model's outputs, and see how it ranks —
with a sample-by-sample comparison. Live at **https://runbenchhub.com**.

## The one big idea: a typed contract

Every piece of data — a dataset field, a leaderboard input, a prediction —
has a declared **kind** (`image`, `mask`, `depth`, `audio`, `label`,
`label_list`, `bboxes`, `scalar`, `text`, `json`, `sequence`,
`coco_detections`). Because the dataset, the `benchhub-client`, and the
metric engine all agree on kinds, data is decoded, scored, and rendered
without guessing. Kinds are defined once in `benchhub/types.py` and listed
live at [`/supported_types`](https://runbenchhub.com/supported_types).

## The end-to-end flow

1. **Import a dataset** — from any HuggingFace repo. The single *Fetch &
   preview* button tries the **tabular** importer (Croissant/parquet) and
   falls back automatically to the **file-tree mapper** for repos of paired
   files, packed archives (`.npz`/`.h5`/`.zip`/`.tar`), or video clips.
   Datasets cache as a small **preview tier**. → docs: *Importing data*.
2. **Create a leaderboard** — bind the dataset, mark fields `input` vs `gt`,
   declare the prediction contract, and choose a **sample subset** to
   materialize at full resolution (head / random / stratified). → docs:
   *Leaderboards*.
3. **Add metrics & visualizations** — Python functions with typed
   signatures, pooled into the leaderboard score; outputs can chain. → docs:
   *Writing metrics*, *Visualizations*.
4. **Submit & compare** — with `benchhub-client`:

   ```python
   import benchhub as bh
   client = bh.Client(token="bh_xxx")          # token from /settings/api_tokens
   sub = client.submission(LB_ID, name="my-model")
   for name, inputs in client.iter_samples(LB_ID):
       pred = my_model(inputs["image"].array)
       sub.predict(name, label_pred=bh.Label(int(pred)))
   sub.submit()
   ```

   → docs: *Submit predictions*, *API & client reference*, *Tutorials*.

## Accounts, visibility, quotas

- **Sign in** with GitHub, Google, or a one-time email code — no passwords.
- **Visibility** per object: `public` / `unlisted` / `private` (yours
  default to private; publish when ready).
- **Storage quotas**: 100 GB public + 10 GB private per user by default.
  Imports are cheap (preview tier); the real cost is per-leaderboard
  materialization.

See `/docs` for the full guide and worked tutorials. Operational/deployment
notes for self-hosting are in [`docs/SELFHOST_RUNBOOK.md`](docs/SELFHOST_RUNBOOK.md).
