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
   signatures, pooled into the leaderboard score; outputs can chain. Author
   + test them locally with the dev kit, then upload:

   ```python
   def my_iou(gt: bh.Mask, pred: bh.Mask): ...
   bh.author.test_metric(my_iou, gt=gt_mask, pred=pred_mask)   # iterate locally
   client.create_metric("my_iou", my_iou)                      # then ship
   ```

   All metric/viz code runs in a **hardened sandbox** (network-isolated,
   read-only), never in-process. You can even register a brand-new data
   `kind` with `client.create_datatype(...)` (its `visualize()` runs in the
   same sandbox). → docs: *Writing metrics*, *Visualizations*, *Data types*.
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
  (The public beta caps total accounts; admins and existing users are
  unaffected.)
- **Visibility** per object: `public` / `unlisted` / `private` (yours
  default to private; publish when ready). Once another user depends on
  something of yours — their leaderboard binds your dataset, or someone
  submits to your leaderboard — it can no longer be made private or deleted.
- **Storage quotas**: 50 GB public + 10 GB private per user by default,
  shown with live usage on [`/settings/account`](https://runbenchhub.com/settings/account).
  Imports are cheap (preview tier); the real cost is per-leaderboard
  materialization.

A high-level diagram of the whole pipeline is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). See `/docs` for the full
guide and worked tutorials; self-hosting/ops notes are in
[`docs/SELFHOST_RUNBOOK.md`](docs/SELFHOST_RUNBOOK.md). Feature requests +
bugs → [GitHub issues](https://github.com/yakirma/BenchHub/issues).
