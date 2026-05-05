# Manual verification checklist

The automated suite (`make test`) covers logic, model state, route status codes,
and a lot of side-effect plumbing. It does **not** cover what the user actually
sees: rendered Jinja templates with real CSS/JS, status-badge polling, drag-drop
interactions, the DLP-safe browser-side encoding, the matplotlib-rendered
visualizations, the depth-map heatmap canvas, etc.

Run this checklist by hand before each release. Skim the whole list first; if
something feels redundant for a given release ("only docs changed"), skip it
and note that in the release notes — but don't remove items.

## Setup

```bash
# Terminal 1
redis-server

# Terminal 2
make worker

# Terminal 3
make dev
# → http://localhost:6060
```

Use a fresh `~/.dtofbenchmarking/` if anything weird shows up; the schema
migrations live in `check_and_migrate_db()` (`app.py`) and are exercised on
startup, so a stale local DB is the most common source of false positives.

You'll also need a small fixture ZIP. The simplest way is to grab one from
`tests/fixtures/` if you've added any — otherwise zip up a `dataset_<name>/`
folder with `config/`, `tags/`, and a `metric_*/` subfolder following the
README conventions.

---

## Project flows

- [ ] `/projects` lists existing projects, "Create Project" works, the new
      project appears in the list and in the top-right project dropdown.
- [ ] **Clone**: clone an existing project. Verify the clone has all the same
      leaderboards, metrics, and metric mappings (settings carry over). The
      original is untouched.
- [ ] **Rename**: rename a project. URLs that previously used the old name
      now 404 / redirect; the new name resolves.
- [ ] **Delete**: delete a project. Its leaderboards are gone. Its datasets
      remain (datasets are global) — verify by visiting `/datasets`.
- [ ] **Selector**: switching projects via the top dropdown sets the
      `active_project_id` cookie; refreshing the page keeps you on the
      selected project.

## Dataset flows

- [ ] Web upload via the "Upload Dataset" form: success flash, redirected to
      `/datasets`, new dataset visible.
- [ ] Inner-folder rename: ZIP a folder named `wrapped_X/`; uploading it
      should result in a dataset named `wrapped_X` regardless of what was
      typed in the name field.
- [ ] **Override** checkbox: re-upload a ZIP with the same name and check
      "Override" — old dataset replaced; sample list updates.
- [ ] `/dataset/<id>` view: samples are listed, tags render as pills, custom
      fields appear in the table in priority order (name → tags → stats →
      config → histograms → images → scalars). The depth-map heatmap loads
      and shows hover values.
- [ ] Display-column toggle saves and persists across reload (note: empty
      selection saves the `__NONE__` sentinel — verify the UI handles this
      gracefully and doesn't fall back to defaults).
- [ ] Download button returns the original ZIP.
- [ ] Delete removes the row + the on-disk `uploads/datasets/<name>/` tree.

## Leaderboard flows

- [ ] Create leaderboard with one dataset attached.
- [ ] Create leaderboard with **multiple** datasets attached. Sample list on
      the comparison view spans both.
- [ ] **Import settings** from another leaderboard: target's metrics +
      visualizations + summary CSV all populate. Verify `lm_<id>` references
      in summary_metrics are remapped to the new IDs (the test for this is
      `test_import_settings_clones_metrics_with_id_remapping` — but the
      template-rendered list should show the right names).
- [ ] Edit-leaderboard tab: pooling-type dropdown shows the percentile field
      only when `pooling_type=percentile`. Saving sticks across reload.
- [ ] Per-metric `tag_filter` field accepts comma-separated tags and updates
      the metric's calculation scope (verify by recalculating and watching
      the per-sample CustomField count change).

## Metric / visualization flows

- [ ] Metrics page lists all global metrics with download/edit/delete buttons.
- [ ] **DLP-safe checkbox**: open browser devtools → Network tab; create a
      metric with the box checked. Confirm the POST payload's `python_code`
      starts with `BASE64:`. Reload the page and edit the metric; the code
      shown should be the decoded original.
- [ ] Upload metric from `.txt` file: contents become the metric body.
- [ ] Add metric to a leaderboard via the form — argument-mapping rows let
      you choose source (gt / sub / scalar) and field name. Verify the
      saved `arg_mappings` (re-open the edit dialog) reflects what you typed.
- [ ] Same as above for a Visualization (per-sample image renders in the
      comparison view; aggregated image renders alongside it).

## Submission flows

- [ ] Upload a single submission ZIP. Status badge cycles through:
      `Pending → Processing: Metric K/N (name) → Generating Visualizations →
      Processed`.
- [ ] **Upload a bulk submissions ZIP** (a ZIP of ZIPs) — each inner ZIP
      becomes its own submission.
- [ ] **Single recalculate** button on a submission row: status flips back
      to `Pending` immediately, then progresses.
- [ ] **Batch recalculate** from the leaderboard: select multiple
      submissions, click recalc. Watch the status column — only one row
      should be in `Processing` at a time (sequential, per commit `8a77b48`).
- [ ] Sample-filter inputs (include / exclude / prefix) on the recalc form
      apply to the calculation. Verify `last_sample_filter` field is set on
      the submission afterwards (visible in the API response, not directly
      in the UI).
- [ ] **Archive / unarchive** via batch action. Archived submissions are
      hidden by default; the "Show archived" toggle reveals them.
- [ ] Tag pills on a submission row open an inline editor; saving updates.
- [ ] Bulk download options:
  - [ ] Download submissions ZIP bundle
  - [ ] Download metrics CSV (one row per submission, columns per metric)
  - [ ] Download per-sample metrics CSV (one row per `(sub, sample)`)
  - [ ] Download "full bulk" (all of the above)

## Comparison view

- [ ] `/<proj>/comparison/<id>` renders with N submissions side-by-side.
      Each per-sample row shows the dataset's GT columns and each
      submission's predictions/metrics.
- [ ] Sort dropdown by metric: ascending / descending swap correctly.
- [ ] Sort by a custom scalar field: column order respects the priority
      table (`get_column_priority` — name first, scalars last).
- [ ] **Pagination preserves `compare_ids`** (regression for commit
      `40ed53a`): select 2 submissions to compare, paginate to page 2 with
      a small per-page setting; verify the URL still has `compare_ids=...`
      and a third (unselected) submission does not appear.
- [ ] Visualizations: per-sample images render inline; aggregated images
      render in the right-hand panel.
- [ ] **GT histogram** column toggles render correctly.
- [ ] Empty state: navigate with `?compare_ids=999999` (no match). Today
      this raises `UnboundLocalError: metric_labels` — see `CLAUDE.md`. Once
      fixed, the page should render with no submissions and an empty table.

## Users page

- [ ] `/users` lists all distinct git authors from datasets + submissions.
- [ ] Click a user to see their stats (count of datasets/submissions).
- [ ] **Merge**: pick two distinct authors, merge A into B. Refreshing the
      leaderboard now shows B's name everywhere A appeared.
- [ ] **Unmerge**: undo the merge — A's attribution returns.
- [ ] Avatar upload + display name editor work; reload shows the updated
      avatar in submission badges.

## App settings

- [ ] `/app-settings`: change scalar / image column widths and theme. Reload
      every other page and confirm the change took effect (settings live in
      `~/.dtofbenchmarking/settings.json`).

## Docs

- [ ] `/docs/` renders the embedded markdown index. Sub-pages
      (`/docs/<page>`) load.

## Release smoke

After deploy, hit each top-level URL once and confirm 2xx:

- `/projects`
- `/<proj>/`
- `/<proj>/leaderboard/<id>`
- `/<proj>/comparison/<id>`
- `/datasets`
- `/dataset/<id>`
- `/<proj>/metrics`
- `/<proj>/visualizations`
- `/users`
- `/app-settings`
- `/docs/`

## Known issues to keep verifying

These are pinned by tests (see `CLAUDE.md`'s "things to be careful with"
list). The tests will start failing the day they're fixed — flip the
assertions then. Until then, keep an eye on:

- Image+scalar precedence in `detect_custom_fields` (folder with both
  `s1.png` and `s1.txt`(=float) → type=image, value=float).
- Orphan `Dataset`/`Submission` rows on partial-failure ingest.
- Aggregated metrics that depend on a per-sample metric: must reference it
  by `lm_<id>`, not by `target_name`. (The UI's metric-add form generates
  the right key, but a hand-edited DB or imported leaderboard could drift.)
- `comparison_view` empty-state crash (`metric_labels` UnboundLocalError).
- Pooling-mode asymmetry: a metric configured with `pooling_type=min` or
  `max` falls through to `mean` on a fresh `process_submission` run, but
  re-aggregating with `reaggregate_submission_metrics` will use the actual
  min/max. Check after every recalc that the displayed value matches what
  you'd expect.
