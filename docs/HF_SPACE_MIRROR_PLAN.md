# HF Space Leaderboard Mirror — Execution Plan

**Goal:** a read-only HuggingFace **Space** that mirrors BenchHub's public leaderboard
standings (view anywhere) while **submissions stay on BenchHub** (the engaging action +
the data moat). Every "Submit" affordance on the Space is an outbound link back to
`runbenchhub.com`; the click is attributed so we can measure HF→BH conversion.

Status: **PLAN** (grounded against the codebase; not yet implemented).

---

## 1. Architecture (locked)

**Producer → data repo → consumer.**

```
BenchHub box                         HuggingFace                     Anywhere
┌───────────────────┐  push (commit) ┌────────────────────────┐ read ┌──────────────┐
│ tasks.push_        │ ─────────────▶ │ dataset repo            │ ───▶ │ Gradio Space │
│ standings_to_hf    │                │ benchhub/leaderboards   │      │ (read-only)  │
│ (Celery)           │                │  index.json             │      │  sortable    │
│ reads MetricResult │                │  leaderboards/<id>.json │      │  tables +    │
│ + LB/Submission    │                │  _manifest.json         │      │  "Submit on  │
│ metadata only      │                │  README.md              │      │  BenchHub →" │
└───────────────────┘                └────────────────────────┘      └──────┬───────┘
                                                                              │ outbound link
                          POST /api/submit  (token-auth, BH only)  ◀──────────┘
```

- **Exported:** already-pooled `MetricResult.value` scalars + column metadata
  (`LeaderboardMetric.target_name`/sort direction) + public submission display metadata.
- **Never exported:** per-sample GT, dataset bytes, prediction files, `arg_mappings`
  internals, private/unlisted LBs, **metric error tracebacks** (see §3 fix #1).
- Submission is impossible on the Space *by construction*: no form, no token, no
  `fetch()` to `/api/submit` (which is `@require_api_token` anyway, app.py:7385).

### Grounding facts (verified against current code)
- **No project-name URL prefix** — the `/<project_name>/` injection machinery was removed
  (app.py:1646-1651). Canonical LB URL is the flat **`/leaderboard/<id>`**.
- **No standings API** exists (`/api/leaderboard/<id>/info` is name/datasets only). Standings
  are assembled inline in `leaderboard_view` (app.py:10116) — the exporter must replicate that query.
- **No HF write code** exists today (pull-only). `HF_TOKEN` in `.env` is read/gated-pull
  scope — a push needs a **separate write-scoped token**.
- **No Celery beat / cron.** The only periodic primitive is the in-repo user-systemd timer
  (`ops/benchhub-db-backup.{service,timer}`); `make_celery` is worker-only (`backend=None`).
- `check_and_migrate_db()` **does** support `CREATE TABLE IF NOT EXISTS` (app.py:19035) —
  new tables are fine, not just ADD COLUMN.

---

## 2. Decisions (open questions, resolved)

| Question | Decision |
|---|---|
| One repo vs per-LB repos | **One** dataset repo, file-per-LB + index. Self-healing reconcile; decouples read load onto HF's CDN. |
| Repo name | **`benchhub/leaderboards`** (owning account TBD — see §6 prereq). |
| File format | **JSON** (human-diffable, content-hash idempotent). Not parquet. |
| "Public" definition | `or_(visibility=='public', owner_user_id IS NULL)` — matches the app's `visible_in_list`. **Exclude `unlisted`** (URL-only by design). |
| Mirrored (PWC) bucket | **Include, segregated + labeled** "reported by authors, not verified by BenchHub" (mirrors the on-app two-table render). One shared config constant. |
| Cadence | **Both** — event-driven (debounced) for freshness + daily full reconcile (mandatory; owns deletes). |
| Space tech | **Gradio** (server-side Python reads the repo, free `gr.Dataframe` sorting) + `robots: noindex,follow`. |
| Mirror schema change | **None** for the mirror itself. The funnel adds 2 columns + 1 table. |

---

## 3. ⚠️ Required fixes (from adversarial review — do not ship without these)

1. **Never export `MetricResult.error_message`.** It is a raw per-sample exception string from
   user metric code running over GT+predictions, so it can embed GT/sample/prediction values.
   Export `None` when the value is missing/errored. Add an assertion in the push task that no
   exported string contains `Traceback`/newlines/path fragments. Delete the recipe's
   `error_message if … else value` branch.
2. **PII scrub on submitter identity.** Author is `owner.display_name or git_author`;
   `display_name` often defaults to the email local-part (app.py:2268). Strip email-derived
   display names, never export raw `git_author`, fall back to an anonymized handle. Disclose in
   the dataset card. (Optional: a per-LB `mirror_to_hf` opt-out column if owners object.)
3. **Reconcile-with-delete is mandatory.** A public→private/unlisted flip, or a deleted/archived
   submission, must remove `leaderboards/<id>.json` (`CommitOperationDelete`) + its index entry,
   else a now-private board leaks. The daily full reconcile owns this; also hook
   `set_leaderboard_visibility` for a prompt reconcile on flip.
4. **Debounce the event trigger.** Metric add/remove + batch recalc call the scoring funnel N
   times; guard with a Redis `SET NX EX 120` per-LB key (Redis is the broker, available even with
   `backend=None`) so one LB change → one push, not N.

---

## 4. Phased build

### Phase 0 — Prerequisite (needs a human decision)
- Pick the **HF org/account** that owns `benchhub/leaderboards`; mint a **write-scoped** token.
- Add `HF_PUSH_TOKEN` + `HF_RESULTS_REPO` to `~/benchhub/.env` (distinct from read-only
  `HF_TOKEN`; never echoed). Then `sudo systemctl restart benchhub-celery` (EnvironmentFile is
  read only at process boot). **This is the one true blocker.**

### Phase 1 — Export builder
- **New** `benchhub/hf_results_export.py` (pure; no Flask-route/Celery deps):
  - `build_lb_standings(lb, MetricResult, LeaderboardMetric)` — the verified recipe: column set
    from `summary_metrics` resolving both `lm_<id>` and display-name tokens (fallback: all metrics
    by id); bulk-load `MetricResult.value`; verified+`Processed` rows; rank by first column's
    `sort_direction` (None/non-numeric to bottom).
  - `build_index(rows)`; `payload_hash(obj)` (stable `json.dumps(sort_keys=True)`→sha256).
  - **Moat-safe whitelist** enforcing fixes #1 and #2.
  - Descriptive text from `Dataset.card_description` (there is **no** `Leaderboard.description`).

### Phase 2 — Push task
- **Edit** `tasks.py`: add `push_standings_to_hf(leaderboard_id=None)` modeled on
  `materialize_leaderboard` (tasks.py:1081), inside `with app.app_context():`.
  - Enumerate public LBs (the §2 filter). Build per-LB JSON + `index.json` + `_manifest.json`
    (stored hashes). Diff against the repo's prior manifest (`hf_hub_download`, best-effort).
  - **Single atomic `create_commit`**: upload changed files + `CommitOperationDelete` for
    de-published LB ids. `create_repo(repo_id=HF_RESULTS_REPO, repo_type='dataset', exist_ok=True)`
    once at the top. Lazy-import `huggingface_hub` at the call site (matches existing convention).
  - Fully re-derivable from the DB → deploy-restart safe.

### Phase 3 — Triggers
- **Event-driven:** hook at `tasks.py:893` (right after `processing_status='Processed'`),
  debounced per fix #4, only for `kind != 'mirrored'` on a public LB.
- **Visibility flip:** hook `set_leaderboard_visibility` to enqueue a reconcile (fix #3).
- **Periodic:** **new** `ops/benchhub-hf-sync.{service,timer}` cloned from
  `ops/benchhub-db-backup.{service,timer}` (daily 04:00, `Persistent=true`, `RandomizedDelaySec`),
  deployed to `~/.config/systemd/user/`; `ExecStart` runs
  `…/.venv/bin/python -c "import tasks; tasks.push_standings_to_hf.delay()"`.
  `systemctl --user enable --now benchhub-hf-sync.timer` (linger already enabled).

### Phase 4 — The Space (separate HF repo, not in this tree)
- `app.py` (Gradio, ~150 lines): `hf_hub_download` `index.json` at startup; per-LB JSON lazily on
  selection; left = searchable LB list; right = sortable `gr.Dataframe`
  (Rank · Submission · Author · `<each metric label>` · Date) + a prominent
  **"Submit your model on BenchHub →"** button and **"View on BenchHub"** link;
  `gr.Blocks(head=…)` injects `robots: noindex,follow` + a "Mirror of BenchHub" banner;
  a Refresh button calls `hf_hub_download(..., force_download=True)`. **No upload/file/token.**
- `requirements.txt`: `gradio`, `huggingface_hub`.
- `README.md` with Space YAML frontmatter (`sdk: gradio`, `app_file: app.py`, `pinned: false`),
  body states it's a read-only mirror and points to `runbenchhub.com` as the source of truth.

### Phase 5 — Funnel + attribution
- **Deep-link:** `https://runbenchhub.com/leaderboard/<id>?utm_source=hf_space&utm_medium=referral&utm_campaign=lb_<id>`
  (GA4 already counts clicks from UTM for free, base.html:31).
- **Bridge the decoupled landing→submit gap via `SubmissionToken`** (minted by Open-in-Colab,
  app.py:10085): stamp first-touch `utm_source` on the token, copy to `Submission.inbound_source`
  at `/api/submit` (app.py:7506). Landing GET and submit POST are otherwise unrelated (the POST is
  a token-auth call from Colab/CLI with no referrer/session), so a session cookie won't survive.
- **Schema (funnel only):** add `Submission.inbound_source` + `SubmissionToken.inbound_source`
  (VARCHAR(60), nullable) + a new `InboundClick` table (id, leaderboard_id, source, user_id?,
  created_at) — the click denominator, logged best-effort in `leaderboard_view` behind a source
  whitelist + try/except. Migrate via `check_and_migrate_db` (CREATE TABLE + ADD COLUMN tuples).
  **Do not** overload `Submission.source_attribution` (it renders as PWC provenance,
  leaderboard.html:1703).
- **Client:** add optional `source=` kwarg to `benchhub/client.py` `post_submission_zip` for the
  bare-client (non-Colab) path; accept it in `api_submit_typed`.
- **Conversion metric:** `count(inbound_source=S) / count(InboundClick source=S)` — **a lower
  bound** (non-Colab CLI submits may be unattributed). Make Open-in-Colab the primary Space CTA.

### Phase 6 — Verify & roll out
- Run the builder offline with `BENCHHUB_DATA_DIR` → a **scratch copy** (never prod) and **diff
  the JSON vs what `leaderboard_view` renders** (same column order, same top row) for several LBs
  incl. one never-viewed (so the `summary_metrics` auto-migration order is checked).
- Assert no payload string contains a traceback/path fragment (fix #1).
- **Dry push to a throwaway `benchhub/leaderboards-test` repo** first; verify no GT/bytes; then
  point `HF_RESULTS_REPO` at the real repo.
- Open the Space: confirm parity, working deep-links, and **zero submission controls**.

### Phase 7 — Docs (same change)
- `docs/SELFHOST_RUNBOOK.md` — `HF_PUSH_TOKEN`/`HF_RESULTS_REPO` keys, the new timer + manual
  trigger, deploy-restart note.
- `templates/docs/*.html` — submit-flow page: read-only mirror, submit on BenchHub.
- Relevant subtree `CLAUDE.md` gotchas.

---

## 5. Data contract (`benchhub/leaderboards` dataset repo)

All fields aggregate/metadata only — no per-sample GT, no bytes, no prediction files, no tracebacks.

**`index.json`** (catalog the Space reads at startup):
```json
{
  "generated_at": "2026-06-09T04:00:00Z",
  "source": "https://runbenchhub.com",
  "leaderboards": [
    {
      "id": 42, "name": "KITTI Depth", "category": "Vision/Depth Estimation",
      "url": "https://runbenchhub.com/leaderboard/42",
      "submit_url": "https://runbenchhub.com/leaderboard/42?utm_source=hf_space&utm_medium=submit_btn",
      "datasets": [{"name": "KITTI", "description": "<Dataset.card_description>", "source_url": "..."}],
      "n_verified": 12, "n_mirrored": 3, "updated_at": "2026-06-08T11:00:00Z"
    }
  ]
}
```

**`leaderboards/<id>.json`** (per-LB standings, pulled lazily on selection):
```json
{
  "id": 42, "name": "KITTI Depth", "category": "Vision/Depth Estimation",
  "url": "https://runbenchhub.com/leaderboard/42",
  "submit_url": "https://runbenchhub.com/leaderboard/42?utm_source=hf_space&utm_medium=submit_btn",
  "columns": [
    {"metric_id": 7, "label": "RMSE", "global_metric": "rmse", "sort_direction": "lower_is_better"}
  ],
  "verified": [
    {"rank": 1, "name": "MyDepthNet", "author": "<scrubbed display name>",
     "created": "2026-06-01T09:00:00Z", "description": "...", "link": null,
     "scores": {"7": 2.13}}
  ],
  "mirrored": [
    {"name": "PaperX", "scores": {"7": 2.40},
     "source_attribution": "Papers With Code", "source_paper_url": "...", "source_external_url": "..."}
  ]
}
```
Contract rules: (1) ranking precomputed by the producer (first column `sort_direction`,
None/non-numeric to bottom); (2) `scores` keyed by `str(metric_id)` matching `columns[].metric_id`;
(3) **`error_message` is always `None`** — never export tracebacks; (4) verified rows require
`kind != 'mirrored'` AND `processing_status == 'Processed'`; (5) **never** include sample names, GT
values, prediction bytes, `arg_mappings` internals, raw `git_author`, or private/unlisted LBs.

---

## 6. File-level change list

| File | Change |
|---|---|
| `benchhub/hf_results_export.py` | **NEW** — pure standings builder + index + moat-safe whitelist + `payload_hash` |
| `tasks.py` | **EDIT** — `push_standings_to_hf` task (~after :1081); debounced event hook at :893 |
| `app.py` | **EDIT** — visibility-flip reconcile hook in `set_leaderboard_visibility`; funnel: `Submission.inbound_source` (~:1165), `SubmissionToken.inbound_source` (~:1085), `InboundClick` model (~:1126), click capture in `leaderboard_view` (~:10119), token stamp in Colab path (~:10085), submission stamp in `api_submit_typed` (~:7506); migrations in `check_and_migrate_db` (~:19035 CREATE TABLE, ~:19686 ADD COLUMN tuples) |
| `benchhub/client.py` | **EDIT** — optional `source=` kwarg on `post_submission_zip` (~:73) |
| `ops/benchhub-hf-sync.{service,timer}` | **NEW** — cloned from `benchhub-db-backup.*` (daily reconcile) |
| `~/benchhub/.env` | **EDIT ON BOX** — `HF_PUSH_TOKEN` (write-scoped) + `HF_RESULTS_REPO` (never echoed) |
| `docs/SELFHOST_RUNBOOK.md`, `templates/docs/*.html` | **EDIT** — new env keys, timer, read-only-mirror submit flow |
| HF Space repo (`app.py`, `requirements.txt`, `README.md`) | **NEW** — Gradio read-only mirror |
| HF dataset repo `benchhub/leaderboards` | **NEW** (auto-created) — `index.json`, `leaderboards/<id>.json`, `_manifest.json`, `README.md` |

---

## 7. Effort & residual risks

**~2 days:** producer ~1d, Space ~0.5d, funnel ~0.5d (verification folded in). No mirror-side
schema change; funnel adds 2 columns + 1 table (auto-migrated on `restart`).

**Not-yet-handled watch items** (decide before/during build):
- **Empty boards** — skip LBs with zero Processed verified submissions, or show "no submissions yet".
- **Payload size** — a per-LB row cap + worst-case size ceiling for very popular LBs.
- **Sync-health heartbeat** — alert if no successful sync in >48h (UI `generated_at` is not enough).
- **HF commit-rate limits** — the event path can produce one commit per LB-change; batched
  single-commit mitigates, but confirm against HF limits.
- **Data retention / erasure** — rewriting a per-LB file purges current data, but **HF git history
  persists old commits**. Given the PII point (§3 #2), keep submitter identity anonymized from day
  one and/or periodically squash the repo.
- **Legacy CustomField-averaged columns** (app.py:10351-10463) — verify no public LB surfaces
  columns via that path instead of `MetricResult`, or extend the builder.
- **Licensed descriptive text** — `Dataset.card_description` scraped from HF cards is republished;
  low risk but unexamined.

---

## 8. Open item for the owner

The single blocker to start: **which HF account/org owns `benchhub/leaderboards`, and a
write-scoped token for it.** Everything else is specified above.
