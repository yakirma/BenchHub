#!/usr/bin/env python3
"""Long-running bulk-importer for popular HuggingFace vision datasets.

Walks a hand-picked list of HF task_categories — depth, segmentation,
object detection, image classification, super-resolution, normal
estimation, captioning, pose estimation, VQA, and a couple of
"surprise me" extras — pulls the most-downloaded datasets from each,
and tries to import each one into the local BenchHub DB via the same
typed-manifest pipeline the admin form uses.

For each candidate:
  - skip if a BH Dataset already points at that repo (source_url match)
  - skip if Croissant isn't available
  - skip if no test / validation / train split could be resolved
  - skip if the chosen split is bigger than `--max-bytes`
  - skip if Croissant doesn't expose enough mapped fields (need >= 2)
  - materialize a bounded sample cap into a staging dir, then import
  - auto-fill Dataset.category from the repo's task tags

Designed to run for hours. Every per-dataset failure is caught + logged
and the loop moves to the next; SIGINT exits cleanly between datasets.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/bulk_import_hf_vision.py \\
        --sample-cap 50 --per-task 15 --max-bytes 524288000

Run with `--dry-run` first to see what it would import without
touching the DB or hitting the network beyond the listing step.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Make `from app import ...` work no matter what cwd the user
# invoked us from — the repo root sits one level up from `scripts/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing `app` binds SQLAlchemy to whatever BENCHHUB_DATA_DIR
# resolves to at import time. Refuse to run against the prod DB
# unless --i-know-what-im-doing is set, so a typo doesn't repeat
# the 2026-05-24 wipe.
DEFAULT_DATA_DIR = os.path.expanduser("~/.dtofbenchmarking")


def _resolve_data_dir(args):
    """Decide which data dir we're writing to + bail loudly if the
    caller hasn't acknowledged the prod path."""
    dd = os.environ.get("BENCHHUB_DATA_DIR") or DEFAULT_DATA_DIR
    if dd == DEFAULT_DATA_DIR and not args.i_know_what_im_doing:
        print(
            "fatal: this script will write to the production data dir "
            f"({dd}). Re-run with --i-know-what-im-doing to confirm, or "
            "set BENCHHUB_DATA_DIR to a sandbox path first.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.environ.setdefault("BENCHHUB_DATA_DIR", dd)
    return dd


# ---------------------------------------------------------------------------
# Target task vocabulary
# ---------------------------------------------------------------------------

# Each entry: (HF task_categories value, friendly label, BH gt-kind
# we expect the dataset to carry). Listed in roughly "most common
# benchmark" → "more niche" order; the script shuffles tasks before
# walking them so a run interrupted mid-way still gets a mix.
_TARGET_TASKS = [
    ("image-classification",          "Image Classification",   "label"),
    ("image-segmentation",            "Image Segmentation",     "mask"),
    ("semantic-segmentation",         "Semantic Segmentation",  "mask"),
    ("instance-segmentation",         "Instance Segmentation",  "mask"),
    ("depth-estimation",              "Depth Estimation",       "depth"),
    ("monocular-depth-estimation",    "Monocular Depth",        "depth"),
    ("object-detection",              "Object Detection",       "bboxes"),
    ("image-to-image",                "Image-to-Image",         "image"),
    ("zero-shot-image-classification", "Zero-Shot Classification", "label"),
    ("image-to-text",                 "Image Captioning",       "text"),
    ("visual-question-answering",     "Visual QA",              "text"),
    ("image-feature-extraction",      "Image Features",         "json"),
    ("mask-generation",               "Mask Generation",        "mask"),
    ("pose-estimation",               "Pose Estimation",        "json"),
    # The HF task list has no normal-estimation slot; surface-normal
    # datasets usually self-tag as `task_categories:image-to-image`
    # and put 'normal' in the name. The downstream Croissant parse
    # will figure out the field kinds either way — they get picked
    # up via the image-to-image bucket.
]


_HF_API_BASE = "https://huggingface.co/api/datasets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class CandidateOutcome:
    """One per attempted repo. Drives the end-of-run summary."""
    repo_id: str
    task: str
    status: str          # 'imported' | 'skipped' | 'failed'
    reason: str = ""
    samples: int = 0
    bytes_on_disk: int = 0
    elapsed_s: float = 0.0


def _fetch_json(url: str, *, timeout: int = 20):
    """Tiny GET helper. Returns parsed JSON or raises."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "benchhub-bulk-import/0.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _top_datasets_for_task(task: str, *, limit: int) -> list[dict]:
    """Top-downloaded public datasets tagged with `task_categories:<task>`."""
    params = urllib.parse.urlencode({
        "filter": f"task_categories:{task}",
        "sort": "downloads",
        "direction": "-1",
        "limit": int(limit),
    })
    try:
        rows = _fetch_json(f"{_HF_API_BASE}?{params}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  [warn] HF list call failed for {task}: {e}", file=sys.stderr)
        return []
    return [r for r in rows if isinstance(r, dict)]


def _pick_split(repo_id: str) -> str | None:
    """Prefer test → validation → val → dev → train. Returns None
    when row-counts couldn't be fetched."""
    from benchhub.hf_search import fetch_split_row_counts
    counts = fetch_split_row_counts(repo_id) or {}
    for cand in ("test", "validation", "val", "dev", "train"):
        if counts.get(cand):
            return cand
    return None


def _is_already_imported(repo_id: str) -> bool:
    from app import Dataset
    url = f"https://huggingface.co/datasets/{repo_id}"
    return Dataset.query.filter(
        (Dataset.source_url == url) | (Dataset.source_url == url.rstrip("/"))
    ).first() is not None


def _admin_user_id() -> int | None:
    from app import User
    u = User.query.filter_by(is_admin=True).order_by(User.id).first()
    return u.id if u else None


def _pick_sample_name_column(fields: list[dict]) -> str | None:
    """If the schema has a text field with a 'filename' / 'image_id'
    style name, use it as the on-disk sample name source — keeps the
    layout introspectable. Otherwise enumeration is fine."""
    PREFERRED = ("file_name", "filename", "image_id", "id", "image",
                 "name", "uuid")
    by_name = {f["name"]: f for f in fields}
    for cand in PREFERRED:
        if cand in by_name and by_name[cand]["kind"] == "text":
            return by_name[cand].get("source_column") or cand
    return None


# ---------------------------------------------------------------------------
# Per-dataset work
# ---------------------------------------------------------------------------


def _try_import_one(
    repo_id: str,
    *,
    task: str,
    sample_cap: int,
    max_bytes: int,
    owner_user_id: int,
    dry_run: bool,
) -> CandidateOutcome:
    """Run the criteria checks + materialize+import for a single
    repo. Always returns a CandidateOutcome; never raises."""
    from app import (
        Dataset, DatasetField, Sample, CustomField, app, db,
        _hf_tags_to_category,
    )
    from benchhub.hf_croissant import (
        fetch_croissant, parse_croissant, CroissantFetchError,
    )
    from benchhub.hf_search import (
        fetch_split_byte_sizes, fetch_dataset_card,
    )

    t0 = time.monotonic()

    def outcome(status: str, reason: str = "", **kw) -> CandidateOutcome:
        return CandidateOutcome(
            repo_id=repo_id, task=task, status=status, reason=reason,
            elapsed_s=time.monotonic() - t0, **kw,
        )

    if _is_already_imported(repo_id):
        return outcome("skipped", "already imported")

    try:
        doc = fetch_croissant(repo_id, timeout=20)
    except CroissantFetchError as e:
        return outcome("skipped", f"no croissant: {e}")
    except Exception as e:
        return outcome("skipped", f"croissant fetch err: {e}")

    try:
        schema = parse_croissant(doc)
    except Exception as e:
        return outcome("skipped", f"croissant parse err: {e}")

    if not schema.fields or len(schema.fields) < 2:
        return outcome("skipped", "schema has < 2 fields")

    split = _pick_split(repo_id)
    if not split:
        return outcome("skipped", "no recognised split")

    byte_sizes = fetch_split_byte_sizes(repo_id) or {}
    chosen_bytes = byte_sizes.get(split) or 0
    if max_bytes > 0 and chosen_bytes and chosen_bytes > max_bytes:
        return outcome(
            "skipped",
            f"{chosen_bytes/1e6:.0f}MB > {max_bytes/1e6:.0f}MB cap",
        )

    # Materialize the schema into the shape the typed-import pipeline
    # consumes. Role defaults to 'gt' for every field; LBs override
    # via field_roles_json after the fact.
    fields_payload = [{
        "name": f.name,
        "source_column": f.source_column or f.name,
        "kind": f.kind,
        "role": "gt",
        "params": {},
    } for f in schema.fields]

    sample_name_from = _pick_sample_name_column(fields_payload)
    dataset_name = repo_id.replace("/", "__")

    if dry_run:
        return outcome(
            "imported",
            f"DRY: split={split} fields={len(fields_payload)} "
            f"bytes={chosen_bytes}",
        )

    # Stand the row up first so it appears in /datasets right away;
    # mirrors the real /admin/import_from_hf/commit flow.
    try:
        with app.app_context():
            ds_row = Dataset(
                name=dataset_name,
                owner_user_id=owner_user_id,
                visibility="public",
                import_status="importing",
                import_progress_json=json.dumps({
                    "phase": "starting", "current": 0, "total": 0,
                    "message": f"bulk-import: {repo_id}",
                }),
            )
            db.session.add(ds_row)
            db.session.commit()
            dataset_id = ds_row.id
    except Exception as e:
        return outcome("failed", f"could not create Dataset row: {e}")

    from benchhub.hf_materialize import materialize_hf_to_typed_dir
    from benchhub.manifest import import_typed_dataset

    try:
        with tempfile.TemporaryDirectory(prefix="bh_bulk_") as staging:
            mat = materialize_hf_to_typed_dir(
                repo_id=repo_id,
                split=split,
                sample_cap=sample_cap,
                staging_dir=staging,
                dataset_name=dataset_name,
                fields=fields_payload,
                hf_token=None,
                sampling="head",      # deterministic for bulk runs
                seed=42,
                sample_name_from=sample_name_from,
            )

            staged_bytes = sum(
                os.path.getsize(os.path.join(d, fn))
                for d, _, files in os.walk(staging) for fn in files
            )

            with app.app_context():
                existing = Dataset.query.get(dataset_id)
                _, summary = import_typed_dataset(
                    staging,
                    db_session=db.session,
                    Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField,
                    DatasetField=DatasetField,
                    upload_folder=app.config["UPLOAD_FOLDER"],
                    existing_dataset=existing,
                )
                existing.source_kind = "hf"
                existing.source_url = f"https://huggingface.co/datasets/{repo_id}"
                existing.source_metadata = json.dumps({
                    "repo_id": repo_id,
                    "split": mat.get("split"),
                    "sample_cap": sample_cap,
                    "sampling": "head",
                    "sampling_seed": 42,
                    "total_rows_in_split": mat.get("total_rows_in_split"),
                    "samples_imported": mat.get("samples"),
                    "rows_written": mat.get("rows_written"),
                    "rows_skipped": mat.get("rows_skipped"),
                })
                existing.import_status = "ready"
                existing.import_error = None
                existing.import_progress_json = json.dumps({
                    "phase": "done",
                    "current": summary["samples"],
                    "total": summary["samples"],
                    "message": f"bulk-imported {summary['samples']} sample(s).",
                })
                # Auto-fill category from HF tags. Same logic the
                # async import task uses; we just inline it here so
                # this script doesn't need a running Celery worker.
                try:
                    card = fetch_dataset_card(repo_id)
                    if card:
                        cat = _hf_tags_to_category(card.get("tags") or [])
                        if cat:
                            existing.category = cat
                except Exception:
                    pass
                db.session.commit()
                return outcome(
                    "imported",
                    f"split={split}",
                    samples=summary.get("samples", 0),
                    bytes_on_disk=staged_bytes,
                )
    except Exception as e:
        # Roll the placeholder Dataset row over to failed so it shows
        # up in the admin's failed-imports list rather than silently
        # vanishing.
        try:
            with app.app_context():
                row = Dataset.query.get(dataset_id)
                if row is not None:
                    row.import_status = "failed"
                    row.import_error = str(e)
                    db.session.commit()
        except Exception:
            pass
        return outcome("failed", f"materialize/import: {e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_STOP = False


def _install_sigint():
    """SIGINT sets a flag so the loop exits cleanly between
    datasets — half-importing a row on Ctrl-C would orphan files."""
    def _handler(signum, frame):
        global _STOP
        _STOP = True
        print("\n[stop] SIGINT received — finishing current dataset, "
              "then exiting…", file=sys.stderr)
    signal.signal(signal.SIGINT, _handler)


def _format_summary(outcomes: list[CandidateOutcome]) -> str:
    counts = {"imported": 0, "skipped": 0, "failed": 0}
    total_samples = 0
    total_bytes = 0
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
        if o.status == "imported":
            total_samples += o.samples
            total_bytes += o.bytes_on_disk
    return (
        f"\n=== summary ({len(outcomes)} attempts) ===\n"
        f"  imported : {counts['imported']:>4} "
        f"({total_samples:,} samples, {total_bytes/1e6:,.0f} MB on disk)\n"
        f"  skipped  : {counts['skipped']:>4}\n"
        f"  failed   : {counts['failed']:>4}\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-task", type=int, default=15,
                   help="how many top-downloaded datasets per task to consider")
    p.add_argument("--max-datasets", type=int, default=0,
                   help="global cap on dataset attempts (0 = no cap)")
    p.add_argument("--sample-cap", type=int, default=50,
                   help="per-dataset sample cap for materialise")
    p.add_argument("--max-bytes", type=int, default=500 * 1024 * 1024,
                   help="skip datasets whose chosen split parquet exceeds this (bytes)")
    p.add_argument("--sleep-between", type=float, default=1.5,
                   help="seconds to sleep between attempts (HF politeness)")
    p.add_argument("--shuffle-tasks", action="store_true",
                   help="walk task categories in random order")
    p.add_argument("--shuffle-within-task", action="store_true",
                   help="shuffle the per-task candidate order")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log", type=str, default=None,
                   help="append per-dataset outcomes to this JSONL file")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be imported without touching the DB or downloading data")
    p.add_argument("--i-know-what-im-doing", action="store_true",
                   help="confirm writes to the production data dir")
    args = p.parse_args(argv)

    _resolve_data_dir(args)
    rng = random.Random(args.seed)

    # Resolve the admin user up front so we don't import 50 datasets
    # and only then realise we have no owner to attribute them to.
    # db.create_all is idempotent on the prod DB and useful when
    # we're pointed at a fresh sandbox path (dry-run testing).
    from app import app, db  # noqa: F401 — sets up SQLAlchemy bindings
    with app.app_context():
        db.create_all()
        owner = _admin_user_id()
    if owner is None:
        print("fatal: no admin user found — set User.is_admin on at "
              "least one user before running.", file=sys.stderr)
        return 1

    _install_sigint()
    global _STOP

    tasks = list(_TARGET_TASKS)
    if args.shuffle_tasks:
        rng.shuffle(tasks)

    log_fh = open(args.log, "a") if args.log else None
    outcomes: list[CandidateOutcome] = []
    attempts = 0
    seen_repos: set[str] = set()

    print(f"[start] data_dir={os.environ.get('BENCHHUB_DATA_DIR')}\n"
          f"        sample_cap={args.sample_cap}, "
          f"per_task={args.per_task}, "
          f"max_bytes={args.max_bytes/1e6:.0f}MB, "
          f"dry_run={args.dry_run}")

    # Long-lived app context so the per-dataset checks
    # (`_is_already_imported`, etc.) can hit SQLAlchemy without
    # spinning a fresh context per call. The per-attempt
    # materialize+import inside `_try_import_one` opens its own
    # nested context for the write phase too — harmless.
    with app.app_context():
        for task_key, task_label, expected_kind in tasks:
            if _STOP:
                break
            print(f"\n--- task: {task_label} (task_categories:{task_key}) ---")
            candidates = _top_datasets_for_task(task_key, limit=args.per_task)
            if args.shuffle_within_task:
                rng.shuffle(candidates)
            for entry in candidates:
                if _STOP:
                    break
                repo_id = entry.get("id") or ""
                if not repo_id or repo_id in seen_repos:
                    continue
                seen_repos.add(repo_id)
                if args.max_datasets and attempts >= args.max_datasets:
                    _STOP = True
                    break
                attempts += 1
                print(f"  [{attempts:03d}] {repo_id} … ", end="", flush=True)
                o = _try_import_one(
                    repo_id,
                    task=task_key,
                    sample_cap=args.sample_cap,
                    max_bytes=args.max_bytes,
                    owner_user_id=owner,
                    dry_run=args.dry_run,
                )
                outcomes.append(o)
                tag = {"imported": "OK ", "skipped": "skip", "failed": "FAIL"}[o.status]
                extra = ""
                if o.status == "imported" and o.samples:
                    extra = f" ({o.samples} samples)"
                print(f"{tag} {o.reason}{extra}  [{o.elapsed_s:.1f}s]")
                if log_fh:
                    log_fh.write(json.dumps({
                        "ts": time.time(),
                        "repo_id": o.repo_id,
                        "task": o.task,
                        "status": o.status,
                        "reason": o.reason,
                        "samples": o.samples,
                        "bytes_on_disk": o.bytes_on_disk,
                        "elapsed_s": o.elapsed_s,
                    }) + "\n")
                    log_fh.flush()
                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)

    if log_fh:
        log_fh.close()
    print(_format_summary(outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
