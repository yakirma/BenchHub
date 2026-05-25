#!/usr/bin/env python3
"""Walk the BH task-domain list in order; for each domain, find the
most-downloaded HF dataset whose layout the agent-mode importer can
pair, and import it. Skips image-classification (cifar10 is already
the canonical entry for that bucket).

This is the "one-per-domain bootstrap" the user asked for after
wiping the DB to just cifar10. Each per-domain attempt is bounded
to a few candidates; once one succeeds we move on, so the script
doesn't churn through 50 unimportable repos per task.

Usage (on the prod box):
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \\
        ~/benchhub/.venv/bin/python scripts/import_one_per_domain.py \\
        --per-task 12 --max-samples 200 --i-know-what-im-doing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from import_hf_agent import (  # noqa: E402
    _detect_layout, _walk_tree, _list_top_for_task, import_one,
)


# Same order as the bulk script; image-classification dropped because
# cifar10 owns that slot.
DOMAINS = [
    ("image-segmentation",            "Image Segmentation"),
    ("semantic-segmentation",         "Semantic Segmentation"),
    ("instance-segmentation",         "Instance Segmentation"),
    ("depth-estimation",              "Depth Estimation"),
    ("monocular-depth-estimation",    "Monocular Depth"),
    ("object-detection",              "Object Detection"),
    ("image-to-image",                "Image-to-Image"),
    ("zero-shot-image-classification", "Zero-Shot Classification"),
    ("image-to-text",                 "Image Captioning"),
    ("visual-question-answering",     "Visual QA"),
    ("image-feature-extraction",      "Image Features"),
    ("mask-generation",               "Mask Generation"),
    ("pose-estimation",               "Pose Estimation"),
]


def _is_already_imported(repo_id: str) -> bool:
    from app import app, Dataset
    url = f"https://huggingface.co/datasets/{repo_id}"
    with app.app_context():
        return Dataset.query.filter(Dataset.source_url == url).first() is not None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-task", type=int, default=12,
                   help="how many top candidates to probe per task before giving up")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-know-what-im-doing", action="store_true")
    args = p.parse_args(argv)

    dd = os.environ.get("BENCHHUB_DATA_DIR") or os.path.expanduser("~/.dtofbenchmarking")
    if dd == os.path.expanduser("~/.dtofbenchmarking") and not args.i_know_what_im_doing:
        print("fatal: production data dir. pass --i-know-what-im-doing",
              file=sys.stderr)
        return 2
    os.environ.setdefault("BENCHHUB_DATA_DIR", dd)

    summary = []
    for task, label in DOMAINS:
        print(f"\n=== {label} (task_categories:{task}) ===")
        cands = _list_top_for_task(task, limit=args.per_task)
        if not cands:
            print("  (no candidates)")
            summary.append((task, "none", "no candidates"))
            continue
        picked = None
        for c in cands:
            repo_id = c.get("id") or ""
            if not repo_id:
                continue
            if _is_already_imported(repo_id):
                print(f"  - {repo_id}: already imported, skip")
                continue
            # Skip gated repos — anonymous tree-walk succeeds but raw
            # file downloads fail with 401 unless the caller's HF
            # token is in the approved list. We don't ship a token.
            gated = c.get("gated")
            if gated and gated != False:
                print(f"  - {repo_id}: gated={gated}, skip")
                continue
            print(f"  probing {repo_id}…", flush=True)
            try:
                # Spot-check: 3000 entries is enough to see paired
                # modality structure when it's there. Walking 100k
                # files for every candidate is what made the probe
                # take hours.
                files = _walk_tree(repo_id, max_files=3000, max_pages=3)
            except Exception as e:
                print(f"    tree-walk failed: {e}")
                continue
            if not files:
                print("    empty tree")
                continue
            layout = _detect_layout(files)
            if not layout:
                print(f"    no paired layout ({len(files)} files)")
                continue
            mods = {m: len({s for s, _ in v}) for m, v in layout.modalities.items()}
            shared = len(layout.sample_ids)
            print(f"    ✓ matches layout {layout.kind}: shared={shared} mods={mods}")
            picked = repo_id
            break
        if not picked:
            print("  (no candidate matched any layout)")
            summary.append((task, "none", "no layout match"))
            continue
        if args.dry_run:
            print(f"  DRY: would import {picked}")
            summary.append((task, picked, "dry-run"))
            continue
        try:
            t0 = time.monotonic()
            r = import_one(picked, max_samples=args.max_samples,
                            dry_run=False, task_label=label)
            elapsed = time.monotonic() - t0
            print(f"  → {picked}: {r}  [{elapsed:.1f}s]")
            summary.append((task, picked, r.get("status", "?")))
        except Exception as e:
            traceback.print_exc()
            summary.append((task, picked, f"FAIL {e}"))

    print("\n=== final summary ===")
    for task, repo, status in summary:
        print(f"  {task:40s} {repo:55s} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
