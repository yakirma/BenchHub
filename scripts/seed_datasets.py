"""Batch-import HuggingFace datasets and (optionally) create a
matching leaderboard with auto-assigned metrics + visualizations.

Usage:
    python scripts/seed_datasets.py path/to/config.json \
        --owner-email you@example.com [--dry-run] [--skip-existing]

Config shape (one JSON file per batch — see seed_data/*.json):

    {
        "domain": "depth",
        "datasets": [
            {
                "hf_repo_id": "sayakpaul/nyu_depth_v2",
                "dataset_name": "nyu_depth_v2",
                "sample_cap": 200,
                "revision": null,
                "auto_create_lb": true,
                "lb_name": "NYU Depth V2 (auto)"
            },
            ...
        ]
    }

The script runs in-process under the Flask app context so it reuses
the same import + auto-LB pipeline the UI exposes. Logs each step;
keeps going on per-dataset failure rather than aborting the whole
batch (HF gating, schema mismatches, etc. happen — the rest of the
batch should still land).
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    app, db,
    Dataset, Leaderboard, User,
    _hf_fetch_features, _llm_infer_mapping, _infer_mapping,
    _import_hf_auto, _auto_create_lb_with_metrics,
)


def _load_user(email):
    user = User.query.filter(db.func.lower(User.email) == email.lower()).first()
    if user is None:
        raise SystemExit(
            f"User with email {email!r} not found. Sign in via OAuth at "
            f"least once, or create the row manually."
        )
    return user


def _resolve_mapping(repo_id, hf_token):
    """Try the LLM-driven mapping first, fall back to the heuristic
    rules. Same precedence as the interactive preview page."""
    try:
        features = _hf_fetch_features(repo_id, hf_token=hf_token)
    except Exception as e:
        print(f"  [skip] features fetch failed: {e}")
        return None, {}
    if not features:
        print("  [skip] no features detected — repo may be gated or empty")
        return None, {}
    mapping = _llm_infer_mapping(features) or _infer_mapping(features)
    return mapping, features


def _seed_one(entry, owner, hf_token, *, skip_existing=False, dry_run=False):
    repo_id = entry['hf_repo_id']
    dataset_name = entry.get('dataset_name') or repo_id.replace('/', '_')
    sample_cap = int(entry.get('sample_cap') or 200)
    revision = entry.get('revision') or None
    auto_lb = bool(entry.get('auto_create_lb', True))
    lb_name = entry.get('lb_name') or f"{dataset_name}_leaderboard"

    print(f"\n=== {repo_id} → {dataset_name} (cap={sample_cap}) ===")

    if skip_existing:
        existing = Dataset.query.filter_by(name=dataset_name).first()
        if existing is not None:
            print(f"  [skip-existing] dataset {dataset_name!r} already imported "
                  f"(id={existing.id})")
            return {'status': 'skipped-existing', 'dataset_id': existing.id}

    mapping, features = _resolve_mapping(repo_id, hf_token)
    if not mapping:
        return {'status': 'failed-features'}

    print("  inferred mapping:")
    for m in mapping:
        print(f"    {m['column']:20} → {m['target_kind']:10} ({m['target_field']})")

    if dry_run:
        return {'status': 'dry-run-ok'}

    t0 = time.time()
    try:
        ok, msg, ds_id = _import_hf_auto(
            repo_id, dataset_name, mapping,
            sample_cap=sample_cap, revision=revision, hf_token=hf_token,
            owner_user_id=owner.id, features=features,
        )
    except Exception as e:
        print(f"  [error] _import_hf_auto raised: {e}")
        traceback.print_exc()
        return {'status': 'crashed', 'error': str(e)}
    dt = time.time() - t0
    if not ok:
        print(f"  [failed] {msg}")
        return {'status': 'failed-import', 'error': msg}
    print(f"  [imported] dataset_id={ds_id} ({dt:.1f}s) — {msg}")

    if not auto_lb:
        return {'status': 'imported', 'dataset_id': ds_id}

    ds = Dataset.query.get(ds_id)
    if Leaderboard.query.filter_by(name=lb_name).first() is not None:
        print(f"  [lb-skip] {lb_name!r} already exists; not auto-creating")
        return {'status': 'imported', 'dataset_id': ds_id}

    lb_ok, lb_msg, lb_id = _auto_create_lb_with_metrics(
        ds, lb_name, owner_user_id=owner.id,
    )
    if lb_ok:
        print(f"  [lb-created] {lb_msg} (id={lb_id})")
        return {
            'status': 'imported+lb', 'dataset_id': ds_id,
            'leaderboard_id': lb_id,
        }
    print(f"  [lb-failed] {lb_msg}")
    return {'status': 'imported-lb-failed', 'dataset_id': ds_id, 'error': lb_msg}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to a JSON batch config.')
    parser.add_argument('--owner-email', required=True,
                        help='Email of the User who will own these imports.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print inferred mappings without importing.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip datasets whose name already exists.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    domain = cfg.get('domain') or '<unspecified>'
    datasets = cfg.get('datasets') or []
    print(f"Loaded {len(datasets)} entries for domain {domain!r}")

    with app.app_context():
        owner = _load_user(args.owner_email)
        print(f"Owner: {owner.email} (id={owner.id})")
        # _resolve_hf_token reads g.current_user, which only exists
        # inside a request — so just read owner.hf_token directly here.
        hf_token = owner.hf_token or os.environ.get('HF_TOKEN')
        if hf_token:
            print("Using owner's saved HF token for gated-repo access.")

        results = []
        for entry in datasets:
            try:
                results.append(_seed_one(
                    entry, owner, hf_token,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                ))
            except SystemExit:
                raise
            except Exception as e:
                print(f"  [crashed] {e}")
                traceback.print_exc()
                results.append({'status': 'crashed', 'error': str(e)})

    print("\n=== Summary ===")
    by_status = {}
    for r in results:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    for status, count in sorted(by_status.items()):
        print(f"  {status:25} {count}")


if __name__ == '__main__':
    main()
