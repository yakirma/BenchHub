#!/usr/bin/env python3
"""Seed 5 clean HuggingFace medical datasets into a "Medical" domain.

Each becomes a public BH Dataset (owner = admin uid 2) with category
"Medical/<Task>", imported through the standard `run_hf_import` path,
capped at 10,000 samples. All map onto existing kinds (image / label /
text) — no new dtype needed (the formats were picked to be clean).

The category is pre-set on the Dataset row before import; run_hf_import
only auto-fills category when it's empty, so the Medical/* value sticks
and the datasets show up under a "Medical" card on the landing page's
browse-by-domain.

Run on the prod box (script imports the app, so the data dir must point
at the live store):

    BENCHHUB_DATA_DIR=~/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/seed_medical_hf.py [--dry-run] [--only SUBSTR]
"""
import argparse
import json
import os
import time

ADMIN_UID = 2
CAP = 10000

DATASETS = [
    {
        "repo": "Falah/Alzheimer_MRI", "name": "Falah__Alzheimer_MRI",
        "category": "Medical/Image Classification",
        "split": "train", "config": None, "sampling": "head",
        "fields": [
            {"name": "image", "source_column": "image", "kind": "image", "role": "input", "params": {}},
            {"name": "label", "source_column": "label", "kind": "label", "role": "gt", "params": {}},
        ],
    },
    {
        "repo": "marmal88/skin_cancer", "name": "marmal88__skin_cancer",
        "category": "Medical/Image Classification",
        "split": "train", "config": None, "sampling": "head",
        "fields": [
            {"name": "image", "source_column": "image", "kind": "image", "role": "input", "params": {}},
            {"name": "dx", "source_column": "dx", "kind": "label", "role": "gt", "params": {}},
            {"name": "localization", "source_column": "localization", "kind": "text", "role": "input", "params": {}},
        ],
    },
    {
        "repo": "trpakov/chest-xray-classification", "name": "trpakov__chest-xray-classification",
        "category": "Medical/Image Classification",
        "split": "train", "config": "full", "sampling": "stratified",
        "fields": [
            {"name": "image", "source_column": "image", "kind": "image", "role": "input", "params": {}},
            {"name": "labels", "source_column": "labels", "kind": "label", "role": "gt", "params": {}},
        ],
    },
    {
        "repo": "openlifescienceai/medmcqa", "name": "openlifescienceai__medmcqa",
        "category": "Medical/Question Answering",
        "split": "validation", "config": None, "sampling": "head",
        "fields": [
            {"name": "question", "source_column": "question", "kind": "text", "role": "input", "params": {}},
            {"name": "opa", "source_column": "opa", "kind": "text", "role": "input", "params": {}},
            {"name": "opb", "source_column": "opb", "kind": "text", "role": "input", "params": {}},
            {"name": "opc", "source_column": "opc", "kind": "text", "role": "input", "params": {}},
            {"name": "opd", "source_column": "opd", "kind": "text", "role": "input", "params": {}},
            {"name": "cop", "source_column": "cop", "kind": "label", "role": "gt", "params": {}},
        ],
    },
    {
        "repo": "keivalya/MedQuad-MedicalQnADataset", "name": "keivalya__MedQuad",
        "category": "Medical/Question Answering",
        "split": "train", "config": None, "sampling": "head",
        "fields": [
            {"name": "question", "source_column": "Question", "kind": "text", "role": "input", "params": {}},
            {"name": "answer", "source_column": "Answer", "kind": "text", "role": "gt", "params": {}},
            {"name": "qtype", "source_column": "qtype", "kind": "text", "role": "input", "params": {}},
        ],
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print plan, import nothing")
    ap.add_argument("--only", default=None, help="run only datasets whose repo contains SUBSTR")
    args = ap.parse_args()

    os.environ.setdefault("BENCHHUB_DATA_DIR", os.path.expanduser("~/.dtofbenchmarking"))
    import app as A
    import tasks as T

    specs = [d for d in DATASETS if (not args.only or args.only in d["repo"])]
    print(f"Seeding {len(specs)} medical dataset(s); cap={CAP}; dry_run={args.dry_run}", flush=True)

    with A.app.app_context():
        for spec in specs:
            name = spec["name"]
            print(f"\n=== {spec['repo']} -> {name}  [{spec['category']}] "
                  f"split={spec['split']} cfg={spec['config']} sampling={spec['sampling']} ===", flush=True)
            if args.dry_run:
                print("  fields:", [f"{f['name']}:{f['kind']}/{f['role']}<-{f['source_column']}" for f in spec["fields"]], flush=True)
                continue

            existing = A.Dataset.query.filter_by(name=name).first()
            if existing and existing.import_status == "ready":
                print(f"  already imported (ready) id={existing.id} — skipping", flush=True)
                continue
            if existing:
                print(f"  removing prior {existing.import_status} row id={existing.id}", flush=True)
                try:
                    A.db.session.delete(existing)
                    A.db.session.commit()
                except Exception as e:
                    A.db.session.rollback()
                    name = name + "_v2"
                    print(f"   delete failed ({e}); using name {name}", flush=True)

            ds = A.Dataset(
                name=name, owner_user_id=ADMIN_UID, visibility="public",
                category=spec["category"], import_status="importing",
                import_progress_json=json.dumps({"phase": "queued", "current": 0, "total": 0, "message": "queued"}),
            )
            A.db.session.add(ds)
            A.db.session.commit()
            did = ds.id

            t0 = time.time()
            try:
                res = T.run_hf_import.apply(kwargs=dict(
                    dataset_id=did, repo_id=spec["repo"], split=spec["split"],
                    config_name=spec.get("config"), sample_cap=CAP, shard_cap=-1,
                    sampling=spec.get("sampling", "head"), sampling_seed=42,
                    dataset_name=name, fields=spec["fields"], sample_name_from=None,
                    hf_token=os.environ.get("HF_TOKEN"), owner_user_id=ADMIN_UID,
                )).result
            except Exception as e:
                res = {"error": f"apply raised: {e}"}
            A.db.session.expire_all()
            ds = A.Dataset.query.get(did)
            dt = int(time.time() - t0)
            print(f"  {dt}s -> status={ds.import_status!r} "
                  f"err={(ds.import_error or '')[:160]!r} result={res}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
