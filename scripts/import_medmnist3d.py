#!/usr/bin/env python3
"""Import a MedMNIST **3D** variant as a typed `volume` dataset.

Generalises scripts/seed_volume_medical.py (which was hardcoded to
NoduleMNIST3D) to every 28^3 MedMNIST v2 3D set. The `volume` DataTypeDef
(npz holding one 3D ndarray; sandboxed cube `visualize`) is registered by
seed_volume_medical.py and reused here — this script only stages the
volume + label fields and imports them at the preview tier.

Usage (prod box):
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking PYTHONPATH=/home/ymatri/Git/BenchHub \\
        ~/benchhub/.venv/bin/python scripts/import_medmnist3d.py \\
        --variant organmnist3d --split test --limit 1000

`--variant` ∈ organmnist3d, adrenalmnist3d, fracturemnist3d,
vesselmnist3d, synapsemnist3d (nodulemnist3d already seeded).
"""
import argparse
import io
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ADMIN_UID = 2
ZENODO_RECORD = "6496656"          # MedMNIST v2 (28px) — holds all 3D sets
CATEGORY = "Medical/Volume Classification"

# Official MedMNIST v2 label names per 3D variant (index order).
VARIANTS = {
    "organmnist3d": {
        "pretty": "OrganMNIST3D",
        "classes": ["liver", "kidney-right", "kidney-left", "femur-right",
                    "femur-left", "bladder", "heart", "lung-right",
                    "lung-left", "spleen", "pancreas"],
        "desc": "Abdominal CT — 11-class organ classification (3D).",
    },
    "adrenalmnist3d": {
        "pretty": "AdrenalMNIST3D",
        "classes": ["normal", "mass"],
        "desc": "Adrenal CT — normal vs. mass (3D shape).",
    },
    "fracturemnist3d": {
        "pretty": "FractureMNIST3D",
        "classes": ["buckle rib fracture", "nondisplaced rib fracture",
                    "displaced rib fracture"],
        "desc": "Rib CT — 3-class rib-fracture classification (3D).",
    },
    "vesselmnist3d": {
        "pretty": "VesselMNIST3D",
        "classes": ["vessel", "aneurysm"],
        "desc": "Brain MRA — healthy vessel vs. aneurysm (3D).",
    },
    "synapsemnist3d": {
        "pretty": "SynapseMNIST3D",
        "classes": ["inhibitory", "excitatory"],
        "desc": "Electron-microscopy — inhibitory vs. excitatory synapse (3D).",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=1000, help="cap samples (0 = all)")
    args = ap.parse_args()

    os.environ.setdefault("BENCHHUB_DATA_DIR", os.path.expanduser("~/.dtofbenchmarking"))
    import numpy as np
    import app as A
    from benchhub.manifest import import_typed_dataset

    spec = VARIANTS[args.variant]
    npz_file = f"{args.variant}.npz"
    dataset_name = f"MedMNIST__{spec['pretty']}"

    with A.app.app_context():
        # The shared 'volume' dtype must already exist (seed_volume_medical.py).
        dt = A.DataTypeDef.query.filter_by(name="volume").first()
        if dt is None:
            print("fatal: 'volume' dtype not registered — run "
                  "seed_volume_medical.py first.", file=sys.stderr)
            return 1

        existing = A.Dataset.query.filter_by(name=dataset_name).first()
        if existing and existing.import_status == "ready":
            print(f"already imported (id={existing.id}) — skipping")
            return 0

        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{npz_file}/content"
        print(f"downloading {url}", flush=True)
        raw = urllib.request.urlopen(url, timeout=300).read()
        z = np.load(io.BytesIO(raw))
        images = z[f"{args.split}_images"]
        labels = z[f"{args.split}_labels"]
        n = len(images)
        if args.limit:
            n = min(n, args.limit)
        print(f"{npz_file}: {args.split} has {len(images)} volumes "
              f"{images.shape[1:]}, importing {n}", flush=True)

        if existing is None:
            existing = A.Dataset(
                name=dataset_name, owner_user_id=ADMIN_UID, visibility="public",
                category=CATEGORY, import_status="importing",
            )
            A.db.session.add(existing)
            A.db.session.commit()

        with tempfile.TemporaryDirectory(prefix="bh_vol3d_") as staging:
            root = Path(staging)
            (root / "volume").mkdir()
            (root / "label").mkdir()
            sample_names = []
            for i in range(n):
                sid = f"s{i:06d}"
                sample_names.append(sid)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=np.asarray(images[i]))
                (root / "volume" / f"{sid}.npz").write_bytes(buf.getvalue())
                cls = int(np.asarray(labels[i]).flatten()[0])
                (root / "label" / f"{sid}.txt").write_text(json.dumps(cls))
            manifest = {
                "name": dataset_name, "version": "1.0",
                "fields": [
                    {"name": "volume", "kind": "volume", "role": "input", "params": {}},
                    {"name": "label", "kind": "label", "role": "gt",
                     "params": {"names": spec["classes"]}},
                ],
                "samples": sample_names,
                "source": {"repo_id": f"MedMNIST/{npz_file}", "split": args.split,
                           "zenodo_record": ZENODO_RECORD},
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

            _, summary = import_typed_dataset(
                staging, db_session=A.db.session,
                Dataset=A.Dataset, Sample=A.Sample, CustomField=A.CustomField,
                DatasetField=A.DatasetField, upload_folder=A.app.config["UPLOAD_FOLDER"],
                existing_dataset=existing, preview_only=True,
                extra_kinds={"volume": ".npz"},
            )
            existing.preview_only = True
            existing.category = CATEGORY
            existing.visibility = "public"
            existing.source_kind = "hf"
            existing.source_url = f"https://zenodo.org/records/{ZENODO_RECORD}"
            existing.source_metadata = json.dumps({
                "repo_id": f"MedMNIST/{npz_file}", "split": args.split,
                "sample_cap": args.limit, "samples_imported": summary["samples"],
                "total_rows_in_split": len(images),
            })
            existing.import_status = "ready"
            existing.import_error = None
            existing.import_progress_json = json.dumps(
                {"phase": "done", "current": summary["samples"],
                 "total": summary["samples"],
                 "message": f"Imported {summary['samples']} volumes."})
            A.db.session.commit()
            print(f"imported id={existing.id} -> {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
