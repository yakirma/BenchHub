#!/usr/bin/env python3
"""Register a `volume` data type and seed a 3D medical dataset.

BenchHub ships no native 3D kind, so this first registers a `volume`
DataTypeDef (bytes = a .npz holding one 3D array; `visualize` renders a
montage of axial slices in the sandbox; `decode` returns the ndarray for
metrics). It then imports MedMNIST's **NoduleMNIST3D** (lung-nodule
malignancy, 28x28x28 uint8 volumes) as a public typed dataset under
category "Medical/Volume Classification" via the standard
`import_typed_dataset(extra_kinds={'volume': '.npz'})` path.

Run on the prod box:

    BENCHHUB_DATA_DIR=~/.dtofbenchmarking PYTHONPATH=/home/ymatri/benchhub \
        ~/benchhub/.venv/bin/python scripts/seed_volume_medical.py [--limit N]
"""
import argparse
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ADMIN_UID = 2
ZENODO_RECORD = "6496656"          # MedMNIST v2 (28px) — has the 3D sets
NPZ_FILE = "nodulemnist3d.npz"
DATASET_NAME = "MedMNIST__NoduleMNIST3D"
CATEGORY = "Medical/Volume Classification"
SPLIT = "train"
CLASS_NAMES = ["benign", "malignant"]   # NoduleMNIST3D labels

VISUALIZE_CODE = '''
def visualize(blob, params):
    import io
    import numpy as np
    from PIL import Image
    z = np.load(io.BytesIO(blob))
    arr = np.asarray(z[z.files[0]])
    while arr.ndim > 3:        # drop a leading channel axis if present
        arr = arr[0]
    a = arr.astype("float32")
    rng = float(a.max() - a.min()) or 1.0
    a = ((a - a.min()) / rng * 255.0).astype("uint8")
    if a.ndim == 2:
        return Image.fromarray(a).convert("RGB")
    D, H, W = a.shape
    n = min(9, D)
    idxs = np.linspace(0, D - 1, n).astype(int)
    cols = 3
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * H, cols * W), dtype="uint8")
    for i, si in enumerate(idxs):
        r, c = divmod(i, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = a[int(si)]
    img = Image.fromarray(canvas).convert("RGB")
    scale = max(1, 384 // max(img.size))
    if scale > 1:
        img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.NEAREST)
    return img
'''

DECODE_CODE = '''
def decode(blob, params):
    import io
    import numpy as np
    z = np.load(io.BytesIO(blob))
    return np.asarray(z[z.files[0]])
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap samples (0 = all in split)")
    args = ap.parse_args()

    os.environ.setdefault("BENCHHUB_DATA_DIR", os.path.expanduser("~/.dtofbenchmarking"))
    import numpy as np
    import app as A
    from benchhub.manifest import import_typed_dataset

    with A.app.app_context():
        # 1. Register the `volume` data type (admin-owned, public) ----------
        dt = A.DataTypeDef.query.filter_by(name="volume").first()
        if dt is None:
            dt = A.DataTypeDef(
                name="volume", description="3D volume (.npz holding one ndarray); "
                "rendered as an axial-slice montage.",
                file_ext=".npz", viz_mime="image/png",
                visualize_code=VISUALIZE_CODE.strip(), decode_code=DECODE_CODE.strip(),
                owner_user_id=ADMIN_UID, visibility="public",
            )
            A.db.session.add(dt)
            A.db.session.commit()
            print(f"registered DataTypeDef 'volume' id={dt.id}", flush=True)
        else:
            print(f"DataTypeDef 'volume' already exists id={dt.id}", flush=True)

        # 2. Download + read the 3D npz -------------------------------------
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{NPZ_FILE}/content"
        print(f"downloading {url}", flush=True)
        raw = urllib.request.urlopen(url, timeout=300).read()
        z = np.load(io.BytesIO(raw))
        images = z[f"{SPLIT}_images"]
        labels = z[f"{SPLIT}_labels"]
        n = len(images)
        if args.limit:
            n = min(n, args.limit)
        print(f"{NPZ_FILE}: {SPLIT} has {len(images)} volumes {images.shape[1:]}, importing {n}", flush=True)

        # 3. Build the typed staging dir -----------------------------------
        with tempfile.TemporaryDirectory(prefix="bh_vol_") as staging:
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
                name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
                (root / "label" / f"{sid}.txt").write_text(name)
            manifest = {
                "name": DATASET_NAME, "version": "1.0",
                "fields": [
                    {"name": "volume", "kind": "volume", "role": "input", "params": {}},
                    {"name": "label", "kind": "label", "role": "gt",
                     "params": {"names": CLASS_NAMES}},
                ],
                "samples": sample_names,
                "source": {"repo_id": f"MedMNIST/{NPZ_FILE}", "split": SPLIT,
                           "zenodo_record": ZENODO_RECORD},
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # 4. Create the Dataset row + import -----------------------------
            existing = A.Dataset.query.filter_by(name=DATASET_NAME).first()
            if existing and existing.import_status == "ready":
                print("already imported (ready) — refreshing in place", flush=True)
            if existing is None:
                existing = A.Dataset(
                    name=DATASET_NAME, owner_user_id=ADMIN_UID, visibility="public",
                    category=CATEGORY, import_status="importing",
                )
                A.db.session.add(existing)
                A.db.session.commit()

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
            existing.import_status = "ready"
            existing.import_error = None
            existing.import_progress_json = json.dumps(
                {"phase": "done", "current": summary["samples"],
                 "total": summary["samples"], "message": f"Imported {summary['samples']} volumes."})
            A.db.session.commit()
            print(f"imported id={existing.id} -> {summary}", flush=True)


if __name__ == "__main__":
    main()
