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
    from PIL import Image, ImageDraw
    z = np.load(io.BytesIO(blob))
    arr = np.asarray(z[z.files[0]])
    while arr.ndim > 3:        # drop a leading channel axis if present
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    a = arr.astype("float32")
    rng = float(a.max() - a.min()) or 1.0
    a = (a - a.min()) / rng                        # normalise 0..1
    D, H, W = a.shape

    # Three orthogonal max-intensity projections -> the cube's visible faces.
    front = a.max(axis=0)                          # (H, W) along depth
    top = a.max(axis=1)                            # (D, W) looking down
    right = a.max(axis=2).T                        # (H, D) from the side

    def to_img(x, bright):
        g = np.clip(x * 255.0 * bright, 0, 255).astype("uint8")
        return Image.fromarray(g).convert("RGB")

    # Cabinet (oblique) projection geometry.
    s = 150
    dep = int(s * 0.55)
    ang = np.deg2rad(32.0)
    ox, oy = int(dep * np.cos(ang)), -int(dep * np.sin(ang))
    pad = 18
    cw = s + abs(ox) + 2 * pad
    ch = s + abs(oy) + 2 * pad
    x0, y0 = pad, pad + abs(oy)
    FTL = (x0, y0); FTR = (x0 + s, y0)
    FBL = (x0, y0 + s); FBR = (x0 + s, y0 + s)
    BTL = (x0 + ox, y0 + oy); BTR = (x0 + s + ox, y0 + oy)
    BBR = (x0 + s + ox, y0 + s + oy)

    canvas = Image.new("RGB", (cw, ch), (17, 15, 28))

    def paste_par(src, p0, p1, p2):
        w, h = src.size
        M = np.array([[p0[0], p0[1], 1.0], [p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0]])
        cx = np.linalg.solve(M, np.array([0.0, float(w), 0.0]))
        cy = np.linalg.solve(M, np.array([0.0, 0.0, float(h)]))
        coeffs = (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
        warped = src.transform((cw, ch), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        mask = Image.new("L", src.size, 255).transform(
            (cw, ch), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        canvas.paste(warped, (0, 0), mask)

    f_img = to_img(front, 1.00).resize((s, s), Image.BILINEAR)
    t_img = to_img(top, 0.80).resize((s, dep), Image.BILINEAR)
    r_img = to_img(right, 0.62).resize((dep, s), Image.BILINEAR)

    paste_par(t_img, BTL, BTR, FTL)                # top face
    paste_par(r_img, FTR, BTR, FBR)                # right face
    canvas.paste(f_img, (x0, y0))                  # front face

    d = ImageDraw.Draw(canvas)
    edge = (215, 210, 235)
    for u, v in [(FTL, FTR), (FTR, FBR), (FBR, FBL), (FBL, FTL),
                 (FTL, BTL), (FTR, BTR), (FBR, BBR), (BTL, BTR), (BTR, BBR)]:
        d.line([u, v], fill=edge, width=1)

    scale = max(1, 360 // max(canvas.size))
    if scale > 1:
        canvas = canvas.resize((canvas.size[0] * scale, canvas.size[1] * scale), Image.NEAREST)
    return canvas
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
    ap.add_argument("--update-viz", action="store_true",
                    help="update the existing volume dtype's visualize/decode code, then exit")
    args = ap.parse_args()

    os.environ.setdefault("BENCHHUB_DATA_DIR", os.path.expanduser("~/.dtofbenchmarking"))
    import numpy as np
    import app as A
    from benchhub.manifest import import_typed_dataset

    with A.app.app_context():
        if args.update_viz:
            dt = A.DataTypeDef.query.filter_by(name="volume").first()
            if dt is None:
                print("no 'volume' dtype registered yet"); return
            dt.visualize_code = VISUALIZE_CODE.strip()
            dt.decode_code = DECODE_CODE.strip()
            A.db.session.commit()
            print(f"updated 'volume' dtype id={dt.id} "
                  f"({len(dt.visualize_code)} chars of visualize); cache busts via code hash")
            return
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
                # Label is stored inline as JSON (Label.decode does json.loads);
                # store the integer class index, names carried in field params.
                (root / "label" / f"{sid}.txt").write_text(json.dumps(cls))
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
