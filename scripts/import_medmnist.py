#!/usr/bin/env python3
"""
Import MedMNIST+ datasets from Zenodo (record 10519652) into BenchHub.

Default behavior — run with no args to get the 12 2D / 224×224 / test-set
collection imported as 12 separate BH Datasets:

    python scripts/import_medmnist.py

Each dataset becomes a BH Dataset with category "Vision/Medical Image
Classification" and the following per-sample fields (BenchHub
`<type>_<field_name>/<sample>.<ext>` layout):

    image_image/s_NNNNNN.png         24-bit RGB PNG (3-channel) or
                                      grayscale "L" PNG (1-channel)
    scalar_label/s_NNNNNN.txt        integer class index (single-label datasets)
    text_class_name/s_NNNNNN.txt     human-readable class name (single-label)
    tags/s_NNNNNN.txt                class name as tag for filtering
    json_labels/s_NNNNNN.json        full one-hot vector (multi-label datasets,
                                      currently only ChestMNIST)

The script reuses BenchHub's normal `process_dataset_zip` pipeline — it
materializes the BH folder layout in a tempdir, zips it, hands the ZIP
off to process_dataset_zip. Result: each MedMNIST variant appears as a
real Dataset row in the catalog, indistinguishable from a user-uploaded
ZIP. No new Attachment kind or schema change required.

Internal — not wired to any HTTP route, run from the shell.

Usage:
    # Default: 12 datasets, 224 size, test split
    python scripts/import_medmnist.py

    # Smaller subset:
    python scripts/import_medmnist.py --datasets retinamnist breastmnist

    # Different size:
    python scripts/import_medmnist.py --size 128

    # Different split (train/val/test):
    python scripts/import_medmnist.py --split val

Author: this script is internal-use only. Keep around for re-runs (new
MedMNIST revisions, fresh DB seeds, etc.) but don't surface to end users.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Ensure we can import the Flask `app` (this script lives in scripts/, the
# app is in the repo root one level up).
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# Per-dataset metadata. Class names mirror the official MedMNIST INFO dict
# (see https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).
# Hardcoded here so the script has no `medmnist` pip dependency.
#
# `task`        — 'multi-class' (single integer label) or 'multi-label'.
# `class_names` — human-readable names indexed by integer label.
# `is_rgb`      — True if images are 3-channel; False = grayscale.
# `category`    — BH "Area/Task" category string for `Dataset.category`.
# ---------------------------------------------------------------------------

MEDMNIST_2D_INFO: dict[str, dict] = {
    'pathmnist': {
        'task': 'multi-class',
        'is_rgb': True,
        'category': 'Vision/Image Classification',
        'class_names': [
            'adipose', 'background', 'debris', 'lymphocytes',
            'mucus', 'smooth muscle', 'normal colon mucosa',
            'cancer-associated stroma', 'colorectal adenocarcinoma epithelium',
        ],
        'description': 'Colon pathology — 9-class classification of '
                       'tissue types in colorectal cancer histology.',
    },
    'chestmnist': {
        'task': 'multi-label',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': [
            'atelectasis', 'cardiomegaly', 'effusion', 'infiltration',
            'mass', 'nodule', 'pneumonia', 'pneumothorax',
            'consolidation', 'edema', 'emphysema', 'fibrosis',
            'pleural', 'hernia',
        ],
        'description': 'Chest X-ray — 14-label multi-label classification '
                       '(NIH Chest X-ray 14 derivative).',
    },
    'dermamnist': {
        'task': 'multi-class',
        'is_rgb': True,
        'category': 'Vision/Image Classification',
        'class_names': [
            'actinic keratoses', 'basal cell carcinoma',
            'benign keratosis-like lesions', 'dermatofibroma',
            'melanoma', 'melanocytic nevi', 'vascular lesions',
        ],
        'description': 'Dermatoscope — 7-class skin-lesion '
                       'classification (HAM10000 derivative).',
    },
    'octmnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': ['CNV', 'DME', 'drusen', 'normal'],
        'description': 'Retinal OCT — 4-class classification of '
                       'macular pathologies.',
    },
    'pneumoniamnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': ['normal', 'pneumonia'],
        'description': 'Chest X-ray — 2-class pneumonia classification.',
    },
    'retinamnist': {
        'task': 'multi-class',
        'is_rgb': True,
        'category': 'Vision/Image Classification',
        'class_names': ['no DR', 'mild', 'moderate', 'severe', 'proliferative'],
        'description': 'Fundus camera — ordinal severity classification '
                       'of diabetic retinopathy (5 grades).',
    },
    'breastmnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': ['malignant', 'normal or benign'],
        'description': 'Breast ultrasound — 2-class malignancy '
                       'classification.',
    },
    'bloodmnist': {
        'task': 'multi-class',
        'is_rgb': True,
        'category': 'Vision/Image Classification',
        'class_names': [
            'basophil', 'eosinophil', 'erythroblast',
            'immature granulocytes', 'lymphocyte', 'monocyte',
            'neutrophil', 'platelet',
        ],
        'description': 'Peripheral blood-cell microscopy — '
                       '8-class cell-type classification.',
    },
    'tissuemnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': [
            'collecting duct, connecting tubule',
            'distal convoluted tubule',
            'glomerular endothelial cells',
            'interstitial endothelial cells',
            'leukocytes',
            'podocytes',
            'proximal tubule segments',
            'thick ascending limb',
        ],
        'description': 'Kidney cortex microscopy — '
                       '8-class tissue-type classification.',
    },
    'organamnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': [
            'bladder', 'femur-left', 'femur-right', 'heart',
            'kidney-left', 'kidney-right', 'liver', 'lung-left',
            'lung-right', 'pancreas', 'spleen',
        ],
        'description': 'Abdominal CT (axial) — 11-class organ '
                       'classification.',
    },
    'organcmnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': [
            'bladder', 'femur-left', 'femur-right', 'heart',
            'kidney-left', 'kidney-right', 'liver', 'lung-left',
            'lung-right', 'pancreas', 'spleen',
        ],
        'description': 'Abdominal CT (coronal) — 11-class organ '
                       'classification.',
    },
    'organsmnist': {
        'task': 'multi-class',
        'is_rgb': False,
        'category': 'Vision/Image Classification',
        'class_names': [
            'bladder', 'femur-left', 'femur-right', 'heart',
            'kidney-left', 'kidney-right', 'liver', 'lung-left',
            'lung-right', 'pancreas', 'spleen',
        ],
        'description': 'Abdominal CT (sagittal) — 11-class organ '
                       'classification.',
    },
}

ZENODO_RECORD = '10519652'
ZENODO_FILE_URL = (
    'https://zenodo.org/api/records/{record}/files/{filename}/content'
)


def _filename(base: str, size: int) -> str:
    """Zenodo filename convention. Size 28 omits the suffix
    (`pathmnist.npz`), other sizes append `_<size>` before .npz."""
    if size == 28:
        return f'{base}.npz'
    return f'{base}_{size}.npz'


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream-download with a basic progress dot per 16 MB."""
    print(f'  downloading {url}', flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'BenchHub-importer'})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get('Content-Length') or 0)
        got = 0
        with open(dest, 'wb') as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                got += len(buf)
                if total:
                    sys.stdout.write(
                        f'\r  {got/1e6:>7.1f} / {total/1e6:.1f} MB '
                        f'({100*got/total:5.1f}%)'
                    )
                    sys.stdout.flush()
    sys.stdout.write('\n')


def _materialize_bh_layout(
    npz_path: Path,
    out_dir: Path,
    info: dict,
    split: str,
) -> int:
    """Read `<split>_images` + `<split>_labels` from the NPZ and lay them
    out under `out_dir` in BenchHub's standard folder convention. Returns
    the number of samples written."""
    z = np.load(npz_path)
    images = z[f'{split}_images']
    labels = z[f'{split}_labels']
    is_multi_label = info['task'] == 'multi-label'
    class_names = info['class_names']

    # BH expects field folders directly under the dataset root. README at
    # the root keeps process_dataset_zip from treating the only-populated
    # subfolder as the dataset wrapper.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'image_image').mkdir(exist_ok=True)
    if is_multi_label:
        (out_dir / 'json_labels').mkdir(exist_ok=True)
    else:
        (out_dir / 'scalar_label').mkdir(exist_ok=True)
        (out_dir / 'text_class_name').mkdir(exist_ok=True)
    (out_dir / 'tags').mkdir(exist_ok=True)

    (out_dir / 'README.md').write_text(
        f"# {npz_path.stem} ({split} split)\n\n"
        f"{info['description']}\n\n"
        f"Imported from Zenodo record {ZENODO_RECORD} "
        f"(https://doi.org/10.5281/zenodo.{ZENODO_RECORD})\n\n"
        f"Task: {info['task']}, classes: {len(class_names)}\n"
    )

    n_written = 0
    for i in range(len(images)):
        sid = f's_{i:06d}'
        arr = images[i]
        if info['is_rgb']:
            # MedMNIST 2D RGB layout is (H, W, 3) uint8 already.
            img = Image.fromarray(arr, mode='RGB')
        else:
            # Grayscale; arr is (H, W) uint8.
            img = Image.fromarray(arr, mode='L')
        img.save(out_dir / 'image_image' / f'{sid}.png', 'PNG', optimize=True)

        if is_multi_label:
            # labels[i] is (n_classes,) binary (or sometimes shape (1, K)).
            vec = labels[i].astype(int).flatten().tolist()
            (out_dir / 'json_labels' / f'{sid}.json').write_text(
                json.dumps(vec)
            )
            # Tag with each present class — usable for filtering / browsing.
            present = [class_names[j] for j, v in enumerate(vec) if v]
            (out_dir / 'tags' / f'{sid}.txt').write_text(
                ','.join(present) if present else 'no findings'
            )
        else:
            cls_idx = int(labels[i].flatten()[0])
            cls_name = (class_names[cls_idx] if 0 <= cls_idx < len(class_names)
                        else f'class_{cls_idx}')
            (out_dir / 'scalar_label' / f'{sid}.txt').write_text(str(cls_idx))
            (out_dir / 'text_class_name' / f'{sid}.txt').write_text(cls_name)
            (out_dir / 'tags' / f'{sid}.txt').write_text(cls_name)
        n_written += 1
    return n_written


def _zip_folder(folder: Path, dest_zip: Path) -> None:
    """ZIP `folder` into `dest_zip`. process_dataset_zip then extracts
    and detects the dataset shape from the folder layout."""
    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in folder.rglob('*'):
            if path.is_file():
                arc = path.relative_to(folder)
                zf.write(path, arcname=str(arc))


def _import_one(base: str, info: dict, size: int, split: str,
                work_root: Path, app_module, force: bool) -> None:
    """Download + lay out + process one MedMNIST variant."""
    dataset_name = f'medmnist-{base}-{size}-{split}'
    print(f'\n=== {dataset_name} ===', flush=True)

    # If a dataset with this name already exists, skip unless --force.
    existing = app_module.Dataset.query.filter_by(name=dataset_name).first()
    if existing:
        if not force:
            print(f'  exists (id={existing.id}) — skipping (use --force to redo)')
            return
        print(f'  exists (id={existing.id}) — --force given, deleting first')
        app_module.db.session.delete(existing)
        app_module.db.session.commit()

    work = work_root / dataset_name
    work.mkdir(parents=True, exist_ok=True)
    npz_path = work / 'src.npz'
    layout_dir = work / 'layout'
    zip_path = work / f'{dataset_name}.zip'

    try:
        if not npz_path.exists() or npz_path.stat().st_size == 0:
            url = ZENODO_FILE_URL.format(
                record=ZENODO_RECORD,
                filename=_filename(base, size),
            )
            _download(url, npz_path)

        print(f'  laying out BH folders…')
        n = _materialize_bh_layout(npz_path, layout_dir, info, split)
        print(f'  wrote {n} samples')

        print(f'  zipping…')
        _zip_folder(layout_dir, zip_path)
        print(f'  zip size: {zip_path.stat().st_size/1e6:.1f} MB')

        print(f'  process_dataset_zip…')
        ok, msg, ds_id = app_module.process_dataset_zip(
            str(zip_path),
            dataset_name,
            owner_user_id=None,
            category=info['category'],
        )
        if not ok:
            print(f'  FAILED: {msg}')
            return
        print(f'  -> Dataset id={ds_id}')

        # Drop the source NPZ + the materialized folder once the ZIP is
        # accepted; the BH copy under uploads/datasets/<name>/ is canonical.
        npz_path.unlink(missing_ok=True)
        shutil.rmtree(layout_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        print(f'  ERROR: {e}')


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Import MedMNIST+ datasets from Zenodo to BenchHub.',
    )
    ap.add_argument(
        '--datasets', nargs='+',
        default=list(MEDMNIST_2D_INFO.keys()),
        choices=list(MEDMNIST_2D_INFO.keys()),
        help='Which MedMNIST 2D datasets to import (default: all 12).',
    )
    ap.add_argument(
        '--size', type=int, default=224, choices=[28, 64, 128, 224],
        help='Image size to import (default: 224).',
    )
    ap.add_argument(
        '--split', default='test', choices=['train', 'val', 'test'],
        help='Which split to extract (default: test).',
    )
    ap.add_argument(
        '--workdir', default=None,
        help='Working tempdir for downloads + ZIPs (default: system tmp).',
    )
    ap.add_argument(
        '--force', action='store_true',
        help='Re-import datasets that already exist (deletes the old row).',
    )
    ap.add_argument(
        '--keep-workdir', action='store_true',
        help='Do not clean up the work directory at the end.',
    )
    args = ap.parse_args()

    import app as app_module  # late import: needs sys.path set above

    work_root = Path(args.workdir or tempfile.mkdtemp(
        prefix='benchhub-medmnist-'))
    print(f'Work dir: {work_root}', flush=True)

    with app_module.app.app_context():
        for base in args.datasets:
            _import_one(
                base, MEDMNIST_2D_INFO[base], args.size, args.split,
                work_root, app_module, args.force,
            )

    if not args.keep_workdir:
        shutil.rmtree(work_root, ignore_errors=True)
        print(f'\nWork dir cleaned up.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
