#!/usr/bin/env python3
"""
Import a subset of BraTS 2021 from the HF mirror `rocky93/BraTS_segmentation`
into BenchHub as a single BH Dataset, then create three Leaderboards
(Whole Tumor, Tumor Core, Enhancing Tumor) sharing it.

The mirror holds 1251 cases as loose `.nii.gz` files per `BraTS2021_XXXXX/`
case folder. We download a configurable subset (default 100 cases),
extract the **axial** tumor-bearing slices (where the seg volume has
any non-zero voxel), and save per-slice samples in BH folder convention:

    image_flair/<case>_zNNN.png         FLAIR modality slice, uint8 normalized
    mask_seg/<case>_zNNN.png            raw 4-class label map (0/1/2/4)
    mask_wt/<case>_zNNN.png             binary {seg != 0}        whole tumor
    mask_tc/<case>_zNNN.png             binary {seg ∈ 1,4}        tumor core
    mask_et/<case>_zNNN.png             binary {seg == 4}         enhancing tumor
    tags/<case>_zNNN.txt                case id + slice index

We only fetch FLAIR + seg per case (skip t1/t1ce/t2) to save bandwidth.
FLAIR is the most-used single-modality input in BraTS literature. If
you want multi-modal inputs later, re-run with --modalities flair t1 t1ce t2.

Three LBs scored on Dice, each comparing a *different* GT mask field
against a per-LB pred field:

    Whole Tumor Segmentation on BraTS2021    gt: mask_wt   pred: wt_pred
    Tumor Core Segmentation on BraTS2021     gt: mask_tc   pred: tc_pred
    Enhancing Tumor Segmentation on BraTS2021 gt: mask_et   pred: et_pred

Internal — runs from the shell.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402
from PIL import Image  # noqa: E402


SRC_REPO = 'rocky93/BraTS_segmentation'
FILE_URL = f'https://huggingface.co/datasets/{SRC_REPO}/resolve/main/'

DATASET_NAME = 'brats2021-subset'
CATEGORY = 'Vision/Medical Image Segmentation'

LB_VARIANTS = [
    {
        'name': 'Whole Tumor Segmentation on BraTS2021',
        'gt_field': 'mask_wt',
        'pred_name': 'wt_pred',
        'desc': 'Binary mask for the whole-tumor region (any tumor label).',
    },
    {
        'name': 'Tumor Core Segmentation on BraTS2021',
        'gt_field': 'mask_tc',
        'pred_name': 'tc_pred',
        'desc': 'Binary mask for the tumor-core region (labels 1 and 4 = '
                'necrotic + enhancing).',
    },
    {
        'name': 'Enhancing Tumor Segmentation on BraTS2021',
        'gt_field': 'mask_et',
        'pred_name': 'et_pred',
        'desc': 'Binary mask for the enhancing-tumor region (label 4).',
    },
]


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BenchHub-importer'})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest, 'wb') as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
        return True
    except Exception as e:
        print(f'  download failed for {url}: {e}')
        return False


def _normalize_to_uint8(vol: np.ndarray) -> np.ndarray:
    """Per-volume p1..p99 stretch to 0..255 uint8 for visualization."""
    arr = vol.astype(np.float32)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _materialize_bh_layout(case_ids: list[str], work: Path, out_dir: Path,
                           slices_per_case_cap: int) -> int:
    """Download FLAIR + seg per case, slice tumor-bearing axial planes,
    write BH folders. Returns total samples written."""
    for sub in ('image_flair', 'mask_seg', 'mask_wt', 'mask_tc',
                'mask_et', 'tags'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    (out_dir / 'README.md').write_text(
        f"# BraTS 2021 — {len(case_ids)}-case subset\n\n"
        f"Sourced from HF mirror `{SRC_REPO}` "
        f"(https://huggingface.co/datasets/{SRC_REPO}).\n\n"
        "FLAIR modality + 4-class segmentation (background, "
        "necrotic=1, edema=2, enhancing=4). Derived binary masks for "
        "the three standard BraTS sub-tasks (WT/TC/ET) are written "
        "alongside the raw seg.\n"
    )

    n_total = 0
    for ci, case in enumerate(case_ids):
        print(f'  [{ci+1}/{len(case_ids)}] {case}', flush=True)
        flair_path = work / f'{case}_flair.nii.gz'
        seg_path = work / f'{case}_seg.nii.gz'
        ok_f = _download(f'{FILE_URL}{case}/{case}_flair.nii.gz', flair_path)
        ok_s = _download(f'{FILE_URL}{case}/{case}_seg.nii.gz', seg_path)
        if not (ok_f and ok_s):
            continue

        try:
            flair_vol = nib.load(str(flair_path)).get_fdata()  # (H, W, D)
            seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.int32)
        except Exception as e:
            print(f'    nibabel load failed: {e}')
            flair_path.unlink(missing_ok=True)
            seg_path.unlink(missing_ok=True)
            continue

        flair_u8 = _normalize_to_uint8(flair_vol)

        # Axial (z) slices that contain any tumor label.
        tumor_slice_mask = (seg_vol != 0).any(axis=(0, 1))
        slice_idxs = np.where(tumor_slice_mask)[0].tolist()
        # Subsample if a single case has many tumor slices, to keep
        # the dataset balanced across cases.
        if slices_per_case_cap and len(slice_idxs) > slices_per_case_cap:
            step = len(slice_idxs) // slices_per_case_cap
            slice_idxs = slice_idxs[::step][:slices_per_case_cap]

        for z in slice_idxs:
            sid = f'{case}_z{z:03d}'
            # Image
            img = Image.fromarray(flair_u8[:, :, z], mode='L')
            img.save(out_dir / 'image_flair' / f'{sid}.png', 'PNG', optimize=True)
            # Raw multiclass seg (encode as P-mode palette PNG)
            seg_slice = seg_vol[:, :, z].astype(np.uint8)
            Image.fromarray(seg_slice, mode='L').save(
                out_dir / 'mask_seg' / f'{sid}.png', 'PNG', optimize=True,
            )
            # Derived binary masks (uint8 0/1 — BH mask detector treats
            # mode='L' with ≤32 unique values as a mask).
            (Image.fromarray((seg_slice != 0).astype(np.uint8), mode='L')
                .save(out_dir / 'mask_wt' / f'{sid}.png', 'PNG', optimize=True))
            (Image.fromarray(np.isin(seg_slice, [1, 4]).astype(np.uint8), mode='L')
                .save(out_dir / 'mask_tc' / f'{sid}.png', 'PNG', optimize=True))
            (Image.fromarray((seg_slice == 4).astype(np.uint8), mode='L')
                .save(out_dir / 'mask_et' / f'{sid}.png', 'PNG', optimize=True))
            (out_dir / 'tags' / f'{sid}.txt').write_text(f'{case},z={z}')
            n_total += 1

        # Reclaim disk between cases.
        flair_path.unlink(missing_ok=True)
        seg_path.unlink(missing_ok=True)
    return n_total


def _zip_folder(folder: Path, dest_zip: Path) -> None:
    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in folder.rglob('*'):
            if path.is_file():
                arc = path.relative_to(folder)
                zf.write(path, arcname=str(arc))


def _create_lbs(app_mod, dataset):
    dice = app_mod.GlobalMetric.query.filter_by(name='mean_dice').first()
    if dice is None:
        raise RuntimeError('GlobalMetric mean_dice not found — abort.')

    for v in LB_VARIANTS:
        existing = app_mod.Leaderboard.query.filter_by(name=v['name']).first()
        if existing:
            print(f'  LB "{v["name"]}" exists (id={existing.id}) — skip')
            continue

        lb = app_mod.Leaderboard(
            name=v['name'],
            dataset_id=dataset.id,
            category=CATEGORY,
            summary_metrics='',
            visibility='public',
            owner_user_id=None,
        )
        lb.required_pred_fields_json = json.dumps([
            {
                'name': v['pred_name'],
                'kind': 'mask',
                'description': v['desc'],
            },
        ])
        app_mod.db.session.add(lb)
        app_mod.db.session.flush()

        app_mod.db.session.add(app_mod.Attachment(
            leaderboard_id=lb.id, dataset_id=dataset.id, role='primary',
        ))

        lm = app_mod.LeaderboardMetric(
            leaderboard_id=lb.id,
            global_metric_id=dice.id,
            arg_mappings=json.dumps({
                'gt': f'gt_{v["gt_field"]}',
                'pred': f'sub_{v["pred_name"]}',
            }),
            target_name='Dice',
            pooling_type='mean',
            sort_direction='higher_is_better',
        )
        app_mod.db.session.add(lm)
        app_mod.db.session.flush()
        lb.summary_metrics = f'lm_{lm.id}'
        app_mod.db.session.commit()
        print(f'  -> created LB id={lb.id} ({v["name"]})')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=100,
                    help='How many BraTS cases to import (default 100).')
    ap.add_argument('--slices-per-case', type=int, default=20,
                    help='Cap on tumor-bearing slices per case '
                         '(default 20). 0 = unlimited.')
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    import app as app_mod

    work = Path(args.workdir or tempfile.mkdtemp(prefix='benchhub-brats-'))
    print(f'Work dir: {work}', flush=True)

    with app_mod.app.app_context():
        existing = app_mod.Dataset.query.filter_by(name=DATASET_NAME).first()
        if existing and not args.force:
            print(f'Dataset {DATASET_NAME} exists (id={existing.id}) — '
                  'skipping import; only creating LBs if missing.')
            dataset = existing
        else:
            if existing and args.force:
                print(f'Dataset {DATASET_NAME} exists — --force given, deleting')
                app_mod.db.session.delete(existing)
                app_mod.db.session.commit()

            # List cases via the HF API.
            import urllib.request
            req = urllib.request.Request(
                f'https://huggingface.co/api/datasets/{SRC_REPO}',
                headers={'User-Agent': 'BenchHub-importer'},
            )
            api = json.loads(urllib.request.urlopen(req, timeout=60).read())
            sibs = [s.get('rfilename', '') for s in api.get('siblings', [])]
            case_ids = sorted({f.split('/')[0]
                              for f in sibs
                              if '/' in f and f.startswith('BraTS2021_')})[:args.cases]
            print(f'  picked {len(case_ids)} cases: '
                  f'{case_ids[0]} … {case_ids[-1]}')

            layout = work / 'layout'
            n = _materialize_bh_layout(case_ids, work, layout,
                                       args.slices_per_case)
            print(f'  wrote {n} samples ({len(case_ids)} cases × tumor slices)')
            if n == 0:
                print('Nothing written — abort.')
                return 1

            zip_path = work / f'{DATASET_NAME}.zip'
            _zip_folder(layout, zip_path)
            print(f'  zip size: {zip_path.stat().st_size/1e6:.1f} MB')

            ok, msg, ds_id = app_mod.process_dataset_zip(
                str(zip_path), DATASET_NAME,
                owner_user_id=None, category=CATEGORY,
            )
            if not ok:
                print(f'  FAILED: {msg}')
                return 1
            print(f'  -> Dataset id={ds_id}')
            dataset = app_mod.db.session.get(app_mod.Dataset, ds_id)
            shutil.rmtree(layout, ignore_errors=True)
            zip_path.unlink(missing_ok=True)

        _create_lbs(app_mod, dataset)

    shutil.rmtree(work, ignore_errors=True)
    print('\nWork dir cleaned up.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
