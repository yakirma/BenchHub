#!/usr/bin/env python3
"""
Import the KITTI Eigen depth test split into BenchHub.

Source: HF repo `exander/kitti-depth-gt` (~2 GB total) — 697 RGB
images from KITTI Raw (left camera, image_02) plus a single 697-entry
object-array .npy with the matching ground-truth depth maps. This is
the canonical Eigen et al. 2014 test split used by ~all monocular
depth-estimation papers (NYU/KITTI is the standard pairing).

Per-sample BH layout:

    image_image/s_NNNNNN.png           original KITTI left-camera image
    depth_raw_depth_map/s_NNNNNN.npz   { depth: float32[H,W] }
    tags/s_NNNNNN.txt                  drive sequence id, for filtering
    json_metadata/s_NNNNNN.json        { kitti_path: "2011_09_26/..." }

KITTI GT is **sparse** (LIDAR returns only — most pixels are 0).
The existing `rmse` / `rms` GlobalMetrics don't mask the zeros, so
the numbers here will be lower than canonical Eigen-eval scores
(which apply a Garg/Eigen crop + valid-pixel mask). Ranking among
submissions is still meaningful as long as all submissions get the
same eval. Mask-aware metrics are a future-work item.

Alignment between `kitti_eigen_split_test.tar` and `gt_depths.npy`
depends on tar-write order matching eigen test_files.txt line order
— high-confidence assumption from the repo's structure but not
guaranteed by the upstream README. If you spot off-by-N drift in
visual previews, the alignment is the first thing to check.

Creates one Leaderboard `Depth Estimation on KITTI Eigen` with
RMSE + RMS metrics, mirroring LB id=3 (NYU-Depth V2) structure.

Internal — runs from the shell.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


SRC_REPO = 'exander/kitti-depth-gt'
TAR_URL = f'https://huggingface.co/datasets/{SRC_REPO}/resolve/main/kitti_eigen_split_test.tar'
NPY_URL = f'https://huggingface.co/datasets/{SRC_REPO}/resolve/main/gt_depths.npy'

DATASET_NAME = 'kitti-eigen-test'
LB_NAME = 'Depth Estimation on KITTI Eigen'
CATEGORY = 'Vision/Depth Estimation'


def _download(url: str, dest: Path) -> None:
    print(f'  downloading {url}', flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'BenchHub-importer'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get('Content-Length') or 0)
        got = 0
        with open(dest, 'wb') as fh:
            while True:
                buf = resp.read(1 << 20)
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


def _materialize_bh_layout(tar_path: Path, npy_path: Path, out_dir: Path) -> int:
    """Walk the tar in iteration order, pair with the same-index entry
    of the npy object array, write BH-shaped folders."""
    print('  loading gt_depths.npy (lazy mmap ok)…', flush=True)
    gt = np.load(npy_path, allow_pickle=True)
    print(f'  gt shape: {gt.shape}, dtype: {gt.dtype}')

    (out_dir / 'image_image').mkdir(parents=True, exist_ok=True)
    (out_dir / 'depth_raw_depth_map').mkdir(parents=True, exist_ok=True)
    (out_dir / 'tags').mkdir(parents=True, exist_ok=True)
    (out_dir / 'json_metadata').mkdir(parents=True, exist_ok=True)

    (out_dir / 'README.md').write_text(
        "# KITTI Eigen depth test\n\n"
        f"Sourced from HF repo `{SRC_REPO}` "
        "(https://huggingface.co/datasets/exander/kitti-depth-gt).\n\n"
        "697 image / sparse-depth pairs from the Eigen et al. 2014 "
        "test split of KITTI Raw.\n\n"
        "  image_image/<id>.png         left-camera RGB (image_02)\n"
        "  depth_raw_depth_map/<id>.npz { depth: float32[H,W] }, 0 = invalid\n"
        "  tags/<id>.txt                drive sequence id (filterable)\n"
        "  json_metadata/<id>.json      kitti_path + tar_name for traceability\n"
    )

    n = 0
    with tarfile.open(tar_path, 'r') as tar:
        # The tar also bundles KITTI Depth Prediction GT PNGs under
        # */proj_depth/groundtruth/image_02/*.png alongside the Eigen-
        # split RGB images. Filter to just the camera frames so the
        # length lines up with gt_depths.npy (697).
        members = [
            m for m in tar.getmembers()
            if m.isfile() and m.name.endswith('.png')
            and '/image_02/data/' in m.name
        ]
        print(f'  tar has {len(members)} image entries; gt has {len(gt)}')
        if len(members) != len(gt):
            print(f'  WARN: image/gt count mismatch — alignment unreliable. '
                  f'Aborting.')
            return 0
        for idx, member in enumerate(members):
            sid = f's_{idx:06d}'
            # Image
            f = tar.extractfile(member)
            if f is None:
                continue
            img_bytes = f.read()
            try:
                img = Image.open(io.BytesIO(img_bytes))
                img.load()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(out_dir / 'image_image' / f'{sid}.png', 'PNG', optimize=True)
            except Exception as e:
                print(f'  WARN: failed to decode image at idx {idx}: {e}')
                continue
            # Depth — gt[idx] is a (H, W) float (probably float32) array
            d = np.asarray(gt[idx], dtype=np.float32)
            np.savez_compressed(out_dir / 'depth_raw_depth_map' / f'{sid}.npz',
                                depth=d)
            # Tag = drive sequence (e.g. '2011_09_26_drive_0002_sync')
            # extracted from the tar member name
            parts = member.name.lstrip('./').split('/')
            drive = parts[1] if len(parts) > 1 else 'unknown'
            (out_dir / 'tags' / f'{sid}.txt').write_text(drive)
            # Metadata
            (out_dir / 'json_metadata' / f'{sid}.json').write_text(
                json.dumps({'kitti_path': member.name.lstrip('./'),
                            'tar_index': idx})
            )
            n += 1
    return n


def _zip_folder(folder: Path, dest_zip: Path) -> None:
    import zipfile
    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in folder.rglob('*'):
            if path.is_file():
                arc = path.relative_to(folder)
                zf.write(path, arcname=str(arc))


def _create_lb(app_mod, dataset):
    name = LB_NAME
    existing = app_mod.Leaderboard.query.filter_by(name=name).first()
    if existing:
        print(f'  LB "{name}" exists (id={existing.id}) — skipping')
        return existing

    rmse = app_mod.GlobalMetric.query.filter_by(name='rmse').first()
    rms = app_mod.GlobalMetric.query.filter_by(name='rms').first()
    if rmse is None or rms is None:
        raise RuntimeError('GlobalMetric rmse or rms missing — abort.')

    lb = app_mod.Leaderboard(
        name=name,
        dataset_id=dataset.id,
        category=CATEGORY,
        summary_metrics='',
        visibility='public',
        owner_user_id=None,
    )
    lb.required_pred_fields_json = json.dumps([
        {
            'name': 'raw_depth_map_pred',
            'kind': 'depth',
            'description': 'Per-pixel depth prediction (float32), '
                           'same resolution as the input image.',
        },
    ])
    app_mod.db.session.add(lb)
    app_mod.db.session.flush()

    app_mod.db.session.add(app_mod.Attachment(
        leaderboard_id=lb.id, dataset_id=dataset.id, role='primary',
    ))
    if dataset not in lb.datasets:
        lb.datasets.append(dataset)

    lm_rmse = app_mod.LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=rmse.id,
        arg_mappings=json.dumps({'gt': 'gt_raw_depth_map',
                                 'pred': 'sub_raw_depth_map_pred'}),
        target_name='RMSE',
        pooling_type='mean',
        sort_direction='lower_is_better',
    )
    lm_rms = app_mod.LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=rms.id,
        arg_mappings=json.dumps({'gt': 'gt_raw_depth_map',
                                 'pred': 'sub_raw_depth_map_pred'}),
        target_name='RMS',
        pooling_type='mean',
        sort_direction='lower_is_better',
    )
    app_mod.db.session.add_all([lm_rmse, lm_rms])
    app_mod.db.session.flush()
    lb.summary_metrics = f'lm_{lm_rmse.id},lm_{lm_rms.id}'
    app_mod.db.session.commit()
    print(f'  -> created LB id={lb.id} ({name})')
    return lb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--keep-workdir', action='store_true')
    args = ap.parse_args()

    import app as app_mod

    work = Path(args.workdir or tempfile.mkdtemp(prefix='benchhub-kitti-'))
    print(f'Work dir: {work}')

    with app_mod.app.app_context():
        existing = app_mod.Dataset.query.filter_by(name=DATASET_NAME).first()
        if existing and not args.force:
            print(f'Dataset {DATASET_NAME} exists (id={existing.id}) — '
                  'skipping import; only creating LB if missing.')
            dataset = existing
        else:
            if existing and args.force:
                print(f'Dataset {DATASET_NAME} exists — --force given, deleting')
                app_mod.db.session.delete(existing)
                app_mod.db.session.commit()
            tar_path = work / 'kitti.tar'
            npy_path = work / 'gt.npy'
            if not tar_path.exists() or tar_path.stat().st_size == 0:
                _download(TAR_URL, tar_path)
            if not npy_path.exists() or npy_path.stat().st_size == 0:
                _download(NPY_URL, npy_path)

            layout = work / 'layout'
            n = _materialize_bh_layout(tar_path, npy_path, layout)
            print(f'  wrote {n} samples')
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
            if dataset is not None and not dataset.source_url:
                dataset.source_url = (
                    'https://huggingface.co/datasets/exander/kitti-depth-gt')
                app_mod.db.session.commit()
            # Cleanup the heavy intermediates.
            tar_path.unlink(missing_ok=True)
            npy_path.unlink(missing_ok=True)
            shutil.rmtree(layout, ignore_errors=True)
            zip_path.unlink(missing_ok=True)

        _create_lb(app_mod, dataset)

    if not args.keep_workdir:
        shutil.rmtree(work, ignore_errors=True)
        print('\nWork dir cleaned up.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
