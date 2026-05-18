#!/usr/bin/env python3
"""
Import DIV2K validation set from EE Zurich into BenchHub.

Default behavior — run with no args to create 3 separate BH Datasets
(one per scale: ×2, ×3, ×4) and 3 LBs scored on PSNR + SSIM:

    python scripts/import_div2k.py

DIV2K is a standard super-resolution benchmark: 800 train + 100 valid
images, each with bicubically downsampled LR versions at ×2, ×3, ×4
(and ×8 in some variants). We import the **valid** split since the
official test set GT is held by EE Zurich for the NTIRE challenges.

Per-sample layout (one Dataset per scale):

    image_lr/0801.png       low-resolution input (×N downsampled)
    image_hr/0801.png       high-resolution ground truth (2K pixels)

Sample names mirror the DIV2K convention: 0801..0900 for the valid
split. No re-numbering.

Each scale gets its own BH Dataset and one LB so that arg_mappings on
the LB clearly tie one LR input to one HR GT. Three LBs total:

    Image Super-Resolution on DIV2K (×2 valid)
    Image Super-Resolution on DIV2K (×3 valid)
    Image Super-Resolution on DIV2K (×4 valid)

Metrics: PSNR + SSIM (existing GlobalMetrics, both kinds=image×image,
mean pooling, higher-is-better).

Internal — not wired to any HTTP route, run from the shell.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Upstream URLs. EE Zurich hosts the canonical DIV2K distribution.
# ---------------------------------------------------------------------------
BASE = 'http://data.vision.ee.ethz.ch/cvl/DIV2K'

# Filename templates: (url_suffix, inner-folder-name-in-zip).
HR_URL = f'{BASE}/DIV2K_valid_HR.zip'
HR_FOLDER = 'DIV2K_valid_HR'
LR_URLS = {
    2: (f'{BASE}/DIV2K_valid_LR_bicubic_X2.zip', 'DIV2K_valid_LR_bicubic/X2', '0801x2.png'),
    3: (f'{BASE}/DIV2K_valid_LR_bicubic_X3.zip', 'DIV2K_valid_LR_bicubic/X3', '0801x3.png'),
    4: (f'{BASE}/DIV2K_valid_LR_bicubic_X4.zip', 'DIV2K_valid_LR_bicubic/X4', '0801x4.png'),
}

CATEGORY = 'Vision/Image Super-Resolution'


def _download(url: str, dest: Path) -> None:
    """Stream-download with periodic progress."""
    print(f'  downloading {url}', flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'BenchHub-importer'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get('Content-Length') or 0)
        got = 0
        with open(dest, 'wb') as fh:
            while True:
                buf = resp.read(1 << 20)  # 1 MB
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


def _extract(zip_path: Path, out_dir: Path) -> Path:
    """Unzip, return the inner folder containing the .png files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # DIV2K_valid_HR/*.png  OR  DIV2K_valid_LR_bicubic/X2/*.png
    return out_dir


def _materialize_bh_layout(hr_root: Path, lr_root: Path, scale: int,
                           out_dir: Path) -> int:
    """Lay out paired LR + HR images in BH folder convention."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'image_lr').mkdir(exist_ok=True)
    (out_dir / 'image_hr').mkdir(exist_ok=True)

    (out_dir / 'README.md').write_text(
        f"# DIV2K validation, ×{scale}\n\n"
        f"Imported from EE Zurich (http://data.vision.ee.ethz.ch/cvl/DIV2K/). "
        f"100 image pairs from the official validation split; LR is "
        f"bicubically downsampled by ×{scale}.\n\n"
        f"image_lr/<id>.png  — low-resolution input (given to submitter)\n"
        f"image_hr/<id>.png  — high-resolution ground truth (held by server)\n"
    )

    # Find the actual subfolders. DIV2K HR archive extracts to
    # DIV2K_valid_HR/0801.png ... 0900.png; LR archive extracts to
    # DIV2K_valid_LR_bicubic/X{scale}/0801x{scale}.png ...
    hr_files = sorted(hr_root.glob('DIV2K_valid_HR/*.png'))
    lr_subdir = lr_root / 'DIV2K_valid_LR_bicubic' / f'X{scale}'
    if not lr_subdir.exists():
        # Some archives extract straight into X2/X3/X4 without the
        # DIV2K_valid_LR_bicubic wrapper. Handle both shapes.
        lr_subdir = lr_root / f'X{scale}'
    if not lr_subdir.exists():
        raise RuntimeError(f"LR subdir not found under {lr_root}: tried "
                           f"DIV2K_valid_LR_bicubic/X{scale} and X{scale}")
    lr_files = sorted(lr_subdir.glob('*.png'))

    if len(hr_files) != len(lr_files):
        print(f'  WARN: HR count {len(hr_files)} != LR count {len(lr_files)}')

    # Sample id = the 4-digit DIV2K image number (0801..0900). HR files
    # are named e.g. 0801.png; LR files are named 0801x2.png. We
    # canonicalise on the HR-style name and rewrite the LR filename to
    # match.
    n = 0
    by_id = {}
    for hp in hr_files:
        sid = hp.stem  # '0801'
        by_id[sid] = {'hr': hp}
    for lp in lr_files:
        # '0801x2.png' -> '0801'
        sid = lp.stem.split('x', 1)[0]
        if sid in by_id:
            by_id[sid]['lr'] = lp
    for sid, pair in sorted(by_id.items()):
        if 'hr' not in pair or 'lr' not in pair:
            print(f'  WARN: incomplete pair for sample {sid}, skipping')
            continue
        # Use plain {sid}.png on both sides (mirrors NYU/imagenet
        # conventions, no scale suffix in the BH dataset).
        shutil.copy(pair['hr'], out_dir / 'image_hr' / f'{sid}.png')
        shutil.copy(pair['lr'], out_dir / 'image_lr' / f'{sid}.png')
        n += 1
    return n


def _zip_folder(folder: Path, dest_zip: Path) -> None:
    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in folder.rglob('*'):
            if path.is_file():
                arc = path.relative_to(folder)
                zf.write(path, arcname=str(arc))


def _create_lb(app_mod, dataset, scale: int):
    """Create the matching Leaderboard + Attachment + 2 LeaderboardMetric
    rows (PSNR + SSIM) for one DIV2K scale dataset."""
    name = f'Image Super-Resolution on DIV2K (×{scale} valid)'

    # Skip if it already exists.
    existing = app_mod.Leaderboard.query.filter_by(name=name).first()
    if existing:
        print(f'  LB "{name}" already exists (id={existing.id}) — skipping')
        return existing

    lb = app_mod.Leaderboard(
        name=name,
        dataset_id=dataset.id,  # legacy back-compat column
        category=CATEGORY,
        summary_metrics='',  # filled below after metric rows have ids
        visibility='public',
        owner_user_id=None,
    )
    # `required_pred_fields_json` declares what the submitter ships in
    # their ZIP. For super-resolution they ship the predicted HR image,
    # one per sample, named to match the dataset's sample ids.
    import json as _json
    lb.required_pred_fields_json = _json.dumps([
        {
            'name': 'hr_pred',
            'kind': 'image',
            'description': f'Super-resolved (×{scale}) reconstruction of '
                           f'the LR input. Must match HR resolution.',
        },
    ])
    app_mod.db.session.add(lb)
    app_mod.db.session.flush()  # get lb.id

    # Attachment: BH-side, pointing at the new Dataset.
    att = app_mod.Attachment(
        leaderboard_id=lb.id,
        dataset_id=dataset.id,
        role='primary',
    )
    app_mod.db.session.add(att)

    # Metrics: PSNR + SSIM, both image×image. The arg_mappings tie the
    # function args to the per-sample context keys:
    #   gt   -> gt_hr           (GT high-res image from the dataset)
    #   pred -> sub_hr_pred     (submission-supplied SR output)
    psnr = app_mod.GlobalMetric.query.filter_by(name='psnr').first()
    ssim = app_mod.GlobalMetric.query.filter_by(name='ssim').first()
    if psnr is None or ssim is None:
        raise RuntimeError('GlobalMetric psnr or ssim missing — abort.')

    lm_psnr = app_mod.LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=psnr.id,
        arg_mappings=_json.dumps({'gt': 'gt_hr', 'pred': 'sub_hr_pred'}),
        target_name='PSNR',
        pooling_type='mean',
        sort_direction='higher_is_better',
    )
    lm_ssim = app_mod.LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=ssim.id,
        arg_mappings=_json.dumps({'gt': 'gt_hr', 'pred': 'sub_hr_pred'}),
        target_name='SSIM',
        pooling_type='mean',
        sort_direction='higher_is_better',
    )
    app_mod.db.session.add_all([lm_psnr, lm_ssim])
    app_mod.db.session.flush()
    # summary_metrics points at the LB-metric ids so they render on the
    # leaderboard table by default.
    lb.summary_metrics = f'lm_{lm_psnr.id},lm_{lm_ssim.id}'
    app_mod.db.session.commit()
    print(f'  -> created LB id={lb.id} ({name})')
    return lb


def _import_one_scale(scale: int, work_root: Path, hr_zip: Path,
                      hr_extracted: Path, app_mod, force: bool) -> None:
    print(f'\n=== DIV2K valid ×{scale} ===', flush=True)
    dataset_name = f'div2k-valid-x{scale}'

    # Existing-dataset gate.
    existing = app_mod.Dataset.query.filter_by(name=dataset_name).first()
    if existing and not force:
        print(f'  Dataset exists (id={existing.id}) — skipping import; '
              f'creating LB only if missing.')
        dataset = existing
    else:
        if existing and force:
            print(f'  Dataset exists (id={existing.id}) — --force given, '
                  f'deleting first')
            app_mod.db.session.delete(existing)
            app_mod.db.session.commit()
        lr_url, lr_inner, _ = LR_URLS[scale]
        lr_zip = work_root / f'lr_x{scale}.zip'
        lr_extracted = work_root / f'lr_x{scale}_extract'
        if not lr_zip.exists() or lr_zip.stat().st_size == 0:
            _download(lr_url, lr_zip)
        _extract(lr_zip, lr_extracted)

        layout_dir = work_root / f'layout_x{scale}'
        n = _materialize_bh_layout(hr_extracted, lr_extracted, scale, layout_dir)
        print(f'  wrote {n} samples')

        zip_path = work_root / f'{dataset_name}.zip'
        _zip_folder(layout_dir, zip_path)
        print(f'  zip size: {zip_path.stat().st_size/1e6:.1f} MB')

        ok, msg, ds_id = app_mod.process_dataset_zip(
            str(zip_path), dataset_name,
            owner_user_id=None, category=CATEGORY,
        )
        if not ok:
            print(f'  FAILED: {msg}')
            return
        print(f'  -> Dataset id={ds_id}')
        dataset = app_mod.db.session.get(app_mod.Dataset, ds_id)
        # Tidy up the per-scale workdir.
        shutil.rmtree(lr_extracted, ignore_errors=True)
        shutil.rmtree(layout_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        lr_zip.unlink(missing_ok=True)

    _create_lb(app_mod, dataset, scale)


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Import DIV2K valid set from EE Zurich to BenchHub.',
    )
    ap.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4],
                    choices=[2, 3, 4],
                    help='Which ×N variants to import (default: 2 3 4).')
    ap.add_argument('--force', action='store_true',
                    help='Re-import existing datasets.')
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--keep-workdir', action='store_true')
    args = ap.parse_args()

    import app as app_mod

    work_root = Path(args.workdir or tempfile.mkdtemp(prefix='benchhub-div2k-'))
    print(f'Work dir: {work_root}', flush=True)

    # HR archive is shared across all scales, fetch once.
    hr_zip = work_root / 'hr.zip'
    hr_extracted = work_root / 'hr_extract'
    with app_mod.app.app_context():
        if not hr_zip.exists() or hr_zip.stat().st_size == 0:
            _download(HR_URL, hr_zip)
        _extract(hr_zip, hr_extracted)
        for scale in args.scales:
            _import_one_scale(scale, work_root, hr_zip, hr_extracted,
                              app_mod, args.force)

    if not args.keep_workdir:
        shutil.rmtree(work_root, ignore_errors=True)
        print('\nWork dir cleaned up.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
