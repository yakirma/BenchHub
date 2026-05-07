"""Generic baseline-seeder for a BenchHub leaderboard.

Walks a list of model specs, runs each on a downloaded GT split, packages
the predictions into a BenchHub submission ZIP, and POSTs it to the
upload API with `source_colab_url` so the submission table back-links to
the notebook that produced it.

Domain-specific stubs (`baselines_depth.py`, `baselines_segmentation.py`,
`baselines_language.py`, `baselines_denoising.py`) define `MODELS` (a
list of model specs) and a `predictor_fn(model, processor, sample_inputs)`
that returns a dict keyed by the LB's pred-field names. This file just
runs them.

Designed to be copy-pasted into a Colab cell or run as a local script.
You provide GPU access — this script provides the bookkeeping.

Example (depth):

    from baselines_depth import MODELS, predictor_fn, load_inputs
    seed_baselines(
        leaderboard_id=42,
        api_token=os.environ['BENCHHUB_API_TOKEN'],
        gt_zip_url='https://benchhub.fly.dev/dataset/17/download',
        models=MODELS,
        predictor_fn=predictor_fn,
        load_inputs=load_inputs,
        source_colab_url=os.environ.get('SOURCE_COLAB_URL', ''),
    )
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable


def _download_and_extract(zip_url: str, dest: Path) -> None:
    """Pull the GT ZIP and extract it under `dest`. Re-uses an existing
    extraction if present so re-running the cell is cheap."""
    marker = dest / '.extracted'
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {zip_url} ...")
    zpath = dest / '_gt.zip'
    urllib.request.urlretrieve(zip_url, str(zpath))
    print(f"  extracting ...")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(str(dest))
    zpath.unlink()
    marker.write_text('1')


def _discover_sample_names(gt_root: Path) -> list[str]:
    """The submission contract is one .txt / .png / .npz per sample, so
    the canonical sample-name list is whatever shows up under any GT
    folder. Strip extensions + _<W>x<H> dim suffixes."""
    seen = set()
    for sub in gt_root.iterdir():
        if not sub.is_dir() or sub.name.startswith('.') or sub.name == '__MACOSX':
            continue
        for f in sub.iterdir():
            stem = f.stem
            # raw_<col>/<sample>_<W>x<H>.npz — strip the dims tail.
            if '_' in stem and stem.rsplit('_', 1)[-1].count('x') == 1:
                head, tail = stem.rsplit('_', 1)
                if 'x' in tail and all(p.isdigit() for p in tail.split('x')):
                    stem = head
            seen.add(stem)
    return sorted(seen)


def _write_prediction(out_root: Path, field: str, sample: str, value) -> None:
    """Write a single per-sample prediction to <field>/<sample>.txt.
    BARE-NAME folders only — `metric_*` is reserved for user-precomputed
    metric values, not raw predictions. Numeric scalars go to .txt;
    other types should round-trip through str()."""
    folder = out_root / field
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{sample}.txt').write_text(str(value))


def _zip_submission(submission_dir: Path) -> Path:
    """Pack the submission folder for upload. Returns the zip path."""
    base = submission_dir.parent / submission_dir.name
    archive = shutil.make_archive(str(base), 'zip', root_dir=str(submission_dir))
    return Path(archive)


def _upload(zip_path: Path, *, leaderboard_id: int, api_token: str,
            base_url: str, submission_name: str,
            source_colab_url: str = '') -> dict:
    """POST to /api/leaderboard/<id>/submission/upload."""
    import requests  # local import — keeps the script Colab-friendly
    url = f"{base_url.rstrip('/')}/api/leaderboard/{leaderboard_id}/submission/upload"
    with open(zip_path, 'rb') as fh:
        resp = requests.post(
            url,
            headers={'Authorization': f'Bearer {api_token}'},
            data={
                'submission_name': submission_name,
                'source_colab_url': source_colab_url,
            },
            files={'submission_zip': (zip_path.name, fh)},
            timeout=300,
        )
    try:
        body = resp.json()
    except ValueError:
        body = {'raw': resp.text[:200]}
    return {'status_code': resp.status_code, 'body': body}


def seed_baselines(
    *,
    leaderboard_id: int,
    api_token: str,
    gt_zip_url: str,
    models: Iterable[dict],
    predictor_fn: Callable,
    load_inputs: Callable[[Path, str], dict] | None = None,
    base_url: str = 'https://benchhub.fly.dev',
    source_colab_url: str = '',
    work_dir: str | None = None,
) -> list[dict]:
    """Run every model spec against the GT and upload one submission
    each. Returns a list of {model, status_code, body, n_samples} dicts.

    `models` is an iterable of dicts with at least `name` (becomes the
    submission_name) and whatever keys `predictor_fn` needs (typically
    `repo_id` + a `load` callable that returns (model, processor)).

    `predictor_fn(spec, model, processor, sample_inputs)` must return a
    dict keyed by the LB's pred-field names (the ones surfaced by the
    LB page's submission contract — typically `<gt_col>_pred`).

    `load_inputs(gt_root, sample_name)` reads whatever GT files this
    domain needs and returns a kwargs dict for the predictor. If None,
    we pass the GT root path + sample name through and let the
    predictor read what it needs.
    """
    work = Path(work_dir or tempfile.mkdtemp(prefix='benchhub-seed-'))
    gt_root = work / 'gt'
    _download_and_extract(gt_zip_url, gt_root)

    sample_names = _discover_sample_names(gt_root)
    if not sample_names:
        raise RuntimeError(f"No samples found under {gt_root}")
    print(f"Found {len(sample_names)} samples")

    results = []
    for spec in models:
        name = spec['name']
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            model, processor = spec['load']()
        except Exception as e:
            print(f"  [skip] load failed: {e}")
            results.append({'model': name, 'status_code': None,
                            'body': {'error': f'load failed: {e}'}})
            continue

        sub_dir = work / f'submission_{name}'
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir()

        n = 0
        for sn in sample_names:
            try:
                if load_inputs is not None:
                    inputs = load_inputs(gt_root, sn)
                else:
                    inputs = {'gt_root': gt_root, 'sample_name': sn}
                preds = predictor_fn(spec, model, processor, inputs)
                if not isinstance(preds, dict):
                    raise TypeError(
                        f'predictor_fn returned {type(preds).__name__}, '
                        f'expected dict of pred_field -> value'
                    )
                for field, value in preds.items():
                    _write_prediction(sub_dir, field, sn, value)
                n += 1
            except Exception as e:
                print(f"  [warn] sample {sn}: {e}")
                continue

        if n == 0:
            print(f"  [skip] zero successful predictions; no upload")
            results.append({'model': name, 'status_code': None,
                            'body': {'error': 'all samples failed'}})
            continue

        zip_path = _zip_submission(sub_dir)
        print(f"  uploading {zip_path.name} ({n} samples)")
        upload_result = _upload(
            zip_path,
            leaderboard_id=leaderboard_id,
            api_token=api_token,
            base_url=base_url,
            submission_name=name,
            source_colab_url=source_colab_url,
        )
        upload_result['model'] = name
        upload_result['n_samples'] = n
        upload_result['elapsed_s'] = round(time.time() - t0, 1)
        print(f"  → {upload_result['status_code']}: {upload_result['body']}")
        results.append(upload_result)

    return results


# ---------------------------------------------------------------------------
# CLI entry — for one-off runs from a terminal. Colab users typically
# call seed_baselines() directly from a notebook cell instead.
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leaderboard-id', type=int, required=True)
    parser.add_argument('--api-token', default=os.environ.get('BENCHHUB_API_TOKEN'))
    parser.add_argument('--gt-zip-url', required=True)
    parser.add_argument('--base-url', default='https://benchhub.fly.dev')
    parser.add_argument('--source-colab-url', default='')
    parser.add_argument('--domain', required=True,
                        choices=['depth', 'segmentation', 'language', 'denoising'])
    args = parser.parse_args()
    if not args.api_token:
        raise SystemExit("API token required (--api-token or $BENCHHUB_API_TOKEN).")

    if args.domain == 'depth':
        from baselines_depth import MODELS, predictor_fn, load_inputs
    elif args.domain == 'segmentation':
        from baselines_segmentation import MODELS, predictor_fn, load_inputs
    elif args.domain == 'language':
        from baselines_language import MODELS, predictor_fn, load_inputs
    elif args.domain == 'denoising':
        from baselines_denoising import MODELS, predictor_fn, load_inputs
    else:  # pragma: no cover - argparse validated above
        raise SystemExit(f"unknown domain: {args.domain}")

    results = seed_baselines(
        leaderboard_id=args.leaderboard_id,
        api_token=args.api_token,
        gt_zip_url=args.gt_zip_url,
        models=MODELS,
        predictor_fn=predictor_fn,
        load_inputs=load_inputs,
        base_url=args.base_url,
        source_colab_url=args.source_colab_url,
    )
    print("\n=== Summary ===")
    print(json.dumps(results, indent=2, default=str))
