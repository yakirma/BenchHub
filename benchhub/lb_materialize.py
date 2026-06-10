"""Stage C: full-resolution re-materialisation of preview-only
datasets, scoped per-Leaderboard.

A `LeaderboardMaterialization` row stores the subset choice
(sample_cap, sampling, seed, stratify_field). This module picks the
samples, re-fetches their full bytes from upstream HF, and writes
them under `uploads/lb_materializations/<lb_id>/<field>/<sample>.<ext>`.

The dataset's preview tier is never touched.

Submission scoring + LB-attached visualisations then prefer the
materialised path; if the materialisation doesn't exist they fall
back to the dataset's preview file via
`materialized_or_preview_path()`.
"""
from __future__ import annotations
import os
import json
import io
import time
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
from PIL import Image as PILImage


# ----- path resolution -----

def materialization_dir(upload_folder: str | os.PathLike, lb_id: int) -> Path:
    return Path(upload_folder) / 'lb_materializations' / str(lb_id)


def list_materialized_samples(
    upload_folder: str | os.PathLike, lb_id: int,
) -> list[str]:
    """Sample names present in this LB's materialisation. Reads the
    first non-empty per-field subdirectory and strips extensions.
    Returns [] if no materialisation directory exists yet."""
    base = materialization_dir(upload_folder, lb_id)
    if not base.is_dir():
        return []
    for field_dir in sorted(base.iterdir()):
        if not field_dir.is_dir():
            continue
        names = sorted({p.stem for p in field_dir.iterdir() if p.is_file()})
        if names:
            return names
    return []


def materialized_or_preview_path(
    upload_folder: str | os.PathLike,
    leaderboard_id: int | None,
    field_name: str,
    sample_name: str,
    preview_rel_path: str,
) -> str:
    """Resolve a per-sample field file path. Prefers the LB's
    materialised file when present; else returns the dataset-tier
    preview path (relative to upload_folder).

    Returns a path RELATIVE to upload_folder (same shape as
    CustomField.value_text).
    """
    if leaderboard_id is not None:
        mdir = materialization_dir(upload_folder, leaderboard_id) / field_name
        if mdir.is_dir():
            # Use whichever extension the materialiser actually
            # wrote (could be .png, .npz, .wav). The sample name is
            # the canonical join key.
            for ext in ('.png', '.jpg', '.jpeg', '.npz', '.wav', '.mp3',
                         '.json', '.txt'):
                candidate = mdir / f'{sample_name}{ext}'
                if candidate.exists():
                    return str(candidate.relative_to(upload_folder))
    return preview_rel_path


# ----- sample picking -----

def pick_samples(
    sample_names: list[str],
    *,
    sample_cap: int,
    sampling: str,
    seed: int,
    stratify_groups: list[Any] | None = None,
) -> list[int]:
    """Return a list of indices into `sample_names` representing the
    chosen subset, in canonical order.

    `stratify_groups`: parallel list of group labels per sample (e.g.
    the class id). Only used when sampling='stratified'.
    """
    n = len(sample_names)
    cap = min(int(sample_cap), n) if sample_cap > 0 else n
    if sampling == 'head' or cap >= n:
        return list(range(cap))
    rng = random.Random(seed)
    if sampling == 'random':
        return sorted(rng.sample(range(n), cap))
    if sampling == 'stratified':
        if not stratify_groups or len(stratify_groups) != n:
            # Fall back to random when groups are missing.
            return sorted(rng.sample(range(n), cap))
        # Bucket indices by group, allocate proportional to group
        # size, round to nearest integer with a minimum of 1 per
        # non-empty group when cap permits.
        buckets: dict[Any, list[int]] = {}
        for i, g in enumerate(stratify_groups):
            buckets.setdefault(g, []).append(i)
        groups = list(buckets.keys())
        alloc: dict[Any, int] = {}
        rem = cap
        for g in groups:
            share = max(1, round(cap * len(buckets[g]) / n))
            alloc[g] = min(share, len(buckets[g]))
            rem -= alloc[g]
        # Distribute / claw back the rounding remainder.
        i = 0
        while rem != 0 and groups:
            g = groups[i % len(groups)]
            if rem > 0 and alloc[g] < len(buckets[g]):
                alloc[g] += 1; rem -= 1
            elif rem < 0 and alloc[g] > 1:
                alloc[g] -= 1; rem += 1
            else:
                # Avoid infinite loop if no group can absorb the diff.
                i += 1
                if i > 4 * len(groups):
                    break
                continue
            i += 1
        chosen: list[int] = []
        for g in groups:
            chosen.extend(rng.sample(buckets[g], alloc[g]))
        return sorted(chosen)
    raise ValueError(f'unknown sampling strategy: {sampling!r}')


# ----- the materialiser -----

def _write_lb_scoped_gt(
    staging_dir: str | os.PathLike,
    fields: list[dict],
    sample_names: list[str],
    *,
    leaderboard_id: int,
    upload_folder: str | os.PathLike,
    db_session,
    CustomField,
    out_dir: Path,
) -> tuple[int, int]:
    """Convert a full-resolution typed staging dir into the LB's OWN GT
    set: per (sample, field) it writes an **LB-scoped** CustomField row
    (`leaderboard_id` set, `sample_id`/`submission_id` NULL, keyed by
    `sample_name`) — the same shape `import_typed_dataset` writes for a
    dataset, but tied to the leaderboard instead of dataset Sample rows.
    File-backed kinds are copied full-res under
    `uploads/lb_materializations/<lb_id>/<field>/`; inline kinds decode
    into value_float/value_text. Returns (cf_rows, files_copied).

    Mirrors manifest.import_typed_dataset's per-kind branch so the eval +
    input paths read these CFs identically to dataset CFs."""
    from benchhub.types import DTYPES
    from benchhub.manifest import expected_file_path

    src_root = Path(staging_dir)
    n_cf = files_copied = 0
    for f in fields:
        if f.get('role', 'gt') == 'pred':
            continue
        kind = f['kind']
        params = f.get('params') or {}
        cls = DTYPES.get(kind)
        # Inline kinds (scalar/label/label_list) have file_ext None.
        # Registered (non-DTYPES) kinds are stored file-backed verbatim.
        is_inline = (cls is not None and cls.file_ext is None)
        field_dir = out_dir / f['name']
        if not is_inline:
            field_dir.mkdir(parents=True, exist_ok=True)
        for s_name in sample_names:
            src = expected_file_path(src_root, f, s_name)
            if not src.exists():
                continue
            cf = CustomField(
                leaderboard_id=leaderboard_id, sample_id=None,
                submission_id=None, sample_name=s_name,
                name=f['name'], data_type=kind,
                source_column=f.get('source_column') or f['name'],
            )
            if params:
                cf.set_params(params)
            if is_inline:
                blob = src.read_bytes()
                if cls is None:
                    cf.value_text = blob.decode('utf-8', 'replace').rstrip('\n')
                else:
                    inst = cls.decode(blob, params)
                    if kind == 'scalar':
                        cf.value_float = float(inst.value)
                    else:
                        cf.value_text = blob.decode('utf-8').rstrip('\n')
            else:
                dst = field_dir / src.name
                shutil.copy2(src, dst)
                files_copied += 1
                cf.value_text = str(dst.relative_to(upload_folder))
                # text/json render their CONTENT from value_text (not the
                # path) — mirror import_typed_dataset's override.
                if kind in ('text', 'json'):
                    try:
                        cf.value_text = dst.read_text(encoding='utf-8').rstrip('\n')
                    except (OSError, UnicodeDecodeError):
                        pass
            db_session.add(cf)
            n_cf += 1
    return n_cf, files_copied


def materialize_for_lb(
    *,
    leaderboard,
    dataset,
    db_session,
    upload_folder: str | os.PathLike,
    CustomField,
    LeaderboardMaterialization,
    progress_cb=None,
) -> dict:
    """Materialise the LB's evaluation set, DECOUPLED from the dataset's
    preview cache. Reads the `LeaderboardMaterialization` row and fetches
    from the SOURCE split directly using the LB's own parameters
    (sample_cap / sampling / seed / shard_cap / split / config), so the LB
    can sample rows the dataset never cached. The chosen samples are written
    as an LB-scoped GT set (full-res files under
    `uploads/lb_materializations/<lb_id>/` + LB-scoped CustomField rows);
    the dataset's preview tier is never touched.

    Non-HF datasets are a noop — they were full-storage to begin with, so
    LB scoring falls back to the dataset's own files.
    """
    # Lazy import — avoid pulling materialize_hf_to_typed_dir into
    # the import path of every module that uses path resolution.
    from benchhub.hf_materialize import materialize_hf_to_typed_dir

    matrow = leaderboard.materialization
    if matrow is None:
        raise ValueError(f'LB {leaderboard.id} has no LeaderboardMaterialization row')

    meta = {}
    try:
        meta = json.loads(dataset.source_metadata or '{}')
    except Exception:
        pass
    repo_id = meta.get('repo_id')
    # LB-level split/config override the dataset's; inherit when unset.
    split = matrow.split or meta.get('split')
    config_name = matrow.config_name or meta.get('config_name') or None
    if not (repo_id and split):
        # Non-HF dataset; nothing to materialise. The LB scoring will
        # fall back to the dataset's own files (which were always full
        # storage).
        matrow.status = 'ready'
        matrow.materialized_at = datetime.utcnow()
        matrow.storage_bytes = 0
        db_session.commit()
        return {'status': 'noop', 'reason': 'non-HF dataset'}

    matrow.status = 'running'
    matrow.error_message = None
    db_session.commit()

    def _progress(phase: str, current: int = 0, total: int = 0, message: str = ''):
        """Persist progress on the matrow + delegate to the optional
        external callback (used by the Celery task to publish state)."""
        payload = {'phase': phase, 'current': current,
                   'total': total, 'message': message}
        matrow.progress_json = json.dumps(payload)
        db_session.commit()
        if progress_cb is not None:
            try:
                progress_cb(payload)
            except Exception:
                pass  # progress reporting is best-effort

    _progress('starting', 0, matrow.sample_cap,
              'Resolving the LB sample set from the source split…')

    # Re-derive the field list the dataset was imported with so the
    # materializer picks the right source columns. DatasetField rows give
    # name+kind+role; source_column defaults to the field name (HF column
    # names are sanitised to match field names at import time).
    fields = []
    for f in dataset.dataset_fields:
        if f.role == 'pred':
            continue
        prms = {}
        try:
            prms = json.loads(f.params) if f.params else {}
        except Exception:
            pass
        fields.append({
            'name': f.name,
            'source_column': f.name,
            'kind': f.kind,
            'role': f.role,
            'params': prms,
        })

    # shard_cap drives how much of the split we pull: -1 = all shards (true
    # whole-split sampling — needed for unbiased random/stratified), 0 =
    # auto (just enough for the cap, head-biased), N = first N.
    shard_cap = matrow.shard_cap if matrow.shard_cap is not None else -1
    out_dir = materialization_dir(upload_folder, leaderboard.id)
    # Idempotent re-materialise: clear prior files + LB-scoped CFs.
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    CustomField.query.filter_by(
        leaderboard_id=leaderboard.id, sample_id=None, submission_id=None,
    ).delete()
    db_session.commit()

    _progress('downloading', 0, matrow.sample_cap,
              f'Sampling {matrow.sample_cap} from {repo_id}[{split}]…')
    with tempfile.TemporaryDirectory(prefix='bh_lbmat_') as staging:
        def _stage_progress(state: dict):
            _progress(
                state.get('phase') or 'downloading',
                current=state.get('current') or 0,
                total=state.get('total') or matrow.sample_cap,
                message=state.get('message') or '',
            )

        # The materializer does the sampling over the WHOLE fetched split
        # (sample_cap + strategy), so the LB's set is independent of the
        # dataset's preview cache. Full-resolution: NO preview_only.
        # Vocabulary bridge: the LeaderboardMaterialization model + wizard use
        # 'random', but materialize_hf_to_typed_dir's _pick_indices speaks
        # 'uniform' for the same seeded-random strategy — translate or every
        # 'random' materialisation (the wizard default) raises "unknown
        # sampling strategy 'random'".
        _pick_sampling = {'random': 'uniform'}.get(matrow.sampling, matrow.sampling)
        materialize_hf_to_typed_dir(
            repo_id=repo_id, split=split,
            sample_cap=matrow.sample_cap,
            shard_cap=shard_cap,
            staging_dir=staging,
            dataset_name=dataset.name,
            fields=fields, hf_token=None,
            sampling=_pick_sampling, seed=matrow.sampling_seed,
            sample_name_from=meta.get('sample_name_from'),
            config_name=config_name,
            progress_cb=_stage_progress,
        )

        manifest = json.loads((Path(staging) / 'manifest.json').read_text())
        sample_names = manifest.get('samples', [])
        _progress('copying', 0, len(sample_names),
                  f'Writing {len(sample_names)} LB-scoped samples…')
        n_cf, copied = _write_lb_scoped_gt(
            staging, fields, sample_names,
            leaderboard_id=leaderboard.id, upload_folder=upload_folder,
            db_session=db_session, CustomField=CustomField, out_dir=out_dir,
        )
        total_bytes = sum(p.stat().st_size for p in out_dir.rglob('*') if p.is_file())

    matrow.status = 'ready'
    matrow.materialized_at = datetime.utcnow()
    matrow.storage_bytes = total_bytes
    _progress('done', len(sample_names), len(sample_names),
              f'Materialised {len(sample_names)} samples ({copied} files, '
              f'{n_cf} GT rows).')
    db_session.commit()
    return {
        'status': 'ready',
        'samples_picked': len(sample_names),
        'files_copied': copied,
        'gt_rows': n_cf,
        'bytes': total_bytes,
    }
