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
import tempfile
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
from PIL import Image as PILImage


# ----- path resolution -----

def materialization_dir(upload_folder: str | os.PathLike, lb_id: int) -> Path:
    return Path(upload_folder) / 'lb_materializations' / str(lb_id)


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

def materialize_for_lb(
    *,
    leaderboard,
    dataset,
    db_session,
    upload_folder: str | os.PathLike,
    CustomField,
    LeaderboardMaterialization,
) -> dict:
    """Run materialisation for the given LB. Reads its
    `LeaderboardMaterialization` row, picks samples, re-fetches full
    bytes from HF, writes to disk, updates the row. Returns a summary.

    Assumes: dataset is HF-sourced (source_metadata.repo_id present)
    and preview_only=True. Non-HF datasets are skipped — they were
    full-storage to begin with so the materialisation is the
    dataset's own files.
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
    split = meta.get('split')
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
    db_session.commit()

    # Resolve stratify groups from the dataset's existing CustomField rows.
    sample_objs = sorted(dataset.samples, key=lambda s: s.name)
    sample_names = [s.name for s in sample_objs]
    stratify_groups = None
    if matrow.sampling == 'stratified' and matrow.stratify_field:
        groups = []
        for s in sample_objs:
            cf = next((c for c in s.custom_fields
                       if c.name == matrow.stratify_field), None)
            groups.append(cf.value_text if cf else None)
        stratify_groups = groups

    chosen_idx = pick_samples(
        sample_names,
        sample_cap=matrow.sample_cap,
        sampling=matrow.sampling,
        seed=matrow.sampling_seed,
        stratify_groups=stratify_groups,
    )
    chosen_set = set(chosen_idx)
    chosen_names = {sample_names[i] for i in chosen_idx}

    # Re-derive the field list the dataset was originally imported
    # with so materialize_hf_to_typed_dir can pick the right HF
    # columns. The dataset's DatasetField rows give name+kind+role;
    # source_column defaults to name (HF column names are sanitised
    # to match field names at import time).
    fields = []
    for f in dataset.dataset_fields:
        if f.role == 'pred':
            continue
        prms = {}
        try: prms = json.loads(f.params) if f.params else {}
        except Exception: pass
        fields.append({
            'name': f.name,
            'source_column': f.name,
            'kind': f.kind,
            'role': f.role,
            'params': prms,
        })

    # Materialise the FULL split via HF datasets-server, then
    # filter to the chosen indices. We accept the temporary on-disk
    # cost during materialisation; only the chosen files survive to
    # uploads/lb_materializations/.
    out_dir = materialization_dir(upload_folder, leaderboard.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='bh_lbmat_') as staging:
        mat = materialize_hf_to_typed_dir(
            repo_id=repo_id, split=split,
            sample_cap=-1,   # full split — we filter locally
            staging_dir=staging,
            dataset_name=dataset.name,
            fields=fields, hf_token=None,
            sampling='head', seed=42, sample_name_from=None,
            # IMPORTANT: do NOT pass preview_only=True here. Stage C
            # materialisations are full-resolution.
        )

        # Copy chosen samples' files into uploads/lb_materializations/<lb_id>/.
        staging_path = Path(staging)
        copied = 0
        for f in fields:
            field_dir = out_dir / f['name']
            src_field_dir = staging_path / f['name']
            if not src_field_dir.is_dir():
                continue
            field_dir.mkdir(parents=True, exist_ok=True)
            for src in src_field_dir.iterdir():
                # File name = "<sample>.<ext>" — use the stem as sample name.
                if src.stem not in chosen_names:
                    continue
                dst = field_dir / src.name
                dst.write_bytes(src.read_bytes())
                copied += 1

        # Tally bytes on disk
        total_bytes = sum(p.stat().st_size for p in out_dir.rglob('*') if p.is_file())

    matrow.status = 'ready'
    matrow.materialized_at = datetime.utcnow()
    matrow.storage_bytes = total_bytes
    db_session.commit()
    return {
        'status': 'ready',
        'samples_picked': len(chosen_idx),
        'files_copied': copied,
        'bytes': total_bytes,
    }
