#!/usr/bin/env python
"""Re-cache a classification dataset's preview tier with full class coverage.

Datasets imported with `head` sampling on a class-sorted HF split end up
caching only the first few classes (e.g. yoga-poses-107 cached 18/107).
This re-runs the HF materialize for an EXISTING dataset_id with the
classification coverage policy now baked into
`benchhub.hf_materialize.materialize_hf_to_typed_dir` — stratified
sampling, cap = max(2000, 10 × n_classes) — swapping the dataset's
Sample + CustomField + DatasetField rows + on-disk previews in place.

LB-scoped GT CustomFields (sample_id IS NULL) are NOT touched, so an
LB's materialised eval set is unaffected (re-materialise the LB
separately if it was built from a skewed cache).

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/recache_classification.py <dataset_id> [<dataset_id> ...]

Prints one `RECACHE_RESULT ...` line per dataset (or `RECACHE_SKIP`/`RECACHE_FAIL`).
"""
import os
import sys
import json
import shutil
import tempfile

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))


def _reconstruct_fields(fields_rows):
    """Rebuild the `fields` list materialize_hf_to_typed_dir expects from
    the dataset's DatasetField rows. source_column defaults to the field
    name (true for the standard HF import where field name == HF column).
    Label fields with the most classes are ordered first so the
    stratified picker keys on the finest granularity."""
    out = []
    for f in fields_rows:
        out.append({
            'name': f.name,
            'source_column': f.name,
            'kind': f.kind,
            'role': f.role or 'gt',
            'params': f.get_params() or {},
        })

    def _nc(fd):
        if fd['kind'] in ('label', 'labellist'):
            return len((fd['params'] or {}).get('names') or [])
        return -1
    out.sort(key=_nc, reverse=True)
    return out


def recache_one(ds_id):
    import app as A
    from app import (db, Dataset, DatasetField, Sample, CustomField,
                     _registered_extra_kinds, check_quota, User, _format_bytes)
    from benchhub.manifest import import_typed_dataset
    from benchhub.hf_materialize import materialize_hf_to_typed_dir

    with A.app.app_context():
        ds = Dataset.query.get(ds_id)
        if ds is None:
            print(f'RECACHE_SKIP {ds_id} not found'); return
        if (ds.source_kind or '') != 'hf':
            print(f'RECACHE_SKIP {ds_id} ({ds.name}) source_kind={ds.source_kind!r}, not hf'); return
        md = {}
        try:
            md = json.loads(ds.source_metadata or '{}')
        except Exception:
            md = {}
        repo_id = md.get('repo_id')
        if not repo_id:
            print(f'RECACHE_SKIP {ds_id} ({ds.name}) no repo_id in metadata'); return
        split = md.get('split')
        config_name = md.get('config_name')

        field_rows = DatasetField.query.filter_by(dataset_id=ds_id).all()
        fields = _reconstruct_fields(field_rows)
        label_fields = [f for f in fields if f['kind'] in ('label', 'labellist')]
        if not label_fields:
            print(f'RECACHE_SKIP {ds_id} ({ds.name}) no label field'); return
        n_classes = max(len((f['params'] or {}).get('names') or []) for f in label_fields)
        policy_cap = max(2000, 10 * n_classes) if n_classes else 2000

        owner = User.query.get(ds.owner_user_id) if ds.owner_user_id else None
        hf_token = getattr(owner, 'hf_token', None) if owner else None

        with tempfile.TemporaryDirectory(prefix='bh_recache_') as staging:
            mat = materialize_hf_to_typed_dir(
                repo_id=repo_id, split=split, sample_cap=policy_cap,
                staging_dir=staging, dataset_name=ds.name, fields=fields,
                hf_token=hf_token, sampling='stratified',
                seed=int(md.get('sampling_seed') or 42),
                sample_name_from=None, config_name=config_name, shard_cap=-1,
            )

            # --- swap in place: delete old scoped rows + previews, then
            #     re-import into the same Dataset row. ---
            old_sample_ids = [s.id for s in Sample.query.filter_by(dataset_id=ds_id).with_entities(Sample.id).all()]
            if old_sample_ids:
                CustomField.query.filter(CustomField.sample_id.in_(old_sample_ids)).delete(synchronize_session=False)
            Sample.query.filter_by(dataset_id=ds_id).delete(synchronize_session=False)
            DatasetField.query.filter_by(dataset_id=ds_id).delete(synchronize_session=False)
            db.session.flush()
            preview_dir = os.path.join(A.app.config['UPLOAD_FOLDER'], 'datasets', str(ds_id))
            shutil.rmtree(preview_dir, ignore_errors=True)

            _, summary = import_typed_dataset(
                staging, db_session=db.session,
                Dataset=Dataset, Sample=Sample, CustomField=CustomField,
                DatasetField=DatasetField,
                upload_folder=A.app.config['UPLOAD_FOLDER'],
                existing_dataset=ds, preview_only=True,
                extra_kinds=_registered_extra_kinds(ds.owner_user_id),
            )
            ds.preview_only = True
            md.update({
                'sampling': 'stratified', 'sample_cap': policy_cap,
                'sampling_seed': int(md.get('sampling_seed') or 42),
                'total_rows_in_split': mat.get('total_rows_in_split'),
                'samples_imported': mat.get('samples'),
                'rows_written': mat.get('rows_written'),
                'recache_policy': f'stratified max(2000,10*{n_classes})={policy_cap}',
            })
            ds.source_metadata = json.dumps(md)
            db.session.commit()
            print(f'RECACHE_RESULT {ds_id} {ds.name} classes={n_classes} '
                  f'cap={policy_cap} samples={summary["samples"]}')


def main():
    ids = [int(x) for x in sys.argv[1:] if x.strip()]
    if not ids:
        print('usage: recache_classification.py <dataset_id> [...]'); return 2
    for ds_id in ids:
        try:
            recache_one(ds_id)
        except Exception as e:
            import traceback
            print(f'RECACHE_FAIL {ds_id} {type(e).__name__}: {e}')
            traceback.print_exc()
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
