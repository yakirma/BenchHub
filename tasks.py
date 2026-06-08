import logging
import json
import os
import shutil
import tempfile
from app import (
    celery, db, Submission, LeaderboardMetric, MetricResult, Sample, app,
    CustomField, Dataset, DatasetField, Leaderboard, LeaderboardMaterialization,
)
from metric_engine import (
    evaluate_dynamic_metric,
    evaluate_in_sandbox,
    get_metric_context,
    sort_metrics_by_dependency,
    _sandbox_enabled,
)
import numpy as np


@celery.task(bind=True, name='tasks.run_hf_import',
             soft_time_limit=1800, time_limit=2100)
def run_hf_import(self, *, dataset_id, repo_id, split, sample_cap, sampling,
                  sampling_seed, dataset_name, fields, sample_name_from,
                  hf_token, owner_user_id, config_name=None, shard_cap=-1):
    """Background HF import. The Dataset row has already been created
    by the route with `import_status='importing'` so it appears on
    /datasets immediately; this task fills it in.

    Progress lands in two places: `Dataset.import_progress_json`
    (persisted, survives page reloads / app restarts) and Celery's
    AsyncResult meta (cheap polling via `state='PROGRESS'`).

    On success: status → 'ready', source_metadata is written.
    On failure: status → 'failed', import_error carries the message.
    The dataset stays on /datasets either way so the admin can see
    what went wrong.
    """
    from benchhub.manifest import import_typed_dataset
    from benchhub.hf_materialize import materialize_hf_to_typed_dir
    from app import _registered_extra_kinds

    with app.app_context():
        def _persist_progress(state):
            ds = Dataset.query.get(dataset_id)
            if ds is None:
                return
            ds.import_progress_json = json.dumps(state)
            db.session.commit()
            try:
                self.update_state(state='PROGRESS', meta=state)
            except Exception:
                pass

        # Marker so /dataset/<id> shows "Starting…" before
        # materialize even reaches the load_dataset step.
        _persist_progress({'phase': 'starting', 'current': 0, 'total': 0,
                           'message': 'Queued — starting import…'})

        with tempfile.TemporaryDirectory(prefix='bh_hf_import_') as staging:
            try:
                mat_summary = materialize_hf_to_typed_dir(
                    repo_id=repo_id,
                    split=split,
                    sample_cap=sample_cap,
                    staging_dir=staging,
                    dataset_name=dataset_name,
                    fields=fields,
                    hf_token=hf_token,
                    sampling=sampling,
                    seed=sampling_seed,
                    sample_name_from=sample_name_from,
                    config_name=config_name,
                    shard_cap=shard_cap,
                    progress_cb=_persist_progress,
                )
                # Post-materialize quota check against actual bytes.
                _persist_progress({'phase': 'quota-check', 'current': 0, 'total': 0,
                                   'message': 'Checking storage quota…'})
                staged_bytes = 0
                for dirpath, _, filenames in os.walk(staging):
                    for fn in filenames:
                        try:
                            staged_bytes += os.path.getsize(os.path.join(dirpath, fn))
                        except OSError:
                            continue
                from app import check_quota, User, _format_bytes
                owner = User.query.get(owner_user_id) if owner_user_id else None
                if owner is not None:
                    ok, msg = check_quota(owner, kind='dataset_create',
                                          incoming_bytes=staged_bytes)
                    if not ok:
                        raise RuntimeError(
                            f"{msg} (this import would write {_format_bytes(staged_bytes)})"
                        )

                _persist_progress({'phase': 'importing', 'current': 0, 'total': 0,
                                   'message': 'Writing rows to the database…'})
                existing = Dataset.query.get(dataset_id)
                # All HF imports default to preview-only now (Stage B):
                # vis modalities land as downscaled JPGs / waveform
                # PNGs; full bytes get materialised per-LB on demand
                # via tasks.materialize_leaderboard.
                _, summary = import_typed_dataset(
                    staging,
                    db_session=db.session,
                    Dataset=Dataset, Sample=Sample, CustomField=CustomField,
                    DatasetField=DatasetField,
                    upload_folder=app.config['UPLOAD_FOLDER'],
                    existing_dataset=existing,
                    preview_only=True,
                    extra_kinds=_registered_extra_kinds(
                        getattr(existing, 'owner_user_id', None)),
                )
                existing.preview_only = True
                existing.source_kind = 'hf'
                existing.source_url = f'https://huggingface.co/datasets/{repo_id}'
                existing.source_metadata = json.dumps({
                    'repo_id': repo_id,
                    'config_name': config_name,
                    'split': mat_summary.get('split'),
                    'sample_cap': sample_cap,
                    'shard_cap': shard_cap,
                    'shards_used': mat_summary.get('shards_used'),
                    'shards_total': mat_summary.get('shards_total'),
                    'sampling': mat_summary.get('sampling'),
                    'sampling_seed': mat_summary.get('seed'),
                    'total_rows_in_split': mat_summary.get('total_rows_in_split'),
                    'samples_imported': mat_summary.get('samples'),
                    'rows_written': mat_summary.get('rows_written'),
                    'rows_skipped': mat_summary.get('rows_skipped'),
                })
                # Auto-fill the Area/Task category from HF's
                # task_categories / task_ids tags so the dataset
                # lands in the right discovery bucket instead of
                # Uncategorized. Only overwrites a NULL category —
                # if the dataset row already carries one (manual
                # tweak, re-import), we leave it alone.
                if not existing.category:
                    try:
                        from benchhub.hf_search import fetch_dataset_card
                        from app import _hf_tags_to_category
                        card = fetch_dataset_card(repo_id)
                        if card:
                            tags = card.get('tags') or []
                            cat = _hf_tags_to_category(tags)
                            if cat:
                                existing.category = cat
                    except Exception:
                        # Category is a nice-to-have; a failure to
                        # fetch the card shouldn't poison the whole
                        # import. The owner can set it on the
                        # dataset settings page.
                        pass
                existing.import_status = 'ready'
                existing.import_error = None
                existing.import_progress_json = json.dumps({
                    'phase': 'done', 'current': summary['samples'],
                    'total': summary['samples'],
                    'message': f"Imported {summary['samples']} sample(s).",
                })
                db.session.commit()
                return {'dataset_id': dataset_id, **summary}
            except Exception as e:
                db.session.rollback()
                ds = Dataset.query.get(dataset_id)
                if ds is not None:
                    ds.import_status = 'failed'
                    ds.import_error = str(e)
                    ds.import_progress_json = json.dumps({
                        'phase': 'failed', 'current': 0, 'total': 0,
                        'message': str(e),
                    })
                    # Clean up any partial bytes on disk.
                    folder = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', str(dataset_id))
                    shutil.rmtree(folder, ignore_errors=True)
                    db.session.commit()
                # Don't re-raise: the failure is already persisted on
                # the Dataset row + import_error. Re-raising would
                # bubble through eager Celery into the calling route
                # (in tests / dev) and surface as a 500. Return an
                # error dict so AsyncResult.successful() is true but
                # the row carries the truth.
                return {'dataset_id': dataset_id, 'error': str(e)}


@celery.task(bind=True, name='tasks.run_file_tree_import',
             soft_time_limit=3600, time_limit=4200)
def run_file_tree_import(self, *, dataset_id, repo_id, spec, dataset_name,
                         sample_cap, hf_token, owner_user_id, token_filter=None,
                         path_prefix=None):
    """Background importer for the user-declared file-tree mapping. The
    Dataset row is pre-created ('importing') by the route. Fetches the
    declared source files via hf_hub_download, decodes them through
    benchhub.file_tree_import.materialize_file_tree into a typed-manifest
    staging dir, then imports preview-only — same downstream path as the
    Croissant importer."""
    from huggingface_hub import HfApi, hf_hub_download
    from benchhub.file_tree_import import materialize_file_tree
    from benchhub.manifest import import_typed_dataset
    from app import _registered_extra_kinds

    with app.app_context():
        def _persist(state):
            ds = Dataset.query.get(dataset_id)
            if ds is None:
                return
            ds.import_progress_json = json.dumps(state)
            db.session.commit()
            try:
                self.update_state(state='PROGRESS', meta=state)
            except Exception:
                pass

        _persist({'phase': 'starting', 'current': 0, 'total': 0,
                  'message': 'Listing repo files…'})
        try:
            api = HfApi()
            if path_prefix:
                # Scoped import (user picked a split/subfolder): list only
                # that subtree's files so a giant repo isn't walked whole.
                from huggingface_hub.hf_api import RepoFile
                files = [it.path for it in api.list_repo_tree(
                    repo_id, path_in_repo=path_prefix, recursive=True,
                    repo_type='dataset', token=hf_token or None)
                    if isinstance(it, RepoFile)]
            else:
                files = api.list_repo_files(repo_id, repo_type='dataset',
                                            token=hf_token or None)

            def _fetch(rel):
                return hf_hub_download(repo_id, rel, repo_type='dataset',
                                       token=hf_token or None)

            with tempfile.TemporaryDirectory(prefix='bh_ft_import_') as staging:
                mat = materialize_file_tree(
                    spec, files, _fetch, staging,
                    sample_cap=sample_cap, dataset_name=dataset_name,
                    token_filter=token_filter, progress_cb=_persist,
                )
                # Quota check against actual staged bytes.
                _persist({'phase': 'quota-check', 'current': 0, 'total': 0,
                          'message': 'Checking storage quota…'})
                staged_bytes = 0
                for dp, _, fns in os.walk(staging):
                    for fn in fns:
                        try:
                            staged_bytes += os.path.getsize(os.path.join(dp, fn))
                        except OSError:
                            continue
                from app import check_quota, User, _format_bytes
                owner = User.query.get(owner_user_id) if owner_user_id else None
                if owner is not None:
                    ok, msg = check_quota(owner, kind='dataset_create',
                                          incoming_bytes=staged_bytes,
                                          visibility='private')
                    if not ok:
                        raise RuntimeError(
                            f"{msg} (this import would write "
                            f"{_format_bytes(staged_bytes)})")

                _persist({'phase': 'importing', 'current': 0, 'total': 0,
                          'message': 'Writing rows to the database…'})
                existing = Dataset.query.get(dataset_id)
                _, summary = import_typed_dataset(
                    staging, db_session=db.session,
                    Dataset=Dataset, Sample=Sample, CustomField=CustomField,
                    DatasetField=DatasetField,
                    upload_folder=app.config['UPLOAD_FOLDER'],
                    existing_dataset=existing, preview_only=True,
                    extra_kinds=_registered_extra_kinds(
                        getattr(existing, 'owner_user_id', None)),
                )
                existing.preview_only = True
                existing.source_kind = 'hf'
                existing.source_url = f'https://huggingface.co/datasets/{repo_id}'
                existing.source_metadata = json.dumps({
                    'repo_id': repo_id, 'importer': 'file_tree',
                    'spec': spec, 'samples_imported': summary['samples'],
                    'total_rows_in_split': mat.get('total_rows_in_split'),
                })
                existing.import_status = 'ready'
                existing.import_error = None
                existing.import_progress_json = json.dumps({
                    'phase': 'done', 'current': summary['samples'],
                    'total': summary['samples'],
                    'message': f"Imported {summary['samples']} sample(s)."})
                db.session.commit()
                return {'dataset_id': dataset_id, **summary}
        except Exception as e:
            db.session.rollback()
            ds = Dataset.query.get(dataset_id)
            if ds is not None:
                ds.import_status = 'failed'
                ds.import_error = str(e)
                ds.import_progress_json = json.dumps({
                    'phase': 'failed', 'current': 0, 'total': 0,
                    'message': str(e)})
                folder = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets',
                                      str(dataset_id))
                shutil.rmtree(folder, ignore_errors=True)
                db.session.commit()
            return {'dataset_id': dataset_id, 'error': str(e)}


# Configure logging
logger = logging.getLogger(__name__)


def _eval_metric_batch(global_metric, contexts, arg_mappings_json):
    """Single dispatch point for evaluating a metric across many contexts.

    Returns a list of (value, error) tuples — one per context.

    When BENCHHUB_SANDBOX_METRICS=1, the work happens in a hardened docker
    container (one spawn for the whole batch — see metric_engine.evaluate_in_sandbox).
    Otherwise it falls back to the in-process exec path.

    The shape is identical for both backends so callers can treat them
    interchangeably."""
    if _sandbox_enabled():
        return evaluate_in_sandbox(global_metric, contexts, arg_mappings_json)
    return [
        evaluate_dynamic_metric(global_metric, ctx, arg_mappings_json)
        for ctx in contexts
    ]


def _process_submission_impl(submission_id, sample_filters=None, task_instance=None):
    # Force remove any existing session to avoid issues in some Celery environments
    db.session.remove()
    session = db.session
    try:
        submission = session.query(Submission).get(submission_id)
        if not submission:
            logger.info(f"Submission {submission_id} not found in task. Retrying...")
            if task_instance:
                try:
                    task_instance.retry(countdown=1)
                except task_instance.MaxRetriesExceededError:
                    logger.error(f"Submission {submission_id} not found after retries.")
            return

        # Phase 15: mirrored submissions opt out of the eval pipeline
        # entirely. Their MetricResult rows were inserted at import
        # time directly from the source benchmark (Papers With Code,
        # etc) — no ZIP, no extraction, no metric exec. Just leave
        # them in 'Mirrored' status.
        if getattr(submission, 'kind', 'verified') == 'mirrored':
            return

        # Strict hash-pin: for remote submissions, refuse to recalc
        # against drifted bytes. Cheap when the cache is warm; on a
        # cache miss this re-fetches from upstream and catches any
        # post-submission edit on the remote URL.
        from app import _verify_remote_submission_hash
        ok, msg = _verify_remote_submission_hash(submission)
        if not ok:
            logger.warning(
                f"Submission {submission_id} rejected: {msg}"
            )
            return

        submission.processing_status = 'Processing'
        session.commit() # Commit the status change
        
        # --- Metric Calculation Logic ---
        leaderboard = submission.leaderboard
        if not leaderboard.leaderboard_metrics:
            logger.info(f"No metrics defined for leaderboard {leaderboard.id}. Skipping calculation.")
        else:
            # 1. Preparing Context Status
            submission.processing_status = 'Processing: Preparing Context'
            session.commit()

            # Fetch all samples the LB evaluates against via the
            # Attachment table (+ the legacy lb.datasets m2m). HF
            # support is gone — datasets are BH-owned files on disk.
            from app import _iter_lb_eval_samples

            dataset_samples = [s for s, _att in _iter_lb_eval_samples(leaderboard)]

            # Filters: applied in Python so they work for any sample
            # source.
            if sample_filters and dataset_samples:
                def _tags_of(s):
                    raw = (s.tags or '') if isinstance(s.tags, str) else ''
                    return [t.strip() for t in raw.split(',') if t.strip()]

                search = sample_filters.get('search')
                if search:
                    needle = search.lower()
                    dataset_samples = [
                        s for s in dataset_samples
                        if needle in (s.name or '').lower()
                    ]

                include = sample_filters.get('include', {})
                if include.get('enabled') and include.get('tags'):
                    wanted = set(include['tags'])
                    dataset_samples = [
                        s for s in dataset_samples
                        if wanted.issubset(set(_tags_of(s)))
                    ]

                exclude = sample_filters.get('exclude', {})
                if exclude.get('enabled') and exclude.get('tags'):
                    banned = set(exclude['tags'])
                    dataset_samples = [
                        s for s in dataset_samples
                        if not (banned & set(_tags_of(s)))
                    ]

                prefix = sample_filters.get('prefix', {})
                if prefix.get('enabled') and prefix.get('tags'):
                    prefixes = tuple(prefix['tags'])
                    dataset_samples = [
                        s for s in dataset_samples
                        if any(t.startswith(prefixes) for t in _tags_of(s))
                    ]

            logger.info(f"Filtered to {len(dataset_samples)} samples for metrics calculation.")

            # Store the filters used for this calculation
            submission.last_sample_filter = json.dumps(sample_filters, sort_keys=True) if sample_filters else None

            # Submission folder. For local subs this is the always-on
            # `uploads/submissions/<id>/`. For remote subs it might be
            # a transient re-extraction from the cached ZIP — the
            # context manager handles either case + cleans up after.
            from app import (
                _pointer_gt_resolver, _make_paired_gt_provider,
                _with_extracted_submission,
            )
            paired_provider = _make_paired_gt_provider(submission.leaderboard)

            with _with_extracted_submission(submission) as submission_folder:
                samples_context = []
                logger.info(
                    f"Building context for {len(dataset_samples)} samples "
                    f"from folder: {submission_folder}"
                )
                for sample in dataset_samples:
                    ctx = get_metric_context(
                        sample, submission,
                        submission_folder=submission_folder,
                        pointer_resolver=_pointer_gt_resolver,
                        paired_gt_provider=paired_provider,
                    )
                    samples_context.append(ctx)
            
            if not samples_context:
                 logger.warning(f"No samples matched filters for submission {submission.id}. Metrics will be null.")

            # Sort metrics by dependency
            sorted_metrics = sort_metrics_by_dependency(leaderboard.leaderboard_metrics)
            metric_is_agg_map = {}
            for lm in leaderboard.leaderboard_metrics:
                 name = lm.target_name if lm.target_name else lm.global_metric.name
                 metric_is_agg_map[name] = lm.global_metric.is_aggregated

            # Which prediction fields did THIS submission actually provide?
            # (CustomFields that aren't metric outputs.) Used below to mark
            # a metric "Not computed" instead of scoring 0.0 when the
            # submission predates the metric / lacks its required pred field
            # — e.g. adding top_5_accuracy (needs label_topk_pred) to an LB
            # whose existing submission only has label_pred.
            sub_pred_field_names = {
                cf.name for cf in submission.custom_fields
                if cf.name and not cf.name.startswith('lm_')
            }
            try:
                _lb_pred_decls = json.loads(
                    leaderboard.required_pred_fields_json or '[]'
                )
                lb_pred_field_names = {
                    e.get('name') for e in _lb_pred_decls
                    if isinstance(e, dict) and e.get('name')
                }
            except (TypeError, ValueError):
                lb_pred_field_names = set()

            def _metric_missing_pred(arg_mappings_json):
                """Return the name of a required pred field the submission
                is missing for this metric, or None if it can be computed.
                A `sub_<field>` mapping (or a bare mapping naming a declared
                pred field) that has no CustomField on the submission means
                the metric isn't applicable to this submission."""
                try:
                    amap = json.loads(arg_mappings_json or '{}')
                except (TypeError, ValueError):
                    return None
                for v in amap.values():
                    if not isinstance(v, str) or v.startswith('SCALAR:') or v.startswith('gt_'):
                        continue
                    field = v[4:] if v.startswith('sub_') else v
                    is_pred_ref = v.startswith('sub_') or field in lb_pred_field_names
                    if is_pred_ref and field not in sub_pred_field_names:
                        return field
                return None
            
            logger.info(f"Evaluating {len(sorted_metrics)} leaderboard metrics...")
            
            total_metrics = len(sorted_metrics)
            for i, lm in enumerate(sorted_metrics):
                global_metric = lm.global_metric
                metric_name = lm.target_name if lm.target_name else global_metric.name
                
                # Update Status: Evaluating Metric X/Y
                submission.processing_status = f'Processing: Metric {i+1}/{total_metrics} ({metric_name})'
                session.commit()

                arg_mappings_json = lm.arg_mappings
                metric_out_name = f"lm_{lm.id}"

                # Check for existing result
                existing_result = session.query(MetricResult).filter_by(
                    submission_id=submission.id, 
                    leaderboard_metric_id=lm.id
                ).first()
                if existing_result:
                    session.delete(existing_result)

                # Skip metrics this submission can't satisfy: a required
                # prediction field is absent (e.g. the metric was added
                # after the submission, or the submission only produced a
                # different pred kind). Record NULL + a "Not computed"
                # message so the table shows that instead of a misleading
                # 0.0. Drop any stale per-sample CFs from a prior run.
                _missing_field = _metric_missing_pred(arg_mappings_json)
                if _missing_field is not None:
                    session.query(CustomField).filter_by(
                        submission_id=submission.id, name=metric_out_name,
                    ).delete(synchronize_session=False)
                    logger.info(
                        f"  [Metric: {metric_out_name}] Not computed — "
                        f"submission has no '{_missing_field}' prediction."
                    )
                    session.add(MetricResult(
                        submission_id=submission.id,
                        leaderboard_metric_id=lm.id,
                        value=None,
                        error_message=f"Not computed (no '{_missing_field}' prediction)",
                    ))
                    continue

                # Tag-based filtering for this specific metric
                current_metric_samples = dataset_samples
                current_metric_ctx = samples_context
                
                if hasattr(lm, 'tag_filter') and lm.tag_filter:
                    tags = [t.strip().lower() for t in lm.tag_filter.split(',')]
                    include_tags = [t for t in tags if not t.startswith('!')]
                    exclude_tags = [t[1:] for t in tags if t.startswith('!')]
                    
                    filtered_indices = []
                    for i_sample, sample in enumerate(dataset_samples):
                        sample_tags = [t.strip().lower() for t in (sample.tags.split(',') if sample.tags else [])]
                        
                        # Check excludes
                        if any(t in sample_tags for t in exclude_tags):
                            continue
                        
                        # Check includes
                        if not include_tags or any(t in sample_tags for t in include_tags):
                            filtered_indices.append(i_sample)
                    
                    current_metric_samples = [dataset_samples[idx] for idx in filtered_indices]
                    current_metric_ctx = [samples_context[idx] for idx in filtered_indices]
                    
                    logger.info(f"  [Metric: {metric_out_name}] Filtered to {len(current_metric_samples)}/{len(dataset_samples)} samples due to tag_filter: {lm.tag_filter}")

                value = None
                error = None
                
                if global_metric.is_aggregated:
                    agg_context = {}
                    try:
                         mappings = json.loads(arg_mappings_json)
                         required_keys = mappings.values()
                    except:
                         required_keys = []
                    
                    for key in required_keys:
                        is_agg_input = metric_is_agg_map.get(key, False)
                        if is_agg_input:
                            if current_metric_ctx:
                                agg_context[key] = current_metric_ctx[0].get(key, None)
                            else:
                                agg_context[key] = None
                        else:
                            vals = []
                            for ctx in current_metric_ctx:
                                vals.append(ctx.get(key, None)) 
                            agg_context[key] = vals

                    # Aggregated metrics: single context, single result.
                    agg_results = _eval_metric_batch(global_metric, [agg_context], arg_mappings_json)
                    value, error = agg_results[0] if agg_results else (None, "no result")
                    logger.info(f"  [Metric: {metric_out_name}] Aggregated calculation. Value: {value}")
                    if value is not None:
                        # Update ALL samples with the aggregated value for this metric
                        for ctx in samples_context:
                            ctx[metric_out_name] = value
                else:
                    # Per-sample: Average the results
                    sample_values = []
                    sample_errors = []
                    
                    # Cleanup existing CustomFields for this metric to avoid stale data
                    session.query(CustomField).filter_by(
                         submission_id=submission.id, 
                         name=metric_out_name
                    ).delete(synchronize_session=False)

                    # Per-sample metrics: batch evaluate once. With the
                    # in-process backend this is N exec() calls; with the
                    # sandbox backend it's a single docker spawn for the
                    # whole submission's worth of contexts. Same shape
                    # either way (a list of (value, error) tuples).
                    per_sample_results = _eval_metric_batch(
                        global_metric, current_metric_ctx, arg_mappings_json
                    )

                    for i_ctx, (val, err) in enumerate(per_sample_results):
                        ctx = current_metric_ctx[i_ctx]
                        if val is not None:
                            sample_values.append(val)
                            ctx[metric_out_name] = val

                            # Persist to CustomField for Visualization
                            try:
                                current_sample = current_metric_samples[i_ctx]
                                cf = CustomField(
                                    submission_id=submission.id,
                                    sample_id=current_sample.id,
                                    sample_name=current_sample.name,
                                    name=metric_out_name,
                                    data_type='scalar',
                                    value_float=float(val)
                                )
                                session.add(cf)
                            except Exception as e:
                                logger.error(f"Error persisting custom field {metric_out_name} for sample {i_ctx}: {e}")

                        if err:
                            sample_errors.append(err)
                    
                    if sample_values:
                        # Aggregation Logic
                        pooling_type = getattr(lm, 'pooling_type', 'mean')
                        pooling_percentile = getattr(lm, 'pooling_percentile', None)
                        
                        try:
                            if pooling_type == 'median':
                                value = float(np.median(sample_values))
                                agg_desc = "Median"
                            elif pooling_type == 'percentile' and pooling_percentile is not None:
                                value = float(np.percentile(sample_values, pooling_percentile))
                                agg_desc = f"{pooling_percentile}th Percentile"
                            else:
                                # Default to Mean
                                value = float(np.mean(sample_values))
                                agg_desc = "Mean"
                                
                            logger.info(f"  [Metric: {metric_out_name}] Per-sample {agg_desc} of {len(sample_values)} samples. Result: {value}")
                        except Exception as e:
                            logger.error(f"  [Metric: {metric_out_name}] Aggregation ({pooling_type}) failed: {e}")
                            value = None
                            error = f"Aggregation failed: {str(e)}"

                    elif sample_errors:
                        error = sample_errors[0]
                        logger.warning(f"  [Metric: {metric_out_name}] Per-sample calculation failed. Error: {error}")
                
                # Store Result
                result = MetricResult(
                    submission_id=submission.id,
                    leaderboard_metric_id=lm.id,
                    value=value,
                    error_message=error
                )
                session.add(result)
            
            session.commit()
        
        # Update Submission with filter state
        if sample_filters:
            submission.last_sample_filter = json.dumps(sample_filters)
        else:
            submission.last_sample_filter = None
            
        submission.processing_status = 'Generating Visualizations'
        session.commit()

        # Determine if we should optionally cache Visualizations
        if leaderboard.leaderboard_visualizations:
            from app import generate_and_cache_agg_viz
            for lv in leaderboard.leaderboard_visualizations:
                if lv.global_visualization.is_aggregated:
                     logger.info(f"Pre-caching aggregated visualization: {lv.id}")
                     try:
                         generate_and_cache_agg_viz(lv, submission)
                     except Exception as e:
                         logger.error(f"Error pre-caching visualization {lv.id}: {e}")

        # Per-sample viz assets: small colormap PNGs of dense
        # predictions (depth maps, seg masks, image outputs) so
        # the comparison page can render thumbnails cheaply without
        # decoding the raw prediction bytes per request. Bounded by
        # SUBMISSION_VIZ_MAX_SAMPLES so ImageNet-scale runs stay sane.
        # Run inside _with_extracted_submission so remote submissions
        # see a populated folder even though their primary extraction
        # has already been torn down.
        try:
            from app import (
                _generate_submission_viz_assets,
                _with_extracted_submission,
            )
            with _with_extracted_submission(submission) as sub_folder:
                n_viz = _generate_submission_viz_assets(
                    submission, leaderboard, sub_folder,
                )
                if n_viz:
                    logger.info(
                        f"Wrote {n_viz} viz PNG(s) for submission {submission_id}"
                    )
                # Typed-ingest flow (Phase B) writes pred CustomField
                # rows up-front in `import_typed_submission`, so the
                # legacy `_persist_pred_scalars_from_disk` step is
                # gone — it used to delete the pred rows and re-read
                # them from `.txt` files only, which mangled non-scalar
                # kinds (depth/image/audio/etc.).
        except Exception as e:
            logger.warning(
                f"Submission {submission_id} viz-asset generation failed: {e}"
            )

        submission.processing_status = 'Processed'
        session.commit()
        logger.info(f"Processing submission {submission_id} done.")

        # Disk-savings closeout: for remote submissions, tear down
        # the extracted `uploads/submissions/<id>/` folder now that
        # CustomFields are persisted. The cached ZIP under bench_cache
        # is the canonical source; subsequent recalcs re-extract on
        # demand via `_with_extracted_submission`.
        # NOTE: viz/ subdir under the submission folder (written
        # above) survives this evict because it lives at the top of
        # the submission folder; for *remote* subs, viz writes have
        # to land somewhere persistent — see the dedicated viz dir
        # logic in _generate_submission_viz_assets's caller below if
        # we ever break that invariant.
        try:
            from app import _evict_extracted_submission_folder
            _evict_extracted_submission_folder(submission)
        except Exception as e:
            logger.warning(
                f"Submission {submission_id} extracted-folder evict failed: {e}"
            )

    except Exception as e:
        session.rollback()
        try:
           submission = session.query(Submission).get(submission_id)
           if submission:
               submission.processing_status = f'Error: {e}'
               session.commit()
        except Exception as inner_e:
            logger.critical(f"Critical error updating submission status: {inner_e}")
        logger.exception(f"Error processing submission {submission_id}: {e}")
    finally:
        session.remove()

@celery.task(bind=True, max_retries=5, ignore_result=True)
def process_submission(self, submission_id, sample_filters=None):
    """
    Standard background task to calculate a single submission.
    """
    _process_submission_impl(submission_id, sample_filters, self)

@celery.task(bind=True, max_retries=5, ignore_result=True)
def process_submissions_batch_sequential(self, submission_ids, sample_filters=None):
    """
    Background task to calculate a list of submissions sequentially.
    """
    logger.info(f"Starting sequential batch calculation for {len(submission_ids)} submissions: {submission_ids}")
    for idx, sub_id in enumerate(submission_ids):
        logger.info(f"Batch processing {idx+1}/{len(submission_ids)}: Submission {sub_id}")
        _process_submission_impl(sub_id, sample_filters, task_instance=None)
    logger.info("Sequential batch calculation complete.")

@celery.task(bind=True, max_retries=3, ignore_result=True)
def reaggregate_submission_metrics(self, submission_id):
    """
    Re-calculates aggregated metric results (mean/max/etc) from existing per-sample 
    CustomField values, without re-running the python metric code.
    Used when only aggregation settings change.
    """
    # Force remove any existing session
    db.session.remove()
    session = db.session
    try:
        submission = session.query(Submission).get(submission_id)
        if not submission:
            return

        # safety check: if submission is 'dirty' (was calculated on subset), 
        # we can't produce a valid full-dataset aggregation from it.
        if submission.last_sample_filter:
             logger.info(f"Submission {submission_id} has filtered results. Falling back to full calculation.")
             process_submission(submission_id)
             return
             
        submission.processing_status = 'Processing'
        session.commit()

        leaderboard = submission.leaderboard
        logger.info(f"Re-aggregating metrics for submission {submission_id} (Optimized)...")

        updated_count = 0
        
        # Pre-calc total
        total_metrics = 0
        for lm in leaderboard.leaderboard_metrics:
            if not lm.global_metric.is_aggregated:
                total_metrics += 1
                
        processed_i = 0
        for lm in leaderboard.leaderboard_metrics:
            # Skip global aggregated metrics (they don't use pooling)
            if lm.global_metric.is_aggregated:
                continue
            
            processed_i += 1
            metric_name = lm.target_name if lm.target_name else lm.global_metric.name
            submission.processing_status = f'Re-aggregating: {processed_i}/{total_metrics} ({metric_name})'
            session.commit()

            metric_out_name = f"lm_{lm.id}"
            
            # 1. Fetch all per-sample scalar values from CustomField
            # We filter by name AND type='scalar' to be sure
            # Note: process_submission ensures these are created
            cfs = session.query(CustomField.value_float).filter_by(
                submission_id=submission.id,
                name=metric_out_name,
                data_type='scalar'
            ).all()
            
            sample_values = [r[0] for r in cfs if r[0] is not None]

            # If no values found, it might mean the metric failed previously or wasn't calculated.
            # In this case, we can't aggregate. 
            if not sample_values:
                continue
                
            # 2. Perform Aggregation
            pooling_type = getattr(lm, 'pooling_type', 'mean')
            pooling_percentile = getattr(lm, 'pooling_percentile', None)
            
            value = None
            error = None
            agg_desc = pooling_type
            
            try:
                if pooling_type == 'median':
                    value = float(np.median(sample_values))
                    agg_desc = "Median"
                elif pooling_type == 'percentile' and pooling_percentile is not None:
                    value = float(np.percentile(sample_values, pooling_percentile))
                    agg_desc = f"{pooling_percentile}th Percentile"
                elif pooling_type == 'min': 
                     value = float(np.min(sample_values))
                     agg_desc = "Min"
                elif pooling_type == 'max':
                     value = float(np.max(sample_values))
                     agg_desc = "Max"
                else:
                    # Default to Mean
                    value = float(np.mean(sample_values))
                    agg_desc = "Mean"
                    
                logger.info(f"  [Metric: {metric_out_name}] Re-aggregated {agg_desc} of {len(sample_values)} samples. Result: {value}")
                
            except Exception as e:
                logger.error(f"  [Metric: {metric_out_name}] Re-aggregation ({pooling_type}) failed: {e}")
                error = f"Aggregation failed: {str(e)}"

            # 3. Update MetricResult
            result = session.query(MetricResult).filter_by(
                submission_id=submission.id,
                leaderboard_metric_id=lm.id
            ).first()
            
            if not result:
                result = MetricResult(
                    submission_id=submission.id,
                    leaderboard_metric_id=lm.id
                )
                session.add(result)
            
            result.value = value
            result.error_message = error
            updated_count += 1
            
        submission.processing_status = 'Processed'
        session.commit()
        logger.info(f"Re-aggregation complete for {submission_id}. Updated {updated_count} metrics.")
        
    except Exception as e:
        session.rollback()
        logger.exception(f"Error re-aggregating submission {submission_id}: {e}")
        # Mark as error so user knows something went wrong
        try:
             submission = session.query(Submission).get(submission_id)
             if submission:
                 submission.processing_status = "Error Recalc"
                 session.commit()
        except: pass
    finally:
        session.remove()




@celery.task(bind=True, name='tasks.materialize_leaderboard',
              max_retries=2, ignore_result=True)
def materialize_leaderboard(self, leaderboard_id: int):
    """Stage C: full-resolution re-fetch from HF for the LB's chosen
    sample subset. Reads the LeaderboardMaterialization row already
    persisted by the create_leaderboard route, runs the materialiser,
    updates status. Failure puts the row in status='failed' with the
    error message; the LB page surfaces a retry button.
    """
    from benchhub.lb_materialize import materialize_for_lb

    logger = logging.getLogger(__name__)
    with app.app_context():
        lb = Leaderboard.query.get(leaderboard_id)
        if lb is None:
            logger.warning(f'materialize: LB {leaderboard_id} not found')
            return
        matrow = LeaderboardMaterialization.query.filter_by(
            leaderboard_id=leaderboard_id).first()
        if matrow is None:
            logger.warning(f'materialize: LB {leaderboard_id} has no '
                           'LeaderboardMaterialization row')
            return
        ds = (lb.datasets or [None])[0]
        if ds is None:
            matrow.status = 'failed'
            matrow.error_message = 'no backing dataset'
            db.session.commit()
            return
        def _publish(state):
            # Push to Celery's task meta so any AsyncResult poller
            # also sees the progress. The DB write is done inside
            # materialize_for_lb's progress callback already.
            try:
                self.update_state(state='PROGRESS', meta=state)
            except Exception:
                pass
        try:
            summary = materialize_for_lb(
                leaderboard=lb, dataset=ds, db_session=db.session,
                upload_folder=app.config['UPLOAD_FOLDER'],
                CustomField=CustomField,
                LeaderboardMaterialization=LeaderboardMaterialization,
                progress_cb=_publish,
            )
            logger.info(f'materialize: LB {leaderboard_id} → {summary}')
        except Exception as e:
            logger.exception(f'materialize LB {leaderboard_id} failed: {e}')
            matrow.status = 'failed'
            matrow.error_message = str(e)[:500]
            db.session.commit()
