import logging
import json
import os
from app import celery, db, Submission, LeaderboardMetric, MetricResult, Sample, app, CustomField
from metric_engine import (
    evaluate_dynamic_metric,
    evaluate_in_sandbox,
    get_metric_context,
    sort_metrics_by_dependency,
    _sandbox_enabled,
)
# Top-level so the celery main process loads it once at startup; the
# prefork child inherits the cached module via fork. Lazy import inside
# the task body fails in the child with ModuleNotFoundError because
# the prefork worker's sys.path/cwd don't always match the parent's.
import pwc_client
import numpy as np

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

            # Fetch all samples the LB evaluates against. Use
            # _iter_lb_eval_samples so HF-attached LBs (no BH Dataset
            # row, just an Attachment(kind='hf')) get streamed
            # _VirtualSample objects instead of an empty SQL result.
            # Previously this filtered on Sample.dataset_id, which is
            # NULL for HF-attached LBs → zero samples → silent
            # status='Processed' with no MetricResult values.
            from app import _iter_lb_eval_samples
            hf_token = None
            try:
                hf_token = getattr(submission.owner, 'hf_token', None)
            except Exception:
                pass
            dataset_samples = [
                s for s, _att in _iter_lb_eval_samples(
                    leaderboard, hf_token=hf_token,
                )
            ]

            # Filters: applied in Python now (was SQL) so it works for
            # both BH Sample rows and HF _VirtualSamples uniformly.
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
                                    field_type='scalar',
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

@celery.task(bind=True, max_retries=0, ignore_result=True,
             time_limit=4200, soft_time_limit=3600)
def build_pwc_index(self):
    """Build the Papers With Code static-archive index. Out-of-band so
    the web request doesn't time out / OOM. Reports progress to a file
    the web tier reads via pwc_client.index_progress_message().

    Caller (the route handler) is expected to have already called
    pwc_client.begin_build_marker() — that's the de-duplication point
    for "user clicked Build twice in a row." This task just executes;
    it doesn't decide whether one was already running."""
    if os.path.exists(pwc_client._index_path()):
        logger.info("PWC index already exists; skipping build.")
        pwc_client.clear_build_marker()
        return
    try:
        pwc_client.update_progress("Downloading PWC archive shards from HuggingFace…")
        snap = pwc_client._ensure_snapshot()
        logger.info(f"PWC: snapshot at {snap}; building index…")
        pwc_client.update_progress("Snapshot ready — opening parquet shards…")
        pwc_client._build_index_into(
            pwc_client._index_path(), snap,
            progress_cb=pwc_client.update_progress,
        )
        logger.info("PWC: index built.")
        pwc_client.update_progress("Done.")
    except Exception as e:
        logger.exception(f"PWC index build failed: {e}")
        try:
            with open(pwc_client._build_error_path(), 'w') as f:
                f.write(str(e)[:2000])
        except OSError:
            pass
    finally:
        pwc_client.clear_build_marker()


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
                field_type='scalar'
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
