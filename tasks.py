import logging
import json
import os
from app import celery, db, Submission, LeaderboardMetric, MetricResult, Sample, app, CustomField
from metric_engine import evaluate_dynamic_metric, get_metric_context, sort_metrics_by_dependency
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

@celery.task(bind=True, max_retries=5, ignore_result=True)
def process_submission(self, submission_id, sample_filters=None):
    # Force remove any existing session to avoid issues in some Celery environments
    db.session.remove()
    session = db.session
    try:
        submission = session.query(Submission).get(submission_id)
        if not submission:
            logger.info(f"Submission {submission_id} not found in task. Retrying...")
            try:
                self.retry(countdown=1)
            except self.MaxRetriesExceededError:
                logger.error(f"Submission {submission_id} not found after retries.")
            return

        submission.processing_status = 'Processing'
        session.commit() # Commit the status change
        
        # --- Metric Calculation Logic ---
        leaderboard = submission.leaderboard
        if not leaderboard.leaderboard_metrics:
            logger.info(f"No metrics defined for leaderboard {leaderboard.id}. Skipping calculation.")
        else:
            # Fetch all samples for this dataset
            dataset_samples_query = session.query(Sample).filter_by(dataset_id=leaderboard.dataset_id)
            
            # Apply filters if provided
            if sample_filters:
                # 1. Search Filter (by name)
                if sample_filters.get('search'):
                    dataset_samples_query = dataset_samples_query.filter(Sample.name.ilike(f"%{sample_filters['search']}%"))
                
                # 2. Tag Filters
                def tag_match_filter(tag):
                    from sqlalchemy import or_
                    return or_(
                        Sample.tags == tag,
                        Sample.tags.ilike(f'{tag},%'),
                        Sample.tags.ilike(f'%,{tag}'),
                        Sample.tags.ilike(f'%,{tag},%')
                    )

                include = sample_filters.get('include', {})
                if include.get('enabled') and include.get('tags'):
                    for tag in include['tags']:
                        dataset_samples_query = dataset_samples_query.filter(tag_match_filter(tag))
                
                exclude = sample_filters.get('exclude', {})
                if exclude.get('enabled') and exclude.get('tags'):
                    from sqlalchemy import or_, not_
                    exclude_conditions = [tag_match_filter(tag) for tag in exclude['tags']]
                    dataset_samples_query = dataset_samples_query.filter(not_(or_(*exclude_conditions)))

                prefix = sample_filters.get('prefix', {})
                if prefix.get('enabled') and prefix.get('tags'):
                    from sqlalchemy import or_
                    for p in prefix['tags']:
                        dataset_samples_query = dataset_samples_query.filter(or_(
                            Sample.tags.ilike(f'{p}%'),
                            Sample.tags.ilike(f'%,{p}%'),
                            Sample.tags.ilike(f'%, {p}%')
                        ))

            dataset_samples = dataset_samples_query.all()
            logger.info(f"Filtered to {len(dataset_samples)} samples for metrics calculation.")

            # Store the filters used for this calculation
            submission.last_sample_filter = json.dumps(sample_filters, sort_keys=True) if sample_filters else None
            
            # Submission folder path
            submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id))
            
            samples_context = []
            logger.info(f"Building context for {len(dataset_samples)} samples from folder: {submission_folder}")
            for sample in dataset_samples:
                ctx = get_metric_context(sample, submission, submission_folder=submission_folder)
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

            for lm in sorted_metrics:
                global_metric = lm.global_metric
                arg_mappings_json = lm.arg_mappings
                metric_out_name = lm.target_name if lm.target_name else global_metric.name

                # Check for existing result
                existing_result = session.query(MetricResult).filter_by(
                    submission_id=submission.id, 
                    leaderboard_metric_id=lm.id
                ).first()
                if existing_result:
                    session.delete(existing_result)
                
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
                            if samples_context:
                                agg_context[key] = samples_context[0].get(key, None)
                            else:
                                agg_context[key] = None
                        else:
                            vals = []
                            for ctx in samples_context:
                                vals.append(ctx.get(key, None)) 
                            agg_context[key] = vals

                    value, error = evaluate_dynamic_metric(global_metric, agg_context, arg_mappings_json)
                    logger.info(f"  [Metric: {metric_out_name}] Aggregated calculation. Value: {value}")
                    if value is not None:
                        for ctx in samples_context:
                            ctx[metric_out_name] = value
                else:
                    # Per-sample: Average the results
                    sample_values = []
                    sample_errors = []
                    
                    # Cleanup existing CustomFields for this metric to avoid stale data
                    # Use delete with synchronize_session=False for performance/bulk
                    session.query(CustomField).filter_by(
                         submission_id=submission.id, 
                         name=metric_out_name
                    ).delete(synchronize_session=False)

                    for i, ctx in enumerate(samples_context):
                        val, err = evaluate_dynamic_metric(global_metric, ctx, arg_mappings_json)
                        if val is not None:
                            sample_values.append(val)
                            ctx[metric_out_name] = val
                            
                            # Persist to CustomField for Visualization
                            try:
                                current_sample = dataset_samples[i]
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
                                logger.error(f"Error persisting custom field {metric_out_name} for sample {i}: {e}")

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
            
        submission.processing_status = 'Processed'
        session.commit()
        logger.info(f"Processing submission {submission_id} done.")

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
