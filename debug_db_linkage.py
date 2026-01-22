import sys
import os
# Force unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Starting debug script...", flush=True)

try:
    from app import app, db, Submission, CustomField, Sample, Leaderboard
    print("Imports successful.", flush=True)

    with app.app_context():
        sub_id = 4
        print(f"--- Inspecting Submission {sub_id} ---", flush=True)
        
        sub = Submission.query.get(sub_id)
        if not sub:
            print("Submission not found", flush=True)
            sys.exit(1)
            
        print(f"Submission {sub.id} belongs to Leaderboard {sub.leaderboard_id}", flush=True)
        lb = Leaderboard.query.get(sub.leaderboard_id)
        dataset_id = lb.dataset_id
        print(f"Linked Dataset ID: {dataset_id}", flush=True)
        
        # Check Samples
        samples = Sample.query.filter_by(dataset_id=dataset_id).all()
        print(f"Found {len(samples)} samples in dataset.", flush=True)
        if samples:
            print(f"Sample IDs: {[s.id for s in samples]}", flush=True)
        
        # Check CustomFields (Metrics)
        cfs = CustomField.query.filter_by(submission_id=sub_id, field_type='metric').all()
        print(f"Found {len(cfs)} CustomMetric fields.", flush=True)
        
        if len(cfs) > 0:
            print(f"First 5 CFs:", flush=True)
            for cf in cfs[:5]:
                print(f"  CF {cf.id}: Name={cf.name}, SampleID={cf.sample_id}", flush=True)

        linked_count = sum(1 for cf in cfs if cf.sample_id is not None)
        print(f"Total Linked CFs: {linked_count} / {len(cfs)}", flush=True)

except Exception as e:
    print(f"CRITICAL ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Script finished.", flush=True)
