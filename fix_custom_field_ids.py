import sys
from app import app, db, Submission, CustomField, Sample, Leaderboard

def fix_custom_fields():
    with app.app_context():
        print("Starting backfill of CustomField.sample_id...")
        
        # Find all CFs with missing sample_id
        # We can do this efficiently by processing submission by submission
        submissions = Submission.query.all()
        
        total_fixed = 0
        
        for sub in submissions:
            print(f"Processing Submission {sub.id}: {sub.name}")
            
            # Get dataset samples map
            lb = Leaderboard.query.get(sub.leaderboard_id)
            if not lb:
                print(f"  [WARN] Leaderboard {sub.leaderboard_id} not found. Skipping.")
                continue
                
            dataset_samples = Sample.query.filter_by(dataset_id=lb.dataset_id).all()
            sample_map = {s.name: s.id for s in dataset_samples}
            
            # Get bad custom fields
            cfs = CustomField.query.filter_by(submission_id=sub.id).filter(CustomField.sample_id == None).all()
            
            if not cfs:
                print("  No missing sample_ids.")
                continue
                
            count = 0
            for cf in cfs:
                if cf.sample_name in sample_map:
                    cf.sample_id = sample_map[cf.sample_name]
                    count += 1
                else:
                    print(f"  [WARN] Sample '{cf.sample_name}' not found in dataset {lb.dataset_id} for CF {cf.id}")
            
            if count > 0:
                db.session.commit()
                print(f"  Fixed {count} fields.")
                total_fixed += count
            else:
                print("  No fields matched.")
                
        print(f"DONE. Total fixed: {total_fixed}")

if __name__ == "__main__":
    fix_custom_fields()
