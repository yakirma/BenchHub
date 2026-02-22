import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path so we can import app
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    from app import app, db, Leaderboard, Dataset, leaderboard_datasets
except ImportError as e:
    print(f"Error: Could not import app modules. Make sure you are running this from the BenchHub root or scripts directory. {e}")
    sys.exit(1)

def migrate():
    print(f"--- BenchHub Migration: Multiple Datasets per Leaderboard ---")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with app.app_context():
        # 1. Create the association table if it doesn't exist
        try:
            db.create_all()
            print("[1/2] Ensured all tables (including leaderboard_datasets) exist.")
        except Exception as e:
            print(f"Error during table creation: {e}")
            sys.exit(1)
        
        # 2. Migrate existing data from Leaderboard.dataset_id to association table
        leaderboards = Leaderboard.query.all()
        migrated_count = 0
        already_migrated = 0
        
        for lb in leaderboards:
            # Check if this leaderboard already has entries in the association table
            # We use a raw SQL check or check the relationship
            existing_count = db.session.query(leaderboard_datasets).filter_by(leaderboard_id=lb.id).count()
            
            if existing_count > 0:
                already_migrated += 1
                continue
                
            if lb.dataset_id:
                # Insert the legacy dataset_id into the new many-to-many relationship
                # We can do this by adding to the relationship list
                ds = Dataset.query.get(lb.dataset_id)
                if ds:
                    lb.datasets.append(ds)
                    migrated_count += 1
                    print(f"  + Migrated Leaderboard '{lb.name}' (ID: {lb.id}) -> Dataset: {ds.name}")
                else:
                    print(f"  ! Warning: Leaderboard '{lb.name}' has invalid dataset_id: {lb.dataset_id}")
            else:
                print(f"  . Leaderboard '{lb.name}' (ID: {lb.id}) has no dataset_id, skipping.")
        
        try:
            db.session.commit()
            print(f"[2/2] Data migration completed.")
            print(f"Summary:")
            print(f"  - Newly migrated: {migrated_count}")
            print(f"  - Already migrated: {already_migrated}")
            print(f"  - Total leaderboards: {len(leaderboards)}")
        except Exception as e:
            db.session.rollback()
            print(f"Error during data migration: {e}")
            sys.exit(1)

    print("--- Migration Finished Successfully ---")

if __name__ == "__main__":
    migrate()
