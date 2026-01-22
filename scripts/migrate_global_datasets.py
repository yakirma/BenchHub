import sqlite3
import os
import sys

# Default path, works for standard installation
DB_PATH = os.path.expanduser('~/.dtofbenchmarking/database.db')

def migrate(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return False

    print(f"Migrating database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. Check if 'project_id' column exists in 'leaderboard'
        cursor.execute("PRAGMA table_info(leaderboard)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'project_id' in columns:
            print(" - Column 'project_id' already exists in 'leaderboard'.")
        else:
            print(" - Adding 'project_id' column to 'leaderboard' table...")
            cursor.execute("ALTER TABLE leaderboard ADD COLUMN project_id INTEGER REFERENCES project(id)")
            print("   Done.")
            
        # 2. Backfill Data: Copy project_id from dataset to leaderboard
        # In the old schema, Datasets belonged to Projects.
        # We are moving this association to Leaderboards.
        print(" - Backfilling 'project_id' in leadboards from their datasets...")
        
        cursor.execute("""
            UPDATE leaderboard 
            SET project_id = (
                SELECT project_id 
                FROM dataset 
                WHERE dataset.id = leaderboard.dataset_id
            )
            WHERE project_id IS NULL
        """)
        
        rows = cursor.rowcount
        print(f"   Updated {rows} leaderboards with project_id.")
        
        # 3. Optional: Verify
        cursor.execute("SELECT count(*) FROM leaderboard WHERE project_id IS NULL")
        orphans = cursor.fetchone()[0]
        if orphans > 0:
            print(f"   Warning: {orphans} leaderboards still have NULL project_id (maybe their dataset had no project?).")
        
        conn.commit()
        print("Migration completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Allow optional command line argument for db path
    path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    migrate(path)
