import sqlite3
import os

DB_PATH = os.path.expanduser('~/.dtofbenchmarking/database.db')

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Check if column exists
        cursor.execute("PRAGMA table_info(leaderboard)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'project_id' in columns:
            print("column 'project_id' already exists in 'leaderboard'. Skipping add.")
        else:
            print("Adding 'project_id' column to 'leaderboard'...")
            cursor.execute("ALTER TABLE leaderboard ADD COLUMN project_id INTEGER REFERENCES project(id)")
            
        # 2. Backfill Data
        print("Backfilling project_id from dataset relation...")
        # Update leaderboard.project_id = dataset.project_id where leaderboard.dataset_id = dataset.id
        # SQLite supports this via subquery in UPDATE
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
        print(f"Updated {rows} leaderboards.")
        
        conn.commit()
        print("Migration successful.")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
