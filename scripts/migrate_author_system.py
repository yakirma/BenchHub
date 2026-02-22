import sqlite3
import os
from datetime import datetime

def migrate():
    user_home = os.path.expanduser("~")
    db_path = os.path.join(user_home, ".dtofbenchmarking", "database.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("--- Starting Author System Migration ---")

    # 1. Ensure AuthorProfile table exists (manually to be safe)
    print("Checking 'author_profile' table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS author_profile (
            id INTEGER PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            display_name VARCHAR(100),
            avatar_filename VARCHAR(255),
            created_at DATETIME
        )
    ''')
    conn.commit()

    # 2. Add merged_into_username column if missing
    print("Checking for 'merged_into_username' column...")
    cursor.execute("PRAGMA table_info(author_profile)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'merged_into_username' not in columns:
        print("Adding 'merged_into_username' column...")
        cursor.execute("ALTER TABLE author_profile ADD COLUMN merged_into_username VARCHAR(100) DEFAULT NULL")
        conn.commit()
    else:
        print("'merged_into_username' column already exists.")

    # 3. Clean up legacy authors (Rename NULL/Empty to 'Yakir')
    print("Cleaning up legacy anonymous authors...")
    
    # Update Submissions
    cursor.execute("UPDATE submission SET git_author = 'Yakir' WHERE git_author IS NULL OR git_author = '' OR git_author = 'N/A'")
    subs_updated = cursor.rowcount
    
    # Update Datasets
    cursor.execute("UPDATE dataset SET git_author = 'Yakir' WHERE git_author IS NULL OR git_author = '' OR git_author = 'N/A'")
    ds_updated = cursor.rowcount
    
    conn.commit()
    
    print(f"--- Migration Complete ---")
    print(f"Cleaned up {subs_updated} submissions.")
    print(f"Cleaned up {ds_updated} datasets.")
    
    conn.close()

if __name__ == '__main__':
    migrate()
