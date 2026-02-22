import sqlite3
import os

def migrate_schema():
    user_home = os.path.expanduser("~")
    db_path = os.path.join(user_home, ".dtofbenchmarking", "database.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Adding merged_into_username column to author_profile table...")
    try:
        cursor.execute("ALTER TABLE author_profile ADD COLUMN merged_into_username TEXT")
        conn.commit()
        print("Column added successfully.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Column already exists.")
        else:
            print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_schema()
