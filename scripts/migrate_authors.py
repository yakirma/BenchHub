import sqlite3
import os

def migrate():
    user_home = os.path.expanduser("~")
    db_path = os.path.join(user_home, ".dtofbenchmarking", "database.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Updating Submissions...")
    cursor.execute("UPDATE submission SET git_author = 'Yakir' WHERE git_author IS NULL OR git_author = '' OR git_author = 'N/A'")
    count1 = cursor.rowcount
    
    print("Updating Datasets...")
    cursor.execute("UPDATE dataset SET git_author = 'Yakir' WHERE git_author IS NULL OR git_author = '' OR git_author = 'N/A'")
    count2 = cursor.rowcount

    conn.commit()
    conn.close()
    print(f"Migration complete. Updated {count1} submissions and {count2} datasets.")

if __name__ == '__main__':
    migrate()
