import sqlite3
import os
import sys

# Path to database
# Assuming standard path based on app.py config
user_home = os.path.expanduser("~")
dtof_data_dir = os.path.join(user_home, ".dtofbenchmarking")
db_path = os.path.join(dtof_data_dir, 'database.db')

print(f"Checking database at: {db_path}")

if not os.path.exists(db_path):
    print("Database file not found!")
    sys.exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check columns in leaderboard_metric dictionary
    cursor.execute("PRAGMA table_info(leaderboard_metric)")
    columns_info = cursor.fetchall()
    
    # Extract column names (index 1 is name)
    column_names = [col[1] for col in columns_info]
    
    if 'sort_direction' in column_names:
        print("Column 'sort_direction' already exists. No action needed.")
    else:
        print("Column 'sort_direction' missing. Adding it now...")
        try:
            # Add column with default value 'higher_is_better'
            cursor.execute("ALTER TABLE leaderboard_metric ADD COLUMN sort_direction VARCHAR(20) DEFAULT 'higher_is_better'")
            conn.commit()
            print("Successfully added 'sort_direction' column.")
        except Exception as e:
            print(f"Failed to add column: {e}")
            conn.rollback()

    conn.close()

except Exception as e:
    print(f"Database error: {e}")
    sys.exit(1)
