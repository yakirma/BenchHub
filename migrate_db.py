import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'database.db')
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}. Skipping migration.")
    exit(0)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Add selected_metrics to dataset
try:
    cursor.execute("ALTER TABLE dataset ADD COLUMN selected_metrics VARCHAR(500) DEFAULT ''")
    print("Added selected_metrics to dataset table.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e).lower():
        print("dataset.selected_metrics already exists.")
    else:
        print(f"Error adding column to dataset: {e}")

# 2. Add selected_metrics to leaderboard
try:
    cursor.execute("ALTER TABLE leaderboard ADD COLUMN selected_metrics VARCHAR(500) DEFAULT ''")
    print("Added selected_metrics to leaderboard table.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e).lower():
        print("leaderboard.selected_metrics already exists.")
    else:
        print(f"Error adding column to leaderboard: {e}")

# 3. Rename metrics to summary_metrics in leaderboard
try:
    cursor.execute("ALTER TABLE leaderboard RENAME COLUMN metrics TO summary_metrics")
    print("Renamed leaderboard.metrics to leaderboard.summary_metrics.")
except sqlite3.OperationalError as e:
    if "no such column" in str(e).lower():
        print("leaderboard.metrics does not exist (possibly already renamed).")
    elif "summary_metrics" in str(e).lower():
        print("leaderboard.summary_metrics already exists.")
    else:
        print(f"Error renaming column in leaderboard: {e}")

conn.commit()
conn.close()
print("Migration completed.")
