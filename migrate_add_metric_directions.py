import sqlite3
import os

user_home = os.path.expanduser("~")
dtof_data_dir = os.path.join(user_home, ".dtofbenchmarking")
db_path = os.path.join(dtof_data_dir, 'database.db')

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}. Skipping migration.")
    exit(0)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add metric_directions to leaderboard
try:
    cursor.execute("ALTER TABLE leaderboard ADD COLUMN metric_directions TEXT DEFAULT '{}'")
    print("Added metric_directions to leaderboard table.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e).lower():
        print("leaderboard.metric_directions already exists.")
    else:
        print(f"Error adding column to leaderboard: {e}")

conn.commit()
conn.close()
print("Migration completed.")
