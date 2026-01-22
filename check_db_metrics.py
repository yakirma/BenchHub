import sqlite3
import os

DB_PATH = '/Users/ymatari/.dtofbenchmarking/database.db'

def check_metrics():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Checking GlobalMetrics...")
    cursor.execute("SELECT id, name FROM global_metric WHERE name LIKE '%hist_entropy%'")
    gms = cursor.fetchall()
    for gm in gms:
        print(f"  Found GM: {gm}")

    print("Checking LeaderboardMetrics...")
    cursor.execute("SELECT lm.id, lm.leaderboard_id, gm.name, lm.target_name FROM leaderboard_metric lm JOIN global_metric gm ON lm.global_metric_id = gm.id WHERE gm.name LIKE '%hist_entropy%' OR lm.target_name LIKE '%hist_entropy%'")
    lms = cursor.fetchall()
    for lm in lms:
        print(f"  Found LM: LB={lm[1]} Metric={lm[2]} Target={lm[3]}")

    conn.close()

if __name__ == "__main__":
    check_metrics()
