import sqlite3
import os

DB_PATH = '/Users/ymatari/.dtofbenchmarking/database.db'

def remove_hist_entropy():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all leaderboards
    cursor.execute("SELECT id, selected_metrics, summary_metrics FROM leaderboard")
    leaderboards = cursor.fetchall()
    
    print(f"Checking {len(leaderboards)} leaderboards...")

    for lb_id, selected, summary in leaderboards:
        new_selected = None
        new_summary = None
        updated = False

        if selected and 'hist_entropy' in selected:
            parts = [p.strip() for p in selected.split(',') if p.strip() != 'hist_entropy']
            new_selected = ','.join(parts)
            print(f"  LB {lb_id}: Removing hist_entropy from selected_metrics.")
            updated = True
        
        if summary and 'hist_entropy' in summary:
            parts = [p.strip() for p in summary.split(',') if p.strip() != 'hist_entropy']
            new_summary = ','.join(parts)
            print(f"  LB {lb_id}: Removing hist_entropy from summary_metrics.")
            updated = True
        
        if updated:
            # Update DB
            # Handle None/empty cases if needed, though join returns empty string
            s_val = new_selected if new_selected is not None else selected
            sum_val = new_summary if new_summary is not None else summary
            
            cursor.execute("UPDATE leaderboard SET selected_metrics = ?, summary_metrics = ? WHERE id = ?", (s_val, sum_val, lb_id))

    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    remove_hist_entropy()
