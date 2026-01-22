
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db, Leaderboard

def cleanup_zombie(project_name, metric_name):
    with app.app_context():
        # Find leaderboard by name
        # The user said "dataset_6_LB". 
        # Note: Leaderboard names might be just "dataset_6_LB" or something else.
        # Let's search for it.
        lbs = Leaderboard.query.filter(Leaderboard.name.ilike(f'%{project_name}%')).all()
        
        if not lbs:
            print(f"No leaderboard found matching '{project_name}'")
            # List all to help debug
            all_lbs = Leaderboard.query.all()
            print(f"Available leaderboards: {[lb.name for lb in all_lbs]}")
            return

        for lb in lbs:
            print(f"Checking Leaderboard: {lb.name} (ID: {lb.id})")
            print(f"  Raw selected_metrics: '{lb.selected_metrics}'")
            print(f"  Raw summary_metrics: '{lb.summary_metrics}'")
            
            # List actual existng metrics
            print("  Active LeaderboardMetrics:")
            for lm in lb.leaderboard_metrics:
                name = lm.target_name if lm.target_name else lm.global_metric.name
                print(f"    - {name} (ID: {lm.id})")

            changed = False
            
            # Check selected_metrics
            if lb.selected_metrics:
                current_metrics = [m.strip() for m in lb.selected_metrics.split(',') if m.strip()]
                if metric_name in current_metrics:
                    print(f"  - Found '{metric_name}' in selected_metrics. Removing...")
                    current_metrics.remove(metric_name)
                    lb.selected_metrics = ','.join(current_metrics)
                    changed = True
            
            # Check summary_metrics (just in case)
            if lb.summary_metrics:
                 # It's likely comma separated too
                current_summary = [m.strip() for m in lb.summary_metrics.split(',') if m.strip()]
                if metric_name in current_summary:
                    print(f"  - Found '{metric_name}' in summary_metrics. Removing...")
                    current_summary.remove(metric_name)
                    lb.summary_metrics = ','.join(current_summary)
                    changed = True

            # 2. Clean metric_directions
            if lb.metric_directions:
                try:
                    directions = json.loads(lb.metric_directions)
                    print(f"  Raw metric_directions keys: {list(directions.keys())}")
                    if metric_name in directions:
                        print(f"  - Found '{metric_name}' in metric_directions. Removing...")
                        del directions[metric_name]
                        lb.metric_directions = json.dumps(directions)
                        changed = True
                except Exception as e:
                    print(f"  - Error parsing metric_directions: {e}")

            if changed:
                print("  => Saving changes...")
                db.session.commit()
                print("  => Done.")
            else:
                print("  => No changes needed.")

if __name__ == "__main__":
    # Hardcoded for the user request
    cleanup_zombie("dataset_6_LB", "random")
