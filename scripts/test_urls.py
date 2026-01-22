import sys
import os

# Add project root to path
sys.path.append('/Users/ymatari/Git/dtof_benchmarking')

try:
    from app import app, db, Project, Dataset, Leaderboard
    from flask import url_for, g

    with app.test_request_context():
        # 1. Test url_for without project context
        try:
            url_idx = url_for('index', project_name='ProjectA')
            print(f"Index URL (explicit): {url_idx}")
        except Exception as e:
            print(f"Error Index URL: {e}")

        # 2. Test url_for with project context in g
        g.project_name = 'ProjectA'
        try:
            url_lb = url_for('leaderboard_view', leaderboard_id=1)
            print(f"Leaderboard URL (auto): {url_lb}")
        except Exception as e:
            print(f"Error Leaderboard URL: {e}")

        # 3. Test list_projects (should NOT have prefix)
        try:
            url_list = url_for('list_projects')
            print(f"List Projects URL: {url_list}")
        except Exception as e:
            print(f"Error List Projects: {e}")

        # 4. Test root redirect
        try:
            url_root = url_for('root_redirect')
            print(f"Root Redirect URL: {url_root}")
        except Exception as e:
            print(f"Error Root Redirect: {e}")

except Exception as e:
    print(f"Verification script failed: {e}")
    import traceback
    traceback.print_exc()
