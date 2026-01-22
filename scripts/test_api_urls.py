import sys
import os

# Add project root to path
sys.path.append('/Users/ymatari/Git/dtof_benchmarking')

try:
    from app import app, db, Project, Dataset, Leaderboard
    from flask import url_for, g

    with app.test_request_context():
        g.project_name = 'ProjectA'
        
        print("Testing Project-Prefixed API URLs:")
        
        # 1. Leaderboard info by name
        try:
            url = url_for('get_leaderboard_info_by_name_api', project_name='ProjectA', leaderboard_name='LB1')
            print(f"Fetch LB Info: {url}")
        except Exception as e:
            print(f"Error Fetch LB Info: {e}")

        # 2. Dataset download
        try:
            url = url_for('api_download_dataset', project_name='ProjectA', dataset_id=5)
            print(f"Dataset Download: {url}")
        except Exception as e:
            print(f"Error Dataset Download: {e}")

        # 3. Submission upload
        try:
            url = url_for('submission_upload_api', project_name='ProjectA', leaderboard_id=10)
            print(f"Submission Upload: {url}")
        except Exception as e:
            print(f"Error Submission Upload: {e}")

except Exception as e:
    print(f"Verification script failed: {e}")
    import traceback
    traceback.print_exc()
