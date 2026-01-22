from app import app
from tasks import process_submission
import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO)

with app.app_context():
    print("Triggering process_submission(1)...")
    try:
        process_submission(1)
        print("Process complete.")
    except Exception as e:
        print(f"Process failed: {e}")
