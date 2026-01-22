import sys
import os
sys.path.append(os.getcwd())
from app import app, db, CustomField, Submission, Sample

with app.app_context():
    # Find any custom field with name 'pred_pick'
    cfs = CustomField.query.filter_by(name='pred_pick').limit(5).all()
    print(f"Found {len(cfs)} entries for 'pred_pick'.")
    for cf in cfs:
        print(f"  ID: {cf.id}, SubID: {cf.submission_id}, SampleID: {cf.sample_id}, SampleName: {cf.sample_name}, Value: {cf.get_value()}")
