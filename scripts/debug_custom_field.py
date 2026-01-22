from app import app, db, CustomField, Submission, Sample

with app.app_context():
    # Find any custom field with name 'pred_pick'
    cfs = CustomField.query.filter_by(name='pred_pick').limit(5).all()
    print(f"Found {len(cfs)} entries for 'pred_pick'.")
    for cf in cfs:
        print(f"  ID: {cf.id}, SubID: {cf.submission_id}, SampleID: {cf.sample_id}, Type: {cf.field_type}, Value: {cf.get_value()}")
        
    # Check specifically for submission 1 (from previous logs)
    sub = Submission.query.get(1)
    if sub:
        print(f"Submission 1: {sub.name}")
        cfs_sub = CustomField.query.filter_by(name='pred_pick', submission_id=1).all()
        print(f"  Entries for Sub 1: {len(cfs_sub)}")
        if cfs_sub:
            print(f"  Sample ID of first entry: {cfs_sub[0].sample_id}")
            # Verify if this sample exists
            samp = Sample.query.get(cfs_sub[0].sample_id)
            print(f"  Sample Name: {samp.name if samp else 'None'}")
