"""
Migration script to fix CustomField data leakage issue.
Sets sample_id to NULL for all submission custom fields.
"""
from app import db, app, CustomField

with app.app_context():
    # Find all CustomFields that belong to submissions (have submission_id)
    # and incorrectly have sample_id set
    affected = CustomField.query.filter(
        CustomField.submission_id.isnot(None),
        CustomField.sample_id.isnot(None)
    ).count()
    
    print(f"Found {affected} submission custom fields with incorrect sample_id linkage")
    
    if affected > 0:
        # Set sample_id to NULL for all submission custom fields
        CustomField.query.filter(
            CustomField.submission_id.isnot(None)
        ).update({CustomField.sample_id: None}, synchronize_session=False)
        
        db.session.commit()
        print(f"✅ Fixed {affected} custom field records")
    else:
        print("✅ No records need fixing")
    
    # Verify the fix
    remaining = CustomField.query.filter(
        CustomField.submission_id.isnot(None),
        CustomField.sample_id.isnot(None)
    ).count()
    
    if remaining == 0:
        print("✅ Migration successful - no submission fields have sample_id set")
    else:
        print(f"⚠️  Warning: {remaining} records still have both submission_id and sample_id set")
