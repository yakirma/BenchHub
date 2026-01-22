import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, Submission, CustomField, Sample

def inspect_custom_fields():
    with app.app_context():
        # Get the first submission that has custom fields
        submission = Submission.query.join(CustomField, CustomField.submission_id == Submission.id).first()
        
        if not submission:
            print("No submissions with custom fields found.")
            return

        print(f"Inspecting Submission ID: {submission.id}, Name: {submission.name}")
        
        # Get all custom fields for this submission
        custom_fields = CustomField.query.filter_by(submission_id=submission.id).all()
        
        print(f"Found {len(custom_fields)} custom fields.")
        print("-" * 60)
        print(f"{'ID':<5} | {'Name':<30} | {'Type':<10} | {'Sample Name':<20} | {'Sample ID':<10} | {'Value'}")
        print("-" * 60)
        
        for cf in custom_fields:
            val = cf.value_float if cf.field_type in ['scalar', 'metric'] else (cf.value_text[:20] + '...' if cf.value_text else 'None')
            print(f"{cf.id:<5} | {cf.name:<30} | {cf.field_type:<10} | {str(cf.sample_name):<20} | {str(cf.sample_id):<10} | {val}")
            
            if 'L1' in cf.name or 'l1' in cf.name:
                print(f"  ^^^ POTENTIAL MATCH FOR L1 ^^^")

        print("-" * 60)

        print("-" * 60)
        
        # Check specific sample
        sample_name = custom_fields[0].sample_name
        if sample_name:
             print(f"\nChecking exact lookup for sample_name='{sample_name}' and name='{custom_fields[0].name}'...")
             
             # Test ID lookup
             cf_id = CustomField.query.filter_by(submission_id=submission.id, name=custom_fields[0].name, sample_id=custom_fields[0].sample_id).first()
             print(f"Lookup by ID: {'FOUND' if cf_id else 'NOT FOUND'}")
             
             # Test Name lookup
             cf_name = CustomField.query.filter_by(submission_id=submission.id, name=custom_fields[0].name, sample_name=sample_name).first()
             print(f"Lookup by Name: {'FOUND' if cf_name else 'NOT FOUND'}")

        print("-" * 60)
        print("Checking Sample Table names:")
        samples = Sample.query.all()
        for s in samples[:20]:
            print(f"ID: {s.id:<4} | Name: '{s.name}'")

        if samples and custom_fields:
            s_match = next((s for s in samples if s.name == custom_fields[0].sample_name), None)
            print(f"\nDirect Match Check: Sample '{custom_fields[0].sample_name}' exists in Sample table? {'YES' if s_match else 'NO'}")

        print("-" * 60)
        print("Checking fields for 'sample9' specifically:")
        s9_fields = CustomField.query.filter_by(submission_id=submission.id, sample_name='sample9').all()
        if not s9_fields:
             # Try determining sample9 by ID if name lookup failed in script
             sample9 = Sample.query.filter_by(name='sample9').first()
             if sample9:
                 s9_fields = CustomField.query.filter_by(submission_id=submission.id, sample_id=sample9.id).all()
        
        for cf in s9_fields:
             print(f"Name: '{cf.name}' | Type: {cf.field_type} | Val: {cf.value_float or cf.value_text}")

        print("-" * 60)
        print("Checking for 'L1' anywhere in this submission:")
        l1_any = CustomField.query.filter_by(submission_id=submission.id, name='L1').first()
        print(f"Does 'L1' exist for ANY sample? {'YES' if l1_any else 'NO'}")

        from app import GlobalMetric
        print("-" * 60)
        print("Checking Global Metrics:")
        gms = GlobalMetric.query.all()
        for gm in gms:
            print(f"ID: {gm.id} | Name: '{gm.name}'")

if __name__ == "__main__":
    inspect_custom_fields()
