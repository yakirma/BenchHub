from app import app, db, Sample

def cleanup_ds_store():
    with app.app_context():
        # Find all samples named .DS_Store
        ds_store_samples = Sample.query.filter_by(name='.DS_Store').all()
        
        if not ds_store_samples:
            print("No .DS_Store samples found.")
            return
            
        print(f"Found {len(ds_store_samples)} .DS_Store samples. Deleting...")
        for sample in ds_store_samples:
            # Cascading deletes will handle MetricResults and CustomFields
            db.session.delete(sample)
            
        try:
            db.session.commit()
            print("Successfully deleted .DS_Store samples.")
        except Exception as e:
            db.session.rollback()
            print(f"Error during deletion: {e}")

if __name__ == "__main__":
    cleanup_ds_store()
