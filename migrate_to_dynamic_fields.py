"""
Migration script to convert existing HistogramData and SignalShape to CustomField format.
This enables dynamic field handling for hist, pick, and wave_shape data.
"""
from app import db, app, Dataset, Sample, HistogramData, SignalShape, CustomField
import os

with app.app_context():
    print("=== Migrating hist, pick, wave_shape to CustomField ===\n")
    
    # 1. Migrate HistogramData to CustomField
    hist_count = 0
    histograms = HistogramData.query.all()
    print(f"Found {len(histograms)} histogram records to migrate")
    
    for hist_data in histograms:
        # Check if already migrated
        existing = CustomField.query.filter_by(
            sample_id=hist_data.sample_id,
            name='hist',
            field_type='histogram'
        ).first()
        
        if not existing:
            # Store bins/counts as JSON in value_text since it's not a simple scalar
            import json
            hist_custom = CustomField(
                name='hist',
                field_type='histogram',
                value_text=f'{{"bins": {hist_data.bins}, "counts": {hist_data.counts}}}',
                sample_id=hist_data.sample_id
            )
            db.session.add(hist_custom)
            hist_count += 1
    
    db.session.commit()
    print(f"✅ Migrated {hist_count} histogram records to CustomField\n")
    
    # 2. Migrate SignalShape to CustomField  
    shape_count = 0
    shapes = SignalShape.query.all()
    print(f"Found {len(shapes)} signal shape records to migrate")
    
    for shape_data in shapes:
        sample = Sample.query.get(shape_data.id)
        if sample:
            existing = CustomField.query.filter_by(
                sample_id=sample.id,
                name='wave_shape',
                field_type='scalar'
            ).first()
            
            if not existing:
                shape_custom = CustomField(
                    name='wave_shape',
                    field_type='scalar',
                    value_text=shape_data.shape_name,  # Store as text
                    sample_id=sample.id
                )
                db.session.add(shape_custom)
                shape_count += 1
    
    db.session.commit()
    print(f"✅ Migrated {shape_count} wave_shape records to CustomField\n")
    
    # 3. Create 'pick' CustomFields from pick/ folder if it exists
    pick_count = 0
    datasets = Dataset.query.all()
    
    for dataset in datasets:
        dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset.name)
        pick_folder = os.path.join(dataset_folder, 'pick')
        
        if os.path.exists(pick_folder):
            print(f"Found pick folder for dataset: {dataset.name}")
            samples = Sample.query.filter_by(dataset_id=dataset.id).all()
            
            for sample in samples:
                pick_file = os.path.join(pick_folder, f'{sample.name}.txt')
                if os.path.exists(pick_file):
                    existing = CustomField.query.filter_by(
                        sample_id=sample.id,
                        name='pick',
                        field_type='scalar'
                    ).first()
                    
                    if not existing:
                        try:
                            with open(pick_file, 'r') as f:
                                pick_value = float(f.read().strip())
                            pick_custom = CustomField(
                                name='pick',
                                field_type='scalar',
                                value_float=pick_value,
                                sample_id=sample.id
                            )
                            db.session.add(pick_custom)
                            pick_count += 1
                        except Exception as e:
                            print(f"  ⚠️  Failed to read pick for {sample.name}: {e}")
    
    db.session.commit()
    print(f"✅ Created {pick_count} pick CustomField records\n")
    
    print(f"=== Migration Complete ===")
    print(f"Total: {hist_count} hist + {shape_count} wave_shape + {pick_count} pick = {hist_count + shape_count + pick_count} new CustomFields")
