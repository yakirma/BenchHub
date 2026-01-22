"""
Migration script to set histogram entropy as default metric for existing datasets.
"""
from app import db, app, Dataset

with app.app_context():
    # Find datasets with empty selected_metrics
    datasets_without_metrics = Dataset.query.filter(
        (Dataset.selected_metrics == '') | (Dataset.selected_metrics is None)
   ).all()
    
    print(f"Found {len(datasets_without_metrics)} datasets without metrics enabled")
    
    if datasets_without_metrics:
        for dataset in datasets_without_metrics:
            dataset.selected_metrics = 'hist_entropy'
            print(f"  - Enabled hist_entropy for dataset: {dataset.name}")
        
        db.session.commit()
        print(f"✅ Updated {len(datasets_without_metrics)} datasets")
    else:
        print("✅ No datasets need updating")
