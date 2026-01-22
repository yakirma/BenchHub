"""
Script to reorder tags column to be right after name column
"""
from app import app, db, Leaderboard, Dataset

def reorder_columns(column_string, name_key, tags_key):
    """Reorder columns so tags appears right after name"""
    if not column_string:
        return column_string
    
    columns = [col.strip() for col in column_string.split(',') if col.strip()]
    
    # Remove tags from current position
    if tags_key in columns:
        columns.remove(tags_key)
    
    # Find name and insert tags right after it
    if name_key in columns:
        name_idx = columns.index(name_key)
        columns.insert(name_idx + 1, tags_key)
    
    return ','.join(columns)

with app.app_context():
    # Update all leaderboards
    leaderboards = Leaderboard.query.all()
    for lb in leaderboards:
        if lb.comparison_display_columns:
            new_order = reorder_columns(lb.comparison_display_columns, 'sample_name', 'dataset_tags')
            if new_order != lb.comparison_display_columns:
                print(f"Updating leaderboard {lb.id} ({lb.name})")
                print(f"  Old: {lb.comparison_display_columns}")
                print(f"  New: {new_order}")
                lb.comparison_display_columns = new_order
    
    # Update all datasets
    datasets = Dataset.query.all()
    for ds in datasets:
        if ds.display_columns:
            new_order = reorder_columns(ds.display_columns, 'sample_name', 'tags')
            if new_order != ds.display_columns:
                print(f"Updating dataset {ds.id} ({ds.name})")
                print(f"  Old: {ds.display_columns}")
                print(f"  New: {new_order}")
                ds.display_columns = new_order
    
    db.session.commit()
    print("\nDone! Column order updated.")
