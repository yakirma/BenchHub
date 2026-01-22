from app import app, db, GlobalMetric

with app.app_context():
    m = GlobalMetric.query.filter_by(name='f1_score').first()
    if m:
        print("Updating f1_score logic...")
        m.python_code = """def f1_score(precision, recall):
    # Retrieve single value from list if passed as list (aggregated context)
    # Also handle None values safely (default to 0.0)
    
    p_val = precision[0] if isinstance(precision, list) and precision else precision
    p = p_val if isinstance(p_val, (int, float)) else 0.0
    
    r_val = recall[0] if isinstance(recall, list) and recall else recall
    r = r_val if isinstance(r_val, (int, float)) else 0.0
    
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)"""
        db.session.commit()
        print("Done.")
    else:
        print("f1_score metric not found.")
