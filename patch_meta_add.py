
from app import app, db, GlobalMetric

new_code = """
def meta_add(arg1, arg2):
    import numpy as np
    # Filter Nones
    v1 = [x for x in arg1 if x is not None]
    v2 = [x for x in arg2 if x is not None]
    
    m1 = np.mean(v1) if v1 else 0.0
    m2 = np.mean(v2) if v2 else 0.0
    return m1 + m2
""".strip()

with app.app_context():
    metric = GlobalMetric.query.filter_by(name='meta_add').first()
    if metric:
        print(f"Updating meta_add code...")
        metric.python_code = new_code
        db.session.commit()
        print("Update successful.")
    else:
        print("meta_add metric not found.")
