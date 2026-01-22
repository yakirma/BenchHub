
from app import app, db, GlobalMetric

with app.app_context():
    print("--- GlobalMetrics ---")
    metrics = GlobalMetric.query.all()
    for m in metrics:
        print(f"ID: {m.id}, Name: {m.name}")
        print(f"Code:\n{m.python_code}")
        print("-" * 20)