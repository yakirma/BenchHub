from app import app, db
from sqlalchemy import text

with app.app_context():
    print("Adding 'accepts_aggregated_inputs' column to 'global_metric' table...")
    try:
        # SQLite syntax for adding a column
        db.session.execute(text("ALTER TABLE global_metric ADD COLUMN accepts_aggregated_inputs BOOLEAN DEFAULT 0"))
        db.session.commit()
        print("Success.")
    except Exception as e:
        print(f"Error (maybe column exists?): {e}")
        db.session.rollback()
