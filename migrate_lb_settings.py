from app import app, db
import sqlite3
import os

def migrate_db():
    print("Checking database for missing columns...")
    with app.app_context():
        # Get database path from app config
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if db_uri.startswith('sqlite:///'):
            db_path = db_uri.replace('sqlite:///', '')
            # If path is relative, make it absolute based on instance folder or current dir
            # But here app.py uses absolute path construction for dtof_data_dir usually
            # Let's rely on exact path if possible, or use the one from config
            if not os.path.isabs(db_path):
                db_path = os.path.join(os.path.dirname(__file__), db_path)
            
            print(f"Database path: {db_path}")
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check for scalar_width
                try:
                    cursor.execute("SELECT scalar_width FROM leaderboard LIMIT 1")
                    print("Column 'scalar_width' already exists.")
                except sqlite3.OperationalError:
                    print("Adding column 'scalar_width'...")
                    cursor.execute("ALTER TABLE leaderboard ADD COLUMN scalar_width TEXT")
                    
                # Check for image_width
                try:
                    cursor.execute("SELECT image_width FROM leaderboard LIMIT 1")
                    print("Column 'image_width' already exists.")
                except sqlite3.OperationalError:
                    print("Adding column 'image_width'...")
                    cursor.execute("ALTER TABLE leaderboard ADD COLUMN image_width TEXT")
                
                conn.commit()
                conn.close()
                print("Migration complete.")
            except Exception as e:
                print(f"Migration failed: {e}")
        else:
            print("Non-SQLite database detected. Please handle migration manually.")

if __name__ == "__main__":
    migrate_db()
