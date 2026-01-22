"""
Database migration script to add CustomField table.

Run this script to update your database schema:
    python migrate_add_custom_fields.py
"""

import sqlite3
import os

def migrate():
    db_path = os.path.join(os.path.dirname(__file__), 'database.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("The database will be created automatically when you run the app.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if custom_field table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='custom_field'")
    if cursor.fetchone():
        print("CustomField table already exists. Migration not needed.")
        conn.close()
        return
    
    print("Creating custom_field table...")
    
    # Create custom_field table
    cursor.execute('''
        CREATE TABLE custom_field (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            field_type VARCHAR(20) NOT NULL,
            value_text TEXT,
            value_float FLOAT,
            sample_id INTEGER,
            submission_id INTEGER,
            sample_name VARCHAR(100),
            FOREIGN KEY(sample_id) REFERENCES sample(id),
            FOREIGN KEY(submission_id) REFERENCES submission(id)
        )
    ''')
    
    # Create indexes for better query performance
    cursor.execute('CREATE INDEX ix_custom_field_sample_id ON custom_field(sample_id)')
    cursor.execute('CREATE INDEX ix_custom_field_submission_id ON custom_field(submission_id)')
    cursor.execute('CREATE INDEX ix_custom_field_name ON custom_field(name)')
    cursor.execute('CREATE INDEX ix_custom_field_type ON custom_field(field_type)')
    
    conn.commit()
    conn.close()
    
    print("Migration completed successfully!")
    print("CustomField table created with indexes.")

if __name__ == '__main__':
    migrate()
