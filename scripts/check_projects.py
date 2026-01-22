from app import app, db, Project

with app.app_context():
    projects = Project.query.all()
    print("-" * 30)
    print(f"Total Projects: {len(projects)}")
    for p in projects:
        print(f"ID: {p.id} | Name: '{p.name}' | Created: {p.created_at}")
    print("-" * 30)
