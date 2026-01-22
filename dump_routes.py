from app import app

print("Dumping all registered routes:")
for rule in app.url_map.iter_rules():
    print(f"{rule} -> {rule.endpoint}")
