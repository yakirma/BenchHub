from app import app

print("Dumping relevant app config:")
print(f"SERVER_NAME: {app.config.get('SERVER_NAME')}")
print(f"APPLICATION_ROOT: {app.config.get('APPLICATION_ROOT')}")
print(f"PREFERRED_URL_SCHEME: {app.config.get('PREFERRED_URL_SCHEME')}")
print(f"DEBUG: {app.config.get('DEBUG')}")
