from flask import Flask, g

app = Flask(__name__)

@app.url_value_preprocessor
def pull_project_name(endpoint, values):
    print(f"Preprocessor called for {endpoint}, values: {values}")
    g.project_name = values.pop('project_name', None) if values else None
    print(f"Popped project_name, remaining values: {values}")

@app.route('/<project_name>/hello')
def hello(project_name):
    return f"Hello from {project_name}"

if __name__ == "__main__":
    with app.test_client() as client:
        try:
            print("Requesting /TestProject/hello")
            resp = client.get('/TestProject/hello')
            print(f"Response status: {resp.status_code}")
            print(f"Response data: {resp.data.decode('utf-8')}")
        except Exception as e:
            print(f"Exception during request: {e}")
            import traceback
            traceback.print_exc()
