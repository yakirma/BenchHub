from flask import Flask, render_template_string
import json

app = Flask(__name__)
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['APPLICATION_ROOT'] = '/'
app.config['PREFERRED_URL_SCHEME'] = 'http'

class MockDataset:
    def __init__(self):
        self.id = 1
        self.name = "Test Dataset"
        self.locked = False
        self.ground_truth_metadata = {}
        from datetime import datetime
        self.upload_date = datetime.now()
        self.description = "Test Description"
        self.num_samples = 10

class MockProject:
    def __init__(self):
        self.id = 1
        self.name = "Test Project"
        self.dataset_id = 1

@app.route('/')
def test_render():
    try:
        # Mocking empty lists for robustness
        samples_data_for_charts = [] 
        all_dataset_tags = []
        
        # Read the template file
        with open('templates/dataset_view.html', 'r') as f:
            template_content = f.read()
            
        # Render
        rendered = render_template_string(
            template_content, 
            dataset=MockDataset(),
            project=MockProject(),
            samples_data_for_charts=samples_data_for_charts,
            all_dataset_tags=all_dataset_tags,
            selected_display_columns=[],
            active_metrics=[],
            custom_scalar_metrics=[],
            sample_metric_options={},
            dataset_metrics_data=[],
            dataset_display_options={},
            custom_fields_map={},
            all_field_types=[],
            per_page_options=[10, 20, 50],
            current_per_page=10,
            paginated_samples=type('obj', (object,), {'items': [], 'total': 0, 'pages': 0, 'page': 1})(),
            global_settings={'image_width': 200, 'histogram_width': 300},
            enumerate=enumerate,
            len=len,
            list=list,
            str=str,
            zip=zip,
            url_for=lambda endpoint, **values: f"/mock/{endpoint}",
            current_user={'is_authenticated': True, 'username': 'test'},
            get_flashed_messages=lambda **kwargs: [],
            request={'endpoint': 'dataset_view', 'path': '/test'}
        )
        print("Template rendered successfully.")
        return rendered
    except Exception as e:
        print(f"Error rendering template: {e}")
        return str(e)

if __name__ == '__main__':
    with app.app_context():
        test_render()
