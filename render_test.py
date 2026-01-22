from jinja2 import Environment, FileSystemLoader
import json

env = Environment(loader=FileSystemLoader('templates'))
template = env.get_template('comparison.html')

class Dummy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getattr__(self, name): return ""
    def __getitem__(self, key): return ""
    def __iter__(self): return iter([])

context = {
    'leaderboard': Dummy(visualizations=''),
    'submissions': [],
    'comparison_data': [],
    'selected_metrics': ['mae'],
    'chart_metrics_data': [],
    'submissions_json': [],
    'sample_metric_options': {'mae': 'MAE'},
    'custom_metrics': [],
    'active_metrics': [],
    'all_sample_tag_names': [],
    'all_sample_prefixes': [],
    'selected_comparison_display_columns': [],
    'leaderboard_viz_list': [],
    'aggregated_viz_list': [],
    'all_custom_fields': [],
    'all_field_types': {},
    'dataset_custom_fields': set(),
    'submission_custom_fields': set(),
    'submission_has_histogram': {},
    'paginated_samples': Dummy(items=[]),
    'per_page_options': [5],
    'current_per_page': 5,
    'comparison_display_options': {},
    'active_metrics': []
}

try:
    rendered = template.render(**context)
    with open('rendered_comp.html', 'w') as f:
        f.write(rendered)
    print("SUCCESS")
except Exception as e:
    print(f"Error: {e}")
