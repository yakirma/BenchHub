import shutil
import sys
import re
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, flash, abort, after_this_request, g, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import zipfile
from datetime import datetime
from celery import Celery
import json
import numpy as np
import threading
# For loading npz files
from scipy.optimize import curve_fit
from sqlalchemy import or_, not_, func
import io
import csv
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from metric_engine import evaluate_dynamic_metric, get_metric_context


app = Flask(__name__)
__version__ = "1.0.0"
app.secret_key = 'supersecretkey' # Needed for session management
# basedir = os.path.abspath(os.path.dirname(__file__)) # No longer used for data
user_home = os.path.expanduser("~")
dtof_data_dir = os.path.join(user_home, ".dtofbenchmarking")

# Ensure data directory exists
if not os.path.exists(dtof_data_dir):
    os.makedirs(dtof_data_dir, exist_ok=True)
    print(f"Created data directory at: {dtof_data_dir}")
else:
    print(f"Using data directory at: {dtof_data_dir}")

app.config.update(
    SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(dtof_data_dir, 'database.db'),
    UPLOAD_FOLDER=os.path.join(dtof_data_dir, 'uploads'),
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    SQLALCHEMY_ENGINE_OPTIONS={'connect_args': {'timeout': 60}}  # 60 seconds timeout
)

# Enable Write-Ahead Logging (WAL) for better concurrency
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=60000") # 60 seconds
    cursor.close()

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



# List of all available metrics
AVAILABLE_METRICS = []

# Helper to determine column priority for sorting
def get_column_priority(key, column_type=None, is_dataset_field=False):
    """
    Returns a numeric priority for a column key.
    Lower number means higher priority (appears first).
    Desired order: sample name - tags - charts - config - histograms - images - scalars
    Dataset fields (GT) always come before Submission fields within the same type.
    """
    # Group 0: Metadata
    if key == 'sample_name': return 0
    
    # Group 1: Tags (right after name)
    if key in ['dataset_tags', 'tags']: return 5
    
    # Group 2: Charts (Metrics)
    # gt_metrics removed
    # per_sample_metrics removed
    if key == 'per_source_stats': return 12
    
    # Group 3: Config/JSON
    if key in ['gt_config', 'signal_shape']: return 30
    if key == 'config': return 31
    # GT JSON fields: 35, Submission JSON fields: 36
    if column_type == 'json':
        return 35 if is_dataset_field else 36
    
    # Group 4: Histograms
    if key == 'gt_histogram': return 40
    if key == 'histogram' or key.startswith('histogram_'): return 41
    
    # Group 5: Images (Custom fields)
    # GT images: 50, Submission images: 51
    if column_type in ['image', 'depth']:
        return 50 if is_dataset_field else 51
    
    # Group 6: Scalars (Custom fields)
    # GT scalars: 60, Submission scalars: 61
    if column_type == 'scalar':
        return 60 if is_dataset_field else 61
        
    return 100
 

# Define available display options for Dataset View
# Reordered by priority
# Define available display options for Dataset View
# Reordered by priority: Name > Charts > Tags > Config > Histograms
DATASET_DISPLAY_OPTIONS = {
    'sample_name': {'label': 'Sample Name', 'type': 'text', 'default_width': '150px'},
    'tags': {'label': 'Tags', 'type': 'text', 'default_width': '150px'},
    # per_sample_metrics removed - GT metrics chart no longer needed
    'per_source_stats': {'label': 'GT Stats', 'type': 'stats', 'default_width': '200px'},
    'signal_shape': {'label': 'Signal Shape', 'type': 'text', 'default_width': '30px'},
    'histogram': {'label': 'Histogram', 'type': 'chart', 'default_width': '150px'},
}
DEFAULT_DATASET_DISPLAY_COLUMNS = ','.join(DATASET_DISPLAY_OPTIONS.keys()) # All enabled by default

# Define available display options for Comparison View
# Reordered by priority: Name > Charts > Tags > Config > Histograms
COMPARISON_DISPLAY_OPTIONS = {
    'sample_name': {'label': 'Sample Name', 'type': 'text', 'default_width': '150px'},
    'dataset_tags': {'label': 'Tags', 'type': 'text', 'default_width': '150px'},
    # gt_metrics removed - GT metrics chart no longer needed
    'per_source_stats': {'label': 'Source Stats (Scalars & Metrics)', 'type': 'text', 'default_width': '300px'},
    'per_sample_metrics': {'label': 'Metrics chart', 'type': 'chart', 'default_width': '300px'},
    'gt_config': {'label': 'GT Config', 'type': 'json', 'default_width': '150px'},
    'gt_histogram': {'label': 'GT Histogram', 'type': 'chart', 'default_width': '150px'},
}
DEFAULT_COMPARISON_DISPLAY_COLUMNS = ','.join([k for k in COMPARISON_DISPLAY_OPTIONS.keys() if k not in ['per_source_stats']])

# Define available visualisations
VISUALIZATION_OPTIONS = {
    'trend_overlay': 'Trend Overlay',
    'histogram_fit': 'Histogram Fit'
}

# Define available sample-level metrics (choosable like visualisations)
SAMPLE_METRIC_OPTIONS = {
    # Histogram entropy removed - now handled via custom fields only
}



def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

# Celery initialization moved after models to avoid circular imports
db = SQLAlchemy(app)

# Association Table for Submissions and Tags
submission_tags = db.Table('submission_tags',
    db.Column('submission_id', db.Integer, db.ForeignKey('submission.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

# Models
class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    # project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True) # Deprecated: Global Datasets
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    git_commit = db.Column(db.String(100))
    git_branch = db.Column(db.String(100))
    git_message = db.Column(db.String(200))
    display_columns = db.Column(db.String(500), nullable=False, default=DEFAULT_DATASET_DISPLAY_COLUMNS)
    visualizations = db.Column(db.String(500), nullable=False, default='') # Active visualizers
    selected_metrics = db.Column(db.String(500), nullable=False, default='') # No default metrics
    leaderboards = db.relationship('Leaderboard', backref='dataset', lazy=True, cascade="all, delete-orphan")
    samples = db.relationship('Sample', backref='dataset', lazy=True, cascade="all, delete-orphan")

class Sample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    tags = db.Column(db.String(500)) # Stores comma-separated tags

    histogram_data = db.relationship('HistogramData', backref='sample', uselist=False, lazy=True, cascade="all, delete-orphan")

    signal_shape = db.relationship('SignalShape', backref='sample', uselist=False, lazy=True, cascade="all, delete-orphan")
    config_data = db.relationship('ConfigData', backref='sample', uselist=False, lazy=True, cascade="all, delete-orphan")
    custom_fields = db.relationship('CustomField', backref='sample', lazy=True, cascade="all, delete-orphan", foreign_keys='CustomField.sample_id')
    
    # Backward compatibility helpers for migrated fields
    @property
    def histogram_data(self):
        """Get histogram data from either old HistogramData model or new CustomField"""
        # Try old model first (for data not yet migrated)
        old_hist = HistogramData.query.filter_by(sample_id=self.id).first()
        if old_hist:
            return old_hist
        
        # Try new CustomField
        hist_field = CustomField.query.filter_by(sample_id=self.id, name='hist', field_type='histogram').first()
        if hist_field and hist_field.value_text:
            # Create a mock HistogramData-like object
            # Note: value_text contains full JSON, we need to extract bins and counts as JSON strings
            import json
            data = json.loads(hist_field.value_text)
            class MockHistData:
                def __init__(self, bins_json, counts_json):
                    self.bins = bins_json  # Store as JSON string
                    self.counts = counts_json  # Store as JSON string
            # Re-serialize bins and counts as JSON strings
            return MockHistData(json.dumps(data['bins']), json.dumps(data['counts']))
        return None
    
    @property
    def signal_shape(self):
        """Get signal shape from either old SignalShape model or new CustomField"""
        # Try old model first
        old_shape = SignalShape.query.filter_by(id=self.id).first()
        if old_shape:
            return old_shape
        
        # Try new CustomField
        shape_field = CustomField.query.filter_by(sample_id=self.id, name='wave_shape', field_type='scalar').first()
        if shape_field:
            # Create a mock SignalShape-like object
            class MockSignalShape:
                def __init__(self, shape_name):
                    self.shape_name = shape_name
            return MockSignalShape(shape_field.value_text or 'gaussian')
        return None

class HistogramData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.Integer, db.ForeignKey('sample.id'), nullable=False)
    bins = db.Column(db.Text, nullable=False)
    counts = db.Column(db.Text, nullable=False)



class SignalShape(db.Model):
    id = db.Column(db.Integer, db.ForeignKey('sample.id'), primary_key=True)
    shape_name = db.Column(db.String(50), nullable=False)

class ConfigData(db.Model):
    id = db.Column(db.Integer, db.ForeignKey('sample.id'), primary_key=True)
    config_json = db.Column(db.Text, nullable=False)

    @property
    def parsed_config(self):
        return json.loads(self.config_json)

class CustomField(db.Model):
    """Store custom fields from datasets and submissions (images, scalars, metrics)"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # Folder name (e.g., 'custom_viz', 'metric_accuracy')
    field_type = db.Column(db.String(20), nullable=False)  # 'image', 'scalar', 'metric'
    value_text = db.Column(db.Text, nullable=True)  # For image paths or text values
    value_float = db.Column(db.Float, nullable=True)  # For scalar/metric values
    sample_id = db.Column(db.Integer, db.ForeignKey('sample.id'), nullable=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=True)
    sample_name = db.Column(db.String(100), nullable=True)  # Sample name for submission custom fields

    def get_value(self):
        """Helper to get the appropriate value based on type"""
        if self.field_type in ['scalar', 'metric'] and self.value_float is not None:
            return self.value_float
        return self.value_text

class GlobalMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True) # Unique name for global reference
    description = db.Column(db.Text, nullable=True)
    python_code = db.Column(db.Text, nullable=False)  # The function definition: def metric_func(...)
    is_aggregated = db.Column(db.Boolean, default=False, nullable=False)
    accepts_aggregated_inputs = db.Column(db.Boolean, default=False)
    # Default/Example mappings for UI hints? (Optional, maybe later)

class LeaderboardMetric(db.Model):
    """Link table between Leaderboard and GlobalMetric with specific settings"""
    id = db.Column(db.Integer, primary_key=True)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=False)
    global_metric_id = db.Column(db.Integer, db.ForeignKey('global_metric.id'), nullable=False)
    
    # Argument mappings: {arg_name: field_name} specific to this leaderboard's dataset
    arg_mappings = db.Column(db.Text, nullable=False)  # JSON string
    
    # Custom display name for this specific leaderboard usage
    # Custom display name for this specific leaderboard usage
    target_name = db.Column(db.String(100), nullable=True) 

    # Aggregation Settings
    pooling_type = db.Column(db.String(20), default='mean', nullable=False) # mean, median, percentile
    pooling_percentile = db.Column(db.Float, nullable=True) # Valid if pooling_type == 'percentile'
    
    # Optimization Goal
    sort_direction = db.Column(db.String(20), default='higher_is_better') # higher_is_better, lower_is_better

    global_metric = db.relationship('GlobalMetric', backref='leaderboard_usages')
    leaderboard = db.relationship('Leaderboard', backref=db.backref('leaderboard_metrics', lazy=True, cascade="all, delete-orphan"))

class GlobalVisualization(db.Model):
    """Global visualization definition (analogous to GlobalMetric but returns PIL.Image)"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)
    python_code = db.Column(db.Text, nullable=False)  # Function definition: def viz_func(...) -> PIL.Image
    is_aggregated = db.Column(db.Boolean, default=False, nullable=False)  # True: single image, False: per-sample
    accepts_aggregated_inputs = db.Column(db.Boolean, default=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)

class LeaderboardVisualization(db.Model):
    """Link table between Leaderboard and GlobalVisualization"""
    id = db.Column(db.Integer, primary_key=True)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=False)
    global_visualization_id = db.Column(db.Integer, db.ForeignKey('global_visualization.id'), nullable=False)
    
    # Argument mappings: {arg_name: field_name} specific to this leaderboard's dataset
    arg_mappings = db.Column(db.Text, nullable=False)  # JSON string
    
    # Custom display name for this specific leaderboard usage
    target_name = db.Column(db.String(100), nullable=True)
    
    # Display order for visualizations
    display_order = db.Column(db.Integer, default=0)
    
    global_visualization = db.relationship('GlobalVisualization', backref='leaderboard_usages')
    leaderboard = db.relationship('Leaderboard', backref=db.backref('leaderboard_visualizations', lazy=True, cascade="all, delete-orphan"))

class MetricResult(db.Model):
    """Stores calculated results for metrics to avoid re-calculation"""
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    leaderboard_metric_id = db.Column(db.Integer, db.ForeignKey('leaderboard_metric.id'), nullable=False)
    
    value = db.Column(db.Float, nullable=True) # Computed value
    error_message = db.Column(db.Text, nullable=True) # Traceback if failed
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    submission = db.relationship('Submission', backref=db.backref('metric_results', lazy='dynamic', cascade="all, delete-orphan"))
    leaderboard_metric = db.relationship('LeaderboardMetric', backref=db.backref('results', cascade='all, delete-orphan'))

class Leaderboard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    summary_metrics = db.Column(db.String(200), nullable=False) # are, l1, l2 etc.
    comparison_display_columns = db.Column(db.String(500), nullable=False, default=DEFAULT_COMPARISON_DISPLAY_COLUMNS)
    visualizations = db.Column(db.String(500), nullable=False, default='') # Active visualizers
    selected_metrics = db.Column(db.String(500), default='') # Comma separated list of targets
    summary_metrics = db.Column(db.String(500), default='') # Initial metrics to show
    metric_directions = db.Column(db.Text, default='{}') # JSON: metric_name -> "higher_is_better" or "lower_is_better"
    metric_aggregation = db.Column(db.Text, default='{}') # JSON: metric_name -> {"type": "mean|median|percentile", "percentile": 95}
    scalar_width = db.Column(db.String(50), nullable=True) # Override for scalar column width
    image_width = db.Column(db.String(50), nullable=True) # Override for image column width
    last_sample_filter = db.Column(db.Text, nullable=True) # JSON string: store last used filter settings
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True) # Added for Project Refactor (Migrated)
    submissions = db.relationship('Submission', backref='leaderboard', lazy=True, cascade="all, delete-orphan")

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Datasets are now global. 
    # datasets = db.relationship('Dataset', backref='project', lazy=True, cascade="all, delete-orphan") 
    leaderboards = db.relationship('Leaderboard', backref='project', lazy=True, cascade="all, delete-orphan")

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=False)
    git_commit = db.Column(db.String(100))
    git_branch = db.Column(db.String(100))
    git_message = db.Column(db.String(200))
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_archived = db.Column(db.Boolean, default=False, nullable=False)
    processing_status = db.Column(db.String(50), default='Pending')
    last_sample_filter = db.Column(db.Text, nullable=True) # JSON store of filters used for metrics
    tags = db.relationship('Tag', secondary=submission_tags, lazy='subquery', backref=db.backref('submissions', lazy=True))
    custom_fields = db.relationship('CustomField', backref='submission', lazy=True, cascade="all, delete-orphan", foreign_keys='CustomField.submission_id')

# Initialize Celery after models are defined
celery = make_celery(app)

# Import tasks to register them with Celery
import tasks  # noqa: F401

@app.context_processor
def inject_version():
    return dict(version=__version__)



from metric_engine import evaluate_dynamic_metric, get_metric_context, sort_metrics_by_dependency

def process_dataset_zip(zip_path, dataset_name, override=False):
    """
    Helper to process a dataset zip file and create database entries.
    If override is True, deletes existing dataset with the same name.
    Returns (success: bool, message: str, dataset_id: int or None)
    """
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_dataset_extract_' + datetime.now().strftime('%Y%m%d%H%M%S%f'))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Initial Collision Check
        existing = Dataset.query.filter_by(name=dataset_name).first()
        if existing:
            if override:
                # Delete existing dataset and its files
                dataset_folder_name = secure_filename(existing.name)
                shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name), ignore_errors=True)
                db.session.delete(existing)
                db.session.commit()
            else:
                return False, f"Dataset '{dataset_name}' already exists.", None

        # Create preliminary entry
        new_dataset = Dataset(name=dataset_name)
        db.session.add(new_dataset)
        db.session.flush()

        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
            # Identify root folder
            extracted_items = [item for item in os.listdir(temp_dir) 
                             if item != '__MACOSX' and not item.startswith('.') and item != os.path.basename(zip_path)]
            
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
                real_dataset_name = extracted_items[0]
                dataset_content_path = os.path.join(temp_dir, real_dataset_name)
            else:
                real_dataset_name = dataset_name
                dataset_content_path = temp_dir

            # Re-check collision if name changed based on zip content
            if real_dataset_name != dataset_name:
                existing = Dataset.query.filter_by(name=real_dataset_name).first()
                if existing:
                    if override:
                        # Delete existing dataset and its files
                        dataset_folder_name = secure_filename(existing.name)
                        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name), ignore_errors=True)
                        db.session.delete(existing)
                        db.session.commit()
                    else:
                        return False, f"Dataset '{real_dataset_name}' (extracted from ZIP) already exists.", None
                new_dataset.name = real_dataset_name
                db.session.add(new_dataset)
                db.session.flush()

            # Permanent storage setup
            dataset_folder_name = secure_filename(new_dataset.name)
            dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Copy original ZIP
            original_zip_dest = os.path.join(dataset_dir, f"{dataset_folder_name}.zip")
            shutil.copy2(zip_path, original_zip_dest)

            # Git metadata
            git_info_path = os.path.join(dataset_content_path, 'git.info')
            if os.path.exists(git_info_path):
                try:
                    with open(git_info_path, 'r') as git_file:
                        git_data = json.load(git_file)
                        new_dataset.git_commit = git_data.get('commit', '')
                        new_dataset.git_branch = git_data.get('branch', '')
                        new_dataset.git_message = git_data.get('message', '')
                except: pass

            # Discover samples - scan all folders dynamically
            sample_names = set()
            for folder_name in os.listdir(dataset_content_path):
                folder_path = os.path.join(dataset_content_path, folder_name)
                if os.path.isdir(folder_path) and folder_name not in ['__MACOSX', 'git.info']:
                    for fname in os.listdir(folder_path):
                        sample_names.add(os.path.splitext(fname)[0])
            
            if not sample_names:
                return False, "No valid samples (hist, config, etc.) found in ZIP.", None

            # Create sample records
            for s_name in sample_names:
                sample = Sample(dataset_id=new_dataset.id, name=s_name)
                
                db.session.add(sample)
                db.session.flush()


                # NOTE: hist and wave_shape are now handled as dynamic custom fields
                # No special processing needed here
                
                # Config data - DEPRECATED: Now handled as regular json custom field
                # Keep for backward compatibility with existing data that has ConfigData
                # config_file = os.path.join(dataset_content_path, 'config', f'{s_name}.json')
                # if os.path.exists(config_file):
                #     try:
                #         with open(config_file, 'r') as f:
                #             db.session.add(ConfigData(id=sample.id, config_json=json.dumps(json.load(f))))
                #     except: pass

            # Custom fields detection
            dataset_custom_fields_dir = os.path.join(dataset_dir, 'images')
            os.makedirs(dataset_custom_fields_dir, exist_ok=True)
            
            known_folders = {'git.info'}  # Only special metadata, config is now a regular json field
            custom_fields = detect_custom_fields(dataset_content_path, sample_names, known_folders, is_submission=False)
            
            for field_name, field_info in custom_fields.items():
                field_type = field_info['type']
                for s_name, value in field_info['data'].items():
                    sample = Sample.query.filter_by(dataset_id=new_dataset.id, name=s_name).first()
                    if sample:
                        if field_type == 'image':
                            field_folder = os.path.join(dataset_custom_fields_dir, field_name)
                            os.makedirs(field_folder, exist_ok=True)
                            dest_path = os.path.join(field_folder, os.path.basename(value))
                            shutil.copy2(value, dest_path)
                            rel_path = os.path.relpath(dest_path, app.config['UPLOAD_FOLDER'])
                            custom_field = CustomField(name=field_name, field_type='image', value_text=rel_path, sample_id=sample.id)
                        elif field_type == 'depth':
                            field_folder = os.path.join(dataset_dir, 'depth_maps', field_name)
                            os.makedirs(field_folder, exist_ok=True)
                            dest_path = os.path.join(field_folder, os.path.basename(value))
                            shutil.copy2(value, dest_path)
                            rel_path = os.path.relpath(dest_path, app.config['UPLOAD_FOLDER'])
                            custom_field = CustomField(name=field_name, field_type='depth', value_text=rel_path, sample_id=sample.id)
                        elif field_type == 'json':
                            # Store JSON files preserving original folder structure
                            field_folder = os.path.join(dataset_dir, field_name)
                            os.makedirs(field_folder, exist_ok=True)
                            dest_path = os.path.join(field_folder, os.path.basename(value))
                            shutil.copy2(value, dest_path)
                            rel_path = os.path.relpath(dest_path, app.config['UPLOAD_FOLDER'])
                            custom_field = CustomField(name=field_name, field_type='json', value_text=rel_path, sample_id=sample.id)
                        elif field_type == 'histogram':
                            # Load histogram data from npz and store as JSON in value_text
                            try:
                                with np.load(value) as data:
                                    hist_json = json.dumps({
                                        'bins': data['bins'].tolist(),
                                        'counts': data['counts'].tolist()
                                    })
                                    custom_field = CustomField(name=field_name, field_type='histogram', value_text=hist_json, sample_id=sample.id)
                            except Exception as e:
                                print(f"Failed to load histogram {value}: {e}")
                                continue
                        elif field_type == 'text':
                            custom_field = CustomField(name=field_name, field_type='text', value_text=value, sample_id=sample.id)
                            # Special handling for 'tags' field to populate the built-in sample.tags
                            if field_name == 'tags':
                                # Normalize tags (replace newlines/spaces with commas)
                                cleaned_tags = re.sub(r'[\n\r]+', ',', value).strip()
                                sample.tags = cleaned_tags
                        else: # scalar
                            custom_field = CustomField(name=field_name, field_type='scalar', value_float=value, sample_id=sample.id)
                        db.session.add(custom_field)
            
            db.session.commit()
            return True, f"Uploaded '{new_dataset.name}' ({len(sample_names)} samples)", new_dataset.id

    except Exception as e:
        db.session.rollback()
        return False, f"Error: {str(e)}", None
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def detect_custom_fields(base_path, sample_names, known_folders, is_submission=False):
    """
    Detect custom fields (images, scalars, metrics) in a dataset or submission folder.
    
    Args:
        base_path: Path to the dataset or submission folder
        sample_names: List of sample names to look for
        known_folders: Set of known folder names to exclude
        is_submission: True if detecting submission fields, False for dataset fields
    
    Returns:
        Dictionary mapping field_name -> {type, sample_name -> value/path}
    """
    custom_fields = {}
    
    if not os.path.exists(base_path):
        return custom_fields
    
    # Get all folders in the base path
    all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    # Filter out known folders
    custom_folders = [f for f in all_folders if f not in known_folders]
    
    for folder_name in custom_folders:
        folder_path = os.path.join(base_path, folder_name)
        
        # Determine field type
        is_metric = is_submission and folder_name.startswith('metric_')
        field_type = 'metric' if is_metric else None
        
        # Check what's inside the folder
        field_data = {}
        
        folder_files = os.listdir(folder_path)
        
        for sample_name in sample_names:
            # Check for image files
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                img_file_name = f'{sample_name}{ext}'
                if img_file_name in folder_files:
                    if field_type is None:
                        field_type = 'image'
                    field_data[sample_name] = os.path.join(folder_path, img_file_name)
                    break
            
            # Check for text files with scalar values or text tags
            txt_file_name = f'{sample_name}.txt'
            if txt_file_name in folder_files:
                try:
                    with open(os.path.join(folder_path, txt_file_name), 'r') as f:
                        content = f.read().strip()
                        try:
                            val = float(content)
                            if field_type is None or field_type == 'metric':
                                 field_type = 'metric' if is_metric else 'scalar'
                            field_data[sample_name] = val
                        except ValueError:
                            # If not a float, treat as text (e.g., tags)
                            if field_type is None:
                                field_type = 'text'
                            field_data[sample_name] = content
                except:
                    pass
            
            # Check for JSON files
            json_file_name = f'{sample_name}.json'
            if json_file_name in folder_files:
                if field_type is None:
                    field_type = 'json'
                field_data[sample_name] = os.path.join(folder_path, json_file_name)

            # Check for histogram files (.npz) if folder starts with 'hist_', is 'raw_histogram', or is 'hist'
            if folder_name.startswith('hist_') or folder_name in ['raw_histogram', 'hist']:
                 npz_file_name = f'{sample_name}.npz'
                 if npz_file_name in folder_files:
                     if field_type is None:
                         field_type = 'histogram'
                     field_data[sample_name] = os.path.join(folder_path, npz_file_name)

            # Check for depth files (.npz) if folder starts with 'raw_'
            if folder_name.startswith('raw_'):
                 # Pattern: <sample_name>_<width>x<height>.npz
                 # Since we don't know width/height, we search the file list
                 prefix = f"{sample_name}_"
                 for fname in folder_files:
                     if fname.startswith(prefix) and fname.endswith('.npz'):
                         # Verify middle part is dimensions (optional but good for strictness)
                         # parts: [sample_name, dims.npz]
                         rest = fname[len(prefix):]
                         if re.match(r'\d+x\d+\.npz$', rest):
                             if field_type is None:
                                 field_type = 'depth'
                             field_data[sample_name] = os.path.join(folder_path, fname)
                             break
        
        if field_type:
            custom_fields[folder_name] = {'type': field_type, 'data': field_data}
            
    return custom_fields

def calculate_submission_metrics(submission, sample, gt_pick, signal_shape, active_metrics):

    pred_data = {'sub_id': submission.id, 'peak': None, 'bins': [], 'counts': []}
    metrics = {}
    
    submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id))
    
    # try/except block removed as specific file loading was removed.
    # Note: exception handling for separate steps (like histograms) is inside loop.

    # Helper to calculate entropy
    def calc_entropy(counts):
         if len(counts) == 0: return 0.0
         c = counts[counts > 0]
         if c.sum() > 0:
             p = c / c.sum()
             res = float(-np.sum(p * np.log2(p)))
             # print(f"DEBUG: calc_entropy input sum={c.sum()} shape={counts.shape} result={res}")
             return res
         return 0.0

    # General approach: Find any folder starting with "hist_" that has an .npz for this sample
    if os.path.exists(submission_folder):
        for folder_name in os.listdir(submission_folder):
            folder_path = os.path.join(submission_folder, folder_name)
            # Strict check for 'hist_' prefix for dynamic ones, plus legacy 'raw_histogram'
            if os.path.isdir(folder_path) and (folder_name.startswith('hist_') or folder_name == 'raw_histogram'):
                hist_file = os.path.join(folder_path, f'{sample.name}.npz')
                # print(f"DEBUG: Checking hist file: {hist_file} (Exists: {os.path.exists(hist_file)})")
                if os.path.exists(hist_file):
                    try:
                        with np.load(hist_file) as data:
                            bins = data['bins'].tolist()
                            counts = data['counts'].tolist()
                            
                            # Store properly namespaced data for all histograms
                            pred_data[f'histogram_{folder_name}'] = {'bins': bins, 'counts': counts}
                            
                            # Restore top-level bins/counts for backward compatibility (e.g. zoom modal fallback)
                            # especially for the primary 'raw_histogram'
                            if folder_name == 'raw_histogram' or (not pred_data['bins'] and not pred_data['counts']):
                                pred_data['bins'] = bins
                                pred_data['counts'] = counts
                            
                            # Custom entropy metric removed as obsolete
                            # entropy_val = calc_entropy(np.array(counts))
                            # metric_name = f'hist_entropy_{folder_name}'
                            # pred_data[metric_name] = entropy_val
                            # metrics[metric_name] = entropy_val
                    except Exception as e:
                        print(f"Error loading histogram from {folder_name}: {e}")
                        # metrics[f'hist_entropy_{folder_name}'] = -1.0

    # Fallback: if 'hist_entropy' wasn't set but is requested
    # We leave it as None. Existing logic set it to 0 if missing, which caused confusion in charts.
    # if 'hist_entropy' in active_metrics and pred_data['hist_entropy'] is None:
    #      pred_data['hist_entropy'] = 0.0



    if pred_data['bins'] and pred_data['counts']:
        bins_np, counts_np = np.array(pred_data['bins']), np.array(pred_data['counts'])
         # Removed curve_fit_error calculation logic as requested

    
    # Add custom fields to pred_data and metrics
    custom_fields_for_sample = [cf for cf in submission.custom_fields if cf.sample_name == sample.name]
    print(f"DEBUG: Found {len(custom_fields_for_sample)} custom fields for sample {sample.name}, submission {submission.id}")
    for cf in custom_fields_for_sample:
        print(f"DEBUG:   Custom field: {cf.name}, type: {cf.field_type}, value: {cf.value_float if cf.field_type != 'image' else cf.value_text}")
        if cf.field_type == 'image':
            # Store image path
            pred_data[cf.name] = cf.value_text
        elif cf.field_type == 'scalar':
            # Store scalar value
            pred_data[cf.name] = cf.value_float
        elif cf.field_type == 'metric':
            # Store metric value
            metrics[cf.name] = cf.value_float
            pred_data[cf.name] = cf.value_float
            print(f"DEBUG:   Added metric {cf.name} = {cf.value_float} to metrics dict")

    return pred_data, metrics

# --- Global Settings Management ---
SETTINGS_FILE = os.path.join(dtof_data_dir, 'global_settings.json')

class GlobalSettings:
    def __init__(self):
        self.defaults = {
            'scalar_width': '150px',
            'image_width': '300px',
            'theme_mode': 'light'
        }
        self.settings = self.load_settings()

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return self.defaults.copy()
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                # Merge with defaults to ensure all keys exist
                settings = self.defaults.copy()
                settings.update(saved)
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self.defaults.copy()

    def save_settings(self, new_settings):
        try:
            # Update internal state with valid keys only
            for key in self.defaults:
                if key in new_settings:
                    self.settings[key] = new_settings[key]
            
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
            
    def get(self, key):
        return self.settings.get(key, self.defaults.get(key))

    @property
    def scalar_width(self):
        return self.settings.get('scalar_width', '150px')

    @property
    def image_width(self):
        return self.settings.get('image_width', '300px')

    @property
    def theme_mode(self):
        return self.settings.get('theme_mode', 'light')

global_settings = GlobalSettings()

@app.context_processor
def inject_projects():
    if g.get('current_project'):
        # Just optimization: if we are in a project, we might want to list others.
        # But actually we want the list always available if we show the dropdown.
        # We'll just fetch all. It's usually small.
        return dict(all_projects=Project.query.order_by(Project.name).all())
    return dict(all_projects=[])

@app.context_processor
def inject_settings():
    # Per-user theme preference from cookie
    user_theme = request.cookies.get('theme_mode')
    
    # Create a wrapper or use the object directly but inject the user preference
    settings_dict = {
        'scalar_width': request.cookies.get('scalar_width', global_settings.scalar_width),
        'image_width': request.cookies.get('image_width', global_settings.image_width),
        'metric_chart_width': request.cookies.get('metric_chart_width', '300px'),
        'tags_width': request.cookies.get('tags_width', '150px'),
        'config_width': request.cookies.get('config_width', '150px'),
        'name_width': request.cookies.get('name_width', '150px'),
        'histogram_width': request.cookies.get('histogram_width', '150px'),
        'theme_mode': user_theme if user_theme in ['light', 'dark'] else global_settings.theme_mode
    }
    return {'global_settings': settings_dict}

@app.route('/app-settings', methods=['GET', 'POST'])
def app_settings():
    if request.method == 'POST':
        theme_mode = request.form.get('theme_mode', 'light')
        scalar_width = request.form.get('scalar_width', '150px')
        image_width = request.form.get('image_width', '300px')
        metric_chart_width = request.form.get('metric_chart_width', '300px')
        tags_width = request.form.get('tags_width', '150px')
        config_width = request.form.get('config_width', '150px')
        name_width = request.form.get('name_width', '150px')
        histogram_width = request.form.get('histogram_width', '150px')
        
        # We redirect back to the same page but with cookies set
        resp = make_response(redirect(url_for('app_settings')))
        resp.set_cookie('theme_mode', theme_mode, max_age=30*24*60*60) # 30 days
        # Store all widths in cookies to make them per-user
        resp.set_cookie('scalar_width', scalar_width, max_age=30*24*60*60)
        resp.set_cookie('image_width', image_width, max_age=30*24*60*60)
        resp.set_cookie('metric_chart_width', metric_chart_width, max_age=30*24*60*60)
        resp.set_cookie('tags_width', tags_width, max_age=30*24*60*60)
        resp.set_cookie('config_width', config_width, max_age=30*24*60*60)
        resp.set_cookie('name_width', name_width, max_age=30*24*60*60)
        resp.set_cookie('histogram_width', histogram_width, max_age=30*24*60*60)
        
        flash('General settings updated successfully!', 'success')
        return resp
        
    # GET: Prepare current settings from cookies or global defaults
    settings = {
        'theme_mode': request.cookies.get('theme_mode', global_settings.theme_mode),
        'scalar_width': request.cookies.get('scalar_width', global_settings.scalar_width),
        'image_width': request.cookies.get('image_width', global_settings.image_width),
        'metric_chart_width': request.cookies.get('metric_chart_width', '300px'),
        'tags_width': request.cookies.get('tags_width', '150px'),
        'config_width': request.cookies.get('config_width', '150px'),
        'name_width': request.cookies.get('name_width', '150px'),
        'histogram_width': request.cookies.get('histogram_width', '150px')
    }
    return render_template('app_settings.html', settings=settings)

@app.route('/<project_name>/settings', methods=['GET', 'POST'])
def settings_page(project_name):
    if request.method == 'POST':
        try:
            # Handle Project Renaming & Description
            if hasattr(g, 'current_project') and g.current_project:
                changes_made = False
                
                # Update Name
                new_project_name = request.form.get('project_name')
                if new_project_name and new_project_name != g.current_project.name:
                    # Check for uniqueness
                    existing = Project.query.filter_by(name=new_project_name).first()
                    if existing:
                        flash(f'Project name "{new_project_name}" is already taken.', 'danger')
                    else:
                        g.current_project.name = new_project_name
                        changes_made = True
                        
                # Update Description
                new_description = request.form.get('project_description')
                if new_description is not None and new_description != (g.current_project.description or ''):
                    g.current_project.description = new_description
                    changes_made = True

                if changes_made:
                    db.session.commit()
                    flash('Project settings updated successfully.', 'success')


        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('settings_page'))
    
    # If no current project, redirect to dashboard
    if not hasattr(g, 'current_project') or not g.current_project:
        return redirect(url_for('list_projects'))

    return render_template('settings.html')


# --- End Global Settings ---

# --- Project Management Logic ---

@app.url_value_preprocessor
def pull_project_name(endpoint, values):
    g.project_name = values.get('project_name') if values else None

@app.url_defaults
def add_project_name(endpoint, values):
    # Only add project_name if the endpoint expects it
    if app.url_map.is_endpoint_expecting(endpoint, 'project_name'):
        if 'project_name' in values:
            return
        if g.get('project_name'):
            values['project_name'] = g.project_name
        elif g.get('current_project'):
            values['project_name'] = g.current_project.name

# Helper to check if an endpoint expects a variable
def is_endpoint_expecting(self, endpoint, variable):
    rules = self._rules_by_endpoint.get(endpoint, [])
    for rule in rules:
        if variable in rule.arguments:
            return True
    return False

# Monkey patch Flask's url_map to easily check for expected arguments
from werkzeug.routing import Map
Map.is_endpoint_expecting = is_endpoint_expecting

@app.before_request
def load_project_context():
    # Public routes and API routes that don't need project context check
    # list_projects, app_settings, docs NOW ALLOW context loading (via cookie) so tabs stay visible
    public_endpoints = ['static', 'create_project', 'select_project', 'check_and_migrate_db', 
                        'legacy_leaderboard_redirect', 'legacy_comparison_redirect']
    if request.endpoint and (request.endpoint in public_endpoints or request.endpoint.startswith('static')):
        return
    
    # Programmatic API access should skip redirection
    if request.path.startswith('/api/'):
        return

    # 1. Check for project name in URL (populated by pull_project_name) OR query param
    project_name = g.get('project_name') or request.args.get('project_name')
    
    if project_name:
        project = Project.query.filter_by(name=project_name).first()
        if project:
            g.current_project = project
            
            # Sync cookie to match the URL-specified project so global pages (like /datasets)
            # remember this context.
            @after_this_request
            def remember_project_cookie(response):
                response.set_cookie('active_project_id', str(project.id), max_age=30*24*60*60)
                return response
            return
        elif g.get('project_name'):
            # Only redirect if it was a path parameter that failed; query args might be stale/typos we can ignore
            flash(f"Project '{project_name}' not found.", "error")
            return redirect(url_for('list_projects'))

    # 2. Check for active project cookie (Legacy/Fallback)
    project_id = request.cookies.get('active_project_id')
    
    if project_id:
        project = Project.query.get(project_id)
        if project:
            g.current_project = project
            # If we are on a route that SHOULD have a project prefix but doesn't,
            # we might want to redirect. However, for now, we'll let it be as-is
            # unless we are on the root '/'.
            if request.path == '/' or request.path == '/deprecated_index':
                return redirect(url_for('index', project_name=project.name))
            return

    # If no valid project selected, and not on a public page, redirect to project selector
    # We allow '/' to redirect, but maybe we want '/' to be the selector if no cookie?
    # Let's say: if visiting root or any other page without project -> redirect to /projects
    if request.endpoint != 'list_projects':
        return redirect(url_for('list_projects'))


@app.route('/projects')
def list_projects():
    projects = Project.query.order_by(Project.created_at).all()
    active_project_id = request.cookies.get('active_project_id')
    return render_template('projects.html', projects=projects, active_project_id=active_project_id, version=__version__)

@app.route('/projects/create', methods=['POST'])
def create_project():
    name = request.form.get('name')
    description = request.form.get('description')
    
    if not name:
        flash("Project name is required.", "danger")
        return redirect(url_for('list_projects'))
        
    if Project.query.filter_by(name=name).first():
        flash(f"Project '{name}' already exists.", "danger")
        return redirect(url_for('list_projects'))
        
    new_project = Project(name=name, description=description)
    db.session.add(new_project)
    db.session.commit()
    
    flash(f"Project '{name}' created!", "success")
    return redirect(url_for('list_projects'))

@app.route('/projects/select/<int:project_id>')
def select_project(project_id):
    project = Project.query.get_or_404(project_id)
    
    # Determine redirect target
    next_page = request.args.get('next')
    if next_page == 'dashboard':
        target_url = url_for('index', project_name=project.name)
    else:
        target_url = url_for('list_projects')
        
    resp = redirect(target_url)
    resp.set_cookie('active_project_id', str(project.id), max_age=30*24*60*60) # 30 days
    
    if next_page != 'dashboard':
        flash(f"Project '{project.name}' is now active.", "success")
        
    return resp

@app.route('/projects/<int:project_id>/rename', methods=['POST'])
def rename_project(project_id):
    project = Project.query.get_or_404(project_id)
    new_name = request.form.get('name')
    
    if not new_name:
         flash("Name cannot be empty.", "danger")
    elif Project.query.filter(Project.name == new_name, Project.id != project_id).first():
         flash(f"Project name '{new_name}' is already taken.", "danger")
    else:
        project.name = new_name
        db.session.commit()
        flash("Project renamed successfully.", "success")
        
    return redirect(url_for('list_projects'))


@app.route('/')
def root_redirect():
    if hasattr(g, 'current_project'):
        return redirect(url_for('index', project_name=g.current_project.name))
    return redirect(url_for('list_projects'))

# --- Legacy Redirects (Backward Compatibility) ---
def get_fallback_project_name():
    """Helper to guess a project for legacy URLs."""
    if hasattr(g, 'current_project') and g.current_project:
        return g.current_project.name
    
    # Try cookie manually if g didn't catch it (though before_request should have)
    project_id = request.cookies.get('active_project_id')
    if project_id:
        p = Project.query.get(project_id)
        if p: return p.name
        
    # Fallback to first available project
    p = Project.query.order_by(Project.created_at).first()
    if p: return p.name
    
    return None

@app.route('/leaderboard/<int:leaderboard_id>')
def legacy_leaderboard_redirect(leaderboard_id):
    p_name = get_fallback_project_name()
    if p_name:
        return redirect(url_for('leaderboard_view', project_name=p_name, leaderboard_id=leaderboard_id))
    return redirect(url_for('list_projects'))



@app.route('/comparison/<int:leaderboard_id>')
def legacy_comparison_redirect(leaderboard_id):
    p_name = get_fallback_project_name()
    if p_name:
        return redirect(url_for('comparison_view', project_name=p_name, leaderboard_id=leaderboard_id))
    return redirect(url_for('list_projects'))
# -------------------------------------------------

@app.route('/<project_name>/')
def index(project_name):
    if not hasattr(g, 'current_project'):
        # Should be caught by before_request, but safety check
        return redirect(url_for('list_projects'))
        
    # Updated for Global Datasets architecture
    leaderboards = g.current_project.leaderboards
    # Datasets are global, passed for "Create Leaderboard" dropdown
    datasets = Dataset.query.order_by(Dataset.name).all()
    
    return render_template('index.html', leaderboards=leaderboards, datasets=datasets)

# --- End Project Management Logic ---

@app.route('/deprecated_index') # Renamed old index to preserve code structure if needed, or we just replaced it above.
def deprecated_index():
    datasets = Dataset.query.all()
    leaderboards = Leaderboard.query.all()
    return render_template('index.html', datasets=datasets, leaderboards=leaderboards, available_metrics=AVAILABLE_METRICS)

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/edit', methods=['GET', 'POST'])
def edit_leaderboard(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    if request.method == 'POST':
        if 'name' in request.form:
            leaderboard.name = request.form.get('name', leaderboard.name)
        if 'scalar_width' in request.form:
            leaderboard.scalar_width = request.form.get('scalar_width') or None
        if 'image_width' in request.form:
            leaderboard.image_width = request.form.get('image_width') or None
        
        if 'summary_metrics' in request.form:
            summary_metrics = request.form.getlist('summary_metrics')
            leaderboard.summary_metrics = ','.join(summary_metrics)
        
        # Save Metric Directions
        directions = json.loads(leaderboard.metric_directions) if leaderboard.metric_directions else {}
        has_direction_updates = False
        for key, value in request.form.items():
            if key.startswith('direction_'):
                metric_name = key.replace('direction_', '')
                directions[metric_name] = value
                has_direction_updates = True
        if has_direction_updates:
            leaderboard.metric_directions = json.dumps(directions)
            # Synchronize with LeaderboardMetric objects
            for lm in leaderboard.leaderboard_metrics:
                target = lm.target_name if lm.target_name else lm.global_metric.name
                if target in directions:
                    lm.sort_direction = directions[target]
        
        # Update Aggregation Settings for existing metrics AND custom metrics
        metric_aggregation = {}
        if leaderboard.metric_aggregation:
             try:
                 metric_aggregation = json.loads(leaderboard.metric_aggregation)
             except:
                 metric_aggregation = {}

        has_aggregation_updates = False
        for key, value in request.form.items():
            if key.startswith('aggregation_type_'):
                metric_name = key.replace('aggregation_type_', '')
                agg_type = value
                agg_perc_key = f"aggregation_percentile_{metric_name}"
                agg_perc = request.form.get(agg_perc_key)
                
                perc_val = None
                if agg_perc and agg_perc.strip():
                    try:
                        perc_val = float(agg_perc)
                    except ValueError:
                        perc_val = None
                
                has_aggregation_updates = True
                
                # Check if it is a dynamic metric
                lm = next((m for m in leaderboard.leaderboard_metrics if (m.target_name == metric_name or m.global_metric.name == metric_name)), None)
                
                if lm:
                    lm.pooling_type = agg_type
                    lm.pooling_percentile = perc_val
                else:
                    # Save to JSON
                    metric_aggregation[metric_name] = {
                        'type': agg_type,
                        'percentile': perc_val
                    }
        
        if has_aggregation_updates:
            leaderboard.metric_aggregation = json.dumps(metric_aggregation)

        db.session.commit()
        
        if has_aggregation_updates:
            # Trigger recalculation for all submissions only if aggregation changed
            submissions = Submission.query.filter_by(leaderboard_id=leaderboard.id).all()
            for sub in submissions:
                 tasks.process_submission.delay(sub.id)
            flash('Leaderboard configuration updated. Recalculation started.', 'success')
        elif has_direction_updates:
            flash('Leaderboard coloring updated.', 'success')
        else:
            flash('Leaderboard settings updated.', 'success')
            
        return redirect(url_for('edit_leaderboard', project_name=project_name, leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))
        
    # Get available fields for mapping (sampling)
    fields_set = set()
    
    # 1. Check GT data
    samples = Sample.query.filter_by(dataset_id=leaderboard.dataset_id).all()
    if any(s.histogram_data for s in samples):
        fields_set.add('gt_histogram')
    
    # 2. Check Submission data
    submissions = Submission.query.filter_by(leaderboard_id=leaderboard.id, processing_status='Processed').all()
    sub_ids = [sub.id for sub in submissions]
    
    # Separate fields for UI datalists
    dataset_fields_set = set(['peak', 'entropy', 'num_samples']) # Standard stuff
    submission_fields_set = set(['sub_peak', 'sub_entropy'])
    
    # 3. Check Custom Fields (Database)
    # GT Custom Fields
    dataset_custom_fields = CustomField.query.filter(CustomField.sample_id.in_([s.id for s in samples])).all()
    for cf in dataset_custom_fields:
        if cf.field_type in ['metric', 'scalar', 'image']:
            dataset_fields_set.add(cf.name)
            
    # Submission Custom Fields
    if sub_ids:
        submission_custom_fields = CustomField.query.filter(CustomField.submission_id.in_(sub_ids)).all()
        for cf in submission_custom_fields:
            if cf.field_type in ['metric', 'scalar', 'image']:
                submission_fields_set.add(cf.name)
                
    # 4. Include already defined Leaderboard Metrics (to allow chaining/dependencies)
    per_sample_metrics = set([])
    aggregated_metrics_list = set([])
    
    for lm in leaderboard.leaderboard_metrics:
        # Use target_name if available (custom alias), otherwise global metric name
        name_to_add = lm.target_name if lm.target_name else lm.global_metric.name
        
        if lm.global_metric.is_aggregated:
            aggregated_metrics_list.add(name_to_add)
        else:
            per_sample_metrics.add(name_to_add)
            
    # Add to submission fields for backward compatibility/default view if needed,
    # but also prepare separate lists for UI
    for m in per_sample_metrics:
        submission_fields_set.add(m)
        
    # Note: aggregated metrics are usually NOT mixed with per-sample submission fields 
    # unless we explicitly want them to show up in "Submission/Metric" dropdown.
    # The user wants "New Category". So we will pass 'aggregated_metrics_list' separately.

    # Standard calculated metrics (Wait, these are results, not sources usually.
    # But ARE/L1/L2 could be used as input for other metrics?)
    # Let's keep them in submission fields or creating a mixed list? 
    # Actually, they are "Calculated Metrics".
    # But user might want to map "gt_list" to "gt_peak" (dataset field).
    
    dataset_fields = sorted(list(dataset_fields_set))
    submission_fields = sorted(list(submission_fields_set))
    
    
    # helper for editing metric directions: get all possible metrics that could appear
    all_known_metrics = set([])
    # Dynamic metrics
    for lm in leaderboard.leaderboard_metrics:
        # Use target_name if available, consistent with metric_to_lm
        all_known_metrics.add(lm.target_name if lm.target_name else lm.global_metric.name)
    # Custom metrics from database (linked to this dataset/submissions)
    # Similar logic to leaderboard_view discovery
    dataset_custom_metrics = CustomField.query.filter(CustomField.sample_id.in_([s.id for s in samples]), CustomField.field_type == 'metric').all()
    for cf in dataset_custom_metrics:
        all_known_metrics.add(f'gt_{cf.name}') # Although leaderboard usually aggregates sub metrics? 
        # Actually leaderboard.html only shows standard, dynamic, and sub-custom metrics. Not GT custom metrics usually (unless dynamic uses them).
        # But let's stick to what's shown in leaderboard table loop.
    
    # Submission custom metrics
    if sub_ids:
        submission_custom_metrics = CustomField.query.filter(CustomField.submission_id.in_(sub_ids), CustomField.field_type =='metric').all()
        for cf in submission_custom_metrics:
            all_known_metrics.add(cf.name)
            
    sorted_metrics = sorted(list(all_known_metrics))
    current_directions = json.loads(leaderboard.metric_directions) if leaderboard.metric_directions else {}
    current_aggregation = json.loads(leaderboard.metric_aggregation) if leaderboard.metric_aggregation else {}
    
    # Get all global metrics for selection in UI
    global_metrics = GlobalMetric.query.order_by(GlobalMetric.name).all()
    
    # Get all global visualizations for selection in UI
    global_visualizations = GlobalVisualization.query.order_by(GlobalVisualization.name).all()

    # Create map for efficient Template lookup
    metric_to_lm = {}
    for lm in leaderboard.leaderboard_metrics:
         name = lm.target_name if lm.target_name else lm.global_metric.name
         metric_to_lm[name] = lm

    return render_template('edit_leaderboard.html', 
                           leaderboard=leaderboard,
                           dataset_fields=dataset_fields,
                           submission_fields=submission_fields,
                           aggregated_metrics=sorted(list(aggregated_metrics_list)),
                           per_sample_metrics=sorted(list(per_sample_metrics)),
                           available_metrics=sorted_metrics,
                           all_known_metrics=sorted_metrics,
                           current_directions=current_directions,
                           current_aggregation=current_aggregation,
                           metric_to_lm=metric_to_lm,
                           global_metrics=global_metrics,
                           global_visualizations=global_visualizations,
                           all_projects=Project.query.all())
                           

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/leaderboard_metric/add', methods=['POST'])
def add_leaderboard_metric(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    try:
        global_metric_id = request.form.get('global_metric_id')
        if not global_metric_id:
            raise ValueError("No global metric selected")
            
        gm = GlobalMetric.query.get(global_metric_id)
        if not gm:
            raise ValueError("Global metric not found")

        # Arg mappings
        arg_names = request.form.getlist('arg_name[]')
        sources = request.form.getlist('source[]')
        fields = request.form.getlist('field_name[]')
        
        arg_mappings = {}
        for arg, source, field in zip(arg_names, sources, fields):
            if arg and field:
                 # Construct internal field key based on source
                 if source == 'gt':
                     key = f'gt_{field}'
                 elif source == 'sub':
                     key = f'sub_{field}'
                 elif source == 'scalar':
                     key = f'SCALAR:{field}'
                 else:
                     key = field 
                 arg_mappings[arg] = key
        
        
        # Determine unique display name
        requested_name = request.form.get('display_name', '').strip()
        
        # If no display name requested, generate one from inputs
        if not requested_name:
            field_mappings = [f"{a}={v}" for a, v in zip(arg_names, fields) if a and v]
            if field_mappings:
                requested_name = f"{gm.name}({', '.join(field_mappings)})"
            else:
                requested_name = gm.name
        
        existing_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=leaderboard.id).all()
        existing_names = set()
        for m in existing_metrics:
            existing_names.add(m.target_name if m.target_name else m.global_metric.name)
            
        final_name = requested_name
        counter = 1
        while final_name in existing_names:
            final_name = f"{requested_name}_{counter}"
            counter += 1
        
        lm = LeaderboardMetric(
            leaderboard_id=leaderboard.id,
            global_metric_id=gm.id,
            arg_mappings=json.dumps(arg_mappings),
            target_name=final_name,
            pooling_type='mean',
            pooling_percentile=None,
            sort_direction=request.form.get('sort_direction', 'higher_is_better')
        )
        db.session.add(lm)
        
        # Auto-add to summary metrics for display
        current_metrics = [m.strip() for m in leaderboard.summary_metrics.split(',') if m.strip()]
        if final_name not in current_metrics:
            current_metrics.append(final_name)
            leaderboard.summary_metrics = ','.join(current_metrics)
            
        db.session.commit()
        
        # Trigger recalculation for all submissions
        submissions = Submission.query.filter_by(leaderboard_id=leaderboard.id).all()
        for sub in submissions:
             tasks.process_submission.delay(sub.id)
             
        flash(f'Metric "{final_name}" added to leaderboard. Recalculation started.', 'success')
    except Exception as e:
        flash(f'Error adding metric: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/leaderboard/<int:leaderboard_id>/import_settings', methods=['POST'])
def import_leaderboard_settings(leaderboard_id):
    target_lb = Leaderboard.query.get_or_404(leaderboard_id)
    source_lb_id = request.form.get('source_leaderboard_id')
    
    if not source_lb_id:
        flash("Please select a source leaderboard.", "warning")
        return redirect(url_for('edit_leaderboard', project_name=target_lb.project.name, leaderboard_id=target_lb.id, _anchor=request.form.get('active_tab')))
        
    source_lb = Leaderboard.query.get_or_404(source_lb_id)
    
    try:
        # 1. Copy direct fields
        target_lb.summary_metrics = source_lb.summary_metrics
        target_lb.visualizations = source_lb.visualizations
        target_lb.selected_metrics = source_lb.selected_metrics
        target_lb.metric_directions = source_lb.metric_directions
        target_lb.metric_aggregation = source_lb.metric_aggregation
        target_lb.comparison_display_columns = source_lb.comparison_display_columns
        target_lb.scalar_width = source_lb.scalar_width
        target_lb.image_width = source_lb.image_width
        target_lb.last_sample_filter = source_lb.last_sample_filter
        
        # 2. Copy LeaderboardMetric entries (complex objects)
        # First, clear existing metrics on target to avoid duplicates/conflicts
        for old_metric in target_lb.leaderboard_metrics:
            db.session.delete(old_metric)
            
        # Then clone from source
        for src_metric in source_lb.leaderboard_metrics:
            new_metric = LeaderboardMetric(
                leaderboard_id=target_lb.id,
                global_metric_id=src_metric.global_metric_id,
                arg_mappings=src_metric.arg_mappings,
                target_name=src_metric.target_name,
                pooling_type=src_metric.pooling_type,
                pooling_percentile=src_metric.pooling_percentile,
                sort_direction=src_metric.sort_direction
            )
            db.session.add(new_metric)

        # 3. Copy LeaderboardVisualization entries
        # First, clear existing visualizations on target
        for old_vis in target_lb.leaderboard_visualizations:
            db.session.delete(old_vis)
            
        # Then clone from source
        for src_vis in source_lb.leaderboard_visualizations:
            new_vis = LeaderboardVisualization(
                leaderboard_id=target_lb.id,
                global_visualization_id=src_vis.global_visualization_id,
                arg_mappings=src_vis.arg_mappings,
                target_name=src_vis.target_name,
                display_order=src_vis.display_order
            )
            db.session.add(new_vis)
            
        db.session.commit()
        flash(f"Settings imported from '{source_lb.name}'.", "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"Error importing settings: {e}", "error")
        
    return redirect(url_for('edit_leaderboard', project_name=target_lb.project.name, leaderboard_id=target_lb.id, _anchor=request.form.get('active_tab')))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/leaderboard_metric/<int:metric_id>/edit', methods=['POST'])
def edit_leaderboard_metric(project_name, leaderboard_id, metric_id):
    lm = LeaderboardMetric.query.get_or_404(metric_id)
    if lm.leaderboard_id != leaderboard_id:
        abort(403)
        
    try:
        old_mappings_json = lm.arg_mappings
        
        arg_names = request.form.getlist('arg_name[]')
        sources = request.form.getlist('source[]')
        fields = request.form.getlist('field_name[]')
        

        
        
        arg_mappings = {}
        for arg, source, field in zip(arg_names, sources, fields):
            if arg and field:
                 # Construct internal field key based on source
                 if source == 'gt':
                     key = f'gt_{field}'
                 elif source == 'sub':
                     key = f'sub_{field}'
                 elif source == 'scalar':
                     key = f'SCALAR:{field}'
                 else:
                     key = field 
                 arg_mappings[arg] = key
        
        lm.arg_mappings = json.dumps(arg_mappings)

        # Update Display Name (target_name)
        requested_name = request.form.get('display_name', '').strip()
        
        # If no display name requested, generate one from inputs
        if not requested_name:
            field_values = [v for v in fields if v]
            if field_values:
                requested_name = f"{lm.global_metric.name}[{', '.join(field_values)}]"
            else:
                requested_name = lm.global_metric.name
        
        # Uniqueness check against OTHER metrics in this leaderboard
        existing_metrics = LeaderboardMetric.query.filter(
            LeaderboardMetric.leaderboard_id == leaderboard_id,
            LeaderboardMetric.id != metric_id
        ).all()
        existing_names = set()
        for m in existing_metrics:
            existing_names.add(m.target_name if m.target_name else m.global_metric.name)
            
        final_name = requested_name
        counter = 1
        while final_name in existing_names:
            final_name = f"{requested_name}_{counter}"
            counter += 1
            
        lm.target_name = final_name
        
        # Update Sort Direction
        if 'sort_direction' in request.form:
            lm.sort_direction = request.form.get('sort_direction')

        db.session.commit()
        
        # Determine if recalculation is actually needed
        new_mappings_json = json.dumps(arg_mappings)
        needs_recalculation = (new_mappings_json != old_mappings_json)
        
        if needs_recalculation:
            # Trigger recalculation for all submissions
            submissions = Submission.query.filter_by(leaderboard_id=leaderboard_id).all()
            for sub in submissions:
                 tasks.process_submission.delay(sub.id)
            flash(f'Metric "{final_name}" updated. Recalculation started.', 'success')
        else:
            flash(f'Metric "{final_name}" updated.', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating metric: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/leaderboard_metric/<int:metric_id>/delete', methods=['POST'])
def delete_leaderboard_metric(project_name, leaderboard_id, metric_id):
    lm = LeaderboardMetric.query.get_or_404(metric_id)
    if lm.leaderboard_id != leaderboard_id:
        abort(403)
        
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    metric_name = lm.target_name if lm.target_name else lm.global_metric.name
    
    # 1. Remove from selected_metrics (Display Columns)
    if leaderboard.selected_metrics:
        current_selected = [m.strip() for m in leaderboard.selected_metrics.split(',') if m.strip()]
        if metric_name in current_selected:
            current_selected.remove(metric_name)
            leaderboard.selected_metrics = ','.join(current_selected)
            
    # 2. Delete the record
    db.session.delete(lm)
    db.session.commit()
    
    # 3. Trigger recalculation for all submissions
    submissions = Submission.query.filter_by(leaderboard_id=leaderboard.id).all()
    for sub in submissions:
         tasks.process_submission.delay(sub.id)
         
    flash(f'Metric "{metric_name}" removed. Recalculation started.', 'success')
    return redirect(url_for('edit_leaderboard', project_name=project_name, leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

# ==================== Leaderboard Visualization Management Routes ====================

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/leaderboard_visualization/add', methods=['POST'])
def add_leaderboard_visualization(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    try:
        global_viz_id = request.form.get('global_visualization_id')
        if not global_viz_id:
            raise ValueError("No global visualization selected")
            
        gv = GlobalVisualization.query.get(global_viz_id)
        if not gv:
            raise ValueError("Global visualization not found")

        # Arg mappings
        arg_names = request.form.getlist('viz_arg_name[]')
        sources = request.form.getlist('viz_source[]')
        fields = request.form.getlist('viz_field_name[]')
        
        arg_mappings = {}
        for arg, source, field in zip(arg_names, sources, fields):
            if arg and field:
                 if source == 'gt':
                     key = f'gt_{field}'
                 elif source == 'sub':
                     key = f'sub_{field}'
                 elif source == 'scalar':
                     key = f'SCALAR:{field}'
                 else:
                     key = field 
                 arg_mappings[arg] = key
        
        # Determine unique display name
        requested_name = request.form.get('display_name', '').strip()
        if not requested_name:
            requested_name = gv.name
        
        existing_vizs = LeaderboardVisualization.query.filter_by(leaderboard_id=leaderboard.id).all()
        existing_names = set()
        for v in existing_vizs:
            existing_names.add(v.target_name if v.target_name else v.global_visualization.name)
            
        final_name = requested_name
        counter = 1
        while final_name in existing_names:
            final_name = f"{requested_name}_{counter}"
            counter += 1
        
        lv = LeaderboardVisualization(
            leaderboard_id=leaderboard.id,
            global_visualization_id=gv.id,
            arg_mappings=json.dumps(arg_mappings),
            target_name=final_name,
            display_order=int(request.form.get('display_order', 0))
        )
        db.session.add(lv)
        db.session.commit()
        flash(f'Visualization "{final_name}" added to leaderboard', 'success')
    except Exception as e:
        flash(f'Error adding visualization: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', project_name=project_name, leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/leaderboard_visualization/<int:viz_id>/edit', methods=['POST'])
def edit_leaderboard_visualization(project_name, leaderboard_id, viz_id):
    lv = LeaderboardVisualization.query.get_or_404(viz_id)
    try:
        # Arg mappings
        arg_names = request.form.getlist('viz_arg_name[]')
        sources = request.form.getlist('viz_source[]')
        fields = request.form.getlist('viz_field_name[]')
        
        arg_mappings = {}
        for arg, source, field in zip(arg_names, sources, fields):
            if arg and field:
                 if source == 'gt':
                     key = f'gt_{field}'
                 elif source == 'sub':
                     key = f'sub_{field}'
                 elif source == 'scalar':
                     key = f'SCALAR:{field}'
                 else:
                     key = field 
                 arg_mappings[arg] = key
        
        lv.arg_mappings = json.dumps(arg_mappings)
        lv.target_name = request.form.get('display_name', '').strip()
        lv.display_order = int(request.form.get('display_order', 0))
        
        db.session.commit()
        flash(f'Visualization updated', 'success')
    except Exception as e:
        flash(f'Error updating visualization: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', project_name=project_name, leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/leaderboard_visualization/<int:viz_id>/delete', methods=['POST'])
def delete_leaderboard_visualization(project_name, leaderboard_id, viz_id):
    lv = LeaderboardVisualization.query.get_or_404(viz_id)
    try:
        db.session.delete(lv)
        db.session.commit()
        flash('Visualization removed from leaderboard', 'success')
    except Exception as e:
        flash(f'Error removing visualization: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', project_name=project_name, leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

# ==================== Visualization Execution Route ====================

def extract_viz_arg_value(sample, submission, field_key):
    """Helper to extract argument value for visualization from sample/submission."""
    import json
    if field_key:
        field_key = field_key.strip()
    
    value = None
    if field_key.startswith('gt_'):
        field_name = field_key[3:]
        if not sample: return None
        
        # Optimize: Preloaded custom fields would be better, but simple query for now
        cf = CustomField.query.filter_by(sample_id=sample.id, name=field_name).first()
        if cf:
            value = cf.get_value()
        elif field_name == 'histogram':
            value = sample.histogram_data
        elif field_name == 'config':
            value = sample.config_data
            
    elif field_key.startswith('sub_'):
        field_name = field_key[4:]
        if submission and sample:
            cf = CustomField.query.filter_by(submission_id=submission.id, sample_id=sample.id, name=field_name).first()
            if not cf:
                # Fallback to sample_name if sample_id is missing (common for uploaded submissions)
                cf = CustomField.query.filter_by(submission_id=submission.id, sample_name=sample.name, name=field_name).first()
            
            if cf:
                value = cf.get_value()
                
    elif field_key.startswith('SCALAR:'):
        value = field_key[7:]
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
            
    else:
        # Fallback: Try as a direct metric/scalar lookup on the submission
        if submission and sample:
            # Debug logging
            print(f"DEBUG: Lookup field='{field_key}', sub={submission.id}, sample='{sample.name}' (ID: {sample.id})")
            
            # 1. Try by sample_id
            cf = CustomField.query.filter_by(submission_id=submission.id, sample_id=sample.id, name=field_key).first()
            
            if not cf:
                 # 2. Fallback to sample_name
                cf = CustomField.query.filter_by(submission_id=submission.id, sample_name=sample.name, name=field_key).first()
                if not cf:
                    print(f"DEBUG: FAILED lookup for '{field_key}' on sample '{sample.name}'")
                    # Check if it's a known GlobalMetric that hasn't been computed
                    from app import GlobalMetric
                    gm = GlobalMetric.query.filter_by(name=field_key).first()
                    if gm:
                         print(f"DEBUG: '{field_key}' is a GlobalMetric (ID: {gm.id}) but has no computed value for this submission. The user needs to run 'Recalculate' for this submission.")
            
            if cf:
                value = cf.get_value()
            
    return value

@app.route('/<project_name>/visualization/<int:lv_id>/execute/<int:sample_id>')
@app.route('/<project_name>/visualization/<int:lv_id>/execute/<int:sample_id>/<int:submission_id>')
def execute_visualization(project_name, lv_id, sample_id, submission_id=None):
    """Execute a visualization and return the image as PNG."""
    import io
    import hashlib
    import time
    from PIL import Image
    
    lv = LeaderboardVisualization.query.get_or_404(lv_id)
    gv = lv.global_visualization
    sample = Sample.query.get_or_404(sample_id)
    submission = Submission.query.get(submission_id) if submission_id else None
    
    # Generate cache key
    code_hash = hashlib.md5((gv.python_code or "").encode()).hexdigest()
    mapping_hash = hashlib.md5((lv.arg_mappings or "").encode()).hexdigest()
    cache_key = f"viz_{lv_id}_{sample_id}_{submission_id or 'none'}_{code_hash}_{mapping_hash}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_dir = os.path.join(os.getcwd(), 'data', 'viz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_hash}.png")
    
    # Return cached image if exists
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype='image/png')
    
    try:
        # Parse arg mappings
        arg_mappings = json.loads(lv.arg_mappings) if lv.arg_mappings else {}
        
        # Build argument values from sample/submission data
        kwargs = {}
        for arg_name, field_key in arg_mappings.items():
            kwargs[arg_name] = extract_viz_arg_value(sample, submission, field_key)
        
        # Execute visualization code
        exec_globals = {
            'Image': Image,
            'PIL': __import__('PIL'),
            'numpy': __import__('numpy'),
            'np': __import__('numpy'),
        }
        exec(gv.python_code, exec_globals)
        
        # Find the function
        func_name = None
        import re
        match = re.search(r'def\s+(\w+)\s*\(', gv.python_code)
        if match:
            func_name = match.group(1)
            
        if func_name and func_name in exec_globals:
            viz_func = exec_globals[func_name]
            result_image = viz_func(**kwargs)
            
            if isinstance(result_image, Image.Image):
                # Save to cache
                result_image.save(cache_path, 'PNG')
                
                # Return image
                img_io = io.BytesIO()
                result_image.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
        
        # Fallback: return placeholder
        return create_error_image("No result")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_error_image(str(e)[:50])

@app.route('/<project_name>/visualization/<int:lv_id>/execute_aggregated')
@app.route('/<project_name>/visualization/<int:lv_id>/execute_aggregated/<int:submission_id>')
def execute_aggregated_visualization(project_name, lv_id, submission_id=None):
    """Execute an aggregated visualization (across all samples) and return the image as PNG."""
    import io
    import hashlib
    import time
    from PIL import Image
    
    lv = LeaderboardVisualization.query.get_or_404(lv_id)
    if not lv.global_visualization.is_aggregated:
        return create_error_image("Not an aggregated visualization")

    leaderboard = lv.leaderboard
    submission = Submission.query.get(submission_id) if submission_id else None
    
    # Generate cache key with hashes
    code_hash = hashlib.md5((lv.global_visualization.python_code or "").encode()).hexdigest()
    mapping_hash = hashlib.md5((lv.arg_mappings or "").encode()).hexdigest()
    cache_key = f"viz_agg_{lv_id}_{submission_id or 'none'}_{code_hash}_{mapping_hash}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_dir = os.path.join(os.getcwd(), 'data', 'viz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_hash}.png")
    
    # Return cached image if exists
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype='image/png')
            
    try:
        # Fetch all samples for the dataset
        # Respect filters if we could access them, but for MVP we use all samples
        all_samples = Sample.query.filter_by(dataset_id=leaderboard.dataset_id).order_by(Sample.name).all()

        # Parse arg mappings
        arg_mappings = json.loads(lv.arg_mappings) if lv.arg_mappings else {}

        # Build argument values - LISTS of values across all samples
        kwargs = {}
        for arg_name, field_key in arg_mappings.items():
            if field_key.startswith('SCALAR:'):
                kwargs[arg_name] = extract_viz_arg_value(None, None, field_key)
            else:
                # Iterate all samples
                values_list = []
                for sample in all_samples:
                    val = extract_viz_arg_value(sample, submission, field_key)
                    values_list.append(val)
                kwargs[arg_name] = values_list

        # Execute Code
        # We need the same execution logic as non-aggregated
        code = lv.global_visualization.python_code
        exec_globals = globals().copy()
        exec_globals.update({'np': np, 'plt': plt, 'Image': Image}) # Add convenience imports

        local_scope = {} # kwargs handling inside wrapper?

        # We need to find the function, same as execute_visualization
        exec(code, exec_globals)

        func_name = None
        import re
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            func_name = match.group(1)

        if func_name and func_name in exec_globals:
            viz_func = exec_globals[func_name]
            result_image = viz_func(**kwargs)

            if isinstance(result_image, Image.Image):
                # Save to cache
                result_image.save(cache_path, 'PNG')

                # Return image
                img_io = io.BytesIO()
                result_image.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')

        return create_error_image("No result or invalid function")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_error_image(f"Execution Error: {str(e)}")

def create_error_image(error_text):
    """Create a simple error image with text."""
    from PIL import Image, ImageDraw
    import io
    
    img = Image.new('RGB', (200, 100), color=(255, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), f"Error: {error_text[:30]}", fill=(100, 0, 0))
    
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


# Helper for DLP-safe Base64 decoding
def handle_dlp_safe_code(code_str):
    """Detect and decode Base64 obfuscated code"""
    if not code_str:
        return code_str
    
    code_str = code_str.strip()
    if code_str.startswith('BASE64:'):
        import base64
        try:
            # Remove prefix and decode
            encoded_part = code_str[7:]
            decoded_bytes = base64.b64decode(encoded_part)
            # Try decoding as utf-8, fallback to latin-1
            try:
                return decoded_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return decoded_bytes.decode('latin-1')
        except Exception as e:
            app.logger.error(f"DLP Decoding Error: {e}")
            # If decoding fails, return as is or maybe flash warning
    return code_str

# Deprecated/Removed: add_dynamic_metric
    pass
    
@app.route('/<project_name>/metrics')
def metrics_view(project_name):
    metrics = GlobalMetric.query.order_by(GlobalMetric.name).all()
    return render_template('metrics.html', metrics=metrics)

def extract_code_from_file(file_storage):
    """Extract Python code from a .txt, .py, or .zip file with robustness for high-security environments."""
    if not file_storage or not file_storage.filename:
        return None
    
    try:
        file_storage.seek(0)
        filename = file_storage.filename.lower()
        
        if filename.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_storage, 'r') as z:
                    # Filter namelist to ignore Mac metadata and directories
                    valid_files = [n for n in z.namelist() 
                                  if n.lower().endswith(('.txt', '.py')) 
                                  and not n.startswith('__MACOSX') 
                                  and not os.path.basename(n).startswith('.')]
                    
                    if not valid_files:
                        app.logger.warning(f"ZIP upload: No valid .txt or .py files found in {filename}")
                        return None
                    
                    # Take the most likely candidate (shallowest path first)
                    valid_files.sort(key=lambda x: x.count('/'))
                    target_name = valid_files[0]
                    content = z.read(target_name)
                    
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            return content.decode(encoding)
                        except UnicodeDecodeError:
                            continue
            except Exception as e:
                app.logger.error(f"ZIP Extraction Error: {e}")
                return None
        elif filename.endswith(('.txt', '.py')):
            content = file_storage.read()
            # Try to decode content first
            text_content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content:
                # Apply DLP decoding to the text content (handles obfuscated .txt files)
                return handle_dlp_safe_code(text_content)
    except Exception as e:
        print(f"File Storage Error: {e}")
    
    return None

@app.route('/<project_name>/metrics/create', methods=['POST'])
def create_global_metric(project_name):
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = handle_dlp_safe_code(request.form.get('python_code', '').strip())
        is_aggregated = 'is_aggregated' in request.form
        accepts_aggregated_inputs = 'accepts_aggregated_inputs' in request.form

        # Handle file upload if present
        metric_file = request.files.get('metric_file')
        file_code = extract_code_from_file(metric_file)
        
        if file_code:
            python_code = file_code
        elif metric_file and metric_file.filename:
            flash(f'Found file {metric_file.filename} but could not extract Python code from it. Check file encoding or zip content.', 'warning')
            # If the user intentionally used the placeholder, don't save it
            if "Implementation will be loaded from ZIP" in python_code:
                 return redirect(url_for('metrics_view'))

        if not python_code or not python_code.strip() or "Implementation will be loaded from ZIP" in python_code:
            flash('Metric code is required and cannot be the ZIP placeholder.', 'danger')
            return redirect(url_for('metrics_view'))

        metric = GlobalMetric(
            name=name,
            description=description,
            python_code=python_code,
            is_aggregated=is_aggregated,
            accepts_aggregated_inputs=accepts_aggregated_inputs
        )
        db.session.add(metric)
        db.session.commit()
        flash(f'Metric "{name}" created successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating metric: {e}', 'danger')
    
    return redirect(url_for('metrics_view'))

@app.route('/<project_name>/metrics/<int:metric_id>/edit', methods=['POST'])
def edit_global_metric(project_name, metric_id):
    metric = GlobalMetric.query.get_or_404(metric_id)
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = handle_dlp_safe_code(request.form.get('python_code', '').strip())
        
        # Handle file upload if present
        metric_file = request.files.get('metric_file')
        file_code = extract_code_from_file(metric_file)
        
        if file_code:
            python_code = file_code
        elif metric_file and metric_file.filename:
            flash(f'Found file {metric_file.filename} but could not extract Python code from it.', 'warning')
            if "Implementation will be loaded from ZIP" in python_code:
                 return redirect(url_for('metrics_view'))

        if not python_code or not python_code.strip() or "Implementation will be loaded from ZIP" in python_code:
            # Fallback to current code if they didn't provide a valid new one but we're in edit mode
            # unless they specifically sent the placeholder.
            if "Implementation will be loaded from ZIP" in python_code:
                flash('Invalid code submitted (placeholder). Update canceled.', 'danger')
                return redirect(url_for('metrics_view'))
        
        metric.name = name
        metric.description = description
        metric.python_code = python_code
        metric.is_aggregated = 'is_aggregated' in request.form
        metric.accepts_aggregated_inputs = 'accepts_aggregated_inputs' in request.form
        
        db.session.commit()
        flash(f'Metric "{metric.name}" updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating metric: {e}', 'danger')
    
    return redirect(url_for('metrics_view'))

@app.route('/projects/<int:project_id>/clone', methods=['POST'])
def clone_project(project_id):
    import sys
    print(f"DEBUG: Entering clone_project for ID {project_id}", file=sys.stderr)
    original_project = Project.query.get_or_404(project_id)
    new_name = request.form.get('name')
    
    if not new_name:
        flash('Project name is required.', 'danger')
        return redirect(url_for('list_projects'))
        
    if Project.query.filter_by(name=new_name).first():
        flash(f'Project "{new_name}" already exists.', 'danger')
        return redirect(url_for('list_projects'))
        
    created_paths = []
    try:
        # 1. Create New Project
        new_project = Project(name=new_name, description=f"Clone of {original_project.name}")
        db.session.add(new_project)
        db.session.flush() # Get ID
        
        # 2. Clone Leaderboards (Link to SAME Datasets)
        # Note: Datasets are global now. We just clone the LB metadata and submissions.
        
        for old_lb in original_project.leaderboards:
            new_lb = Leaderboard(
                name=old_lb.name,
                project_id=new_project.id, # Belongs to New Project
                dataset_id=old_lb.dataset_id, # Link to SAME Dataset
                summary_metrics=old_lb.summary_metrics,
                comparison_display_columns=old_lb.comparison_display_columns,
                visualizations=old_lb.visualizations,
                selected_metrics=old_lb.selected_metrics,
                metric_directions=old_lb.metric_directions,
                metric_aggregation=old_lb.metric_aggregation,
                scalar_width=old_lb.scalar_width,
                image_width=old_lb.image_width,
                last_sample_filter=old_lb.last_sample_filter
            )
            db.session.add(new_lb)
            db.session.flush()
            
            # Clone Leaderboard Metrics
            for old_lm in old_lb.leaderboard_metrics:
                new_lm = LeaderboardMetric(
                    leaderboard_id=new_lb.id,
                    global_metric_id=old_lm.global_metric_id,
                    arg_mappings=old_lm.arg_mappings,
                    target_name=old_lm.target_name,
                    pooling_type=old_lm.pooling_type,
                    pooling_percentile=old_lm.pooling_percentile,
                    sort_direction=old_lm.sort_direction
                )
                db.session.add(new_lm)
                
            # Clone Leaderboard Visualizations
            for old_lv in old_lb.leaderboard_visualizations:
                new_lv = LeaderboardVisualization(
                    leaderboard_id=new_lb.id,
                    global_visualization_id=old_lv.global_visualization_id,
                    arg_mappings=old_lv.arg_mappings,
                    target_name=old_lv.target_name,
                    display_order=old_lv.display_order
                )
                db.session.add(new_lv)
            
            # Clone Submissions
            for old_sub in old_lb.submissions:
                new_sub = Submission(
                    name=old_sub.name,
                    leaderboard_id=new_lb.id,
                    git_commit=old_sub.git_commit,
                    git_branch=old_sub.git_branch,
                    git_message=old_sub.git_message,
                    is_archived=old_sub.is_archived,
                    processing_status=old_sub.processing_status, 
                    last_sample_filter=old_sub.last_sample_filter
                )
                db.session.add(new_sub)
                db.session.flush() # Get new_sub.id
                
                # Copy Submission Folder
                old_sub_path = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(old_sub.id))
                new_sub_path = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(new_sub.id))
                
                if os.path.exists(new_sub_path):
                     shutil.rmtree(new_sub_path, ignore_errors=True)

                if os.path.exists(old_sub_path):
                    shutil.copytree(old_sub_path, new_sub_path, dirs_exist_ok=True)
                    created_paths.append(new_sub_path)
                    
                for tag in old_sub.tags:
                    new_sub.tags.append(tag)
                    
                for cf in old_sub.custom_fields:
                    # Reuse same sample_id since we share the dataset
                    new_cf = CustomField(
                        name=cf.name,
                        field_type=cf.field_type,
                        value_text=cf.value_text,
                        value_float=cf.value_float,
                        submission_id=new_sub.id,
                        sample_id=cf.sample_id, 
                        sample_name=cf.sample_name
                    )
                    db.session.add(new_cf)
                    
        db.session.commit()
        print(f"CLONE SUCCESS: Project cloned as {new_name}", file=sys.stderr)
        flash(f'Project cloned successfully as "{new_name}"', 'success')
        return redirect(url_for('list_projects'))
        
    except Exception as e:
        print(f"CLONE ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        db.session.rollback()
        # Clean up created paths
        for path in created_paths:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                
        flash(f'Error cloning project: {e}', 'danger')
        return redirect(url_for('list_projects'))

@app.route('/projects/<int:project_id>/delete', methods=['POST'])
def delete_project(project_id):
    project = Project.query.get_or_404(project_id)
    import sys
    print(f"DEBUG: Deleting project {project_id}", file=sys.stderr)
    try:
        project = Project.query.get_or_404(project_id)
        
        # Delete project folder if exists? logic...
        
        db.session.delete(project)
        db.session.commit()
        print(f"DEBUG: Delete Success for ID {project_id}", file=sys.stderr)
        
        flash(f'Project "{project.name}" deleted successfully.', 'success')
        
        response = make_response(redirect(url_for('list_projects')))
        # Check cookie manually or via g
        active_id = request.cookies.get('active_project_id')
        if active_id and str(active_id) == str(project_id):
            response.set_cookie('active_project_id', '', expires=0)
            
        return response
        
    except Exception as e:
        print(f"DEBUG: Delete Error {e}", file=sys.stderr)
        db.session.rollback()
        # Check integrity error (should be handled by cascade, but just in case)
        flash(f'Error deleting project: {str(e)}', 'danger')
        return redirect(url_for('list_projects'))

@app.route('/<project_name>/metrics/<int:metric_id>/delete', methods=['POST'])
def delete_global_metric(project_name, metric_id):
    metric = GlobalMetric.query.get_or_404(metric_id)
    try:
        db.session.delete(metric)
        db.session.commit()
        flash(f'Metric "{metric.name}" deleted.', 'success')
    except Exception as e:
         db.session.rollback()
         flash(f'Error deleting metric: {e}. It might be used by a leaderboard.', 'danger')
         
    return redirect(url_for('metrics_view'))

@app.route('/<project_name>/metrics/<int:metric_id>/download')
def download_metric(project_name, metric_id):
    """Download metric code as a .txt file"""
    metric = GlobalMetric.query.get_or_404(metric_id)
    
    # Create response with metric code
    response = make_response(metric.python_code)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = f'attachment; filename={metric.name}.txt'
    
    return response

# ==================== Visualization Management Routes ====================

@app.route('/<project_name>/visualizations')
def visualizations_view(project_name):
    """List all visualizations for the project."""
    visualizations = GlobalVisualization.query.order_by(GlobalVisualization.name).all()
    return render_template('visualizations.html', visualizations=visualizations, project_name=project_name)

@app.route('/<project_name>/create_visualization', methods=['POST'])
def create_visualization(project_name):
    """Create a new visualization."""
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = handle_dlp_safe_code(request.form.get('python_code', '').strip())
        
        # Handle file upload if present
        viz_file = request.files.get('visualization_file')
        file_code = extract_code_from_file(viz_file)
        
        if file_code:
            python_code = file_code
        elif viz_file and viz_file.filename:
            flash(f'Found file {viz_file.filename} but could not extract Python code from it.', 'warning')
            if "Implementation will be loaded from ZIP" in python_code:
                return redirect(url_for('visualizations_view', project_name=project_name))

        if not python_code or not python_code.strip() or "Implementation will be loaded from ZIP" in python_code:
            flash('Visualization code is required.', 'danger')
            return redirect(url_for('visualizations_view', project_name=project_name))
        
        # Create new visualization
        new_viz = GlobalVisualization(
            name=name,
            description=description,
            python_code=python_code,
            is_aggregated='is_aggregated' in request.form,
            accepts_aggregated_inputs='accepts_aggregated_inputs' in request.form
        )
        
        db.session.add(new_viz)
        db.session.commit()
        flash(f'Visualization "{new_viz.name}" created successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating visualization: {e}', 'danger')
    
    return redirect(url_for('visualizations_view', project_name=project_name))

@app.route('/<project_name>/visualizations/<int:viz_id>/edit', methods=['POST'])
def edit_visualization(project_name, viz_id):
    """Edit an existing visualization."""
    viz = GlobalVisualization.query.get_or_404(viz_id)
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = handle_dlp_safe_code(request.form.get('python_code', '').strip())
        
        # Handle file upload if present
        viz_file = request.files.get('visualization_file')
        file_code = extract_code_from_file(viz_file)
        
        if file_code:
            python_code = file_code
        elif viz_file and viz_file.filename:
            flash(f'Found file {viz_file.filename} but could not extract Python code from it.', 'warning')
            if "Implementation will be loaded from ZIP" in python_code:
                return redirect(url_for('visualizations_view', project_name=project_name))

        if not python_code or not python_code.strip() or "Implementation will be loaded from ZIP" in python_code:
            if "Implementation will be loaded from ZIP" in python_code:
                flash('Invalid code submitted (placeholder). Update canceled.', 'danger')
                return redirect(url_for('visualizations_view', project_name=project_name))
        
        viz.name = name
        viz.description = description
        viz.python_code = python_code
        viz.is_aggregated = 'is_aggregated' in request.form
        viz.accepts_aggregated_inputs = 'accepts_aggregated_inputs' in request.form
        
        db.session.commit()
        flash(f'Visualization "{viz.name}" updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating visualization: {e}', 'danger')
    
    return redirect(url_for('visualizations_view', project_name=project_name))

@app.route('/<project_name>/visualizations/<int:viz_id>/delete', methods=['POST'])
def delete_visualization(project_name, viz_id):
    """Delete a visualization."""
    viz = GlobalVisualization.query.get_or_404(viz_id)
    try:
        db.session.delete(viz)
        db.session.commit()
        flash(f'Visualization "{viz.name}" deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting visualization: {e}. It might be used by a leaderboard.', 'danger')
        
    return redirect(url_for('visualizations_view', project_name=project_name))

@app.route('/<project_name>/visualizations/<int:viz_id>/download')
def download_visualization(project_name, viz_id):
    """Download visualization code as a .txt file."""
    viz = GlobalVisualization.query.get_or_404(viz_id)
    
    # Create response with visualization code
    response = make_response(viz.python_code)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = f'attachment; filename={viz.name}.txt'
    
    return response

@app.route('/<project_name>/metrics/upload', methods=['POST'])
def upload_metric(project_name):
    """Upload/update metric from a .txt file"""
    try:
        metric_file = request.files.get('metric_file')
        metric_name = request.form.get('metric_name', '').strip()
        description = request.form.get('description', '').strip()
        is_aggregated = 'is_aggregated' in request.form
        accepts_aggregated_inputs = 'accepts_aggregated_inputs' in request.form
        
        if not metric_file:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('metrics_view'))
        
        # Read Python code from file
        python_code = metric_file.read().decode('utf-8')
        
        # Basic validation
        if not python_code.strip():
            flash('Uploaded file is empty.', 'danger')
            return redirect(url_for('metrics_view'))
        
        # Try to compile to check syntax
        try:
            compile(python_code, '<string>', 'exec')
        except SyntaxError as e:
            flash(f'Python syntax error in uploaded file: {e}', 'danger')
            return redirect(url_for('metrics_view'))
        
        # Check if metric with this name exists
        existing_metric = GlobalMetric.query.filter_by(name=metric_name).first()
        
        if existing_metric:
            # Update existing metric
            existing_metric.python_code = python_code
            existing_metric.description = description or existing_metric.description
            existing_metric.is_aggregated = is_aggregated
            existing_metric.accepts_aggregated_inputs = accepts_aggregated_inputs
            db.session.commit()
            flash(f'Metric "{metric_name}" updated successfully.', 'success')
        else:
            # Create new metric
            if not metric_name:
                flash('Metric name is required for new metrics.', 'danger')
                return redirect(url_for('metrics_view'))
            
            new_metric = GlobalMetric(
                name=metric_name,
                description=description,
                python_code=python_code,
                is_aggregated=is_aggregated,
                accepts_aggregated_inputs=accepts_aggregated_inputs
            )
            db.session.add(new_metric)
            db.session.commit()
            flash(f'Metric "{metric_name}" created successfully.', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error uploading metric: {e}', 'danger')
    
    return redirect(url_for('metrics_view'))

@app.route('/<project_name>/submission/<int:submission_id>/recalculate', methods=['POST'])
def recalculate_submission(project_name, submission_id):
    submission = Submission.query.get_or_404(submission_id)
    
    # Set to processing state immediately for UI feedback
    submission.processing_status = 'Pending'
    db.session.commit()
    
    # Extract Sample Filters from Request
    sample_filters = {
        'search': request.form.get('sample_search_query', ''),
        'include': {'enabled': request.form.get('enable_sample_include') == 'true', 'tags': sorted([t.strip() for t in request.form.get('sample_include_tags', '').split(',') if t.strip()])},
        'exclude': {'enabled': request.form.get('enable_sample_exclude') == 'true', 'tags': sorted([t.strip() for t in request.form.get('sample_exclude_tags', '').split(',') if t.strip()])},
        'prefix': {'enabled': request.form.get('enable_sample_prefix') == 'true', 'tags': sorted([t.strip() for t in request.form.get('sample_prefix_tags', '').split(',') if t.strip()])}
    }

    # Trigger the task
    tasks.process_submission.delay(submission.id, sample_filters=sample_filters)
    
    flash(f'Recalculation started for submission "{submission.name}".', 'info')
    
    # Preserve filter parameters in redirect
    redirect_args = {
        'leaderboard_id': submission.leaderboard.id,
        'search_query': request.form.get('search_query', ''),
        'show_archived': request.form.get('show_archived', 'false'),
        'enable_include': request.form.get('enable_include', 'false'),
        'enable_exclude': request.form.get('enable_exclude', 'false'),
        'enable_prefix': request.form.get('enable_prefix', 'false'),
        'include_tags': request.form.get('include_tags', ''),
        'exclude_tags': request.form.get('exclude_tags', ''),
        'prefix_tags': request.form.get('prefix_tags', ''),
        'sort_metric': request.form.get('sort_metric', ''),
        'sort_order': request.form.get('sort_order', 'asc'),
        'sample_search_query': request.form.get('sample_search_query', ''),
        'enable_sample_include': request.form.get('enable_sample_include', 'false'),
        'enable_sample_exclude': request.form.get('enable_sample_exclude', 'false'),
        'enable_sample_prefix': request.form.get('enable_sample_prefix', 'false'),
        'sample_include_tags': request.form.get('sample_include_tags', ''),
        'sample_exclude_tags': request.form.get('sample_exclude_tags', ''),
        'sample_prefix_tags': request.form.get('sample_prefix_tags', '')
    }
    
    # Redirect back to the leaderboard
    return redirect(url_for('leaderboard_view', **redirect_args))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>')
def leaderboard_view(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    show_archived = request.args.get('show_archived', 'false').lower() == 'true'
    sort_metric = request.args.get('sort_metric', '')
    sort_order = request.args.get('sort_order', 'asc')
    
    # Submission Filter Params
    search_query = request.args.get('search_query', '')
    enable_include = request.args.get('enable_include', 'false') == 'true'
    enable_exclude = request.args.get('enable_exclude', 'false') == 'true'
    enable_prefix = request.args.get('enable_prefix', 'false') == 'true'
    include_tags = [t.strip() for t in request.args.get('include_tags', '').split(',') if t.strip()]
    exclude_tags = [t.strip() for t in request.args.get('exclude_tags', '').split(',') if t.strip()]
    prefix_tags = [t.strip() for t in request.args.get('prefix_tags', '').split(',') if t.strip()]

    # New Sample Filter Params
    sample_search_query = request.args.get('sample_search_query', '')
    enable_sample_include = request.args.get('enable_sample_include', 'false') == 'true'
    enable_sample_exclude = request.args.get('enable_sample_exclude', 'false') == 'true'
    enable_sample_prefix = request.args.get('enable_sample_prefix', 'false') == 'true'

    sample_include_tags = [t.strip() for t in request.args.get('sample_include_tags', '').split(',') if t.strip()]
    sample_exclude_tags = [t.strip() for t in request.args.get('sample_exclude_tags', '').split(',') if t.strip()]
    sample_prefix_tags = [t.strip() for t in request.args.get('sample_prefix_tags', '').split(',') if t.strip()]

    # Construct current sample filter object for comparison
    current_sample_filters = {
        'search': sample_search_query,
        'include': {'enabled': enable_sample_include, 'tags': sorted(sample_include_tags)},
        'exclude': {'enabled': enable_sample_exclude, 'tags': sorted(sample_exclude_tags)},
        'prefix': {'enabled': enable_sample_prefix, 'tags': sorted(sample_prefix_tags)}
    }
    current_filters_json = json.dumps(current_sample_filters, sort_keys=True)

    query = Submission.query.filter_by(leaderboard_id=leaderboard.id)
    if not show_archived:
        query = query.filter_by(is_archived=False)
    
    # 1. Search Filter (by name)
    if search_query:
        query = query.filter(Submission.name.ilike(f'%{search_query}%'))

    # 2. Tag Filters
    # Include (AND): Submission must have ALL these tags
    if enable_include and include_tags:
        for tag_name in include_tags:
            query = query.filter(Submission.tags.any(Tag.name == tag_name))
            
    # Exclude (OR): Submission must have NONE of these tags
    if enable_exclude and exclude_tags:
        query = query.filter(~Submission.tags.any(Tag.name.in_(exclude_tags)))
        
    # Prefix (AND): Submission must have AT LEAST ONE tag starting with EACH prefix
    if enable_prefix and prefix_tags:
        for prefix in prefix_tags:
            query = query.filter(Submission.tags.any(Tag.name.ilike(f'{prefix}%')))

    submissions = query.order_by(Submission.upload_date.desc()).all()
    
    all_tags = Tag.query.join(Tag.submissions).filter(Submission.leaderboard_id == leaderboard.id).distinct().all()
    # Also get all sample tags for autocomplete
    all_sample_tag_names, all_sample_prefixes = get_all_sample_tags(leaderboard.dataset_id)

    processed_submissions = [s for s in submissions if s.processing_status == 'Processed']
    selected_metrics = [m for m in leaderboard.summary_metrics.split(',') if m.strip()]
    
    # Get all custom metrics from submissions
    custom_metrics = set()
    for sub in processed_submissions:
        for cf in sub.custom_fields:
            if cf.field_type == 'metric':
                custom_metrics.add(cf.name)
    
    # Get all dynamic metrics (linked global metrics) - mapping target_name to lm
    leaderboard_metrics_map = { (lm.target_name if lm.target_name else lm.global_metric.name): lm for lm in leaderboard.leaderboard_metrics }
    
    # Add custom and dynamic metrics to selected_metrics for display
    # Use target_name for dynamic metrics to ensure uniqueness
    # Fix: Deduplicate while preserving selected_metrics order
    discovered_metrics = set(custom_metrics) | set(leaderboard_metrics_map.keys())
    all_metrics = list(selected_metrics)
    for m in sorted(list(discovered_metrics)):
        if m not in all_metrics:
            all_metrics.append(m)
    
    metrics_ranges = {}
    calculated_dynamic_values = {} # sub_id -> metric_name -> value
    calculated_dynamic_values = {} # sub_id -> metric_name -> value
    
    leaderboard_metrics_names = set()
    if leaderboard.leaderboard_metrics:
        leaderboard_metrics_names = {lm.global_metric.name for lm in leaderboard.leaderboard_metrics}

    if processed_submissions:
        # Fetch calculated results from DB
        sub_ids = [s.id for s in processed_submissions]
        results = MetricResult.query.filter(MetricResult.submission_id.in_(sub_ids)).all()
        
        for res in results:
            if res.submission_id not in calculated_dynamic_values:
                calculated_dynamic_values[res.submission_id] = {}
            
            # Use the target_name (unique)
            metric_name = res.leaderboard_metric.target_name if res.leaderboard_metric.target_name else res.leaderboard_metric.global_metric.name
            
            if res.error_message:
                calculated_dynamic_values[res.submission_id][metric_name] = str(res.error_message) 
            else:
                calculated_dynamic_values[res.submission_id][metric_name] = res.value


        for metric in all_metrics:
            if metric in AVAILABLE_METRICS:
                # Standard metric
                values = [getattr(s, metric) for s in processed_submissions if getattr(s, metric) is not None]
            elif metric in leaderboard_metrics_map:
                values = [calculated_dynamic_values.get(s.id, {}).get(metric) for s in processed_submissions]
                values = [v for v in values if isinstance(v, (int, float))]
            else:
                # Custom metric - aggregate from custom_fields with filters
                values = []
                for sub in processed_submissions:
                    if sub.id not in calculated_dynamic_values:
                        calculated_dynamic_values[sub.id] = {}
                    
                    # Group by metric name and execute (Same logic as status API)
                    # Note: After CustomField fix, submission fields don't have sample_id set
                    # So we match by sample_name instead
                    query = db.session.query(
                        func.avg(CustomField.value_float)
                    ).filter(
                        CustomField.submission_id == sub.id,
                        CustomField.name == metric,
                        CustomField.field_type == 'metric'
                    )
                    
                    # Apply sample filters by sample_name (not via Sample join)
                    
                    
                    # Apply Normalized filters by getting matching sample names first
                    if current_sample_filters:
                        sample_names_query = db.session.query(Sample.name).filter_by(dataset_id=leaderboard.dataset_id)
                        
                        if current_sample_filters.get('search'):
                            sample_names_query = sample_names_query.filter(Sample.name.ilike(f"%{current_sample_filters['search']}%"))
                        
                        def tag_match_filter(tag):
                            return or_(
                                Sample.tags == tag,
                                Sample.tags.ilike(f'{tag},%'),
                                Sample.tags.ilike(f'%,{tag}'),
                                Sample.tags.ilike(f'%,{tag},%')
                            )

                        include = current_sample_filters.get('include', {})
                        if include.get('enabled') and include.get('tags'):
                            for tag in include['tags']:
                                sample_names_query = sample_names_query.filter(tag_match_filter(tag))
                        
                        exclude = current_sample_filters.get('exclude', {})
                        if exclude.get('enabled') and exclude.get('tags'):
                            exclude_conditions = [tag_match_filter(tag) for tag in exclude['tags']]
                            if exclude_conditions:
                                sample_names_query = sample_names_query.filter(not_(or_(*exclude_conditions)))

                        prefix = current_sample_filters.get('prefix', {})
                        if prefix.get('enabled') and prefix.get('tags'):
                            prefix_conds = []
                            for p in prefix['tags']:
                                prefix_conds.append(or_(
                                    Sample.tags.ilike(f'{p}%'),
                                    Sample.tags.ilike(f'%,{p}%'),
                                    Sample.tags.ilike(f'%, {p}%')
                                ))
                            if prefix_conds:
                                sample_names_query = sample_names_query.filter(or_(*prefix_conds))
                        
                        # Get matching sample names
                        matching_sample_names = [name for (name,) in sample_names_query.all()]
                    if matching_sample_names:
                        query = query.filter(CustomField.sample_name.in_(matching_sample_names))
                    else:
                        query = query.filter(False)

                    # Fetch raw values for aggregation
                    raw_values = [r[0] for r in query.with_entities(CustomField.value_float).all()]
                    
                    avg_val = None
                    if raw_values:
                        # Determine aggregation method
                        agg_config = current_aggregation.get(metric, {}) if 'current_aggregation' in locals() else {}
                        if not agg_config and leaderboard.metric_aggregation:
                             try:
                                 current_aggregation = json.loads(leaderboard.metric_aggregation)
                                 agg_config = current_aggregation.get(metric, {})
                             except:
                                 agg_config = {}
                        
                        pooling_type = agg_config.get('type', 'mean')
                        pooling_percentile = agg_config.get('percentile')
                        
                        try:
                            if pooling_type == 'mean':
                                avg_val = float(np.mean(raw_values))
                            elif pooling_type == 'median':
                                avg_val = float(np.median(raw_values))
                            elif pooling_type == 'percentile' and pooling_percentile is not None:
                                avg_val = float(np.percentile(raw_values, float(pooling_percentile)))
                            else:
                                avg_val = float(np.mean(raw_values))
                        except Exception as e:
                            print(f"Error calculating aggregation for {metric}: {e}")
                            avg_val = None

                    calculated_dynamic_values[sub.id][metric] = avg_val
                    if avg_val is not None:
                        values.append(avg_val)
            
            if values:
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    metrics_ranges[metric] = {'min': min(numeric_values), 'max': max(numeric_values)}
                else:
                    metrics_ranges[metric] = {'min': 0, 'max': 0}
            else:
                metrics_ranges[metric] = {'min': 0, 'max': 0}
    
    # Apply sorting if requested
    if sort_metric and sort_metric in all_metrics:
        def get_metric_value(sub):
            if sort_metric in AVAILABLE_METRICS:
                val = getattr(sub, sort_metric)
                return val if val is not None else float('inf')
            elif sort_metric in all_metrics:
                # Dynamic or Custom metric
                val = calculated_dynamic_values.get(sub.id, {}).get(sort_metric)
                return val if val is not None else float('inf')
            return float('inf')
        
        submissions.sort(key=get_metric_value, reverse=(sort_order == 'desc'))

    # Prepare Aggregation Info for UI
    from sqlalchemy import inspect
    # Reuse loop logic or helper? Helper is better but inline for speed now.
    metric_agg_info = {}
    current_agg_config = json.loads(leaderboard.metric_aggregation) if leaderboard.metric_aggregation else {}
    
    for metric in all_metrics:
        label = None
        # Check dynamic
        lm = next((m for m in leaderboard.leaderboard_metrics if (m.target_name == metric or m.global_metric.name == metric)), None)
        if lm:
            if lm.pooling_type == 'percentile':
                label = f"{lm.pooling_percentile}% Percentile" if lm.pooling_percentile else "Percentile"
            else:
                # Default to Mean if None, or capitalize existing
                t = lm.pooling_type if lm.pooling_type else 'mean'
                label = t.capitalize()
        else:
            # Check custom
            conf = current_agg_config.get(metric, {})
            t = conf.get('type', 'mean')
            if t == 'percentile':
                 p = conf.get('percentile')
                 label = f"{p}% Percentile" if p else "Percentile"
            else:
                label = t.capitalize()
        
        if label:
            metric_agg_info[metric] = label

    # Merge existing config with LeaderboardMetric directions for template
    metric_directions_dict = json.loads(leaderboard.metric_directions) if leaderboard.metric_directions else {}
    if leaderboard.leaderboard_metrics:
        for lm in leaderboard.leaderboard_metrics:
            target = lm.target_name if lm.target_name else lm.global_metric.name
            
            if lm.sort_direction:
                metric_directions_dict[target] = lm.sort_direction

    return render_template('leaderboard.html', 
                           project_name=project_name, 
                           leaderboard=leaderboard,
                           submissions=submissions,
                           all_metrics=all_metrics,
                           selected_metrics=all_metrics,
                           metrics_ranges=metrics_ranges,
                           dynamic_values=calculated_dynamic_values,
                           sort_metric=sort_metric,
                           sort_order=sort_order,
                           metric_agg_info=metric_agg_info,
                           show_archived=show_archived,
                           all_tags=all_tags,
                           all_sample_tag_names=all_sample_tag_names,
                           all_sample_prefixes=all_sample_prefixes,
                           search_query=search_query,
                           sample_search_query=sample_search_query,
                           enable_include=enable_include,
                           enable_exclude=enable_exclude,
                           enable_prefix=enable_prefix,
                           include_tags=include_tags,
                           exclude_tags=exclude_tags,
                           prefix_tags=prefix_tags,
                           enable_sample_include=enable_sample_include,
                           enable_sample_exclude=enable_sample_exclude,
                           enable_sample_prefix=enable_sample_prefix,
                           sample_include_tags=sample_include_tags,
                           sample_exclude_tags=sample_exclude_tags,
                           sample_prefix_tags=sample_prefix_tags,
                           metric_directions=metric_directions_dict)

@app.route('/<project_name>/upload_dataset', methods=['POST'])
def upload_dataset(project_name):
    dataset_names_input = request.form.get('dataset_name', '')
    files = request.files.getlist('dataset_zip')
    override = request.form.get('override_dataset') == 'true'

    if not files:
        flash("No files uploaded.", "warning")
        return redirect(url_for('index'))

    # If names are provided, split them by comma. Otherwise, auto-generate from filenames.
    if dataset_names_input:
        provided_names = [name.strip() for name in dataset_names_input.split(',') if name.strip()]
    else:
        provided_names = [file.filename.replace('.zip', '') for file in files]

    if len(provided_names) != len(files):
        flash("Number of names provided does not match number of files uploaded.", "danger")
        return redirect(url_for('index'))

    project_id = g.current_project.id if g.get('current_project') else None

    for i, file in enumerate(files):
        dataset_name = provided_names[i]
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_dataset_pre_process')
        os.makedirs(temp_dir, exist_ok=True)
        temp_zip_path = os.path.join(temp_dir, filename)
        file.save(temp_zip_path)

        success, message, ds_id = process_dataset_zip(temp_zip_path, dataset_name, override=override)

        if success:
            flash(message, "success")
        else:
            flash(message, "danger")

        # Cleanup temp zip
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    return redirect(url_for('datasets_list'))

@app.route('/<project_name>/create_leaderboard', methods=['POST'])
def create_leaderboard(project_name):
    project = Project.query.filter_by(name=project_name).first_or_404()
    leaderboard_name = request.form['leaderboard_name']
    
    # Check if leaderboard with this name already exists in this project
    existing = Leaderboard.query.filter_by(name=leaderboard_name, project_id=project.id).first()
    if existing:
        if request.form.get('overwrite'):
            db.session.delete(existing)
            db.session.commit()
            flash(f'Overwriting existing leaderboard "{leaderboard_name}".', 'warning')
        else:
            flash(f'Leaderboard "{leaderboard_name}" already exists in this project. Please choose a different name or check "Overwrite".', 'danger')
            return redirect(url_for('index', project_name=project_name))
    
    new_leaderboard = Leaderboard(
        name=leaderboard_name, 
        dataset_id=request.form['dataset_id'],
        project_id=project.id,
        summary_metrics=','.join(request.form.getlist('summary_metrics'))
    )
    db.session.add(new_leaderboard)
    db.session.commit() # Commit to get ID
    
    # Pre-populate Standard Aggregated Metrics
    aggregated_metrics = []
    
    for m in aggregated_metrics:
        # Check if GlobalMetric exists, else create it
        gm = GlobalMetric.query.filter_by(name=m['name']).first()
        if not gm:
            gm = GlobalMetric(
                name=m['name'],
                description=f"Default aggregated metric: {m['name']}",
                python_code=m['code'],
                is_aggregated=True
            )
            db.session.add(gm)
            db.session.flush() # Get ID
        
        # Link to leaderboard via LeaderboardMetric
        lm = LeaderboardMetric(
            leaderboard_id=new_leaderboard.id,
            global_metric_id=gm.id,
            arg_mappings=json.dumps(m['mappings'])
        )
        db.session.add(lm)
    db.session.commit()
    
    db.session.commit()
    
    flash(f'Leaderboard "{leaderboard_name}" created successfully!', 'success')
    return redirect(url_for('index', project_name=project_name))

def process_submission_zip(leaderboard_id, submission_name, zip_path):
    """
    Helper function to process a single submission zip file.
    Create DB entry, extract files, and queue processing task.
    """
    try:
        new_submission = Submission(name=submission_name, leaderboard_id=leaderboard_id, processing_status='Queued')
        db.session.add(new_submission)
        db.session.flush() # Get ID for folder name

        submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(new_submission.id))
        
        # --- Ensure submission_folder is clean before extraction ---
        if os.path.exists(submission_folder):
            shutil.rmtree(submission_folder)
        os.makedirs(submission_folder, exist_ok=True) # Recreate the empty folder
        
        # Save the original ZIP file for download
        dest_zip_path = os.path.join(submission_folder, 'submission.zip')
        shutil.copy2(zip_path, dest_zip_path)
        
        # Unzip contents, handling potential nested root folder
        temp_extract_dir = os.path.join(submission_folder, '_temp_extract')
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_extract_dir)
            
        # Determine if there's a single top-level folder (ignoring __MACOSX)
        extracted_items = [item for item in os.listdir(temp_extract_dir) if item != '__MACOSX' and not item.startswith('.')]
        
        source_dir = temp_extract_dir
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract_dir, extracted_items[0])):
             # It's a nested folder, use it as source
             folder_name = extracted_items[0]
             source_dir = os.path.join(temp_extract_dir, folder_name)
             
             # UPDATE SUBMISSION NAME to match folder name
             new_submission.name = folder_name
             db.session.add(new_submission)
             db.session.flush()
        
        # Move all contents from source_dir to submission_folder
        for item in os.listdir(source_dir):
            if item == 'submission.zip' or item == '_temp_extract': continue
            s = os.path.join(source_dir, item)
            d = os.path.join(submission_folder, item)
            if os.path.isdir(s):
                if os.path.exists(d): shutil.rmtree(d)
                shutil.move(s, d)
            else:
                shutil.move(s, d)
                
        # Cleanup temp
        shutil.rmtree(temp_extract_dir)
            
        print(f"Extracted submission to: {submission_folder}")
        # --- End clean up and extraction ---

        # Read git.info from its final location
        git_info_path = os.path.join(submission_folder, 'git.info')
        if os.path.exists(git_info_path):
            with open(git_info_path, 'r') as git_file:
                git_data = json.load(git_file)
                # Map various possible keys to the model fields
                new_submission.git_commit = git_data.get('commit') or git_data.get('commit_sha')
                new_submission.git_branch = git_data.get('branch') or git_data.get('repo_url') # Fallback repo_url to branch for visibility if branch missing
                new_submission.git_message = git_data.get('message')
        
        # Parse tags from tags.txt or tags/ folder
        tags_to_add = set()
        
        # 1. tags.txt
        tags_txt_path = os.path.join(submission_folder, 'tags.txt')
        if os.path.exists(tags_txt_path):
            try:
                with open(tags_txt_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        for tag in content.split(','):
                            clean_tag = tag.strip()
                            if clean_tag:
                                tags_to_add.add(clean_tag)
            except Exception as e:
                print(f"Error reading tags.txt: {e}")
                

                
        # Add tags to submission and database
        for tag_name in tags_to_add:
            tag = Tag.query.filter_by(name=tag_name).first()
            if not tag:
                tag = Tag(name=tag_name)
                db.session.add(tag)
            new_submission.tags.append(tag)
        
        # Detect and store custom fields (including metrics)
        # We explicitly EXCLUDE nothing now. Generic detection logic prevails. 
        # so that if they contain images/scalars they are picked up, or if they are empty/irrelevant they are ignored.
        # But wait, checking logic: detect_custom_fields iterates folders NOT in known_folders.
        # Previously they WERE in known_folders, meaning they were SKIPPED by detect_custom_fields.
        # The user said "stop expecting specific fields... anymore".
        # If I remove them from known_folders, they will be SCANNED.
        # If they contain .npz files (histograms), they will be ignored by detect_custom_fields (which looks for images/txt).
        # This is correct.
        
        leaderboard = Leaderboard.query.get(leaderboard_id)
        dataset_samples = Sample.query.filter_by(dataset_id=leaderboard.dataset_id).all()
        sample_names = [s.name for s in dataset_samples]
        
        # Debug: Log submission structure
        print(f"\n=== CUSTOM FIELD DETECTION DEBUG ===")
        print(f"Submission folder: {submission_folder}")
        print(f"Submission folder contents:")
        try:
            for item in os.listdir(submission_folder):
                item_path = os.path.join(submission_folder, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    print(f"  {item}/ ({len(files)} files)")
                else:
                    print(f"  {item}")
        except Exception as e:
            print(f"  Error listing contents: {e}")
        
        print(f"Dataset sample names: {sample_names}")
        
        # Only exclude internal metadata folders we definitively don't want as custom fields
        known_folders = {'git.info', '__MACOSX'} 
        
        print(f"Known folders (excluded from custom fields): {known_folders}")
        
        custom_fields = detect_custom_fields(submission_folder, sample_names, known_folders, is_submission=True)
        
        print(f"Detected custom fields: {list(custom_fields.keys())}")
        for field_name, field_info in custom_fields.items():
            print(f"  {field_name}: type={field_info['type']}, samples={list(field_info['data'].keys())}")
        print(f"=== END DEBUG ===\n")
        
        # Create sample name to ID map
        sample_map = {s.name: s.id for s in dataset_samples}

        # Store custom fields in database
        for field_name, field_info in custom_fields.items():
            field_type = field_info['type']
            for s_name, value in field_info['data'].items():
                
                # Resolve sample_id
                s_id = sample_map.get(s_name)
                
                if field_type == 'image':
                    # Create permanent folder for submission images
                    submission_images_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(new_submission.id), 'images', field_name)
                    os.makedirs(submission_images_dir, exist_ok=True)

                    filename = os.path.basename(value)
                    dest_path = os.path.join(submission_images_dir, filename)
                    shutil.copy2(value, dest_path)
                    
                    # Store relative path from uploads folder
                    rel_path = os.path.relpath(dest_path, app.config['UPLOAD_FOLDER'])
                    custom_field = CustomField(
                        name=field_name,
                        field_type='image',
                        value_text=rel_path,
                        submission_id=new_submission.id,
                        sample_name=s_name
                    )
                elif field_type == 'histogram':
                    # Store path in value_text for histograms
                    custom_field = CustomField(
                        name=field_name,
                        field_type='histogram',
                        value_text=value, 
                        submission_id=new_submission.id,
                        sample_name=s_name
                    )

                elif field_type == 'depth':
                     # Store processed depth map .npz
                    submission_depth_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(new_submission.id), 'depth_maps', field_name)
                    os.makedirs(submission_depth_dir, exist_ok=True)

                    filename = os.path.basename(value)
                    dest_path = os.path.join(submission_depth_dir, filename)
                    shutil.copy2(value, dest_path)
                    
                    rel_path = os.path.relpath(dest_path, app.config['UPLOAD_FOLDER'])
                    custom_field = CustomField(
                        name=field_name,
                        field_type='depth',
                        value_text=rel_path,
                        submission_id=new_submission.id,
                        sample_name=s_name
                    )
                elif field_type == 'json':
                    # Store JSON files preserving original folder structure
                    submission_json_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(new_submission.id), field_name)
                    os.makedirs(submission_json_dir, exist_ok=True)

                    filename = os.path.basename(value)
                    dest_path = os.path.join(submission_json_dir, filename)
                    shutil.copy2(value, dest_path)
                    
                    rel_path = os.path.relpath(dest_path, app.config['UPLOAD_FOLDER'])
                    custom_field = CustomField(
                        name=field_name,
                        field_type='json',
                        value_text=rel_path,
                        submission_id=new_submission.id,
                        sample_name=s_name
                    )
                else:  # scalar or metric
                    custom_field = CustomField(
                        name=field_name,
                        field_type=field_type,
                        value_float=value,
                        submission_id=new_submission.id,
                        sample_name=s_name
                    )
                db.session.add(custom_field)
        
        db.session.commit()

        # Send task to Celery
        tasks.process_submission.delay(new_submission.id)
        return True, None

    except Exception as e:
        db.session.rollback()
        print(f"Error processing submission {submission_name}: {e}")
        return False, str(e)

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/upload_submission', methods=['POST'])
def upload_submission(project_name, leaderboard_id):
    files = request.files.getlist('submission_zip')
    submission_names_input = request.form.get('submission_name')

    if not files:
        return redirect(url_for('leaderboard_view', leaderboard_id=leaderboard_id))

    # Handle explicitly provided names only if number matches
    provided_names = []
    if submission_names_input:
        provided_names = [name.strip() for name in submission_names_input.split(',') if name.strip()]
    
    
    # We will prioritize checking if it's a BULK upload first.
    # If it is bulk, we ignore the provided name (which likely auto-filled as the zip name).
    # If not bulk, we use the provided name.

    for i, file in enumerate(files):
        # Save temporary
        temp_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload_zip', secure_filename(file.filename))
        os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
        file.save(temp_zip_path)
        
        is_bulk = False
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zf:
                file_list = zf.namelist()
                # Check for predictions file to confirm it's a SINGLE submission
                has_predictions = any(f.endswith('predictions.csv') or f.endswith('predictions.json') for f in file_list)
                
                # Check for nested zips to suspect BATCH
                has_zips = any(f.endswith('.zip') for f in file_list)
                
                if not has_predictions and has_zips:
                    is_bulk = True
                
                with open("debug_log.txt", "a") as f:
                    f.write(f"DEBUG: file_list={file_list}\\n")
                    f.write(f"DEBUG: has_predictions={has_predictions}, has_zips={has_zips}, is_bulk={is_bulk}\\n")
        except Exception as e:
            with open("debug_log.txt", "a") as f:
                f.write(f"DEBUG: Exception reading zip: {e}\\n")
            pass # Fallback to single

        if is_bulk:
            # Process as Batch
            extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_bulk_extract', secure_filename(file.filename).replace('.zip',''))
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir)

            with zipfile.ZipFile(temp_zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            for root, dirs, filenames in os.walk(extract_dir):
                for filename in filenames:
                    if filename.endswith('.zip') and not filename.startswith('__MACOSX'):
                        inner_zip_path = os.path.join(root, filename)
                        sub_name = filename.replace('.zip', '')
                        process_submission_zip(leaderboard_id, sub_name, inner_zip_path)
            
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)

        else:
            # Process as Single
            # HERE we use the provided name if available and valid
            if provided_names and i < len(provided_names):
                sub_name = provided_names[i]
            else:
                sub_name = file.filename.replace('.zip', '')
                
            process_submission_zip(leaderboard_id, sub_name, temp_zip_path)

        # Cleanup original upload
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    return redirect(url_for('leaderboard_view', leaderboard_id=leaderboard_id))



@app.route('/<project_name>/submissions/batch_action', methods=['POST'])
def batch_action(project_name):
    action = request.form.get('action')
    submission_ids = request.form.getlist('submission_ids')
    leaderboard_id = request.form.get('leaderboard_id')
    if not submission_ids:
        return redirect(url_for('leaderboard_view', leaderboard_id=leaderboard_id))
    if action == 'compare':
        # Removed session based compare_ids setting for shareability
        compare_ids_str = ','.join(submission_ids)
        return redirect(url_for('comparison_view', leaderboard_id=leaderboard_id, compare_ids=compare_ids_str))
    submissions = Submission.query.filter(Submission.id.in_(submission_ids)).all()
    if action == 'archive':
        for sub in submissions: sub.is_archived = True
    elif action == 'unarchive':
        for sub in submissions: sub.is_archived = False
    elif action == 'delete':
        for sub in submissions:
            shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(sub.id)), ignore_errors=True)
            db.session.delete(sub)
    elif action == 'add_tags':
        tag_names = [tag.strip() for tag in request.form.get('tags', '').split(',') if tag.strip()]
        for sub in submissions:
            for tag_name in tag_names:
                tag = Tag.query.filter_by(name=tag_name).first() or Tag(name=tag_name)
                if tag not in sub.tags:
                    sub.tags.append(tag)
    elif action == 'recalculate':
        # Extract Sample Filters from Request
        sample_filters = {
            'search': request.form.get('sample_search_query', ''),
            'include': {'enabled': request.form.get('enable_sample_include') == 'true', 'tags': sorted([t.strip() for t in request.form.get('sample_include_tags', '').split(',') if t.strip()])},
            'exclude': {'enabled': request.form.get('enable_sample_exclude') == 'true', 'tags': sorted([t.strip() for t in request.form.get('sample_exclude_tags', '').split(',') if t.strip()])},
            'prefix': {'enabled': request.form.get('enable_sample_prefix') == 'true', 'tags': sorted([t.strip() for t in request.form.get('sample_prefix_tags', '').split(',') if t.strip()])}
        }
        for sub in submissions:
            sub.processing_status = 'Pending'
            tasks.process_submission.delay(sub.id, sample_filters=sample_filters)
        flash(f'Started recalculation for {len(submissions)} submissions.', 'info')
    db.session.commit()
    
    # Preserve filter parameters in redirect
    redirect_args = {
        'leaderboard_id': leaderboard_id,
        'search_query': request.form.get('search_query', ''),
        'show_archived': request.form.get('show_archived', 'false'),
        'enable_include': request.form.get('enable_include', 'false'),
        'enable_exclude': request.form.get('enable_exclude', 'false'),
        'enable_prefix': request.form.get('enable_prefix', 'false'),
        'include_tags': request.form.get('include_tags', ''),
        'exclude_tags': request.form.get('exclude_tags', ''),
        'prefix_tags': request.form.get('prefix_tags', ''),
        'sort_metric': request.form.get('sort_metric', ''),
        'sort_order': request.form.get('sort_order', 'asc'),
        'sample_search_query': request.form.get('sample_search_query', ''),
        'enable_sample_include': request.form.get('enable_sample_include', 'false'),
        'enable_sample_exclude': request.form.get('enable_sample_exclude', 'false'),
        'enable_sample_prefix': request.form.get('enable_sample_prefix', 'false'),
        'sample_include_tags': request.form.get('sample_include_tags', ''),
        'sample_exclude_tags': request.form.get('sample_exclude_tags', ''),
        'sample_prefix_tags': request.form.get('sample_prefix_tags', '')
    }
    
    return redirect(url_for('leaderboard_view', **redirect_args))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/update_metrics', methods=['POST'])
def update_leaderboard_metrics(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    selected_metrics = request.form.getlist('metrics')
    leaderboard.selected_metrics = ','.join(selected_metrics)
    db.session.commit()
    return redirect(url_for('comparison_view', leaderboard_id=leaderboard_id))

@app.route('/<project_name>/comparison/<int:leaderboard_id>')
def comparison_view(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    
    # Check for compare_ids in query parameters first (for shareable URLs)
    compare_ids_arg = request.args.get('compare_ids')
    if compare_ids_arg:
        compare_ids = compare_ids_arg.split(',')
    else:
        # Fallback to session (backward compatibility or direct nav cases)
        compare_ids = session.get('compare_ids', [])
    
    if compare_ids:
        # Filter if user explicitly selected a subset
        submissions = [s for s in leaderboard.submissions if str(s.id) in compare_ids and not s.is_archived]
    else:
        submissions = [s for s in leaderboard.submissions if not s.is_archived]
    
    if not submissions:
        return render_template('comparison.html', 
                               leaderboard=leaderboard, 
                               submissions=[], 
                               comparison_data=[], 
                               chart_metrics_data="[]", 
                               selected_metrics=[], 
                               paginated_samples=None, 
                               submissions_json=[], 
                               selected_comparison_display_columns=[], 
                               all_comparison_tags=[],
                               all_sample_tag_names=[],
                               all_sample_prefixes=[],
                               all_custom_fields=[],
                               all_field_types={},
                               dataset_custom_fields=set(),
                               submission_custom_fields={},
                               submission_has_histogram={},
                               per_page_options=[5, 10, 20, 100], 
                               current_per_page=request.args.get('per_page', 5, type=int), 
                               search_query=request.args.get('search_query', ''), 
                               comparison_display_options=DATASET_DISPLAY_OPTIONS, 
                               visualization_options=VISUALIZATION_OPTIONS, 
                               active_visualizations=[],
                               visualization_configs={},
                               leaderboard_viz_list=[],
                               sort_by='', 
                               sort_order='asc', 
                               sample_metric_options=SAMPLE_METRIC_OPTIONS, 
                               active_metrics=[],
                               project_name=project_name,
                               current_compare_ids=compare_ids_arg)

    # Pagination params
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 5, type=int)
    search_query = request.args.get('search_query', '')
    sort_by = request.args.get('sort_by', '') # Default to empty (no sort)
    sort_order = request.args.get('sort_order', 'asc')

    # Base query for samples
    samples_query = Sample.query.filter_by(dataset_id=leaderboard.dataset_id)
    
    # Apply search filter
    if search_query:
        samples_query = samples_query.filter(Sample.name.ilike(f'%{search_query}%'))
    
    # Apply tag filters
    samples_query = apply_tag_filters(samples_query, request.args)

    # Collect unique custom field names efficiently
    # Dataset fields
    dataset_custom_fields_query = db.session.query(CustomField.name, CustomField.field_type).join(Sample).filter(
        Sample.dataset_id == leaderboard.dataset_id,
        CustomField.submission_id == None
    ).distinct().all()
    
    dataset_custom_fields = {name for name, ftype in dataset_custom_fields_query if ftype in ['image', 'depth', 'scalar', 'metric']}
    dataset_field_types = {name: ftype for name, ftype in dataset_custom_fields_query}
    
    # Submission fields
    sub_ids = [s.id for s in submissions]
    submission_custom_fields_query = db.session.query(CustomField.submission_id, CustomField.name, CustomField.field_type).filter(
        CustomField.submission_id.in_(sub_ids)
    ).distinct().all()
    
    submission_custom_fields = {}
    submission_field_types = {}
    for sub_id, name, ftype in submission_custom_fields_query:
        if ftype in ['image', 'depth', 'scalar', 'metric']:
            if sub_id not in submission_custom_fields:
                submission_custom_fields[sub_id] = set()
            submission_custom_fields[sub_id].add(name)
            submission_field_types[name] = ftype

    all_submission_fields = set()
    for fields in submission_custom_fields.values():
        all_submission_fields.update(fields)
    all_custom_fields = sorted(list(dataset_custom_fields | all_submission_fields))
    all_field_types = dataset_field_types.copy()
    all_field_types.update(submission_field_types)
    
    # Standard metrics
    for m in ['mse', 'rmse', 'mae', 'mape', 'smape']:
        all_field_types[m] = 'metric'
    
    for lm in leaderboard.leaderboard_metrics:
        all_field_types[lm.global_metric.name] = 'metric'
    
    custom_scalar_metric_names = [name for name, ftype in all_field_types.items() if ftype in ['scalar', 'metric']]

    # Sorting
    if sort_by:
        if sort_by == 'name':
            if sort_order == 'desc':
                samples_query = samples_query.order_by(Sample.name.desc())
            else:
                samples_query = samples_query.order_by(Sample.name.asc())
            total = samples_query.count()
            paginated_items = samples_query.offset((page-1)*per_page).limit(per_page).all()
        elif ':' in sort_by:
            # Sort by submission metric - this is still heavy if not pre-calculated
            # For now, we fetch ALL to sort, but we should optimize this later with MetricResult joins
            all_filtered_samples = samples_query.all()
            parts = sort_by.split(':')
            metric_key, sub_id_str = parts
            sub_id = int(sub_id_str)
            target_sub = Submission.query.get(sub_id)
            if target_sub:
                target_lm = next((lm for lm in leaderboard.leaderboard_metrics if (lm.target_name == metric_key or lm.global_metric.name == metric_key)), None)
                submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(target_sub.id))
                
                # Fetch pre-calculated results for the metric if possible
                precalc_results = {res.sample_name: res.value for res in db.session.query(CustomField.sample_name, CustomField.value_float).filter(
                    CustomField.submission_id == sub_id,
                    CustomField.name == metric_key,
                    CustomField.field_type == 'metric'
                ).all()}

                def get_sort_val(s):
                    if s.name in precalc_results:
                        val = precalc_results[s.name]
                    else:
                        # Fallback to dynamic calculation
                        if target_lm:
                            context = get_metric_context(s, target_sub, submission_folder=submission_folder)
                            val, err = evaluate_dynamic_metric(target_lm.global_metric, context, target_lm.arg_mappings)
                        else:
                            val = None
                    
                    if val is None: return -float('inf') if sort_order == 'desc' else float('inf')
                    return val
                
                all_filtered_samples.sort(key=get_sort_val, reverse=(sort_order == 'desc'))
                # Slice manually since we had to fetch all
                total = len(all_filtered_samples)
                paginated_items = all_filtered_samples[(page-1)*per_page : page*per_page]
            else:
                total = samples_query.count()
                paginated_items = samples_query.offset((page-1)*per_page).limit(per_page).all()
        elif sort_by in custom_scalar_metric_names:
            # Sort by custom scalar/dataset metric
            # Fetch all filtered IDs and values for sorting
            sort_vals = {s_name: v for s_name, v in db.session.query(Sample.name, CustomField.value_float).join(CustomField).filter(
                Sample.dataset_id == leaderboard.dataset_id,
                CustomField.name == sort_by,
                CustomField.submission_id == None
            ).all()}
            
            all_filtered_samples = samples_query.all()
            all_filtered_samples.sort(key=lambda x: sort_vals.get(x.name, 0), reverse=(sort_order == 'desc'))
            total = len(all_filtered_samples)
            paginated_items = all_filtered_samples[(page-1)*per_page : page*per_page]
        else:
             total = samples_query.count()
             paginated_items = samples_query.offset((page-1)*per_page).limit(per_page).all()
    else:
        total = samples_query.count()
        paginated_items = samples_query.offset((page-1)*per_page).limit(per_page).all()

    # Create pagination object
    class SimplePagination:
        def __init__(self, items, page, per_page, total):
            self.items = items
            self.page = page
            self.per_page = per_page
            self.total = total
            self.pages = (total + per_page - 1) // per_page
            self.has_prev = page > 1
            self.has_next = page < self.pages
            self.prev_num = page - 1
            self.next_num = page + 1

    paginated_samples = SimplePagination(paginated_items, page, per_page, total)
    samples_on_page = paginated_samples.items

    # 2-pass calculation setup
    comparison_data, chart_metrics_data = [], []
    active_metrics = list(set([m for m in leaderboard.selected_metrics.split(',') if m.strip()] + 
                             [m for m in leaderboard.summary_metrics.split(',') if m.strip()]))

    # Helper to handle zero values for log scale visualization
    def get_log_safe_value(val):
        if val is None: return None
        return float(val) if val > 0 else 1e-9 # Ensure float conversion and handle zero

    comparison_data, chart_metrics_data = [], []

    selected_comparison_display_columns = [col for col in leaderboard.comparison_display_columns.split(',') if col.strip()]
    
    # Pre-calculate metrics/scalars ONLY for paginated samples to fix NameError
    sample_metrics_map = {}
    for s in samples_on_page:
        sample_metrics_map[s.id] = {'name': s.name}
        for cf in s.custom_fields:
            if cf.field_type in ['scalar', 'metric'] and cf.submission_id is None:
                sample_metrics_map[s.id][cf.name] = cf.value_float

    print(f"DEBUG: Processing {len(samples_on_page)} samples for comparison view.")
    for sample in samples_on_page:
        print(f"DEBUG: Processing sample: {sample.name} (ID: {sample.id})")
        signal_shape = sample.signal_shape.shape_name if sample.signal_shape else 'gaussian'
        gt_bins = json.loads(sample.histogram_data.bins) if sample.histogram_data else []
        gt_counts = json.loads(sample.histogram_data.counts) if sample.histogram_data else []
        config_json = sample.config_data.parsed_config if sample.config_data else {}

        sample_info = {
            'sample_id': sample.id,
            'sample_name': sample.name,
            'dataset_tags': [t.strip() for t in sample.tags.split(',')] if sample.tags else [],
            'ground_truth': {
                'config': sample.config_data.parsed_config if sample.config_data else {},
                'bins': gt_bins,
                'counts': gt_counts,
                'custom_fields': {cf.name: cf.value_float for cf in sample.custom_fields if cf.field_type in ['scalar', 'metric'] and cf.submission_id is None}
            },
            'predictions': [],
            'custom_fields': {},
            'custom_metrics': {}
        }
        
        # Add GT custom fields for this sample
        for cf in sample.custom_fields:
            if cf.field_type in ['image', 'depth', 'scalar']:
                sample_info['custom_fields'][cf.name] = {
                    'gt_field_id': cf.id if cf.field_type in ['image', 'depth'] else None,
                    'gt_scalar_value': cf.value_float if cf.field_type == 'scalar' else None,
                    'submissions': {},
                    'sub_scalars': {}
                }


        sample_chart_metrics_for_this_sample = {
            'sample_name': sample.name,
            'metrics': {}
        }

        all_observed_metrics = set()

        for sub in submissions:
            print(f"DEBUG:   Processing submission: {sub.name} (ID: {sub.id}) for sample {sample.name}")
            gt_pick = sample_metrics_map[sample.id].get('pick', 0)
            pred_data, current_sample_metrics = calculate_submission_metrics(sub, sample, gt_pick, signal_shape, active_metrics)
            
            sample_info['predictions'].append(pred_data)
            
            # Add submission custom fields for this sample
            for cf in sub.custom_fields:
                if cf.field_type in ['image', 'depth', 'scalar'] and cf.sample_name == sample.name:
                    if cf.name not in sample_info['custom_fields']:
                        sample_info['custom_fields'][cf.name] = {
                            'gt_field_id': None,
                            'gt_scalar_value': None,
                            'submissions': {},
                            'sub_scalars': {}
                        }
                    if cf.field_type in ['image', 'depth']:
                        sample_info['custom_fields'][cf.name]['submissions'][sub.id] = cf.id
                    elif cf.field_type == 'scalar':
                        sample_info['custom_fields'][cf.name]['sub_scalars'][sub.id] = cf.value_float
                
                # Add submission custom metric fields for this sample
                if cf.field_type == 'metric' and cf.sample_name == sample.name:
                    if cf.name not in sample_info['custom_metrics']:
                        sample_info['custom_metrics'][cf.name] = {
                            'submissions': {}
                        }
                    sample_info['custom_metrics'][cf.name]['submissions'][sub.id] = cf.value_float

            
            # Include both standard and custom metrics
            log_safe_metrics = {}
            
            # Add custom metrics (any metric not in the standard list)
            for metric_name, metric_value in current_sample_metrics.items():
                 log_safe_metrics[metric_name] = get_log_safe_value(metric_value)
            
            # # Dynamic histogram entropies only. Legacy 'hist_entropy' excluded as requested.
            # for k, v in pred_data.items():
            #     if k.startswith('hist_entropy_'):
            #          log_safe_metrics[k] = get_log_safe_value(v)
            
            # Add Dynamic Metrics
            dynamic_ctx = get_metric_context(sample, sub)
            # Add calculated standard metrics to ctx for dynamic functions
            for km, vm in current_sample_metrics.items():
                 if vm is not None: dynamic_ctx[km] = vm
            
            # Add Dynamic Metrics linked to this leaderboard
            for lm in leaderboard.leaderboard_metrics:
                if lm.global_metric.is_aggregated: continue
                
                # Use GlobalMetric code + LeaderboardMetric mapping
                val, err = evaluate_dynamic_metric(lm.global_metric, dynamic_ctx, lm.arg_mappings)
                if val is not None:
                    # Determine if it's a submission metric or GT metric
                    arg_mappings = {}
                    try: arg_mappings = json.loads(lm.arg_mappings)
                    except: pass
                    is_sub_metric = any(field_name.startswith('sub_') for field_name in arg_mappings.values())
                    
                    log_safe_val = get_log_safe_value(val)
                    if is_sub_metric:
                        log_safe_metrics[lm.global_metric.name] = log_safe_val
                    else:
                        # Purely GT based metric - add to Ground Truth in chart
                        if 'Ground Truth' not in sample_chart_metrics_for_this_sample['metrics']:
                            sample_chart_metrics_for_this_sample['metrics']['Ground Truth'] = {}
                        sample_chart_metrics_for_this_sample['metrics']['Ground Truth'][lm.global_metric.name] = log_safe_val
                    
                    # Also add to comparison table
                    if lm.global_metric.name not in sample_info['custom_metrics']:
                        sample_info['custom_metrics'][lm.global_metric.name] = {'submissions': {}}
                    sample_info['custom_metrics'][lm.global_metric.name]['submissions'][sub.id] = val
                    
                    # Ensure it's in pred_data for the per_sample_values column
                    pred_data[lm.global_metric.name] = val

            sample_chart_metrics_for_this_sample['metrics'][sub.name] = log_safe_metrics
            all_observed_metrics.update(log_safe_metrics.keys())
            print(f"DEBUG:     Log-safe metrics for {sample.name} (sub {sub.id}): {log_safe_metrics}")

        # Add Ground Truth to chart metrics
        # Only add dynamic hist_entropy_* variants if they exist in submissions
        sample_chart_metrics_for_this_sample['metrics']['Ground Truth'] = {}
        
        # # Map to observed dynamic entropy keys (so GT aligns with Submissions)
        # for k in all_observed_metrics:
        #      if k.startswith('hist_entropy_'):
        #          # These are dynamic variants - would need GT values from custom fields
        #          # For now, skip GT values for these
        #          pass

        comparison_data.append(sample_info)
        chart_metrics_data.append(sample_chart_metrics_for_this_sample)
    
    # Collect all unique tags from the comparison data for the filter dropdown
    all_comparison_tags = set()
    # (Removed per-sample submission tags collection)
    
    # Also collect global submission tags
    for sub in submissions:
        all_comparison_tags.update([tag.name for tag in sub.tags])

    # Also get ALL sample tags from the dataset for auto-suggestions
    all_sample_tag_names, all_sample_prefixes = get_all_sample_tags(leaderboard.dataset_id)

    # Collect all custom metrics from submissions
    custom_metrics = set()
    for sub in submissions:
        for cf in sub.custom_fields:
            if cf.field_type == 'metric':
                custom_metrics.add(cf.name)
    
    # Add dynamic metrics names to custom_metrics
    leaderboard_metrics_map = {lm.global_metric.name: lm for lm in leaderboard.leaderboard_metrics}
    for dm_name, lm in leaderboard_metrics_map.items():
        custom_metrics.add(dm_name)

    # # Also collect dynamic metrics found in chart_metrics_data (e.g. hist_entropy_*)
    # # These are not in sub.custom_fields but are in the calculated metrics
    # for item in chart_metrics_data:
    #     for sub_name, metrics_dict in item['metrics'].items():
    #          for m_key in metrics_dict.keys():
    #              if m_key.startswith('hist_entropy_') and m_key != 'hist_entropy':
    #                  custom_metrics.add(m_key)
                 
    # Force remove 'hist_entropy' if it somehow got added
    # if 'hist_entropy' in custom_metrics:
    #     custom_metrics.remove('hist_entropy')

    # Combine standard metrics with custom metrics
    standard_metrics = [m for m in leaderboard.summary_metrics.split(',') if m.strip()]
    all_selected_metrics = standard_metrics + sorted(list(custom_metrics))
    
    # Prune hist_entropy from custom_metrics if no submission has it?
    # No, it's per submission. The Chart rendering iterates per submission.
    # The frontend uses 'custom_metrics' (now 'per_sample_custom_metrics') as the label list for the chart.
    # If 'hist_entropy' is in that list, it reserves a spot on the X-axis for it.
    # If a specific submission lacks data for it, it shows 0.
    # To hide it completely for a submission that doesn't have it, we can't easily do that if other submissions DO have it
    # because the X-axis is shared (unless we want dynamic X-axes per submission card, which works).
    # The current frontend logic shares the label set?
    # No, 'allMetricsForChart.map(m => sampleMetricData.metrics['GT'][m] ...'
    # Wait, the frontend iterates `datasetMetricsData` but for SUBMISSIONS it iterates `chartMetricsData`.
    # Let's check `comparison.html` again.
    
    print(f"DEBUG: Standard metrics: {standard_metrics}")
    print(f"DEBUG: Custom metrics: {custom_metrics}")
    print(f"DEBUG: All selected metrics: {all_selected_metrics}")
    
    # Build submissions_json with both standard and custom metrics
    submissions_json = []
    # Map to store aggregated dynamic values for averages in submissions_json
    dynamic_v_sums = {s.id: {dm_name: [] for dm_name in leaderboard_metrics_map} for s in submissions}
    for item in chart_metrics_data:
        for sub_name, metrics_dict in item['metrics'].items():
            # Find sub_id from name (inefficient but works for now)
            sub_id = next((s.id for s in submissions if s.name == sub_name), None)
            if sub_id:
                for dm_name in leaderboard_metrics_map:
                    if dm_name in metrics_dict and metrics_dict[dm_name] is not None:
                        # Convert back from log_safe if possible? No, we need raw value.
                        # Wait, the comparison view loop above added RAW values to sample_info['custom_metrics'].
                        # Let's use that.
                        pass

    # --- Efficient Metric Fetching (Replacing slow 2-Pass Python loops) ---
    calculated_dynamic_values = {} # {sub_id: {metric_name: value}}
    sub_ids = [s.id for s in submissions]

    # 1. Fetch pre-calculated MetricResult records (Aggregated/Global Metrics)
    results = MetricResult.query.filter(MetricResult.submission_id.in_(sub_ids)).all()
    for res in results:
        if res.submission_id not in calculated_dynamic_values:
            calculated_dynamic_values[res.submission_id] = {}
        metric_name = res.leaderboard_metric.target_name or res.leaderboard_metric.global_metric.name
        calculated_dynamic_values[res.submission_id][metric_name] = res.value if not res.error_message else str(res.error_message)

    # 2. Fetch/Aggregate Custom Metric fields (e.g. standard custom metrics)
    for sub in submissions:
        if sub.id not in calculated_dynamic_values:
            calculated_dynamic_values[sub.id] = {}
        for m in custom_metrics:
            if m in calculated_dynamic_values[sub.id]: continue
            
            # Use same logic as leaderboard_view for aggregation
            avg_val = db.session.query(func.avg(CustomField.value_float)).filter(
                CustomField.submission_id == sub.id,
                CustomField.name == m,
                CustomField.field_type == 'metric'
            ).scalar()
            calculated_dynamic_values[sub.id][m] = avg_val

    # Build submissions_json using the pre-calculated values
    submissions_json = []
    for s in submissions:
        sub_data = {'id': s.id, 'name': s.name}
        # Add standard metrics
        for m in AVAILABLE_METRICS:
            sub_data[m] = getattr(s, m)
            
        # Add custom/dynamic metrics
        for metric_name in custom_metrics:
            if metric_name in leaderboard_metrics_map:
                # Use value from 2-pass calculation
                sub_data[metric_name] = calculated_dynamic_values[s.id].get(metric_name)
            else:
                # Standard Database Custom Metric
                metric_fields = [cf for cf in s.custom_fields if cf.name == metric_name and cf.field_type == 'metric']
                if metric_fields:
                    avg_value = sum(cf.value_float for cf in metric_fields if cf.value_float is not None) / len(metric_fields)
                    sub_data[metric_name] = avg_value
                else:
                    sub_data[metric_name] = None
                    
        submissions_json.append(sub_data)
    # Discovery moved early for sorting
    
    print(f"DEBUG: Final chart_metrics_data before JSON dump: {chart_metrics_data}")
    print(f"DEBUG: Submissions JSON: {submissions_json}")
    
    # Build active visualizations from LeaderboardVisualization model
    # Legacy: active_visualizations = [v for v in leaderboard.visualizations.split(',') if v.strip()]
    leaderboard_viz_list = sorted(leaderboard.leaderboard_visualizations, key=lambda x: x.display_order)
    active_visualizations = [lv.target_name or lv.global_visualization.name for lv in leaderboard_viz_list]
    
    # Build visualization metadata for template
    visualization_configs = {}
    for lv in leaderboard_viz_list:
        viz_name = lv.target_name or lv.global_visualization.name
        visualization_configs[viz_name] = {
            'id': lv.id,
            'is_aggregated': lv.global_visualization.is_aggregated,
            'global_viz_id': lv.global_visualization.id,
        }
    
    # Filter active_metrics to ONLY include per-sample metrics for the detailed charts
    # Aggregated metrics should only appear in the summary chart/table
    aggregated_metric_names = {lm.global_metric.name for lm in leaderboard.leaderboard_metrics if lm.global_metric.is_aggregated}
    raw_selected_metrics = [m for m in leaderboard.selected_metrics.split(',') if m.strip()]
    active_metrics = [m for m in raw_selected_metrics if m not in aggregated_metric_names]
    
    print(f"DEBUG: leaderboard.selected_metrics = '{leaderboard.selected_metrics}'")
    print(f"DEBUG: aggregated_metric_names = {aggregated_metric_names}")
    print(f"DEBUG: active_metrics (filtered) = {active_metrics}")
 
    # Fix: Deduplicate sort options. Remove active_metrics from all_custom_fields
    # so they don't appear twice in the dropdown (once as Active, once as Custom)
    all_custom_fields = [f for f in all_custom_fields if f not in active_metrics]

    # Determine available columns based on data existence
    available_display_options = COMPARISON_DISPLAY_OPTIONS.copy()
    
    # Inject Custom Fields into display options with proper ordering
    # Dynamic sample metric options (no custom metrics - they auto-appear in charts)
    sample_metric_options_dynamic = SAMPLE_METRIC_OPTIONS.copy()
    
    # Ensure all active_metrics have a label so they appear in the Sort dropdown
    for m in active_metrics:
        if m not in sample_metric_options_dynamic:
            # Use the metric name as a label (or fallback)
            # We preserve the original case for the 'Active' group
            sample_metric_options_dynamic[m] = m

    # Clean up available_display_options: remove metrics that should now be handled via sample_metric_options_dynamic
    # Actually, they weren't added to available_display_options yet in dataset_view, but I should be careful.
    
    # Determine available columns based on data existence
    available_display_options = COMPARISON_DISPLAY_OPTIONS.copy()
    
    # Inject Custom Fields into display options with proper ordering
    sample_metric_options_dynamic = SAMPLE_METRIC_OPTIONS.copy()
    
    for m in active_metrics:
        if m not in sample_metric_options_dynamic:
            sample_metric_options_dynamic[m] = m

    custom_image_fields = []
    dataset_fields_dict = {}
    submission_fields_dict = {}
    
    for field_name in all_custom_fields:
        field_type = all_field_types.get(field_name, 'image')
        if field_name in dataset_custom_fields:
            dataset_fields_dict[field_name] = field_type
        else:
            submission_fields_dict[field_name] = field_type
    
    # Add dataset fields first
    for field_name in sorted(dataset_fields_dict.keys()):
        field_type = dataset_fields_dict[field_name]
        if field_type in ['scalar', 'metric']: continue
        available_display_options[field_name] = {'label': field_name, 'type': field_type, 'default_width': '300px'}
        if field_type in ['image', 'depth']: custom_image_fields.append(field_name)
    
    # Build submission field mapping
    submission_field_map = {}
    for sub_id, field_names in submission_custom_fields.items():
        for field_name in field_names:
            if field_name not in dataset_custom_fields:
                if field_name not in submission_field_map: submission_field_map[field_name] = []
                submission_field_map[field_name].append((sub_id, field_name))
    
    for base_name in sorted(submission_field_map.keys()):
        sorted_entries = sorted(submission_field_map[base_name], key=lambda x: x[0])
        for sub_id, field_name in sorted_entries:
            field_type = all_field_types.get(field_name, 'image')
            if field_type in ['scalar', 'metric']: continue
            available_display_options[field_name] = {'label': field_name, 'type': field_type, 'default_width': '300px'}
            if field_type in ['image', 'depth']: custom_image_fields.append(field_name)

    # 1. Check GT fields availability (only for samples on current page for UI rendering hint)
    has_gt_hist = any(s.histogram_data for s in samples_on_page)
    has_gt_config = any(s.config_data for s in samples_on_page)
    has_dataset_tags = any(s.tags for s in samples_on_page)
    
    if not has_gt_hist: available_display_options.pop('gt_histogram', None)
    if not has_gt_config: available_display_options.pop('gt_config', None)
    if not has_dataset_tags: available_display_options.pop('dataset_tags', None)
    
    
    # 2. Check Submission fields (heuristic: check first sample for each submission)
    has_pred_peak = False
    submission_has_histogram = {} 
    
    # Discover available histogram types dynamically
    available_hist_types = set()
    
    if samples_on_page:
        test_sample = samples_on_page[0]
        # We also need to iterate a bit more broadly if different submissions have different fields?
        # But for display COLUMNS, we usually want the union of what's possible.
        # Let's check comparison_data to see what keys exist in 'predictions'
        if comparison_data:
             # Check the first sample's predictions
             first_sample_preds = comparison_data[0]['predictions']
             for pred in first_sample_preds:
                 for key in pred.keys():
                     if key.startswith('histogram_'):
                         available_hist_types.add(key)

        for sub in submissions:
            sub_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(sub.id))
            # Check specific histogram existence for display availability
            if sub.id not in submission_has_histogram:
                submission_has_histogram[sub.id] = set()
            
            for hist_type in available_hist_types:
                # hist_type is e.g. 'histogram_hist_pred'
                # File path expected: .../hist_pred/sample.npz
                folder_name = hist_type.replace('histogram_', '')
                if os.path.exists(os.path.join(sub_folder, folder_name, f'{test_sample.name}.npz')):
                    submission_has_histogram[sub.id].add(hist_type)
            
            if not has_pred_peak:
                if os.path.exists(os.path.join(sub_folder, 'metric_peak', f'{test_sample.name}.txt')):
                    has_pred_peak = True
            
    # Also check if we have any custom metrics/charts to show in 'per_sample_metrics'
    if not all_selected_metrics:
        available_display_options.pop('per_sample_metrics', None)

    # Add discovered histograms to display options
    for hist_key in available_hist_types:
        folder_name = hist_key.replace('histogram_', '')
        label = folder_name.replace('hist_', '').replace('_', ' ').title()
        available_display_options[hist_key] = {'label': label, 'type': 'chart', 'default_width': '150px'}
    
    # Remove legacy hardcoded 'pred_histogram' to prefer the dynamic ones
    available_display_options.pop('pred_histogram', None) 
    # For now, let's allow it to be driven by available_hist_types.
    
    # Legacy: if 'pred_histogram' was in options/columns, map it if available
    # But wait, existing saved leaderboards might have 'pred_histogram' in their column string.
    # We should handle that mapping in the template or here.
    # If dynamic 'histogram_*' is found, we can alias it or just rely on new key.
    # Let's just add the dynamic ones.

    # Add per-sample visualizations as display columns
    for viz_name, viz_config in visualization_configs.items():
        if not viz_config['is_aggregated']:  # Only per-sample visualizations are columns
            available_display_options[f'viz_{viz_name}'] = {
                'label': f'📊 {viz_name}',
                'type': 'visualization',
                'default_width': '250px',
                'viz_id': viz_config['id']
            }


    # Filter selected columns again to ensure we don't try to display a column that we just determined is unavailable
    # This handles the "Display columns row" request by updating the default set passed to template

    # Filter selected columns matches available options
    selected_comparison_display_columns = [col for col in selected_comparison_display_columns if col in available_display_options]
    
    # Enforce column order: Name -> Metric Charts -> Other Fields
    def column_sort_key(col):
        if col == 'sample_name': return 0
        if col == 'per_source_stats': return 1
        return 2 # Everything else

    selected_comparison_display_columns.sort(key=column_sort_key)
    

    
    # NOTE: The template currently iterates `selected_comparison_display_columns` AND `custom_metrics` AND `custom_image_fields`.
    # To fully satisfy "Display columns fields should include all columns... enabled by default",
    # I ideally need to move the rendering entirely to the generic loop.
    # However, that is a bigger refactor of the template.
    # For now, I will just ensure they appear in the OPTIONS list so they can be toggled.
    # The display visibility is controlled by `comparison_display_columns` string.
    # If the user toggles it in the form, it updates the string.
    # So if I add them to `available_display_options`, they appear in the form.
    # But for them to render, the template needs to check this string for these keys.
    # The template currently has explicit loops: `{% for custom_met in custom_metrics %}`.
    # This implies they are ALWAYS shown regardless of the display_columns setting?
    # NO, wait. `custom_metrics` variable is just a list of available metrics.
    # If I want them controllable, I should change the template to ONLY render them if in `selected_comparison_display_columns`.
    # I will proceed with adding them to `available_display_options` here.
    
    # Make sure we don't lose the separate lists needed for data extraction in template if we don't fully refactor rendering yet.
    # But for the FORM, this is key.

    print(f"DEBUG: available_display_options.keys() = {list(available_display_options.keys())}")
    print(f"DEBUG: selected_comparison_display_columns = {selected_comparison_display_columns}")
    
    # Filter custom_metrics to exclude aggregated metrics (like F1 score) so they don't appear in per-sample charts/tables
    per_sample_custom_metrics = [m for m in sorted(list(custom_metrics)) if m not in aggregated_metric_names]
    
    # Sort available_display_options by priority for the "View Options" form
    sorted_display_options = dict(sorted(available_display_options.items(), 
                                          key=lambda x: get_column_priority(x[0], x[1].get('type'), x[0] in dataset_custom_fields)))
    
    # Enable all columns by default ONLY on first visit (when no selection is saved)
    # Otherwise, respect the saved selection to allow users to disable columns
    if leaderboard.comparison_display_columns == '__NONE__':
        # User explicitly wants nothing selected
        selected_comparison_display_columns = []
    elif not leaderboard.comparison_display_columns.strip() or leaderboard.comparison_display_columns == DEFAULT_COMPARISON_DISPLAY_COLUMNS:
        # First time or using old defaults - enable all available columns except per_sample_values
        selected_comparison_display_columns = [k for k in available_display_options.keys() if k != 'per_sample_values']
    # else: use the saved selection as loaded earlier
    
    # Also ensure selected_comparison_display_columns are sorted by priority
    selected_comparison_display_columns = sorted(selected_comparison_display_columns, 
                                                  key=lambda x: get_column_priority(x, available_display_options.get(x, {}).get('type'), x in dataset_custom_fields))

    # Pass metric directions for coloring
    metric_directions = json.loads(leaderboard.metric_directions) if leaderboard.metric_directions else {}

    return render_template('comparison.html', 
                           leaderboard=leaderboard, 
                           submissions=submissions,
                           comparison_data=comparison_data,
                           selected_metrics=all_selected_metrics,
                           chart_metrics_data=chart_metrics_data, 
                           submissions_json=submissions_json,
                           all_tags=all_comparison_tags, # Assuming all_tags should be all_comparison_tags
                           custom_metrics=per_sample_custom_metrics,
                           custom_image_fields=custom_image_fields,
                           all_comparison_tags=list(all_comparison_tags),
                           all_sample_tag_names=all_sample_tag_names,
                           all_sample_prefixes=all_sample_prefixes,
                           all_custom_fields=all_custom_fields,
                           all_field_types=all_field_types,
                           dataset_custom_fields=dataset_custom_fields,
                           submission_custom_fields=submission_custom_fields,
                           submission_has_histogram=submission_has_histogram,
                           paginated_samples=paginated_samples, 
                           per_page_options=[5, 10, 20, 100], 
                           current_per_page=per_page, 
                           search_query=search_query, 
                           comparison_display_options=sorted_display_options, 
                           selected_comparison_display_columns=selected_comparison_display_columns, 
                           visualization_options=VISUALIZATION_OPTIONS, 
                           active_visualizations=active_visualizations,
                           visualization_configs=visualization_configs,
                           leaderboard_viz_list=leaderboard_viz_list,
                           sort_by=sort_by, 
                           sort_order=sort_order, 
                           sample_metric_options=sample_metric_options_dynamic, 
                           metric_directions=metric_directions,
                           active_metrics=active_metrics,
                           project_name=project_name,
                           current_compare_ids=compare_ids_arg)





@app.route('/docs/')
@app.route('/docs/<path:page>')
def docs(page='index'):
    if not page:
        page = 'index'
    
    if not page.endswith('.html'):
        template_name = f"docs/{page}.html"
    else:
        template_name = f"docs/{page}"
    
    try:
        # Pass page to template so sidebar can highlight active link
        return render_template(template_name, page=page.replace('.html', ''))
    except Exception:
        return render_template('docs/index.html', page='index')


@app.route('/api/depth_image/<path:filepath>')
def serve_depth_image(filepath):
    """
    Serve a depth map .npz as a heatmap image.
    filepath should be relative to UPLOAD_FOLDER.
    """
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    
    if not os.path.exists(full_path):
        return abort(404, description="File not found")

    try:
        # Load npz
        with np.load(full_path) as data:
            # Heuristic: Find first 2D array
            arr = None
            for key in data.files:
                if len(data[key].shape) == 2:
                    arr = data[key]
                    break
            
            if arr is None:
                # Fallback to first available if any
                if data.files:
                    arr = data[data.files[0]]
                else:
                    return abort(500, description="Empty npz file")

        # Plot
        fig = plt.figure(figsize=(4, 3), dpi=72) # Slightly wider for colorbar
        ax = fig.add_axes([0, 0, 0.85, 1]) # Adjust specific axes for image
        ax.set_axis_off()
        im = ax.imshow(arr, cmap='turbo', aspect='auto')
        
        # Add colorbar
        cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
        fig.colorbar(im, cax=cax)
        
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)
        output.seek(0)
        
        return send_file(output, mimetype='image/png')
            
    except Exception as e:
        print(f"Error serving depth image: {e}")
        return abort(500, description=str(e))

@app.route('/api/depth_data/<path:filepath>')
def serve_depth_data(filepath):
    """
    Serve raw depth map data as JSON for interactive visualization.
    filepath should be relative to UPLOAD_FOLDER.
    """
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    
    if not os.path.exists(full_path):
        return abort(404, description="File not found")

    try:
        # Load npz
        with np.load(full_path) as data:
            # Heuristic: Find first 2D array
            arr = None
            for key in data.files:
                if len(data[key].shape) == 2:
                    arr = data[key]
                    break
            
            if arr is None:
                # Fallback to first available if any
                if data.files:
                    arr = data[data.files[0]]
                else:
                    return abort(500, description="Empty npz file")
        
        # Check if we need to json-serialize types (numpy types to python types)
        return jsonify({
            'data': arr.tolist(),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'shape': arr.shape
        })
            
    except Exception as e:
        print(f"Error serving depth data: {e}")
        return abort(500, description=str(e))
            
    except Exception as e:
        print(f"Error serving depth image: {e}")
        return abort(500, description=str(e))


@app.route('/datasets')
def datasets_list():
    datasets = Dataset.query.order_by(Dataset.upload_date.desc()).all()
    return render_template('datasets.html', datasets=datasets)

@app.route('/dataset/<int:dataset_id>')
def dataset_view(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    page = request.args.get('page', 1, type=int)
    samples_per_page = request.args.get('per_page', 5, type=int)
    sort_by = request.args.get('sort_by', 'name')
    sort_order = request.args.get('sort_order', 'asc')
    selected_display_columns = dataset.display_columns.split(',')
    
    # Collect all unique tags and prefixes for this dataset for filtering and auto-suggestions
    all_dataset_tags, all_dataset_prefixes = get_all_sample_tags(dataset.id)
    
    samples_query = Sample.query.filter_by(dataset_id=dataset.id)
    
    # Apply Tag Filters
    samples_query = apply_tag_filters(samples_query, request.args)

    # Collect unique custom field names from the dataset early for sorting/display
    # Use a distinct query instead of iterating through all samples
    custom_field_query = db.session.query(CustomField.name, CustomField.field_type).join(Sample).filter(
        Sample.dataset_id == dataset.id,
        CustomField.submission_id == None
    ).distinct().all()
    
    custom_field_names = set(custom_field_query)
    custom_scalar_metric_names = [name for name, ftype in custom_field_names if ftype == 'scalar']

    # Sorting
    if sort_by == 'name':
        if sort_order == 'desc':
            samples_query = samples_query.order_by(Sample.name.desc())
        else:
            samples_query = samples_query.order_by(Sample.name.asc())
    elif sort_by in custom_scalar_metric_names:
        # Complex sort by custom field requires joining CustomField
        # We join and order by value_float
        samples_query = samples_query.join(CustomField, isouter=True).filter(
            or_(CustomField.name == sort_by, CustomField.id == None)
        )
        if sort_order == 'desc':
            samples_query = samples_query.order_by(CustomField.value_float.desc())
        else:
            samples_query = samples_query.order_by(CustomField.value_float.asc())

    # Pagination
    total = samples_query.count()
    paginated_samples = samples_query.paginate(page=page, per_page=samples_per_page, error_out=False)

    active_metrics = [m for m in dataset.selected_metrics.split(',') if m.strip()]

    dataset_metrics_data = [] # For per_sample_metrics chart
    samples_data_for_charts = []
    
    # Only process samples on the current page
    for sample in paginated_samples.items:
        tags_list = [t.strip() for t in sample.tags.split(',')] if sample.tags else []
        
        # Standard metrics
        metrics = {}
        # Add custom scalar metrics for this sample
        for cf in sample.custom_fields:
            if cf.field_type == 'scalar' and cf.submission_id is None:
                metrics[cf.name] = cf.value_float

        dataset_metrics_data.append({
            'sample_id': sample.id,
            'sample_name': sample.name,
            'metrics': {'GT': metrics}
        })

        if 'histogram' in selected_display_columns and sample.histogram_data:
            samples_data_for_charts.append({
                'id': sample.id, 
                'name': sample.name, 
                'bins': json.loads(sample.histogram_data.bins), 
                'counts': json.loads(sample.histogram_data.counts), 
                'tags': tags_list
            })
        else:
            samples_data_for_charts.append({'id': sample.id, 'name': sample.name, 'bins': [], 'counts': [], 'tags': tags_list})
    
    # Create a map of sample_id -> {field_name: value}
    # Only for samples on current page
    custom_fields_map = {}
    for sample in paginated_samples.items:
        custom_fields_map[sample.id] = {}
        for cf in sample.custom_fields:
            # We skip submission-specific custom fields in the dataset view
            if cf.submission_id is not None:
                continue
                
            if cf.field_type == 'image':
                custom_fields_map[sample.id][cf.name] = {'type': 'image', 'value': cf.value_text, 'field_id': cf.id}
            elif cf.field_type == 'scalar':
                custom_fields_map[sample.id][cf.name] = {'type': 'scalar', 'value': cf.value_float, 'field_id': cf.id}
            elif cf.field_type == 'depth':
                 custom_fields_map[sample.id][cf.name] = {'type': 'depth', 'value': cf.value_text, 'field_id': cf.id}
            elif cf.field_type == 'json':
                 custom_fields_map[sample.id][cf.name] = {'type': 'json', 'value': cf.value_text, 'field_id': cf.id}
            elif cf.field_type == 'text':
                 custom_fields_map[sample.id][cf.name] = {'type': 'text', 'value': cf.value_text, 'field_id': cf.id}
    
    # Determine available columns based on data existence
    available_display_options = DATASET_DISPLAY_OPTIONS.copy()
    
    # Inject detected custom fields (excluding scalars - they appear in GT Stats)
    for field_name, field_type in custom_field_names:
        if field_type == 'image':
            available_display_options[field_name] = {'label': field_name, 'type': 'image', 'default_width': '150px'}
        elif field_type == 'depth':
            available_display_options[field_name] = {'label': field_name, 'type': 'depth', 'default_width': '150px'}
        elif field_type == 'json':
            available_display_options[field_name] = {'label': field_name, 'type': 'json', 'default_width': '150px'}
        # Scalars are not added as individual columns - they appear in per_source_stats
    
    # Check if any sample has data for these fields
    # Using all_dataset_tags check from above for 'tags'
    has_tags = bool(all_dataset_tags)
    
    # For DB fields, we can do efficient queries or simple checks on all_samples
    all_samples = dataset.samples
    has_gt = any(sample.custom_fields for sample in all_samples if any(cf.name == 'pick' for cf in sample.custom_fields))
    has_hist = any(s.histogram_data for s in all_samples)
    has_shape = any(s.signal_shape for s in all_samples)
    

    if not has_hist: available_display_options.pop('histogram', None)
    if not has_shape: available_display_options.pop('signal_shape', None)
    if not has_tags: available_display_options.pop('tags', None)

    
    # Filter selected columns to ensure they exist in available options
    selected_display_columns = [col for col in selected_display_columns if col in available_display_options]

    active_visualizations = [v for v in dataset.visualizations.split(',') if v.strip()]
    # Dynamic sample metric options (no custom metrics - they auto-appear in charts)
    sample_metric_options_dynamic = SAMPLE_METRIC_OPTIONS.copy()
    
    # Extract custom scalar metric names for auto-inclusion in charts
    custom_scalar_metrics = [field_name for field_name, field_type in custom_field_names if field_type == 'scalar']
    
    # Create a set of all custom field names for priority sorting
    # In dataset view, all custom fields belong to the dataset (no submissions)
    dataset_custom_fields = {field_name for field_name, _ in custom_field_names}
    
    # Sort available_display_options by priority for the "View Options" form
    sorted_display_options = dict(sorted(available_display_options.items(), 
                                          key=lambda x: get_column_priority(x[0], x[1].get('type'), x[0] in dataset_custom_fields)))
    
    # Enable all columns by default ONLY on first visit (when no selection is saved)
    # Otherwise, respect the saved selection to allow users to disable columns
    if dataset.display_columns == '__NONE__':
        # User explicitly wants nothing selected
        selected_display_columns = []
    elif not dataset.display_columns.strip() or dataset.display_columns == DEFAULT_DATASET_DISPLAY_COLUMNS:
        # First time or using old defaults - enable all available columns
        selected_display_columns = list(available_display_options.keys())
    # else: use the saved selection as loaded earlier
    
    # Also ensure selected_display_columns are sorted by priority
    selected_display_columns = sorted(selected_display_columns, 
                                       key=lambda x: get_column_priority(x, available_display_options.get(x, {}).get('type'), x in dataset_custom_fields))

    return render_template('dataset_view.html', 
                           dataset=dataset, 
                           paginated_samples=paginated_samples, 
                           samples_data_for_charts=samples_data_for_charts, 
                           dataset_display_options=sorted_display_options, 
                           selected_display_columns=selected_display_columns, 
                           per_page_options=[5, 10, 20, 100], 
                           current_per_page=samples_per_page, 
                           all_dataset_tags=all_dataset_tags,
                           all_dataset_prefixes=all_dataset_prefixes,
                           visualization_options=VISUALIZATION_OPTIONS, 
                           active_visualizations=active_visualizations, 
                           dataset_metrics_data=dataset_metrics_data,
                           sort_by=sort_by, 
                           sort_order=sort_order, 
                           sample_metric_options=sample_metric_options_dynamic, 
                           active_metrics=active_metrics,
                           custom_field_names=sorted(list(custom_field_names)),
                           custom_fields_map=custom_fields_map,
                           custom_scalar_metrics=custom_scalar_metrics)


@app.route('/dataset/<int:dataset_id>/update_display_columns', methods=['POST'])
def update_dataset_display_columns(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    cols = request.form.getlist('display_columns')
    
    # We need to look up types for sorting. 
    # Since we don't have available_display_options here easily without repeating logic,
    # we'll do a simple key-based sort here, and the view render will do the definitive sort.
    # Actually, saving in the requested order is good.
    # Use a sentinel value to distinguish "user wants nothing" from "use defaults"
    dataset.display_columns = ','.join(cols) if cols else '__NONE__'

    db.session.commit()
    return redirect(request.referrer)

@app.route('/dataset/<int:dataset_id>/update_visualizations', methods=['POST'])
def update_dataset_visualizations(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset.visualizations = ','.join(request.form.getlist('visualizations'))
    db.session.commit()
    return redirect(request.referrer)

@app.route('/dataset/<int:dataset_id>/update_metrics', methods=['POST'])
def update_dataset_metrics(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset.selected_metrics = ','.join(request.form.getlist('metrics'))
    db.session.commit()
    return redirect(request.referrer)

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/update_visualizations', methods=['POST'])
def update_leaderboard_visualizations(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    leaderboard.visualizations = ','.join(request.form.getlist('visualizations'))
    db.session.commit()
    return redirect(request.referrer)

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/update_comparison_display_columns', methods=['POST'])
def update_comparison_display_columns(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    cols = request.form.getlist('comparison_display_columns')
    # Use a sentinel value to distinguish "user wants nothing" from "use defaults"
    leaderboard.comparison_display_columns = ','.join(cols) if cols else '__NONE__'
    db.session.commit()
    return redirect(request.referrer)

@app.route('/<project_name>/submission/<int:submission_id>/download')
def download_submission(project_name, submission_id):
    submission = Submission.query.get_or_404(submission_id)
    submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id))
    zip_path = os.path.join(submission_folder, 'submission.zip')
    
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name=f"{submission.name}.zip")
    else:
        flash("Original submission ZIP not found.", "warning")
        return redirect(url_for('leaderboard_view', leaderboard_id=submission.leaderboard_id))

@app.route('/<project_name>/leaderboard/<int:leaderboard_id>/download_submissions', methods=['POST'])
def download_submissions_bulk(project_name, leaderboard_id):
    submission_ids = request.form.getlist('submission_ids')
    redirect_args = {
        'leaderboard_id': leaderboard_id,
        'search_query': request.form.get('search_query', ''),
        'show_archived': request.form.get('show_archived', 'false'),
        'enable_include': request.form.get('enable_include', 'false'),
        'enable_exclude': request.form.get('enable_exclude', 'false'),
        'enable_prefix': request.form.get('enable_prefix', 'false'),
        'include_tags': request.form.get('include_tags', ''),
        'exclude_tags': request.form.get('exclude_tags', ''),
        'prefix_tags': request.form.get('prefix_tags', ''),
        'sort_metric': request.form.get('sort_metric', ''),
        'sort_order': request.form.get('sort_order', 'asc'),
        'sample_search_query': request.form.get('sample_search_query', ''),
        'enable_sample_include': request.form.get('enable_sample_include', 'false'),
        'enable_sample_exclude': request.form.get('enable_sample_exclude', 'false'),
        'enable_sample_prefix': request.form.get('enable_sample_prefix', 'false'),
        'sample_include_tags': request.form.get('sample_include_tags', ''),
        'sample_exclude_tags': request.form.get('sample_exclude_tags', ''),
        'sample_prefix_tags': request.form.get('sample_prefix_tags', '')
    }
    if not submission_ids:
        flash("No submissions selected.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))
        
    submissions = Submission.query.filter(Submission.id.in_(submission_ids), Submission.leaderboard_id == leaderboard_id).all()
    
    if not submissions:
        flash("No valid submissions found.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))

    # Create a temporary zip file
    import tempfile
    from zipfile import ZipFile
    
    # We use a temp file that survives the block so we can send it
    tmp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close() # Close handle so ZipFile can open it

    try:
        with ZipFile(tmp_path, 'w') as zf:
            for sub in submissions:
                sub_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(sub.id))
                sub_zip_path = os.path.join(sub_folder, 'submission.zip')
                
                if os.path.exists(sub_zip_path):
                    # Add to zip with a nice name
                    zf.write(sub_zip_path, arcname=f"{sub.name}_{sub.id}.zip")
    except Exception as e:
        if os.path.exists(tmp_path):
             os.remove(tmp_path)
        flash(f"Error creating bulk zip: {e}", "danger")
        return redirect(url_for('leaderboard_view', **redirect_args))
    
    @after_this_request
    def remove_file(response):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as error:
            app.logger.error("Error removing temp bulk zip", error)
        return response

    return send_file(tmp_path, as_attachment=True, download_name="bulk_submissions.zip")

@app.route('/dataset/<int:dataset_id>/delete', methods=['POST'])
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset_folder_name = secure_filename(dataset.name)
    shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name), ignore_errors=True)
    db.session.delete(dataset)
    db.session.commit()
    return redirect(url_for('datasets_list'))

@app.route('/<project_name>/delete_leaderboard/<int:leaderboard_id>', methods=['POST'])
def delete_leaderboard(project_name, leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    db.session.delete(leaderboard)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/<project_name>/delete_submission/<int:submission_id>', methods=['POST'])
def delete_submission(project_name, submission_id):
    submission = Submission.query.get_or_404(submission_id)
    shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id)), ignore_errors=True)
    db.session.delete(submission)
    db.session.commit()
    return redirect(url_for('leaderboard_view', leaderboard_id=submission.leaderboard_id))

@app.route('/<project_name>/custom_field_image/<int:field_id>')
def serve_custom_field_image(project_name, field_id):
    """Serve a custom field image or depth map"""
    custom_field = CustomField.query.get_or_404(field_id)
    
    if custom_field.field_type == 'depth':
        return serve_depth_image(custom_field.value_text)
    
    if custom_field.field_type != 'image':
        return "Not an image/depth field", 400
    
    # value_text contains the relative path from uploads folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], custom_field.value_text)
    
    if not os.path.exists(image_path):
        return "Image not found", 404
    
    return send_file(image_path)

@app.route('/<project_name>/api/custom_field_depth_data/<int:field_id>')
def serve_custom_field_depth_data(project_name, field_id):
    """Serve raw depth data for a custom field as JSON."""
    custom_field = CustomField.query.get_or_404(field_id)
    
    if custom_field.field_type != 'depth':
        return abort(400, description="Not a depth field")
        
    return serve_depth_data(custom_field.value_text)

@app.route('/<project_name>/api/custom_field_json/<int:field_id>')
def serve_custom_field_json(project_name, field_id):
    """Serve JSON data for a custom field."""
    custom_field = CustomField.query.get_or_404(field_id)
    
    if custom_field.field_type != 'json':
        return abort(400, description="Not a JSON field")
    
    # value_text contains the relative path from uploads folder
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], custom_field.value_text)
    
    if not os.path.exists(json_path):
        return abort(404, description="JSON file not found")
    
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        return jsonify(json_data)
    except Exception as e:
        return abort(500, description=f"Error reading JSON file: {str(e)}")


@app.route('/<project_name>/sample/<int:sample_id>/download')
def download_sample(project_name, sample_id):
    sample = Sample.query.get_or_404(sample_id)
    # Optional submission IDs to include in the zip
    submission_ids = request.args.getlist('submission_id', type=int)
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # 1. Add Ground Truth (Dataset) fields reconstructed from DB
        # Tags
        if sample.tags:
            zf.writestr(f'ground_truth/tags/{sample.name}.txt', sample.tags)
        
        # Peak
        for cf in sample.custom_fields:
            if cf.name == 'pick' and cf.field_type == 'scalar':
                zf.writestr(f'ground_truth/pick/{sample.name}.txt', str(cf.value_float))
                break
            
        # Shape
        if sample.signal_shape:
            zf.writestr(f'ground_truth/wave_shape/{sample.name}.txt', sample.signal_shape.shape_name)
            
        # Config
        if sample.config_data:
            zf.writestr(f'ground_truth/config/{sample.name}.json', sample.config_data.config_json)
            
        # Histogram (.npz)
        if sample.histogram_data:
            bins = np.array(json.loads(sample.histogram_data.bins))
            counts = np.array(json.loads(sample.histogram_data.counts))
            hist_buf = io.BytesIO()
            np.savez_compressed(hist_buf, bins=bins, counts=counts)
            zf.writestr(f'ground_truth/hist/{sample.name}.npz', hist_buf.getvalue())
        
        # Custom fields from dataset
        for cf in sample.custom_fields:
            if cf.field_type == 'scalar':
                # Write scalar value to text file
                zf.writestr(f'ground_truth/{cf.name}/{sample.name}.txt', str(cf.value_float))
            elif cf.field_type == 'image':
                # Copy image file from uploads folder
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], cf.value_text)
                if os.path.exists(img_path):
                    # Determine file extension from original path
                    ext = os.path.splitext(cf.value_text)[1]
                    zf.write(img_path, f'ground_truth/{cf.name}/{sample.name}{ext}')


        # 2. Add Submission fields from disk
        for sub_id in submission_ids:
            sub = Submission.query.get(sub_id)
            if not sub: continue
            
            sub_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(sub.id))
            if not os.path.exists(sub_folder): continue
            
            # Walk through sub fields: histograms, metric_peak, etc.
            # We want to maintain a structure like submissions/sub_name/folder/sample_name.ext
            sub_label = f"submission_{sub.name}_{sub.id}"
            
            for root, dirs, files in os.walk(sub_folder):
                for file in files:
                    if os.path.splitext(file)[0] == sample.name:
                        # Get relative path from sub_folder
                        rel_path = os.path.relpath(os.path.join(root, file), sub_folder)
                        zf.write(os.path.join(root, file), f'{sub_label}/{rel_path}')

        # 3. Add active visualizations
        # Determine active visualizations for this sample's context
        active_vis = []
        # If part of a leaderboard (passed via submission_ids mostly)
        if submission_ids:
            # Assuming all submissions in submission_ids belong to the same leaderboard
            # Or, if no submissions, then it's just the dataset's visualizations
            if submission_ids:
                first_sub = Submission.query.get(submission_ids[0])
                if first_sub:
                    leaderboard = first_sub.leaderboard
                    active_vis = [v for v in leaderboard.visualizations.split(',') if v.strip()]
            if not active_vis and sample.dataset: # Fallback to dataset if no leaderboard context or no submissions
                active_vis = [v for v in sample.dataset.visualizations.split(',') if v.strip()]
        elif sample.dataset: # If no submission_ids, use dataset's visualizations
            active_vis = [v for v in sample.dataset.visualizations.split(',') if v.strip()]
            
        for vis_type in active_vis:
            # For ground truth context
            img_data = get_vis_image_bytes(vis_type, sample)
            if img_data:
                zf.writestr(f'visualizations/ground_truth_{vis_type}_{sample.name}.png', img_data)
            
            # For each submission
            for sub_id in submission_ids:
                sub = Submission.query.get(sub_id)
                if not sub: continue
                img_data = get_vis_image_bytes(vis_type, sample, sub_id=sub.id)
                if img_data:
                    zf.writestr(f'visualizations/submission_{sub.name}_{vis_type}_{sample.name}.png', img_data)

    memory_file.seek(0)
    return send_file(memory_file, download_name=f'{sample.name}_data.zip', as_attachment=True)

    
def get_all_sample_tags(dataset_id):
    """
    Retrieves all unique sample tags and prefixes for a given dataset.
    Returns (all_tag_names, all_prefixes) as sorted lists.
    """
    all_sample_tags_query = db.session.query(Sample.tags).filter(Sample.dataset_id == dataset_id).all()
    all_sample_tag_names = set()
    for (tags_str,) in all_sample_tags_query:
        if tags_str:
            for t in tags_str.split(','):
                t_trimmed = t.strip()
                if t_trimmed:
                    all_sample_tag_names.add(t_trimmed)
    
    all_tag_names_list = sorted(list(all_sample_tag_names))
    # Consistent with frontend: split by :, =, or -
    all_prefixes = set()
    for t in all_tag_names_list:
        # Using re.split to handle multiple possible separators
        parts = re.split(r'[=:-]', t)
        if len(parts) > 1:
            prefix = parts[0].strip()
            if prefix:
                all_prefixes.add(prefix)
            
    return all_tag_names_list, sorted(list(all_prefixes))

@app.route('/sample/<int:sample_id>/update_tags', methods=['POST'])
def update_sample_tags(sample_id):
    sample = Sample.query.get_or_404(sample_id)
    new_tags = request.form.get('tags', '').strip()
    
    # Optional: sanitize tags (e.g., remove duplicates, empty strings)
    tags_list = [t.strip() for t in new_tags.split(',') if t.strip()]
    sample.tags = ','.join(tags_list)
    
    db.session.commit()
    
    flash(f'Tags updated for sample {sample.name}', 'success')
    return redirect(request.referrer or url_for('dataset_view', dataset_id=sample.dataset_id))

def apply_tag_filters(query, args):
    """
    Applies tag hashing filters to a SQLAlchemy query based on request arguments.
    Supports Include (AND), Exclude (OR), and Prefix (AND) logic.
    """
    enable_include = args.get('enable_include', 'false') == 'true'
    enable_exclude = args.get('enable_exclude', 'false') == 'true'
    enable_prefix = args.get('enable_prefix', 'false') == 'true'

    include_tags = [t.strip().lower() for t in args.get('include_tags', '').split(',') if t.strip()]
    exclude_tags = [t.strip().lower() for t in args.get('exclude_tags', '').split(',') if t.strip()]
    prefix_tags = [t.strip().lower() for t in args.get('prefix_tags', '').split(',') if t.strip()]

    # Helper for exact tag matching in CSV string: "tag", "tag,other", "other,tag", "other,tag,other"
    def tag_match_filter(tag):
        return or_(
            Sample.tags == tag,
            Sample.tags.ilike(f'{tag},%'),
            Sample.tags.ilike(f'%,{tag}'),
            Sample.tags.ilike(f'%,{tag},%')
        )

    # Helper for prefix matching: starts with prefix, or contains ",prefix"
    def prefix_match_filter(prefix):
        return or_(
            Sample.tags.ilike(f'{prefix}%'),
            Sample.tags.ilike(f'%,{prefix}%'), # matches ",prefix..."
            Sample.tags.ilike(f'%, {prefix}%') # matches ", prefix..." just in case of spaces
        )

    if enable_include and include_tags:
        # AND Logic: Must contain ALL include tags
        for tag in include_tags:
            query = query.filter(tag_match_filter(tag))

    if enable_exclude and exclude_tags:
        # OR Logic: Exclude if ANY exclude tag is present
        # i.e., NOT (Has Tag A OR Has Tag B)
        exclude_conditions = [tag_match_filter(tag) for tag in exclude_tags]
        query = query.filter(not_(or_(*exclude_conditions)))

    if enable_prefix and prefix_tags:
        # AND Logic: Must have a tag matching ALL prefixes
        for prefix in prefix_tags:
            query = query.filter(prefix_match_filter(prefix))
            
    return query


# --- API Endpoints ---


@app.route('/api/dataset/upload', methods=['POST'])
def dataset_upload_api():
    """Programmatic dataset upload API"""
    if 'dataset_zip' not in request.files:
        return jsonify({'error': 'No dataset_zip file provided'}), 400
    
    file = request.files['dataset_zip']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    dataset_name = request.form.get('dataset_name', file.filename.replace('.zip', ''))
    # project_id logic removed (Global Datasets)
    override = request.form.get('override', 'false').lower() == 'true'
    
    filename = secure_filename(file.filename)
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_dataset_api_pre')
    os.makedirs(temp_dir, exist_ok=True)
    temp_zip_path = os.path.join(temp_dir, filename)
    file.save(temp_zip_path)

    try:
        success, message, ds_id = process_dataset_zip(temp_zip_path, dataset_name, override=override)
        
        if success:
            return jsonify({'message': message, 'dataset_id': ds_id}), 201
        else:
            return jsonify({'error': message}), 400
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.route('/api/leaderboard/<int:leaderboard_id>/info', methods=['GET'])
def get_leaderboard_info_api(leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    return jsonify({
        'id': leaderboard.id,
        'name': leaderboard.name,
        'dataset': {
            'id': leaderboard.dataset.id,
            'name': leaderboard.dataset.name
        }
    })

@app.route('/<project_name>/api/leaderboard/by_name/<leaderboard_name>/info', methods=['GET'])
def get_leaderboard_info_by_name_api(project_name, leaderboard_name):
    # Look up project
    project = Project.query.filter_by(name=project_name).first()
    if not project:
        return jsonify({'error': f'Project "{project_name}" not found'}), 404
    
    # Filter leaderboard by both name and project
    leaderboard = Leaderboard.query.filter_by(name=leaderboard_name, project_id=project.id).first()
    if not leaderboard:
        return jsonify({'error': f'Leaderboard "{leaderboard_name}" not found in project "{project_name}"'}), 404
    return jsonify({
        'id': leaderboard.id,
        'name': leaderboard.name,
        'dataset': {
            'id': leaderboard.dataset.id,
            'name': leaderboard.dataset.name
        }
    })

@app.route('/api/leaderboard/suggest_name', methods=['GET'])
def suggest_leaderboard_name():
    """Suggest an available leaderboard name, adding _2, _3, etc. if needed."""
    base_name = request.args.get('name', '')
    if not base_name:
        return jsonify({'error': 'No name provided'}), 400
    
    # Check if base name is available
    if not Leaderboard.query.filter_by(name=base_name).first():
        return jsonify({'suggested_name': base_name})
    
    # Try suffixes _2, _3, etc.
    counter = 2
    while True:
        suggested_name = f"{base_name}_{counter}"
        if not Leaderboard.query.filter_by(name=suggested_name).first():
            return jsonify({'suggested_name': suggested_name})
        counter += 1
        # Safety limit to prevent infinite loop
        if counter > 1000:
            return jsonify({'error': 'Could not find available name'}), 500


@app.route('/<project_name>/dataset/<int:dataset_id>/download')
def download_dataset(project_name, dataset_id):
    """Download dataset ZIP file"""
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset_folder_name = secure_filename(dataset.name)
    dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name)
    zip_file = f"{dataset_folder_name}.zip"
    zip_path = os.path.join(dataset_dir, zip_file)
    
    if not os.path.exists(zip_path):
        return "Dataset file not found", 404
    
    return send_file(zip_path, as_attachment=True, download_name=zip_file)

@app.route('/test_ping')
def test_ping():
    return "PONG: Server is running modified code"

@app.route('/test_verification_123')
def test_verification():
    return "VERIFICATION: This route was added at 13:02 on Dec 28"

@app.route('/<project_name>/api/dataset/<dataset_id>/download', methods=['GET'], endpoint='api_download_dataset')
def api_download_dataset(project_name, dataset_id):
    """Download stored dataset ZIP"""
    import sys
    import traceback
    
    # Force convert to int here to handle relaxed typing
    try:
        dataset_id = int(dataset_id)
    except ValueError:
        return jsonify({'error': 'Invalid dataset ID'}), 400

    try:
        # Check if dataset exists using a simpler query first
        exists = db.session.query(Dataset.id).filter_by(id=dataset_id).scalar()
        sys.stderr.write(f"DEBUG DOWNLOAD: Dataset ID {dataset_id} exists in DB: {exists}\n")
        
        if not exists:
             sys.stderr.write(f"DEBUG DOWNLOAD: Dataset {dataset_id} NOT FOUND in DB\n")
             return jsonify({'error': f'Dataset {dataset_id} not found in database'}), 404

        dataset = Dataset.query.get(dataset_id)
        
        # Path to stored ZIP
        dataset_folder_name = secure_filename(dataset.name)
        dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name)
        original_zip_name = f"{dataset_folder_name}.zip"
        original_zip_path = os.path.join(dataset_dir, original_zip_name)
        
        sys.stderr.write(f"DEBUG DOWNLOAD: name={dataset.name}\n")
        sys.stderr.write(f"DEBUG DOWNLOAD: path={original_zip_path}\n")
        sys.stderr.write(f"DEBUG DOWNLOAD: exists={str(os.path.exists(original_zip_path))}\n")
        sys.stderr.flush()

        if not os.path.exists(original_zip_path):
            return jsonify({'error': f'Dataset source file not found at {original_zip_path}'}), 404
            
        return send_file(original_zip_path, as_attachment=True, download_name=original_zip_name)
    except Exception as e:
        sys.stderr.write(f"DEBUG DOWNLOAD: EXCEPTION: {str(e)}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return jsonify({'error': str(e)}), 500

@app.route('/<project_name>/api/leaderboard/<int:leaderboard_id>/submission/upload', methods=['POST'])
def submission_upload_api(project_name, leaderboard_id):
    """Programmatic submission upload API"""
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    if 'submission_zip' not in request.files:
        return jsonify({'error': 'No submission_zip provided'}), 400
        
    file = request.files['submission_zip']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    submission_name = request.form.get('submission_name', file.filename.replace('.zip', ''))
    
    # Process
    temp_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload_zip', secure_filename(file.filename))
    os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
    file.save(temp_zip_path)
    
    try:
        success, error = process_submission_zip(leaderboard.id, submission_name, temp_zip_path)
        if success:
             return jsonify({'success': True, 'message': 'Submission queued'})
        else:
             return jsonify({'error': error}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.route('/api/leaderboard/<int:leaderboard_id>/recalculate_async', methods=['POST'])
def recalculate_leaderboard_async(leaderboard_id):
    """Trigger async recalculation for submissions."""
    data = request.get_json()
    submission_ids = data.get('submission_ids', [])
    sample_filters = data.get('sample_filters', {})
    
    if not submission_ids:
        return jsonify({'error': 'No submission IDs provided'}), 400

    submissions = Submission.query.filter(Submission.id.in_(submission_ids), Submission.leaderboard_id == leaderboard_id).all()
    
    triggered_count = 0
    for sub in submissions:
        sub.processing_status = 'Pending'
        tasks.process_submission.delay(sub.id, sample_filters=sample_filters)
        triggered_count += 1
        
    db.session.commit()
    return jsonify({'success': True, 'triggered_count': triggered_count})

@app.route('/api/leaderboard/<int:leaderboard_id>/metrics_status', methods=['POST'])
def leaderboard_metrics_status(leaderboard_id):
    """Get processing status and metrics for submissions."""
    data = request.get_json()
    submission_ids = data.get('submission_ids', [])
    include_per_sample = data.get('include_per_sample', False)
    sample_ids = data.get('sample_ids', [])
    
    if not submission_ids:
        return jsonify({'error': 'No submission IDs provided'}), 400
        
    from sqlalchemy import func, or_, not_

    submissions = Submission.query.filter(Submission.id.in_(submission_ids), Submission.leaderboard_id == leaderboard_id).all()
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    
    result = {}
    for sub in submissions:
        sub_data = {'status': sub.processing_status}
        
        if sub.processing_status == 'Processed':
            # Fetch numeric results (Leaderboard Metrics)
            metric_results = MetricResult.query.filter_by(submission_id=sub.id).all()
            metrics = {}
            for res in metric_results:
                metric_name = res.leaderboard_metric.global_metric.name
                metrics[metric_name] = res.value
            
            # Fetch custom metrics (scalars) - Aggregate with Filters
            # Parse filters from stored state
            filters = {}
            if sub.last_sample_filter:
                try:
                    filters = json.loads(sub.last_sample_filter)
                except:
                    pass
            
            # Build Aggregation Query - Fetch raw values
            query = db.session.query(
                CustomField.name, 
                CustomField.value_float
            ).join(Sample).filter(
                CustomField.submission_id == sub.id,
                CustomField.field_type == 'metric'
            )
            
            # Apply Filters (Same logic as tasks.py)
            if filters:
                if filters.get('search'):
                    query = query.filter(Sample.name.ilike(f"%{filters['search']}%"))
                
                def tag_match_filter(tag):
                    return or_(
                        Sample.tags == tag,
                        Sample.tags.ilike(f'{tag},%'),
                        Sample.tags.ilike(f'%,{tag}'),
                        Sample.tags.ilike(f'%,{tag},%')
                    )

                include = filters.get('include', {})
                if include.get('enabled') and include.get('tags'):
                    for tag in include['tags']:
                        query = query.filter(tag_match_filter(tag))
                
                exclude = filters.get('exclude', {})
                if exclude.get('enabled') and exclude.get('tags'):
                    exclude_conditions = [tag_match_filter(tag) for tag in exclude['tags']]
                    if exclude_conditions:
                        query = query.filter(not_(or_(*exclude_conditions)))

                prefix = filters.get('prefix', {})
                if prefix.get('enabled') and prefix.get('tags'):
                    prefix_conds = []
                    for p in prefix['tags']:
                        prefix_conds.append(or_(
                            Sample.tags.ilike(f'{p}%'),
                            Sample.tags.ilike(f'%,{p}%'),
                            Sample.tags.ilike(f'%, {p}%')
                        ))
                    if prefix_conds:
                        query = query.filter(or_(*prefix_conds))
            
            # Group by metric name and aggregate in Python
            raw_data = {}
            for name, val in query.all():
                if name not in raw_data:
                    raw_data[name] = []
                if val is not None:
                    raw_data[name].append(val)
            
            # Load aggregation config
            current_aggregation = json.loads(leaderboard.metric_aggregation) if leaderboard.metric_aggregation else {}
            
            for name, values in raw_data.items():
                if values:
                    agg_config = current_aggregation.get(name, {})
                    pooling_type = agg_config.get('type', 'mean')
                    pooling_percentile = agg_config.get('percentile')
                    
                    try:
                        if pooling_type == 'mean':
                            avg_val = float(np.mean(values))
                        elif pooling_type == 'median':
                            avg_val = float(np.median(values))
                        elif pooling_type == 'percentile' and pooling_percentile is not None:
                            avg_val = float(np.percentile(values, float(pooling_percentile)))
                        else:
                            avg_val = float(np.mean(values))
                    except:
                        avg_val = None
                    
                    metrics[name] = avg_val
            
            sub_data['metrics'] = metrics
            
            # Format metrics for display
            formatted = {}
            for name, val in metrics.items():
                if val is None:
                    formatted[name] = '-'
                else:
                    try:
                        formatted[name] = "{:.4f}".format(val)
                    except:
                        formatted[name] = str(val)
            sub_data['metrics_formatted'] = formatted

            # NEW: Per-sample metrics if requested
            if include_per_sample:
                sample_metrics = {}
                if sample_ids:
                    sm_query = db.session.query(
                        CustomField.sample_id,
                        CustomField.name,
                        CustomField.value_float
                    ).filter(
                        CustomField.submission_id == sub.id,
                        CustomField.field_type == 'metric',
                        CustomField.sample_id.in_(sample_ids)
                    ).all()
                    for sid, name, val in sm_query:
                        if sid not in sample_metrics: sample_metrics[sid] = {}
                        sample_metrics[sid][name] = val
                sub_data['sample_metrics'] = sample_metrics
        
        result[sub.id] = sub_data
            
    # Now calculate global ranges for color-mapping
    # We include ALL processed submissions for this leaderboard to get correct global min/max
    processed_submissions = Submission.query.filter_by(
        leaderboard_id=leaderboard_id, 
        processing_status='Processed'
    ).all()
    
    # We need the same logic as in leaderboard_view to gather all metrics
    selected_metrics = [m for m in leaderboard.summary_metrics.split(',') if m.strip()]
    custom_metrics = set()
    for sub in processed_submissions:
        for cf in sub.custom_fields:
            if cf.field_type == 'metric':
                custom_metrics.add(cf.name)
    
    leaderboard_metrics_map = { (lm.target_name if lm.target_name else lm.global_metric.name): lm for lm in leaderboard.leaderboard_metrics }
    discovered_metrics = set(custom_metrics) | set(leaderboard_metrics_map.keys())
    all_metrics = list(selected_metrics)
    for m in sorted(list(discovered_metrics)):
        if m not in all_metrics:
            all_metrics.append(m)

    # Gather all values to calculate ranges
    metrics_ranges = {}
    
    # For directions
    metric_directions_dict = json.loads(leaderboard.metric_directions) if leaderboard.metric_directions else {}
    if leaderboard.leaderboard_metrics:
        for lm in leaderboard.leaderboard_metrics:
            target = lm.target_name if lm.target_name else lm.global_metric.name
            if lm.sort_direction:
                metric_directions_dict[target] = lm.sort_direction

    # We need to fetch/calculate values for ALL processed submissions if we want accurate global ranges
    # For now, let's just use the submissions we have in the current 'result' if it's too expensive?
    # No, the user wants "correctly displayed", which implies relative to the whole leaderboard.
    
    # Simple approach: fetch results for all processed
    all_sub_ids = [s.id for s in processed_submissions]
    all_results = MetricResult.query.filter(MetricResult.submission_id.in_(all_sub_ids)).all()
    
    all_values = {} # metric -> list of values
    for res in all_results:
        m_name = res.leaderboard_metric.target_name if res.leaderboard_metric.target_name else res.leaderboard_metric.global_metric.name
        if m_name not in all_values: all_values[m_name] = []
        if res.value is not None: all_values[m_name].append(res.value)
        
    for m in all_metrics:
        if m in AVAILABLE_METRICS:
            vals = [getattr(s, m) for s in processed_submissions if getattr(s, m) is not None]
            if m not in all_values: all_values[m] = []
            all_values[m].extend(vals)
        # Note: Custom metrics would require aggregating for ALL submissions here.
        # This might be slow. Optimization: only do this if any processed submission is a custom metric.
        # For now, let's focus on the submissions in the current request's result for ranges if they are custom.
    
    for m, vals in all_values.items():
        if vals:
             numeric_vals = [v for v in vals if isinstance(v, (int, float))]
             if numeric_vals:
                 metrics_ranges[m] = {'min': min(numeric_vals), 'max': max(numeric_vals)}

    return jsonify({
        'submissions': result,
        'ranges': metrics_ranges,
        'directions': metric_directions_dict
    })


@app.template_filter('from_json')
def from_json_filter(s):
    try:
        return json.loads(s)
    except:
        return {}

def check_and_migrate_db():
    """
    Checks for missing columns/tables (legacy support) and adds them if necessary.
    Handles:
    1. Leaderboard columns: scalar_width, image_width, last_sample_filter
    2. Project support: 'project' table, 'dataset.project_id', and migration to 'General' project.
    """
    print("Checking database schema for missing columns/tables...")
    with app.app_context():
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if db_uri.startswith('sqlite:///'):
            db_path = db_uri.replace('sqlite:///', '')
            if not os.path.isabs(db_path):
                 if os.path.basename(db_path) == db_path:
                     db_path = os.path.join(dtof_data_dir, db_path)
            
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # --- 1. Project Table migration ---
                # Check if 'project' table exists
                try:
                    cursor.execute("SELECT id FROM project LIMIT 1")
                except sqlite3.OperationalError:
                    print("Migrating DB: Creating 'project' table...")
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS project (
                            id INTEGER PRIMARY KEY,
                            name VARCHAR(100) NOT NULL UNIQUE,
                            description VARCHAR(255),
                            created_at DATETIME
                        )
                    ''')
                    conn.commit()
                    print("Created 'project' table.")

                # Helper to get/create General project
                def ensure_general_project():
                    cursor.execute("SELECT id FROM project WHERE name = 'General'")
                    res = cursor.fetchone()
                    if res:
                        return res[0]
                    print("Creating default 'General' project for orphaned/legacy data...")
                    cursor.execute("INSERT INTO project (name, description, created_at) VALUES (?, ?, ?)",
                                   ('General', 'Default project for legacy data', datetime.utcnow()))
                    conn.commit()
                    return cursor.lastrowid

                # --- 2. Dataset & Leaderboard Project ID migration ---
                tables_to_migrate = ['dataset', 'leaderboard']
                for table in tables_to_migrate:
                    needs_migration = False
                    try:
                        cursor.execute(f"SELECT project_id FROM {table} LIMIT 1")
                    except sqlite3.OperationalError:
                        print(f"Migrating DB: Adding 'project_id' to '{table}' table...")
                        try:
                            cursor.execute(f"ALTER TABLE {table} ADD COLUMN project_id INTEGER")
                            conn.commit()
                            needs_migration = True
                        except Exception as e:
                            print(f"Failed to add project_id to {table}: {e}")
                            continue
                    
                    # Also check for NULL values if column existed
                    if not needs_migration:
                        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE project_id IS NULL")
                        res = cursor.fetchone()
                        if res and res[0] > 0:
                            needs_migration = True
                    
                    if needs_migration:
                        gen_id = ensure_general_project()
                        print(f"Migrating orphaned records in '{table}' to Project '{gen_id}' (General)...")
                        cursor.execute(f"UPDATE {table} SET project_id = ? WHERE project_id IS NULL", (gen_id,))
                        conn.commit()

                # --- 3. Leaderboard Columns (Legacy) ---
                columns_to_check = {
                    'scalar_width': 'TEXT',
                    'image_width': 'TEXT',
                    'last_sample_filter': 'TEXT',
                    'metric_aggregation': 'TEXT DEFAULT "{}"',
                    'comparison_display_columns': 'TEXT DEFAULT "{}"' 
                }
                
                for col_name, col_type in columns_to_check.items():
                    try:
                        cursor.execute(f"SELECT {col_name} FROM leaderboard LIMIT 1")
                    except sqlite3.OperationalError:
                        print(f"Migrating DB: Adding missing column '{col_name}' to 'leaderboard' table...")
                        try:
                            cursor.execute(f"ALTER TABLE leaderboard ADD COLUMN {col_name} {col_type}")
                            conn.commit()
                            print(f"Successfully added '{col_name}'.")
                        except Exception as e:
                            print(f"Failed to add '{col_name}': {e}")
                            
                conn.close()
                
                # --- 4. LeaderboardMetric Aggregation Columns ---
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check for 'pooling_type' in leaderboard_metric
                cursor.execute("PRAGMA table_info(leaderboard_metric)")
                columns = [info[1] for info in cursor.fetchall()]

                
                if 'pooling_type' not in columns:
                    print("Migrating DB: Adding 'pooling_type' to 'leaderboard_metric'...")
                    try:
                        cursor.execute("ALTER TABLE leaderboard_metric ADD COLUMN pooling_type VARCHAR(20) DEFAULT 'mean' NOT NULL")
                        conn.commit()
                        print("Successfully added 'pooling_type'.")
                    except Exception as e:
                        print(f"Failed to add 'pooling_type': {e}")

                if 'pooling_percentile' not in columns:
                    print("Migrating DB: Adding 'pooling_percentile' to 'leaderboard_metric'...")
                    try:
                        cursor.execute("ALTER TABLE leaderboard_metric ADD COLUMN pooling_percentile FLOAT")
                        conn.commit()
                        print("Successfully added 'pooling_percentile'.")
                    except Exception as e:
                        print(f"Failed to add 'pooling_percentile': {e}")
                
                conn.close()

                print("Database schema check complete.")
            except Exception as e:
                print(f"Error during DB migration check: {e}")
        else:
            print("Non-SQLite database detected. Skipping auto-migration check.")


@app.route('/submission/<int:submission_id>/download_metrics')
def download_submission_metrics(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    
    # Create an in-memory CSV file
    si = io.StringIO()
    writer = csv.writer(si)
    
    # Write header
    writer.writerow(['Metric', 'Value'])
    
    # Fetch results joined with LeaderboardMetric to get display names
    results = db.session.query(MetricResult, LeaderboardMetric).join(
        LeaderboardMetric, MetricResult.leaderboard_metric_id == LeaderboardMetric.id
    ).filter(MetricResult.submission_id == submission_id).all()
    
    for result, lb_metric in results:
        # Determine display name (same logic as in template)
        display_name = lb_metric.target_name or lb_metric.global_metric.label or lb_metric.global_metric.name
        
        # Format value
        value = result.value
        if value is None:
            value_str = 'N/A'
            if result.error_message:
                value_str = f"Error: {result.error_message}"
        else:
            value_str = f"{value:.6f}"
            
        writer.writerow([display_name, value_str])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=submission_{submission_id}_aggregated_metrics.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/submission/<int:submission_id>/download_sample_metrics_csv')
def download_sample_metrics_csv(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    leaderboard = submission.leaderboard
    
    # Get all samples for this dataset
    samples = Sample.query.filter_by(dataset_id=leaderboard.dataset_id).order_by(Sample.name).all()
    sample_names = [s.name for s in samples]
    sample_ids = [s.id for s in samples]
    
    # Get metric display names
    lb_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=leaderboard.id).all()
    metric_names = []
    metric_display_map = {}
    for lm in lb_metrics:
        if lm.global_metric.is_aggregated:
            continue
        display_name = lm.target_name or lm.global_metric.label or lm.global_metric.name
        metric_names.append(display_name)
        metric_display_map[display_name] = (lm.target_name if lm.target_name else lm.global_metric.name)

    # Fetch per-sample metrics from CustomField
    # Rows: Metrics, Columns: Samples
    si = io.StringIO()
    writer = csv.writer(si)
    
    # Header: Metric, Sample1, Sample2, ...
    writer.writerow(['Metric'] + sample_names)
    
    # Data rows
    for m_display in metric_names:
        m_raw = metric_display_map[m_display]
        row = [m_display]
        # Fetch all custom fields for this metric and submission at once for efficiency
        cfs = CustomField.query.filter_by(submission_id=submission_id, name=m_raw, field_type='scalar').all()
        cf_map = {cf.sample_id: cf.value_float for cf in cfs}
        
        for s_id in sample_ids:
            val = cf_map.get(s_id)
            row.append(f"{val:.6f}" if val is not None else 'N/A')
        writer.writerow(row)
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=submission_{submission_id}_per_sample_metrics.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/leaderboard/<int:leaderboard_id>/download_sample_metrics_bulk', methods=['POST'])
def download_submissions_sample_metrics_bulk(leaderboard_id):
    submission_ids = request.form.getlist('submission_ids')
    redirect_args = {
        'project_name': g.current_project.name if g.get('current_project') else 'dTOF',
        'leaderboard_id': leaderboard_id
    }
    
    if not submission_ids:
        flash("No submissions selected.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))
        
    submissions = Submission.query.filter(Submission.id.in_(submission_ids), Submission.leaderboard_id == leaderboard_id).all()
    
    if not submissions:
        flash("No valid submissions found.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))

    import tempfile
    from zipfile import ZipFile
    
    tmp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        with ZipFile(tmp_path, 'w') as zf:
            for sub in submissions:
                # Reuse the logic from download_sample_metrics_csv but write to zip
                samples = Sample.query.filter_by(dataset_id=sub.leaderboard.dataset_id).order_by(Sample.name).all()
                sample_names = [s.name for s in samples]
                sample_ids = [s.id for s in samples]
                
                lb_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=sub.leaderboard_id).all()
                metric_names = []
                metric_display_map = {}
                for lm in lb_metrics:
                    if lm.global_metric.is_aggregated:
                        continue
                    display_name = lm.target_name or lm.global_metric.label or lm.global_metric.name
                    metric_names.append(display_name)
                    metric_display_map[display_name] = (lm.target_name if lm.target_name else lm.global_metric.name)

                si = io.StringIO()
                writer = csv.writer(si)
                writer.writerow(['Metric'] + sample_names)
                
                for m_display in metric_names:
                    m_raw = metric_display_map[m_display]
                    row = [m_display]
                    cfs = CustomField.query.filter_by(submission_id=sub.id, name=m_raw, field_type='scalar').all()
                    cf_map = {cf.sample_id: cf.value_float for cf in cfs}
                    for s_id in sample_ids:
                        val = cf_map.get(s_id)
                        row.append(f"{val:.6f}" if val is not None else 'N/A')
                    writer.writerow(row)
                
                zf.writestr(f"{sub.name}_{sub.id}_per_sample_metrics.csv", si.getvalue())
                
    except Exception as e:
        if os.path.exists(tmp_path):
             os.remove(tmp_path)
        flash(f"Error creating bulk sample metrics zip: {e}", "danger")
        return redirect(url_for('leaderboard_view', **redirect_args))
    
    @after_this_request
    def remove_file(response):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as error:
            app.logger.error("Error removing temp bulk sample zip", error)
        return response

    return send_file(tmp_path, as_attachment=True, download_name="bulk_per_sample_metrics.zip")
@app.route('/leaderboard/<int:leaderboard_id>/download_metrics_bulk', methods=['POST'])
def download_submissions_metrics_bulk(leaderboard_id):
    submission_ids = request.form.getlist('submission_ids')
    redirect_args = {
        'project_name': g.current_project.name if g.get('current_project') else 'dTOF', # Fallback
        'leaderboard_id': leaderboard_id
    }
    # Add other filter args for redirect if needed, similar to existing bulk download
    
    if not submission_ids:
        flash("No submissions selected.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))
        
    submissions = Submission.query.filter(Submission.id.in_(submission_ids), Submission.leaderboard_id == leaderboard_id).all()
    
    if not submissions:
        flash("No valid submissions found.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))

    import tempfile
    from zipfile import ZipFile
    
    tmp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        with ZipFile(tmp_path, 'w') as zf:
            for sub in submissions:
                # Generate CSV
                si = io.StringIO()
                writer = csv.writer(si)
                writer.writerow(['Metric', 'Value'])
                
                results = db.session.query(MetricResult, LeaderboardMetric).join(
                    LeaderboardMetric, MetricResult.leaderboard_metric_id == LeaderboardMetric.id
                ).filter(MetricResult.submission_id == sub.id).all()
                
                for result, lb_metric in results:
                    display_name = lb_metric.target_name or lb_metric.global_metric.label or lb_metric.global_metric.name
                    value = result.value
                    if value is None:
                        value_str = 'N/A'
                        if result.error_message:
                            value_str = f"Error: {result.error_message}"
                    else:
                        value_str = str(value)
                    writer.writerow([display_name, value_str])
                
                # Add to zip
                zf.writestr(f"{sub.name}_{sub.id}_metrics.csv", si.getvalue())
                
    except Exception as e:
        if os.path.exists(tmp_path):
             os.remove(tmp_path)
        flash(f"Error creating bulk metrics zip: {e}", "danger")
        return redirect(url_for('leaderboard_view', **redirect_args))
    
    @after_this_request
    def remove_file(response):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as error:
            app.logger.error("Error removing temp bulk zip", error)
        return response

    return send_file(tmp_path, as_attachment=True, download_name="bulk_metrics.zip")

@app.route('/leaderboard/<int:leaderboard_id>/download_full_bulk', methods=['POST'])
def download_submissions_full_bulk(leaderboard_id):
    submission_ids = request.form.getlist('submission_ids')
    redirect_args = {
        'project_name': g.current_project.name if g.get('current_project') else 'dTOF',
        'leaderboard_id': leaderboard_id
    }
    
    if not submission_ids:
        flash("No submissions selected.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))
        
    submissions = Submission.query.filter(Submission.id.in_(submission_ids), Submission.leaderboard_id == leaderboard_id).all()
    
    if not submissions:
        flash("No valid submissions found.", "warning")
        return redirect(url_for('leaderboard_view', **redirect_args))

    import tempfile
    from zipfile import ZipFile
    
    tmp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        with ZipFile(tmp_path, 'w') as zf:
            for sub in submissions:
                # 1. Add Original Zip
                sub_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(sub.id))
                sub_zip_path = os.path.join(sub_folder, 'submission.zip')
                if os.path.exists(sub_zip_path):
                    zf.write(sub_zip_path, arcname=f"{sub.name}_{sub.id}.zip")
                
                # 2. Add Metrics CSV
                si = io.StringIO()
                writer = csv.writer(si)
                writer.writerow(['Metric', 'Value'])
                
                results = db.session.query(MetricResult, LeaderboardMetric).join(
                    LeaderboardMetric, MetricResult.leaderboard_metric_id == LeaderboardMetric.id
                ).filter(MetricResult.submission_id == sub.id).all()
                
                for result, lb_metric in results:
                    display_name = lb_metric.target_name or lb_metric.global_metric.label or lb_metric.global_metric.name
                    value = result.value
                    if value is None:
                        value_str = 'N/A'
                        if result.error_message:
                            value_str = f"Error: {result.error_message}"
                    else:
                        value_str = f"{value:.6f}"
                    writer.writerow([display_name, value_str])
                
                zf.writestr(f"{sub.name}_{sub.id}_aggregated_metrics.csv", si.getvalue())

                # 3. Add Per-Sample Metrics CSV
                samples = Sample.query.filter_by(dataset_id=sub.leaderboard.dataset_id).order_by(Sample.name).all()
                sample_names = [s.name for s in samples]
                sample_ids = [s.id for s in samples]
                
                lb_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=sub.leaderboard_id).all()
                metric_names = []
                metric_display_map = {}
                for lm in lb_metrics:
                    if lm.global_metric.is_aggregated:
                        continue
                    display_name = lm.target_name or lm.global_metric.label or lm.global_metric.name
                    metric_names.append(display_name)
                    metric_display_map[display_name] = (lm.target_name if lm.target_name else lm.global_metric.name)

                si_ps = io.StringIO()
                writer_ps = csv.writer(si_ps)
                writer_ps.writerow(['Metric'] + sample_names)
                
                for m_display in metric_names:
                    m_raw = metric_display_map[m_display]
                    row = [m_display]
                    cfs = CustomField.query.filter_by(submission_id=sub.id, name=m_raw, field_type='scalar').all()
                    cf_map = {cf.sample_id: cf.value_float for cf in cfs}
                    for s_id in sample_ids:
                        val = cf_map.get(s_id)
                        row.append(f"{val:.6f}" if val is not None else 'N/A')
                    writer_ps.writerow(row)
                
                zf.writestr(f"{sub.name}_{sub.id}_per_sample_metrics.csv", si_ps.getvalue())
                
    except Exception as e:
        if os.path.exists(tmp_path):
             os.remove(tmp_path)
        flash(f"Error creating bulk zip: {e}", "danger")
        return redirect(url_for('leaderboard_view', **redirect_args))
    
    @after_this_request
    def remove_file(response):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as error:
            app.logger.error("Error removing temp bulk zip", error)
        return response

    return send_file(tmp_path, as_attachment=True, download_name="bulk_submissions_full.zip")
if __name__ == '__main__':
    with app.app_context():
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # Check and migrate DB before creating/accessing
        check_and_migrate_db()
        db.create_all()

    app.run(host='0.0.0.0', port=6060, debug=True)
