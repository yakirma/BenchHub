import shutil
import urllib.parse
from urllib.parse import quote
import sys
import re
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, flash, abort, after_this_request, g, make_response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import zipfile
from datetime import datetime, timedelta
from celery import Celery
import json
import math
import numpy as np
import threading
# For loading npz files
from scipy.optimize import curve_fit
from sqlalchemy import or_, and_, not_, func
import io
import csv
import secrets
import tempfile
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from metric_engine import evaluate_dynamic_metric, get_metric_context, sort_metrics_by_dependency
import subprocess
from functools import wraps
from authlib.integrations.flask_client import OAuth

# Import local configuration (optional, with fallback)
try:
    from local_config import GIT_REPO_PATH
except ImportError:
    GIT_REPO_PATH = None  # Fallback to None if config doesn't exist

def get_author_from_git_commit(commit_hash, repo_path=None, branch_name=None):
    """
    Extract author name from a git branch using remote origin.
    Uses ONLY branch name as requested.
    Uses: git log -1 --format='%an' origin/BRANCH_NAME
    
    Args:
        commit_hash: (Ignored) Kept for compatibility.
        repo_path: Optional path to git repository.
        branch_name: The git branch name to query (REQUIRED)
    
    Returns:
        Author name string or None if not found
    """
    if not branch_name or branch_name == 'N/A':
        return None
    
    git_path = repo_path or GIT_REPO_PATH
    
    try:
        # Fetch from remote first to ensure we have latest data
        fetch_cmd = ['git', 'fetch', 'origin']
        if git_path:
            fetch_cmd = ['git', '-C', git_path, 'fetch', 'origin']
        
        subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=30)
        
        # Get author from remote branch
        cmd = ['git', 'log', '-1', '--format=%an', f'origin/{branch_name}']
        if git_path:
            cmd = ['git', '-C', git_path, 'log', '-1', '--format=%an', f'origin/{branch_name}']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            author = result.stdout.strip()
            if author:
                return author
        
        print(f"Warning: Could not find author for branch '{branch_name}' from remote")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout while extracting git author from branch '{branch_name}'")
        return None
    except Exception as e:
        print(f"Warning: Could not extract git author from branch '{branch_name}': {e}")
        return None



def format_tag_value(val):
    """
    Parses and formats a tag value string into its appropriate type (int, float, bool as 0/1, or str).
    Returns 'N/A' if val is None.
    """
    if val is None:
        return 'N/A'
    v_lower = str(val).lower().strip()
    if v_lower in ('true', 'yes'):
        return 1
    if v_lower in ('false', 'no'):
        return 0
    # Try numeric conversion
    try:
        f_val = float(val)
        if math.isfinite(f_val):
            if f_val == int(f_val):
                return int(f_val)
            return f"{f_val:.6f}"
    except (ValueError, TypeError, OverflowError):
        pass
    return val

def get_distinguishable_metric_name(lm):
    """
    Constructs a metric name for CSV exports as <metric_name>_<metric_id>.
    """
    base_name = lm.target_name or lm.global_metric.label or lm.global_metric.name
    # Keep it clean as per user request
    return f"{base_name}_{lm.id}"


app = Flask(__name__)
__version__ = "1.0.0"
app.secret_key = os.environ.get('SECRET_KEY') or 'supersecretkey'  # Override in prod via SECRET_KEY env

# Honor X-Forwarded-Proto / X-Forwarded-Host from the Fly + Cloudflare edges
# so url_for(_external=True) produces https:// URLs (and the right host) when
# the container itself only sees plain HTTP. Without this the GitHub OAuth
# redirect_uri ends up http://, GitHub rejects it as "not associated with
# this application."
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
# basedir = os.path.abspath(os.path.dirname(__file__)) # No longer used for data
user_home = os.path.expanduser("~")
dtof_data_dir = os.environ.get('BENCHHUB_DATA_DIR') or os.path.join(user_home, ".dtofbenchmarking")

# Ensure data directory exists
if not os.path.exists(dtof_data_dir):
    os.makedirs(dtof_data_dir, exist_ok=True)
    print(f"Created data directory at: {dtof_data_dir}")
else:
    print(f"Using data directory at: {dtof_data_dir}")

def _ensure_redis_ssl_param(url):
    """Celery 5+ refuses rediss:// URLs without an ssl_cert_reqs parameter
    (raises "A rediss:// URL must have parameter ssl_cert_reqs..."). When
    the secret comes from Upstash as a bare connection string, append it
    here so the URL parser picks it up natively — sidesteps the
    BROKER_USE_SSL / redis_backend_use_ssl naming dance and the Celery 5
    old/new key mix-check."""
    import urllib.parse as _urlparse
    if not url or not url.startswith('rediss://'):
        return url
    parsed = _urlparse.urlparse(url)
    qs = dict(_urlparse.parse_qsl(parsed.query, keep_blank_values=True))
    qs.setdefault('ssl_cert_reqs', 'CERT_REQUIRED')
    return parsed._replace(query=_urlparse.urlencode(qs)).geturl()


_redis_url = _ensure_redis_ssl_param(os.environ.get('REDIS_URL') or 'redis://localhost:6379/0')
_celery_broker = _ensure_redis_ssl_param(os.environ.get('CELERY_BROKER_URL') or _redis_url)
_celery_backend = _ensure_redis_ssl_param(os.environ.get('CELERY_RESULT_BACKEND') or _redis_url)

app.config.update(
    SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(dtof_data_dir, 'database.db'),
    UPLOAD_FOLDER=os.path.join(dtof_data_dir, 'uploads'),
    CELERY_BROKER_URL=_celery_broker,
    CELERY_RESULT_BACKEND=_celery_backend,
    SQLALCHEMY_ENGINE_OPTIONS={'connect_args': {'timeout': 120}},  # 120 seconds timeout
)

# Enable Write-Ahead Logging (WAL) for better concurrency
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=120000") # 120 seconds
    cursor.close()

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --- OAuth (Phase 1 multi-tenancy) ---
# Authlib client config. GITHUB_CLIENT_ID/SECRET come from `fly secrets set`
# in prod and from a local .env (or your shell) in dev. Missing creds means
# the /login/github route will return a 503 — the rest of the app keeps
# working, so local dev without OAuth set up is fine.
oauth = OAuth(app)
oauth.register(
    name='github',
    client_id=os.environ.get('GITHUB_CLIENT_ID'),
    client_secret=os.environ.get('GITHUB_CLIENT_SECRET'),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'read:user user:email'},
)


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

# Discovery tags (separate concept from per-submission tags). Reuses the
# Tag table — same string namespace, so a "depth" tag on a dataset and
# a "depth" tag on a submission are the same row.
dataset_tags = db.Table('dataset_tags',
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True),
)
leaderboard_tags = db.Table('leaderboard_tags',
    db.Column('leaderboard_id', db.Integer, db.ForeignKey('leaderboard.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True),
)

# Per-row collaborators: owner can grant other users access to a private
# dataset / leaderboard. Visibility checks honor the share so 'private'
# stops being binary owner-only.
dataset_shares = db.Table('dataset_shares',
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id'), primary_key=True),
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
)
leaderboard_shares = db.Table('leaderboard_shares',
    db.Column('leaderboard_id', db.Integer, db.ForeignKey('leaderboard.id'), primary_key=True),
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
)

# Association Table for Leaderboards and Datasets
leaderboard_datasets = db.Table('leaderboard_datasets',
    db.Column('leaderboard_id', db.Integer, db.ForeignKey('leaderboard.id'), primary_key=True),
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id'), primary_key=True)
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
    git_author = db.Column(db.String(100))  # Git commit author
    display_columns = db.Column(db.String(500), nullable=False, default=DEFAULT_DATASET_DISPLAY_COLUMNS)
    visualizations = db.Column(db.String(500), nullable=False, default='') # Active visualizers
    selected_metrics = db.Column(db.String(500), nullable=False, default='') # No default metrics
    # Phase 1 multi-tenancy.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    # server_default (not just `default`) so raw SQL INSERTs in the legacy
    # migration code don't trip the NOT NULL constraint.
    visibility = db.Column(db.String(20), nullable=False, default='public', server_default='public')
    # Phase 5: BenchHub-curated content marker. True for the seeded official
    # datasets that should appear in the landing-page "Curated benchmarks"
    # Phase 7: cached storage usage (bytes) summed across uploads/datasets/<id>.
    # Updated alongside file writes/deletes so quota checks don't have to du
    # the volume on every request.
    storage_bytes = db.Column(db.BigInteger, nullable=False, default=0, server_default='0')
    owner = db.relationship('User', foreign_keys=[owner_user_id])
    # leaderboards = db.relationship('Leaderboard', backref='dataset', lazy=True, cascade="all, delete-orphan") # Deprecated: use many-to-many
    samples = db.relationship('Sample', backref='dataset', lazy=True, cascade="all, delete-orphan")
    tags = db.relationship('Tag', secondary=dataset_tags, lazy='subquery',
                           backref=db.backref('datasets', lazy=True))
    collaborators = db.relationship('User', secondary=dataset_shares, lazy='subquery',
                                    backref=db.backref('shared_datasets', lazy=True))

    # Provenance for re-import. source_kind ∈ {zip, hf-bench, hf-parquet, hf-webdataset}.
    # source_metadata stashes the {repo_id, revision, mapping, sample_cap} so a
    # later "Refresh from HF" can replay the import deterministically.
    source_kind = db.Column(db.String(32), nullable=True)
    source_metadata = db.Column(db.Text, nullable=True)

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
    # Phase 1 Slice 4 multi-tenancy. NOTE: `name` is still globally unique;
    # two users can't both ship a metric called "L1Loss". For Phase 1 that's
    # acceptable (rename your fork). Composite (owner, name) uniqueness is a
    # bigger schema change deferred to a later slice.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    visibility = db.Column(db.String(20), nullable=False, default='public', server_default='public')
    owner = db.relationship('User', foreign_keys=[owner_user_id])
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

    # Per-sample Filtering
    tag_filter = db.Column(db.Text, nullable=True) # comma-separated tags

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
    # Phase 1 Slice 4 multi-tenancy.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    visibility = db.Column(db.String(20), nullable=False, default='public', server_default='public')
    owner = db.relationship('User', foreign_keys=[owner_user_id])

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
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True) # Deprecated: migration to leaderboard_datasets
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
    # Phase 1 multi-tenancy.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    visibility = db.Column(db.String(20), nullable=False, default='public', server_default='public')
    owner = db.relationship('User', foreign_keys=[owner_user_id])
    submissions = db.relationship('Submission', backref='leaderboard', lazy=True, cascade="all, delete-orphan")
    datasets = db.relationship('Dataset', secondary=leaderboard_datasets, backref=db.backref('leaderboards', lazy='dynamic'))
    tags = db.relationship('Tag', secondary=leaderboard_tags, lazy='subquery',
                           backref=db.backref('leaderboards', lazy=True))
    collaborators = db.relationship('User', secondary=leaderboard_shares, lazy='subquery',
                                    backref=db.backref('shared_leaderboards', lazy=True))


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=False)
    git_commit = db.Column(db.String(100))
    git_branch = db.Column(db.String(100))
    git_message = db.Column(db.String(200))
    git_author = db.Column(db.String(100))  # Git commit author
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_archived = db.Column(db.Boolean, default=False, nullable=False)
    processing_status = db.Column(db.String(50), default='Pending')
    last_sample_filter = db.Column(db.Text, nullable=True) # JSON store of filters used for metrics
    # Phase 1 multi-tenancy. Submissions don't get their own visibility — they
    # inherit the leaderboard's. Owner is whoever uploaded the submission.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    owner = db.relationship('User', foreign_keys=[owner_user_id])
    tags = db.relationship('Tag', secondary=submission_tags, lazy='subquery', backref=db.backref('submissions', lazy=True))
    custom_fields = db.relationship('CustomField', backref='submission', lazy=True, cascade="all, delete-orphan", foreign_keys='CustomField.submission_id')

class AuthorProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False) # The git_author string
    display_name = db.Column(db.String(100), nullable=True)
    avatar_filename = db.Column(db.String(255), nullable=True)
    merged_into_username = db.Column(db.String(100), nullable=True) # Username this user is merged into
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    """Authenticated account. Phase 1 of the public-web rollout — every other
    multi-tenant feature (owner_user_id FKs, visibility, quotas) hangs off this.
    OAuth-only by design: no password column, no signup form."""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(120))
    avatar_url = db.Column(db.String(500))
    oauth_provider = db.Column(db.String(20), nullable=False)  # 'github', later 'google'
    oauth_sub = db.Column(db.String(120), nullable=False)      # provider's user id
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = db.Column(db.DateTime)

    # Phase 7 quotas. Caps are server-side enforced before disk writes happen
    # (so a malicious user can't fill the volume). Defaults below are the
    # free-tier values; bump per-user when you launch a paid tier. NULL caps
    # are allowed → "unlimited" (used for the `system` curated-content user).
    quota_max_storage_bytes = db.Column(
        db.BigInteger, nullable=False, default=200 * 1024 * 1024,
        server_default=str(200 * 1024 * 1024),
    )  # 200 MB
    quota_max_datasets = db.Column(
        db.Integer, nullable=False, default=5, server_default='5',
    )
    quota_max_submissions_per_day = db.Column(
        db.Integer, nullable=False, default=50, server_default='50',
    )

    # Phase 8: API token for programmatic access. Stored verbatim (not
    # hashed) — by design: the user views/copies it from the settings
    # page, so we need to be able to display it. Treat the DB as
    # secret-bearing. Rotate via the regenerate endpoint, which writes
    # a new value and invalidates the old one (no grace window).
    api_token = db.Column(db.String(64), unique=True, nullable=True, index=True)

    # Admin bit. Two ways to be an admin:
    # 1. Email is on the BENCHHUB_ADMIN_EMAILS env-var allow-list
    #    (immutable from the running app — set via Fly secret).
    # 2. This DB-backed flag (mutable from /settings/admins by another
    #    admin). Bootstrapped from the env-var allow-list at first login.
    is_admin = db.Column(db.Boolean, nullable=False, default=False, server_default='0')

    __table_args__ = (
        db.UniqueConstraint('oauth_provider', 'oauth_sub', name='uq_user_oauth_identity'),
    )


def get_canonical_username(username, profiles):
    """Resolve a username to its canonical identity by following merge chains."""
    visited = set()
    curr = username
    while curr in profiles and profiles[curr].merged_into_username:
        if profiles[curr].merged_into_username in visited:  # Cycle detection
            break
        visited.add(curr)
        curr = profiles[curr].merged_into_username
    return curr


# Initialize Celery after models are defined
celery = make_celery(app)

# Import tasks to register them with Celery
import tasks  # noqa: F401

@app.context_processor
def inject_version():
    profiles = AuthorProfile.query.all()
    mapping = {p.username: {
        'display_name': p.display_name,
        'avatar_url': url_for('serve_author_avatar', filename=p.avatar_filename) if p.avatar_filename else None,
        'merged_into': p.merged_into_username
    } for p in profiles}
    return dict(version=__version__, author_profiles_json=json.dumps(mapping))


def process_dataset_zip(zip_path, dataset_name, owner_user_id=None):
    """
    Helper to process a dataset zip file and create database entries.
    Name collisions are rejected — users must delete the existing dataset
    explicitly. owner_user_id (Phase 1 multi-tenancy) is the User who
    uploaded; None means "legacy / scripted" upload.
    Returns (success: bool, message: str, dataset_id: int or None)
    """
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_dataset_extract_' + datetime.now().strftime('%Y%m%d%H%M%S%f'))
    os.makedirs(temp_dir, exist_ok=True)

    # Track the prelim Dataset row so all failure paths can clean it up.
    # The row is committed early (to grab an id and release the write lock)
    # which means anything failing mid-extraction would otherwise leave an
    # orphan tying up the user's quota until the periodic prune.
    prelim_dataset = None

    def _cleanup_prelim():
        """Remove the prelim Dataset row + any partial folder. Idempotent."""
        if prelim_dataset is None:
            return
        try:
            db.session.rollback()
            detached = Dataset.query.get(prelim_dataset.id)
            if detached is not None:
                folder = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'datasets',
                    secure_filename(detached.name),
                )
                if os.path.isdir(folder):
                    shutil.rmtree(folder, ignore_errors=True)
                db.session.delete(detached)
                db.session.commit()
        except Exception as cleanup_err:
            db.session.rollback()
            print(f"process_dataset_zip self-cleanup failed: {cleanup_err}")

    try:
        # Initial collision check.
        existing = Dataset.query.filter_by(name=dataset_name).first()
        if existing:
            return False, f"Dataset '{dataset_name}' already exists.", None

        # Create preliminary entry
        new_dataset = Dataset(name=dataset_name, owner_user_id=owner_user_id)
        db.session.add(new_dataset)
        db.session.commit() # Commit immediately to release lock and get ID
        prelim_dataset = new_dataset

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
                    _cleanup_prelim()
                    return False, f"Dataset '{real_dataset_name}' (extracted from ZIP) already exists.", None
                new_dataset.name = real_dataset_name
                db.session.add(new_dataset)
                db.session.commit()

            # Permanent storage setup
            dataset_folder_name = secure_filename(new_dataset.name)
            dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Copy original ZIP
            original_zip_dest = os.path.join(dataset_dir, f"{dataset_folder_name}.zip")
            shutil.copy2(zip_path, original_zip_dest)

            # Git metadata
            git_info_path = os.path.join(dataset_content_path, 'git_info.json')
            if not os.path.exists(git_info_path):
                git_info_path = os.path.join(dataset_content_path, 'git.info')
            
            if os.path.exists(git_info_path):
                try:
                    with open(git_info_path, 'r') as git_file:
                        git_data = json.load(git_file)
                        new_dataset.git_commit = git_data.get('commit', '')
                        new_dataset.git_branch = git_data.get('branch', '')
                        new_dataset.git_message = git_data.get('message', '')
                        # Extract author from git_info.json or from git commit
                        author = git_data.get('author', '')
                        if not author and new_dataset.git_commit:
                            # Try to extract author from git using the commit hash
                            author = get_author_from_git_commit(new_dataset.git_commit, branch_name=new_dataset.git_branch)
                        new_dataset.git_author = author or ''
                except Exception: pass

            # Discover samples - scan all folders dynamically.
            # Strip trailing _<W>x<H> dimension suffix in raw_ depth folders;
            # the convention is `<sample>_<W>x<H>.npz`, so the bare
            # extension-strip would otherwise turn a depth file into a
            # phantom sample with no other fields.
            sample_names = set()
            _depth_dim_re = re.compile(r'^(.*)_\d+x\d+$')
            for folder_name in os.listdir(dataset_content_path):
                folder_path = os.path.join(dataset_content_path, folder_name)
                if os.path.isdir(folder_path) and folder_name not in ['__MACOSX', 'git.info']:
                    is_raw_depth = folder_name.startswith('raw_')
                    for fname in os.listdir(folder_path):
                        base = os.path.splitext(fname)[0]
                        if is_raw_depth:
                            m = _depth_dim_re.match(base)
                            if m:
                                base = m.group(1)
                        sample_names.add(base)
            
            if not sample_names:
                _cleanup_prelim()
                return False, "No valid samples (hist, config, etc.) found in ZIP.", None

            # Create sample records
            for s_name in sample_names:
                if s_name == '.DS_Store': continue
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
            
            # Phase 7: cache the on-disk size so quota checks don't du.
            new_dataset.storage_bytes = _path_size_bytes(dataset_dir)
            db.session.commit()
            return True, f"Uploaded '{new_dataset.name}' ({len(sample_names)} samples)", new_dataset.id

    except Exception as e:
        # Crash mid-extraction → roll back the prelim row + any partial folder
        # via the same cleanup the early-return paths use.
        _cleanup_prelim()
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
                except Exception:
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
             return res
         return 0.0

    # General approach: Find any folder starting with "hist_" that has an .npz for this sample
    if os.path.exists(submission_folder):
        for folder_name in os.listdir(submission_folder):
            folder_path = os.path.join(submission_folder, folder_name)
            # Strict check for 'hist_' prefix for dynamic ones, plus legacy 'raw_histogram'
            if os.path.isdir(folder_path) and (folder_name.startswith('hist_') or folder_name == 'raw_histogram'):
                hist_file = os.path.join(folder_path, f'{sample.name}.npz')
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
    for cf in custom_fields_for_sample:
        if cf.field_type == 'image':
            # Store image path
            pred_data[cf.name] = cf.value_text
        elif cf.field_type in ['scalar', 'metric']:
            # Store scalar/metric value
            pred_data[cf.name] = cf.value_float
            metrics[cf.name] = cf.value_float
            
            # [FIX] If the field name is lm_{id}, also register it under its friendly name for backward compatibility
            # and to satisfy consumers expecting friendly names.
            if cf.name.startswith('lm_'):
                try:
                    lm_id = int(cf.name[3:])
                    lm = LeaderboardMetric.query.get(lm_id)
                    if lm:
                        friendly_name = lm.target_name or lm.global_metric.name
                        metrics[friendly_name] = cf.value_float
                        pred_data[friendly_name] = cf.value_float
                except Exception: pass

    # [FIX] strict mode cleanup:
    # If the submission uses lm_{id} for a metric (globally), ensure we don't leak raw/legacy values 
    # for samples where that metric was filtered out (and thus lm_{id} is missing).
    if metrics:
        submission_lm_ids = {
            int(cf.name[3:]) for cf in submission.custom_fields 
            if cf.name.startswith('lm_') and cf.name[3:].isdigit()
        }
        
        if submission_lm_ids:
            lb_metrics = submission.leaderboard.leaderboard_metrics
            # Map friendly name to list of IDs (support multiple flavours)
            friendly_to_ids = {}
            for lm in lb_metrics:
                name = lm.target_name or lm.global_metric.name
                if name not in friendly_to_ids:
                    friendly_to_ids[name] = []
                friendly_to_ids[name].append(lm.id)
            
            for key in list(metrics.keys()):
                if key in friendly_to_ids:
                    # Check if ANY of the flavours for this name are tracked by this submission (New Mode)
                    tracked_ids = [mid for mid in friendly_to_ids[key] if mid in submission_lm_ids]
                    
                    if tracked_ids:
                        # If tracked, we expect at least one of the tracked lm_{id}s to be present in the sample metrics
                        # If NONE are present, it implies all flavours were filtered out for this sample.
                        # In that case, we remove the friendly name (which is likely a raw data leak).
                        any_flavour_present = any(f'lm_{mid}' in metrics for mid in tracked_ids)
                        
                        if not any_flavour_present:
                             del metrics[key]
                             if key in pred_data:
                                 del pred_data[key]

    return pred_data, metrics

# --- Global Settings Management ---
SETTINGS_FILE = os.path.join(dtof_data_dir, 'global_settings.json')

class GlobalSettings:
    def __init__(self):
        self.defaults = {
            'scalar_width': '150px',
            'image_width': '300px',
            'theme_mode': 'dark'
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
        return self.settings.get('theme_mode', 'dark')

global_settings = GlobalSettings()

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
        theme_mode = request.form.get('theme_mode', 'dark')
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


# --- End Global Settings ---

# (Removed in projects-removal refactor:
#   - load_project_context @before_request
#   - pull_project_name @url_value_preprocessor
#   - add_project_name @url_defaults
#   - is_endpoint_expecting helper + werkzeug Map monkey-patch
# URLs are no longer prefixed with /<project_name>/.)


# ===================== Authentication routes =====================

@app.before_request
def load_current_user():
    """Populate g.current_user from session on every request. None if anonymous."""
    user_id = session.get('user_id')
    g.current_user = User.query.get(user_id) if user_id else None


@app.context_processor
def inject_current_user():
    """Make current_user (and current_user_is_admin) available in every
    Jinja template."""
    user = getattr(g, 'current_user', None)
    return {
        'current_user': user,
        'current_user_is_admin': is_admin(user),
    }


def login_required(view):
    """Redirect anonymous users to /login, preserving the intended path so the
    callback can bounce them back. Use on any route that needs an account."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not getattr(g, 'current_user', None):
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)
    return wrapped


def owner_required(model_cls, id_kwarg='id'):
    """Gate a route to the owner of a row.

    Usage::
        @app.route('/foo/<int:foo_id>/edit', methods=['POST'])
        @login_required
        @owner_required(Foo, 'foo_id')
        def edit_foo(foo_id):
            ...

    Order matters: stack `@login_required` ABOVE `@owner_required` so anon
    users get the login redirect before we look up the row.

    Rules:
    - row not found        -> 404
    - row.owner_user_id is NULL (legacy data, pre-migration) -> allow
    - row.owner_user_id matches g.current_user.id            -> allow
    - anything else                                          -> 403
    """
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            row_id = kwargs.get(id_kwarg)
            if row_id is None:
                abort(404)
            row = model_cls.query.get(row_id)
            if row is None:
                abort(404)
            owner_id = getattr(row, 'owner_user_id', None)
            current = getattr(g, 'current_user', None)
            if owner_id is None:
                # Legacy / unowned. Permit for now; the eventual backfill
                # plan assigns an owner so this branch becomes unreachable.
                return view(*args, **kwargs)
            # Admin (BENCHHUB_ADMIN_EMAILS) bypass — staff can act on
            # anything, e.g. delete abusive content.
            if is_admin(current):
                return view(*args, **kwargs)
            if current is None or current.id != owner_id:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def visible_in_list(model_cls, user):
    """SQL filter: rows that should appear in a *list* page.

    Show:
    - rows with visibility='public'
    - legacy NULL-owner rows (treated as public until backfill)
    - **owner's own non-unlisted rows** (so I can find my private stuff)
    - **rows shared with the user** (collaborator on a private row)

    Hide:
    - unlisted (URL-only by design — even from the owner's list pages)
    - other users' private rows the user isn't a collaborator on

    Use as ``Model.query.filter(visible_in_list(Model, g.current_user))``."""
    public_or_legacy = or_(
        model_cls.visibility == 'public',
        model_cls.owner_user_id.is_(None),
    )
    if user is None:
        return public_or_legacy

    # Collaborator share-table lookup, model-aware.
    share_clause = None
    if model_cls is Dataset:
        share_clause = db.session.query(dataset_shares.c.user_id).filter(
            dataset_shares.c.dataset_id == model_cls.id,
            dataset_shares.c.user_id == user.id,
        ).exists()
    elif model_cls is Leaderboard:
        share_clause = db.session.query(leaderboard_shares.c.user_id).filter(
            leaderboard_shares.c.leaderboard_id == model_cls.id,
            leaderboard_shares.c.user_id == user.id,
        ).exists()

    own_clause = and_(
        model_cls.owner_user_id == user.id,
        model_cls.visibility != 'unlisted',
    )
    if share_clause is not None:
        return or_(public_or_legacy, own_clause, share_clause)
    return or_(public_or_legacy, own_clause)


def visibility_required(model_cls, id_kwarg='id'):
    """Gate a *detail* route based on the row's visibility.

    Rules:
    - row not found                                 -> 404
    - viewer is the owner                           -> allow
    - row.owner_user_id IS NULL (legacy)            -> allow
    - row.visibility in ('public', 'unlisted')      -> allow
                                                       (unlisted is "by URL only" — that's
                                                        what this branch represents: list
                                                        pages exclude it via visible_in_list,
                                                        but a direct URL goes through.)
    - row.visibility == 'private' and not owner     -> 404 (don't leak existence)
    """
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            row_id = kwargs.get(id_kwarg)
            if row_id is None:
                abort(404)
            row = model_cls.query.get(row_id)
            if row is None:
                abort(404)
            current = getattr(g, 'current_user', None)
            owner_id = getattr(row, 'owner_user_id', None)
            visibility = getattr(row, 'visibility', 'public')

            if owner_id is None:
                return view(*args, **kwargs)
            if current is not None and current.id == owner_id:
                return view(*args, **kwargs)
            # Admin can view anything (incl. private rows of other users).
            if is_admin(current):
                return view(*args, **kwargs)
            # Collaborator: row was explicitly shared with this user.
            collaborators = getattr(row, 'collaborators', None)
            if (current is not None and collaborators is not None
                    and current in collaborators):
                return view(*args, **kwargs)
            if visibility in ('public', 'unlisted'):
                return view(*args, **kwargs)
            abort(404)
        return wrapped
    return decorator


# ---------------------------------------------------------------------------
# Phase 7: per-user quotas
# ---------------------------------------------------------------------------
# Why server-side caps (and not just an apologetic UI hint): without them,
# a single user can fill the volume in minutes and brick the app for
# everyone else. Each helper below is the source of truth for one cap;
# enforce_quota_or_flash composes them at upload routes.

def storage_used_bytes(user):
    """Total bytes this user is currently consuming on the data volume.
    Sums the cached `Dataset.storage_bytes` (set during dataset ingest)
    over all of the user's datasets. Submission ZIPs are not counted
    against the user's quota — the leaderboard owner pays for inbound
    submissions. Treats NULL as 0."""
    if user is None:
        return 0
    total = db.session.query(func.coalesce(func.sum(Dataset.storage_bytes), 0)).filter(
        Dataset.owner_user_id == user.id
    ).scalar()
    return int(total or 0)


def daily_submission_count(user):
    """Submissions this user uploaded in the trailing 24h. Used for the
    rate-limit cap; deliberately a rolling window not a calendar day so a
    user can't dump 50 at 23:59 and another 50 at 00:01."""
    if user is None:
        return 0
    cutoff = datetime.utcnow() - timedelta(days=1)
    return db.session.query(func.count(Submission.id)).filter(
        Submission.owner_user_id == user.id,
        Submission.upload_date >= cutoff,
    ).scalar() or 0


def dataset_count(user):
    if user is None:
        return 0
    return db.session.query(func.count(Dataset.id)).filter(
        Dataset.owner_user_id == user.id
    ).scalar() or 0


def _format_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def check_quota(user, *, kind, incoming_bytes=0):
    """Return (ok, message). `kind` is one of:
       - 'dataset_create'  : count cap + storage cap (incoming_bytes)
       - 'submission'      : daily rate cap
    """
    if user is None:
        return False, "Sign in required."

    if kind == 'dataset_create':
        if dataset_count(user) >= user.quota_max_datasets:
            return False, (
                f"Dataset limit reached ({user.quota_max_datasets}). "
                "Delete an existing dataset or contact us for a higher cap."
            )
        used = storage_used_bytes(user)
        if used + incoming_bytes > user.quota_max_storage_bytes:
            return False, (
                f"Storage limit would be exceeded: "
                f"{_format_bytes(used)} used + {_format_bytes(incoming_bytes)} new "
                f"> {_format_bytes(user.quota_max_storage_bytes)} cap."
            )
        return True, None

    if kind == 'submission':
        if daily_submission_count(user) >= user.quota_max_submissions_per_day:
            return False, (
                f"Daily submission limit reached "
                f"({user.quota_max_submissions_per_day}/24h). Try again later."
            )
        return True, None

    return True, None


# ---------------------------------------------------------------------------
# Phase 8: API token authentication
# ---------------------------------------------------------------------------
# Anon /api/* endpoints predate OAuth and are about to be exposed to the
# open internet. Add a Bearer-token gate so headless callers (CI, scripts)
# can keep working but anonymous strangers can't.

def generate_api_token():
    """Cryptographically random URL-safe token. 32 bytes of entropy →
    ~43 chars. Stored verbatim; treat the DB as secret-bearing."""
    return secrets.token_urlsafe(32)


def _bearer_token_from_request():
    """Extract a token from the Authorization header. Accepts both
    'Bearer <token>' (recommended) and a bare token (lenient — some
    older client scripts already send it that way)."""
    h = request.headers.get('Authorization', '')
    if not h:
        return None
    if h.lower().startswith('bearer '):
        return h[7:].strip()
    return h.strip()


def require_api_token(view):
    """Authenticate by Authorization: Bearer <token>; populate g.current_user.

    Use on programmatic /api/* endpoints. Returns 401 JSON on missing
    or invalid token. Inside the view, g.current_user is the row that
    owns the token, so quota helpers and owner_user_id assignments
    work the same as on cookie-authed routes."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        token = _bearer_token_from_request()
        if not token:
            return jsonify({'error': 'API token required (Authorization: Bearer <token>)'}), 401
        user = User.query.filter_by(api_token=token).first()
        if user is None:
            return jsonify({'error': 'Invalid API token'}), 401
        g.current_user = user
        return view(*args, **kwargs)
    return wrapped


def _admin_emails():
    """Allow-list of admin email addresses, env-var driven so we don't
    need a DB migration to grant admin. Comma-separated. Empty by default
    (no admins → /api/admin/* is locked down)."""
    raw = os.environ.get('BENCHHUB_ADMIN_EMAILS', '') or ''
    return {e.strip().lower() for e in raw.split(',') if e.strip()}


def is_admin(user):
    """A user is admin if EITHER their email is on the env-var allow-list
    OR their User.is_admin DB flag is set. Env-var users get the flag
    auto-set on first OAuth callback so the DB-backed admin list always
    reflects reality."""
    if user is None:
        return False
    if getattr(user, 'is_admin', False):
        return True
    return (user.email or '').strip().lower() in _admin_emails()


def require_admin(view):
    """Stack ON TOP of @require_api_token. Returns 403 if the
    token's user isn't on the BENCHHUB_ADMIN_EMAILS allow-list."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not is_admin(getattr(g, 'current_user', None)):
            return jsonify({'error': 'Admin access required'}), 403
        return view(*args, **kwargs)
    return wrapped


def _path_size_bytes(path):
    """Walk a directory, sum file sizes. Falls back to os.path.getsize for
    a single file. Returns 0 on any error so a stat blip can't block an
    upload that would otherwise succeed."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    except Exception:
        return 0


@app.route('/login')
def login():
    return render_template('login.html', next=request.args.get('next', ''))


@app.route('/login/github')
def login_github():
    if not os.environ.get('GITHUB_CLIENT_ID') or not os.environ.get('GITHUB_CLIENT_SECRET'):
        # Most common dev mistake — make it loud rather than crashing in Authlib.
        return ("GitHub OAuth not configured: set GITHUB_CLIENT_ID and "
                "GITHUB_CLIENT_SECRET (env vars or Fly secrets)."), 503
    # Stash the post-login redirect in the session so the OAuth state stays clean.
    session['oauth_next'] = request.args.get('next') or url_for('home')
    redirect_uri = url_for('oauth_callback_github', _external=True)
    return oauth.github.authorize_redirect(redirect_uri)


@app.route('/oauth/callback/github')
def oauth_callback_github():
    try:
        token = oauth.github.authorize_access_token()
    except Exception as e:
        flash(f"GitHub login failed: {e}", "danger")
        return redirect(url_for('login'))

    # GitHub: /user gives the profile, /user/emails gives the verified primary email.
    profile_resp = oauth.github.get('user', token=token)
    profile = profile_resp.json()
    oauth_sub = str(profile.get('id'))
    if not oauth_sub:
        flash("GitHub didn't return a user id.", "danger")
        return redirect(url_for('login'))

    email = profile.get('email')
    if not email:
        try:
            emails_resp = oauth.github.get('user/emails', token=token)
            emails = emails_resp.json() or []
            primary = next((e for e in emails if e.get('primary') and e.get('verified')), None)
            email = (primary or {}).get('email') or (emails[0] if emails else {}).get('email')
        except Exception:
            email = None

    if not email:
        flash("GitHub login succeeded but no email was available — make sure your GitHub "
              "account has at least one verified email.", "warning")
        return redirect(url_for('login'))

    # Upsert: provider+sub is the stable identity; email can change on GitHub side.
    user = User.query.filter_by(oauth_provider='github', oauth_sub=oauth_sub).first()
    if user is None:
        # Email collision against a different provider would land here on a new
        # provider. For now: GitHub-only, so this just means a brand-new account.
        user = User(
            email=email,
            display_name=profile.get('name') or profile.get('login') or email.split('@')[0],
            avatar_url=profile.get('avatar_url'),
            oauth_provider='github',
            oauth_sub=oauth_sub,
        )
        db.session.add(user)
    else:
        # Refresh denormalized profile fields in case the user changed them on GitHub.
        user.email = email
        user.display_name = profile.get('name') or profile.get('login') or user.display_name
        user.avatar_url = profile.get('avatar_url') or user.avatar_url
    user.last_login_at = datetime.utcnow()
    # Bootstrap: anyone whose email is on the env-var allow-list gets the
    # DB-backed admin bit set at login. Lets the runtime UI show + edit
    # the admin list without losing the env-var users on every restart.
    if (user.email or '').strip().lower() in _admin_emails() and not user.is_admin:
        user.is_admin = True
    db.session.commit()

    session['user_id'] = user.id
    flash(f"Logged in as {user.display_name}.", "success")
    next_url = session.pop('oauth_next', None) or url_for('home')
    return redirect(next_url)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('oauth_next', None)
    flash("Logged out.", "info")
    return redirect(url_for('login'))


# ===================== Settings: API tokens (Phase 8) =====================

@app.route('/settings/api_tokens', methods=['GET'])
@login_required
def api_tokens():
    return render_template('api_tokens.html', token=g.current_user.api_token)


@app.route('/settings/api_tokens/regenerate', methods=['POST'])
@login_required
def api_tokens_regenerate():
    """Generate (or rotate) the user's API token. Old token is invalidated
    immediately — no grace window. Display once on the resulting page;
    we keep it server-side so the user can revisit /settings/api_tokens
    to grab it again, but rotating produces a new value."""
    g.current_user.api_token = generate_api_token()
    db.session.commit()
    flash("New API token generated. Update any scripts that use the old one.", "success")
    return redirect(url_for('api_tokens'))


@app.route('/settings/api_tokens/revoke', methods=['POST'])
@login_required
def api_tokens_revoke():
    g.current_user.api_token = None
    db.session.commit()
    flash("API token revoked.", "warning")
    return redirect(url_for('api_tokens'))


# ===================== Admin management =====================
# DB-backed admin list, manageable from /settings/admins. Bootstrapped
# from BENCHHUB_ADMIN_EMAILS at OAuth callback.

@app.route('/settings/admins', methods=['GET'])
@login_required
def admins_settings():
    if not is_admin(g.current_user):
        abort(403)
    admin_users = User.query.filter_by(is_admin=True).order_by(User.email).all()
    env_emails = sorted(_admin_emails())
    return render_template(
        'admins_settings.html',
        admin_users=admin_users,
        env_emails=env_emails,
    )


@app.route('/settings/admins/grant', methods=['POST'])
@login_required
def admins_grant():
    if not is_admin(g.current_user):
        abort(403)
    email = (request.form.get('email') or '').strip().lower()
    if not email:
        flash("Email required.", "warning")
        return redirect(url_for('admins_settings'))
    user = User.query.filter(func.lower(User.email) == email).first()
    if user is None:
        flash(f"No BenchHub user with email '{email}'. They need to sign in once first.", "warning")
        return redirect(url_for('admins_settings'))
    if user.is_admin:
        flash(f"{user.display_name or user.email} is already an admin.", "info")
        return redirect(url_for('admins_settings'))
    user.is_admin = True
    db.session.commit()
    flash(f"Granted admin to {user.display_name or user.email}.", "success")
    return redirect(url_for('admins_settings'))


@app.route('/settings/admins/revoke/<int:user_id>', methods=['POST'])
@login_required
def admins_revoke(user_id):
    if not is_admin(g.current_user):
        abort(403)
    target = User.query.get(user_id)
    if target is None:
        abort(404)
    # Don't let an admin demote themselves — too easy to lock everyone out.
    # Another admin can do it, or revert via the env-var bootstrap.
    if target.id == g.current_user.id:
        flash("You can't revoke your own admin. Ask another admin.", "warning")
        return redirect(url_for('admins_settings'))
    target.is_admin = False
    db.session.commit()
    flash(f"Revoked admin from {target.display_name or target.email}.", "success")
    return redirect(url_for('admins_settings'))


# ===================== Account deletion (GDPR) =====================
# Right-to-be-forgotten flow. Wipes everything the user owns plus their
# user row. Submissions sitting in someone *else's* leaderboard get their
# owner detached (set to NULL) rather than deleted — the leaderboard
# owner's benchmark history stays intact, but the personal-data link is
# severed.

@app.route('/settings/account', methods=['GET'])
@login_required
def account_settings():
    return render_template('account_settings.html')


@app.route('/settings/account/delete', methods=['POST'])
@login_required
def account_delete():
    """Delete the user's account and all owned data.

    Confirm-text gate: the user must type their email exactly. We use
    email (not display name or "DELETE") because email is a thing the
    user demonstrably knows and can't accidentally tab-complete from
    a UI hint.
    """
    typed = (request.form.get('confirm_email') or '').strip().lower()
    expected = (g.current_user.email or '').strip().lower()
    if typed != expected:
        flash("Email confirmation didn't match. Account not deleted.", "danger")
        return redirect(url_for('account_settings'))

    user = g.current_user
    user_id = user.id

    # 1) Submissions in OTHER users' leaderboards: detach owner only.
    foreign_subs = (
        db.session.query(Submission)
        .join(Leaderboard, Submission.leaderboard_id == Leaderboard.id)
        .filter(
            Submission.owner_user_id == user_id,
            or_(Leaderboard.owner_user_id != user_id,
                Leaderboard.owner_user_id.is_(None)),
        )
        .all()
    )
    for sub in foreign_subs:
        sub.owner_user_id = None

    # 2) Owned leaderboards (cascade deletes their submissions).
    for lb in Leaderboard.query.filter_by(owner_user_id=user_id).all():
        db.session.delete(lb)

    # 4) Owned datasets (cascades samples + custom fields). Also nuke
    # the on-disk dataset folder so we're not paying for orphaned bytes.
    for ds in Dataset.query.filter_by(owner_user_id=user_id).all():
        folder_name = secure_filename(ds.name)
        ds_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', folder_name)
        if os.path.isdir(ds_dir):
            shutil.rmtree(ds_dir, ignore_errors=True)
        db.session.delete(ds)

    # 5) Owned global metrics + visualizations.
    GlobalMetric.query.filter_by(owner_user_id=user_id).delete(synchronize_session=False)
    GlobalVisualization.query.filter_by(owner_user_id=user_id).delete(synchronize_session=False)

    # 6) The user row itself. Drops the cookie next.
    db.session.delete(user)
    db.session.commit()

    session.pop('user_id', None)
    session.pop('oauth_next', None)
    flash("Your account and owned content have been deleted.", "success")
    return redirect(url_for('landing'))


# ===================== Legal stubs (Phase 8) =====================
# Placeholder content. Treating these as real legal documents will require
# actual lawyer review before public launch — this is a launch-blocker
# checkbox, not the final wording.

@app.route('/terms')
def terms():
    return render_template('legal_terms.html')


@app.route('/privacy')
def privacy():
    return render_template('legal_privacy.html')


# ===================== Project routes =====================


@app.route('/')
def landing():
    """Public marketing landing page (Phase 6 Slice 1).

    Replaces the old `redirect('/projects')` so anonymous visitors see
    a real homepage. Logged-in users see the same page but with a
    "Go to dashboard" CTA instead of "Log in".

    Featured leaderboards: top public leaderboards by submission activity
    in the last 30 days. Visibility filter excludes private + unlisted —
    same rules as /explore (when that lands).
    """
    cutoff = datetime.utcnow() - timedelta(days=30)

    # Subquery: count submissions per leaderboard in the activity window.
    # Use SQLAlchemy func.count + group_by so the DB does the work.
    activity = (
        db.session.query(
            Submission.leaderboard_id.label('lb_id'),
            func.count(Submission.id).label('recent_count'),
        )
        .filter(Submission.upload_date >= cutoff, Submission.is_archived.is_(False))
        .group_by(Submission.leaderboard_id)
        .subquery()
    )

    featured = (
        db.session.query(Leaderboard, activity.c.recent_count)
        .outerjoin(activity, Leaderboard.id == activity.c.lb_id)
        # Only public lists; pre-Phase-1 NULL-owner rows fall through the
        # same branch as 'public' per visible_in_list semantics.
        .filter(or_(
            Leaderboard.visibility == 'public',
            Leaderboard.owner_user_id.is_(None),
        ))
        # Order by recent activity (NULL = no submissions in window → 0).
        .order_by(func.coalesce(activity.c.recent_count, 0).desc(),
                  Leaderboard.upload_date.desc())
        .limit(3)
        .all()
    )

    # Render-friendly wrapper: list of (leaderboard, recent_count_int).
    featured_rows = [(lb, int(c or 0)) for lb, c in featured]


    return render_template(
        'landing.html',
        featured=featured_rows,
    )


def _dataset_thumb_url(ds):
    """Return a URL for a representative thumbnail of `ds`, or None.

    Prefers an `image_*` custom field on any sample (rendered by the
    existing /custom_field_image endpoint). Falls back to a depth field
    (which the depth-image endpoint will render to PNG). Returns None
    if the dataset has no visualizable content yet (e.g. metric-only)."""
    sample_ids = [s.id for s in Sample.query.filter_by(dataset_id=ds.id).limit(20).all()]
    if not sample_ids:
        return None
    cf = (
        CustomField.query
        .filter(CustomField.sample_id.in_(sample_ids),
                CustomField.field_type.in_(('image', 'depth')))
        .order_by(CustomField.field_type.desc())  # 'image' before 'depth'
        .first()
    )
    if cf is None:
        return None
    return url_for('custom_field_image', field_id=cf.id)


@app.route('/home')
@login_required
def home():
    """User dashboard: their datasets + leaderboards, recent first, each
    with a sample thumbnail when one exists. Becomes the post-login
    destination."""
    user = g.current_user

    datasets = (
        Dataset.query
        .filter(Dataset.owner_user_id == user.id)
        .order_by(Dataset.upload_date.desc())
        .limit(24)
        .all()
    )
    leaderboards = (
        Leaderboard.query
        .filter(Leaderboard.owner_user_id == user.id)
        .order_by(Leaderboard.upload_date.desc())
        .limit(24)
        .all()
    )

    # For each leaderboard, pick a thumbnail from its first dataset.
    dataset_thumbs = {ds.id: _dataset_thumb_url(ds) for ds in datasets}
    leaderboard_thumbs = {}
    for lb in leaderboards:
        lb_datasets = list(lb.datasets)
        leaderboard_thumbs[lb.id] = (
            _dataset_thumb_url(lb_datasets[0]) if lb_datasets else None
        )

    return render_template(
        'home.html',
        datasets=datasets,
        leaderboards=leaderboards,
        dataset_thumbs=dataset_thumbs,
        leaderboard_thumbs=leaderboard_thumbs,
    )


@app.route('/explore')
def explore():
    """Public catalog of leaderboards (Phase 6 Slice 2).

    Visible to everyone (anonymous or signed-in). Filtering/sorting via
    query string:
        ?q=<text>     — case-insensitive name match
        ?sort=activity  (default) recent submissions in last 30 days, then upload_date
              recent  newest leaderboards first
              popular total submissions across all time
    """
    q = (request.args.get('q') or '').strip()
    tag_filter = (request.args.get('tag') or '').strip().lower()
    sort = request.args.get('sort', 'activity')
    if sort not in ('activity', 'recent', 'popular'):
        sort = 'activity'

    cutoff = datetime.utcnow() - timedelta(days=30)

    # Two activity counts: recent (for default sort + display) and total
    # (for the "popular" sort + the "N submissions" badge).
    recent_activity = (
        db.session.query(
            Submission.leaderboard_id.label('lb_id'),
            func.count(Submission.id).label('recent_count'),
        )
        .filter(Submission.upload_date >= cutoff, Submission.is_archived.is_(False))
        .group_by(Submission.leaderboard_id)
        .subquery()
    )
    total_activity = (
        db.session.query(
            Submission.leaderboard_id.label('lb_id'),
            func.count(Submission.id).label('total_count'),
        )
        .filter(Submission.is_archived.is_(False))
        .group_by(Submission.leaderboard_id)
        .subquery()
    )

    base = (
        db.session.query(
            Leaderboard,
            func.coalesce(recent_activity.c.recent_count, 0).label('recent_count'),
            func.coalesce(total_activity.c.total_count, 0).label('total_count'),
        )
        .outerjoin(recent_activity, Leaderboard.id == recent_activity.c.lb_id)
        .outerjoin(total_activity, Leaderboard.id == total_activity.c.lb_id)
        .filter(visible_in_list(Leaderboard, getattr(g, 'current_user', None)))
    )


    if q:
        base = base.filter(Leaderboard.name.ilike(f'%{q}%'))

    if tag_filter:
        base = (
            base.join(leaderboard_tags, leaderboard_tags.c.leaderboard_id == Leaderboard.id)
                .join(Tag, Tag.id == leaderboard_tags.c.tag_id)
                .filter(Tag.name == tag_filter)
        )

    if sort == 'recent':
        base = base.order_by(Leaderboard.upload_date.desc())
    elif sort == 'popular':
        base = base.order_by(
            func.coalesce(total_activity.c.total_count, 0).desc(),
            Leaderboard.upload_date.desc(),
        )
    else:  # activity
        base = base.order_by(
            func.coalesce(recent_activity.c.recent_count, 0).desc(),
            Leaderboard.upload_date.desc(),
        )

    rows = [
        {'lb': lb, 'recent': int(r or 0), 'total': int(t or 0)}
        for lb, r, t in base.limit(60).all()
    ]

    # Tag cloud: count of *visible* leaderboards per tag, plus dataset
    # tag counts folded in. Only tags with at least one visible item show.
    visible_lb_filter = visible_in_list(Leaderboard, getattr(g, 'current_user', None))
    visible_ds_filter = visible_in_list(Dataset, getattr(g, 'current_user', None))
    lb_tag_counts = (
        db.session.query(Tag.name, func.count(Leaderboard.id).label('cnt'))
        .join(leaderboard_tags, leaderboard_tags.c.tag_id == Tag.id)
        .join(Leaderboard, Leaderboard.id == leaderboard_tags.c.leaderboard_id)
        .filter(visible_lb_filter)
        .group_by(Tag.name)
        .all()
    )
    ds_tag_counts = (
        db.session.query(Tag.name, func.count(Dataset.id).label('cnt'))
        .join(dataset_tags, dataset_tags.c.tag_id == Tag.id)
        .join(Dataset, Dataset.id == dataset_tags.c.dataset_id)
        .filter(visible_ds_filter)
        .group_by(Tag.name)
        .all()
    )
    combined = {}
    for name, cnt in lb_tag_counts:
        combined[name] = combined.get(name, 0) + int(cnt or 0)
    for name, cnt in ds_tag_counts:
        combined[name] = combined.get(name, 0) + int(cnt or 0)
    if combined:
        max_cnt = max(combined.values())
        # Bucket each tag into one of 5 size tiers for the cloud rendering.
        # Linear bucketing is fine for the small sizes we expect (<100 tags).
        tag_cloud = []
        for name, cnt in sorted(combined.items(), key=lambda kv: (-kv[1], kv[0])):
            tier = 1 + min(4, int((cnt / max_cnt) * 4))  # 1..5
            tag_cloud.append({'name': name, 'count': cnt, 'tier': tier})
    else:
        tag_cloud = []

    return render_template(
        'explore.html',
        rows=rows,
        q=q,
        sort=sort,
        tag_cloud=tag_cloud,
        active_tag=tag_filter,
    )


@app.route('/u/<int:user_id>')
def user_profile(user_id):
    """Public profile page (Phase 6 Slice 2).

    Lists the user's PUBLIC datasets, leaderboards, and recent
    submissions. Private + unlisted rows aren't surfaced here even when
    the viewer is the profile owner — those live in their own dashboard.
    """
    user = User.query.get(user_id)
    if user is None:
        abort(404)

    viewer = getattr(g, 'current_user', None)

    # On the public profile we want the *publicly visible* slice of the
    # user's stuff, regardless of who's looking. Build a stricter filter
    # than visible_in_list (which would also include the viewer's private
    # rows belonging to *that* viewer — irrelevant on someone else's page).
    def public_only_filter(model_cls):
        return or_(
            model_cls.visibility == 'public',
            model_cls.owner_user_id.is_(None),
        )

    datasets = (
        Dataset.query
        .filter(Dataset.owner_user_id == user.id, public_only_filter(Dataset))
        .order_by(Dataset.upload_date.desc())
        .limit(12)
        .all()
    )
    leaderboards = (
        Leaderboard.query
        .filter(Leaderboard.owner_user_id == user.id, public_only_filter(Leaderboard))
        .order_by(Leaderboard.upload_date.desc())
        .limit(12)
        .all()
    )
    # Submissions: show recent ones whose leaderboard is public (don't
    # leak that a user submitted to a private leaderboard).
    recent_subs = (
        db.session.query(Submission, Leaderboard)
        .join(Leaderboard, Submission.leaderboard_id == Leaderboard.id)
        .filter(
            Submission.owner_user_id == user.id,
            Submission.is_archived.is_(False),
            public_only_filter(Leaderboard),
        )
        .order_by(Submission.upload_date.desc())
        .limit(10)
        .all()
    )

    return render_template(
        'user_profile.html',
        profile_user=user,
        viewer_is_owner=(viewer is not None and viewer.id == user.id),
        datasets=datasets,
        leaderboards=leaderboards,
        recent_subs=recent_subs,
    )
# -------------------------------------------------

# --- End Project Management Logic ---


@app.route('/leaderboard/<int:leaderboard_id>/edit', methods=['GET', 'POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def edit_leaderboard(leaderboard_id):
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
                lmid = f"lm_{lm.id}"
                if lmid in directions:
                    lm.sort_direction = directions[lmid]
        
        # Update Aggregation Settings for existing metrics AND custom metrics
        metric_aggregation = {}
        if leaderboard.metric_aggregation:
             try:
                 metric_aggregation = json.loads(leaderboard.metric_aggregation)
             except Exception:
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
                
                # Check if it is a dynamic metric using unique ID
                lm = next((m for m in leaderboard.leaderboard_metrics if f"lm_{m.id}" == metric_name), None)
                
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
                 tasks.reaggregate_submission_metrics.delay(sub.id)
            flash('Leaderboard configuration updated. Aggregation updated (Optimized).', 'success')
        elif has_direction_updates:
            flash('Leaderboard coloring updated.', 'success')
        else:
            flash('Leaderboard settings updated.', 'success')
            
        return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))
        
    # Get available fields for mapping (sampling)
    fields_set = set()
    
    # 1. Check GT data
    dataset_ids = [d.id for d in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
    samples = Sample.query.filter(Sample.dataset_id.in_(dataset_ids)).all()
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
    
    # Map internal IDs to labels
    metric_labels = { m: m for m in dataset_fields_set | submission_fields_set }
    
    for lm in leaderboard.leaderboard_metrics:
        # Use target_name if available (custom alias), otherwise global metric name
        name_to_add = lm.target_name if lm.target_name else lm.global_metric.name
        lmid = f"lm_{lm.id}"
        metric_labels[lmid] = name_to_add
        
        if lm.global_metric.is_aggregated:
            aggregated_metrics_list.add(lmid)
        else:
            per_sample_metrics.add(lmid)
            
    # Add to submission fields for backward compatibility/default view if needed,
    # but also prepare separate lists for UI
    for m in per_sample_metrics:
        submission_fields_set.add(m)
        
    # Note: aggregated metrics are usually NOT mixed with per-sample submission fields 
    # unless we explicitly want them to show up in "Submission/Metric" dropdown.
    # The user wants "New Category". So we will pass 'aggregated_metrics_list' separately.

    # Fetch sample tags for auto-suggestions
    all_sample_tag_names, all_sample_prefixes = get_all_sample_tags(dataset_ids)
    all_sample_tags = sorted(list(set(all_sample_tag_names + all_sample_prefixes)))
    
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
        # Use unique internal ID
        all_known_metrics.add(f"lm_{lm.id}")
    # Custom metrics from database (linked to this dataset/submissions)
    # Similar logic to leaderboard_view discovery
    dataset_custom_metrics = CustomField.query.filter(CustomField.sample_id.in_([s.id for s in samples]), CustomField.field_type == 'metric').all()
    for cf in dataset_custom_metrics:
        all_known_metrics.add(f'gt_{cf.name}') # Although leaderboard usually aggregates sub metrics? 
        # Actually leaderboard.html only shows standard, dynamic, and sub-custom metrics. Not GT custom metrics usually (unless dynamic uses them).
        # But let's stick to what's shown in leaderboard table loop.
    
    # Submission custom metrics
    if sub_ids:
        submission_custom_metrics = CustomField.query.filter(
            CustomField.submission_id.in_(sub_ids), 
            CustomField.field_type.in_(['metric', 'scalar'])
        ).all()
        for cf in submission_custom_metrics:
            if not cf.name.startswith('lm_'):
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
         metric_to_lm[f"lm_{lm.id}"] = lm

    return render_template('edit_leaderboard.html', 
                           leaderboard=leaderboard,
                           dataset_fields=dataset_fields,
                           submission_fields=submission_fields,
                           aggregated_metrics=sorted(list(aggregated_metrics_list)),
                           per_sample_metrics=sorted(list(per_sample_metrics)),
                           available_metrics=sorted_metrics,
                           all_known_metrics=sorted_metrics,
                           current_directions=current_directions,
                           all_sample_tags=all_sample_tags,
                           current_aggregation=current_aggregation,
                           metric_labels=metric_labels,
                           metric_to_lm=metric_to_lm,
                           global_metrics=global_metrics,
                           global_visualizations=global_visualizations,
                           all_datasets=Dataset.query.all())
                           

@app.route('/leaderboard/<int:leaderboard_id>/leaderboard_metric/add', methods=['POST'])
def add_leaderboard_metric(leaderboard_id):
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
        
        final_name = requested_name
        
        lm = LeaderboardMetric(
            leaderboard_id=leaderboard.id,
            global_metric_id=gm.id,
            arg_mappings=json.dumps(arg_mappings),
            target_name=final_name,
            pooling_type='mean',
            pooling_percentile=None,
            sort_direction=request.form.get('sort_direction', 'higher_is_better'),
            tag_filter=request.form.get('tag_filter', '').strip() or None
        )
        db.session.add(lm)
        db.session.flush() # Ensure ID is generated for use below
        
        # Auto-add to summary metrics for display using unique ID
        current_metrics = [m.strip() for m in leaderboard.summary_metrics.split(',') if m.strip()]
        lmid = f"lm_{lm.id}"
        if lmid not in current_metrics:
            current_metrics.append(lmid)
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
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def import_leaderboard_settings(leaderboard_id):
    target_lb = Leaderboard.query.get_or_404(leaderboard_id)
    source_lb_id = request.form.get('source_leaderboard_id')
    
    if not source_lb_id:
        flash("Please select a source leaderboard.", "warning")
        return redirect(url_for('edit_leaderboard', leaderboard_id=target_lb.id, _anchor=request.form.get('active_tab')))
        
    source_lb = Leaderboard.query.get_or_404(source_lb_id)
    
    try:
        # 1. Clear existing items on target first to avoid conflicts
        for old_metric in target_lb.leaderboard_metrics:
            db.session.delete(old_metric)
        for old_vis in target_lb.leaderboard_visualizations:
            db.session.delete(old_vis)
        
        # Flush to ensure deletions happen
        db.session.flush()

        # 2. Clone Metrics and build ID map
        metric_id_map = {} # old_id -> new_id
        
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
            db.session.flush() # Get new ID
            metric_id_map[src_metric.id] = new_metric.id

        # 3. Clone Visualizations and build ID map
        viz_id_map = {} # old_id -> new_id
        
        for src_vis in source_lb.leaderboard_visualizations:
            new_vis = LeaderboardVisualization(
                leaderboard_id=target_lb.id,
                global_visualization_id=src_vis.global_visualization_id,
                arg_mappings=src_vis.arg_mappings,
                target_name=src_vis.target_name,
                display_order=src_vis.display_order
            )
            db.session.add(new_vis)
            db.session.flush() # Get new ID
            viz_id_map[src_vis.id] = new_vis.id

        # Helper to replace IDs in comma-separated strings (CSV)
        def replace_ids_csv(text):
            if not text: return text
            parts = [p.strip() for p in text.split(',')]
            new_parts = []
            for p in parts:
                if p.startswith('lm_'):
                    try:
                        old_id = int(p[3:])
                        if old_id in metric_id_map:
                            new_parts.append(f"lm_{metric_id_map[old_id]}")
                            continue
                    except Exception: pass
                elif p.startswith('viz_'):
                    try:
                        old_id = int(p[4:])
                        if old_id in viz_id_map:
                            new_parts.append(f"viz_{viz_id_map[old_id]}")
                            continue
                    except Exception: pass
                
                # Keep original if no match (e.g. custom metric name)
                new_parts.append(p)
            return ','.join(new_parts)

        # Helper to replace IDs in JSON objects (keys)
        def replace_ids_json(json_str):
            if not json_str: return json_str
            try:
                data = json.loads(json_str)
                new_data = {}
                for k, v in data.items():
                    new_key = k
                    if k.startswith('lm_'):
                        try:
                            old_id = int(k[3:])
                            if old_id in metric_id_map:
                                new_key = f"lm_{metric_id_map[old_id]}"
                        except Exception: pass
                    new_data[new_key] = v
                return json.dumps(new_data)
            except Exception:
                return json_str

        # 4. Copy and Remap Fields
        target_lb.summary_metrics = replace_ids_csv(source_lb.summary_metrics)
        target_lb.visualizations = replace_ids_csv(source_lb.visualizations)
        target_lb.selected_metrics = replace_ids_csv(source_lb.selected_metrics)
        target_lb.comparison_display_columns = replace_ids_csv(source_lb.comparison_display_columns)
        
        target_lb.metric_directions = replace_ids_json(source_lb.metric_directions)
        target_lb.metric_aggregation = replace_ids_json(source_lb.metric_aggregation)
        
        # Copy direct fields (no IDs involved)
        target_lb.scalar_width = source_lb.scalar_width
        target_lb.image_width = source_lb.image_width
        target_lb.last_sample_filter = source_lb.last_sample_filter
            
        db.session.commit()
        flash(f"Settings imported from '{source_lb.name}' with ID remapping.", "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"Error importing settings: {e}", "error")
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=target_lb.id, _anchor=request.form.get('active_tab')))

@app.route('/leaderboard/<int:leaderboard_id>/leaderboard_metric/<int:metric_id>/edit', methods=['POST'])
def edit_leaderboard_metric(leaderboard_id, metric_id):
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
        
        old_name = lm.target_name if lm.target_name else lm.global_metric.name
        lm.target_name = requested_name
        final_name = requested_name
        
        # Sync summary_metrics: Replace old name with lmid to ensure it stays visible/valid
        leaderboard = Leaderboard.query.get(leaderboard_id)
        if leaderboard and leaderboard.summary_metrics:
            current_selected = [m.strip() for m in leaderboard.summary_metrics.split(',') if m.strip()]
            lmid = f"lm_{lm.id}"
            if old_name in current_selected:
                # Replace with lmid
                new_selected = [lmid if m == old_name else m for m in current_selected]
                leaderboard.summary_metrics = ','.join(new_selected)
                # No need to commit here, it will be committed below with lm.target_name
        
        # Update Sort Direction
        if 'sort_direction' in request.form:
            lm.sort_direction = request.form.get('sort_direction', 'higher_is_better')
        lm.tag_filter = request.form.get('tag_filter', '').strip() or None
        
        db.session.commit()
        
        # Determine if recalculation is actually needed
        new_mappings_json = json.dumps(arg_mappings)
        needs_recalculation = (new_mappings_json != old_mappings_json)
        
        if needs_recalculation:
            # Invalidate submissions instead of automatic recalculation
            submissions = Submission.query.filter_by(leaderboard_id=leaderboard_id).all()
            for sub in submissions:
                 sub.processing_status = 'Outdated'
            db.session.commit()
            flash(f'Metric "{final_name}" updated. Submissions marked as Outdated.', 'success')
        else:
            flash(f'Metric "{final_name}" updated.', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating metric: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/leaderboard/<int:leaderboard_id>/leaderboard_metric/<int:metric_id>/delete', methods=['POST'])
def delete_leaderboard_metric(leaderboard_id, metric_id):
    lm = LeaderboardMetric.query.get_or_404(metric_id)
    if lm.leaderboard_id != leaderboard_id:
        abort(403)
        
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    metric_name = lm.target_name if lm.target_name else lm.global_metric.name
    
    # 1. Remove from selected_metrics (Display Columns)
    if leaderboard.summary_metrics:
        # Prune both name and lm_{id} for safety
        current_summary = [m.strip() for m in leaderboard.summary_metrics.split(',') if m.strip()]
        lmid = f"lm_{lm.id}"
        metric_name = lm.target_name if lm.target_name else lm.global_metric.name
        
        new_summary = [m for m in current_summary if m != lmid and m != metric_name]
        if len(new_summary) != len(current_summary):
            leaderboard.summary_metrics = ','.join(new_summary)
            
    # 2. Delete the record
    db.session.delete(lm)
    db.session.commit()
    
    # 3. Trigger recalculation for all submissions
    submissions = Submission.query.filter_by(leaderboard_id=leaderboard.id).all()
    for sub in submissions:
         tasks.process_submission.delay(sub.id)
         
    flash(f'Metric "{metric_name}" removed. Recalculation started.', 'success')
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

# ==================== Leaderboard Visualization Management Routes ====================

@app.route('/leaderboard/<int:leaderboard_id>/leaderboard_visualization/add', methods=['POST'])
def add_leaderboard_visualization(leaderboard_id):
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
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/leaderboard/<int:leaderboard_id>/leaderboard_visualization/<int:viz_id>/edit', methods=['POST'])
def edit_leaderboard_visualization(leaderboard_id, viz_id):
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
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

@app.route('/leaderboard/<int:leaderboard_id>/leaderboard_visualization/<int:viz_id>/delete', methods=['POST'])
def delete_leaderboard_visualization(leaderboard_id, viz_id):
    lv = LeaderboardVisualization.query.get_or_404(viz_id)
    try:
        db.session.delete(lv)
        db.session.commit()
        flash('Visualization removed from leaderboard', 'success')
    except Exception as e:
        flash(f'Error removing visualization: {e}', 'danger')
        
    return redirect(url_for('edit_leaderboard', leaderboard_id=leaderboard_id, _anchor=request.form.get('active_tab')))

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
            
            # 1. Try by sample_id
            cf = CustomField.query.filter_by(submission_id=submission.id, sample_id=sample.id, name=field_key).first()
            
            if not cf:
                 # 2. Fallback to sample_name
                cf = CustomField.query.filter_by(submission_id=submission.id, sample_name=sample.name, name=field_key).first()
                if not cf:
                    # Check if it's a known GlobalMetric
                    # [FIX] Import GlobalMetric inside to avoid circular imports if any, 
                    # but mainly we need to ensure we are in a valid session context.
                    # This function is likely called from the Flask routes above, so context should exist.
                    # The error 'Flask app is not registered' suggests we might be detached or using a stale session?
                    # Explicitly using db.session might help if the query object is detached.
                    
                    from app import GlobalMetric
                    gm = GlobalMetric.query.filter_by(name=field_key).first()
                    if gm:
                        pass  # GlobalMetric exists but no computed value for this submission
            if cf:
                value = cf.get_value()
            
    return value

@app.route('/visualization/<int:lv_id>/execute/<int:sample_id>')
@app.route('/visualization/<int:lv_id>/execute/<int:sample_id>/<int:submission_id>')
def execute_visualization(lv_id, sample_id, submission_id=None):
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

def generate_and_cache_agg_viz(lv, submission=None):
    """Generates an aggregated visualization and saves it to cache. Returns the cache path."""
    import hashlib
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    leaderboard = lv.leaderboard
    
    # Generate cache key with hashes
    code_hash = hashlib.md5((lv.global_visualization.python_code or "").encode()).hexdigest()
    mapping_hash = hashlib.md5((lv.arg_mappings or "").encode()).hexdigest()
    cache_key = f"viz_agg_{lv.id}_{submission.id if submission else 'none'}_{code_hash}_{mapping_hash}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_dir = os.path.join(os.getcwd(), 'data', 'viz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_hash}.png")
    
    # Return cached image if exists
    if os.path.exists(cache_path):
        return cache_path
            
    try:
        # Fetch all samples for the dataset(s)
        dataset_ids = [d.id for d in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
        all_samples = Sample.query.filter(Sample.dataset_id.in_(dataset_ids)).order_by(Sample.name).all()

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
                return cache_path

        return None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

@app.route('/visualization/<int:lv_id>/execute_aggregated')
@app.route('/visualization/<int:lv_id>/execute_aggregated/<int:submission_id>')
def execute_aggregated_visualization(lv_id, submission_id=None):
    """Execute an aggregated visualization (across all samples) and return the image as PNG."""
    lv = LeaderboardVisualization.query.get_or_404(lv_id)
    if not lv.global_visualization.is_aggregated:
        return create_error_image("Not an aggregated visualization")

    submission = Submission.query.get(submission_id) if submission_id else None
    
    cache_path = generate_and_cache_agg_viz(lv, submission)
    if cache_path and os.path.exists(cache_path):
        return send_file(cache_path, mimetype='image/png')
    
    return create_error_image("No result or execution error")

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


@app.route('/metrics')
def metrics_view():
    metrics = (
        GlobalMetric.query
        .filter(visible_in_list(GlobalMetric, getattr(g, 'current_user', None)))
        .order_by(GlobalMetric.name)
        .all()
    )
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
                return text_content
    except Exception as e:
        print(f"File Storage Error: {e}")
    
    return None

@app.route('/metrics/create', methods=['POST'])
@login_required
def create_global_metric():
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = request.form.get('python_code', ''.strip())
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
            accepts_aggregated_inputs=accepts_aggregated_inputs,
            owner_user_id=g.current_user.id,
        )
        db.session.add(metric)
        db.session.commit()
        flash(f'Metric "{name}" created successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating metric: {e}', 'danger')
    
    return redirect(url_for('metrics_view'))

@app.route('/metrics/<int:metric_id>/edit', methods=['POST'])
@login_required
@owner_required(GlobalMetric, 'metric_id')
def edit_global_metric(metric_id):
    metric = GlobalMetric.query.get_or_404(metric_id)
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = request.form.get('python_code', ''.strip())
        
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

@app.route('/metrics/<int:metric_id>/delete', methods=['POST'])
@login_required
@owner_required(GlobalMetric, 'metric_id')
def delete_global_metric(metric_id):
    metric = GlobalMetric.query.get_or_404(metric_id)
    try:
        db.session.delete(metric)
        db.session.commit()
        flash(f'Metric "{metric.name}" deleted.', 'success')
    except Exception as e:
         db.session.rollback()
         flash(f'Error deleting metric: {e}. It might be used by a leaderboard.', 'danger')
         
    return redirect(url_for('metrics_view'))

@app.route('/metrics/<int:metric_id>/download')
def download_metric(metric_id):
    """Download metric code as a .txt file"""
    metric = GlobalMetric.query.get_or_404(metric_id)
    
    response = make_response(metric.python_code)
    response.headers['Content-Type'] = 'text/plain'
    filename = f"{metric.name}.txt"
    response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{quote(filename)}"
    
    return response

# ==================== Visualization Management Routes ====================

@app.route('/visualizations')
def visualizations_view():
    """List all visualizations for the project."""
    visualizations = (
        GlobalVisualization.query
        .filter(visible_in_list(GlobalVisualization, getattr(g, 'current_user', None)))
        .order_by(GlobalVisualization.name)
        .all()
    )
    return render_template('visualizations.html', visualizations=visualizations)

@app.route('/create_visualization', methods=['POST'])
@login_required
def create_visualization():
    """Create a new visualization."""
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = request.form.get('python_code', ''.strip())
        
        # Handle file upload if present
        viz_file = request.files.get('visualization_file')
        file_code = extract_code_from_file(viz_file)
        
        if file_code:
            python_code = file_code
        elif viz_file and viz_file.filename:
            flash(f'Found file {viz_file.filename} but could not extract Python code from it.', 'warning')
            if "Implementation will be loaded from ZIP" in python_code:
                return redirect(url_for('visualizations_view'))

        if not python_code or not python_code.strip() or "Implementation will be loaded from ZIP" in python_code:
            flash('Visualization code is required.', 'danger')
            return redirect(url_for('visualizations_view'))
        
        # Create new visualization
        new_viz = GlobalVisualization(
            name=name,
            description=description,
            python_code=python_code,
            is_aggregated='is_aggregated' in request.form,
            accepts_aggregated_inputs='accepts_aggregated_inputs' in request.form,
            owner_user_id=g.current_user.id,
        )
        
        db.session.add(new_viz)
        db.session.commit()
        flash(f'Visualization "{new_viz.name}" created successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating visualization: {e}', 'danger')
    
    return redirect(url_for('visualizations_view'))

@app.route('/visualizations/<int:viz_id>/edit', methods=['POST'])
@login_required
@owner_required(GlobalVisualization, 'viz_id')
def edit_visualization(viz_id):
    """Edit an existing visualization."""
    viz = GlobalVisualization.query.get_or_404(viz_id)
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        python_code = request.form.get('python_code', ''.strip())
        
        # Handle file upload if present
        viz_file = request.files.get('visualization_file')
        file_code = extract_code_from_file(viz_file)
        
        if file_code:
            python_code = file_code
        elif viz_file and viz_file.filename:
            flash(f'Found file {viz_file.filename} but could not extract Python code from it.', 'warning')
            if "Implementation will be loaded from ZIP" in python_code:
                return redirect(url_for('visualizations_view'))

        if not python_code or not python_code.strip() or "Implementation will be loaded from ZIP" in python_code:
            if "Implementation will be loaded from ZIP" in python_code:
                flash('Invalid code submitted (placeholder). Update canceled.', 'danger')
                return redirect(url_for('visualizations_view'))
        
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
    
    return redirect(url_for('visualizations_view'))

@app.route('/visualizations/<int:viz_id>/delete', methods=['POST'])
@login_required
@owner_required(GlobalVisualization, 'viz_id')
def delete_visualization(viz_id):
    """Delete a visualization."""
    viz = GlobalVisualization.query.get_or_404(viz_id)
    try:
        db.session.delete(viz)
        db.session.commit()
        flash(f'Visualization "{viz.name}" deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting visualization: {e}. It might be used by a leaderboard.', 'danger')
        
    return redirect(url_for('visualizations_view'))

@app.route('/visualizations/<int:viz_id>/download')
def download_visualization(viz_id):
    """Download visualization code as a .txt file."""
    viz = GlobalVisualization.query.get_or_404(viz_id)
    
    response = make_response(viz.python_code)
    response.headers['Content-Type'] = 'text/plain'
    filename = f"{viz.name}.txt"
    response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{quote(filename)}"
    
    return response

@app.route('/metrics/upload', methods=['POST'])
@login_required
def upload_metric():
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
            # Multi-tenancy guard: only the owner (or anyone if it's legacy NULL-owner)
            # can overwrite an existing metric by name.
            if (existing_metric.owner_user_id is not None
                    and existing_metric.owner_user_id != g.current_user.id):
                flash(f'Metric "{metric_name}" is owned by another user — pick a different name.', 'danger')
                return redirect(url_for('metrics_view'))
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
                accepts_aggregated_inputs=accepts_aggregated_inputs,
                owner_user_id=g.current_user.id,
            )
            db.session.add(new_metric)
            db.session.commit()
            flash(f'Metric "{metric_name}" created successfully.', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error uploading metric: {e}', 'danger')
    
    return redirect(url_for('metrics_view'))

@app.route('/submission/<int:submission_id>/recalculate', methods=['POST'])
@login_required
@owner_required(Submission, 'submission_id')
def recalculate_submission(submission_id):
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

@app.route('/leaderboard/<int:leaderboard_id>')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_view(leaderboard_id):
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

    if sort_metric == 'date_uploaded':
        if sort_order == 'asc':
            query = query.order_by(Submission.upload_date.asc())
        else:
            query = query.order_by(Submission.upload_date.desc())
    elif sort_metric == 'name':
        if sort_order == 'asc':
            query = query.order_by(Submission.name.asc())
        else:
            query = query.order_by(Submission.name.desc())
    else:
        query = query.order_by(Submission.upload_date.desc())
        
    submissions = query.all()
    
    all_tags = Tag.query.join(Tag.submissions).filter(Submission.leaderboard_id == leaderboard.id).distinct().all()
    # Also get all sample tags for autocomplete
    dataset_ids = [d.id for d in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
    all_sample_tag_names, all_sample_prefixes = get_all_sample_tags(dataset_ids)

    processed_submissions = [s for s in submissions if s.processing_status == 'Processed']
    selected_metrics = [m for m in leaderboard.summary_metrics.split(',') if m.strip()]
    
    # Get all custom metrics from submissions
    custom_metrics = set()
    for sub in processed_submissions:
        for cf in sub.custom_fields:
            if cf.field_type in ['metric', 'scalar'] and not cf.name.startswith('lm_'):
                custom_metrics.add(cf.name)
    
    # Map internal IDs to labels
    metric_labels = {}
    for m in custom_metrics:
        metric_labels[m] = m
        
    # Get all dynamic metrics (linked global metrics) - mapping target_name to lm
    # Key by lm_{id} to support duplicate display names
    leaderboard_metrics_map = { f"lm_{lm.id}": lm for lm in leaderboard.leaderboard_metrics }
    for lmid, lm in leaderboard_metrics_map.items():
        metric_labels[lmid] = lm.target_name if lm.target_name else lm.global_metric.name

    # Map from name to list of matching lmids for support in summary_metrics
    name_to_lmids = {}
    for lmid, lm in leaderboard_metrics_map.items():
        name = lm.target_name if lm.target_name else lm.global_metric.name
        if name not in name_to_lmids:
            name_to_lmids[name] = []
        name_to_lmids[name].append(lmid)

    # State for consumption
    consumed_counts = {}

    # Convert selected_metrics to use IDs for LeaderboardMetrics and PRUNE stale ones
    updated_selected = []
    for m in selected_metrics:
        if m in name_to_lmids:
            # If the metric name maps to multiple IDs (flavours), we add all of them
            # Sort them by ID to ensure consistent order
            mapped_ids = sorted(name_to_lmids[m])
            
            for mid in mapped_ids:
                if mid not in updated_selected:
                    updated_selected.append(mid)
        elif m in leaderboard_metrics_map:
            # Already a valid unique ID
            # [FIX] Also expand to peers (other flavours of the same metric name) 
            # to ensure we show all variants even if only one was saved as ID.
            lm = leaderboard_metrics_map[m]
            target_name = lm.target_name if lm.target_name else lm.global_metric.name
            
            if target_name in name_to_lmids:
                 mapped_ids = sorted(name_to_lmids[target_name])
                 for mid in mapped_ids:
                     if mid not in updated_selected:
                         updated_selected.append(mid)
            else:
                 if m not in updated_selected:
                    updated_selected.append(m)
        elif m in custom_metrics:
            # Valid static or custom submission metric
            updated_selected.append(m)
        else:
            # Legacy/stale name that doesn't match any current metric
            # PRUNED: do not add to updated_selected
            pass
    
    # Auto-migrate summary_metrics to use unique IDs permanently and REMOVE stale names
    if updated_selected != selected_metrics:
        leaderboard.summary_metrics = ','.join(updated_selected)
        
        # Proactively clean up JSON settings to remove stale/pruned metrics
        valid_set = set(updated_selected)
        
        if leaderboard.metric_directions:
            try:
                directions = json.loads(leaderboard.metric_directions)
                new_directions = {k: v for k, v in directions.items() if k in valid_set}
                if len(new_directions) != len(directions):
                    leaderboard.metric_directions = json.dumps(new_directions)
            except: pass
            
        if leaderboard.metric_aggregation:
            try:
                aggs = json.loads(leaderboard.metric_aggregation)
                new_aggs = {k: v for k, v in aggs.items() if k in valid_set}
                if len(new_aggs) != len(aggs):
                    leaderboard.metric_aggregation = json.dumps(new_aggs)
            except: pass

        db.session.commit()
    
    selected_metrics = updated_selected

    # discovered_metrics contains lm_ids (strings) and custom_metrics (names)
    discovered_metrics = set(custom_metrics) | set(leaderboard_metrics_map.keys())
    all_metrics = list(selected_metrics)
    
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
            
            # Use the internal ID to avoid collisions
            metric_identifier = f"lm_{res.leaderboard_metric_id}"
            
            val = res.value
            if res.error_message:
                val = str(res.error_message)

            # Store by ID
            calculated_dynamic_values[res.submission_id][metric_identifier] = val
            
            # Store by Global Name
            calculated_dynamic_values[res.submission_id][res.leaderboard_metric.global_metric.name] = val
            
            # Store by Target Name (if exists)
            if res.leaderboard_metric.target_name:
                calculated_dynamic_values[res.submission_id][res.leaderboard_metric.target_name] = val


        for metric in all_metrics:
            if metric in leaderboard_metrics_map:
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
                        CustomField.field_type.in_(['metric', 'scalar'])
                    )
                    
                    # Apply sample filters by sample_name (not via Sample join)
                    
                    
                    # Apply Normalized filters by getting matching sample names first
                    if current_sample_filters:
                        dataset_ids = [d.id for d in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
                        sample_names_query = db.session.query(Sample.name).filter(Sample.dataset_id.in_(dataset_ids))
                        
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
                             except Exception:
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
            val = calculated_dynamic_values.get(sub.id, {}).get(sort_metric)
            return val if val is not None else float('inf')
        
        submissions.sort(key=get_metric_value, reverse=(sort_order == 'desc'))

    # Prepare Aggregation Info for UI
    from sqlalchemy import inspect
    # Reuse loop logic or helper? Helper is better but inline for speed now.
    metric_agg_info = {}
    current_agg_config = json.loads(leaderboard.metric_aggregation) if leaderboard.metric_aggregation else {}
    
    for metric in all_metrics:
        label = None
        # Check dynamic
        lm = next((m for m in leaderboard.leaderboard_metrics if f"lm_{m.id}" == metric), None)
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
            target = f"lm_{lm.id}"
            if lm.sort_direction:
                metric_directions_dict[target] = lm.sort_direction

    return render_template('leaderboard.html', 
                           leaderboard=leaderboard,
                           submissions=submissions,
                           all_metrics=all_metrics,
                           selected_metrics=all_metrics,
                           metrics_ranges=metrics_ranges,
                            dynamic_values=calculated_dynamic_values,
                            metric_to_lm=leaderboard_metrics_map,
                            metric_labels=metric_labels,
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

@app.route('/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    dataset_names_input = request.form.get('dataset_name', '')
    files = request.files.getlist('dataset_zip')

    if not files:
        flash("No files uploaded.", "warning")
        return redirect(url_for('datasets_list'))

    # If names are provided, split them by comma. Otherwise, auto-generate from filenames.
    if dataset_names_input:
        provided_names = [name.strip() for name in dataset_names_input.split(',') if name.strip()]
    else:
        provided_names = [file.filename.replace('.zip', '') for file in files]

    if len(provided_names) != len(files):
        flash("Number of names provided does not match number of files uploaded.", "danger")
        return redirect(url_for('datasets_list'))

    for i, file in enumerate(files):
        dataset_name = provided_names[i]
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_dataset_pre_process')
        os.makedirs(temp_dir, exist_ok=True)
        temp_zip_path = os.path.join(temp_dir, filename)
        file.save(temp_zip_path)

        # Phase 7 quota gate: count + projected-bytes (the ZIP itself is a
        # rough upper bound on the extracted size for the cap math).
        incoming = _path_size_bytes(temp_zip_path)
        ok, msg = check_quota(g.current_user, kind='dataset_create', incoming_bytes=incoming)
        if not ok:
            try:
                os.remove(temp_zip_path)
            except OSError:
                pass
            flash(msg, "danger")
            continue

        success, message, ds_id = process_dataset_zip(
            temp_zip_path, dataset_name,
            owner_user_id=g.current_user.id,
        )

        if success:
            flash(message, "success")
        else:
            flash(message, "danger")

        # Cleanup temp zip
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    return redirect(url_for('datasets_list'))


# ===================== HuggingFace BYO (Phase 4 — simple) =====================
# Constraint: the HF repo MUST already follow BenchHub's folder convention
# (metric_*/, hist_*/, raw_*/, image_*/ folders with one file per sample).
# We don't translate arbitrary HF Datasets schemas — that's a much bigger
# design problem that needs explicit user input on which datasets to support.
# This path is "snapshot a structured repo + reuse the existing ZIP pipeline."

def import_dataset_from_hf(repo_id, dataset_name, *, revision=None,
                           hf_token=None, owner_user_id=None):
    """Pull an HF repo to a temp dir, zip it, and feed it to
    process_dataset_zip. Returns the same (success, message, ds_id) tuple.

    `huggingface_hub` is imported lazily so the rest of the app boots
    without it (and so tests can patch the import path).
    """
    from huggingface_hub import snapshot_download

    work_dir = tempfile.mkdtemp(prefix='benchhub-hf-')
    try:
        snap_dir = snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            revision=revision,
            token=hf_token,
            local_dir=os.path.join(work_dir, 'snap'),
        )
        # Re-zip the snapshot so process_dataset_zip can ingest it without
        # needing a directory-mode branch. The cost is one extra
        # write/read of the bytes; cheap relative to the network pull.
        zip_base = os.path.join(work_dir, secure_filename(dataset_name))
        zip_path = shutil.make_archive(zip_base, 'zip', root_dir=snap_dir)
        return process_dataset_zip(
            zip_path, dataset_name,
            owner_user_id=owner_user_id,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.route('/import_from_hf', methods=['POST'])
@login_required
def import_from_hf():
    repo_id = (request.form.get('hf_repo_id') or '').strip()
    dataset_name = (request.form.get('dataset_name') or '').strip()
    revision = (request.form.get('hf_revision') or '').strip() or None
    hf_token = (request.form.get('hf_token') or '').strip() or None

    if not repo_id:
        flash("Missing HuggingFace repo ID.", "danger")
        return redirect(url_for('datasets_list'))
    if not dataset_name:
        # Default to the repo's last segment.
        dataset_name = repo_id.rstrip('/').split('/')[-1]

    # Quota gate: count check only (we don't know the size until we pull).
    # The storage cap re-checks after extraction inside process_dataset_zip
    # would be ideal, but we trust the count gate + the post-pull caller
    # to abort if the resulting dataset would push storage over.
    ok, msg = check_quota(g.current_user, kind='dataset_create', incoming_bytes=0)
    if not ok:
        flash(msg, "danger")
        return redirect(url_for('datasets_list'))

    try:
        success, message, _ = import_dataset_from_hf(
            repo_id, dataset_name,
            revision=revision, hf_token=hf_token,
            owner_user_id=g.current_user.id,
        )
        flash(message, "success" if success else "danger")
    except ImportError:
        flash("HuggingFace import is not available on this server (huggingface_hub not installed).", "danger")
    except Exception as e:
        flash(f"HuggingFace import failed: {e}", "danger")

    return redirect(url_for('datasets_list'))


# --- HuggingFace auto-import (Level 2) -----------------------------------
# Inspect a HF parquet dataset's `features` schema, infer which columns
# belong in BenchHub's image_/raw_/metric_/hist_ folders, show the user
# the inferred mapping, and on confirm stream the rows into a
# BenchHub-shaped folder + run them through process_dataset_zip-equivalent
# logic. Operator can override the mapping before kicking off.
#
# Why parquet only: it's the dominant HF Datasets format and we get
# typed schema info without downloading shards. WebDataset / Arrow could
# follow the same pattern, but adapting them is its own slice.

# Pixel-resolutions that look "depth-y" — used in the inference rules.
_HF_DEPTH_NAMES = {'depth', 'depth_map', 'gt_depth', 'disparity'}
_HF_RGB_NAMES = {'image', 'rgb', 'color', 'photo', 'pixel_values'}
_HF_METRIC_NAMES = {'label', 'class', 'score', 'metric', 'target', 'y'}


def _hf_fetch_features(repo_id, revision=None, hf_token=None):
    """Pull the `features` JSON from huggingface.co/api/datasets/<repo>.
    Returns a dict {column_name: feature_descriptor} or {} if unavailable.
    Anonymous when no token; uses the token for gated repos.
    """
    import requests as _r
    headers = {}
    if hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    if revision:
        url += f"?revision={revision}"
    resp = _r.get(url, headers=headers, timeout=15)
    if resp.status_code == 401:
        raise RuntimeError(
            "HuggingFace returned 401: gated dataset. "
            "Accept its terms on huggingface.co and pass an access token."
        )
    resp.raise_for_status()
    info = resp.json()
    # The features can live in a few nested places depending on how the
    # dataset was uploaded. Walk the common ones.
    card = info.get('cardData') or {}
    dataset_info_blocks = card.get('dataset_info') or info.get('dataset_info') or []
    if isinstance(dataset_info_blocks, dict):
        dataset_info_blocks = [dataset_info_blocks]
    for blk in dataset_info_blocks:
        feats = blk.get('features')
        if feats:
            return _normalize_features(feats)
    return {}


def _normalize_features(feats):
    """HF stores features as either a list of {name, dtype/...} or a dict.
    Normalize to {name: {type: <kind>, length: <opt>}} where kind is one
    of 'Image', 'Audio', 'ClassLabel', 'Value:<dtype>', 'Sequence:<inner>',
    'unknown'."""
    out = {}
    if isinstance(feats, dict):
        items = feats.items()
    elif isinstance(feats, list):
        items = [(f.get('name'), f) for f in feats if isinstance(f, dict) and f.get('name')]
    else:
        return out
    for name, desc in items:
        if not isinstance(desc, dict):
            out[name] = {'type': 'unknown'}
            continue
        # HF feature `_type` (or the implicit shape via dtype) tells us what it is.
        kind = desc.get('_type') or desc.get('feature', {}).get('_type') or 'Value'
        if kind == 'Image':
            out[name] = {'type': 'Image'}
        elif kind == 'Audio':
            out[name] = {'type': 'Audio'}
        elif kind == 'ClassLabel':
            out[name] = {'type': 'ClassLabel', 'names': desc.get('names', [])}
        elif kind == 'Value':
            out[name] = {'type': f"Value:{desc.get('dtype', 'unknown')}"}
        elif kind == 'Sequence':
            inner = desc.get('feature') or {}
            inner_t = inner.get('_type') or 'Value'
            inner_dtype = inner.get('dtype', '') if inner_t == 'Value' else inner_t
            out[name] = {
                'type': f"Sequence:{inner_dtype}",
                'length': desc.get('length', -1),
            }
        else:
            out[name] = {'type': kind}
    return out


def _infer_mapping(features):
    """Heuristic: map each feature into a BenchHub field type, or 'skip'.
    Returns a list of {column, target_kind, target_field, reason}."""
    out = []
    for col, desc in features.items():
        col_lc = col.lower()
        t = desc.get('type', '')
        if t == 'Image':
            if any(k in col_lc for k in _HF_DEPTH_NAMES):
                out.append({'column': col, 'target_kind': 'depth',
                            'target_field': f'raw_{col}',
                            'reason': "Image-typed column with depth-suggesting name"})
            else:
                out.append({'column': col, 'target_kind': 'image',
                            'target_field': f'image_{col}',
                            'reason': "Image-typed column → image_*"})
        elif t.startswith('Value:'):
            dtype = t.split(':', 1)[1]
            if dtype in ('int8', 'int16', 'int32', 'int64',
                         'uint8', 'uint16', 'uint32',
                         'float16', 'float32', 'float64', 'bool'):
                out.append({'column': col, 'target_kind': 'metric',
                            'target_field': f'metric_{col}',
                            'reason': f"Numeric scalar ({dtype}) → metric_*"})
            elif dtype == 'string':
                if col_lc in ('caption', 'text', 'tag', 'tags'):
                    out.append({'column': col, 'target_kind': 'text',
                                'target_field': col,
                                'reason': "Text-shaped column"})
                else:
                    out.append({'column': col, 'target_kind': 'skip',
                                'target_field': '',
                                'reason': "String column with no obvious mapping"})
            else:
                out.append({'column': col, 'target_kind': 'skip',
                            'target_field': '',
                            'reason': f"Unknown dtype '{dtype}'"})
        elif t == 'ClassLabel':
            out.append({'column': col, 'target_kind': 'metric',
                        'target_field': f'metric_{col}',
                        'reason': "ClassLabel → store integer index as metric_*"})
        elif t.startswith('Sequence:'):
            inner = t.split(':', 1)[1]
            length = desc.get('length', -1)
            if inner in ('int32', 'int64', 'uint8', 'uint16', 'uint32') and length in (256, 512, 1024, 2048):
                out.append({'column': col, 'target_kind': 'histogram',
                            'target_field': f'hist_{col}',
                            'reason': f"Fixed-length int sequence ({length}) → hist_*"})
            else:
                out.append({'column': col, 'target_kind': 'skip',
                            'target_field': '',
                            'reason': f"Sequence:{inner} (length={length}) — no auto-mapping"})
        else:
            out.append({'column': col, 'target_kind': 'skip',
                        'target_field': '',
                        'reason': f"Unsupported feature type '{t}'"})
    return out


def _import_hf_auto(repo_id, dataset_name, mapping, *, sample_cap=200,
                   split='train', revision=None, hf_token=None,
                   owner_user_id=None):
    """Stream up to `sample_cap` rows from the HF dataset, lay them out
    as BenchHub folders per the mapping, and feed the result through the
    existing process_dataset_zip pipeline by re-zipping the folder.

    `mapping` is a list of {column, target_kind, target_field}. Anything
    with target_kind='skip' is ignored. Returns (success, message, ds_id).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return False, ("HF auto-import needs the `datasets` package; "
                       "install it (or rebuild the image with it pinned)."), None

    work_dir = tempfile.mkdtemp(prefix='benchhub-hf-auto-')
    try:
        # Pre-create folders for each non-skip mapping target.
        for m in mapping:
            kind = m.get('target_kind')
            if kind in ('image', 'depth', 'histogram', 'metric', 'text'):
                os.makedirs(os.path.join(work_dir, m['target_field']), exist_ok=True)
        # README at root keeps process_dataset_zip from picking the only
        # populated subfolder as the dataset wrapper.
        with open(os.path.join(work_dir, 'README.md'), 'w') as f:
            f.write(
                f"# {dataset_name}\n\nAuto-imported from HuggingFace `{repo_id}` "
                f"(first {sample_cap} samples, revision={revision or 'main'}).\n"
            )

        ds = load_dataset(repo_id, split=split, streaming=True, revision=revision,
                          token=hf_token)

        from PIL import Image as _PILImage
        import numpy as _np

        n_written = 0
        for example in ds:
            if n_written >= sample_cap:
                break
            sample_id = f"s{n_written:05d}"
            wrote_anything = False
            for m in mapping:
                col = m['column']
                kind = m.get('target_kind')
                if kind == 'skip' or col not in example:
                    continue
                value = example[col]
                target_dir = os.path.join(work_dir, m['target_field'])
                try:
                    if kind == 'image' and value is not None:
                        img = value if isinstance(value, _PILImage.Image) else _PILImage.fromarray(_np.array(value))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(os.path.join(target_dir, f"{sample_id}.png"), 'PNG')
                        wrote_anything = True
                    elif kind == 'depth' and value is not None:
                        if isinstance(value, _PILImage.Image):
                            arr = _np.array(value)
                        else:
                            arr = _np.array(value)
                        if arr.ndim == 3 and arr.shape[2] == 1:
                            arr = arr.squeeze(-1)
                        h, w = arr.shape[:2]
                        _np.savez(os.path.join(target_dir, f"{sample_id}_{w}x{h}.npz"),
                                  depth=arr)
                        wrote_anything = True
                    elif kind == 'metric' and value is not None:
                        with open(os.path.join(target_dir, f"{sample_id}.txt"), 'w') as f:
                            f.write(str(float(value)))
                        wrote_anything = True
                    elif kind == 'text' and value is not None:
                        with open(os.path.join(target_dir, f"{sample_id}.txt"), 'w') as f:
                            f.write(str(value))
                        wrote_anything = True
                    elif kind == 'histogram' and value is not None:
                        counts = _np.array(value, dtype='int64')
                        bins = _np.arange(len(counts) + 1, dtype='int64')
                        _np.savez(os.path.join(target_dir, f"{sample_id}.npz"),
                                  bins=bins, counts=counts)
                        wrote_anything = True
                except Exception as conv_err:
                    print(f"Skipping sample {sample_id} col {col}: {conv_err}")
            if wrote_anything:
                n_written += 1

        if n_written == 0:
            return False, "No samples could be converted with the chosen mapping.", None

        # Zip the work_dir and feed it to the existing pipeline.
        zip_base = os.path.join(work_dir, secure_filename(dataset_name) + '_zipped')
        zip_path = shutil.make_archive(zip_base, 'zip', root_dir=work_dir)
        success, message, ds_id = process_dataset_zip(
            zip_path, dataset_name, owner_user_id=owner_user_id,
        )
        if success and ds_id is not None:
            ds = Dataset.query.get(ds_id)
            if ds is not None:
                ds.source_kind = 'hf-parquet'
                ds.source_metadata = json.dumps({
                    'repo_id': repo_id,
                    'revision': revision,
                    'split': split,
                    'sample_cap': sample_cap,
                    'mapping': mapping,
                    'samples_written': n_written,
                })
                db.session.commit()
        return success, message, ds_id
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.route('/import_from_hf/preview', methods=['POST'])
@login_required
def import_from_hf_preview():
    """Step 1 of the auto-import flow: paste repo, get back the inferred
    mapping for review. Renders a confirmation page.
    """
    repo_id = (request.form.get('hf_repo_id') or '').strip()
    revision = (request.form.get('hf_revision') or '').strip() or None
    hf_token = (request.form.get('hf_token') or '').strip() or None
    sample_cap = int(request.form.get('sample_cap') or 200)
    sample_cap = max(10, min(sample_cap, 2000))
    dataset_name = (request.form.get('dataset_name') or '').strip()
    if not dataset_name:
        dataset_name = repo_id.rstrip('/').split('/')[-1] if repo_id else ''

    if not repo_id:
        flash("Missing HuggingFace repo ID.", "danger")
        return redirect(url_for('datasets_list'))

    try:
        features = _hf_fetch_features(repo_id, revision=revision, hf_token=hf_token)
    except Exception as e:
        msg = str(e)
        if '401' in msg or 'gated' in msg.lower():
            flash(
                f"HuggingFace returned 401 — '{repo_id}' is a gated dataset. "
                "Accept its terms on huggingface.co/datasets/" + repo_id +
                " first, then paste an access token under Advanced.",
                "danger",
            )
        else:
            flash(f"Couldn't read schema: {e}", "danger")
        return redirect(url_for('datasets_list'))

    if not features:
        flash(
            f"No `features` schema found in {repo_id}. The dataset may not "
            "be parquet-formatted; try the manual import path instead.",
            "warning",
        )
        return redirect(url_for('datasets_list'))

    mapping = _infer_mapping(features)
    return render_template(
        'hf_import_preview.html',
        repo_id=repo_id,
        revision=revision,
        hf_token=hf_token,
        dataset_name=dataset_name,
        sample_cap=sample_cap,
        features=features,
        mapping=mapping,
    )


@app.route('/import_from_hf/auto', methods=['POST'])
@login_required
def import_from_hf_auto():
    """Step 2: form has the (potentially edited) mapping; run the import."""
    repo_id = (request.form.get('hf_repo_id') or '').strip()
    revision = (request.form.get('hf_revision') or '').strip() or None
    hf_token = (request.form.get('hf_token') or '').strip() or None
    dataset_name = (request.form.get('dataset_name') or '').strip()
    sample_cap = max(10, min(int(request.form.get('sample_cap') or 200), 2000))

    if not repo_id or not dataset_name:
        flash("Missing repo or dataset name.", "danger")
        return redirect(url_for('datasets_list'))

    # Mapping comes back as parallel arrays (one per row in the preview).
    cols = request.form.getlist('mapping_column[]')
    kinds = request.form.getlist('mapping_target_kind[]')
    fields = request.form.getlist('mapping_target_field[]')
    mapping = []
    for col, kind, field in zip(cols, kinds, fields):
        if not col:
            continue
        mapping.append({
            'column': col,
            'target_kind': kind,
            'target_field': field or col,
        })

    ok, msg = check_quota(g.current_user, kind='dataset_create', incoming_bytes=0)
    if not ok:
        flash(msg, "danger")
        return redirect(url_for('datasets_list'))

    try:
        success, message, ds_id = _import_hf_auto(
            repo_id, dataset_name, mapping,
            sample_cap=sample_cap, revision=revision, hf_token=hf_token,
            owner_user_id=g.current_user.id,
        )
        flash(message, "success" if success else "danger")
        if success and ds_id:
            return redirect(url_for('dataset_view', dataset_id=ds_id))
    except Exception as e:
        flash(f"HuggingFace auto-import failed: {e}", "danger")
    return redirect(url_for('datasets_list'))


# --- HuggingFace dataset listing (Round C) -------------------------------
# Proxy the HF public datasets index so users can pick a repo from a
# searchable list inside BenchHub instead of context-switching to
# huggingface.co. Server-side cache keeps us under HF's anonymous rate
# limit (the result is the same for everyone for an hour).

_hf_datasets_cache = {'fetched_at': 0.0, 'sort': None, 'q': '', 'rows': []}


def _fetch_hf_datasets(*, sort='likes', q='', limit=50):
    """Hit huggingface.co/api/datasets and return a list of dicts. Cache
    for 1h on (sort, q). Returns [] on any error so the UI degrades to
    the manual-entry path."""
    import time as _time
    now = _time.time()
    if (_hf_datasets_cache['sort'] == sort
            and _hf_datasets_cache['q'] == q
            and now - _hf_datasets_cache['fetched_at'] < 3600
            and _hf_datasets_cache['rows']):
        return _hf_datasets_cache['rows']

    try:
        import requests as _r
        url = "https://huggingface.co/api/datasets"
        params = {'sort': sort, 'limit': limit, 'full': 'false'}
        if q:
            params['search'] = q
        resp = _r.get(url, params=params, timeout=8)
        resp.raise_for_status()
        raw = resp.json() or []
        rows = []
        for entry in raw:
            rid = entry.get('id') or ''
            if not rid:
                continue
            rows.append({
                'id': rid,
                'downloads': int(entry.get('downloads') or 0),
                'likes': int(entry.get('likes') or 0),
                'updated': entry.get('lastModified') or '',
                'tags': [t for t in (entry.get('tags') or []) if isinstance(t, str)][:6],
            })
        _hf_datasets_cache.update({
            'fetched_at': now, 'sort': sort, 'q': q, 'rows': rows,
        })
        return rows
    except Exception as e:
        print(f"_fetch_hf_datasets failed: {e}")
        return []


@app.route('/api/hf/datasets')
@login_required
def api_hf_datasets():
    """JSON listing of HuggingFace datasets, used by the picker on /datasets.

    Query string: ?sort=likes|downloads|trending  ?q=<keyword>
    """
    sort = request.args.get('sort', 'likes')
    if sort not in ('likes', 'downloads', 'trending'):
        sort = 'likes'
    q = (request.args.get('q') or '').strip()
    rows = _fetch_hf_datasets(sort=sort, q=q)
    return jsonify({'rows': rows, 'sort': sort, 'q': q})


@app.route('/create_leaderboard', methods=['POST'])
@login_required
def create_leaderboard():
    leaderboard_name = request.form['leaderboard_name']

    # Names are global now (no project namespace).
    existing = Leaderboard.query.filter_by(name=leaderboard_name).first()
    if existing:
        if request.form.get('overwrite'):
            db.session.delete(existing)
            db.session.commit()
            flash(f'Overwriting existing leaderboard "{leaderboard_name}".', 'warning')
        else:
            flash(f'Leaderboard "{leaderboard_name}" already exists. Choose a different name or check "Overwrite".', 'danger')
            return redirect(url_for('datasets_list'))

    new_leaderboard = Leaderboard(
        name=leaderboard_name,
        summary_metrics=','.join(request.form.getlist('summary_metrics')),
        owner_user_id=g.current_user.id,
    )
    
    # Handle multiple datasets
    dataset_ids = request.form.getlist('dataset_ids')
    if not dataset_ids and 'dataset_id' in request.form:
        dataset_ids = [request.form['dataset_id']] # Fallback to single ID if present
        
    if dataset_ids:
        # Link all selected datasets
        datasets = Dataset.query.filter(Dataset.id.in_(dataset_ids)).all()
        new_leaderboard.datasets = datasets
        # Also set the legacy field to the first one for backwards compatibility
        if datasets:
            new_leaderboard.dataset_id = datasets[0].id

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
    return redirect(url_for('datasets_list'))

def process_submission_zip(leaderboard_id, submission_name, zip_path, owner_user_id=None):
    """
    Helper function to process a single submission zip file.
    Create DB entry, extract files, and queue processing task.
    owner_user_id (Phase 1 multi-tenancy): the User who uploaded.
    """
    try:
        new_submission = Submission(
            name=submission_name,
            leaderboard_id=leaderboard_id,
            processing_status='Queued',
            owner_user_id=owner_user_id,
        )
        db.session.add(new_submission)
        db.session.commit() # Commit immediately to release lock and get ID

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
             db.session.commit()
        
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
            
        # --- End clean up and extraction ---

        # Read git_info.json from its final location
        git_info_path = os.path.join(submission_folder, 'git_info.json')
        if not os.path.exists(git_info_path):
            git_info_path = os.path.join(submission_folder, 'git.info')

        if os.path.exists(git_info_path):
            with open(git_info_path, 'r') as git_file:
                git_data = json.load(git_file)
                # Map various possible keys to the model fields
                new_submission.git_commit = git_data.get('commit') or git_data.get('commit_sha')
                new_submission.git_branch = git_data.get('branch') or git_data.get('repo_url') # Fallback repo_url to branch for visibility if branch missing
                new_submission.git_message = git_data.get('message')
                # Extract author from git_info.json or from git commit
                author = git_data.get('author', '')
                if not author and new_submission.git_commit:
                    # Try to extract author from git using the commit hash
                    author = get_author_from_git_commit(new_submission.git_commit, branch_name=new_submission.git_branch)
                new_submission.git_author = author or ''
        
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
        dataset_ids = [d.id for d in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
        dataset_samples = Sample.query.filter(Sample.dataset_id.in_(dataset_ids)).all()
        sample_names = [s.name for s in dataset_samples]
        


        # Only exclude internal metadata folders we definitively don't want as custom fields
        known_folders = {'git.info', '__MACOSX'} 
        
        
        custom_fields = detect_custom_fields(submission_folder, sample_names, known_folders, is_submission=True)
        


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
                else:  # scalar or metric or text
                    if field_type == 'text':
                        custom_field = CustomField(
                            name=field_name,
                            field_type=field_type,
                            value_text=str(value),
                            submission_id=new_submission.id,
                            sample_name=s_name
                        )
                    else: # scalar or metric
                        custom_field = CustomField(
                            name=field_name,
                            field_type=field_type,
                            value_float=float(value),
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

@app.route('/leaderboard/<int:leaderboard_id>/upload_submission', methods=['POST'])
@login_required
def upload_submission(leaderboard_id):
    files = request.files.getlist('submission_zip')
    submission_names_input = request.form.get('submission_name')

    if not files:
        return redirect(url_for('leaderboard_view', leaderboard_id=leaderboard_id))

    # Phase 7 quota gate: rolling 24h submission rate. Checked before the
    # save loop, not inside it — bulk uploads count once at intake.
    ok, msg = check_quota(g.current_user, kind='submission')
    if not ok:
        flash(msg, "danger")
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
                        process_submission_zip(
                            leaderboard_id, sub_name, inner_zip_path,
                            owner_user_id=g.current_user.id,
                        )

            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)

        else:
            # Process as Single
            # HERE we use the provided name if available and valid
            if provided_names and i < len(provided_names):
                sub_name = provided_names[i]
            else:
                sub_name = file.filename.replace('.zip', '')

            process_submission_zip(
                leaderboard_id, sub_name, temp_zip_path,
                owner_user_id=g.current_user.id,
            )

        # Cleanup original upload
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    return redirect(url_for('leaderboard_view', leaderboard_id=leaderboard_id))



@app.route('/submissions/batch_action', methods=['POST'])
def batch_action():
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
        db.session.commit()
        
        tasks.process_submissions_batch_sequential.delay([s.id for s in submissions], sample_filters=sample_filters)
        flash(f'Started sequential recalculation for {len(submissions)} submissions.', 'info')
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


@app.route('/submission/<int:submission_id>/update_tags', methods=['POST'])
@login_required
@owner_required(Submission, 'submission_id')
def update_submission_tags(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    new_tags_str = request.form.get('tags', '').strip()
    
    # Get tag names
    tag_names = [t.strip() for t in new_tags_str.split(',') if t.strip()]
    
    # Clear existing tags
    submission.tags = []
    
    # Add new tags
    for tag_name in tag_names:
        tag = Tag.query.filter_by(name=tag_name).first() or Tag(name=tag_name)
        if tag not in submission.tags:
            submission.tags.append(tag)
    
    db.session.commit()
    flash(f'Tags updated for submission {submission.name}', 'success')
    return redirect(request.referrer or url_for('leaderboard_view', leaderboard_id=submission.leaderboard_id))

@app.route('/leaderboard/<int:leaderboard_id>/update_metrics', methods=['POST'])
def update_leaderboard_metrics(leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    selected_metrics = request.form.getlist('metrics')
    leaderboard.selected_metrics = ','.join(selected_metrics)
    db.session.commit()
    return redirect(url_for('comparison_view', leaderboard_id=leaderboard_id))

@app.route('/comparison/<int:leaderboard_id>')
@visibility_required(Leaderboard, 'leaderboard_id')
def comparison_view(leaderboard_id):
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
                               current_compare_ids=compare_ids_arg,
                               metric_labels=metric_labels)

    # Pagination params
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 5, type=int)
    search_query = request.args.get('search_query', '')
    sort_by = request.args.get('sort_by', '') # Default to empty (no sort)
    sort_order = request.args.get('sort_order', 'asc')

    # Base query for samples
    dataset_ids = [ds.id for ds in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
    samples_query = Sample.query.filter(Sample.dataset_id.in_(dataset_ids))
    
    # Apply search filter
    if search_query:
        samples_query = samples_query.filter(Sample.name.ilike(f'%{search_query}%'))
    
    # Apply tag filters
    samples_query = apply_tag_filters(samples_query, request.args)

    # Collect unique custom field names efficiently
    # Dataset fields
    dataset_custom_fields_query = db.session.query(CustomField.name, CustomField.field_type).join(Sample).filter(
        Sample.dataset_id.in_(dataset_ids),
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
    
    # Map internal IDs to labels for all metrics including dynamic ones
    metric_labels = {}
    for m in all_submission_fields:
        metric_labels[m] = m
    
    leaderboard_metrics_map = { f"lm_{lm.id}": lm for lm in leaderboard.leaderboard_metrics }
    for lmid, lm in leaderboard_metrics_map.items():
        metric_labels[lmid] = f"{lm.target_name if lm.target_name else lm.global_metric.name} ({lmid})"
        all_field_types[lmid] = 'metric'
    
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
            target_sub = db.session.get(Submission, sub_id)
            if target_sub:
                target_lm = next((lm for lm in leaderboard.leaderboard_metrics if (f"lm_{lm.id}" == metric_key or lm.target_name == metric_key or lm.global_metric.name == metric_key)), None)
                submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(target_sub.id))
                
                # Fetch pre-calculated results for the metric if possible
                precalc_results = {res.sample_name: res.value_float for res in db.session.query(CustomField.sample_name, CustomField.value_float).filter(
                    CustomField.submission_id == sub_id,
                    CustomField.name == metric_key,
                    CustomField.field_type.in_(['scalar', 'metric'])
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
                Sample.dataset_id.in_(dataset_ids),
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

    for sample in samples_on_page:
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
                    m_id_name = f"lm_{lm.id}"
                    log_safe_val = get_log_safe_value(val)
                    log_safe_metrics[m_id_name] = log_safe_val
                    
                    # Also add to comparison table
                    if m_id_name not in sample_info['custom_metrics']:
                        sample_info['custom_metrics'][m_id_name] = {'submissions': {}}
                    sample_info['custom_metrics'][m_id_name]['submissions'][sub.id] = val
                    
                    # Ensure it's in pred_data for the per-sample values column
                    pred_data[m_id_name] = val

            sample_chart_metrics_for_this_sample['metrics'][sub.name] = log_safe_metrics
            all_observed_metrics.update(log_safe_metrics.keys())

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
    dataset_ids = [d.id for d in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
    all_sample_tag_names, all_sample_prefixes = get_all_sample_tags(dataset_ids)

    # Collect all custom metrics from submissions
    custom_metrics = set()
    for sub in submissions:
        for cf in sub.custom_fields:
            if cf.field_type == 'metric':
                custom_metrics.add(cf.name)
    
    # Deduplicate: Remove names that are already represented as leaderboard metrics
    lb_metric_names = {lm.global_metric.name for lm in leaderboard.leaderboard_metrics} | \
                      {lm.target_name for lm in leaderboard.leaderboard_metrics if lm.target_name}
    custom_metrics = {m for m in custom_metrics if m not in lb_metric_names and not m.startswith('lm_')}
    
    # Add dynamic metrics names to custom_metrics (IDs already handled in transmissions_json)
    # leaderboard_metrics_map = {lm.global_metric.name: lm for lm in leaderboard.leaderboard_metrics}
    # for dm_name, lm in leaderboard_metrics_map.items():
    #     custom_metrics.add(dm_name)

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
        # Use UNIQUE ID (lm_ID) instead of target_name to avoid collisions and match metric_labels
        metric_key = f"lm_{res.leaderboard_metric_id}"
        calculated_dynamic_values[res.submission_id][metric_key] = res.value if not res.error_message else str(res.error_message)

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

            
        # Add values from calculated_dynamic_values (includes both MetricResult and aggregated CustomFields)
        if s.id in calculated_dynamic_values:
            for m_key, m_val in calculated_dynamic_values[s.id].items():
                sub_data[m_key] = m_val
        
        # Ensure we also populate by names for all leaderboard metrics to match chart labels
        for lm in leaderboard.leaderboard_metrics:
             lmid = f"lm_{lm.id}"
             val = calculated_dynamic_values.get(s.id, {}).get(lmid)
             if val is not None:
                 # Provide by global name and target name if they were selected in settings
                 sub_data[lm.global_metric.name] = val
                 if lm.target_name:
                      sub_data[lm.target_name] = val

        submissions_json.append(sub_data)
    # Discovery moved early for sorting
    
    
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
    agg_lm_ids = {f"lm_{lm.id}" for lm in leaderboard.leaderboard_metrics if lm.global_metric.is_aggregated}
    agg_metric_names = {lm.global_metric.name for lm in leaderboard.leaderboard_metrics if lm.global_metric.is_aggregated}
    
    raw_selected_metrics = [m for m in leaderboard.selected_metrics.split(',') if m.strip()]
    
    active_metrics = []
    for m in raw_selected_metrics:
        if m in agg_lm_ids: continue
        if m in agg_metric_names: continue
        active_metrics.append(m)
    
 
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
            # Try to use metric_labels if available, else m
            sample_metric_options_dynamic[m] = metric_labels.get(m, m)

    # Clean up available_display_options: remove metrics that should now be handled via sample_metric_options_dynamic
    # Actually, they weren't added to available_display_options yet in dataset_view, but I should be careful.
    
    # Determine available columns based on data existence
    available_display_options = COMPARISON_DISPLAY_OPTIONS.copy()
    
    # Inject Custom Fields into display options with proper ordering
    # (Redundant copy removed, variable already updated above)
    
    # Ensure all selected_metrics (including aggregated ones) have labels for the Chart
    for m in all_selected_metrics:
        if m not in sample_metric_options_dynamic:
             sample_metric_options_dynamic[m] = metric_labels.get(m, m)
             
    # Ensure aggregated metrics specifically are covered (since they might be filtered out of active_metrics)
    for lmid in agg_lm_ids:
        if lmid not in sample_metric_options_dynamic:
            sample_metric_options_dynamic[lmid] = metric_labels.get(lmid, lmid)

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

    
    # Filter custom_metrics to exclude aggregated metrics (like F1 score) so they don't appear in per-sample charts/tables
    per_sample_custom_metrics = [m for m in sorted(list(custom_metrics)) if m not in agg_metric_names and m not in agg_lm_ids]
    
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
                           metric_labels=metric_labels,
                           active_metrics=active_metrics,
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
    # Belt-and-suspenders: clean up orphan rows on the way in so the list
    # reflects what's actually usable. process_dataset_zip's except handler
    # is the primary defense; this catches anything that escaped (e.g. a
    # SIGKILL'd worker mid-extraction). Cheap because the typical case
    # finds nothing.
    try:
        prune_incomplete_datasets()
    except Exception as e:
        # Never let cleanup break the listing.
        print(f"datasets_list inline prune failed (non-fatal): {e}")

    datasets = (
        Dataset.query
        .filter(visible_in_list(Dataset, getattr(g, 'current_user', None)))
        .order_by(Dataset.upload_date.desc())
        .all()
    )
    
    # Pre-calculate activity stats for leaderboards used by these datasets
    now = datetime.utcnow()
    for ds in datasets:
        # ds.leaderboards is a dynamic relationship query object
        lbs = ds.leaderboards.all()
        ds.active_leaderboards = []
        
        # Track the very latest activity across all LBs for this dataset for sorting
        ds_last_activity = ds.upload_date
        
        for lb in lbs:
            submissions = lb.submissions # This is lazy loaded
            
            # Find the last submission date for this LB
            lb_last_sub = None
            if submissions:
                lb_last_sub = max(s.upload_date for s in submissions)
            
            # Update dataset's overall max activity:
            # 1. Check LB creation date
            if lb.upload_date > ds_last_activity:
                ds_last_activity = lb.upload_date
                
            # 2. Check last submission date
            if lb_last_sub and lb_last_sub > ds_last_activity:
                ds_last_activity = lb_last_sub
            
            # Check for recent activity (using same 7-day window as active)
            subs_last_24h = sum(1 for s in submissions if s.upload_date > now - timedelta(days=7)) 
            
            # Create a simple dict for the template
            ds.active_leaderboards.append({
                'id': lb.id,
                'name': lb.name,
                'subs_last_24h': subs_last_24h
            })
        
        # Store for sorting
        ds.last_associated_activity = ds_last_activity

    # Sort datasets by their last associated activity (most recent first)
    datasets.sort(key=lambda x: x.last_associated_activity, reverse=True)
            
    return render_template('datasets.html', datasets=datasets)

@app.route('/author_avatars/<filename>')
def serve_author_avatar(filename):
    avatar_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'author_avatars')
    return send_from_directory(avatar_dir, filename)

@app.route('/dataset/<int:dataset_id>')
@visibility_required(Dataset, 'dataset_id')
def dataset_view(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    page = request.args.get('page', 1, type=int)
    samples_per_page = request.args.get('per_page', 5, type=int)
    sort_by = request.args.get('sort_by', 'name')
    sort_order = request.args.get('sort_order', 'asc')
    selected_display_columns = dataset.display_columns.split(',')
    
    # 1. OPTIMIZED TAGS/PREFIXES COLLECTION
    distinct_tags_query = db.session.query(func.distinct(Sample.tags)).filter(Sample.dataset_id == dataset.id).all()
    all_dataset_tags = set()
    all_dataset_prefixes = set()
    for t_str, in distinct_tags_query:
        if t_str:
            for t in t_str.split(','):
                cleaned_t = t.strip()
                if cleaned_t:
                    all_dataset_tags.add(cleaned_t)
                    if ':' in cleaned_t:
                        all_dataset_prefixes.add(cleaned_t.split(':')[0])
    all_dataset_tags = sorted(list(all_dataset_tags))
    all_dataset_prefixes = sorted(list(all_dataset_prefixes))

    samples_query = Sample.query.filter_by(dataset_id=dataset.id)
    # Apply tag filters (port from comparison view)
    samples_query = apply_tag_filters(samples_query, request.args)

    # Collect unique custom field names from the dataset early for sorting/display
    # Use a distinct query instead of iterating through all samples
    custom_field_query = db.session.query(CustomField.name, CustomField.field_type).join(Sample).filter(
        Sample.dataset_id == dataset.id,
        CustomField.submission_id == None
    ).distinct().all()
    
    custom_field_names = set(custom_field_query)
    custom_scalar_metric_names = [name for name, ftype in custom_field_names if ftype in ('scalar', 'metric')]

    # Sorting
    if sort_by == 'name':
        if sort_order == 'desc':
            samples_query = samples_query.order_by(Sample.name.desc())
        else:
            samples_query = samples_query.order_by(Sample.name.asc())
    elif sort_by in custom_scalar_metric_names:
        # Optimized sort by custom field
        samples_query = samples_query.outerjoin(
            CustomField, 
            and_(CustomField.sample_id == Sample.id, CustomField.name == sort_by, CustomField.submission_id == None)
        )
        if sort_order == 'desc':
            samples_query = samples_query.order_by(CustomField.value_float.desc())
        else:
            samples_query = samples_query.order_by(CustomField.value_float.asc())

    # Pagination with eager loading to avoid N+1 queries
    # Load related data in bulk for better performance
    # Note: histogram_data and config_data are @property methods that shadow relationships,
    # so we can't eager load them. We only eager load custom_fields.
    from sqlalchemy.orm import joinedload
    
    samples_query = samples_query.options(
        joinedload(Sample.custom_fields)
    )
    
    total = samples_query.count()
    paginated_samples = samples_query.paginate(page=page, per_page=samples_per_page, error_out=False)

    # 6. ROW RENDERING DATA PREPARATION (Fix N+1)
    current_sample_ids = [s.id for s in paginated_samples.items]
    hist_map = {h.sample_id: h for h in HistogramData.query.filter(HistogramData.sample_id.in_(current_sample_ids)).all()}
    shape_map = {s.id: s for s in SignalShape.query.filter(SignalShape.id.in_(current_sample_ids)).all()}

    active_metrics = [m for m in dataset.selected_metrics.split(',') if m.strip()]

    dataset_metrics_data = [] # For per_sample_metrics chart
    samples_data_for_charts = []
    
    # Create a map of sample_id -> {field_name: value}
    # Only for samples on current page
    custom_fields_map = {}

    # Only process samples on the current page
    for sample in paginated_samples.items:
        # Resolve histogram data using our bulk-fetched map + eager loaded custom_fields
        sample_hist = hist_map.get(sample.id)
        if not sample_hist:
            hist_cf = next((cf for cf in sample.custom_fields if cf.name == 'hist' and cf.field_type == 'histogram'), None)
            if hist_cf and hist_cf.value_text:
                try:
                    data = json.loads(hist_cf.value_text)
                    class MockHist: pass
                    sample_hist = MockHist()
                    sample_hist.bins = json.dumps(data['bins'])
                    sample_hist.counts = json.dumps(data['counts'])
                except Exception: pass

        # Resolve signal shape
        sample_shape = shape_map.get(sample.id)
        if not sample_shape:
            shape_cf = next((cf for cf in sample.custom_fields if cf.name == 'wave_shape' and cf.field_type == 'scalar'), None)
            if shape_cf:
                class MockShape: pass
                sample_shape = MockShape()
                sample_shape.shape_name = shape_cf.value_text or 'gaussian'

        # Cache on sample object for template access without triggering properties
        sample._resolved_hist = sample_hist
        sample._resolved_shape = sample_shape

        tags_list = [t.strip() for t in sample.tags.split(',')] if sample.tags else []
        
        # Standard metrics
        metrics = {}
        cf_vals = {}
        for cf in sample.custom_fields:
            if cf.submission_id is not None: continue
            
            if cf.field_type in ('scalar', 'metric'):
                metrics[cf.name] = cf.value_float
                cf_vals[cf.name] = {'type': cf.field_type, 'value': cf.value_float, 'field_id': cf.id}
            else:
                cf_vals[cf.name] = {'type': cf.field_type, 'value': cf.value_text, 'field_id': cf.id}

        dataset_metrics_data.append({
            'sample_id': sample.id,
            'sample_name': sample.name,
            'metrics': {'GT': metrics}
        })

        if 'histogram' in selected_display_columns and sample_hist:
            samples_data_for_charts.append({
                'id': sample.id, 
                'name': sample.name, 
                'bins': json.loads(sample_hist.bins), 
                'counts': json.loads(sample_hist.counts), 
                'tags': tags_list
            })
        else:
            samples_data_for_charts.append({'id': sample.id, 'name': sample.name, 'bins': [], 'counts': [], 'tags': tags_list})
        
        custom_fields_map[sample.id] = cf_vals
    
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
    has_tags = bool(all_dataset_tags)
    
    # Efficient existence checks
    has_hist = any(ft == 'histogram' for fn, ft in custom_field_names) or db.session.query(HistogramData.id).join(Sample).filter(Sample.dataset_id == dataset.id).limit(1).first() is not None
    has_shape = any(fn == 'wave_shape' for fn, ft in custom_field_names) or db.session.query(SignalShape.id).join(Sample).filter(Sample.dataset_id == dataset.id).limit(1).first() is not None

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


def _resolve_tags(raw):
    """Turn a free-text comma/whitespace separated string into a list of
    Tag rows, creating any that don't exist yet. Lowercases and strips
    so 'Depth, Segmentation' and 'depth,segmentation' end up the same."""
    names = []
    seen = set()
    for chunk in re.split(r'[,\n]+', raw or ''):
        name = chunk.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    if not names:
        return []
    existing = {t.name: t for t in Tag.query.filter(Tag.name.in_(names)).all()}
    out = []
    for n in names:
        tag = existing.get(n)
        if tag is None:
            tag = Tag(name=n)
            db.session.add(tag)
            db.session.flush()
        out.append(tag)
    return out


@app.route('/dataset/<int:dataset_id>/update_tags', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def update_dataset_tags(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset.tags = _resolve_tags(request.form.get('tags', ''))
    db.session.commit()
    flash("Tags updated.", "success")
    return redirect(request.referrer or url_for('dataset_view', dataset_id=dataset_id))


@app.route('/leaderboard/<int:leaderboard_id>/update_tags', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def update_leaderboard_tags(leaderboard_id):
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    lb.tags = _resolve_tags(request.form.get('tags', ''))
    db.session.commit()
    flash("Tags updated.", "success")
    return redirect(request.referrer or url_for('leaderboard_view', leaderboard_id=leaderboard_id))


# --- Collaborators (private sharing) ---


def _share_helper(row, redirect_url):
    """Add a collaborator by email. Owner-only (caller already gated)."""
    email = (request.form.get('email') or '').strip().lower()
    if not email:
        flash("Email required.", "warning")
        return redirect(redirect_url)
    user = User.query.filter(func.lower(User.email) == email).first()
    if user is None:
        flash(f"No BenchHub user with email '{email}'. They need to sign in once first.", "warning")
        return redirect(redirect_url)
    if user.id == row.owner_user_id:
        flash("You're already the owner — no need to share with yourself.", "info")
        return redirect(redirect_url)
    if user in row.collaborators:
        flash(f"{user.display_name or user.email} already has access.", "info")
        return redirect(redirect_url)
    row.collaborators.append(user)
    db.session.commit()
    flash(f"Granted access to {user.display_name or user.email}.", "success")
    return redirect(redirect_url)


def _unshare_helper(row, user_id, redirect_url):
    user = User.query.get(user_id)
    if user and user in row.collaborators:
        row.collaborators.remove(user)
        db.session.commit()
        flash(f"Removed {user.display_name or user.email}.", "success")
    return redirect(redirect_url)


@app.route('/dataset/<int:dataset_id>/share', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def share_dataset(dataset_id):
    ds = Dataset.query.get_or_404(dataset_id)
    return _share_helper(ds, url_for('dataset_view', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/unshare/<int:user_id>', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def unshare_dataset(dataset_id, user_id):
    ds = Dataset.query.get_or_404(dataset_id)
    return _unshare_helper(ds, user_id, url_for('dataset_view', dataset_id=dataset_id))


@app.route('/leaderboard/<int:leaderboard_id>/share', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def share_leaderboard(leaderboard_id):
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    return _share_helper(lb, url_for('leaderboard_view', leaderboard_id=leaderboard_id))


@app.route('/leaderboard/<int:leaderboard_id>/unshare/<int:user_id>', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def unshare_leaderboard(leaderboard_id, user_id):
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    return _unshare_helper(lb, user_id, url_for('leaderboard_view', leaderboard_id=leaderboard_id))

@app.route('/leaderboard/<int:leaderboard_id>/update_visualizations', methods=['POST'])
def update_leaderboard_visualizations(leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    leaderboard.visualizations = ','.join(request.form.getlist('visualizations'))
    db.session.commit()
    return redirect(request.referrer)

@app.route('/leaderboard/<int:leaderboard_id>/update_comparison_display_columns', methods=['POST'])
def update_comparison_display_columns(leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    cols = request.form.getlist('comparison_display_columns')
    # Use a sentinel value to distinguish "user wants nothing" from "use defaults"
    leaderboard.comparison_display_columns = ','.join(cols) if cols else '__NONE__'
    db.session.commit()
    return redirect(request.referrer)

@app.route('/submission/<int:submission_id>/download')
def download_submission(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    submission_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id))
    zip_path = os.path.join(submission_folder, 'submission.zip')
    
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name=f"{submission.name}.zip")
    else:
        flash("Original submission ZIP not found.", "warning")
        return redirect(url_for('leaderboard_view', leaderboard_id=submission.leaderboard_id))

@app.route('/leaderboard/<int:leaderboard_id>/download_submissions', methods=['POST'])
def download_submissions_bulk(leaderboard_id):
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
@login_required
@owner_required(Dataset, 'dataset_id')
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset_folder_name = secure_filename(dataset.name)
    shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name), ignore_errors=True)
    db.session.delete(dataset)
    db.session.commit()
    return redirect(url_for('datasets_list'))

@app.route('/delete_leaderboard/<int:leaderboard_id>', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def delete_leaderboard(leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    db.session.delete(leaderboard)
    db.session.commit()
    return redirect(url_for('datasets_list'))

@app.route('/delete_submission/<int:submission_id>', methods=['POST'])
@login_required
@owner_required(Submission, 'submission_id')
def delete_submission(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id)), ignore_errors=True)
    db.session.delete(submission)
    db.session.commit()
    return redirect(url_for('leaderboard_view', leaderboard_id=submission.leaderboard_id))

@app.route('/custom_field_image/<int:field_id>')
def serve_custom_field_image(field_id):
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

@app.route('/api/custom_field_depth_data/<int:field_id>')
def serve_custom_field_depth_data(field_id):
    """Serve raw depth data for a custom field as JSON."""
    custom_field = CustomField.query.get_or_404(field_id)
    
    if custom_field.field_type != 'depth':
        return abort(400, description="Not a depth field")
        
    return serve_depth_data(custom_field.value_text)

@app.route('/api/custom_field_json/<int:field_id>')
def serve_custom_field_json(field_id):
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


@app.route('/sample/<int:sample_id>/download')
def download_sample(sample_id):
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
        
        # Custom fields from dataset (full type coverage — previously only
        # scalar+image were included, so depth maps and json/text fields
        # silently dropped from the bundle).
        for cf in sample.custom_fields:
            if cf.field_type == 'scalar':
                zf.writestr(f'ground_truth/{cf.name}/{sample.name}.txt', str(cf.value_float))

            elif cf.field_type in ('image', 'depth', 'json'):
                # All three store a relative path under UPLOAD_FOLDER.
                # Preserve the original filename (depth files carry a
                # `_<W>x<H>` suffix that the importer expects on round-trip).
                src_path = os.path.join(app.config['UPLOAD_FOLDER'], cf.value_text or '')
                if os.path.exists(src_path):
                    arc_filename = os.path.basename(cf.value_text)
                    zf.write(src_path, f'ground_truth/{cf.name}/{arc_filename}')

            elif cf.field_type == 'histogram':
                # value_text is the JSON {bins, counts}; round-trip to .npz
                # so the bundle matches the original ZIP convention.
                try:
                    h = json.loads(cf.value_text)
                    buf = io.BytesIO()
                    np.savez_compressed(buf, bins=np.array(h['bins']), counts=np.array(h['counts']))
                    zf.writestr(f'ground_truth/{cf.name}/{sample.name}.npz', buf.getvalue())
                except Exception:
                    pass

            elif cf.field_type == 'text':
                zf.writestr(f'ground_truth/{cf.name}/{sample.name}.txt', cf.value_text or '')


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

    
def get_all_sample_tags(dataset_ids):
    """
    Retrieves all unique sample tags and prefixes for a given list of datasets.
    Returns (all_tag_names, all_prefixes) as sorted lists.
    """
    if isinstance(dataset_ids, int):
        dataset_ids = [dataset_ids]
        
    all_sample_tags_query = db.session.query(Sample.tags).filter(Sample.dataset_id.in_(dataset_ids)).all()
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
@require_api_token
def dataset_upload_api():
    """Programmatic dataset upload API. Authenticated via Bearer token
    (see /settings/api_tokens). Owner of the new Dataset is the token's
    user; quotas apply same as the interactive upload."""
    if 'dataset_zip' not in request.files:
        return jsonify({'error': 'No dataset_zip file provided'}), 400

    file = request.files['dataset_zip']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    dataset_name = request.form.get('dataset_name', file.filename.replace('.zip', ''))

    filename = secure_filename(file.filename)
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_dataset_api_pre')
    os.makedirs(temp_dir, exist_ok=True)
    temp_zip_path = os.path.join(temp_dir, filename)
    file.save(temp_zip_path)

    # Phase 7 quota gate (now possible because the request is authenticated).
    incoming = _path_size_bytes(temp_zip_path)
    ok, msg = check_quota(g.current_user, kind='dataset_create', incoming_bytes=incoming)
    if not ok:
        try:
            os.remove(temp_zip_path)
        except OSError:
            pass
        return jsonify({'error': msg}), 429

    try:
        success, message, ds_id = process_dataset_zip(
            temp_zip_path, dataset_name,
            owner_user_id=g.current_user.id,
        )

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
        'datasets': [{
            'id': ds.id,
            'name': ds.name
        } for ds in leaderboard.datasets],
        'dataset': {
            'id': leaderboard.datasets[0].id if leaderboard.datasets else None,
            'name': leaderboard.datasets[0].name if leaderboard.datasets else None
        }
    })

@app.route('/api/leaderboard/by_name/<leaderboard_name>/info', methods=['GET'])
def get_leaderboard_info_by_name_api(leaderboard_name):
    leaderboard = Leaderboard.query.filter_by(name=leaderboard_name).first()
    if not leaderboard:
        return jsonify({'error': f'Leaderboard "{leaderboard_name}" not found'}), 404
    return jsonify({
        'id': leaderboard.id,
        'name': leaderboard.name,
        'datasets': [{
            'id': ds.id,
            'name': ds.name
        } for ds in leaderboard.datasets],
        'dataset': {
            'id': leaderboard.datasets[0].id if leaderboard.datasets else None,
            'name': leaderboard.datasets[0].name if leaderboard.datasets else None
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


@app.route('/dataset/<int:dataset_id>/download')
def download_dataset(dataset_id):
    """Download dataset ZIP file"""
    dataset = Dataset.query.get_or_404(dataset_id)
    dataset_folder_name = secure_filename(dataset.name)
    dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', dataset_folder_name)
    zip_file = f"{dataset_folder_name}.zip"
    zip_path = os.path.join(dataset_dir, zip_file)
    
    if not os.path.exists(zip_path):
        return "Dataset file not found", 404
    
    return send_file(zip_path, as_attachment=True, download_name=zip_file)



@app.route('/api/dataset/<dataset_id>/download', methods=['GET'], endpoint='api_download_dataset')
def api_download_dataset(dataset_id):
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

@app.route('/api/leaderboard/<int:leaderboard_id>/submission/upload', methods=['POST'])
@require_api_token
def submission_upload_api(leaderboard_id):
    """Programmatic submission upload API. Bearer-token authenticated;
    owner_user_id of the resulting Submission is the token's user."""
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    if 'submission_zip' not in request.files:
        return jsonify({'error': 'No submission_zip provided'}), 400

    file = request.files['submission_zip']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Phase 7 daily-rate quota.
    ok, msg = check_quota(g.current_user, kind='submission')
    if not ok:
        return jsonify({'error': msg}), 429

    submission_name = request.form.get('submission_name', file.filename.replace('.zip', ''))

    temp_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload_zip', secure_filename(file.filename))
    os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
    file.save(temp_zip_path)

    try:
        success, error = process_submission_zip(
            leaderboard.id, submission_name, temp_zip_path,
            owner_user_id=g.current_user.id,
        )
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
    # Iterate and commit in batches to avoid long locks if many submissions
    for sub in submissions:
        sub.processing_status = 'Pending'
        db.session.add(sub)
        # Commit immediately? No, that might be too slow.
        # But locking for all might cause issues.
        # Let's verify if we can do it in one go but with a short transaction.
        # Actually, the error happens at commit() time.
        # Maybe just processing them one by one is safer for SQLite concurrent access.
        
        db.session.commit() # Commit each status change individually to keep transaction short
        
        tasks.process_submission.delay(sub.id, sample_filters=sample_filters)
        triggered_count += 1
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
                lm = res.leaderboard_metric
                lmid = f"lm_{lm.id}"
                val = res.value
                
                # Provide by ID
                metrics[lmid] = val
                # Provide by Names
                metrics[lm.global_metric.name] = val
                if lm.target_name:
                    metrics[lm.target_name] = val
            
            # Fetch custom metrics (scalars) - Aggregate with Filters
            # Parse filters from stored state
            filters = {}
            if sub.last_sample_filter:
                try:
                    filters = json.loads(sub.last_sample_filter)
                except Exception:
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
                    except Exception:
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
                    except Exception:
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
        lm = res.leaderboard_metric
        val = res.value
        if val is None: continue
        
        lmid = f"lm_{lm.id}"
        keys_to_update = [lmid, lm.global_metric.name]
        if lm.target_name: keys_to_update.append(lm.target_name)
        
        for k in keys_to_update:
            if k not in all_values: all_values[k] = []
            all_values[k].append(val)
        

    
    for m, vals in all_values.items():
        if vals:
             numeric_vals = [v for v in vals if isinstance(v, (int, float))]
             if numeric_vals:
                 metrics_ranges[m] = {'min': min(numeric_vals), 'max': max(numeric_vals)}

    # Ensure directions are also available by lm_ID and names
    final_directions = {}
    if metric_directions_dict:
        final_directions.update(metric_directions_dict)
    
    for lm in leaderboard.leaderboard_metrics:
        if lm.sort_direction:
            lmid = f"lm_{lm.id}"
            final_directions[lmid] = lm.sort_direction
            final_directions[lm.global_metric.name] = lm.sort_direction
            if lm.target_name:
                final_directions[lm.target_name] = lm.sort_direction

    return jsonify({
        'submissions': result,
        'ranges': metrics_ranges,
        'directions': final_directions
    })


@app.template_filter('from_json')
def from_json_filter(s):
    try:
        return json.loads(s)
    except Exception:
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

                # --- 6. Tag Filter migration ---
                # Check if leaderboard_metric table has tag_filter column using PRAGMA
                cursor.execute("PRAGMA table_info(leaderboard_metric)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'tag_filter' not in columns:
                    print("Migrating DB: Adding 'tag_filter' to 'leaderboard_metric' table...")
                    try:
                        cursor.execute("ALTER TABLE leaderboard_metric ADD COLUMN tag_filter TEXT DEFAULT NULL")
                        conn.commit()
                        print("Migration successful: Added 'tag_filter' column.")
                    except Exception as e:
                        print(f"Migration error (tag_filter): {e}")

                # --- 7. Git Author migration ---
                # Add git_author column to dataset table
                cursor.execute("PRAGMA table_info(dataset)")
                dataset_columns = [row[1] for row in cursor.fetchall()]
                
                if 'git_author' not in dataset_columns:
                    print("Migrating DB: Adding 'git_author' to 'dataset' table...")
                    try:
                        cursor.execute("ALTER TABLE dataset ADD COLUMN git_author VARCHAR(100) DEFAULT NULL")
                        conn.commit()
                        print("Migration successful: Added 'git_author' column to dataset.")
                    except Exception as e:
                        print(f"Migration error (dataset.git_author): {e}")
                
                # Add git_author column to submission table
                cursor.execute("PRAGMA table_info(submission)")
                submission_columns = [row[1] for row in cursor.fetchall()]
                
                if 'git_author' not in submission_columns:
                    print("Migrating DB: Adding 'git_author' to 'submission' table...")
                    try:
                        cursor.execute("ALTER TABLE submission ADD COLUMN git_author VARCHAR(100) DEFAULT NULL")
                        conn.commit()
                        print("Migration successful: Added 'git_author' column to submission.")
                    except Exception as e:
                        print(f"Migration error (submission.git_author): {e}")

                # (Legacy "ensure General project" + project_id backfill removed
                # in the projects-removal refactor. The Stage 2 teardown later
                # in this function drops the project table outright.)

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

                # --- 8. AuthorProfile merging migration ---
                try:
                    cursor.execute("PRAGMA table_info(author_profile)")
                    ap_columns = [row[1] for row in cursor.fetchall()]
                    if ap_columns and 'merged_into_username' not in ap_columns:
                        print("Migrating DB: Adding 'merged_into_username' to 'author_profile' table...")
                        cursor.execute("ALTER TABLE author_profile ADD COLUMN merged_into_username VARCHAR(100) DEFAULT NULL")
                        conn.commit()
                        print("Migration successful: Added 'merged_into_username' column.")
                except Exception as e:
                    print(f"Migration error (author_profile.merged_into_username): {e}")
                            
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

                # --- User table (Phase 1 multi-tenancy) ---
                # check_and_migrate_db runs BEFORE db.create_all(), so we
                # explicitly create the User table here so existing installs
                # (with a populated DB but no `user` table) get it without
                # reinstalling. db.create_all() handles fresh installs.
                try:
                    cursor.execute("SELECT id FROM user LIMIT 1")
                except sqlite3.OperationalError:
                    print("Migrating DB: Creating 'user' table...")
                    try:
                        cursor.execute('''
                            CREATE TABLE user (
                                id INTEGER PRIMARY KEY,
                                email VARCHAR(255) NOT NULL UNIQUE,
                                display_name VARCHAR(120),
                                avatar_url VARCHAR(500),
                                oauth_provider VARCHAR(20) NOT NULL,
                                oauth_sub VARCHAR(120) NOT NULL,
                                created_at DATETIME NOT NULL,
                                last_login_at DATETIME
                            )
                        ''')
                        cursor.execute('CREATE INDEX ix_user_email ON user (email)')
                        cursor.execute(
                            'CREATE UNIQUE INDEX uq_user_oauth_identity '
                            'ON user (oauth_provider, oauth_sub)'
                        )
                        conn.commit()
                        print("Created 'user' table.")
                    except Exception as e:
                        print(f"Failed to create 'user' table: {e}")

                # --- Owner / visibility columns (Phase 1 Slice 2) ---
                # Add owner_user_id + visibility to project / dataset /
                # leaderboard, and owner_user_id only to submission.
                # Each ALTER is wrapped so a half-applied state is recoverable
                # by re-running the migration.
                _ownership_migrations = [
                    # project.* entries kept for back-compat on installs that
                    # still have the table around — they're harmless once the
                    # table is dropped (the loop checks PRAGMA table_info first).
                    ("project",              "visibility",    "VARCHAR(20) NOT NULL DEFAULT 'public'"),
                    ("dataset",              "owner_user_id", "INTEGER"),
                    ("dataset",              "visibility",    "VARCHAR(20) NOT NULL DEFAULT 'public'"),
                    ("leaderboard",          "owner_user_id", "INTEGER"),
                    ("leaderboard",          "visibility",    "VARCHAR(20) NOT NULL DEFAULT 'public'"),
                    ("submission",           "owner_user_id", "INTEGER"),
                    # Slice 4: shared "library" assets.
                    ("global_metric",        "owner_user_id", "INTEGER"),
                    ("global_metric",        "visibility",    "VARCHAR(20) NOT NULL DEFAULT 'public'"),
                    ("global_visualization", "owner_user_id", "INTEGER"),
                    ("global_visualization", "visibility",    "VARCHAR(20) NOT NULL DEFAULT 'public'"),
                    # Phase 5: curated-content marker (legacy on Project, then relocated).
                    # Phase 7: per-user quotas. Existing rows pick up the
                    # free-tier defaults; bump per-row to grant a paid tier.
                    ("user",                 "quota_max_storage_bytes",        f"BIGINT NOT NULL DEFAULT {200 * 1024 * 1024}"),
                    ("user",                 "quota_max_datasets",             "INTEGER NOT NULL DEFAULT 5"),
                    ("user",                 "quota_max_submissions_per_day",  "INTEGER NOT NULL DEFAULT 50"),
                    # Phase 8: programmatic-access token. Nullable; users
                    # opt-in by clicking "Generate" in /settings/api_tokens.
                    ("user",                 "api_token",                      "VARCHAR(64)"),
                    # DB-backed admin flag (env-var allow-list still works
                    # in parallel as the bootstrap mechanism).
                    ("user",                 "is_admin",                       "BOOLEAN NOT NULL DEFAULT 0"),
                    # HF auto-import provenance.
                    ("dataset",              "source_kind",                    "VARCHAR(32)"),
                    ("dataset",              "source_metadata",                "TEXT"),
                    # Phase 7: cached storage usage on the dataset itself —
                    # cheaper than du'ing the volume on every upload.
                    ("dataset",              "storage_bytes",                  "BIGINT NOT NULL DEFAULT 0"),
                ]
                for tbl, col, coldef in _ownership_migrations:
                    cursor.execute(f"PRAGMA table_info({tbl})")
                    existing = {row[1] for row in cursor.fetchall()}
                    # PRAGMA returns no rows on a missing table (no exception).
                    # On a fresh install the table doesn't exist yet —
                    # db.create_all() below will build it with the right
                    # columns, so we skip rather than try to ALTER nothing.
                    if not existing:
                        continue
                    if col in existing:
                        continue
                    try:
                        cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {coldef}")
                        conn.commit()
                        print(f"Added {tbl}.{col}.")
                    except Exception as e:
                        print(f"Failed to add {tbl}.{col}: {e}")

                # Stage 2 — projects + curated removal teardown.
                # ALTER TABLE DROP COLUMN with a foreign-key column trips
                # SQLite's FK validation ("unknown column ... in foreign key
                # definition") because the FK metadata still references the
                # dropped target table. legacy_alter_table=ON skips that
                # validation, which is what we want here — the FK target is
                # gone and we're tearing the column out anyway.
                try:
                    cursor.execute("PRAGMA legacy_alter_table = ON")
                    cursor.execute("PRAGMA foreign_keys = OFF")
                    for tbl, col in [
                        ('leaderboard', 'project_id'),
                        ('leaderboard', 'is_curated'),
                        ('dataset', 'is_curated'),
                        ('global_visualization', 'project_id'),
                    ]:
                        try:
                            cursor.execute(f"PRAGMA table_info({tbl})")
                            cols = {row[1] for row in cursor.fetchall()}
                            if col in cols:
                                cursor.execute(f"ALTER TABLE {tbl} DROP COLUMN {col}")
                                conn.commit()
                                print(f"Dropped {tbl}.{col}.")
                        except Exception as e:
                            print(f"Could not drop {tbl}.{col} (will retry next boot): {e}")
                    try:
                        cursor.execute("DROP TABLE IF EXISTS project")
                        conn.commit()
                        print("Dropped project table.")
                    except Exception as e:
                        print(f"Could not drop project table: {e}")
                finally:
                    cursor.execute("PRAGMA legacy_alter_table = OFF")
                    cursor.execute("PRAGMA foreign_keys = ON")

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
    writer.writerow(['Metric', 'Value', 'Mapping', 'Tags'])
    
    # Fetch results joined with LeaderboardMetric to get display names
    results = db.session.query(MetricResult, LeaderboardMetric).join(
        LeaderboardMetric, MetricResult.leaderboard_metric_id == LeaderboardMetric.id
    ).filter(MetricResult.submission_id == submission_id).all()
    
    for result, lb_metric in results:
        # Determine display name
        display_name = get_distinguishable_metric_name(lb_metric)
        
        # Format value
        value = result.value
        if value is None:
            value_str = 'N/A'
            if result.error_message:
                value_str = f"Error: {result.error_message}"
        else:
            value_str = f"{value:.6f}"
            
        writer.writerow([display_name, value_str, lb_metric.arg_mappings, lb_metric.tag_filter or ''])
        
    output = make_response(si.getvalue())
    filename = f"{submission.name}_metrics.csv"
    output.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/submission/<int:submission_id>/download_sample_metrics_csv')
def download_sample_metrics_csv(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    leaderboard = submission.leaderboard
    
    # [FIX] Support multiple datasets
    dataset_ids = [ds.id for ds in leaderboard.datasets]
    if not dataset_ids and leaderboard.dataset_id:
        dataset_ids = [leaderboard.dataset_id]
        
    # Get all samples for all associated datasets
    samples = Sample.query.filter(Sample.dataset_id.in_(dataset_ids)).order_by(Sample.name).all()
    sample_names = [s.name for s in samples]
    sample_ids = [s.id for s in samples]
    
    # Get metric display names
    lb_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=leaderboard.id).all()
    metric_names = []
    
    # Map both friendly names and lm_{id} to LeaderboardMetric objects
    metric_id_to_lm = {}
    for lm in lb_metrics:
        if lm.global_metric.is_aggregated:
            continue
        
        display_name = get_distinguishable_metric_name(lm)
        metric_names.append(display_name)
        
        # Index by distinguish name, friendly name, and lm_{id}
        metric_id_to_lm[display_name] = lm
        metric_id_to_lm[f"lm_{lm.id}"] = lm
        friendly_name = lm.target_name if lm.target_name else lm.global_metric.name
        metric_id_to_lm[friendly_name] = lm

    # Fetch per-sample metrics from CustomField
    # Rows: Metrics, Columns: Samples
    si = io.StringIO()
    writer = csv.writer(si)
    
    # Header: Metric, Mapping, Tags, Sample1, Sample2, ...
    writer.writerow(['Metric', 'Mapping', 'Tags'] + sample_names)
    
    for m_display in metric_names:
        lm = metric_id_to_lm.get(m_display)
        if not lm: continue
        
        m_id_name = f"lm_{lm.id}"
        friendly_name = lm.target_name if lm.target_name else lm.global_metric.name
        row = [m_display, lm.arg_mappings, lm.tag_filter or '']
        
        # Fetch results for both name variants and both types ('scalar' and legacy 'metric')
        cfs = CustomField.query.filter(
            CustomField.submission_id == submission_id,
            CustomField.name.in_([m_id_name, friendly_name, f"metric_{friendly_name}"]),
            CustomField.field_type.in_(['scalar', 'metric'])
        ).all()
        
        # Check if ANY lm_{id} fields exist for this metric/submission
        # If so, we are in "New Mode" and should strictly ignore friendly-name fallbacks (which might be raw/unfiltered data)
        has_new_format = any(cf.name == m_id_name for cf in cfs)
        
        cf_map = {}
        for cf in cfs:
            if has_new_format and cf.name != m_id_name:
                continue
            
            if cf.name == m_id_name or cf.sample_id not in cf_map:
                cf_map[cf.sample_id] = cf.value_float
        
        for s_id in sample_ids:
            val = cf_map.get(s_id)
            row.append(f"{val:.6f}" if val is not None else 'N/A')
        writer.writerow(row)

    # [NEW] Add Tag Parsing Rows
    # Dataset tags are stored in the 'tags' column of the Sample model
    parsed_tags_by_sample = {} # sample_id -> {tag_key: tag_value}
    all_tag_keys = set()
    
    for sample in samples:
        if not sample.tags:
            continue
        parts = sample.tags.split(',')
        sample_tags = {}
        for part in parts:
            if '=' in part:
                try:
                    k, v = part.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if k:
                        sample_tags[k] = v
                        all_tag_keys.add(k)
                except ValueError:
                    continue
        parsed_tags_by_sample[sample.id] = sample_tags

    sorted_tag_keys = sorted(list(all_tag_keys))
    
    for tag_key in sorted_tag_keys:
        row = [f"tag_{tag_key}", "", ""]
        for s_id in sample_ids:
            tags = parsed_tags_by_sample.get(s_id, {})
            val = tags.get(tag_key)
            row.append(format_tag_value(val))
        writer.writerow(row)
        
    output = make_response(si.getvalue())
    filename = f"{submission.name}_per_sample_metrics.csv"
    output.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/leaderboard/<int:leaderboard_id>/download_sample_metrics_bulk', methods=['POST'])
def download_submissions_sample_metrics_bulk(leaderboard_id):
    submission_ids = request.form.getlist('submission_ids')
    redirect_args = {
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
                # [FIX] Support multiple datasets
                dataset_ids = [ds.id for ds in sub.leaderboard.datasets]
                if not dataset_ids and sub.leaderboard.dataset_id:
                    dataset_ids = [sub.leaderboard.dataset_id]
                    
                # Get all samples for all associated datasets
                samples = Sample.query.filter(Sample.dataset_id.in_(dataset_ids)).order_by(Sample.name).all()
                sample_names = [s.name for s in samples]
                sample_ids = [s.id for s in samples]
                
                lb_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=sub.leaderboard_id).all()
                metric_names = []
                metric_id_to_lm = {}
                for lm in lb_metrics:
                    if lm.global_metric.is_aggregated:
                        continue
                    display_name = get_distinguishable_metric_name(lm)
                    metric_names.append(display_name)
                    
                    metric_id_to_lm[display_name] = lm
                    metric_id_to_lm[f"lm_{lm.id}"] = lm
                    friendly_name = lm.target_name if lm.target_name else lm.global_metric.name
                    metric_id_to_lm[friendly_name] = lm

                si = io.StringIO()
                writer = csv.writer(si)
                writer.writerow(['Metric', 'Mapping', 'Tags'] + sample_names)
                
                for m_display in metric_names:
                    lm = metric_id_to_lm.get(m_display)
                    if not lm: continue
                    m_id_name = f"lm_{lm.id}"
                    friendly_name = lm.target_name if lm.target_name else lm.global_metric.name
                    row = [m_display, lm.arg_mappings, lm.tag_filter or '']
                    
                    cfs = CustomField.query.filter(
                        CustomField.submission_id == sub.id,
                        CustomField.name.in_([m_id_name, friendly_name, f"metric_{friendly_name}"]),
                        CustomField.field_type.in_(['scalar', 'metric'])
                    ).all()
                    
                    # Check if ANY lm_{id} fields exist for this metric/submission
                    found_any_flavour_in_submission = False
                    
                    # Optimization: Check if this submission has *any* lm_ fields generally (New Mode)
                    # We can cache this per submission if needed, but doing a simple query is okay for now.
                    # Or check against all lb_metrics IDs.
                    
                    # Let's verify if any of the Friendly Name's flavours are present in this submission
                    other_flavour_ids = []
                    for other_lm in lb_metrics:
                        other_name = other_lm.target_name if other_lm.target_name else other_lm.global_metric.name
                        if other_name == friendly_name:
                            other_flavour_ids.append(f"lm_{other_lm.id}")
                            
                    # Check if any of these exist in the database for this submission
                    # (This confirms "New Mode" for this specific metric name)
                    has_new_mode_specific = db.session.query(CustomField.id).filter(
                        CustomField.submission_id == sub.id,
                        CustomField.name.in_(other_flavour_ids)
                    ).first() is not None

                    use_strict_id = has_new_mode_specific
                    
                    cf_map = {}
                    for cf in cfs:
                        if use_strict_id:
                            if cf.name == m_id_name:
                                cf_map[cf.sample_id] = cf.value_float
                        else:
                            # Legacy mode: accept friendly name or id
                            if cf.name == m_id_name or cf.sample_id not in cf_map:
                                cf_map[cf.sample_id] = cf.value_float
                    
                    for cf in cfs:
                        if use_strict_id:
                            if cf.name == m_id_name:
                                cf_map[cf.sample_id] = cf.value_float
                        else:
                            # Legacy mode: accept friendly name or id
                            if cf.name == m_id_name or cf.sample_id not in cf_map:
                                cf_map[cf.sample_id] = cf.value_float
                    
                    for s_id in sample_ids:
                        val = cf_map.get(s_id)
                        row.append(f"{val:.6f}" if val is not None else 'N/A')
                    writer.writerow(row)

                # [NEW] Add Tag Parsing Rows
                parsed_tags_by_sample = {}
                all_tag_keys = set()
                for sample in samples:
                    if not sample.tags: continue
                    parts = sample.tags.split(',')
                    sample_tags = {}
                    for part in parts:
                        if '=' in part:
                            try:
                                k, v = part.split('=', 1)
                                k, v = k.strip(), v.strip()
                                if k:
                                    sample_tags[k] = v
                                    all_tag_keys.add(k)
                            except ValueError: continue
                    parsed_tags_by_sample[sample.id] = sample_tags
                
                sorted_tag_keys = sorted(list(all_tag_keys))

                for tag_key in sorted_tag_keys:
                    row = [f"tag_{tag_key}", "", ""]
                    for s_id in sample_ids:
                        tags = parsed_tags_by_sample.get(s_id, {})
                        val = tags.get(tag_key)
                        row.append(format_tag_value(val))
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
                writer.writerow(['Metric', 'Value', 'Mapping', 'Tags'])
                
                results = db.session.query(MetricResult, LeaderboardMetric).join(
                    LeaderboardMetric, MetricResult.leaderboard_metric_id == LeaderboardMetric.id
                ).filter(MetricResult.submission_id == sub.id).all()
                
                for result, lb_metric in results:
                    display_name = get_distinguishable_metric_name(lb_metric)
                    value = result.value
                    if value is None:
                        value_str = 'N/A'
                        if result.error_message:
                            value_str = f"Error: {result.error_message}"
                    else:
                        value_str = str(value)
                    writer.writerow([display_name, value_str, lb_metric.arg_mappings, lb_metric.tag_filter or ''])
                
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
                writer.writerow(['Metric', 'Value', 'Mapping', 'Tags'])
                
                results = db.session.query(MetricResult, LeaderboardMetric).join(
                    LeaderboardMetric, MetricResult.leaderboard_metric_id == LeaderboardMetric.id
                ).filter(MetricResult.submission_id == sub.id).all()
                
                for result, lb_metric in results:
                    display_name = get_distinguishable_metric_name(lb_metric)
                    value = result.value
                    if value is None:
                        value_str = 'N/A'
                        if result.error_message:
                            value_str = f"Error: {result.error_message}"
                    else:
                        value_str = f"{value:.6f}"
                    writer.writerow([display_name, value_str, lb_metric.arg_mappings, lb_metric.tag_filter or ''])
                
                zf.writestr(f"{sub.name}_{sub.id}_metrics.csv", si.getvalue())

                # 3. Add Per-Sample Metrics CSV
                # [FIX] Support multiple datasets
                dataset_ids = [ds.id for ds in sub.leaderboard.datasets]
                if not dataset_ids and sub.leaderboard.dataset_id:
                    dataset_ids = [sub.leaderboard.dataset_id]
                
                # Get all samples for all associated datasets
                samples = Sample.query.filter(Sample.dataset_id.in_(dataset_ids)).order_by(Sample.name).all()
                sample_names = [s.name for s in samples]
                sample_ids = [s.id for s in samples]
                
                lb_metrics = LeaderboardMetric.query.filter_by(leaderboard_id=sub.leaderboard_id).all()
                metric_names = []
                metric_display_map = {}
                for lm in lb_metrics:
                    if lm.global_metric.is_aggregated:
                        continue
                    display_name = get_distinguishable_metric_name(lm)
                    metric_names.append(display_name)
                    metric_display_map[display_name] = (lm.target_name if lm.target_name else lm.global_metric.name)

                si_ps = io.StringIO()
                writer_ps = csv.writer(si_ps)
                writer_ps.writerow(['Metric', 'Mapping', 'Tags'] + sample_names)
                
                # Store lm mapping for easier access to Mapping/Tags
                metric_id_to_lm = { get_distinguishable_metric_name(lm): lm for lm in lb_metrics }

                for m_display in metric_names:
                    lm = metric_id_to_lm.get(m_display)
                    if not lm: continue
                    m_id_name = f"lm_{lm.id}"
                    row = [m_display, lm.arg_mappings, lm.tag_filter or '']
                    cfs = CustomField.query.filter_by(submission_id=sub.id, name=m_id_name, field_type='scalar').all()
                    cf_map = {cf.sample_id: cf.value_float for cf in cfs}
                    for s_id in sample_ids:
                        val = cf_map.get(s_id)
                        row.append(f"{val:.6f}" if val is not None else 'N/A')
                    writer_ps.writerow(row)

                # [NEW] Add Tag Parsing Rows
                parsed_tags_by_sample = {}
                all_tag_keys = set()
                for sample in samples:
                    if not sample.tags: continue
                    parts = sample.tags.split(',')
                    sample_tags = {}
                    for part in parts:
                        if '=' in part:
                            try:
                                k, v = part.split('=', 1)
                                k, v = k.strip(), v.strip()
                                if k:
                                    sample_tags[k] = v
                                    all_tag_keys.add(k)
                            except ValueError: continue
                    parsed_tags_by_sample[sample.id] = sample_tags
                
                sorted_tag_keys = sorted(list(all_tag_keys))

                for tag_key in sorted_tag_keys:
                    row = [f"tag_{tag_key}", "", ""]
                    for s_id in sample_ids:
                        tags = parsed_tags_by_sample.get(s_id, {})
                        val = tags.get(tag_key)
                        row.append(format_tag_value(val))
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


def prune_incomplete_datasets():
    """Remove datasets that landed in the DB but never finished uploading.

    A "fully uploaded" dataset has at least one Sample row AND its
    on-disk folder under uploads/datasets/<safe-name>/ exists. If either
    is missing — typical signature of a process_dataset_zip() crash, an
    interrupted ZIP extraction, or a deleted volume snapshot — the row
    is just orphaned bookkeeping. Cascade-delete it (and any leftover
    folder bytes) so the datasets list is honest about what's actually
    available.

    Runs at boot from run_migrations(); also exposed for tests.

    Returns the number of datasets removed.
    """
    upload_folder = app.config['UPLOAD_FOLDER']
    datasets_root = os.path.join(upload_folder, 'datasets')
    removed = 0

    for ds in Dataset.query.all():
        sample_count = Sample.query.filter_by(dataset_id=ds.id).count()
        folder_path = os.path.join(datasets_root, secure_filename(ds.name))
        folder_ok = os.path.isdir(folder_path)

        if sample_count == 0 or not folder_ok:
            print(
                f"prune_incomplete_datasets: removing dataset {ds.id} "
                f"'{ds.name}' (samples={sample_count}, folder_present={folder_ok})"
            )
            # Remove on-disk folder if it exists at all (may be partially
            # extracted bytes even when the row says zero samples).
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
            db.session.delete(ds)
            removed += 1

    if removed:
        db.session.commit()
    return removed


def run_migrations():
    """Idempotent boot-time migration runner. Safe to call concurrently —
    SQLite WAL + 120s busy_timeout absorb the race, and both `db.create_all()`
    (CREATE TABLE IF NOT EXISTS) and `check_and_migrate_db()` (try/except
    around each ALTER) are no-ops when the schema is already current."""
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        check_and_migrate_db()
        db.create_all()
        try:
            prune_incomplete_datasets()
        except Exception as e:
            # Boot must continue even if the cleanup hits a snag.
            print(f"prune_incomplete_datasets failed (non-fatal): {e}")


# Auto-run migrations at module import time when the env var is set. This is
# how the schema gets onto Fly's persistent volume: Fly's `release_command`
# runs in a temp VM that doesn't mount our volume, so we can't rely on it for
# SQLite-on-volume setups. Setting BENCHHUB_AUTO_MIGRATE=1 in fly.toml means
# every gunicorn / celery process boots through run_migrations() against the
# real volume-backed DB.
#
# Tests deliberately don't set this env var — the pytest conftest manages
# the schema itself via db.create_all() per test.
if os.environ.get('BENCHHUB_AUTO_MIGRATE') == '1':
    run_migrations()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'migrate':
        run_migrations()
        sys.exit(0)

    run_migrations()
    port = int(os.environ.get('PORT', '6060'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
