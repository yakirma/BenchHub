import contextlib
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
import bench_cache
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

# Pointer-mode + remote-submission cache. Bytes streamed from HF / from
# user-owned remote storage land here under a bounded LRU. Default
# location is alongside uploads so the same Fly volume covers both.
app.config.setdefault(
    'CACHE_FOLDER',
    os.environ.get('BENCHHUB_CACHE_FOLDER')
    or os.path.join(app.config['UPLOAD_FOLDER'], '_cache'),
)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)


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
    # Keep the broker (Redis) but disable the result backend — every
    # task in this app uses ignore_result=True, so the result backend
    # was just burning Upstash request quota for state nobody reads.
    # Polling tunables below cut the idle-worker poll rate that hit
    # the 500K/month free-tier cap on Upstash.
    celery = Celery(
        app.import_name,
        backend=None,
        broker=app.config['CELERY_BROKER_URL'],
    )
    celery.conf.update(app.config)
    celery.conf.update(
        result_backend=None,
        # Default visibility_timeout is 1h with frequent polls; 6h
        # is safer for our long-running viz/index tasks anyway.
        broker_transport_options={'visibility_timeout': 21600},
        # Skip mingle/gossip/heartbeat — single-worker single-machine
        # setup gains nothing from them, and each is a Redis poll.
        worker_disable_rate_limits=True,
        # Don't DB-poll for tasks every 50ms (default).
        broker_pool_limit=1,
    )

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
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id'), primary_key=True),
    # Per-attachment role. Lets a single LB pair an "input" dataset
    # with one or more "GT-source" datasets — solves the dirty-docs
    # /-train + /-cleaned shape and SIDD-style splits without
    # merging upstream HF repos.
    #
    # Conventions:
    #   'primary'   → owns the sample-name list submissions key off.
    #                 First non-gt_source LB attachment is primary.
    #   'gt_source' → matches sample names against the primary; its
    #                 CustomFields fold into get_metric_context as
    #                 additional gt_<col> entries.
    db.Column('role', db.String(20), nullable=False,
              default='primary', server_default='primary'),
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
    # New model: track which columns the user explicitly *hid*. Empty
    # string (or NULL) → nothing hidden → all available columns visible.
    # This way new custom fields added after the user's last save
    # default to visible instead of disappearing until re-selected.
    hidden_display_columns = db.Column(db.Text, nullable=True)
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
    # 'local'      → samples have on-disk files (images / depth NPZs).
    # 'hf-pointer' → samples carry source_ref_json + per-CustomField
    #                source_column; the engine streams from HF on
    #                demand and caches via bench_cache.
    storage_mode = db.Column(db.String(20), nullable=False,
                             default='local', server_default='local')

    @property
    def source_metadata_parsed(self):
        """Parsed source_metadata or {} on any error. Used by templates
        that want to read e.g. .repo_id without writing JSON-decoding
        boilerplate inline."""
        if not self.source_metadata:
            return {}
        try:
            return json.loads(self.source_metadata)
        except Exception:
            return {}

class Sample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    tags = db.Column(db.String(500)) # Stores comma-separated tags
    # Pointer-mode storage. Populated only for samples that come from
    # a streaming HF dataset; NULL for everything imported as bytes-on-
    # disk. Stores a JSON blob {repo_id, revision, split, row_idx} so
    # the engine can fetch this row's image / depth on-demand via
    # bench_cache. See Dataset.storage_mode.
    source_ref_json = db.Column(db.Text, nullable=True)

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
    # HF-attached LBs have no Sample rows. The eval task snapshots
    # streamed GT scalars/text into CustomField rows tagged with
    # leaderboard_id (sample_id NULL, submission_id NULL) so the
    # comparison view can render the GT column without re-streaming
    # from huggingface.co. Indexed for the LB-scoped lookup.
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=True, index=True)
    sample_name = db.Column(db.String(100), nullable=True)  # Sample name for submission custom fields
    # Pointer-mode: name of the column in the upstream HF row that
    # produces this CustomField when fetched. Populated only when the
    # parent Sample's source_ref_json is set (i.e. the dataset is
    # storage_mode='hf-pointer'). value_text stays NULL for image /
    # depth pointer fields; the engine resolves them via bench_cache
    # using (sample.source_ref_json, source_column) as the cache key.
    source_column = db.Column(db.String(120), nullable=True)

    def get_value(self):
        """Helper to get the appropriate value based on type"""
        if self.field_type in ['scalar', 'metric'] and self.value_float is not None:
            return self.value_float
        return self.value_text


def _smart_num(value):
    """Render a scalar without a trailing '.0' when it's actually an
    integer (HF ClassLabel indices, count metrics, etc.). Floats with
    a fractional part keep their decimals. Non-numerics pass through.
    Registered as a Jinja filter further down."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if f != f:  # NaN
        return value
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return value

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
    # New: same flip as Dataset.hidden_display_columns. Track exclusions.
    hidden_comparison_display_columns = db.Column(db.Text, nullable=True)
    visualizations = db.Column(db.String(500), nullable=False, default='') # Active visualizers
    selected_metrics = db.Column(db.String(500), default='') # Comma separated list of targets
    summary_metrics = db.Column(db.String(500), default='') # Initial metrics to show
    metric_directions = db.Column(db.Text, default='{}') # JSON: metric_name -> "higher_is_better" or "lower_is_better"
    metric_aggregation = db.Column(db.Text, default='{}') # JSON: metric_name -> {"type": "mean|median|percentile", "percentile": 95}
    scalar_width = db.Column(db.String(50), nullable=True) # Override for scalar column width
    image_width = db.Column(db.String(50), nullable=True) # Override for image column width
    last_sample_filter = db.Column(db.Text, nullable=True) # JSON string: store last used filter settings
    # Cached Colab submission notebook. Stored as JSON
    # {"sig": "<crc32-of-lb-structure>", "notebook": "<.ipynb json>"}.
    # Self-invalidating: when the LB's datasets/metrics change, the
    # signature drifts and the next request regenerates.
    colab_notebook_cache = db.Column(db.Text, nullable=True)
    # Personal vs admin-promoted canonical. /explore + /home filter
    # to canonicality='public' by default; personal LBs stay visible
    # to their owner + collaborators only.
    canonicality = db.Column(
        db.String(20), nullable=False,
        default='personal', server_default='personal',
    )
    # When set, this LB is the canonical surface for the named HF repo
    # (e.g. 'cifar10', 'sayakpaul/nyu_depth_v2'). At most one row per
    # repo with this set — enforced at promote time.
    canonical_for_repo = db.Column(db.String(200), nullable=True, index=True)
    # Pred fields the LB requires submissions to ship even when no
    # metric/viz consumes them — e.g. an organizer wants to host
    # raw predictions for human review. JSON list of dicts:
    # [{name, gt_field, kind, description}]. Surfaced in the
    # "Submission contract" widget alongside metric-derived fields.
    required_pred_fields_json = db.Column(db.Text, nullable=True)
    # Two-level taxonomy "area/task" (e.g. "Vision/Depth Estimation").
    # Auto-populated for PWC imports from the source task name via
    # `_pwc_task_to_category`; manual LBs can override via the
    # Settings page. /explore renders a tree sidebar by splitting on '/'.
    category = db.Column(db.String(120), nullable=True, index=True)
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
    # Colab provenance: when a submission was uploaded via the API
    # from a Colab notebook, store the gist URL so reviewers can re-open
    # the exact notebook that produced these predictions. Set by the
    # API endpoint when the form includes `source_colab_url`, or
    # auto-populated from the user's per-user gist for this LB.
    source_colab_url = db.Column(db.String(500), nullable=True)
    # Storage:
    #   'local'  → ZIP was uploaded directly to BenchHub's volume.
    #   'remote' → ZIP lives at `remote_url` (https:// or hf://).
    #              Fetched on-demand into bench_cache; extracted to
    #              `uploads/submissions/<id>/` for eval, may evict.
    storage_mode = db.Column(db.String(20), nullable=False,
                             default='local', server_default='local')
    remote_url = db.Column(db.String(500), nullable=True)
    # SHA-256 of the submission ZIP captured at first eval. On re-eval
    # the bytes get hashed again and rejected on mismatch — strict
    # reproducibility (same posture as `revision` pinning for HF GT).
    content_hash = db.Column(db.String(64), nullable=True)
    # 'verified' (default) — went through the full upload + eval pipeline.
    # 'mirrored'           — score row imported from an external source
    #                        (Papers With Code etc). No predictions on
    #                        disk, no Celery task, MetricResult rows are
    #                        inserted directly. Two-table render on the
    #                        LB page keeps the trust gradient visible.
    kind = db.Column(db.String(20), nullable=False,
                     default='verified', server_default='verified',
                     index=True)
    source_attribution = db.Column(db.String(200), nullable=True)  # e.g. 'Papers With Code'
    source_paper_url = db.Column(db.String(500), nullable=True)
    source_paper_title = db.Column(db.String(300), nullable=True)
    source_external_url = db.Column(db.String(500), nullable=True)  # e.g. PWC entry URL
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

    # HuggingFace access token, used by HF imports (auto + direct + gated
    # wizard). Auto-saved when the user supplies one; user can review or
    # remove from /settings/hf_token. Treat the DB as secret-bearing —
    # same posture as api_token.
    hf_token = db.Column(db.String(200), nullable=True)

    __table_args__ = (
        db.UniqueConstraint('oauth_provider', 'oauth_sub', name='uq_user_oauth_identity'),
    )


class UserColabGist(db.Model):
    """Per-(user, leaderboard) GitHub gist holding a Colab notebook with
    the user's BenchHub API token baked in. The LB-level cache on
    Leaderboard.colab_notebook_cache keeps the *generic* notebook (token
    placeholder); this table maps user → personalized gist so each
    authed user gets their own one-click Colab link."""
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    leaderboard_id = db.Column(
        db.Integer, db.ForeignKey('leaderboard.id', ondelete='CASCADE'),
        primary_key=True,
    )
    gist_id = db.Column(db.String(64), nullable=True)
    gist_owner = db.Column(db.String(120), nullable=True)
    sig = db.Column(db.String(16), nullable=True)


# Default cap on how many rows we iterate from an HF-ref attachment
# during metric eval. Override per-attachment via Attachment.hf_sample_cap.
HF_DEFAULT_SAMPLE_CAP = 10_000


class Attachment(db.Model):
    """Replaces the legacy `leaderboard_datasets` association table.
    Each row attaches one source to one Leaderboard. The source is
    EITHER a BH-managed Dataset (uploaded ZIP, samples on disk) OR a
    HuggingFace reference (live-streamed at eval time, no DB rows).

    Contract:
      - `dataset_id` set, hf_* null  → BH attachment.
      - `dataset_id` null, `hf_repo_id` set → HF-ref attachment.
      - `role` ∈ {'primary', 'gt_source'}: gt_source attachments fold
        their fields into the metric context for primary samples whose
        names match (paired-dataset shape).
    """
    id = db.Column(db.Integer, primary_key=True)
    leaderboard_id = db.Column(
        db.Integer, db.ForeignKey('leaderboard.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    dataset_id = db.Column(
        db.Integer, db.ForeignKey('dataset.id', ondelete='CASCADE'),
        nullable=True,
    )
    hf_repo_id = db.Column(db.String(200), nullable=True, index=True)
    hf_revision = db.Column(db.String(120), nullable=True)
    hf_split = db.Column(db.String(50), nullable=True)
    # Mapping from HF column name → BH semantic field, persisted as JSON
    # list of {column, target_kind, target_field}. Same shape the
    # importer's preview produces, just attached to the LB rather than
    # consumed at import time.
    hf_mapping_json = db.Column(db.Text, nullable=True)
    role = db.Column(db.String(20), nullable=False,
                     default='primary', server_default='primary')
    # Per-attachment cap on iterated rows. NULL → use HF_DEFAULT_SAMPLE_CAP.
    hf_sample_cap = db.Column(db.Integer, nullable=True)

    leaderboard = db.relationship(
        'Leaderboard',
        backref=db.backref('attachments', lazy='subquery',
                           cascade='all, delete-orphan'),
    )
    dataset = db.relationship('Dataset')

    @property
    def kind(self):
        return 'bh' if self.dataset_id is not None else 'hf'


class HfDatasetVisit(db.Model):
    """Lightweight log of HF datasets a user has explored via the
    /hf/<repo_id> live-preview page. NOT a materialized Dataset —
    the user hasn't committed to creating an LB yet. Used for the
    'Your recent HF picks' widget + the 'Trending across the
    platform' aggregate.

    Composite PK keeps each (user, repo) at one row; revisits bump
    `last_visited_at` + `visit_count` rather than appending.
    """
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),
                        primary_key=True)
    repo_id = db.Column(db.String(200), primary_key=True)
    last_visited_at = db.Column(db.DateTime, nullable=False,
                                default=datetime.utcnow,
                                index=True)
    visit_count = db.Column(db.Integer, nullable=False, default=1,
                            server_default='1')


class CacheEntry(db.Model):
    """Disk-bounded LRU cache backing pointer-mode HF datasets and
    remote submissions. Each row is one cached blob; bench_cache.py
    is the only thing that should write to this table.

    `cache_key` is a free-form string — typically
    `gt:<repo_id>@<revision>:<split>:<row_idx>` for HF GT rows, or
    `sub:<submission_id>:<col>:<sample>` for streamed submission
    bytes. The on-disk filename is sha256(cache_key) so the keyspace
    can carry slashes / colons / query-strings safely.

    `origin` ∈ {'gt', 'submission'}; submissions evict first when
    budget tightens (they're cheap to re-fetch from the user's
    external store, GT is on HF and rate-limited).
    """
    cache_key = db.Column(db.String(512), primary_key=True)
    size_bytes = db.Column(db.BigInteger, nullable=False, default=0)
    origin = db.Column(db.String(16), nullable=False)
    last_accessed_at = db.Column(db.DateTime, nullable=False,
                                 default=datetime.utcnow, index=True)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=datetime.utcnow)


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

        # Determine field type purely from file extension + content. The
        # legacy reserved prefixes (metric_, hist_, raw_) were dropped:
        # any folder can hold any type, and we infer from what's inside.
        # The only remaining hard-coded folder is `tags` (always text,
        # never re-parsed as numeric — otherwise ClassLabel name lists
        # like ['0', '1'] get coerced to scalars and the sample loses
        # its tags).
        is_tags_folder = (folder_name == 'tags')
        field_type = 'text' if is_tags_folder else None

        # Check what's inside the folder
        field_data = {}

        folder_files = os.listdir(folder_path)

        def _classify_npz(path):
            """Peek at the npz's keys to decide histogram vs depth.
            Bin+counts → histogram; explicit `depth` key → depth; else
            we still tag as depth (a single-array .npz is overwhelmingly
            a dense map). Returns the field-type string."""
            try:
                with np.load(path) as data:
                    keys = set(data.keys())
            except Exception:
                return None
            if {'bins', 'counts'}.issubset(keys):
                return 'histogram'
            return 'depth'

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
                        if field_type == 'text':
                            field_data[sample_name] = content
                        else:
                            try:
                                val = float(content)
                                if field_type is None:
                                    field_type = 'scalar'
                                field_data[sample_name] = val
                            except ValueError:
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

            # Check for npz files. New convention: `<sample>.npz` with
            # the kind decided by inspecting the archive's keys. Keep
            # the legacy `<sample>_<W>x<H>.npz` shape for backward
            # compat — older depth submissions are still on disk.
            npz_candidate = None
            simple_npz = f'{sample_name}.npz'
            if simple_npz in folder_files:
                npz_candidate = os.path.join(folder_path, simple_npz)
            else:
                prefix = f"{sample_name}_"
                for fname in folder_files:
                    if (fname.startswith(prefix) and fname.endswith('.npz')
                            and re.match(r'\d+x\d+\.npz$', fname[len(prefix):])):
                        npz_candidate = os.path.join(folder_path, fname)
                        break
            if npz_candidate:
                kind = _classify_npz(npz_candidate)
                if kind:
                    if field_type is None:
                        field_type = kind
                    field_data[sample_name] = npz_candidate

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

    # Find any per-sample .npz that holds histogram data. The legacy
    # hist_/raw_histogram prefix gate was dropped — we now inspect
    # archive keys instead, so any folder whose npz carries
    # `bins` + `counts` surfaces as a histogram.
    if os.path.exists(submission_folder):
        for folder_name in os.listdir(submission_folder):
            folder_path = os.path.join(submission_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue
            hist_file = os.path.join(folder_path, f'{sample.name}.npz')
            if os.path.exists(hist_file):
                try:
                    with np.load(hist_file) as data:
                        if not {'bins', 'counts'}.issubset(data.keys()):
                            continue
                        bins = data['bins'].tolist()
                        counts = data['counts'].tolist()

                        # Store properly namespaced data for all histograms
                        pred_data[f'histogram_{folder_name}'] = {'bins': bins, 'counts': counts}

                        # Restore top-level bins/counts when no other
                        # histogram has filled them yet (legacy
                        # zoom-modal fallback expects the keys present).
                        if not pred_data['bins'] and not pred_data['counts']:
                            pred_data['bins'] = bins
                            pred_data['counts'] = counts
                            
                except Exception as e:
                    print(f"Error loading histogram from {folder_name}: {e}")

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


# Jinja filter: drop trailing '.0' on integer-valued floats so scalars
# imported from int sources (ClassLabel, count metrics) render as ints.
app.jinja_env.filters['smart_num'] = _smart_num


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


# ===================== HuggingFace token =====================
# Stored alongside the user row so HF imports don't need to ask for a
# token every time. Auto-saved on import; manageable here.

@app.route('/settings/hf_token', methods=['GET'])
@login_required
def hf_token_settings():
    return render_template('hf_token_settings.html', token=g.current_user.hf_token)


@app.route('/settings/hf_token/save', methods=['POST'])
@login_required
def hf_token_save():
    token = (request.form.get('hf_token') or '').strip()
    if not token:
        flash("Empty token — nothing saved.", "warning")
        return redirect(url_for('hf_token_settings'))
    g.current_user.hf_token = token
    db.session.commit()
    flash("HuggingFace token saved.", "success")
    return redirect(url_for('hf_token_settings'))


@app.route('/settings/hf_token/remove', methods=['POST'])
@login_required
def hf_token_remove():
    g.current_user.hf_token = None
    db.session.commit()
    flash("HuggingFace token removed. Future imports will need it again.", "warning")
    return redirect(url_for('hf_token_settings'))


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


@app.route('/admin/cache_stats', methods=['GET'])
@login_required
def admin_cache_stats():
    """Summarize disk usage for the bench_cache, the LB-scoped GT
    snapshots, and uploaded submission folders. Useful for spotting
    a runaway HF repo or a submission that hasn't been evicted yet.

    All counts/sizes are DB-derived where possible (cache rows carry
    their own size_bytes via bench_cache.cache_put). Submission-folder
    disk usage is computed with os.walk so it doesn't depend on a
    monotonically-maintained counter."""
    if not is_admin(g.current_user):
        abort(403)
    from collections import defaultdict
    from bench_cache import resolve_budget_bytes

    # 1. bench_cache: split by origin, then group GT entries by
    #    (repo, split, column) and submission entries by submission id
    #    (the submission id is the first chunk after the `submission:`
    #    prefix in our caching helpers).
    cache_root = app.config.get('CACHE_FOLDER') or ''
    budget_bytes = resolve_budget_bytes(cache_root) if cache_root else 0
    rows = CacheEntry.query.with_entities(
        CacheEntry.cache_key, CacheEntry.size_bytes,
        CacheEntry.origin, CacheEntry.last_accessed_at,
    ).all()
    gt_groups = defaultdict(lambda: {'count': 0, 'bytes': 0, 'last': None})
    sub_groups = defaultdict(lambda: {'count': 0, 'bytes': 0, 'last': None})
    total_bytes_gt = 0
    total_bytes_sub = 0
    for key, size, origin, last in rows:
        if origin == 'gt':
            # gt_viz:repo@rev:split:col:idx  → group on (repo, split, col)
            label = key
            if key.startswith('gt_viz:'):
                tail = key[len('gt_viz:'):]
                bits = tail.split(':')
                if len(bits) >= 4:
                    repo_rev = bits[0]
                    repo = repo_rev.split('@', 1)[0]
                    split = bits[1]
                    col = bits[2]
                    label = f"{repo} · {split} · {col}"
            entry = gt_groups[label]
            entry['count'] += 1
            entry['bytes'] += int(size or 0)
            if last is not None and (entry['last'] is None or last > entry['last']):
                entry['last'] = last
            total_bytes_gt += int(size or 0)
        elif origin == 'submission':
            label = key.split(':', 2)[1] if ':' in key else key
            entry = sub_groups[label]
            entry['count'] += 1
            entry['bytes'] += int(size or 0)
            if last is not None and (entry['last'] is None or last > entry['last']):
                entry['last'] = last
            total_bytes_sub += int(size or 0)

    def _sorted(groups):
        return sorted(
            [{'label': k, **v} for k, v in groups.items()],
            key=lambda r: r['bytes'], reverse=True,
        )

    # 2. LB-scoped GT snapshot rows (CustomField with no sample_id / no
    #    submission_id, scoped to a leaderboard).
    snapshot_summary = []
    snapshot_rows = (
        db.session.query(
            CustomField.leaderboard_id,
            CustomField.field_type,
            func.count(CustomField.id),
        )
        .filter(CustomField.leaderboard_id.isnot(None))
        .filter(CustomField.submission_id.is_(None))
        .filter(CustomField.sample_id.is_(None))
        .group_by(CustomField.leaderboard_id, CustomField.field_type)
        .all()
    )
    snap_by_lb = defaultdict(lambda: {'count': 0, 'by_kind': defaultdict(int)})
    for lb_id, ftype, n in snapshot_rows:
        snap_by_lb[lb_id]['count'] += n
        snap_by_lb[lb_id]['by_kind'][ftype] += n
    for lb_id, info in snap_by_lb.items():
        lb = Leaderboard.query.get(lb_id)
        snapshot_summary.append({
            'lb_id': lb_id,
            'lb_name': lb.name if lb else f"(deleted lb {lb_id})",
            'total': info['count'],
            'by_kind': dict(info['by_kind']),
        })
    snapshot_summary.sort(key=lambda r: r['total'], reverse=True)

    # 3. Submission folder usage — du across uploads/submissions/<id>/.
    #    Cheap on typical Fly volumes (a few hundred subdirs); cap the
    #    listing at the top 20 to keep the render fast.
    sub_dir = os.path.join(app.config.get('UPLOAD_FOLDER', ''), 'submissions')
    sub_folders = []
    if os.path.isdir(sub_dir):
        for entry in os.listdir(sub_dir):
            full = os.path.join(sub_dir, entry)
            if not os.path.isdir(full):
                continue
            size = 0
            for root, _dirs, files in os.walk(full):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            sub_folders.append({'name': entry, 'bytes': size})
    sub_folders.sort(key=lambda r: r['bytes'], reverse=True)
    total_sub_folder_bytes = sum(r['bytes'] for r in sub_folders)
    top_sub_folders = sub_folders[:20]

    # 4. Datasets — already track a counter via Dataset.storage_bytes.
    dataset_rows = (
        Dataset.query.with_entities(
            Dataset.id, Dataset.name, Dataset.storage_bytes,
        )
        .order_by(Dataset.storage_bytes.desc())
        .all()
    )
    datasets_summary = [
        {'id': did, 'name': dname, 'bytes': int(dsize or 0)}
        for did, dname, dsize in dataset_rows
    ]
    total_dataset_bytes = sum(r['bytes'] for r in datasets_summary)

    return render_template(
        'admin_cache_stats.html',
        cache_root=cache_root,
        budget_bytes=budget_bytes,
        cache_total_bytes=total_bytes_gt + total_bytes_sub,
        gt_total_bytes=total_bytes_gt,
        sub_total_bytes=total_bytes_sub,
        gt_groups=_sorted(gt_groups),
        sub_groups=_sorted(sub_groups),
        snapshot_summary=snapshot_summary,
        top_sub_folders=top_sub_folders,
        total_sub_folder_bytes=total_sub_folder_bytes,
        sub_folder_count=len(sub_folders),
        datasets_summary=datasets_summary,
        total_dataset_bytes=total_dataset_bytes,
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


@app.route('/admin/leaderboard/<int:lb_id>/sota_picker', methods=['GET'])
@login_required
def admin_lb_sota_picker(lb_id):
    """Admin-only: pick a SOTA model from HF Hub trending (filtered to
    the LB's task slug) and seed a Colab submission notebook for it.
    Free-form input is also offered for cases where the heuristic
    misclassifies the task or the admin wants something the trending
    list doesn't surface."""
    if not is_admin(g.current_user):
        abort(403)
    lb = Leaderboard.query.get_or_404(lb_id)
    task_slug = _hf_task_for_lb(lb)
    # The HF dataset the LB attaches to (PWC-imported LBs always have
    # this; legacy BH-only LBs may not). Used to filter the model list
    # to ones actually trained on the right dataset.
    hf_dataset = None
    for att in (lb.attachments or []):
        if getattr(att, 'kind', None) == 'hf' and att.role == 'primary' and att.hf_repo_id:
            hf_dataset = att.hf_repo_id
            break
    if not hf_dataset and lb.canonical_for_repo:
        hf_dataset = lb.canonical_for_repo
    result = _hf_trending_sota_models(
        task_slug, limit=20, dataset_repo=hf_dataset,
    ) if (task_slug or hf_dataset) else {
        'models': [], 'filter_level': 'unfiltered', 'tried': [],
    }
    return render_template(
        'admin_lb_sota_picker.html',
        leaderboard=lb,
        task_slug=task_slug,
        hf_dataset=hf_dataset,
        models=result['models'],
        filter_level=result['filter_level'],
        filters_tried=result['tried'],
    )


def _sota_cache_path(lb_id, model_id):
    """On-disk cache for SOTA notebooks keyed by (lb, model). The
    Claude generation is a 1-3 min call we don't want to repeat for
    every admin click on the same row. Cache lives on the volume so
    it survives gunicorn worker recycles."""
    base = os.environ.get('BENCHHUB_DATA_DIR') or os.path.expanduser('~/.dtofbenchmarking')
    cache_dir = os.path.join(base, '_cache', 'sota')
    os.makedirs(cache_dir, exist_ok=True)
    safe_model = re.sub(r'[^A-Za-z0-9_-]+', '_', model_id) or 'model'
    return os.path.join(cache_dir, f'{lb_id}__{safe_model}.json')


def _read_sota_cache(lb_id, model_id):
    """Return {notebook, gist_id, gist_owner} or None if missing /
    unparseable / older than the prompt-version cap. The PROMPT_VERSION
    bump invalidates every cached entry, which is the migration path
    for prompt changes (Claude generated stale code under the old
    rules)."""
    PROMPT_VERSION = 6  # bump when _llm_colab_notebook prompt changes
    path = _sota_cache_path(lb_id, model_id)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get('prompt_version') != PROMPT_VERSION:
        return None
    if not data.get('notebook'):
        return None
    return data


def _write_sota_cache(lb_id, model_id, *, notebook, gist_id=None, gist_owner=None):
    PROMPT_VERSION = 6
    path = _sota_cache_path(lb_id, model_id)
    try:
        with open(path, 'w') as f:
            json.dump({
                'prompt_version': PROMPT_VERSION,
                'model_id': model_id,
                'notebook': notebook,
                'gist_id': gist_id,
                'gist_owner': gist_owner,
            }, f)
    except OSError as e:
        print(f"_write_sota_cache failed for lb={lb_id} model={model_id!r}: {e}")


def _push_one_off_gist(notebook_json, *, filename, description):
    """Create a fresh secret GitHub gist for this notebook and return
    (gist_url, gist_id, gist_owner). Returns (None, None, None) when
    BENCHHUB_GITHUB_GIST_TOKEN is missing or GitHub rejects the call.

    Unlike _ensure_colab_gist this never PATCHes an existing gist —
    the SOTA flow generates one notebook per (LB, model), so each
    deserves its own gist. We don't cache the id either; admin can
    regenerate freely."""
    token = os.environ.get('BENCHHUB_GITHUB_GIST_TOKEN')
    if not token:
        return None, None, None
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    payload = {
        'description': description,
        'public': False,
        'files': {filename: {'content': notebook_json}},
    }
    try:
        import requests as _r
        resp = _r.post(
            'https://api.github.com/gists',
            headers=headers, json=payload, timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        return (
            body.get('html_url'),
            body.get('id'),
            (body.get('owner') or {}).get('login'),
        )
    except Exception as e:
        print(f"_push_one_off_gist failed: {e}")
        return None, None, None


@app.route('/admin/leaderboard/<int:lb_id>/sota_notebook', methods=['POST'])
@login_required
def admin_lb_sota_notebook(lb_id):
    """Generate a SOTA-model-specific Colab notebook for this LB and
    push it to a GitHub gist so we can redirect straight to
    colab.research.google.com/gist/<owner>/<id>. Falls back to a
    direct .ipynb download when the gist token isn't configured or
    the GitHub API call fails — admin can still upload the file
    manually."""
    if not is_admin(g.current_user):
        abort(403)
    lb = Leaderboard.query.get_or_404(lb_id)
    model_id = (request.form.get('model_id') or '').strip()
    if not model_id:
        flash("HF model id is required.", "warning")
        return redirect(url_for('admin_lb_sota_picker', lb_id=lb.id))
    force = request.form.get('force') == '1'

    # Cache hit: serve the previous notebook (and its gist if we have
    # one) — saves the 1-3 min Claude call. Force regen via the
    # picker form's "Regenerate even if cached" checkbox; also
    # auto-invalidates when the PROMPT_VERSION bump in
    # _read_sota_cache fires (deploys with prompt changes).
    cached = None if force else _read_sota_cache(lb.id, model_id)

    safe_model = re.sub(r'[^A-Za-z0-9_-]+', '_', model_id) or 'model'
    safe_lb = re.sub(r'[^A-Za-z0-9_-]+', '_', lb.name) or f'lb_{lb.id}'
    filename = f'{safe_lb}__{safe_model}_sota.ipynb'
    description = (
        f"BenchHub SOTA submission scaffold for '{lb.name}' "
        f"(model={model_id})"
    )

    if cached:
        nb = cached['notebook']
        gist_id = cached.get('gist_id')
        gist_owner = cached.get('gist_owner')
        # If we cached a gist URL last time AND it still exists on
        # GitHub, redirect immediately. Otherwise re-push as a one-off.
        if not gist_id:
            _gist_url, gist_id, gist_owner = _push_one_off_gist(
                nb, filename=filename, description=description,
            )
            if gist_id:
                _write_sota_cache(lb.id, model_id, notebook=nb,
                                  gist_id=gist_id, gist_owner=gist_owner)
    else:
        nb = _llm_sota_colab_notebook(lb, model_id)
        if not nb:
            flash(
                f"Couldn't generate a SOTA notebook for {model_id} "
                "(LLM unavailable, model id rejected by Claude, or "
                "response wasn't valid notebook JSON). Try a different "
                "model or check ANTHROPIC_API_KEY.",
                "danger",
            )
            return redirect(url_for('admin_lb_sota_picker', lb_id=lb.id))
        _gist_url, gist_id, gist_owner = _push_one_off_gist(
            nb, filename=filename, description=description,
        )
        # Cache regardless of gist outcome so the next click skips
        # the LLM call. Gist push is cheap to retry on read-back.
        _write_sota_cache(lb.id, model_id, notebook=nb,
                          gist_id=gist_id, gist_owner=gist_owner)

    if gist_id:
        path = f'{gist_owner}/{gist_id}' if gist_owner else gist_id
        return redirect(f'https://colab.research.google.com/gist/{path}')
    # Fallback: direct download. Inline the flash so admin understands
    # why they got a file instead of a Colab tab.
    flash(
        "Couldn't publish the SOTA notebook as a GitHub gist "
        "(BENCHHUB_GITHUB_GIST_TOKEN missing or GitHub rejected the "
        "call). Downloaded the .ipynb instead — upload it manually "
        "via Colab → File → Upload notebook.",
        "warning",
    )
    return app.response_class(
        nb,
        mimetype='application/x-ipynb+json',
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"',
        },
    )


@app.route('/admin/leaderboard/<int:lb_id>/promote', methods=['POST'])
@login_required
def admin_promote_leaderboard(lb_id):
    """Mark an LB as the canonical public leaderboard for an HF repo.

    Why: anyone can fork a public dataset into their own personal LB; we
    only want one entry on /explore per repo so the catalog stays
    legible. Admin-curated single-source-of-truth.

    Form params:
      - canonicality: 'public' (promote) or 'personal' (demote)
      - canonical_for_repo: HF repo id (only used on 'public')
    """
    if not is_admin(g.current_user):
        abort(403)
    lb = Leaderboard.query.get_or_404(lb_id)
    target = (request.form.get('canonicality') or 'public').strip()
    if target not in ('public', 'personal'):
        flash("canonicality must be 'public' or 'personal'.", "warning")
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    if target == 'personal':
        lb.canonicality = 'personal'
        lb.canonical_for_repo = None
        db.session.commit()
        flash(f"Demoted '{lb.name}' to personal.", "info")
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))

    # Promotion path. Repo binding is optional (a public LB can exist
    # without a 1:1 HF repo binding), but if provided, enforce uniqueness.
    repo = (request.form.get('canonical_for_repo') or '').strip() or None
    if repo:
        clash = (Leaderboard.query
                 .filter(Leaderboard.canonical_for_repo == repo,
                         Leaderboard.id != lb.id)
                 .first())
        if clash is not None:
            flash(
                f"Repo '{repo}' is already canonicalized by leaderboard "
                f"'{clash.name}' (id={clash.id}). Demote that one first.",
                "warning",
            )
            return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    lb.canonicality = 'public'
    lb.canonical_for_repo = repo
    db.session.commit()
    label = f"public canonical for {repo}" if repo else "public"
    flash(f"Promoted '{lb.name}' to {label}.", "success")
    return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))


# ===================== Papers With Code import =====================
# Admin-only mirror flow: search PWC benchmarks → preview → confirm to
# create a canonical leaderboard whose primary attachment is the
# benchmark's HuggingFace mirror, plus one mirrored Submission per
# PWC result row. Mirrored submissions skip Celery entirely; their
# scores are inserted directly as MetricResult rows.


@app.route('/admin/pwc/import', methods=['GET'])
@login_required
def admin_pwc_import():
    """Search the PWC static archive. The actual archive download +
    index build is heavy enough to OOM/timeout a gunicorn worker, so
    it lives in a Celery task. The page renders one of three states:
      ready    → real search.
      building → "still indexing, refresh later".
      error / absent → "Build the index" CTA that enqueues the task.
    """
    if not is_admin(g.current_user):
        abort(403)
    from pwc_client import (
        index_status, index_error_message, index_progress_message,
    )
    status = index_status()
    err_message = index_error_message() if status == 'error' else ''
    progress = index_progress_message() if status == 'building' else ''
    q = (request.args.get('q') or '').strip()
    rows = []
    error = None
    if status == 'ready':
        # Empty query → top-N most-populated datasets so the admin
        # can browse without having to know what they're looking for.
        try:
            from pwc_client import search_datasets
            rows = search_datasets(q, limit=30)
        except Exception as e:
            error = str(e)
    return render_template(
        'admin_pwc_import.html',
        q=q, rows=rows, error=error,
        index_status=status, index_error=err_message,
        index_progress=progress,
    )


@app.route('/admin/pwc/index/build', methods=['POST'])
@login_required
def admin_pwc_index_build():
    """Disabled in v110. The pyarrow decode of the pwc-archive parquet's
    nested struct schema turned out to be unreliable on a 2 GB / 4 GB
    fly worker — successive optimization rounds (recursion cap, column
    projection, batch sizing, sqlite PRAGMA tuning) all stuck inside
    the first batch's decode. Built the SQLite offline on a beefier
    box instead and shipped it via `scripts/upload_pwc_index.py`.

    The endpoint stays around as a 403/flash rather than a 404 so an
    admin who bookmarked it gets a sensible message, not a generic
    error page."""
    if not is_admin(g.current_user):
        abort(403)
    flash(
        "PWC index is shipped pre-built — no in-prod build available. "
        "Ask the maintainer to refresh it via "
        "scripts/upload_pwc_index.py if you need fresher data.",
        "info",
    )
    return redirect(url_for('admin_pwc_import'))


@app.route('/admin/pwc/import/dataset/<int:dataset_id>', methods=['GET'])
@login_required
def admin_pwc_import_dataset(dataset_id):
    """List benchmarks tracked on a single dataset. `hf_repo` is the
    admin's pick from the search page (or whatever they typed) — we
    pass it through so the preview/confirm steps know what HF repo
    to attach the LB to."""
    if not is_admin(g.current_user):
        abort(403)
    hf_repo = (request.args.get('hf_repo') or '').strip()
    try:
        from pwc_client import list_evaluations_for_dataset
        evals = list_evaluations_for_dataset(dataset_id)
    except Exception as e:
        flash(f"PWC error: {e}", "danger")
        return redirect(url_for('admin_pwc_import'))
    return render_template(
        'admin_pwc_import_dataset.html',
        dataset_id=dataset_id, hf_repo=hf_repo, evals=evals,
    )


@app.route('/admin/pwc/import/preview/<int:evaluation_id>', methods=['GET'])
@login_required
def admin_pwc_import_preview(evaluation_id):
    """Preview what would be created. `hf_repo` is admin-editable on
    this page since the archive rarely carries it; the form on this
    page POSTs both the LB name and the (possibly-edited) hf_repo to
    /confirm/."""
    if not is_admin(g.current_user):
        abort(403)
    hf_repo = (request.args.get('hf_repo') or '').strip()
    try:
        from pwc_client import get_evaluation, suggest_hf_repo
        evaluation = get_evaluation(evaluation_id)
    except Exception as e:
        flash(f"PWC error: {e}", "danger")
        return redirect(url_for('admin_pwc_import'))
    task = evaluation.get('task') or 'benchmark'
    dataset = evaluation.get('dataset') or hf_repo or 'unknown-dataset'
    # If admin didn't carry an HF repo through from the search page,
    # ask HF Hub for the best match. The preview page will pre-fill
    # the field with this guess and offer alternatives.
    hf_suggestion = None
    hf_alternatives = []
    if not hf_repo and dataset:
        try:
            hf_suggestion, hf_alternatives = suggest_hf_repo(dataset)
        except Exception as e:
            print(f"HF suggest failed for {dataset!r}: {e}")
    suggested_name = f"{task} on {dataset}"
    return render_template(
        'admin_pwc_import_preview.html',
        evaluation=evaluation, hf_repo=hf_repo,
        hf_suggestion=hf_suggestion,
        hf_alternatives=hf_alternatives,
        suggested_name=suggested_name,
    )


@app.route('/admin/pwc/import/confirm/<int:evaluation_id>', methods=['POST'])
@login_required
def admin_pwc_import_confirm(evaluation_id):
    """Create the canonical LB + mirrored submissions in one transaction."""
    if not is_admin(g.current_user):
        abort(403)
    hf_repo = (request.form.get('hf_repo') or '').strip()
    lb_name = (request.form.get('leaderboard_name') or '').strip()
    if not hf_repo or not lb_name:
        flash("Missing HF repo or LB name.", "danger")
        return redirect(url_for('admin_pwc_import'))
    try:
        from pwc_client import get_evaluation
        evaluation = get_evaluation(evaluation_id)
    except Exception as e:
        flash(f"PWC error: {e}", "danger")
        return redirect(url_for('admin_pwc_import'))
    try:
        lb_id = _create_lb_from_pwc_benchmark(
            evaluation, hf_repo=hf_repo, lb_name=lb_name,
            owner_user_id=g.current_user.id,
        )
    except Exception as e:
        db.session.rollback()
        flash(f"Import failed: {e}", "danger")
        return redirect(url_for('admin_pwc_import'))
    flash(
        f"Imported PWC benchmark — created '{lb_name}' with "
        f"{len(evaluation.get('results') or [])} mirrored submissions.",
        "success",
    )
    return redirect(url_for('leaderboard_view', leaderboard_id=lb_id))


def _bulk_import_pwc_benchmarks(*, max_imports=25, min_results=10,
                                 owner_user_id, logger=None):
    """Sweep the PWC archive for evaluations with at least `min_results`
    result rows, where an HF dataset repo can be inferred from the
    PWC dataset's `links_json`, and import them as canonical
    leaderboards.

    Skips evaluations whose inferred HF repo already has a canonical
    LB (we don't want duplicates), and skips per-benchmark failures
    (HF schema probe error, LLM call error, etc.) so one bad import
    doesn't abort the batch. Imports are committed individually.

    Returns a dict::

        {
          'imported':  [(eval_id, lb_id, lb_name, hf_repo, n_results), ...],
          'skipped':   [(eval_id, dataset, reason), ...],
          'failed':    [(eval_id, dataset, reason), ...],
        }
    """
    import sqlite3
    from pwc_client import (
        _index_path, _hf_repo_from_links, get_evaluation, suggest_hf_repo,
    )

    _log = (logger or print)
    out = {'imported': [], 'skipped': [], 'failed': []}

    # 1. Pull candidate (evaluation_id, dataset_name, links_json,
    # n_results) tuples ordered by n_results desc so the most
    # heavily-populated benchmarks come first.
    try:
        conn = sqlite3.connect(_index_path())
    except Exception as e:
        _log(f"PWC archive open failed: {e}")
        return out
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT e.id, d.name, d.links_json, "
            "       json_array_length(e.results_json) AS n_results "
            "FROM pwc_evaluation e "
            "JOIN pwc_dataset d ON d.id = e.dataset_id "
            "WHERE json_array_length(e.results_json) >= ? "
            "ORDER BY n_results DESC, e.id ASC",
            (int(min_results),),
        )
        candidates = cur.fetchall()
    finally:
        conn.close()

    # 2. Pre-compute the set of HF repos that already have a canonical
    # LB so we can skip-without-DB-roundtrip.
    existing = {
        repo for (repo,) in db.session.query(Leaderboard.canonical_for_repo)
        .filter(Leaderboard.canonical_for_repo.isnot(None))
        .all()
    }

    # 3. Walk candidates, attempt import for each. Cap when we reach
    # max_imports successful creations. Two-tier HF-repo inference:
    # cheap `_hf_repo_from_links` from the PWC archive's link
    # metadata first; fall back to `suggest_hf_repo` (queries HF Hub)
    # when the archive doesn't carry an HF link, which is the common
    # case. Cap the suggest-fallback attempts so a 1500-candidate run
    # doesn't burn HF's anonymous quota chasing dead-end names.
    suggest_budget = max(max_imports * 8, 100)
    suggest_attempts = 0
    for eval_id, dataset, links_json, n_results in candidates:
        if len(out['imported']) >= max_imports:
            break
        hf_repo = _hf_repo_from_links(links_json)
        if not hf_repo and dataset and suggest_attempts < suggest_budget:
            suggest_attempts += 1
            try:
                guessed, _alts = suggest_hf_repo(dataset)
            except Exception as e:
                guessed = None
                _log(f"suggest_hf_repo({dataset!r}) raised: {e}")
            if guessed:
                hf_repo = guessed
        if not hf_repo:
            out['skipped'].append((eval_id, dataset, 'no inferable HF repo'))
            continue
        if hf_repo in existing:
            out['skipped'].append(
                (eval_id, dataset, f'already imported ({hf_repo})')
            )
            continue
        try:
            evaluation = get_evaluation(eval_id)
        except Exception as e:
            out['failed'].append((eval_id, dataset, f'get_evaluation: {e}'))
            continue
        task = evaluation.get('task') or 'benchmark'
        lb_name = f"{task} on {dataset}"
        # Cap at 100 chars to fit the Leaderboard.name column.
        if len(lb_name) > 100:
            lb_name = lb_name[:97] + '...'
        # Skip if an LB with this exact name already exists (rare —
        # mostly a previous failed import that committed the LB row
        # before bailing).
        if Leaderboard.query.filter_by(name=lb_name).first():
            out['skipped'].append(
                (eval_id, dataset, f'name collision: {lb_name!r}')
            )
            continue
        try:
            lb_id = _create_lb_from_pwc_benchmark(
                evaluation, hf_repo=hf_repo, lb_name=lb_name,
                owner_user_id=owner_user_id,
            )
            out['imported'].append((eval_id, lb_id, lb_name, hf_repo, n_results))
            existing.add(hf_repo)  # don't re-import the same repo this batch
            _log(
                f"[{len(out['imported'])}/{max_imports}] "
                f"imported eval={eval_id} lb={lb_id} {lb_name!r} "
                f"({n_results} mirrored)"
            )
        except Exception as e:
            db.session.rollback()
            out['failed'].append((eval_id, dataset, repr(e)[:200]))
            _log(f"FAILED eval={eval_id} dataset={dataset!r}: {e}")

    return out


def _create_lb_from_pwc_benchmark(evaluation, *, hf_repo, lb_name,
                                   owner_user_id):
    """Persist a canonical-public LB whose primary attachment is the
    HF dataset behind the PWC benchmark, plus one mirrored Submission
    per PWC result row with MetricResult rows pre-populated. All in
    one transaction.

    The created GlobalMetric rows use slugified PWC metric names so
    future verified submissions land in the same column the mirrored
    rows populate — that's the "metric matching" promise from the
    earlier design discussion.
    """
    from pwc_client import slugify_metric_name
    pwc_metrics = evaluation.get('metrics') or []
    pwc_results = evaluation.get('results') or []

    # 1. Create LB. Admin imports → mark canonical for the HF repo so
    # the LB shows up on /explore and "submit there instead" hints
    # surface for users importing the same repo themselves.
    # summary_metrics is matched against LeaderboardMetric.target_name (or
    # lm_<id>) by the LB view's metric resolver — slugified GlobalMetric
    # names won't resolve and get auto-pruned, leaving the view with no
    # metric columns. Use the PWC names verbatim so name_to_lmids resolves.
    lb = Leaderboard(
        name=lb_name,
        summary_metrics=','.join(m['name'] for m in pwc_metrics),
        owner_user_id=owner_user_id,
        visibility='public',
        canonicality='public',
        canonical_for_repo=hf_repo,
        category=_pwc_task_to_category(evaluation.get('task')),
    )
    db.session.add(lb); db.session.flush()

    # 2. HF-ref attachment. Probe the dataset's parquet schema and run
    # _infer_mapping so the LeaderboardMetric arg_mappings can target a
    # real GT field (e.g. `gt_label` for cifar10) instead of placeholder
    # 'gt_unknown'. Generated SOTA notebooks get real PRED_FIELDS to
    # write into, so submissions actually score. Falls through to an
    # empty mapping if the schema fetch fails — admin can still wire
    # it manually via the LB Settings page later.
    try:
        features = _hf_fetch_features(hf_repo)
    except Exception as e:
        print(f"PWC import: schema probe for {hf_repo} failed: {e}")
        features = {}
    inferred_mapping = _infer_mapping(features) if features else []
    db.session.add(Attachment(
        leaderboard_id=lb.id,
        hf_repo_id=hf_repo,
        hf_split='train',
        hf_mapping_json=json.dumps(inferred_mapping),
        role='primary',
    ))
    db.session.flush()

    # Pick the canonical GT field for arg_mappings. Priority:
    # scalar (ClassLabel-y) before depth before image — the most common
    # PWC-imported task is image classification where the scalar label
    # is what we want to score against.
    gt_field = None
    for kind in ('scalar', 'depth', 'image', 'mask', 'text'):
        for m in inferred_mapping:
            if m.get('target_kind') == kind:
                gt_field = m.get('target_field') or m.get('column')
                if gt_field:
                    break
        if gt_field:
            break
    pred_arg_field = f"sub_{gt_field}_pred" if gt_field else 'sub_unknown_pred'
    gt_arg_field = f"gt_{gt_field}" if gt_field else 'gt_unknown'

    # 3. Build GlobalMetric + LeaderboardMetric for each PWC metric.
    # PWC doesn't ship runnable code (just names + descriptions), so
    # mirrored-only metrics get a stub that errors at eval time —
    # that's intentional: until someone authors the matching code via
    # the LB's Settings page (Add a metric / AI authoring), verified
    # submissions on this column would have nothing to score against.
    metric_id_by_pwc_name = {}  # PWC name → LeaderboardMetric.id
    metric_dir_by_pwc_name = {}  # for sort_direction lookup
    for pm in pwc_metrics:
        gn = slugify_metric_name(pm['name'])
        sd = pm.get('sort_direction') or 'higher_is_better'
        gm = GlobalMetric.query.filter_by(name=gn).first()
        if gm is None:
            # Author runnable code for the metric via Claude. PWC's
            # metric names are well-known (top-1 accuracy, BLEU-4,
            # RMSE, mIoU, etc.) so the LLM produces solid code most
            # of the time. Fall back to a NotImplementedError stub
            # only when the API key is missing or the call fails —
            # admin can fix it later via the LB Settings → Add a
            # metric flow (Phase 8).
            llm_hint = (
                f"PWC metric '{pm['name']}'. {pm.get('description') or ''} "
                f"Per-sample function `def {gn}(gt, pred)` returning a "
                f"float. Use stdlib + numpy (already imported as np). "
                f"Be defensive about types — gt/pred may arrive as int, "
                f"float, str, list, or numpy array depending on the "
                f"underlying dataset. Direction: "
                f"{'lower' if sd == 'lower_is_better' else 'higher'} is better."
            )
            authored_code = None
            try:
                authored_code = _llm_generate_metric_code(gn, llm_hint)
            except Exception as e:
                print(f"PWC metric authoring failed for {gn}: {e}")
            python_code = authored_code or (
                f"def {gn}(gt, pred):\n"
                f"    \"\"\"PWC-imported metric. AI authoring was\n"
                f"    unavailable at import time. Edit this on the\n"
                f"    LB Settings page before accepting verified\n"
                f"    submissions on this column.\"\"\"\n"
                f"    raise NotImplementedError(\n"
                f"        '{gn} needs a reference implementation. "
                f"Edit it on the LB Settings page.'\n"
                f"    )\n"
            )
            gm = GlobalMetric(
                name=gn,
                description=(
                    f"{pm['name']} — imported from Papers With Code. "
                    f"{pm.get('description') or ''}"
                ).strip(),
                python_code=python_code,
                is_aggregated=False, accepts_aggregated_inputs=False,
                owner_user_id=owner_user_id, visibility='public',
            )
            db.session.add(gm); db.session.flush()
        lm = LeaderboardMetric(
            leaderboard_id=lb.id, global_metric_id=gm.id,
            target_name=pm['name'],
            arg_mappings=json.dumps({'gt': gt_arg_field,
                                     'pred': pred_arg_field}),
            sort_direction=sd,
            pooling_type='mean',
        )
        db.session.add(lm); db.session.flush()
        metric_id_by_pwc_name[pm['name']] = lm.id
        metric_dir_by_pwc_name[pm['name']] = sd

    # 4. Per PWC result row → one mirrored Submission + per-metric
    # MetricResult rows. Use the paper title (or a generated stub) as
    # the submission name so the LB page reads naturally.
    n_subs = 0
    for r in pwc_results:
        title = (r.get('paper_title') or r.get('methodology')
                 or f"PWC result {r.get('id')}").strip()
        sub = Submission(
            name=title[:100], leaderboard_id=lb.id,
            kind='mirrored', processing_status='Mirrored',
            source_attribution='Papers With Code',
            source_paper_title=title[:300] or None,
            source_paper_url=(r.get('paper_url') or r.get('paper')) or None,
            source_external_url=r.get('external_source_url'),
            owner_user_id=owner_user_id,
        )
        db.session.add(sub); db.session.flush()
        # PWC result.metrics is a dict of {metric_name → value-string}.
        for pwc_name, raw_val in (r.get('metrics') or {}).items():
            lm_id = metric_id_by_pwc_name.get(pwc_name)
            if lm_id is None:
                continue
            try:
                val = float(re.sub(r'[^0-9.\-]', '', str(raw_val))) if raw_val else None
            except (TypeError, ValueError):
                val = None
            if val is None:
                continue
            db.session.add(MetricResult(
                submission_id=sub.id,
                leaderboard_metric_id=lm_id,
                value=val,
            ))
        n_subs += 1
    db.session.commit()
    return lb.id


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


def _compute_explorable_lb_ids(lb_ids):
    """Return the subset of `lb_ids` whose LB has cached GT samples
    somewhere — i.e. clicking "Explore samples" will surface real rows.
    BH-attached LBs qualify once the `Sample` table has rows for any of
    their attached datasets. HF-attached LBs qualify once
    `populate_lb_samples` has written LB-scoped GT CustomFields
    (sample_id+submission_id both NULL). Templates render a
    green-checkmark pill on cards in this set and a "no samples yet"
    pill on the rest, so users don't waste a click on a LB whose GT
    pipeline hasn't run (or failed)."""
    if not lb_ids:
        return set()
    lb_ids = list(lb_ids)
    bh_explorable = (
        db.session.query(leaderboard_datasets.c.leaderboard_id)
        .join(Sample, Sample.dataset_id == leaderboard_datasets.c.dataset_id)
        .filter(leaderboard_datasets.c.leaderboard_id.in_(lb_ids))
        .distinct()
    )
    hf_explorable = (
        db.session.query(CustomField.leaderboard_id)
        .filter(
            CustomField.leaderboard_id.in_(lb_ids),
            CustomField.submission_id.is_(None),
            CustomField.sample_id.is_(None),
        )
        .distinct()
    )
    return (
        {row[0] for row in bh_explorable.all()}
        | {row[0] for row in hf_explorable.all()}
    )


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
    explorable_lb_ids = _compute_explorable_lb_ids([lb.id for lb, _ in featured])


    return render_template(
        'landing.html',
        featured=featured_rows,
        explorable_lb_ids=explorable_lb_ids,
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
    return url_for('serve_custom_field_image', field_id=cf.id)


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
    # Split user's LBs into "public canonical" (admin-promoted, also
    # appear on /explore) and "personal" (their own work that hasn't
    # been promoted). /home shows both, public first.
    public_lbs = (
        Leaderboard.query
        .filter(Leaderboard.owner_user_id == user.id,
                Leaderboard.canonicality == 'public')
        .order_by(Leaderboard.upload_date.desc())
        .limit(24)
        .all()
    )
    personal_lbs = (
        Leaderboard.query
        .filter(Leaderboard.owner_user_id == user.id,
                Leaderboard.canonicality != 'public')
        .order_by(Leaderboard.upload_date.desc())
        .limit(24)
        .all()
    )
    leaderboards = public_lbs + personal_lbs  # legacy template var

    dataset_thumbs = {ds.id: _dataset_thumb_url(ds) for ds in datasets}
    leaderboard_thumbs = {}
    for lb in leaderboards:
        lb_datasets = list(lb.datasets)
        leaderboard_thumbs[lb.id] = (
            _dataset_thumb_url(lb_datasets[0]) if lb_datasets else None
        )
    explorable_lb_ids = _compute_explorable_lb_ids([lb.id for lb in leaderboards])

    return render_template(
        'home.html',
        datasets=datasets,
        leaderboards=leaderboards,
        public_lbs=public_lbs,
        personal_lbs=personal_lbs,
        dataset_thumbs=dataset_thumbs,
        leaderboard_thumbs=leaderboard_thumbs,
        explorable_lb_ids=explorable_lb_ids,
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
    # ?category=Vision               → filter by area
    # ?category=Vision/Depth Estim.. → filter by area + task
    category_filter = (request.args.get('category') or '').strip()
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
        # /explore is the canonical catalog. Personal LBs live on their
        # owner's /home; surfacing every fork here makes the page noisy
        # and dilutes admin-promoted entries.
        .filter(Leaderboard.canonicality == 'public')
    )


    if q:
        base = base.filter(Leaderboard.name.ilike(f'%{q}%'))

    if tag_filter:
        base = (
            base.join(leaderboard_tags, leaderboard_tags.c.leaderboard_id == Leaderboard.id)
                .join(Tag, Tag.id == leaderboard_tags.c.tag_id)
                .filter(Tag.name == tag_filter)
        )

    if category_filter:
        # Single-arg ?category=Vision filters every LB whose category
        # starts with "Vision/" (or is exactly "Vision"). Two-segment
        # ?category=Vision/Depth+Estimation matches that exact path.
        if '/' in category_filter:
            base = base.filter(Leaderboard.category == category_filter)
        else:
            base = base.filter(or_(
                Leaderboard.category == category_filter,
                Leaderboard.category.like(f"{category_filter}/%"),
            ))

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
    # Per-LB "explorability" flag (see _compute_explorable_lb_ids).
    explorable_lb_ids = _compute_explorable_lb_ids([r['lb'].id for r in rows])

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
        # Tier controls SIZE only (1..5, by count). Color is now picked
        # per-tag-name via a deterministic hash so two different tags
        # never look identical even when they have the same count.
        # 12-hue palette is enough that adjacent tags don't collide
        # often; use crc32 instead of hash() so the assignment is
        # stable across processes.
        import zlib as _zlib
        tag_cloud = []
        for name, cnt in sorted(combined.items(), key=lambda kv: (-kv[1], kv[0])):
            tier = 1 + min(4, int((cnt / max_cnt) * 4))  # 1..5
            color_idx = _zlib.crc32(name.encode('utf-8')) % 12
            tag_cloud.append({
                'name': name, 'count': cnt,
                'tier': tier, 'color_idx': color_idx,
            })
    else:
        tag_cloud = []

    # Per-LB thumbnail (first dataset's thumb) so the explore cards
    # match the visual treatment on /home and the LB header.
    leaderboard_thumbs = {}
    for row in rows:
        # `rows` mixes dataclass / dict shapes depending on the path —
        # use [] indexing or attribute access defensively.
        lb = row['lb'] if isinstance(row, dict) else row.lb
        lb_datasets = list(lb.datasets)
        leaderboard_thumbs[lb.id] = (
            _dataset_thumb_url(lb_datasets[0]) if lb_datasets else None
        )

    # Category tree: per-area / per-task counts over visible+canonical LBs.
    # We compute counts independent of the current category filter so the
    # tree always shows the full breakdown (clicking a leaf scopes the
    # results panel but the tree itself stays stable).
    cat_rows = (
        db.session.query(Leaderboard.category, func.count(Leaderboard.id))
        .filter(visible_lb_filter)
        .filter(Leaderboard.canonicality == 'public')
        .filter(Leaderboard.category.isnot(None))
        .group_by(Leaderboard.category)
        .all()
    )
    category_tree = {}  # area → {'count': int, 'tasks': [(task, cnt)]}
    for cat, cnt in cat_rows:
        if not cat:
            continue
        if '/' in cat:
            area, task = cat.split('/', 1)
        else:
            area, task = cat, ''
        bucket = category_tree.setdefault(area, {'count': 0, 'tasks': {}})
        bucket['count'] += int(cnt or 0)
        if task:
            bucket['tasks'][task] = bucket['tasks'].get(task, 0) + int(cnt or 0)
    # Stable render: areas alphabetised, tasks by count desc then name.
    category_tree = [
        {
            'area': area,
            'count': v['count'],
            'tasks': sorted(
                [{'name': t, 'count': c} for t, c in v['tasks'].items()],
                key=lambda x: (-x['count'], x['name']),
            ),
        }
        for area, v in sorted(category_tree.items())
    ]

    return render_template(
        'explore.html',
        rows=rows,
        q=q,
        sort=sort,
        tag_cloud=tag_cloud,
        active_tag=tag_filter,
        leaderboard_thumbs=leaderboard_thumbs,
        category_tree=category_tree,
        active_category=category_filter,
        explorable_lb_ids=explorable_lb_ids,
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
        # Per-attached-dataset role updates: form sends one
        # `dataset_role_<id>` field per dataset row.
        for ds in leaderboard.datasets:
            new_role = request.form.get(f'dataset_role_{ds.id}')
            if new_role in ('primary', 'gt_source'):
                _set_lb_dataset_role(leaderboard.id, ds.id, new_role)
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

    # Per-attached-dataset role lookup so the template can render the
    # role dropdown next to each dataset row. Defaults to 'primary'.
    dataset_roles = {
        ds.id: (_lb_dataset_role(leaderboard.id, ds.id) or 'primary')
        for ds in leaderboard.datasets
    }
    return render_template('edit_leaderboard.html',
                           leaderboard=leaderboard,
                           dataset_fields=dataset_fields,
                           dataset_roles=dataset_roles,
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


@app.route('/supported_types')
def supported_types():
    """Standalone top-level page documenting every BenchHub field type
    and its storage convention. Sits in the primary nav next to
    Datasets / Leaderboards / Metrics / Visualizations — same
    discoverability as those, not buried inside /docs."""
    return render_template('supported_types.html')


# Heuristic-only metric → domain bucket. Inspect the lowercased
# metric name + description for known tokens; first match wins.
# Kept ordered so e.g. 'segmentation' beats 'accuracy' when both
# words appear in one name.
_METRIC_DOMAIN_RULES = [
    ('Segmentation & Detection', (
        r'\bm?iou\b', r'\bdice\b', r'\bmap\b', r'\bap50\b',
        r'\bap75\b', r'\bap\b', r'segmentation', r'detection',
    )),
    ('Regression & Depth', (
        r'\bmae\b', r'\bmse\b', r'\brmse\b', r'\brms\b', r'\bnmse\b',
        r'\bmape\b', r'absrel', r'sqrel', r'log10', r'silog',
        r'\bdelta\b', r'depth',
    )),
    ('Image Quality & Generation', (
        r'\bpsnr\b', r'\bssim\b', r'\blpips\b', r'\bfid\b', r'\bkid\b',
        r'\bis\b.*inception',
    )),
    ('NLP / Speech', (
        r'\bbleu\b', r'\brouge\b', r'\bmeteor\b', r'\bwer\b', r'\bcer\b',
        r'perplex', r'\bppl\b', r'\bf1\b.*\b(nlp|qa|squad)\b',
    )),
    ('Classification', (
        r'top.?\d', r'accuracy', r'\bf1\b', r'precision', r'recall',
        r'\bauc\b', r'\broc\b', r'\bmcc\b',
    )),
    ('Trajectory & Pose', (
        r'\bade\b', r'\bfde\b', r'\bpose\b', r'\bpck\b',
    )),
    ('Loss / Generic Error', (
        r'\bloss\b', r'\berror\b',
    )),
]


# PWC task name → top-level area. First regex hit wins, so order matters:
# Medical/Speech/Code MUST come before Vision/NLP so domain-specific words
# (e.g. "medical image segmentation", "speech recognition", "code generation")
# don't get grabbed by the broader Vision/NLP regexes.
_PWC_AREA_RULES = [
    ('Medical', (
        r'medical', r'\bmri\b', r'\bct\b.*scan', r'pathology',
        r'tumor', r'lesion', r'clinical', r'radiograph', r'\bx.?ray\b',
    )),
    ('Speech & Audio', (
        r'speech', r'\basr\b', r'voice', r'\baudio\b', r'\bsound\b',
        r'music', r'keyword.spotting', r'\btts\b',
    )),
    ('Code', (
        r'code.generation', r'code.completion', r'program.synthesis',
        r'\bprogramming\b',
    )),
    ('Reinforcement Learning', (
        r'reinforcement', r'\brl\b', r'atari', r'\bgame\b',
    )),
    ('Graph', (
        r'\bgraph\b', r'node.classification', r'link.prediction',
        r'molecul', r'protein.folding',
    )),
    ('Recommendation', (
        r'recommendation', r'recommender', r'collaborative.filtering',
    )),
    ('Time Series', (
        r'time.series', r'forecasting',
    )),
    ('Tabular', (
        r'tabular', r'feature.engineering',
    )),
    ('NLP', (
        r'translation', r'summarization', r'question.answering',
        r'sentiment', r'\bner\b', r'language.mod', r'\bnli\b',
        r'natural.language', r'\btext\b', r'parsing', r'reading.comprehension',
        r'dialog', r'paraphras', r'entail', r'relation.extraction',
        r'common.sense', r'reasoning', r'entity.linking', r'word.sense',
    )),
    ('Vision', (
        r'depth', r'segmentation', r'object.detection', r'image.classification',
        r'image.recognition', r'action.recognition', r'face.recognition',
        r'super.resolution', r'denoising', r'inpainting', r'optical.flow',
        r'tracking', r'\bocr\b', r'\b3d\b', r'reconstruction', r'pose.estimation',
        r'\bimage\b', r'\bvideo\b', r'visual', r'face', r'point.cloud',
        r'\bgan\b', r'image.generation', r'video.generation', r'deblur',
        r'\bre.identification\b', r'anomaly.detection', r'\bsr\b',
    )),
]


def _pwc_task_to_category(task_name):
    """PWC `task` string → "Area/Task".

    Two-step:
      1. Strip domain-modality prefixes (Medical, Aerial, Satellite,
         3D, Few-Shot, Zero-Shot, …) from the task name so e.g.
         "Medical Image Segmentation" → "Image Segmentation". The
         modality is metadata, not the task itself.
      2. Classify the stripped name into an area via the regex rules.

    Empty/unknown task → None so the caller can leave Leaderboard.
    category null."""
    if not task_name:
        return None
    raw = str(task_name).strip()
    # Order: domain-only prefixes (e.g. "Medical ") first, so
    # "Medical Image Segmentation" → "Image Segmentation". Multi-word
    # prefixes that swallow part of the task ("medical image ", which
    # would leave just "Segmentation") are explicitly NOT listed.
    _DOMAIN_PREFIXES = (
        'medical ', 'biomedical ',
        'aerial ', 'satellite ', 'remote sensing ',
        'document ', 'scientific ',
        'monocular ', 'stereo ',
        'few-shot ', 'few shot ', 'zero-shot ', 'zero shot ',
        'one-shot ', 'one shot ',
        'self-supervised ', 'self supervised ',
        'weakly-supervised ', 'weakly supervised ',
        'semi-supervised ', 'semi supervised ',
        'unsupervised ',
        'multi-task ', 'multi task ',
        'low-light ', 'low light ',
    )
    stripped = raw
    lower = raw.lower()
    for prefix in _DOMAIN_PREFIXES:
        if lower.startswith(prefix):
            stripped = raw[len(prefix):].strip()
            break
    # Title-case the rebuilt task: "image segmentation" → "Image Segmentation".
    # PWC's source casing varies (some use Title Case, some lowercase
    # after the prefix); normalise so the /explore tree groups
    # "Image Segmentation" rows from different sources together.
    if stripped:
        stripped = ' '.join(
            w.capitalize() if w.islower() else w
            for w in stripped.split()
        )
    else:
        stripped = raw
    area = 'Other'
    blob = stripped.lower()
    for label, patterns in _PWC_AREA_RULES:
        if any(re.search(p, blob) for p in patterns):
            area = label
            break
    return f"{area}/{stripped}"


def _metric_domain(metric):
    """Return the bucket label for one GlobalMetric. Falls through to
    'Other' when no rule matches — that bucket is the catchall and
    intentionally last in the sidebar render order."""
    blob = ' '.join([
        (metric.name or ''),
        (metric.description or ''),
    ]).lower()
    for label, patterns in _METRIC_DOMAIN_RULES:
        for pat in patterns:
            if re.search(pat, blob):
                return label
    return 'Other'


@app.route('/metrics')
def metrics_view():
    metrics = (
        GlobalMetric.query
        .filter(visible_in_list(GlobalMetric, getattr(g, 'current_user', None)))
        .order_by(GlobalMetric.name)
        .all()
    )
    # Group by domain heuristic. Preserve the bucket order from the
    # _METRIC_DOMAIN_RULES table so the sidebar reads in a sensible
    # order (Classification first, generic Loss/Error toward the end,
    # Other last).
    groups = {}
    for m in metrics:
        groups.setdefault(_metric_domain(m), []).append(m)
    order = [label for label, _ in _METRIC_DOMAIN_RULES] + ['Other']
    grouped_metrics = [
        (label, groups[label]) for label in order if label in groups
    ]
    # Selected metric id for the right pane (?selected=<id>). Falls
    # back to the first metric in the first non-empty group.
    selected_id = request.args.get('selected', type=int)
    selected = None
    if selected_id:
        selected = next((m for m in metrics if m.id == selected_id), None)
    if selected is None and grouped_metrics:
        selected = grouped_metrics[0][1][0]
    return render_template(
        'metrics.html',
        metrics=metrics,
        grouped_metrics=grouped_metrics,
        selected=selected,
    )

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
    """List all visualizations. Mirrors /metrics layout: domain-bucketed
    sidebar on md+, single picker on mobile, detail pane for the
    `?selected=<id>` viz (defaults to the first)."""
    visualizations = (
        GlobalVisualization.query
        .filter(visible_in_list(GlobalVisualization, getattr(g, 'current_user', None)))
        .order_by(GlobalVisualization.name)
        .all()
    )
    groups = {}
    for v in visualizations:
        groups.setdefault(_metric_domain(v), []).append(v)
    order = [label for label, _ in _METRIC_DOMAIN_RULES] + ['Other']
    grouped = [(label, groups[label]) for label in order if label in groups]
    selected_id = request.args.get('selected', type=int)
    selected = None
    if selected_id:
        selected = next((v for v in visualizations if v.id == selected_id), None)
    if selected is None and grouped:
        selected = grouped[0][1][0]
    return render_template(
        'visualizations.html',
        visualizations=visualizations,
        grouped_visualizations=grouped,
        selected=selected,
    )

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

@app.route('/leaderboard/<int:leaderboard_id>/colab_open')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_colab_open(leaderboard_id):
    """Materialize the LB's notebook as a GitHub gist (Colab-whitelisted)
    and redirect to Colab's gist importer. Falls back to the modal with
    a "couldn't open directly, download instead" flash if no GitHub
    token is configured or gist creation fails."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    user = getattr(g, 'current_user', None)
    if user and getattr(user, 'api_token', None):
        gist_url, gist_id, gist_owner = _ensure_user_colab_gist(lb, user)
    else:
        gist_url, gist_id, gist_owner = _ensure_colab_gist(lb)
    if gist_id:
        # Colab's gist URL pattern is `/gist/<owner>/<gist_id>` — the
        # bare-id form is rejected with "Unexpected GitHub Gist path".
        path = f'{gist_owner}/{gist_id}' if gist_owner else gist_id
        return redirect(f'https://colab.research.google.com/gist/{path}')
    flash(
        "Couldn't open directly in Colab — set BENCHHUB_GITHUB_GIST_TOKEN "
        "or download the notebook and upload it manually.",
        "warning",
    )
    return redirect(url_for('leaderboard_view', leaderboard_id=leaderboard_id))


@app.route('/leaderboard/<int:leaderboard_id>/colab_notebook.ipynb')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_colab_notebook(leaderboard_id):
    """Serve the per-LB Colab submission notebook. Generates on first
    hit (LLM if ANTHROPIC_API_KEY set, static fallback otherwise),
    caches on Leaderboard.colab_notebook_cache, self-invalidates when
    the LB's structure signature drifts."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    notebook_json, _source = _get_or_generate_colab_notebook(lb)
    # Inline-personalize for direct downloads too, so a logged-in user
    # who clicks "Download notebook" gets their token pre-filled.
    user = getattr(g, 'current_user', None)
    if user and getattr(user, 'api_token', None):
        notebook_json = _personalize_notebook_for_user(notebook_json, user)
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', lb.name) or f'lb_{lb.id}'
    resp = app.response_class(
        notebook_json,
        mimetype='application/x-ipynb+json',
        headers={
            'Content-Disposition': f'inline; filename="{safe_name}_submit.ipynb"',
            # Permissive CORS so Colab's URL importer can fetch it.
            'Access-Control-Allow-Origin': '*',
        },
    )
    return resp


@app.route('/leaderboard/<int:leaderboard_id>/colab_bootstrap.py')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_colab_bootstrap(leaderboard_id):
    """Standalone .py equivalent of the Colab notebook. Runs the full
    submission flow locally: pip-installs deps, fetches the dataset
    GT zip, defines my_model, builds the prediction folder, zips it,
    and optionally uploads via the user's API token. Derived by
    flattening the cached .ipynb JSON so the two artifacts stay in
    lockstep — same template version, same field schema, same
    PRED_FIELDS — without a second LLM call."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    notebook_json, _src = _get_or_generate_colab_notebook(lb)
    user = getattr(g, 'current_user', None)
    if user and getattr(user, 'api_token', None):
        notebook_json = _personalize_notebook_for_user(notebook_json, user)
    body = _ipynb_to_python(notebook_json, lb)
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', lb.name) or f'lb_{lb.id}'
    return app.response_class(
        body, mimetype='text/x-python',
        headers={
            'Content-Disposition': f'attachment; filename="{safe_name}_submit.py"',
        },
    )


# Rewriter rules:
#   "!pip install X Y"   → subprocess.check_call([sys.executable, '-m',
#                          'pip', 'install', 'X', 'Y'])
#   "!apt ..." / other ! → comment out (no portable equivalent off-Colab).
#   "from google.colab"  → comment out + print the zip path so the user
#                          still gets a clear next step.
#   files.download(...)  → comment out, replaced by a print.
# Markdown cells become `# ` line-prefixed comments. The RUN-LOCALLY
# BOOTSTRAP cell from _static_colab_notebook is dropped — it's a no-op
# when *this* script IS the local entry point.
_NB_BOOTSTRAP_MARKER = '>>> RUN-LOCALLY BOOTSTRAP <<<'


def _rewrite_jupyter_line(line):
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]
    # !pip install <args>
    m = re.match(r'!\s*pip\s+(.*)', stripped)
    if m:
        args = m.group(1).split()
        args_repr = ', '.join(repr(a) for a in args)
        return (
            f"{indent}__import__('subprocess').check_call("
            f"[__import__('sys').executable, '-m', 'pip', {args_repr}])"
        )
    # Any other shell magic — neutralize so the script still compiles.
    if stripped.startswith('!'):
        return f"{indent}# (jupyter magic disabled outside Colab) {stripped}"
    # google.colab is Colab-only; comment out, but if the same line
    # also calls files.download(<expr>), emit a substitute print of
    # the expr so the user still knows where the artifact landed.
    if re.search(r'\bgoogle\.colab\b', line) or re.search(r'\bfiles\.download\(', line):
        replacement = f"{indent}# (Colab-only, disabled outside Colab) {stripped}"
        m = re.search(r'\bfiles\.download\((.+?)\)', line)
        if m:
            replacement += (
                f"\n{indent}print('Submission ZIP written to:',"
                f" {m.group(1).strip()})"
            )
        return replacement
    return line


def _ipynb_to_python(notebook_json, lb=None):
    """Flatten an .ipynb JSON to a single runnable .py. Markdown cells
    become `# ` comments; code cells are rewritten line-by-line via
    _rewrite_jupyter_line so IPython-flavored syntax (`!pip ...`,
    `from google.colab ...`, `files.download(...)`) becomes pure
    Python that runs in a plain interpreter."""
    try:
        nb = json.loads(notebook_json)
    except Exception:
        # Defensive: if the cached blob is corrupt, fall through with
        # a 1-line stub that at least tells the user something useful.
        return (
            f"# Could not parse cached notebook for LB"
            f"{f' id={lb.id}' if lb else ''}.\n"
            f"# Re-open the LB page and try again.\n"
        )

    out_lines = []
    if lb is not None:
        safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', lb.name) or f'lb_{lb.id}'
        out_lines.append(
            f"# {lb.name} — BenchHub submission script (LB id={lb.id})"
        )
        out_lines.append(
            f"# Self-contained equivalent of the Colab notebook. Edit my_model()"
        )
        out_lines.append(
            f"# (and API_TOKEN if you want auto-upload), then: python {safe_name}_submit.py"
        )
        out_lines.append("")
    for cell in nb.get('cells', []):
        src = cell.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        if not src.strip():
            continue
        ctype = cell.get('cell_type', 'code')
        if ctype == 'markdown':
            for ln in src.splitlines() or ['']:
                out_lines.append(f"# {ln}" if ln else "#")
            out_lines.append("")
            continue
        if ctype != 'code':
            continue
        # Drop the bootstrap-the-notebook cell — redundant when this
        # script IS the local entry point.
        if _NB_BOOTSTRAP_MARKER in src:
            continue
        for ln in src.splitlines():
            rewritten = _rewrite_jupyter_line(ln)
            out_lines.append(rewritten)
        out_lines.append("")
    return '\n'.join(out_lines).rstrip() + '\n'


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

    # Defensive fallback: an LB with leaderboard_metrics but an empty
    # selected list (e.g. after a bad summary_metrics value got auto-
    # pruned, or for fresh PWC imports before any picks were saved)
    # would render no metric columns at all. Default to showing every
    # LB metric in that case so mirrored / verified rows aren't blank.
    if not selected_metrics and leaderboard_metrics_map:
        selected_metrics = sorted(leaderboard_metrics_map.keys())

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

    # Per-dataset thumbnails for the LB header (mirrors /datasets and
    # /home logic: first image-or-depth custom field on any sample).
    dataset_thumbs = {ds.id: _dataset_thumb_url(ds) for ds in leaderboard.datasets}

    pred_field_schema = _lb_submission_pred_fields(leaderboard)
    # Phase 15: split mirrored (PWC-style) submissions into a separate
    # render bucket. They share the Submission table but show in a
    # second "Reported scores" section so the trust gradient stays
    # visible — verified beats mirrored is a real signal.
    verified_submissions = [s for s in submissions
                            if (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored']
    mirrored_submissions = [s for s in submissions
                            if (getattr(s, 'kind', 'verified') or 'verified') == 'mirrored']
    lb_explorable = leaderboard.id in _compute_explorable_lb_ids([leaderboard.id])
    return render_template('leaderboard.html',
                           leaderboard=leaderboard,
                           lb_explorable=lb_explorable,
                           dataset_thumbs=dataset_thumbs,
                           pred_field_schema=pred_field_schema,
                           submissions=verified_submissions,
                           mirrored_submissions=mirrored_submissions,
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

@app.route('/import_from_hf/gated', methods=['GET'])
@login_required
def import_from_hf_gated_wizard():
    """Step-by-step gated-dataset unlock page. Reached from any HF
    import path that hit a 401 / 'gated' error. The user opens the
    dataset's HF page (new tab) → accepts terms → opens HF token
    settings (new tab) → pastes the token here → resubmits."""
    repo_id = (request.args.get('repo_id') or '').strip()
    dataset_name = (request.args.get('dataset_name') or '').strip()
    revision = (request.args.get('revision') or '').strip()
    sample_cap = request.args.get('sample_cap', type=int) or 200
    flow = request.args.get('flow', 'auto')  # 'auto' or 'direct'
    if not repo_id:
        return redirect(url_for('datasets_list'))
    return render_template(
        'hf_gated_wizard.html',
        repo_id=repo_id,
        dataset_name=dataset_name or repo_id.rstrip('/').split('/')[-1],
        revision=revision,
        sample_cap=sample_cap,
        flow=flow,
    )


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


def _resolve_hf_token(form_token):
    """Return the HF token to use for this request: prefer the form
    field (operator just typed it in for this attempt), fall back to
    the saved user.hf_token. Also persist a non-empty form value so
    future imports skip the password input."""
    form_token = (form_token or '').strip() or None
    user = getattr(g, 'current_user', None)
    if form_token:
        if user is not None and user.hf_token != form_token:
            user.hf_token = form_token
            db.session.commit()
        return form_token
    if user is not None and user.hf_token:
        return user.hf_token
    return None


def _hf_fetch_features(repo_id, revision=None, hf_token=None):
    """Pull the `features` JSON from huggingface.co/api/datasets/<repo>.
    Returns a dict {column_name: feature_descriptor} or {} if unavailable.
    Anonymous when no token; uses the token for gated repos.

    When the REST API blob doesn't carry `cardData.dataset_info`
    (common for community datasets uploaded as raw image folders or
    backed by a script rather than parquet), fall back to actually
    streaming the dataset via the `datasets` library and reading
    `ds.features`. That works for any repo `datasets.load_dataset`
    can open, which is a much wider set than the ones HF chooses to
    surface in dataset_info.
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
    # API didn't expose features → ask the datasets lib to introspect.
    return _hf_features_via_streaming(repo_id, revision=revision, hf_token=hf_token)


def _hf_features_via_streaming(repo_id, revision=None, hf_token=None):
    """Fallback path for repos with no dataset_info in the API response:
    open the dataset in streaming mode (no full download) and read its
    `.features` directly. Returns the same normalized dict shape as
    `_normalize_features`. {} on any failure so the caller falls
    through cleanly."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {}
    # Try a few common splits; many repos only ship one of these.
    for split in ('train', 'validation', 'test', 'default'):
        try:
            # `trust_remote_code=True` is required for script-backed
            # repos (like NYU Depth V2 that ships nyu_depth_v2.py).
            # We're pinned to datasets<3.0 because 3.x removed script
            # support entirely.
            ds = load_dataset(repo_id, split=split, streaming=True,
                              revision=revision, token=hf_token,
                              trust_remote_code=True)
        except Exception:
            continue
        feats = getattr(ds, 'features', None)
        if not feats:
            # No declared schema even via streaming; try to peek a row.
            try:
                example = next(iter(ds))
            except Exception:
                continue
            return _features_from_example(example)
        return _features_from_datasets_features(feats)
    return {}


def _features_from_datasets_features(features):
    """Convert a `datasets.Features` mapping into BenchHub's normalized
    feature dict."""
    out = {}
    for name, feat in (features.items() if hasattr(features, 'items') else []):
        out[name] = _describe_feature(feat)
    return out


def _describe_feature(feat):
    """One feature object → {type: ...[, names, length]} dict."""
    cls_name = type(feat).__name__
    if cls_name == 'Image':
        return {'type': 'Image'}
    if cls_name == 'Audio':
        return {'type': 'Audio'}
    if cls_name == 'ClassLabel':
        return {'type': 'ClassLabel',
                'names': list(getattr(feat, 'names', []) or [])}
    if cls_name == 'Value':
        return {'type': f"Value:{getattr(feat, 'dtype', 'unknown')}"}
    if cls_name in ('Sequence', 'List'):
        inner = getattr(feat, 'feature', None)
        inner_desc = _describe_feature(inner) if inner is not None else {}
        inner_type = inner_desc.get('type', 'unknown')
        # `Sequence:<inner_dtype>` (e.g. 'int32') matches the API shape.
        if inner_type.startswith('Value:'):
            inner_type = inner_type[len('Value:'):]
        return {'type': f"Sequence:{inner_type}",
                'length': getattr(feat, 'length', -1)}
    return {'type': 'unknown'}


def _features_from_example(example):
    """Last-resort schema inference: peek one row and guess the type
    from the Python type of each value. Used only when streaming
    couldn't surface a declared schema."""
    out = {}
    for name, value in (example.items() if isinstance(example, dict) else []):
        if hasattr(value, 'mode') and hasattr(value, 'size'):  # PIL.Image
            out[name] = {'type': 'Image'}
        elif isinstance(value, bool):
            out[name] = {'type': 'Value:bool'}
        elif isinstance(value, int):
            out[name] = {'type': 'Value:int64'}
        elif isinstance(value, float):
            out[name] = {'type': 'Value:float32'}
        elif isinstance(value, str):
            out[name] = {'type': 'Value:string'}
        elif isinstance(value, (list, tuple)):
            out[name] = {'type': 'Sequence:unknown', 'length': -1}
        else:
            out[name] = {'type': 'unknown'}
    return out


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
            dtype = desc.get('dtype', 'unknown')
            # Some popular repos (ILSVRC/imagenet-1k included) ship
            # features with `_type` missing/'Value' but a non-canonical
            # dtype payload — `'image'` for image columns, a nested
            # `{'class_label': {'names': {...}}}` dict for class
            # labels. Reshape those into the proper kinds before they
            # confuse _infer_mapping.
            if dtype == 'image':
                out[name] = {'type': 'Image'}
            elif dtype == 'audio':
                out[name] = {'type': 'Audio'}
            elif isinstance(dtype, dict) and 'class_label' in dtype:
                cls = dtype.get('class_label') or {}
                names = cls.get('names')
                # `names` may be a list or a {'<idx>': '<name>'} dict.
                if isinstance(names, dict):
                    try:
                        names = [names[k] for k in sorted(names, key=int)]
                    except (KeyError, ValueError):
                        names = list(names.values())
                out[name] = {
                    'type': 'ClassLabel',
                    'names': list(names or []),
                }
            else:
                out[name] = {'type': f"Value:{dtype}"}
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


# --- Auto-tag generation for HF imports ---------------------------------
# Two sources combined: HF's own tags (after filtering out noise prefixes
# like language: and size_categories:) plus, when ANTHROPIC_API_KEY is
# set, Claude-suggested discovery tags from the dataset description.

# HF tag prefixes that are pure metadata noise for our discovery
# purposes — license, size, language, etc. don't help anyone find the
# dataset by topic.
_HF_TAG_PREFIX_DROP = (
    'language:', 'size_categories:', 'license:', 'arxiv:', 'dataset:',
    'pretty_name:', 'paperswithcode:', 'multilinguality:', 'region:',
    'source_datasets:', 'annotations_creators:', 'language_creators:',
    'extra_gated', 'configs:', 'config_names:',
)
# Prefixes whose tail we DO want to keep, sans the prefix.
_HF_TAG_PREFIX_KEEP = ('task_categories:', 'task_ids:', 'task:', 'modality:')


# The single allowed vocabulary for the primary discovery tag. Every
# auto-tagged dataset gets exactly one of these (or none if we can't
# classify confidently). Keeps `/explore` filterable instead of drowning
# in 50 near-duplicate task labels.
_PRIMARY_TASK_TAGS = (
    'depth', 'segmentation', 'classification', 'detection',
    'language', 'audio', 'generation', 'regression', 'tabular',
    'multimodal', 'tracking', 'pose', 'reconstruction',
)

# Map common HF task tags onto our primary vocabulary so the heuristic
# fallback (no API key) can still pick a sensible primary.
_HF_TASK_TO_PRIMARY = {
    'image-classification': 'classification',
    'text-classification': 'classification',
    'token-classification': 'classification',
    'audio-classification': 'classification',
    'tabular-classification': 'tabular',
    'tabular-regression': 'tabular',
    'image-segmentation': 'segmentation',
    'semantic-segmentation': 'segmentation',
    'instance-segmentation': 'segmentation',
    'panoptic-segmentation': 'segmentation',
    'object-detection': 'detection',
    'face-detection': 'detection',
    'depth-estimation': 'depth',
    'monocular-depth-estimation': 'depth',
    'translation': 'language',
    'summarization': 'language',
    'question-answering': 'language',
    'text-generation': 'language',
    'fill-mask': 'language',
    'sentiment-analysis': 'language',
    'text-to-image': 'generation',
    'image-to-image': 'generation',
    'unconditional-image-generation': 'generation',
    'image-to-text': 'multimodal',
    'visual-question-answering': 'multimodal',
    'speech-recognition': 'audio',
    'automatic-speech-recognition': 'audio',
    'audio-to-audio': 'audio',
    'pose-estimation': 'pose',
    'keypoint-detection': 'pose',
    'object-tracking': 'tracking',
    'reinforcement-learning': 'other',  # filtered later
}

# Optional second tag: a qualifier that further specializes the primary.
# Keep this list short — it's the only kebab-cased free-text tag we'll
# accept from the heuristic side. The LLM is allowed to invent its own.
_HF_QUALIFIER_VOCAB = {
    'stereo', 'monocular', 'indoor', 'outdoor', 'medical', 'satellite',
    'aerial', 'lidar', 'ct', 'mri', 'xray', 'autonomous-driving',
    'robotics', 'fine-grained', 'multi-label', 'multilingual',
    'low-light', 'underwater', 'face', 'document',
}


def _normalize_hf_tags(raw_tags):
    """Strip noise prefixes off HF's raw tag list and return a flat
    lowercased list. Used as INPUT to _heuristic_primary_tag — not as
    the dataset's final tag set. (Final tags live behind
    _auto_tags_for_hf and are capped at 1 primary + 1 qualifier.)"""
    out = []
    seen = set()
    for tag in raw_tags or []:
        if not isinstance(tag, str):
            continue
        t = tag.strip().lower()
        if not t:
            continue
        if t.startswith(_HF_TAG_PREFIX_DROP):
            continue
        for keep in _HF_TAG_PREFIX_KEEP:
            if t.startswith(keep):
                t = t[len(keep):]
                break
        t = t.strip().strip(':')
        if len(t) < 2:
            continue
        if t in {'image', 'text', 'audio', 'tabular', 'multimodal',
                 'crowdsourced', 'expert-generated', 'machine-generated',
                 'found', 'other', 'mit', 'apache-2.0', 'cc-by-4.0', 'unknown'}:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _heuristic_primary_tag(normalized_hf_tags):
    """First HF task tag that maps onto our primary vocabulary, or None."""
    for t in normalized_hf_tags:
        primary = _HF_TASK_TO_PRIMARY.get(t)
        if primary and primary in _PRIMARY_TASK_TAGS:
            return primary
        # Sometimes the tag is bare (e.g. 'depth-estimation' without
        # the `task:` prefix); the same lookup still applies.
    return None


def _heuristic_qualifier_tag(normalized_hf_tags, primary):
    """First qualifier-vocab tag that's compatible with the primary,
    or None. Stereo + classification doesn't pair, so we don't bother
    cross-checking — qualifiers are intentionally sparse."""
    for t in normalized_hf_tags:
        if t in _HF_QUALIFIER_VOCAB and t != primary:
            return t
    return None


def _hf_fetch_repo_metadata(repo_id, revision=None, hf_token=None):
    """Fetch repo-level metadata: raw HF tags, description, license. Used
    to feed the auto-tag generator. Returns {} on any error so the import
    keeps working even when this side feature is degraded."""
    import requests as _r
    headers = {}
    if hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    if revision:
        url += f"?revision={revision}"
    try:
        resp = _r.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        info = resp.json()
        return {
            'tags': info.get('tags') or [],
            'description': (info.get('description') or '')[:2000],
            'card_data': info.get('cardData') or {},
        }
    except Exception as e:
        print(f"_hf_fetch_repo_metadata failed: {e}")
        return {}


def _llm_suggest_tags(repo_id, hf_tags, description):
    """Ask Claude for the dataset's primary task tag and at most ONE
    qualifier. Returns [primary] or [primary, qualifier], or [] when
    the API key isn't set or the call fails. Keeps the discovery tag
    set tiny on purpose — see _auto_tags_for_hf."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return []
    try:
        import requests as _r
        primary_list = ', '.join(_PRIMARY_TASK_TAGS)
        system_prompt = (
            "You assign minimal discovery tags to a HuggingFace dataset "
            "for a benchmarking platform. Tags must be SHORT and "
            "INFORMATIVE so /explore stays scannable.\n\n"
            "Output rule: a JSON array with EXACTLY 1 OR 2 strings — "
            "first the primary task tag, then (optional) a qualifier.\n\n"
            f"PRIMARY TAG MUST be one of: {primary_list}.\n"
            "Pick the closest match; never invent a new primary.\n\n"
            "QUALIFIER (optional 2nd tag) is a SINGLE WORD or kebab-case "
            "modifier that further specializes the primary. Examples:\n"
            "- depth + stereo (stereo-depth dataset)\n"
            "- depth + monocular\n"
            "- depth + indoor\n"
            "- segmentation + medical / semantic / instance / panoptic\n"
            "- classification + medical / fine-grained / multi-label\n"
            "- detection + autonomous-driving / face / aerial\n"
            "- language + qa / summarization / sentiment / translation\n"
            "- generation + text-to-image / image-to-image\n"
            "- audio + speech / music\n\n"
            "If you cannot classify confidently, return []. NEVER return "
            "vague or generic tags. NEVER return more than 2 tags. "
            "Return ONLY the JSON array, no prose."
        )
        msg = (
            f"Repo: {repo_id}\n"
            f"HF tags: {', '.join((hf_tags or [])[:25])}\n"
            f"Description:\n{(description or '')[:1500]}"
        )
        resp = _r.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "system": [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": msg}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        text = ''.join(
            block.get('text', '')
            for block in body.get('content', [])
            if block.get('type') == 'text'
        ).strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)
        if not isinstance(parsed, list) or not parsed:
            return []
        # Validate shape: first entry must be in the primary vocab.
        primary = str(parsed[0]).strip().lower().replace(' ', '-')
        if primary not in _PRIMARY_TASK_TAGS:
            return []
        out = [primary]
        if len(parsed) >= 2 and isinstance(parsed[1], str):
            qualifier = parsed[1].strip().lower().replace(' ', '-')
            if 2 <= len(qualifier) <= 40 and qualifier != primary:
                out.append(qualifier)
        return out
    except Exception as e:
        print(f"_llm_suggest_tags failed: {e}")
        return []


def _auto_tags_for_hf(repo_id, hf_token=None, revision=None):
    """Pick at most 2 discovery tags for the dataset: a primary task
    category from `_PRIMARY_TASK_TAGS` plus an optional qualifier.

    LLM-first when ANTHROPIC_API_KEY is set; otherwise the heuristic
    HF-task-tag → primary mapping. Returns [] when neither source can
    classify the dataset — caller leaves it untagged for the user to
    fill in.

    Anti-bloat: never returns more than 2 tags. The previous union-of-
    everything behavior produced noise that made /explore unreadable.
    """
    meta = _hf_fetch_repo_metadata(repo_id, revision=revision, hf_token=hf_token)
    raw = meta.get('tags', []) or []
    description = meta.get('description', '')

    llm_tags = _llm_suggest_tags(repo_id, raw, description)
    if llm_tags:
        return llm_tags[:2]

    # Heuristic fallback (no API key, or LLM returned []).
    normalized = _normalize_hf_tags(raw)
    primary = _heuristic_primary_tag(normalized)
    if not primary:
        return []
    qualifier = _heuristic_qualifier_tag(normalized, primary)
    return [primary, qualifier] if qualifier else [primary]


# --- Colab submission notebook generation (per-LB, LLM-cached) ---------
# Click "Submit via Colab" on a leaderboard → user gets a notebook
# pre-filled with that LB's sample schema and the loop that builds a
# submission ZIP. Cached on Leaderboard.colab_notebook_cache and
# self-invalidated via a structure signature, so changing the LB's
# datasets / metrics triggers a re-generation on the next request.

_COLAB_TEMPLATE_VERSION = 'v8-upload-submission-zip-fieldname'


def _lb_structure_signature(lb):
    """Stable hash of the LB's externally-visible structure: which
    datasets, which custom-field names on the GT samples, which
    metrics. If any of these changes, the cached notebook is stale.
    Returns a short hex string. The template version is folded in so
    that template-side improvements force a regeneration even when the
    LB's structure itself is unchanged."""
    import zlib as _zlib
    parts = {
        'template': _COLAB_TEMPLATE_VERSION,
        'dataset_ids': sorted(d.id for d in lb.datasets),
        'metrics': sorted([
            (lm.global_metric_id, lm.target_name)
            for lm in (lb.leaderboard_metrics or [])
        ]),
    }
    # First-dataset sample-field schema (cheap proxy for the submission
    # shape). Skip if the LB has no samples yet.
    if lb.datasets:
        sample_ids = [s.id for s in lb.datasets[0].samples[:1]]
        if sample_ids:
            parts['fields'] = sorted({
                (cf.name, cf.field_type)
                for cf in CustomField.query.filter(
                    CustomField.sample_id.in_(sample_ids)
                ).all()
            })
    return f"{_zlib.crc32(json.dumps(parts, sort_keys=True, default=str).encode()):08x}"


def _static_colab_notebook(lb):
    """Fallback notebook used when no ANTHROPIC_API_KEY is set or the
    LLM call fails. Generic enough to work for any LB shape — the user
    edits the model placeholder + the per-sample prediction format.

    Two dataset-source shapes:
      - BH-managed: `lb.datasets[0]` exists; fetch via /dataset/<id>/download.
      - HF-attached (PWC imports, /hf/... LBs): no BH Dataset row;
        stream rows via `datasets.load_dataset(repo, split=...)`. Sample
        names follow BenchHub's `_VirtualSample` convention (`s_000000`,
        `s_000001`, ...) so the submission upload path resolves correctly.
    """
    ds = lb.datasets[0] if lb.datasets else None
    ds_name = ds.name if ds else '<dataset>'
    base_url = os.environ.get('BENCHHUB_BASE_URL', 'https://benchhub.fly.dev').rstrip('/')
    # Detect HF attachment (mirrors the LLM path's discovery in _llm_colab_notebook).
    hf_attachment = None
    for att in (lb.attachments or []):
        if (getattr(att, 'kind', None) == 'hf'
                and att.role == 'primary' and att.hf_repo_id):
            hf_attachment = att
            break
    if hf_attachment:
        ds_name = f"{hf_attachment.hf_repo_id} (HF)"
    sample_field_summary = []
    if ds and ds.samples:
        first = ds.samples[0]
        for cf in (first.custom_fields or [])[:8]:
            sample_field_summary.append(f"- `{cf.name}` ({cf.field_type})")
    elif hf_attachment:
        try:
            hf_mapping = json.loads(hf_attachment.hf_mapping_json or '[]')
        except (TypeError, ValueError):
            hf_mapping = []
        for m in hf_mapping[:8]:
            col = m.get('column') or '?'
            kind = m.get('target_kind') or '?'
            sample_field_summary.append(f"- `{col}` ({kind})")
    field_block = '\n'.join(sample_field_summary) or '- *(no custom fields detected)*'
    metric_names = [
        lm.target_name or (lm.global_metric.name if lm.global_metric else '?')
        for lm in (lb.leaderboard_metrics or [])
    ]
    metric_block = ', '.join(metric_names) or '_(none configured yet)_'

    pred_fields = _lb_submission_pred_fields(lb)
    if pred_fields:
        pred_block_lines = [
            f"- `{p['name']}/<sample>.txt` &mdash; predicted value for GT "
            f"`{p['gt_field']}`."
            for p in pred_fields
        ]
        pred_block = '\n'.join(pred_block_lines)
    else:
        pred_block = (
            "- _(this leaderboard has no auto-detected prediction fields; "
            "fill `<name>/<sample>.txt` per the LB's metric "
            "definitions)_"
        )

    nb = {
        "cells": [
            {
                "cell_type": "markdown", "metadata": {},
                "source": [
                    f"# Submit to **{lb.name}** — BenchHub\n",
                    "\n",
                    f"Dataset: `{ds_name}` &middot; Metrics: {metric_block}\n",
                    "\n",
                    "**Required submission folders** (one .txt per sample):\n",
                    pred_block + "\n",
                    "\n",
                    "**Runtime choice (Colab):** this notebook runs on CPU by default. "
                    "If your model needs a GPU/TPU, switch via "
                    "*Runtime → Change runtime type* before running the cells below.\n",
                    "\n",
                    "**Want to run locally instead?** The next cell is a self-"
                    "contained Python bootstrap that fetches this notebook + "
                    "the dataset onto your own machine. Copy it, save as "
                    "`bootstrap.py`, and run `python bootstrap.py` outside Colab.\n",
                    "\n",
                    "Workflow:\n",
                    "1. Run the cells below, replacing `my_model(sample)` with your inference code.\n",
                    "2. The notebook builds a submission ZIP in BenchHub's expected folder layout.\n",
                    "3. Either download it (last cell) or upload it via the API token in the cell after.\n",
                    "\n",
                    "Sample schema (from the first sample):\n",
                    field_block + "\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": (
                    [
                        "# >>> RUN-LOCALLY BOOTSTRAP <<<\n",
                        "# This cell is a no-op on Colab. The downloadable .py\n",
                        "# version (LB page → Download submit script) already is\n",
                        "# the local entry point and skips this cell entirely.\n",
                        "import os, subprocess, sys, urllib.request\n",
                        f"BENCHHUB = '{base_url}'\n",
                        f"LEADERBOARD_ID = {lb.id}\n",
                        "if 'google.colab' not in sys.modules:\n",
                        "    nb_url = f'{BENCHHUB}/leaderboard/{LEADERBOARD_ID}/colab_notebook.ipynb'\n",
                        "    urllib.request.urlretrieve(nb_url, 'submit.ipynb')\n",
                        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n",
                        "                           '-q', 'requests', 'numpy', 'pillow', 'datasets'])\n",
                        "    print('Saved submit.ipynb. Open in Jupyter / VS Code, '\n",
                        "          'then run the remaining cells.')\n",
                        "else:\n",
                        "    print('On Colab — skipping local bootstrap.')\n",
                    ] if hf_attachment else [
                        "# >>> RUN-LOCALLY BOOTSTRAP <<<\n",
                        "# This cell is a no-op on Colab (Colab already has the\n",
                        "# notebook + can install pip packages). Copy it into a\n",
                        "# `bootstrap.py` on your machine to set up a local run.\n",
                        "import os, subprocess, sys, urllib.request\n",
                        f"BENCHHUB = '{base_url}'\n",
                        f"LEADERBOARD_ID = {lb.id}\n",
                        f"DATASET_ID = {ds.id if ds else 'None'}\n",
                        "if 'google.colab' not in sys.modules:\n",
                        "    # Local run: fetch the notebook + dataset GT zip, install deps.\n",
                        "    nb_url = f'{BENCHHUB}/leaderboard/{LEADERBOARD_ID}/colab_notebook.ipynb'\n",
                        "    urllib.request.urlretrieve(nb_url, 'submit.ipynb')\n",
                        "    if DATASET_ID is not None:\n",
                        "        urllib.request.urlretrieve(\n",
                        "            f'{BENCHHUB}/dataset/{DATASET_ID}/download', 'gt.zip')\n",
                        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n",
                        "                           '-q', 'requests', 'numpy', 'pillow'])\n",
                        "    print('Saved submit.ipynb + gt.zip. Open submit.ipynb in '\n",
                        "          'Jupyter or VS Code, then run the remaining cells.')\n",
                        "else:\n",
                        "    print('On Colab — skipping local bootstrap.')\n",
                    ]
                ),
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": (
                    [
                        "!pip -q install requests numpy pillow datasets\n",
                        "import os, io, json, zipfile, itertools, requests, numpy as np\n",
                        "from PIL import Image\n",
                        f"BENCHHUB = '{base_url}'\n",
                        f"LEADERBOARD_ID = {lb.id}\n",
                        f"DATASET_NAME = '{ds_name}'\n",
                    ] if hf_attachment else [
                        "!pip -q install requests numpy pillow\n",
                        "import os, io, json, zipfile, requests, numpy as np\n",
                        "from PIL import Image\n",
                        f"BENCHHUB = '{base_url}'\n",
                        f"LEADERBOARD_ID = {lb.id}\n",
                        f"DATASET_NAME = '{ds_name}'\n",
                    ]
                ),
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": (
                    [
                        "# 1. Stream samples directly from the HuggingFace dataset.\n",
                        "#    HF-attached LBs have no BenchHub ZIP — the GT lives on HF.\n",
                        "from datasets import load_dataset\n",
                        "CAP = 200  # bump if your model is fast or you have GPU runtime\n",
                        f"ds_stream = load_dataset({hf_attachment.hf_repo_id!r},\n",
                        f"                        split={(hf_attachment.hf_split or 'train')!r},\n",
                        "                        streaming=True, trust_remote_code=True)\n",
                        "HF_ROWS = list(itertools.islice(ds_stream, CAP))\n",
                        f"print('Loaded', len(HF_ROWS), 'rows from "
                        f"{hf_attachment.hf_repo_id} "
                        f"(split={hf_attachment.hf_split or 'train'})')\n",
                    ] if hf_attachment else [
                        "# 1. Fetch the dataset ZIP from BenchHub so we know what samples to predict on.\n",
                        "ds_zip = requests.get(f'{BENCHHUB}/dataset/" + (str(ds.id) if ds else "<id>") + "/download', timeout=120).content\n",
                        "open('/tmp/gt.zip', 'wb').write(ds_zip)\n",
                        "import zipfile as zf\n",
                        "with zf.ZipFile('/tmp/gt.zip') as z:\n",
                        "    z.extractall('/tmp/gt')\n",
                        "print('Extracted to /tmp/gt/'); print(os.listdir('/tmp/gt'))\n",
                    ]
                ),
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": [
                    "# 2. >>> EDIT THIS <<< Plug your model in here.\n",
                    "# Each field's `kind` tells the writer below how to serialize the\n",
                    "# prediction: 'scalar' → .txt, 'image'/'mask' → .png, 'depth' → .npz.\n",
                    f"PRED_FIELDS = {json.dumps(pred_fields)}\n",
                    "def my_model(sample_name, sample_inputs):\n",
                    "    \"\"\"sample_inputs is a dict of {field_name: numpy_array_or_PIL_image}.\n",
                    "    Return a dict keyed by PRED_FIELDS[*]['name']; values match the\n",
                    "    field kind: scalar ⇒ float, image/mask ⇒ HxWx3 numpy or PIL.Image,\n",
                    "    depth ⇒ HxW numpy float array.\n",
                    "    \"\"\"\n",
                    "    out = {}\n",
                    "    for f in PRED_FIELDS:\n",
                    "        if f['kind'] == 'scalar':\n",
                    "            out[f['name']] = 0\n",
                    "        elif f['kind'] in ('image', 'mask'):\n",
                    "            out[f['name']] = np.zeros((4, 4, 3), dtype=np.uint8)  # placeholder\n",
                    "        elif f['kind'] == 'depth':\n",
                    "            out[f['name']] = np.zeros((4, 4), dtype=np.float32)  # placeholder\n",
                    "    return out\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": (
                    [
                        "# 3. Iterate the HF dataset rows, run predictions, build a submission folder.\n",
                        "#    Sample names follow BenchHub's _VirtualSample convention\n",
                        "#    (s_000000, s_000001, ...) so the upload path resolves.\n",
                        "from pathlib import Path\n",
                        "import shutil\n",
                        "OUT = Path('/tmp/submission')\n",
                        "if OUT.exists(): shutil.rmtree(OUT)\n",
                        "OUT.mkdir(parents=True)\n",
                        "PRED_KIND = {f['name']: f['kind'] for f in PRED_FIELDS}\n",
                        "from PIL import Image\n",
                        "for i, row in enumerate(HF_ROWS):\n",
                        "    name = f's_{i:06d}'\n",
                        "    # row is a dict of {column_name: value}. Pass it as-is; your\n",
                        "    # my_model() can pick out the inputs it needs.\n",
                        "    preds = my_model(name, dict(row))\n",
                    ] if hf_attachment else [
                        "# 3. Walk the GT folder, run predictions, build a submission folder in BenchHub layout.\n",
                        "from pathlib import Path\n",
                        "import shutil\n",
                        "GT = Path('/tmp/gt')\n",
                        "OUT = Path('/tmp/submission')\n",
                        "if OUT.exists(): shutil.rmtree(OUT)\n",
                        "OUT.mkdir(parents=True)\n",
                        "# Discover sample names from the first metric_*/ or image_*/ folder.\n",
                        "sample_names = set()\n",
                        "for sub in GT.iterdir():\n",
                        "    if sub.is_dir() and sub.name not in {'__MACOSX'}:\n",
                        "        for f in sub.iterdir():\n",
                        "            sample_names.add(f.stem.split('_')[0])  # strip W xH suffix from raw_/\n",
                        "sample_names = sorted(sample_names)\n",
                        "print(f'Found {len(sample_names)} samples')\n",
                        "\n",
                        "PRED_KIND = {f['name']: f['kind'] for f in PRED_FIELDS}\n",
                        "from PIL import Image\n",
                        "for name in sample_names:\n",
                        "    # Build per-sample inputs by reading whatever GT files match.\n",
                        "    inputs = {}  # populate with reads as needed\n",
                        "    preds = my_model(name, inputs)\n",
                    ]
                ) + [
                    "    # Each predicted field becomes its own folder. The kind decides\n",
                    "    # the file extension so the BenchHub engine loads it correctly:\n",
                    "    #   scalar → <sample>.txt   (float)\n",
                    "    #   image  → <sample>.png   (HxWx3 RGB)\n",
                    "    #   mask   → <sample>.png   (HxW class IDs OR HxWx3 RGB)\n",
                    "    #   depth  → <sample>.npz   (HxW float, key 'depth')\n",
                    "    for field, value in preds.items():\n",
                    "        out_dir = OUT / field\n",
                    "        out_dir.mkdir(exist_ok=True)\n",
                    "        kind = PRED_KIND.get(field, 'scalar')\n",
                    "        if kind in ('image', 'mask'):\n",
                    "            arr = value if isinstance(value, np.ndarray) else np.asarray(value)\n",
                    "            if arr.ndim == 2:\n",
                    "                arr = np.stack([arr, arr, arr], axis=-1)\n",
                    "            arr_u8 = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr\n",
                    "            Image.fromarray(arr_u8).save(out_dir / f'{name}.png')\n",
                    "        elif kind == 'depth':\n",
                    "            arr = np.asarray(value, dtype=np.float32)\n",
                    "            np.savez(out_dir / f'{name}.npz', depth=arr)\n",
                    "        else:\n",
                    "            (out_dir / f'{name}.txt').write_text(str(value))\n",
                    "print('Submission folder built at', OUT)\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": [
                    "# 4a. ZIP and download to your Drive / local.\n",
                    "import shutil as sh, zipfile as zf\n",
                    "zip_path = sh.make_archive('/tmp/submission', 'zip', '/tmp/submission')\n",
                    "from google.colab import files; files.download(zip_path)\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": [
                    "# 4b. (Optional) upload directly to BenchHub via API token.\n",
                    "# When you're signed in, BenchHub auto-fills your token below.\n",
                    "# Otherwise, generate one at /settings/api_tokens and paste it here.\n",
                    "API_TOKEN = ''  # auto-filled per-user; manually paste otherwise\n",
                    "SUBMISSION_NAME = 'my_first_submission'\n",
                    "# BenchHub also stores this URL on the submission so reviewers\n",
                    "# can re-open the exact notebook that produced the predictions.\n",
                    "SOURCE_COLAB_URL = ''  # auto-filled per-user; safe to leave blank\n",
                    "if API_TOKEN:\n",
                    "    with open(zip_path, 'rb') as fh:\n",
                    "        r = requests.post(\n",
                    f"            f'{{BENCHHUB}}/api/leaderboard/{lb.id}/submission/upload',\n",
                    "            headers={'Authorization': f'Bearer {API_TOKEN}'},\n",
                    "            data={'submission_name': SUBMISSION_NAME,\n",
                    "                  'source_colab_url': SOURCE_COLAB_URL},\n",
                    "            files={'submission_zip': ('submission.zip', fh)},\n",
                    "            timeout=300,\n",
                    "        )\n",
                    "    print(r.status_code, r.text)\n",
                    "else:\n",
                    "    print('Set API_TOKEN above to upload directly.')\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    return json.dumps(nb)


def _hf_task_for_lb(lb):
    """Best-effort guess at the HF Hub task slug for an LB so the
    SOTA picker can filter to relevant models. Reads the LB's first
    metric/task name and maps to the canonical HF task taxonomy.
    Returns None when nothing maps cleanly — the admin then sees an
    unfiltered top-downloads list."""
    if not lb:
        return None
    # PWC-style LBs carry the original task name on the LeaderboardMetric
    # via target_name like "top-1 accuracy". Easier signal: the LB name
    # itself ("Image Classification on ImageNet") if it came from PWC.
    blob_parts = [lb.name or '']
    for lm in (lb.leaderboard_metrics or [])[:6]:
        blob_parts.append(lm.target_name or '')
        if lm.global_metric:
            blob_parts.append(lm.global_metric.name or '')
    blob = ' '.join(blob_parts).lower()
    # Order matters: 'segmentation' before 'classification' so the LB
    # isn't mislabeled when both words appear.
    rules = [
        ('object detection', 'object-detection'),
        ('semantic segmentation', 'image-segmentation'),
        ('instance segmentation', 'image-segmentation'),
        ('image segmentation', 'image-segmentation'),
        ('depth estimation', 'depth-estimation'),
        ('image generation', 'text-to-image'),
        ('image classification', 'image-classification'),
        ('classification', 'image-classification'),
        ('question answering', 'question-answering'),
        ('summarization', 'summarization'),
        ('translation', 'translation'),
        ('sentiment', 'text-classification'),
        ('text classification', 'text-classification'),
        ('language modeling', 'text-generation'),
        ('language modelling', 'text-generation'),
        ('generation', 'text-generation'),
    ]
    for needle, task in rules:
        if needle in blob:
            return task
    return None


def _hf_dataset_filter_variants(repo_id):
    """Build a fall-back ladder of `dataset:` tag values to try when
    filtering HF models. The HF tag system is inconsistent: a model
    trained on `ILSVRC/imagenet-1k` might be tagged with any of
    `dataset:ILSVRC/imagenet-1k`, `dataset:imagenet-1k`, `dataset:imagenet`.
    We walk progressively looser variants and stop at the first that
    returns hits."""
    if not repo_id:
        return []
    variants = []
    # Most specific first.
    variants.append(repo_id)
    if '/' in repo_id:
        bare = repo_id.split('/', 1)[1]
        variants.append(bare)
    # Strip a trailing version-y suffix (-1k, -v2, etc.) for the
    # broadest catch.
    short = variants[-1]
    short = re.sub(r'[-_](?:\d+k?|v\d+|train|test|val|small|tiny)$', '', short, flags=re.IGNORECASE)
    if short and short != variants[-1]:
        variants.append(short)
    # Dedupe while preserving order.
    seen = set()
    out = []
    for v in variants:
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _hf_trending_sota_models(task_slug, *, limit=15, dataset_repo=None):
    """List HF Hub models for one task slug, sorted by download count.
    When `dataset_repo` is given, narrow first to models tagged with
    `dataset:<repo>`; if that returns empty, walk progressively looser
    dataset name variants (e.g. ILSVRC/imagenet-1k → imagenet-1k →
    imagenet) before falling back to task-only. The fallback level
    is recorded in the cache value so the picker UI can label what
    filter actually matched.

    Returns a dict {`models`: [...], `filter_level`: str, `tried`: [...]}.
    `filter_level` is one of:
      'task+dataset:<variant>' — filtered narrowly, hits found.
      'task'                   — dataset filter empty at every level, fell through.
      'unfiltered'             — task slug missing entirely.
    """
    cache_key = ('sota_models', task_slug or '', dataset_repo or '')
    import time
    now = time.time()
    cached = _HF_SOTA_CACHE.get(cache_key)
    if cached and now - cached[0] < 600:
        return cached[1]

    def _one_call(filters):
        """Single API call with `filters` (list of tag strings or a
        single string). `sort='downloads'` gives descending order by
        default in this huggingface_hub version (no `direction=` kwarg
        — it raises TypeError; previously the silent empty result was
        masking that)."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            kwargs = {'sort': 'downloads', 'limit': limit}
            if filters:
                kwargs['filter'] = filters
            return list(api.list_models(**kwargs))
        except Exception as e:
            print(f"_hf_trending_sota_models call {filters!r} failed: {e}")
            return []

    tried = []
    models = []
    filter_level = 'unfiltered'
    if not task_slug and not dataset_repo:
        models = _one_call(None)
    else:
        if task_slug and dataset_repo:
            for variant in _hf_dataset_filter_variants(dataset_repo):
                tag = f"dataset:{variant}"
                tried.append(tag)
                models = _one_call([task_slug, tag])
                if models:
                    filter_level = f'task+{tag}'
                    break
        if not models and task_slug:
            tried.append(task_slug)
            models = _one_call([task_slug])
            if models:
                filter_level = 'task'

    out_rows = []
    for m in models:
        out_rows.append({
            'id': getattr(m, 'id', None) or getattr(m, 'modelId', ''),
            'downloads': getattr(m, 'downloads', 0) or 0,
            'likes': getattr(m, 'likes', 0) or 0,
            'library_name': getattr(m, 'library_name', '') or '',
            'last_modified': str(getattr(m, 'last_modified', '') or '')[:10],
        })
    result = {
        'models': out_rows,
        'filter_level': filter_level,
        'tried': tried,
    }
    _HF_SOTA_CACHE[cache_key] = (now, result)
    return result


_HF_SOTA_CACHE = {}


def _llm_sota_colab_notebook(lb, model_id):
    """Generate a BenchHub Colab notebook tailored to a specific
    HuggingFace model. Single Claude call with the same prompt the
    generic notebook uses + an extra "load this exact model" hint —
    NOT a base-then-rewrite, which doubles the token bill and tends to
    blow past max_tokens for non-trivial notebooks. Falls back to None
    on any failure; the caller flashes + redirects with a useful
    message."""
    return _llm_colab_notebook(lb, model_id_hint=model_id)


def _llm_colab_notebook(lb, model_id_hint=None):
    """Ask Claude to write the notebook. Returns the .ipynb JSON
    string on success, None on any failure.

    `model_id_hint`: optional HF model id. When set, the user_msg
    includes "use exactly this HF model" instructions and we bump
    max_tokens because the model cell has more lines (pip install,
    processor/tokenizer load, inference loop). Single LLM call —
    avoids the base-then-rewrite shape that blew the token cap."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    base_url = os.environ.get('BENCHHUB_BASE_URL', 'https://benchhub.fly.dev').rstrip('/')
    ds = lb.datasets[0] if lb.datasets else None
    ds_id = ds.id if ds else None
    ds_name = ds.name if ds else None
    # PWC-imported (and any HF-attached) LBs have no BH Dataset row;
    # they pull samples directly from a HuggingFace dataset. Read the
    # primary HF attachment so the notebook prompt can tell Claude to
    # use `datasets.load_dataset(repo, split=...)` instead of the
    # /dataset/<id>/download URL (which 404s for these LBs).
    hf_attachment = None
    for att in (lb.attachments or []):
        if (getattr(att, 'kind', None) == 'hf'
                and att.role == 'primary' and att.hf_repo_id):
            hf_attachment = att
            break
    metric_names = [
        lm.target_name or (lm.global_metric.name if lm.global_metric else '?')
        for lm in (lb.leaderboard_metrics or [])
    ]
    sample_fields = []
    if ds and ds.samples:
        for cf in (ds.samples[0].custom_fields or [])[:12]:
            sample_fields.append({'name': cf.name, 'type': cf.field_type})
    pred_fields = _lb_submission_pred_fields(lb)

    system_prompt = (
        "You generate a Google Colab Jupyter notebook (Python 3, "
        "nbformat 4) that helps a BenchHub user submit predictions to "
        "a specific leaderboard.\n\n"
        "BenchHub folder convention (CRITICAL — the LB metrics will\n"
        "FAIL TO EVALUATE if predictions land in the wrong folder):\n"
        "- Each prediction field gets its own folder using the field\n"
        "  name from `PRED_FIELDS` verbatim. No required prefixes —\n"
        "  the engine infers type from the file extension + npz keys.\n"
        "- File naming per kind:\n"
        "  * scalar / text → `<field>/<sample>.txt`\n"
        "  * image / mask  → `<field>/<sample>.png` (HxWx3 RGB for\n"
        "    image; HxW class IDs OR HxWx3 RGB for mask)\n"
        "  * depth         → `<field>/<sample>.npz` with array key\n"
        "    `depth` (HxW float). No width×height in the filename.\n"
        "  * histogram     → `<field>/<sample>.npz` with array keys\n"
        "    `bins` and `counts`.\n\n"
        "The notebook MUST:\n"
        "- Open with a markdown cell explaining the leaderboard, its "
        "  datasets, metrics, and a placeholder for the user's model. "
        "  This cell MUST also include:\n"
        "  * a 'Required submission folders' subsection listing the "
        "    pred fields (from the user_msg) verbatim with their GT "
        "    field origin and the bare-name folder layout.\n"
        "  * a one-line note that the runtime defaults to CPU and the "
        "    user can switch to GPU/TPU via Runtime → Change runtime "
        "    type IF their model needs acceleration. Do NOT pre-set the "
        "    accelerator in metadata — let the user decide.\n"
        "  * a one-line pointer to the run-locally bootstrap cell below.\n"
        "- Include, RIGHT AFTER that markdown cell, a Python code cell "
        "  marked '>>> RUN-LOCALLY BOOTSTRAP <<<'. It detects "
        "  `'google.colab' in sys.modules` and, when run OUTSIDE Colab, "
        "  uses urllib.request to download "
        "  `<BENCHHUB>/leaderboard/<id>/colab_notebook.ipynb` + the "
        "  dataset ZIP, then pip-installs requests/numpy/pillow. On "
        "  Colab, it just prints a skip message — no-op.\n"
        "- Include a code cell that downloads the dataset ZIP from "
        "  BenchHub and extracts it to /tmp/gt/.\n"
        "- Include a clearly-marked `def my_model(sample_name, inputs)` "
        "  cell the user is meant to edit. The cell defines a "
        "  `PRED_FIELDS` list (copy verbatim from the user_msg) and the "
        "  function returns a dict whose keys exactly match `PRED_FIELDS`.\n"
        "- Include a loop that walks `preds.items()` and writes each\n"
        "  value to `<field>/<sample>.txt` under a `submission/` "
        "  directory — bare-name folders, NO `metric_` prefix.\n"
        "- ZIP the submission folder and offer both a `files.download` "
        "  flow and an optional API-token upload to the BenchHub URL "
        "  /api/leaderboard/<id>/submission/upload.\n"
        "- The upload endpoint REQUIRES the multipart form field to be"
        " named exactly `submission_zip` —"
        " `files={'submission_zip': ('submission.zip', fh, 'application/zip')}`."
        " Any other name (e.g. `file`, `zip`) returns 400"
        " `{'error': 'No submission_zip provided'}` and the submission is dropped.\n"
        "- The submission name goes in `data={'submission_name': '...'}`,"
        " NOT as a query string or path component."
        " The Authorization header is `Bearer <API_TOKEN>`.\n\n"
        "DO NOT set `metadata.accelerator` or `metadata.colab.gpuClass` "
        "— the user picks their runtime themselves.\n\n"
        "Return ONLY a JSON object — a valid .ipynb nbformat 4 notebook. "
        "Do not wrap it in markdown fences. Do not include explanatory prose."
    )
    if hf_attachment:
        # HF-attached LB (PWC import etc): the notebook streams samples
        # from the HF dataset directly. There IS no
        # /dataset/<id>/download URL for these LBs.
        try:
            hf_mapping = json.loads(hf_attachment.hf_mapping_json or '[]')
        except (TypeError, ValueError):
            hf_mapping = []
        user_msg = (
            f"BenchHub base URL: {base_url}\n"
            f"Leaderboard id: {lb.id}\n"
            f"Leaderboard name: {lb.name}\n"
            f"Dataset source: HuggingFace dataset {hf_attachment.hf_repo_id} "
            f"(split: {hf_attachment.hf_split or 'train'})\n"
            f"DATASET FETCH (use this instead of any BenchHub /dataset URL):\n"
            f"  from datasets import load_dataset\n"
            f"  ds = load_dataset({hf_attachment.hf_repo_id!r}, "
            f"split={(hf_attachment.hf_split or 'train')!r}, "
            f"streaming=True, trust_remote_code=True)\n"
            f"  Iterate up to ~200 rows for inference (cap to keep Colab "
            f"runtime reasonable). Sample names are 's_000000', 's_000001', "
            f"... matching BenchHub's _VirtualSample naming convention; the "
            f"submission folder must use those names.\n"
            f"Field mapping (column → BenchHub field): "
            f"{json.dumps(hf_mapping)}\n"
            f"Submission upload URL: {base_url}/api/leaderboard/{lb.id}/submission/upload\n"
            f"Required prediction fields (PRED_FIELDS — bare-name folders, "
            f"NOT under metric_*): {json.dumps(pred_fields)}\n"
            f"Metric output names expected: {json.dumps(metric_names)}\n"
            f"DO NOT use urllib.request to GET '{base_url}/dataset/None/download' "
            f"or any /dataset/<id>/download URL — that path doesn't exist "
            f"for HF-attached LBs and 404s. Always go through "
            f"datasets.load_dataset.\n"
        )
    else:
        # Legacy BH-dataset LB (uploaded ZIP): keep the download-ZIP path.
        user_msg = (
            f"BenchHub base URL: {base_url}\n"
            f"Leaderboard id: {lb.id}\n"
            f"Leaderboard name: {lb.name}\n"
            f"Dataset: {ds_name} (id={ds_id})\n"
            f"Dataset download URL: {base_url}/dataset/{ds_id}/download\n"
            f"Submission upload URL: {base_url}/api/leaderboard/{lb.id}/submission/upload\n"
            f"Required prediction fields (PRED_FIELDS — bare-name folders, "
            f"NOT under metric_*): {json.dumps(pred_fields)}\n"
            f"Metric output names expected: {json.dumps(metric_names)}\n"
            f"Sample-field schema (first sample): {json.dumps(sample_fields)}\n"
        )
    if model_id_hint:
        user_msg += (
            f"\n# SOTA model directive\n"
            f"Pre-load THIS HuggingFace model in the my_model cell — the "
            f"user shouldn't have to fill in the model body, just run:\n"
            f"  HF_MODEL_ID = {model_id_hint!r}\n"
            f"Hard requirements when this directive is set:\n"
            f"- Markdown cell at the top mentions {model_id_hint!r} with a "
            f"link to https://huggingface.co/{model_id_hint}.\n"
            f"- The my_model cell starts with `!pip install -q transformers "
            f"torch pillow accelerate sentencepiece` (add datasets/timm/etc only "
            f"if you actually use them). For vision models also pip-install "
            f"`einops` since many HF vision models depend on it.\n"
            f"- Loader choice — DEFAULT to transformers, fall back to timm only "
            f"when the model is unambiguously a timm release:\n"
            f"  * If HF_MODEL_ID starts with 'timm/' OR the model card explicitly "
            f"says 'timm.create_model(...)' → use `timm.create_model("
            f"f'hf_hub:{{HF_MODEL_ID}}', pretrained=True)` and `timm.data."
            f"resolve_data_config({{}}, model=...)` for the preprocessor.\n"
            f"  * Otherwise → use `transformers.AutoModel.from_pretrained("
            f"HF_MODEL_ID, trust_remote_code=True)` paired with "
            f"`AutoImageProcessor` / `AutoTokenizer` / `AutoFeatureExtractor` "
            f"as appropriate. trust_remote_code=True is REQUIRED for many "
            f"recent vision models (MambaVision, EVA, custom architectures) "
            f"that ship modeling code on the Hub. NEVER use timm for repos "
            f"that aren't under the 'timm/' org or don't document a "
            f"`timm.create_model` call — the timm hf-hub loader requires "
            f"an 'architecture' key in config.json that custom-code models "
            f"don't have, and you'll see KeyError: 'architecture'.\n"
            f"  * For LLMs / text generation: `AutoModelForCausalLM` or "
            f"`AutoModelForSeq2SeqLM` + `AutoTokenizer`.\n"
            f"- IMPORTANT processor fallback for vision models: many HF Hub "
            f"vision repos (including MambaVision, some custom classifiers) "
            f"DO NOT ship a `preprocessor_config.json`, so AutoImageProcessor "
            f"raises OSError. Wrap the processor load in try/except; on "
            f"failure, fall back to a stock ImageNet torchvision pipeline:\n"
            f"    from torchvision import transforms\n"
            f"    _IMAGENET_TFM = transforms.Compose([\n"
            f"        transforms.Resize(256),\n"
            f"        transforms.CenterCrop(224),\n"
            f"        transforms.ToTensor(),\n"
            f"        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n"
            f"    ])\n"
            f"  Then in my_model: if processor is None, use "
            f"`pixel_values = _IMAGENET_TFM(img).unsqueeze(0)` instead of "
            f"the processor call. This is the common fallback that works "
            f"for the vast majority of ImageNet-trained classifiers.\n"
            f"- Define an `_install_and_retry(load_fn, max_retries=4)` helper "
            f"that catches the load_fn() exception, extracts the missing "
            f"package name from either `No module named 'X'` (ImportError) "
            f"or `pip install X` / `Run \\`pip install X\\`` (RuntimeError "
            f"raised by transformers when modeling_*.py imports something "
            f"that isn't installed), then runs "
            f"`subprocess.check_call([sys.executable, '-m', 'pip', "
            f"'install', '-q', pkg])` and retries. After max_retries it "
            f"re-raises. This pattern handles the long tail of HF custom-"
            f"code models that depend on mamba_ssm / causal-conv1d / "
            f"flash-attn / etc. The processor + model loads BOTH go through "
            f"this helper.\n"
            f"- Wrap the overall load in a try/except. On final failure, "
            f"print the error and a one-line hint: 'If timm raised KeyError "
            f"architecture, swap to transformers.AutoModel.from_pretrained("
            f"..., trust_remote_code=True). If AutoImageProcessor raised "
            f"OSError about preprocessor_config, the fallback torchvision "
            f"pipeline kicks in automatically. If a deep dep refused to "
            f"install (compile error / GPU-only wheel), the model needs "
            f"GPU runtime — switch via Runtime → Change runtime type → "
            f"GPU and rerun.'\n"
            f"- The my_model(sample_name, inputs) function runs inference end-to-end "
            f"and returns a dict whose keys match PRED_FIELDS exactly. The default "
            f"body is RUNNABLE — the user shouldn't need to edit it for the metric "
            f"to score.\n"
            f"- Default to CPU; mention in the markdown if GPU would speed it up.\n"
        )
    try:
        import requests as _r
        resp = _r.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                # SOTA-mode notebooks have an extra pip-install + model-load
                # cell with import boilerplate, so 4k tokens regularly truncates
                # mid-output. Bump only when the directive is set; the regular
                # generic notebook fits comfortably in 4k.
                "max_tokens": 16000 if model_id_hint else 4000,
                "system": [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
            # SOTA-mode notebooks generate ~12-15k tokens of model
            # boilerplate; the regular notebook is ~3k. Scale the
            # timeout proportionally so neither hits a read-timeout
            # in the steady-state case.
            timeout=300 if model_id_hint else 60,
        )
        resp.raise_for_status()
        body = resp.json()
        text = ''.join(
            block.get('text', '')
            for block in body.get('content', [])
            if block.get('type') == 'text'
        ).strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        # Validate it parses as JSON and is shaped like a notebook.
        try:
            nb = json.loads(text)
        except json.JSONDecodeError as je:
            print(
                f"_llm_colab_notebook(model_id_hint={model_id_hint!r}) "
                f"got unparseable JSON ({je}); "
                f"first 300 chars: {text[:300]!r}"
            )
            return None
        if not isinstance(nb, dict) or 'cells' not in nb or 'nbformat' not in nb:
            print(
                f"_llm_colab_notebook(model_id_hint={model_id_hint!r}) "
                f"returned JSON without notebook shape; "
                f"keys: {list(nb.keys()) if isinstance(nb, dict) else type(nb).__name__}"
            )
            return None
        return text
    except Exception as e:
        print(
            f"_llm_colab_notebook(model_id_hint={model_id_hint!r}) "
            f"raised {type(e).__name__}: {e}"
        )
        return None


def _get_or_generate_colab_notebook(lb):
    """Returns a (notebook_json_str, source) tuple where source is 'cache',
    'llm', or 'static'. Stamps the cache on miss."""
    sig = _lb_structure_signature(lb)
    cache = lb.colab_notebook_cache
    if cache:
        try:
            wrapped = json.loads(cache)
            if wrapped.get('sig') == sig and wrapped.get('notebook'):
                return wrapped['notebook'], 'cache'
        except Exception:
            pass
    nb = _llm_colab_notebook(lb)
    source = 'llm'
    if not nb:
        nb = _static_colab_notebook(lb)
        source = 'static'
    try:
        # Preserve any existing gist_id across regenerations so we can
        # PATCH the gist instead of orphaning it.
        existing_gist = None
        if cache:
            try:
                existing_gist = (json.loads(cache) or {}).get('gist_id')
            except Exception:
                existing_gist = None
        lb.colab_notebook_cache = json.dumps({
            'sig': sig, 'notebook': nb, 'gist_id': existing_gist,
        })
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"colab notebook cache write failed: {e}")
    return nb, source


def _personalize_notebook_for_user(notebook_json, user, source_colab_url=None):
    """Substitute the empty `API_TOKEN = ''` placeholder with the
    user's actual BenchHub API token, and optionally the empty
    `SOURCE_COLAB_URL = ''` placeholder with this notebook's gist URL
    so the upload back-references itself. Safe to apply to either the
    static or LLM-generated notebook form."""
    out = notebook_json
    if user and getattr(user, 'api_token', None):
        safe_token = user.api_token.replace("\\", r"\\").replace("'", r"\'")
        out = re.sub(
            r"API_TOKEN\s*=\s*(?:''|\\\"\\\"|\"\")",
            f"API_TOKEN = '{safe_token}'",
            out, count=1,
        )
    if source_colab_url:
        safe_url = source_colab_url.replace("\\", r"\\").replace("'", r"\'")
        out = re.sub(
            r"SOURCE_COLAB_URL\s*=\s*(?:''|\\\"\\\"|\"\")",
            f"SOURCE_COLAB_URL = '{safe_url}'",
            out, count=1,
        )
    return out


def _ensure_user_colab_gist(lb, user):
    """Per-user variant of _ensure_colab_gist: pushes a personalized
    notebook (with the user's API token baked in) to a per-user gist so
    each authed user gets their own one-click Colab link.

    Falls back to the generic LB-level gist when the user is anonymous
    or has no api_token configured."""
    if not user or not getattr(user, 'api_token', None):
        return _ensure_colab_gist(lb)

    token = os.environ.get('BENCHHUB_GITHUB_GIST_TOKEN')
    if not token:
        return None, None, None

    # Reuse the LB-level generic notebook (LLM/static).
    nb_json, _src = _get_or_generate_colab_notebook(lb)
    sig = _lb_structure_signature(lb)

    record = UserColabGist.query.filter_by(
        user_id=user.id, leaderboard_id=lb.id,
    ).first()
    gist_id = record.gist_id if record else None
    gist_owner = record.gist_owner if record else None

    # When a record already exists we know the gist URL upfront and can
    # bake it into the SOURCE_COLAB_URL placeholder so the cell shows
    # the back-link verbatim. First-time creation uses the empty
    # placeholder; the API endpoint's UserColabGist fallback fills the
    # URL onto the Submission row regardless.
    known_url = None
    if gist_id:
        path = f'{gist_owner}/{gist_id}' if gist_owner else gist_id
        known_url = f'https://colab.research.google.com/gist/{path}'
    nb_json = _personalize_notebook_for_user(
        nb_json, user, source_colab_url=known_url,
    )

    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', lb.name) or f'lb_{lb.id}'
    filename = f'{safe_name}_submit.ipynb'
    description = (
        f"BenchHub submission scaffold for leaderboard '{lb.name}' "
        f"(id={lb.id}) — personalized for {user.email}"
    )
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    payload = {
        'description': description,
        'public': False,
        'files': {filename: {'content': nb_json}},
    }
    try:
        import requests as _r
        gist_resp_json = None
        if gist_id:
            resp = _r.patch(
                f'https://api.github.com/gists/{gist_id}',
                headers=headers, json=payload, timeout=15,
            )
            if resp.status_code == 404:
                gist_id = None
            else:
                resp.raise_for_status()
                gist_resp_json = resp.json()
        if not gist_id:
            resp = _r.post(
                'https://api.github.com/gists',
                headers=headers, json=payload, timeout=15,
            )
            resp.raise_for_status()
            gist_resp_json = resp.json()
            gist_id = gist_resp_json.get('id')
        if not gist_id:
            return None, None, None
        if gist_resp_json:
            new_owner = (gist_resp_json.get('owner') or {}).get('login')
            if new_owner:
                gist_owner = new_owner
        if not gist_owner and gist_resp_json:
            html_url = gist_resp_json.get('html_url', '')
            m = re.match(r'https://gist\.github\.com/([^/]+)/[0-9a-f]+', html_url)
            if m:
                gist_owner = m.group(1)
        try:
            if record:
                record.gist_id = gist_id
                record.gist_owner = gist_owner
                record.sig = sig
            else:
                db.session.add(UserColabGist(
                    user_id=user.id, leaderboard_id=lb.id,
                    gist_id=gist_id, gist_owner=gist_owner, sig=sig,
                ))
            db.session.commit()
        except Exception:
            db.session.rollback()
        return (
            f'https://gist.github.com/{gist_owner}/{gist_id}'
            if gist_owner else f'https://gist.github.com/{gist_id}',
            gist_id,
            gist_owner,
        )
    except Exception as e:
        print(f"_ensure_user_colab_gist failed: {e}")
        return None, None, None


# Heuristic for naming a GT image column as a segmentation mask
# rather than an RGB regression target. Cheap upfront check; metrics
# are still safe-to-run if the heuristic miscategorizes, just less
# semantically apt.
_MASK_NAME_HINTS = ('mask', 'seg', 'label', 'parsing', 'class')


def _gt_columns(ds):
    """Yield (col_name, kind, hints) for each evaluable GT column on
    the first sample. `kind` is one of 'scalar' / 'depth' / 'image' /
    'mask' / 'text'. `hints` carries domain-specific extras:
      - scalar: {'is_classlabel': bool}
      - text: {} (treated as classlabel-style: exact-string-match)
      - depth, image, mask: {} (reserved)
    Class-name sidecars (`<col>_class`) are skipped — they're not
    evaluable on their own."""
    if not ds or not ds.samples:
        return
    first = ds.samples[0]
    field_types = {cf.name: cf.field_type for cf in (first.custom_fields or [])}
    for cf in (first.custom_fields or []):
        if cf.name.endswith('_class'):
            continue
        if cf.field_type == 'scalar':
            is_classlabel = f"{cf.name}_class" in field_types
            yield cf.name, 'scalar', {'is_classlabel': is_classlabel}
        elif cf.field_type == 'depth':
            yield cf.name, 'depth', {}
        elif cf.field_type == 'image':
            name_lc = cf.name.lower()
            if any(h in name_lc for h in _MASK_NAME_HINTS):
                yield cf.name, 'mask', {}
            else:
                yield cf.name, 'image', {}
        elif cf.field_type == 'text':
            # Treat short text fields as classlabel-style GT (e.g.
            # 'neg' / 'pos' for sentiment, 'cat' / 'dog' / ... for
            # animals). Long captions get skipped — `len > 80` is a
            # cheap heuristic to avoid proposing exact-match against
            # free-form prose.
            val = (getattr(cf, 'value_text', None) or '').strip()
            if val and len(val) <= 80:
                yield cf.name, 'text', {}


# Backwards-compat shim — some callers still iterate scalar-only.
def _scalar_gt_columns(ds):
    for name, kind, hints in _gt_columns(ds):
        if kind == 'scalar':
            yield name, hints['is_classlabel']


# ---------------------------------------------------------------------------
# Per-kind metric proposal builders. Each returns a single dict matching
# the proposal schema (target_name, global_name, description, fallback_code,
# arg_mappings, sort_direction, pooling_type, llm_hint, pred_fields).
# ---------------------------------------------------------------------------


def _proposal_top1_classlabel(col):
    global_name = "top1"
    return {
        'target_name': f"top-1 accuracy ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample top-1 accuracy on the `{col}` ClassLabel: "
            f"1.0 when the submission's `{col}_pred` equals GT `{col}`, "
            f"0.0 otherwise. Mean-pooled."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"1.0 if predicted class index matches GT, else 0.0.\"\"\"\n"
            f"    try:\n"
            f"        return 1.0 if int(gt) == int(pred) else 0.0\n"
            f"    except (TypeError, ValueError):\n"
            f"        return 0.0\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'higher_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Per-sample top-1 accuracy: GT integer class index `{col}` "
            f"vs submission's predicted index `{col}_pred`. Function name "
            f"MUST be `{global_name}(gt, pred)`. Returns 1.0 or 0.0."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'scalar', 'gt_field': col,
            'description': f"Per-sample predicted class index for `{col}` (.txt with the integer).",
        }],
    }


def _proposal_mae_scalar(col):
    global_name = "mae"
    return {
        'target_name': f"MAE ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample mean absolute error between submission's "
            f"`{col}_pred` and GT `{col}`."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"|gt - pred| as a float.\"\"\"\n"
            f"    return float(abs(float(gt) - float(pred)))\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'lower_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Mean absolute error between GT scalar `{col}` and submission's "
            f"`{col}_pred`. Function name MUST be `{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'scalar', 'gt_field': col,
            'description': f"Per-sample predicted value for `{col}` (.txt with the number).",
        }],
    }


_TEXT_EVAL_SUITE_CACHE = {}  # keyed by (id(ds), col) → suite dict or None


def _get_or_build_text_eval_suite(ds, col, hints):
    """Memoized wrapper around `_llm_propose_text_evaluation_suite`.
    Both the metric proposer and the viz proposer iterate `_gt_columns`
    independently; without caching they'd each fire an LLM call per
    text column. Cache lives on a module-level dict keyed by id(ds)+col
    so it's per-process per-request — no persistent staleness."""
    key = (id(ds), col)
    if key in _TEXT_EVAL_SUITE_CACHE:
        return _TEXT_EVAL_SUITE_CACHE[key]
    sample_value = None
    try:
        first = ds.samples[0] if ds.samples else None
        if first is not None:
            for cf in (first.custom_fields or []):
                if cf.name == col:
                    sample_value = getattr(cf, 'value_text', None)
                    break
    except Exception:
        pass
    dataset_repo = getattr(ds, 'name', None)
    suite = _llm_propose_text_evaluation_suite(
        col, sample_value, dataset_repo=dataset_repo,
    )
    _TEXT_EVAL_SUITE_CACHE[key] = suite
    return suite


def _llm_propose_text_evaluation_suite(col, sample_value, dataset_repo=None):
    """Ask Claude to classify the text task this column represents and
    author a metric+viz suite tailored to it.

    Why: a text GT column might be 'pos'/'neg' (classification → top-1
    accuracy + F1 makes sense), or a sentence completion (BLEU / ROUGE
    / edit distance), or a span answer (SQuAD-style EM + token-F1), or
    free-form generation. Different tasks need different metrics.
    Heuristics on length/vocab can't tell these apart reliably; an LLM
    can.

    Returns: dict {task_type, metrics: [proposal_dict], visualizations:
    [proposal_dict]} or None on any failure (caller falls back to the
    static top-1+F1 pair). Each proposal_dict matches the same schema
    the static proposers produce (target_name, global_name, description,
    fallback_code, arg_mappings, sort_direction, pooling_type, llm_hint,
    pred_fields, plus is_aggregated/accepts_aggregated_inputs for viz).
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    system_prompt = (
        "You design evaluation suites for benchmarking text-output "
        "tasks. Given a single GT column from a dataset (column name + "
        "one example value), classify the task and propose 1-3 "
        "appropriate metrics plus an optional aggregated visualization.\n\n"
        "Hard requirements:\n"
        "- Output ONLY valid JSON, no fences, no commentary.\n"
        "- Schema:\n"
        "  {\n"
        "    \"task_type\": \"classification\" | \"generation\" | "
        "\"completion\" | \"qa\" | \"summarization\" | \"translation\" | \"other\",\n"
        "    \"metrics\": [{\n"
        "      \"global_name\": <snake_case Python identifier>,\n"
        "      \"target_name\": <human-readable label>,\n"
        "      \"description\": <one-sentence what-it-measures>,\n"
        "      \"sort_direction\": \"higher_is_better\" | \"lower_is_better\",\n"
        "      \"is_aggregated\": <bool>,\n"
        "      \"python_code\": <full Python source>\n"
        "    }],\n"
        "    \"visualization\": <same shape as metric, or null>\n"
        "  }\n\n"
        "Code constraints:\n"
        "- Each `python_code` must define exactly one top-level function whose name equals `global_name`.\n"
        "- Per-sample (is_aggregated=false): signature `def f(gt, pred)`, returns float.\n"
        "- Aggregated (is_aggregated=true): signature `def f(gt, pred)` where gt + pred are PARALLEL lists of strings spanning every sample of one submission. Returns a float (metric) or PIL.Image (viz).\n"
        "- Use stdlib + numpy (np already imported). Don't import packages that may be missing (no nltk, no rouge, no transformers).\n"
        "- For BLEU/ROUGE/edit-distance/etc., implement the formula from scratch.\n"
        "- gt and pred are strings; coerce defensively (str(gt).strip()).\n"
        "- Aggregated viz must return a PIL.Image (≤512x512). Use `from PIL import Image as _PILImage`.\n\n"
        "Task-type guidance:\n"
        "- classification (short class labels like 'pos'/'neg', 'cat'/'dog'): top-1 accuracy + macro F1; viz = confusion matrix. IMPORTANT: use LENIENT string matching — case-insensitive, whitespace + punctuation stripped, prefix-aware so 'pos' / 'Pos' / 'positive' / '  pos!  ' all canonicalize to the same class. Do NOT compare with raw `==` after only `.strip()`.\n"
        "- generation / summarization: corpus BLEU-4 + ROUGE-L (both aggregated).\n"
        "- completion: ROUGE-L + character-level edit-distance ratio.\n"
        "- qa: SQuAD-style exact match + token-level F1.\n"
        "- translation: corpus BLEU-4.\n"
        "- other: top-1 accuracy + a sensible second metric you justify in description.\n"
    )
    user_msg = json.dumps({
        'column_name': col,
        'sample_value': str(sample_value)[:500] if sample_value is not None else '',
        'dataset_repo': dataset_repo or '',
    })
    try:
        import requests as _r
        resp = _r.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 4000,
                "system": [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=45,
        )
        resp.raise_for_status()
        body = resp.json()
        text = ''.join(
            block.get('text', '')
            for block in body.get('content', [])
            if block.get('type') == 'text'
        ).strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)
    except Exception as e:
        print(f"_llm_propose_text_evaluation_suite failed: {e}")
        return None

    metrics_in = parsed.get('metrics') or []
    viz_in = parsed.get('visualization')
    metric_proposals = []
    for m in metrics_in:
        gn = (m.get('global_name') or '').strip()
        if not re.match(r'^[a-z_][a-z0-9_]*$', gn):
            continue
        code = (m.get('python_code') or '').strip()
        if f"def {gn}(" not in code:
            continue
        is_agg = bool(m.get('is_aggregated'))
        sd = m.get('sort_direction')
        if sd not in ('higher_is_better', 'lower_is_better'):
            sd = 'higher_is_better'
        metric_proposals.append({
            'target_name': (m.get('target_name') or gn) + f" ({col})",
            'global_name': gn,
            'description': (m.get('description') or '').strip(),
            'fallback_code': code,
            'python_code': code,
            'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
            'sort_direction': sd,
            'pooling_type': 'mean',
            'is_aggregated': is_agg,
            'accepts_aggregated_inputs': is_agg,
            'llm_hint': m.get('description') or gn,
            'pred_fields': [{
                'name': f'{col}_pred', 'kind': 'scalar', 'gt_field': col,
                'description': (
                    f"Predicted text for `{col}` "
                    f"(per-sample string, ship as `<sample>.txt`)."
                ),
            }],
        })

    viz_proposals = []
    if isinstance(viz_in, dict):
        gn = (viz_in.get('global_name') or '').strip()
        code = (viz_in.get('python_code') or '').strip()
        if (re.match(r'^[a-z_][a-z0-9_]*$', gn)
                and f"def {gn}(" in code):
            viz_proposals.append({
                'target_name': (viz_in.get('target_name') or gn) + f" ({col})",
                'global_name': gn,
                'description': (viz_in.get('description') or '').strip(),
                'fallback_code': code,
                'python_code': code,
                'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
                'is_aggregated': True,
                'accepts_aggregated_inputs': True,
                'llm_hint': viz_in.get('description') or gn,
                'pred_fields': [{
                    'name': f'{col}_pred', 'kind': 'scalar', 'gt_field': col,
                    'description': f"Per-sample predicted text for `{col}`.",
                }],
            })

    if not metric_proposals:
        # All metrics rejected by validation — fall back rather than
        # return an empty suite that would suppress static defaults.
        return None
    return {
        'task_type': parsed.get('task_type') or 'other',
        'metrics': metric_proposals,
        'visualizations': viz_proposals,
    }


def _proposal_top1_text_classlabel(col):
    """Top-1 classification accuracy for text-typed GT columns (e.g.
    sentiment 'neg'/'pos', species 'cat'/'dog', plain string class
    names that aren't ClassLabel-encoded). Mirrors `_proposal_top1_classlabel`
    for ints — lenient string match: case-insensitive + whitespace/
    punctuation normalized + prefix-aware so 'pos' / 'POS!' / 'positive'
    all count as the same class."""
    global_name = "top1_text"
    return {
        'target_name': f"top-1 accuracy ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample top-1 classification accuracy on the `{col}` "
            f"text label. Lenient string match: case-insensitive, "
            f"whitespace + punctuation normalized, prefix-aware "
            f"('pos' / 'Pos' / 'positive' / '  pos!  ' all count as "
            f"the same class)."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"Lenient top-1 accuracy on text class labels.\"\"\"\n"
            f"    import re as _re\n"
            f"    def _norm(x):\n"
            f"        s = str(x).strip().lower()\n"
            f"        s = _re.sub(r'[^\\w\\s]', '', s)\n"
            f"        s = _re.sub(r'\\s+', ' ', s).strip()\n"
            f"        return s\n"
            f"    g = _norm(gt); p = _norm(pred)\n"
            f"    if not g or not p:\n"
            f"        return 0.0\n"
            f"    if g == p:\n"
            f"        return 1.0\n"
            f"    if p.startswith(g) or g.startswith(p):\n"
            f"        return 1.0\n"
            f"    g_toks = set(g.split()); p_toks = set(p.split())\n"
            f"    if g_toks & p_toks:\n"
            f"        return 1.0\n"
            f"    if any(t.startswith(g) or g.startswith(t) for t in p_toks):\n"
            f"        return 1.0\n"
            f"    return 0.0\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'higher_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Per-sample top-1 classification accuracy for text labels "
            f"on `{col}`. Return 1.0 if str(gt).strip() == "
            f"str(pred).strip(), else 0.0. Function name MUST be "
            f"`{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'scalar', 'gt_field': col,
            'description': (
                f"Predicted class name for `{col}` "
                f"(per-sample string, ship as `<sample>.txt`)."
            ),
        }],
    }


def _proposal_macro_f1_text_classlabel(col):
    """Macro-averaged F1 across the (auto-discovered) classes seen in
    the submission. is_aggregated so the metric receives parallel
    lists of (gt, pred) strings and computes per-class precision/
    recall before averaging — useful when one class dominates and
    top-1 alone hides minority-class performance."""
    global_name = "macro_f1_text"
    return {
        'target_name': f"macro F1 ({col})",
        'global_name': global_name,
        'description': (
            f"Macro-averaged F1 on `{col}`. Computes per-class precision "
            f"and recall over the union of GT + predicted classes, then "
            f"averages F1 unweighted across classes. Robust to class "
            f"imbalance — lower than accuracy when minority classes are "
            f"misclassified."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"Macro F1 over text classes with lenient string\n"
            f"    matching (lowercase + punctuation-stripped + prefix-aware\n"
            f"    canonicalization to one of the GT class names).\"\"\"\n"
            f"    import re as _re\n"
            f"    def _norm(x):\n"
            f"        s = str(x).strip().lower()\n"
            f"        s = _re.sub(r'[^\\w\\s]', '', s)\n"
            f"        s = _re.sub(r'\\s+', ' ', s).strip()\n"
            f"        return s\n"
            f"    pairs = [(_norm(g), _norm(p))\n"
            f"             for g, p in zip(gt, pred)\n"
            f"             if g is not None and p is not None]\n"
            f"    if not pairs:\n"
            f"        return 0.0\n"
            f"    # Canonical class set = whatever appears as GT (longer\n"
            f"    # variants first so 'positive' wins over 'pos' when both\n"
            f"    # are GT classes).\n"
            f"    classes = sorted({{g for g, _ in pairs}}, key=len, reverse=True)\n"
            f"    if not classes:\n"
            f"        return 0.0\n"
            f"    def _canon(s):\n"
            f"        if s in classes:\n"
            f"            return s\n"
            f"        for c in classes:\n"
            f"            if s.startswith(c) or c.startswith(s):\n"
            f"                return c\n"
            f"        toks = set(s.split())\n"
            f"        for c in classes:\n"
            f"            if c in toks or any(t.startswith(c) or c.startswith(t) for t in toks):\n"
            f"                return c\n"
            f"        return s  # unmappable — falls into its own bucket\n"
            f"    canon_pairs = [(g, _canon(p)) for g, p in pairs]\n"
            f"    f1s = []\n"
            f"    for c in classes:\n"
            f"        tp = sum(1 for g, p in canon_pairs if g == c and p == c)\n"
            f"        fp = sum(1 for g, p in canon_pairs if g != c and p == c)\n"
            f"        fn = sum(1 for g, p in canon_pairs if g == c and p != c)\n"
            f"        if tp == 0:\n"
            f"            f1s.append(0.0); continue\n"
            f"        prec = tp / (tp + fp) if (tp + fp) else 0.0\n"
            f"        rec  = tp / (tp + fn) if (tp + fn) else 0.0\n"
            f"        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)\n"
            f"    return float(sum(f1s) / len(f1s))\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'higher_is_better',
        'pooling_type': 'mean',  # ignored — is_aggregated below
        'is_aggregated': True,
        'accepts_aggregated_inputs': True,
        'llm_hint': (
            f"Macro-averaged F1 on text class labels for `{col}`. "
            f"`gt` and `pred` are PARALLEL lists of strings spanning "
            f"every sample of one submission. Compute per-class precision "
            f"+ recall over the union of seen classes, average F1 unweighted. "
            f"Function name MUST be `{global_name}(gt, pred)`. Return a float."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'scalar', 'gt_field': col,
            'description': (
                f"Predicted class name for `{col}` "
                f"(per-sample string, ship as `<sample>.txt`)."
            ),
        }],
    }


def _viz_text_confusion_matrix(col):
    """Aggregated confusion matrix for text-class GT. String-based
    counterpart to the int-classlabel confusion matrix — auto-discovers
    classes from the (gt, pred) pairs seen in the submission."""
    global_name = "confusion_matrix_text"
    return {
        'target_name': f"confusion matrix ({col})",
        'global_name': global_name,
        'description': (
            f"Aggregated confusion matrix between GT `{col}` (text class) "
            f"and submission's `{col}_pred`. Classes auto-discovered from "
            f"the union of values seen across all samples."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"Aggregated string-class confusion matrix as a\n"
            f"    256x256 grayscale heatmap. gt + pred are parallel\n"
            f"    lists. Predictions are canonicalized to one of the GT\n"
            f"    class names via lenient matching (lowercase + strip +\n"
            f"    prefix-aware) so 'pos'/'POS!'/'positive' all collapse\n"
            f"    to the same row.\"\"\"\n"
            f"    import re as _re\n"
            f"    import numpy as _np\n"
            f"    from PIL import Image as _PILImage\n"
            f"    def _norm(x):\n"
            f"        s = str(x).strip().lower()\n"
            f"        s = _re.sub(r'[^\\w\\s]', '', s)\n"
            f"        s = _re.sub(r'\\s+', ' ', s).strip()\n"
            f"        return s\n"
            f"    pairs = [(_norm(g), _norm(p))\n"
            f"             for g, p in zip(gt, pred)\n"
            f"             if g is not None and p is not None]\n"
            f"    if not pairs:\n"
            f"        return _PILImage.new('L', (256, 256), 0)\n"
            f"    classes = sorted({{g for g, _ in pairs}}, key=len, reverse=True)\n"
            f"    def _canon(s):\n"
            f"        if s in classes:\n"
            f"            return s\n"
            f"        for c in classes:\n"
            f"            if s.startswith(c) or c.startswith(s):\n"
            f"                return c\n"
            f"        toks = set(s.split())\n"
            f"        for c in classes:\n"
            f"            if c in toks or any(t.startswith(c) or c.startswith(t) for t in toks):\n"
            f"                return c\n"
            f"        return s\n"
            f"    canon_pairs = [(g, _canon(p)) for g, p in pairs]\n"
            f"    all_labels = sorted({{g for g, _ in canon_pairs}} | {{p for _, p in canon_pairs}})\n"
            f"    idx = {{c: i for i, c in enumerate(all_labels)}}\n"
            f"    n = len(all_labels)\n"
            f"    cm = _np.zeros((n, n), dtype=_np.int32)\n"
            f"    for g, p in canon_pairs:\n"
            f"        cm[idx[g], idx[p]] += 1\n"
            f"    norm = (cm / max(int(cm.max()), 1) * 255).astype(_np.uint8)\n"
            f"    img = _PILImage.fromarray(norm)\n"
            f"    return img.resize((256, 256), _PILImage.NEAREST)\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'is_aggregated': True,
        'accepts_aggregated_inputs': True,
        'llm_hint': (
            f"Aggregated confusion-matrix visualization between GT "
            f"text class `{col}` and submission's predicted text class "
            f"`{col}_pred`. `gt` and `pred` arrive as PARALLEL lists. "
            f"Function name MUST be `{global_name}(gt, pred)` and return "
            f"a PIL.Image (grayscale or RGB, ≤ 512x512)."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'scalar',
            'description': f"Per-sample predicted class name for `{col}`.",
            'gt_field': col,
        }],
    }


def _proposal_rmse_depth(col):
    """Standard depth RMSE on the valid-mask. The metric receives the
    GT and predicted depth maps as numpy arrays (engine-side loader
    handles the NPZ → array unmarshalling)."""
    global_name = "rmse"
    return {
        'target_name': f"RMSE ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample RMSE between predicted depth (`{col}_pred`) and "
            f"GT depth (`{col}`), restricted to the GT-valid mask "
            f"(non-zero, finite). Mean-pooled across samples."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    if gt is None or pred is None:\n"
            f"        return float('nan')\n"
            f"    g = np.asarray(gt, dtype=np.float32)\n"
            f"    p = np.asarray(pred, dtype=np.float32)\n"
            f"    if g.shape != p.shape:\n"
            f"        # Resize predicted to GT via simple slicing on the\n"
            f"        # common rectangle — keeps the metric well-defined\n"
            f"        # even when models output a different resolution.\n"
            f"        h = min(g.shape[0], p.shape[0])\n"
            f"        w = min(g.shape[1], p.shape[1])\n"
            f"        g, p = g[:h, :w], p[:h, :w]\n"
            f"    valid = (g > 0) & np.isfinite(g) & np.isfinite(p)\n"
            f"    if not valid.any():\n"
            f"        return float('nan')\n"
            f"    diff = g[valid] - p[valid]\n"
            f"    return float(np.sqrt(np.mean(diff * diff)))\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'lower_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Per-sample RMSE between predicted depth and GT depth, "
            f"masking out invalid (zero/NaN) GT pixels. Both inputs are "
            f"numpy 2D arrays. Function name MUST be `{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'depth', 'gt_field': col,
            'description': (
                f"Per-sample predicted depth map. Ship as "
                f"`{col}_pred/<sample>.npz` with the array under "
                f"key `depth`, matching GT shape."
            ),
        }],
    }


def _proposal_abs_rel_depth(col):
    global_name = "abs_rel"
    return {
        'target_name': f"abs-rel ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample absolute-relative depth error: mean of "
            f"|gt - pred| / gt over the GT-valid mask."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    if gt is None or pred is None:\n"
            f"        return float('nan')\n"
            f"    g = np.asarray(gt, dtype=np.float32)\n"
            f"    p = np.asarray(pred, dtype=np.float32)\n"
            f"    if g.shape != p.shape:\n"
            f"        h = min(g.shape[0], p.shape[0])\n"
            f"        w = min(g.shape[1], p.shape[1])\n"
            f"        g, p = g[:h, :w], p[:h, :w]\n"
            f"    valid = (g > 0) & np.isfinite(g) & np.isfinite(p)\n"
            f"    if not valid.any():\n"
            f"        return float('nan')\n"
            f"    return float(np.mean(np.abs(g[valid] - p[valid]) / g[valid]))\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'lower_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Per-sample absolute-relative depth error: mean of "
            f"|gt - pred| / gt over the GT-valid mask. Function name MUST "
            f"be `{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'depth', 'gt_field': col,
            'description': (
                f"Per-sample predicted depth map (same shape as GT)."
            ),
        }],
    }


def _proposal_a1_depth(col):
    """Depth-thresholded accuracy: fraction of pixels where
    max(gt/pred, pred/gt) < 1.25."""
    global_name = "a1"
    return {
        'target_name': f"δ < 1.25 ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample fraction of pixels whose ratio max(gt/pred, "
            f"pred/gt) is below 1.25 — the standard 'a1' depth accuracy."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    if gt is None or pred is None:\n"
            f"        return float('nan')\n"
            f"    g = np.asarray(gt, dtype=np.float32)\n"
            f"    p = np.asarray(pred, dtype=np.float32)\n"
            f"    if g.shape != p.shape:\n"
            f"        h = min(g.shape[0], p.shape[0])\n"
            f"        w = min(g.shape[1], p.shape[1])\n"
            f"        g, p = g[:h, :w], p[:h, :w]\n"
            f"    valid = (g > 0) & np.isfinite(g) & np.isfinite(p) & (p > 0)\n"
            f"    if not valid.any():\n"
            f"        return float('nan')\n"
            f"    ratio = np.maximum(g[valid] / p[valid], p[valid] / g[valid])\n"
            f"    return float(np.mean(ratio < 1.25))\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'higher_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Standard 'a1' depth accuracy on per-pixel ratios (max of "
            f"gt/pred and pred/gt) below 1.25. Function name MUST be "
            f"`{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'depth', 'gt_field': col,
            'description': "Per-sample predicted depth map (same shape as GT).",
        }],
    }


def _proposal_psnr_image(col):
    """PSNR on RGB arrays; auto-normalizes if input looks uint8."""
    global_name = "psnr"
    return {
        'target_name': f"PSNR ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample peak signal-to-noise ratio between predicted "
            f"image (`{col}_pred`) and GT image (`{col}`), in dB."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    if gt is None or pred is None:\n"
            f"        return float('nan')\n"
            f"    g = np.asarray(gt, dtype=np.float32)\n"
            f"    p = np.asarray(pred, dtype=np.float32)\n"
            f"    if g.shape != p.shape:\n"
            f"        h = min(g.shape[0], p.shape[0])\n"
            f"        w = min(g.shape[1], p.shape[1])\n"
            f"        g, p = g[:h, :w], p[:h, :w]\n"
            f"    # Normalize to [0, 1] for either uint8-range or already-normalized inputs.\n"
            f"    if max(float(g.max()), float(p.max())) > 1.5:\n"
            f"        g = g / 255.0\n"
            f"        p = p / 255.0\n"
            f"    mse = float(np.mean((g - p) ** 2))\n"
            f"    if mse < 1e-10:\n"
            f"        return 100.0\n"
            f"    return float(-10.0 * np.log10(mse))\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'higher_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Per-sample PSNR (dB) between GT and predicted RGB images. "
            f"Inputs are HxWx3 numpy arrays (uint8 or float; auto-detect). "
            f"Function name MUST be `{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'image', 'gt_field': col,
            'description': (
                f"Per-sample predicted RGB image. Ship as "
                f"`{col}_pred/<sample>.png`."
            ),
        }],
    }


def _proposal_miou_mask(col):
    """Mean IoU across the union of class IDs present in either GT or
    pred mask. Both are integer-valued image arrays."""
    global_name = "miou"
    return {
        'target_name': f"mean IoU ({col})",
        'global_name': global_name,
        'description': (
            f"Per-sample mean intersection-over-union across the class "
            f"IDs present in either GT mask (`{col}`) or prediction "
            f"(`{col}_pred`)."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    if gt is None or pred is None:\n"
            f"        return float('nan')\n"
            f"    g = np.asarray(gt).astype(np.int64)\n"
            f"    p = np.asarray(pred).astype(np.int64)\n"
            f"    # Color-encoded masks → flatten to a single channel by\n"
            f"    # treating each unique RGB tuple as one class.\n"
            f"    if g.ndim == 3:\n"
            f"        g = g[..., 0] * 256 * 256 + g[..., 1] * 256 + g[..., 2]\n"
            f"    if p.ndim == 3:\n"
            f"        p = p[..., 0] * 256 * 256 + p[..., 1] * 256 + p[..., 2]\n"
            f"    if g.shape != p.shape:\n"
            f"        h = min(g.shape[0], p.shape[0])\n"
            f"        w = min(g.shape[1], p.shape[1])\n"
            f"        g, p = g[:h, :w], p[:h, :w]\n"
            f"    g, p = g.flatten(), p.flatten()\n"
            f"    classes = np.unique(np.concatenate([g, p]))\n"
            f"    ious = []\n"
            f"    for c in classes:\n"
            f"        if c < 0:\n"
            f"            continue\n"
            f"        inter = int(((g == c) & (p == c)).sum())\n"
            f"        union = int(((g == c) | (p == c)).sum())\n"
            f"        if union > 0:\n"
            f"            ious.append(inter / union)\n"
            f"    return float(np.mean(ious)) if ious else float('nan')\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'sort_direction': 'higher_is_better',
        'pooling_type': 'mean',
        'llm_hint': (
            f"Per-sample mean IoU between GT segmentation mask and "
            f"predicted mask. Both are integer-valued (or RGB-encoded) "
            f"image arrays. Function name MUST be `{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'mask', 'gt_field': col,
            'description': (
                f"Per-sample predicted segmentation mask. Ship as "
                f"`{col}_pred/<sample>.png` with class IDs as pixel values "
                f"(matching GT encoding)."
            ),
        }],
    }


def _propose_metrics_for_dataset(ds):
    """Walk the GT columns and propose per-column metrics. Dispatches
    per kind:
      - scalar (ClassLabel-shaped) → top-1 accuracy
      - scalar (numeric)           → MAE
      - depth                      → RMSE + abs-rel + a1 (3 metrics per column)
      - image                      → PSNR
      - mask                       → mean IoU
    Submissions are expected to ship `<col>_pred/<sample>.<ext>` per
    GT column — extension depends on the kind (.txt / .npz / .png).
    """
    proposals = []
    for col, kind, hints in _gt_columns(ds):
        if kind == 'scalar':
            if hints['is_classlabel']:
                proposals.append(_proposal_top1_classlabel(col))
            else:
                proposals.append(_proposal_mae_scalar(col))
        elif kind == 'text':
            # Try the LLM suite first — it can tell apart classification
            # from generation / completion / QA / etc. and propose
            # task-appropriate metrics. Cache the result on the dataset
            # object so the parallel viz proposer reuses it without a
            # second round-trip.
            suite = _get_or_build_text_eval_suite(ds, col, hints)
            if suite and suite.get('metrics'):
                proposals.extend(suite['metrics'])
            else:
                # Fall back to the static classification-style proposal
                # when the LLM is unavailable or its response is unusable.
                proposals.append(_proposal_top1_text_classlabel(col))
                proposals.append(_proposal_macro_f1_text_classlabel(col))
        elif kind == 'depth':
            proposals.append(_proposal_rmse_depth(col))
            proposals.append(_proposal_abs_rel_depth(col))
            proposals.append(_proposal_a1_depth(col))
        elif kind == 'image':
            proposals.append(_proposal_psnr_image(col))
        elif kind == 'mask':
            proposals.append(_proposal_miou_mask(col))
    return proposals


def _viz_depth_error_heatmap(col):
    """Aggregated per-pixel mean-abs-error heatmap. `gt` and `pred`
    arrive as parallel lists of HxW depth arrays (is_aggregated=True,
    accepts_aggregated_inputs=True). Pads to common shape via simple
    crop so a heterogeneous-resolution submission still renders."""
    global_name = "depth_error_heatmap"
    return {
        'target_name': f"depth error heatmap ({col})",
        'global_name': global_name,
        'description': (
            f"Per-pixel mean absolute depth error averaged across all "
            f"samples of a submission, rendered as a 256x256 grayscale "
            f"PIL image (brighter = larger error)."
        ),
        'fallback_code': (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"Aggregated depth error heatmap. gt and pred are\n"
            f"    parallel lists of HxW numpy arrays.\"\"\"\n"
            f"    import numpy as _np\n"
            f"    from PIL import Image as _PILImage\n"
            f"    pairs = []\n"
            f"    for g, p in zip(gt, pred):\n"
            f"        if g is None or p is None:\n"
            f"            continue\n"
            f"        g = _np.asarray(g, dtype=_np.float32)\n"
            f"        p = _np.asarray(p, dtype=_np.float32)\n"
            f"        h = min(g.shape[0], p.shape[0])\n"
            f"        w = min(g.shape[1], p.shape[1])\n"
            f"        if h == 0 or w == 0:\n"
            f"            continue\n"
            f"        pairs.append((g[:h, :w], p[:h, :w]))\n"
            f"    if not pairs:\n"
            f"        return _PILImage.new('L', (256, 256), 0)\n"
            f"    H = min(pp[0].shape[0] for pp in pairs)\n"
            f"    W = min(pp[0].shape[1] for pp in pairs)\n"
            f"    acc = _np.zeros((H, W), dtype=_np.float64)\n"
            f"    for g, p in pairs:\n"
            f"        acc += _np.abs(g[:H, :W] - p[:H, :W])\n"
            f"    acc /= len(pairs)\n"
            f"    norm = (acc / max(float(acc.max()), 1e-6) * 255).astype(_np.uint8)\n"
            f"    return _PILImage.fromarray(norm).resize((256, 256), _PILImage.NEAREST)\n"
        ),
        'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
        'is_aggregated': True,
        'accepts_aggregated_inputs': True,
        'llm_hint': (
            f"Aggregated depth-error heatmap. `gt` and `pred` are PARALLEL "
            f"lists of 2D depth arrays. Compute mean per-pixel |gt - pred|, "
            f"normalize to [0, 255] grayscale, return a 256x256 PIL.Image. "
            f"Function name MUST be `{global_name}(gt, pred)`."
        ),
        'pred_fields': [{
            'name': f'{col}_pred', 'kind': 'depth', 'gt_field': col,
            'description': "Per-sample predicted depth map (.npz, matches GT shape).",
        }],
    }


def _propose_visualizations_for_dataset(ds):
    """Mirror of _propose_metrics_for_dataset for visualizations.
    One viz per GT column where the auto-proposer can pick a
    canonical summary plot.

    Per kind:
    - scalar (ClassLabel): aggregated confusion-matrix heatmap.
    - depth: aggregated mean-abs-error heatmap (per-pixel mean of
      |gt - pred| across all submission samples).
    - image: not auto-proposed (side-by-side strips need padding to a
      common shape; left for manual authoring).
    - mask: not auto-proposed (per-sample diff is more useful than an
      aggregated mIoU bar; left for manual authoring).
    """
    proposals = []
    for col, kind, hints in _gt_columns(ds):
        if kind == 'depth':
            proposals.append(_viz_depth_error_heatmap(col))
            continue
        if kind == 'text':
            # Mirror the metric path: ask the LLM what viz makes sense
            # for this text task. For classification it'll propose a
            # confusion matrix; for generation/QA it might skip viz
            # entirely. Fall back to the static text confusion matrix
            # when the LLM is unavailable.
            suite = _get_or_build_text_eval_suite(ds, col, hints)
            if suite and suite.get('visualizations'):
                proposals.extend(suite['visualizations'])
            elif suite is None:
                # LLM unavailable / failed — use the static confusion
                # matrix (matches the static-fallback metric pair).
                proposals.append(_viz_text_confusion_matrix(col))
            # If suite returned no viz, that's a deliberate LLM choice
            # (e.g. for free-form generation a confusion matrix is
            # meaningless). Don't second-guess it.
            continue
        if kind != 'scalar' or not hints.get('is_classlabel'):
            continue
        global_name = "confusion_matrix"
        target_name = f"confusion matrix ({col})"
        description = (
            f"Aggregated confusion matrix between GT `{col}` "
            f"(ClassLabel index) and submission's `{col}_pred`."
        )
        fallback_code = (
            f"def {global_name}(gt, pred):\n"
            f"    \"\"\"Aggregated confusion matrix as a 256x256 grayscale heatmap.\n\n"
            f"    `gt` and `pred` are LISTS spanning every sample of a single\n"
            f"    submission (is_aggregated=True, accepts_aggregated_inputs=True).\n"
            f"    \"\"\"\n"
            f"    import numpy as _np\n"
            f"    from PIL import Image as _PILImage\n"
            f"    pairs = [(int(g), int(p)) for g, p in zip(gt, pred)\n"
            f"             if g is not None and p is not None]\n"
            f"    if not pairs:\n"
            f"        return _PILImage.new('L', (256, 256), 0)\n"
            f"    classes = sorted({{g for g, _ in pairs}} | {{p for _, p in pairs}})\n"
            f"    idx = {{c: i for i, c in enumerate(classes)}}\n"
            f"    n = len(classes)\n"
            f"    cm = _np.zeros((n, n), dtype=_np.int32)\n"
            f"    for g, p in pairs:\n"
            f"        cm[idx[g], idx[p]] += 1\n"
            f"    norm = (cm / max(int(cm.max()), 1) * 255).astype(_np.uint8)\n"
            f"    img = _PILImage.fromarray(norm)\n"
            f"    return img.resize((256, 256), _PILImage.NEAREST)\n"
        )
        llm_hint = (
            f"Aggregated confusion-matrix visualization between GT "
            f"integer class index `{col}` and submission's predicted "
            f"index `{col}_pred`. `gt` and `pred` arrive as PARALLEL "
            f"lists. Function name MUST be `{global_name}(gt, pred)` "
            f"and return a PIL.Image (grayscale or RGB, ≤ 512x512)."
        )
        proposals.append({
            'target_name': target_name,
            'global_name': global_name,
            'description': description,
            'fallback_code': fallback_code,
            'arg_mappings': {'gt': f'gt_{col}', 'pred': f'sub_{col}_pred'},
            'is_aggregated': True,
            'accepts_aggregated_inputs': True,
            'llm_hint': llm_hint,
            # Same pred contract as the matching metric proposal. The
            # auto-LB helper unique-deduplicates by name across both
            # proposers so the user only sees `<col>_pred` once.
            'pred_fields': [{
                'name': f'{col}_pred',
                'kind': 'scalar',
                'description': f"Per-sample predicted class index for `{col}`.",
                'gt_field': col,
            }],
        })
    return proposals


def _lb_submission_pred_fields(lb):
    """Derive the prediction-field schema for an LB's submissions by
    walking its metrics + visualizations and pulling out arg_mappings
    keys that reference `sub_<x>_pred`. De-duplicates across metrics
    and visualizations by field name.

    Returns a list of dicts: [{name, gt_field, kind, description,
    used_by}]. `description` infers from the matching GT custom field
    on the first dataset (ClassLabel-shaped → 'predicted class index',
    plain numeric → 'predicted value'). `used_by` lists the LB metric
    + visualization target names that consume this pred field, so the
    LB page can show which scoreboards depend on each prediction."""
    seen = {}

    def _record(field_name, gt_field, used_by):
        if field_name in seen:
            seen[field_name]['used_by'].append(used_by)
            return
        seen[field_name] = {
            'name': field_name,
            'gt_field': gt_field,
            'kind': 'scalar',
            'description': '',  # filled in below
            'used_by': [used_by],
        }

    sources = (
        [('metric', lm) for lm in (lb.leaderboard_metrics or [])]
        + [('viz', lv) for lv in (lb.leaderboard_visualizations or [])]
    )
    for kind, source in sources:
        try:
            mappings = json.loads(source.arg_mappings or '{}')
        except (TypeError, ValueError):
            mappings = {}
        used_by_label = (
            source.target_name
            or (source.global_metric.name if kind == 'metric'
                else source.global_visualization.name)
        )
        for ctx_key in mappings.values():
            if not isinstance(ctx_key, str):
                continue
            # Only `sub_<x>_pred` shapes are submission-side prediction
            # fields; bare `sub_<x>` for precomputed metric values is
            # a different contract and irrelevant here.
            if not ctx_key.startswith('sub_') or not ctx_key.endswith('_pred'):
                continue
            field_name = ctx_key[len('sub_'):]
            gt_field = field_name[:-len('_pred')]
            _record(field_name, gt_field, used_by_label)

    # Now backfill descriptions by inspecting the GT side of the first
    # dataset for each pred field's `gt_field`.
    #
    # Two source shapes:
    #   - BH-attached LBs: read field types off the first sample's
    #     custom_fields.
    #   - HF-attached LBs (no Sample rows): read kinds from the
    #     primary HF Attachment's `hf_mapping_json`, which carries
    #     `{column, target_kind, target_field}` per HF column. Without
    #     this fallback, every HF LB's pred field defaulted to scalar,
    #     so e.g. NYU `raw_depth_map_pred` showed type=scalar instead
    #     of depth.
    gt_field_meta = {}
    for ds in (lb.datasets or []):
        if not ds.samples:
            continue
        first = ds.samples[0]
        for cf in (first.custom_fields or []):
            gt_field_meta.setdefault(cf.name, cf.field_type)
        # Walk one dataset's worth of fields; that's enough to infer.
        break
    if not gt_field_meta:
        for att in (lb.attachments or []):
            if getattr(att, 'kind', None) != 'hf' or att.role != 'primary':
                continue
            try:
                mapping = json.loads(att.hf_mapping_json or '[]')
            except (TypeError, ValueError):
                mapping = []
            for m in mapping:
                target_field = (m.get('target_field') or m.get('column') or '').strip()
                target_kind = (m.get('target_kind') or '').strip()
                if target_field and target_kind:
                    gt_field_meta.setdefault(target_field, target_kind)
            break
    for entry in seen.values():
        gt_name = entry['gt_field']
        sibling_class = f"{gt_name}_class"
        gt_type = gt_field_meta.get(gt_name)
        # Kind drives the submission file extension: scalar → .txt,
        # depth → .npz, image/mask → .png. Match what the proposer set
        # at metric-creation time.
        if sibling_class in gt_field_meta:
            entry['kind'] = 'scalar'
            entry['description'] = (
                f"Predicted class index for `{gt_name}` "
                f"(per-sample integer, ship as `<sample>.txt`)."
            )
        elif gt_type == 'scalar':
            entry['kind'] = 'scalar'
            entry['description'] = (
                f"Predicted numeric value for `{gt_name}` "
                f"(per-sample float, ship as `<sample>.txt`)."
            )
        elif gt_type == 'depth':
            entry['kind'] = 'depth'
            entry['description'] = (
                f"Predicted depth map for `{gt_name}` "
                f"(ship as `<sample>.npz` with array key `depth`)."
            )
        elif gt_type == 'image':
            name_lc = gt_name.lower()
            if any(h in name_lc for h in _MASK_NAME_HINTS):
                entry['kind'] = 'mask'
                entry['description'] = (
                    f"Predicted segmentation mask for `{gt_name}` "
                    f"(ship as `<sample>.png` with class IDs / RGB encoding)."
                )
            else:
                entry['kind'] = 'image'
                entry['description'] = (
                    f"Predicted image for `{gt_name}` "
                    f"(ship as `<sample>.png` matching GT shape)."
                )
        elif gt_type == 'text':
            entry['kind'] = 'scalar'  # ship as <sample>.txt
            entry['description'] = (
                f"Predicted text label for `{gt_name}` "
                f"(per-sample string, ship as `<sample>.txt`)."
            )
        else:
            entry['kind'] = 'scalar'
            entry['description'] = (
                f"Per-sample prediction paired against GT `{gt_name}`."
            )

    # Phase 9: required pred fields declared independently of any
    # metric/viz. Organizer might want raw predictions for human
    # review even without scoring. Merge in.
    #
    # When an extras entry's name matches a metric-derived one, treat
    # it as an *override*: update the kind/description on the derived
    # entry without replacing its `used_by` context. This is how the
    # auto-LB preview's "edit type" affordance reaches the submission
    # contract — the rename lives in the metric's arg_mappings, and
    # the kind override lives in required_pred_fields_json.
    try:
        extras = json.loads(lb.required_pred_fields_json or '[]')
    except (TypeError, ValueError):
        extras = []
    for entry in (extras or []):
        if not isinstance(entry, dict):
            continue
        name = (entry.get('name') or '').strip()
        if not name:
            continue
        if name in seen:
            if entry.get('kind'):
                seen[name]['kind'] = entry['kind']
            if entry.get('description'):
                seen[name]['description'] = entry['description']
            continue
        seen[name] = {
            'name': name,
            'gt_field': (entry.get('gt_field') or name.removesuffix('_pred')) or name,
            'kind': entry.get('kind') or 'scalar',
            'description': entry.get('description') or (
                f"Required prediction field `{name}` (no scoring "
                "metric — organizer wants the raw values)."
            ),
            'used_by': ['(no metric — required by LB)'],
        }
    return list(seen.values())


def _llm_generate_metric_code(global_name, llm_hint):
    """Ask Claude for the body of a per-sample metric. Returns Python
    source defining a single function whose name matches `global_name`,
    or None if the API key is missing / the call fails.

    The callee takes named kwargs `gt` and `pred` (per the proposer's
    arg_mappings) and returns a float. The caller falls back to the
    proposal's `fallback_code` on None."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    system_prompt = (
        "You write tiny per-sample metric functions for the BenchHub "
        "benchmarking platform.\n\n"
        "Hard requirements:\n"
        "- Output ONLY Python source (no explanation, no fences).\n"
        "- Define exactly ONE top-level function. Its name MUST equal "
        "the `global_name` you are given.\n"
        "- Keyword arguments must be `gt` and `pred` (ground-truth + "
        "submitted value).\n"
        "- The function returns a Python float.\n"
        "- Use only Python stdlib + numpy (already imported as `np` "
        "by the harness — do not re-import).\n"
        "- Be defensive about types: gt/pred may arrive as int, float, "
        "or numeric strings.\n"
    )
    user_msg = (
        f"global_name: {global_name}\n"
        f"semantics: {llm_hint}\n"
    )
    try:
        import requests as _r
        resp = _r.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 800,
                "system": [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        body = resp.json()
        text = ''.join(
            block.get('text', '')
            for block in body.get('content', [])
            if block.get('type') == 'text'
        ).strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:python)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        # Sanity-check: the function name must be present.
        if f"def {global_name}(" not in text:
            return None
        return text
    except Exception as e:
        print(f"_llm_generate_metric_code failed: {e}")
        return None


def _llm_generate_visualization_code(global_name, llm_hint,
                                     is_aggregated, accepts_aggregated_inputs):
    """Sister to `_llm_generate_metric_code` for visualizations: ask
    Claude for a function that returns a PIL.Image. Returns the source
    on success, None when the API key is missing or the function-name
    safety check fails. Caller falls back to the proposal's
    `fallback_code`."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    aggregation_clause = (
        "The function is AGGREGATED across a single submission's "
        "samples. `gt` and `pred` arrive as parallel Python lists "
        "spanning every sample. Reduce them into one image."
        if is_aggregated and accepts_aggregated_inputs
        else "The function is PER-SAMPLE. `gt` and `pred` are scalars."
    )
    system_prompt = (
        "You write tiny visualization functions for the BenchHub "
        "benchmarking platform. Each function returns a PIL.Image "
        "summarizing a submission's behavior on a leaderboard.\n\n"
        "Hard requirements:\n"
        "- Output ONLY Python source (no explanation, no fences).\n"
        "- Define exactly ONE top-level function. Its name MUST equal "
        "the `global_name` you are given.\n"
        "- The function MUST return a PIL.Image (max 512x512). Use "
        "`from PIL import Image` inside the function body.\n"
        "- Use only Python stdlib + numpy (already imported as `np` "
        "by the harness — do not re-import).\n"
        "- Be defensive: gt/pred may include None entries or be empty.\n\n"
        f"Aggregation: {aggregation_clause}"
    )
    user_msg = (
        f"global_name: {global_name}\n"
        f"semantics: {llm_hint}\n"
    )
    try:
        import requests as _r
        resp = _r.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1500,
                "system": [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        text = ''.join(
            block.get('text', '')
            for block in body.get('content', [])
            if block.get('type') == 'text'
        ).strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:python)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        if f"def {global_name}(" not in text:
            return None
        return text
    except Exception as e:
        print(f"_llm_generate_visualization_code failed: {e}")
        return None


def _enrich_metric_proposal_with_existing_code(p):
    """If a GlobalMetric with the proposed name already exists, attach
    its current python_code to the proposal so the preview UI shows
    what the user is reusing (and they can override it). Otherwise try
    the LLM to author code (with the deterministic fallback as the
    floor). Mutates `p` in-place: adds `python_code` and `code_source`
    ∈ {'existing', 'llm', 'static'}."""
    existing = GlobalMetric.query.filter_by(name=p['global_name']).first()
    if existing is not None:
        p['python_code'] = existing.python_code
        p['code_source'] = 'existing'
        p['existing_global_metric_id'] = existing.id
        return p
    llm_code = _llm_generate_metric_code(p['global_name'], p['llm_hint'])
    if llm_code:
        p['python_code'] = llm_code
        p['code_source'] = 'llm'
    else:
        p['python_code'] = p['fallback_code']
        p['code_source'] = 'static'
    return p


def _enrich_viz_proposal_with_existing_code(p):
    """Sister of _enrich_metric_proposal_with_existing_code for vis."""
    existing = GlobalVisualization.query.filter_by(name=p['global_name']).first()
    if existing is not None:
        p['python_code'] = existing.python_code
        p['code_source'] = 'existing'
        p['existing_global_visualization_id'] = existing.id
        return p
    llm_code = _llm_generate_visualization_code(
        p['global_name'], p['llm_hint'],
        p['is_aggregated'], p['accepts_aggregated_inputs'],
    )
    if llm_code:
        p['python_code'] = llm_code
        p['code_source'] = 'llm'
    else:
        p['python_code'] = p['fallback_code']
        p['code_source'] = 'static'
    return p


def _collect_auto_lb_proposals(dataset):
    """Run the metric + viz proposers and enrich each entry with the
    code the LB would persist if the user accepts it as-is. Returns
    (metric_proposals, viz_proposals). The preview page renders both
    and the user picks which to keep / edits the code."""
    metric_proposals = [
        _enrich_metric_proposal_with_existing_code(dict(p))
        for p in _propose_metrics_for_dataset(dataset)
    ]
    viz_proposals = [
        _enrich_viz_proposal_with_existing_code(dict(p))
        for p in _propose_visualizations_for_dataset(dataset)
    ]
    return metric_proposals, viz_proposals


def _auto_create_lb_with_metrics(dataset, lb_name, owner_user_id,
                                 *, metric_proposals=None, viz_proposals=None):
    """Create a Leaderboard backed by `dataset`, attach metrics and
    visualizations from the supplied (or freshly proposed) lists, and
    reuse any existing GlobalMetric / GlobalVisualization whose name
    matches (strict name match). When proposals aren't supplied, runs
    the proposers + LLM-fallback code-gen path itself.

    Returns (success, message, lb_id)."""
    if not lb_name:
        return False, "Leaderboard name is required.", None
    if Leaderboard.query.filter_by(name=lb_name).first():
        return False, f'A leaderboard named "{lb_name}" already exists.', None

    if metric_proposals is None and viz_proposals is None:
        metric_proposals, viz_proposals = _collect_auto_lb_proposals(dataset)
    metric_proposals = metric_proposals or []
    viz_proposals = viz_proposals or []
    if not metric_proposals and not viz_proposals:
        return False, ("No GT scalar fields detected on the dataset — "
                       "nothing to auto-attach. Create the leaderboard "
                       "manually and add metrics yourself."), None

    lb = Leaderboard(
        name=lb_name,
        summary_metrics=','.join(p['target_name'] for p in metric_proposals),
        owner_user_id=owner_user_id,
    )
    lb.datasets = [dataset]
    db.session.add(lb)
    db.session.flush()  # get lb.id

    for p in metric_proposals:
        gm = GlobalMetric.query.filter_by(name=p['global_name']).first()
        if gm is None:
            code = (
                p.get('python_code')
                or _llm_generate_metric_code(p['global_name'], p['llm_hint'])
                or p['fallback_code']
            )
            gm = GlobalMetric(
                name=p['global_name'],
                description=p['description'],
                python_code=code,
                is_aggregated=False,
                accepts_aggregated_inputs=False,
                owner_user_id=owner_user_id,
                visibility='public',
            )
            db.session.add(gm)
            db.session.flush()
        lm = LeaderboardMetric(
            leaderboard_id=lb.id,
            global_metric_id=gm.id,
            arg_mappings=json.dumps(p['arg_mappings']),
            target_name=p['target_name'],
            pooling_type=p['pooling_type'],
            sort_direction=p['sort_direction'],
        )
        db.session.add(lm)

    for p in viz_proposals:
        gv = GlobalVisualization.query.filter_by(name=p['global_name']).first()
        if gv is None:
            code = (
                p.get('python_code')
                or _llm_generate_visualization_code(
                    p['global_name'], p['llm_hint'],
                    p['is_aggregated'], p['accepts_aggregated_inputs'],
                )
                or p['fallback_code']
            )
            gv = GlobalVisualization(
                name=p['global_name'],
                description=p['description'],
                python_code=code,
                is_aggregated=p['is_aggregated'],
                accepts_aggregated_inputs=p['accepts_aggregated_inputs'],
                owner_user_id=owner_user_id,
                visibility='public',
            )
            db.session.add(gv)
            db.session.flush()
        lv = LeaderboardVisualization(
            leaderboard_id=lb.id,
            global_visualization_id=gv.id,
            arg_mappings=json.dumps(p['arg_mappings']),
            target_name=p['target_name'],
        )
        db.session.add(lv)

    db.session.commit()
    n_m, n_v = len(metric_proposals), len(viz_proposals)
    summary = []
    if n_m:
        summary.append(f"{n_m} metric{'' if n_m == 1 else 's'}")
    if n_v:
        summary.append(f"{n_v} visualization{'' if n_v == 1 else 's'}")
    # Tell the user what their submissions need to ship — same fields
    # the colab notebook will template into the prediction loop.
    pred_field_names = []
    seen_pred = set()
    for p in metric_proposals + viz_proposals:
        for pf in p.get('pred_fields') or []:
            if pf['name'] in seen_pred:
                continue
            seen_pred.add(pf['name'])
            pred_field_names.append(pf['name'])
    pred_clause = ''
    if pred_field_names:
        pred_clause = (
            ' Submissions must ship per-sample predictions in: '
            + ', '.join(f'`{n}`' for n in pred_field_names) + '.'
        )
    return True, (f'Created leaderboard "{lb_name}" '
                  f'with {" and ".join(summary)}.{pred_clause}'), lb.id


def _ensure_colab_gist(lb):
    """Materialize the LB's notebook as a GitHub gist so Colab's URL
    importer (which only whitelists github.com / gist.github.com /
    drive.google.com / raw.githubusercontent.com) can fetch it.

    Returns (gist_html_url, gist_id, gist_owner) on success, or
    (None, None, None) when BENCHHUB_GITHUB_GIST_TOKEN isn't configured
    or the API call fails. Caller falls back to the manual download path
    in either case.

    Cache shape: lb.colab_notebook_cache = {sig, notebook, gist_id, gist_owner}.
    On signature drift, the notebook is regenerated and the gist is
    PATCH'd in place — no orphan gists per LB version.
    """
    token = os.environ.get('BENCHHUB_GITHUB_GIST_TOKEN')
    if not token:
        return None, None, None

    notebook, _src = _get_or_generate_colab_notebook(lb)
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', lb.name) or f'lb_{lb.id}'
    filename = f'{safe_name}_submit.ipynb'
    description = f"BenchHub submission scaffold for leaderboard '{lb.name}' (id={lb.id})"

    # Re-read cache to get any existing gist_id + owner login.
    gist_id = None
    gist_owner = None
    if lb.colab_notebook_cache:
        try:
            wrapped = json.loads(lb.colab_notebook_cache) or {}
            gist_id = wrapped.get('gist_id')
            gist_owner = wrapped.get('gist_owner')
        except Exception:
            gist_id = None
            gist_owner = None

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    payload = {
        'description': description,
        'public': False,
        'files': {filename: {'content': notebook}},
    }
    try:
        import requests as _r
        gist_resp_json = None
        if gist_id:
            # Update existing gist (idempotent for unchanged content).
            resp = _r.patch(
                f'https://api.github.com/gists/{gist_id}',
                headers=headers, json=payload, timeout=15,
            )
            if resp.status_code == 404:
                gist_id = None  # was deleted upstream; create fresh
            else:
                resp.raise_for_status()
                gist_resp_json = resp.json()
        if not gist_id:
            resp = _r.post(
                'https://api.github.com/gists',
                headers=headers, json=payload, timeout=15,
            )
            resp.raise_for_status()
            gist_resp_json = resp.json()
            gist_id = gist_resp_json.get('id')
        if not gist_id:
            return None, None
        # Owner login is part of the Colab gist URL pattern
        # (colab.research.google.com/gist/<owner>/<gist_id>) — Colab
        # rejects the bare-id form with "Unexpected GitHub Gist path".
        if gist_resp_json:
            owner_obj = gist_resp_json.get('owner') or {}
            new_owner = owner_obj.get('login')
            if new_owner:
                gist_owner = new_owner
        if not gist_owner:
            # Fallback: parse from html_url if we got one (`https://gist.github.com/<owner>/<id>`).
            html_url = (gist_resp_json or {}).get('html_url', '')
            m = re.match(r'https://gist\.github\.com/([^/]+)/[0-9a-f]+', html_url)
            if m:
                gist_owner = m.group(1)
        # Persist gist_id + owner in the cache wrapper so future calls reuse them.
        try:
            wrapped = json.loads(lb.colab_notebook_cache or '{}') or {}
            wrapped['gist_id'] = gist_id
            if gist_owner:
                wrapped['gist_owner'] = gist_owner
            lb.colab_notebook_cache = json.dumps(wrapped)
            db.session.commit()
        except Exception:
            db.session.rollback()
        return (
            f'https://gist.github.com/{gist_owner}/{gist_id}'
            if gist_owner else f'https://gist.github.com/{gist_id}',
            gist_id,
            gist_owner,
        )
    except Exception as e:
        print(f"_ensure_colab_gist failed: {e}")
        return None, None, None


def _llm_infer_mapping(features, dataset_repo=None):
    """Ask Claude to map each HF feature into a BenchHub field type.
    Returns a list of {column, target_kind, target_field, reason} on
    success, or None when the LLM is unavailable / parsing fails — in
    which case the caller falls back to the rule-based _infer_mapping().

    Activates only when ANTHROPIC_API_KEY is set so local-dev / tests
    skip the network round-trip without configuration.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    # Compact features payload — column name + normalized type only.
    payload_features = {col: desc.get('type', 'unknown')
                        for col, desc in features.items()}
    if not payload_features:
        return None

    system_prompt = (
        "You map columns from a HuggingFace dataset's feature schema onto "
        "BenchHub field types for benchmark ingestion.\n\n"
        "Allowed target_kind values:\n"
        "- image: 2D RGB visual data (photos, rendered frames)\n"
        "- depth: 2D depth/disparity/distance maps\n"
        "- scalar: ground-truth label / numeric value the model is "
        "  expected to predict (class index, regression target, etc.)\n"
        "- metric: a USER-PRECOMPUTED per-sample metric value the LB "
        "  should display directly (e.g. an MSE the user already ran "
        "  and is shipping alongside the data). RARE — almost never "
        "  the right pick for a labeled HF dataset.\n"
        "- histogram: count distribution (vector of bin counts)\n"
        "- text: textual descriptions, captions, tags\n"
        "- skip: not useful for benchmarking (file paths, metadata, etc.)\n\n"
        "Rules:\n"
        "- Image() with depth-suggesting name → depth\n"
        "- Image() otherwise → image\n"
        "- ClassLabel → scalar. Store the integer index as a GT scalar.\n"
        "  The importer ALSO automatically emits two derived artifacts\n"
        "  from the ClassLabel.names list, so YOU MUST NOT add separate\n"
        "  entries for them: (a) a `<col>_class` text column with the\n"
        "  human class name, (b) per-sample tags. Treat ClassLabel as a\n"
        "  single source column → one mapping entry, target_kind='scalar'.\n"
        "- Numeric Value (int/float) → scalar (it's a GT label/value).\n"
        "  Only choose `metric` when the column name explicitly says\n"
        "  the value is a precomputed metric result (e.g. `mse_pretrained`,\n"
        "  `accuracy_baseline`).\n"
        "- Sequence of integers, fixed length → histogram\n"
        "- Strings: 'caption'/'text'/'tags'/'description' → text, others → skip\n"
        "- Use the column name's semantics, not just the dtype.\n\n"
        "Output discipline (very important):\n"
        "- Output EXACTLY ONE entry per source column from the input list.\n"
        "  Do not invent additional columns (no `<col>_class`, no `tag`,\n"
        "  no `<col>_name`). The system handles those derivations on its\n"
        "  own from the original feature metadata.\n"
        "- Never split one source column into multiple BenchHub fields.\n"
        "- target_field MUST follow BenchHub conventions verbatim:\n"
        "  image_<col>, raw_<col>, hist_<col>, metric_<col> (only for "
        "  the rare `metric` kind), or the bare column name for `scalar` "
        "  and `text`.\n\n"
        "Output a JSON array, one entry per column, with exactly these keys: "
        '`column`, `target_kind`, `target_field`, `reason`. '
        '`reason` is one short sentence (≤ 15 words) explaining the choice. '
        "Return ONLY the JSON array, no prose."
    )

    user_msg = f"Dataset: {dataset_repo or 'unknown'}\n"
    user_msg += "Columns:\n"
    for col, t in payload_features.items():
        user_msg += f"  - {col}: {t}\n"

    try:
        import requests as _r
        resp = _r.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 2048,
                # Cache the rules — they don't change per request.
                "system": [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        body = resp.json()
        text = ''.join(
            block.get('text', '')
            for block in body.get('content', [])
            if block.get('type') == 'text'
        ).strip()
        # Tolerate the model wrapping in fenced code blocks.
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return None
        # Validate each entry has the required shape and a known target_kind.
        # Defensive dedupe: keep the FIRST entry per source column. The
        # prompt forbids splitting a column, but model output is best-effort.
        valid_kinds = {'image', 'depth', 'scalar', 'metric',
                       'histogram', 'text', 'skip'}
        by_col = {}
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            col = entry.get('column')
            kind = entry.get('target_kind', 'skip')
            if col not in payload_features or kind not in valid_kinds:
                continue
            if col in by_col:
                continue  # Drop derived/duplicate entries (e.g. label_class).
            by_col[col] = {
                'column': col,
                'target_kind': kind,
                'target_field': str(entry.get('target_field') or col)[:120],
                'reason': str(entry.get('reason') or '')[:200],
            }
        # Cover any column the model dropped.
        for col in payload_features:
            if col not in by_col:
                by_col[col] = {
                    'column': col, 'target_kind': 'skip',
                    'target_field': '',
                    'reason': 'Not classified by the model — defaulted to skip.',
                }
        return list(by_col.values())
    except Exception as e:
        print(f"_llm_infer_mapping failed: {e}")
        return None


def _infer_mapping(features):
    """Heuristic: map each feature into a BenchHub field type, or 'skip'.
    Returns a list of {column, target_kind, target_field, reason}.

    PWC bulk imports leaned on this when no LLM is configured, and any
    column flagged 'skip' produces zero GT in the LB-scoped cache (so
    the comparison view ends up empty). String / Audio / arbitrary-
    sequence columns therefore default to 'text' or 'audio' rather than
    'skip' — better to show *something* than nothing.
    """
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
        elif t == 'Audio':
            out.append({'column': col, 'target_kind': 'audio',
                        'target_field': f'audio_{col}',
                        'reason': "Audio-typed column → audio waveform thumb"})
        elif t.startswith('Value:'):
            dtype = t.split(':', 1)[1]
            if dtype in ('int8', 'int16', 'int32', 'int64',
                         'uint8', 'uint16', 'uint32',
                         'float16', 'float32', 'float64', 'bool'):
                out.append({'column': col, 'target_kind': 'scalar',
                            'target_field': col,
                            'reason': (f"Numeric scalar ({dtype}) → GT "
                                       f"scalar field. Pick `metric` "
                                       f"explicitly only when the column "
                                       f"already holds a user-precomputed "
                                       f"metric value.")})
            elif dtype == 'string':
                # Default to text for *any* string column. The previous
                # whitelist-only behaviour (caption/text/tag) skipped
                # almost every QA / code / dialog dataset, which left
                # the GT cache empty for the bulk-PWC imports.
                out.append({'column': col, 'target_kind': 'text',
                            'target_field': col,
                            'reason': "String column → GT text field"})
            else:
                # HF's schema parser sometimes flattens complex nested
                # features (Sequence-of-dict, list-of-list, Translation)
                # into `Value:unknown` instead of preserving the shape.
                # Persist as JSON so the row's actual value (dict / list)
                # still lands in the GT cache for human inspection.
                out.append({'column': col, 'target_kind': 'json',
                            'target_field': col,
                            'reason': f"Value:{dtype} → GT json field (complex/nested)"})
        elif t == 'ClassLabel':
            out.append({'column': col, 'target_kind': 'scalar',
                        'target_field': col,
                        'reason': ("ClassLabel → store integer index as a "
                                   "GT scalar (metric_* is reserved for "
                                   "user-precomputed metric values).")})
        elif t.startswith('Sequence:'):
            inner = t.split(':', 1)[1]
            length = desc.get('length', -1)
            if inner == 'string':
                out.append({'column': col, 'target_kind': 'text',
                            'target_field': col,
                            'reason': "Sequence:string → join into GT text"})
            elif inner in ('int32', 'int64', 'uint8', 'uint16', 'uint32') and length in (256, 512, 1024, 2048):
                out.append({'column': col, 'target_kind': 'histogram',
                            'target_field': f'hist_{col}',
                            'reason': f"Fixed-length int sequence ({length}) → hist_*"})
            else:
                # Lists of floats/ints, ClassLabels (e.g. answer-span
                # offsets in QA datasets), bounding boxes etc. Persist
                # as serialized JSON so the comparison view can render
                # them as a compact key/value card.
                out.append({'column': col, 'target_kind': 'json',
                            'target_field': col,
                            'reason': f"Sequence:{inner} → GT json field"})
        else:
            # Catch-all (dict / Translation / Audio sequence / etc.) —
            # serialize the row's value to JSON so SOMETHING shows up.
            out.append({'column': col, 'target_kind': 'json',
                        'target_field': col,
                        'reason': f"Feature type '{t}' → GT json field"})
    return out


def _lb_dataset_role(lb_id, dataset_id):
    """Read the `role` of a (leaderboard, dataset) attachment. Returns
    'primary' for legacy rows that pre-date the column or attachments
    without an explicit role."""
    row = db.session.execute(
        leaderboard_datasets.select().where(
            (leaderboard_datasets.c.leaderboard_id == lb_id) &
            (leaderboard_datasets.c.dataset_id == dataset_id)
        )
    ).first()
    if row is None:
        return None
    return getattr(row, 'role', None) or 'primary'


def _set_lb_dataset_role(lb_id, dataset_id, role):
    """Update the `role` of a (leaderboard, dataset) attachment.
    Idempotent. Only writes when the value actually changes."""
    if role not in ('primary', 'gt_source'):
        raise ValueError(f"unknown role {role!r}")
    current = _lb_dataset_role(lb_id, dataset_id)
    if current is None or current == role:
        # Either not attached, or already the right role.
        if current is None:
            return False
        return False
    db.session.execute(
        leaderboard_datasets.update()
        .where(
            (leaderboard_datasets.c.leaderboard_id == lb_id) &
            (leaderboard_datasets.c.dataset_id == dataset_id)
        )
        .values(role=role)
    )
    db.session.commit()
    return True


def _gt_source_datasets_for_lb(lb):
    """List of Dataset rows attached to `lb` with role='gt_source'.
    Empty list when no paired GT is configured (the common case)."""
    if lb is None:
        return []
    rows = db.session.execute(
        leaderboard_datasets.select().where(
            (leaderboard_datasets.c.leaderboard_id == lb.id) &
            (leaderboard_datasets.c.role == 'gt_source')
        )
    ).all()
    out = []
    for r in rows:
        ds = Dataset.query.get(r.dataset_id)
        if ds is not None:
            out.append(ds)
    return out


# ---------------------------------------------------------------------------
# Virtual-sample iteration for HF-ref attachments. The Engine works on
# Sample-row-shaped objects today; for HF-ref attachments we synthesize
# in-memory equivalents on demand. No DB rows; bytes (image/depth)
# stream from HF via the same `_pointer_gt_resolver` path.
# ---------------------------------------------------------------------------


class _VirtualCustomField:
    """Quack-like-CustomField for HF-ref iteration. The Engine's
    `_load_gt_array` + `pointer_resolver` paths inspect: name,
    field_type, value_text, value_float, source_column."""
    __slots__ = (
        'name', 'field_type', 'value_text', 'value_float',
        'source_column', 'sample_id', 'submission_id',
    )

    def __init__(self, name, field_type, value_text=None,
                 value_float=None, source_column=None):
        self.name = name
        self.field_type = field_type
        self.value_text = value_text
        self.value_float = value_float
        self.source_column = source_column
        self.sample_id = None
        self.submission_id = None


class _VirtualSample:
    """Quack-like-Sample for HF-ref iteration."""
    __slots__ = (
        'id', 'name', 'tags', 'custom_fields', 'source_ref_json',
        'histogram_data', 'dataset',
    )

    def __init__(self, name, source_ref_json, custom_fields, tags=''):
        self.id = None
        self.name = name
        self.tags = tags
        self.custom_fields = custom_fields
        self.source_ref_json = source_ref_json
        self.histogram_data = None
        # Set by the iterator so `pointer_resolver` can find the
        # attachment's owner for HF-token resolution.
        self.dataset = None


def _virtual_sample_from_hf_row(att, row, row_idx, classlabel_names):
    """Build (_VirtualSample, list_of_inline_field_descriptions) from
    one streamed HF row + the attachment's mapping. Handles the same
    field-kind dispatch the importer used to do; image/depth columns
    become pointer-only fields (engine streams via bench_cache)."""
    sample_name = f"s_{row_idx:06d}"
    cfs = []
    tags_list = []
    try:
        mapping = json.loads(att.hf_mapping_json or '[]')
    except (TypeError, ValueError):
        mapping = []
    for m in mapping:
        col = m.get('column')
        kind = m.get('target_kind')
        target_field = m.get('target_field') or col
        if not col or kind == 'skip' or col not in row:
            continue
        value = row[col]
        if kind in ('image', 'mask', 'depth', 'audio', 'histogram') and value is not None:
            cfs.append(_VirtualCustomField(
                name=target_field, source_column=col,
                field_type=kind,
            ))
        elif kind == 'json' and value is not None:
            # Serialize arbitrary structured values (dicts, lists,
            # bboxes, span offsets) so they can land in CustomField.
            try:
                value_text = json.dumps(value, default=str)
            except Exception:
                value_text = str(value)
            cfs.append(_VirtualCustomField(
                name=target_field, field_type='json',
                value_text=value_text, source_column=col,
            ))
        elif kind in ('scalar', 'metric') and value is not None:
            names = classlabel_names.get(col)
            try:
                int_val = int(value)
            except (TypeError, ValueError):
                int_val = None
            if (names is not None and int_val is not None
                    and 0 <= int_val < len(names)):
                class_name = str(names[int_val])
                cfs.append(_VirtualCustomField(
                    name=target_field, field_type='scalar',
                    value_float=float(int_val), source_column=col,
                ))
                if class_name != str(int_val):
                    cfs.append(_VirtualCustomField(
                        name=f"{target_field}_class", field_type='text',
                        value_text=class_name,
                    ))
                tags_list.append(class_name)
            else:
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    fv = None
                if fv is not None:
                    cfs.append(_VirtualCustomField(
                        name=target_field, field_type='scalar',
                        value_float=fv, source_column=col,
                    ))
                elif isinstance(value, str) and value.strip():
                    # Scalar mapping but the value is a plain string
                    # (e.g. sentiment 'neg'/'pos', species 'cat'/'dog'
                    # without a ClassLabel feature). Store as text so
                    # the proposer's text-kind dispatch picks it up
                    # with an exact-match metric.
                    cfs.append(_VirtualCustomField(
                        name=target_field, field_type='text',
                        value_text=value.strip(), source_column=col,
                    ))
        elif kind == 'text' and value is not None:
            # Lists/tuples (Sequence:string columns from PWC datasets
            # like QA answers / code tests) → join with ' | ' so the
            # comparison view shows the whole thing.
            if isinstance(value, (list, tuple)):
                value_text = ' | '.join(str(x) for x in value if x is not None)
            elif isinstance(value, dict):
                # Translation features land here as {'en': '...', 'de': '...'}.
                value_text = ' | '.join(f"{k}: {v}" for k, v in value.items())
            else:
                value_text = str(value)
            cfs.append(_VirtualCustomField(
                name=target_field, field_type='text',
                value_text=value_text, source_column=col,
            ))
    return _VirtualSample(
        name=sample_name,
        source_ref_json=json.dumps({
            'repo_id': att.hf_repo_id,
            'revision': att.hf_revision,
            'split': att.hf_split or 'train',
            'row_idx': row_idx,
        }),
        custom_fields=cfs,
        tags=','.join(tags_list) if tags_list else '',
    )


# Preference order for which HF split to stream when populating GT.
# Test sets are the cleanest reference for benchmarking — published
# numbers usually report against test. Validation is the runner-up
# (held out, but sometimes used during development). Train is the
# fallback only when neither is available, since it's contaminated
# with anything a model was trained on. The stored `att.hf_split` is
# treated as a hint, not a hard preference, because PWC bulk imports
# default every attachment to 'train' regardless of what's actually
# present in the source repo.
_HF_SPLIT_PREFERENCE = ['test', 'validation', 'val', 'dev', 'train']


def _resolve_hf_split_and_load(att, load_fn, *, on_log=None):
    """Try splits in `_HF_SPLIT_PREFERENCE` order, with `att.hf_split`
    moved to the front when it's already in that list (preserves an
    explicit user override).

    Smart-skip: probe row 0 of each candidate split and verify that the
    columns mapped as GT (target_kind != 'skip') actually carry non-
    null values. A split that loads but has only inputs (e.g. a
    held-out test split for a contest where labels are withheld) is
    skipped in favor of the next preference, so we don't end up
    benchmarking against missing GT.

    Returns the first split that loads AND has GT, or — failing that —
    the first split that just loads (better SOME data than none).
    Auth / gated errors propagate."""
    def _log(msg):
        if on_log:
            on_log(msg)
    try:
        mapping = json.loads(att.hf_mapping_json or '[]')
    except (TypeError, ValueError):
        mapping = []
    gt_cols = [
        m.get('column') for m in mapping
        if m.get('column') and m.get('target_kind') not in (None, '', 'skip')
    ]
    order = list(_HF_SPLIT_PREFERENCE)
    hint = (att.hf_split or '').strip()
    if hint and hint in order:
        order.remove(hint)
        order.insert(0, hint)
    elif hint:
        order.insert(0, hint)
    last_error = None
    fallback_loadable_ds = None
    fallback_split = None
    for split in order:
        if not split:
            continue
        try:
            ds = load_fn(split)
        except ValueError as e:
            last_error = e
            msg = str(e)
            import re as _re
            m = _re.search(r"Available splits:\s*\[(.*?)\]", msg)
            if m:
                for s in m.group(1).split(','):
                    s = s.strip(" '\"")
                    if s and s not in order:
                        order.append(s)
            continue
        except Exception as e:
            low = str(e).lower()
            if ('401' in str(e) or 'gated' in low or 'authenticated' in low
                    or 'restricted' in low or 'access denied' in low):
                raise
            last_error = e
            continue
        # Loaded. If we have GT cols to check, peek at row 0.
        if gt_cols:
            try:
                row0 = next(iter(ds))
                missing = [
                    c for c in gt_cols
                    if row0.get(c) in (None, '', [], {})
                ]
                if missing:
                    _log(
                        f"split={split!r} on {att.hf_repo_id} lacks GT in "
                        f"columns {missing}; trying next split"
                    )
                    # Keep it as a last-resort fallback, but keep searching.
                    if fallback_loadable_ds is None:
                        fallback_loadable_ds = load_fn(split)
                        fallback_split = split
                    continue
            except StopIteration:
                _log(f"split={split!r} on {att.hf_repo_id} is empty; trying next")
                continue
            except Exception as e:
                _log(f"split={split!r} probe err: {e}; using it anyway")
                ds = load_fn(split)
        _log(f"using split={split!r} for {att.hf_repo_id}")
        return ds
    if fallback_loadable_ds is not None:
        _log(
            f"falling back to split={fallback_split!r} on {att.hf_repo_id} "
            f"(no preferred split had full GT)"
        )
        return fallback_loadable_ds
    _log(f"all-split load failed for {att.hf_repo_id}: {last_error}")
    return None


def _iter_hf_attachment_samples(att, *, hf_token=None, cap=None):
    """Stream the HF dataset behind `att` and yield _VirtualSample
    objects, one per row, capped at `cap` (or the attachment's
    own cap, or HF_DEFAULT_SAMPLE_CAP)."""
    cap = cap or att.hf_sample_cap or HF_DEFAULT_SAMPLE_CAP
    try:
        from datasets import load_dataset
    except ImportError:
        return
    def _load(split):
        return load_dataset(
            att.hf_repo_id, split=split,
            streaming=True, revision=att.hf_revision,
            token=hf_token, trust_remote_code=True,
        )
    ds = _resolve_hf_split_and_load(att, _load,
                                    on_log=lambda m: print(f"_iter_hf_attachment_samples: {m}"))
    if ds is None:
        return

    classlabel_names = {}
    try:
        feats = getattr(ds, 'features', None) or {}
        for col, feat in (feats.items() if hasattr(feats, 'items') else []):
            names = getattr(feat, 'names', None)
            if not names:
                inner = getattr(feat, 'feature', None)
                if inner is not None:
                    names = getattr(inner, 'names', None)
            if names:
                classlabel_names[col] = list(names)
    except Exception:
        pass

    for i, row in enumerate(ds):
        if i >= cap:
            break
        yield _virtual_sample_from_hf_row(att, row, i, classlabel_names)


def _iter_lb_eval_samples(lb, *, hf_token=None, hf_cap=None):
    """Yield (sample_handle, attachment) tuples for every sample the
    LB evaluates against. Two source shapes:
      - Attachment rows (new): BH-attached → real Samples; HF-attached →
        streamed _VirtualSample objects.
      - Legacy m2m `lb.datasets` (BH-only LBs created before the
        Attachment model existed): real Sample rows.

    Datasets already represented by an Attachment row are skipped in
    the legacy pass so we don't double-iterate when both wirings
    co-exist on the same LB.

    `hf_cap` overrides each HF attachment's default sample cap. The
    eval pipeline derives a tight cap from the actual submission
    prediction index range (see `_discover_submission_pred_indices`)
    so a submission with predictions for samples 0..187 doesn't
    trigger 10K rows of HF streaming for nothing.
    """
    covered_dataset_ids = set()
    for att in lb.attachments:
        if att.role != 'primary':
            continue
        if att.kind == 'bh':
            if att.dataset is None:
                continue
            covered_dataset_ids.add(att.dataset.id)
            for s in att.dataset.samples:
                yield s, att
        else:
            for vs in _iter_hf_attachment_samples(
                att, hf_token=hf_token, cap=hf_cap,
            ):
                yield vs, att

    # Legacy m2m path: walk `lb.datasets` for any dataset not already
    # surfaced via an Attachment. _auto_create_lb_with_metrics still
    # uses lb.datasets.append(ds) without creating an Attachment row,
    # so existing test fixtures + older LB rows depend on this branch.
    for ds in (lb.datasets or []):
        if ds.id in covered_dataset_ids:
            continue
        for s in (ds.samples or []):
            yield s, None


def _discover_submission_pred_indices(submission_id, pred_field_names):
    """Walk `uploads/submissions/<id>/<field>/` for each `<field>` in
    `pred_field_names` and return the set of sample indices the
    submission has predictions for (sample names follow the
    `_VirtualSample` `s_NNNNNN` convention; non-matching names are
    ignored).

    Used by the eval pipeline to bound HF-attached iteration to the
    actual prediction range — a submission with 8 preds at indices
    up to 187 streams 188 HF rows, not the 10K default cap.

    Returns an empty set if no pred files are on disk (e.g. for legacy
    BH LBs where samples are addressed by name, not index — the
    iteration won't be capped, which matches prior behavior)."""
    indices = set()
    if not pred_field_names:
        return indices
    folder = os.path.join(
        app.config['UPLOAD_FOLDER'], 'submissions', str(submission_id),
    )
    if not os.path.isdir(folder):
        return indices
    pat = re.compile(r'^s_(\d+)\.[^.]+$')
    for field in pred_field_names:
        d = os.path.join(folder, field)
        if not os.path.isdir(d):
            continue
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for entry in entries:
            m = pat.match(entry)
            if m:
                try:
                    indices.add(int(m.group(1)))
                except ValueError:
                    pass
    return indices


def _make_paired_gt_provider(lb):
    """Closure factory: returns a `(primary_sample) -> iterable` that
    yields (CustomField, sample) tuples from gt_source datasets
    attached to `lb`. The yielded fields fold into get_metric_context
    as additional gt_<col> entries.

    Cheap to call — datasets-per-LB is tiny and the gt_source
    attachment lookup is O(N) on a typical 1-3-dataset LB."""
    if lb is None:
        return None
    gt_sources = _gt_source_datasets_for_lb(lb)
    if not gt_sources:
        return None

    def _provider(primary_sample):
        if primary_sample is None:
            return
        for ds in gt_sources:
            partner = Sample.query.filter_by(
                dataset_id=ds.id, name=primary_sample.name,
            ).first()
            if partner is None:
                continue
            for cf in (partner.custom_fields or []):
                if cf.field_type in ('scalar', 'image', 'depth', 'text'):
                    yield cf, partner

    return _provider


def _pointer_gt_resolver(sample, cf):
    """Bench-cache-backed resolver passed to `get_metric_context` so
    pointer-mode samples can serve image / depth GT to metrics
    on-demand. Returns a numpy array on success, None on any failure.

    Cache key shape:
        gt:<repo_id>@<revision>:<split>:<row_idx>:<source_column>

    Submissions evict before GT (see bench_cache.cache_gc) so the
    GT side stays warm across re-evals."""
    if not (sample is not None and getattr(sample, 'source_ref_json', None)
            and getattr(cf, 'source_column', None)):
        return None
    try:
        ref = json.loads(sample.source_ref_json)
    except (TypeError, ValueError):
        return None
    repo_id = ref.get('repo_id')
    if not repo_id:
        return None
    revision = ref.get('revision') or 'main'
    split = ref.get('split') or 'train'
    try:
        row_idx = int(ref.get('row_idx'))
    except (TypeError, ValueError):
        return None
    cache_key = f"gt:{repo_id}@{revision}:{split}:{row_idx}:{cf.source_column}"
    cache_root = app.config.get('CACHE_FOLDER')

    # Pull the user's HF token if the dataset's owner has one saved —
    # gated repos otherwise 401 here at metric-eval time.
    hf_token = None
    try:
        owner = sample.dataset.owner if sample.dataset else None
        hf_token = getattr(owner, 'hf_token', None)
    except Exception:
        pass

    def _writer(path):
        from datasets import load_dataset
        ds = load_dataset(
            repo_id, split=split, streaming=True,
            revision=ref.get('revision'),  # keep None if unpinned at import
            token=hf_token, trust_remote_code=True,
        )
        # `skip(N).take(1)` is the streaming-friendly shape for HF
        # datasets; iterating + breaking works equally and avoids
        # methods that only exist on IterableDataset in some versions.
        target = None
        for i, row in enumerate(ds):
            if i == row_idx:
                target = row
                break
        if target is None:
            raise RuntimeError(
                f"row {row_idx} unreachable in {repo_id}@{revision}/{split}"
            )
        value = target.get(cf.source_column)
        if value is None:
            raise RuntimeError(
                f"column {cf.source_column!r} missing in row {row_idx}"
            )
        # `np.savez` and `np.save` auto-append a file extension when
        # the path doesn't already have one — which collides with
        # bench_cache's hashed (extension-less) filename. Stage to a
        # BytesIO and dump the buffer to the exact path we were given.
        import io as _io
        if cf.field_type == 'image':
            from PIL import Image as _PILImage
            buf = _io.BytesIO()
            if hasattr(value, 'mode') and hasattr(value, 'save'):
                value.convert('RGB').save(buf, 'PNG')
            else:
                arr = np.asarray(value)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                _PILImage.fromarray(arr.astype(np.uint8)).save(buf, 'PNG')
            with open(path, 'wb') as f:
                f.write(buf.getvalue())
        elif cf.field_type == 'depth':
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            buf = _io.BytesIO()
            np.savez(buf, depth=arr.astype(np.float32))
            with open(path, 'wb') as f:
                f.write(buf.getvalue())
        else:
            # Fallback: serialize via numpy savez under key 'value'.
            buf = _io.BytesIO()
            np.savez(buf, value=np.asarray(value))
            with open(path, 'wb') as f:
                f.write(buf.getvalue())

    try:
        cached_path = bench_cache.cache_put(
            db.session, CacheEntry,
            cache_root=cache_root, key=cache_key,
            origin='gt', writer=_writer,
        )
    except Exception as e:
        print(f"DEBUG: pointer fetch failed for {cache_key}: {e}")
        return None

    # Read back via the existing in-memory loader. The bench_cache
    # filename is sha256 of the key (no extension); both PIL and
    # np.load handle that fine.
    try:
        if cf.field_type == 'image':
            from PIL import Image as _PILImage
            return np.asarray(_PILImage.open(cached_path).convert('RGB'))
        if cf.field_type == 'depth':
            with np.load(cached_path) as data:
                if 'depth' in data:
                    return np.asarray(data['depth'])
                first = next(iter(data.keys()))
                return np.asarray(data[first])
    except Exception as e:
        print(f"DEBUG: pointer read-back failed for {cache_key}: {e}")
    return None


def _import_hf_auto(repo_id, dataset_name, mapping, *, sample_cap=200,
                   split='train', revision=None, hf_token=None,
                   owner_user_id=None, features=None):
    """Stream up to `sample_cap` rows from the HF dataset, lay them out
    as BenchHub folders per the mapping, and feed the result through the
    existing process_dataset_zip pipeline by re-zipping the folder.

    `mapping` is a list of {column, target_kind, target_field}. Anything
    with target_kind='skip' is ignored. Returns (success, message, ds_id).

    `features` (optional) is the normalized HF feature schema. When
    provided and a column is a ClassLabel, the importer also writes a
    text column with the human-readable class name and a per-sample
    tag (in tags/<sample>.txt) so users can filter by class downstream.
    """
    # Index ClassLabel columns once so the per-row loop is cheap.
    classlabel_names = {}  # col -> [name0, name1, ...]
    for col, desc in (features or {}).items():
        if isinstance(desc, dict) and desc.get('type') == 'ClassLabel':
            names = desc.get('names') or []
            if isinstance(names, list) and names:
                classlabel_names[col] = names
    # `metric` and `scalar` use identical write logic; the only difference
    # is whether the on-disk folder name carries the `metric_` prefix.
    # Pre-create folders for both kinds in the loop below.
    _metric_like = {'metric', 'scalar'}
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
            if kind in ('image', 'depth', 'histogram', 'metric',
                        'scalar', 'text'):
                os.makedirs(os.path.join(work_dir, m['target_field']), exist_ok=True)
        # README at root keeps process_dataset_zip from picking the only
        # populated subfolder as the dataset wrapper.
        with open(os.path.join(work_dir, 'README.md'), 'w') as f:
            f.write(
                f"# {dataset_name}\n\nAuto-imported from HuggingFace `{repo_id}` "
                f"(first {sample_cap} samples, revision={revision or 'main'}).\n"
            )

        ds = load_dataset(repo_id, split=split, streaming=True, revision=revision,
                          token=hf_token, trust_remote_code=True)

        # Fall back to features inspected from the streamed dataset itself.
        # The HF API features blob doesn't always include ClassLabel.names
        # for modern parquet datasets, but `datasets` lib resolves them
        # from the dataset_info.json on the repo and exposes them on
        # ds.features as ClassLabel objects.
        try:
            ds_feats = getattr(ds, 'features', None) or {}
            for col_name, feat in (ds_feats.items() if hasattr(ds_feats, 'items') else []):
                if col_name in classlabel_names:
                    continue
                names = getattr(feat, 'names', None)
                # Some HF feature types (e.g. Sequence(ClassLabel)) wrap the
                # ClassLabel — peek through one level of wrapper.
                if not names:
                    inner = getattr(feat, 'feature', None)
                    if inner is not None:
                        names = getattr(inner, 'names', None)
                if names:
                    classlabel_names[col_name] = list(names)
        except Exception as e:
            print(f"ClassLabel name extraction from ds.features failed: {e}")

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
                    elif kind in _metric_like and value is not None:
                        # ClassLabel columns: store the class as an int
                        # (1 not 1.0), plus a parallel text column
                        # carrying the class NAME, and a per-sample tag
                        # in tags/ so users can filter the dataset by
                        # class downstream.
                        #
                        # `metric` vs `scalar`: identical write logic.
                        # The folder-name prefix difference (metric_<col>
                        # vs <col>) is encoded in m['target_field'] by
                        # the upstream mapping step.
                        names = classlabel_names.get(col)
                        try:
                            int_val = int(value)
                        except (TypeError, ValueError):
                            int_val = None
                        if names is not None and int_val is not None and 0 <= int_val < len(names):
                            class_name = str(names[int_val])
                            with open(os.path.join(target_dir, f"{sample_id}.txt"), 'w') as f:
                                f.write(str(int_val))
                            # If the names list is just stringified indices
                            # (e.g. ['0','1','2',...]) the side fields would
                            # duplicate the metric column with the same digit
                            # everywhere — skip them, but still emit a tag so
                            # the user can filter by class.
                            redundant = (class_name == str(int_val))
                            if not redundant:
                                class_dir = os.path.join(work_dir, f"{col}_class")
                                os.makedirs(class_dir, exist_ok=True)
                                with open(os.path.join(class_dir, f"{sample_id}.txt"), 'w') as f:
                                    f.write(class_name)
                            tags_dir = os.path.join(work_dir, 'tags')
                            os.makedirs(tags_dir, exist_ok=True)
                            tag_path = os.path.join(tags_dir, f"{sample_id}.txt")
                            existing = ''
                            if os.path.exists(tag_path):
                                with open(tag_path) as f:
                                    existing = f.read().strip()
                            with open(tag_path, 'w') as f:
                                merged = ','.join(
                                    [t for t in (existing.split(',') if existing else []) if t]
                                    + [class_name]
                                )
                                f.write(merged)
                        else:
                            # Regular numeric metric. Round-trip ints as ints.
                            try:
                                fv = float(value)
                                if fv.is_integer():
                                    out_str = str(int(fv))
                                else:
                                    out_str = str(fv)
                            except (TypeError, ValueError):
                                out_str = str(value)
                            with open(os.path.join(target_dir, f"{sample_id}.txt"), 'w') as f:
                                f.write(out_str)
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
                # Auto-tag from HF metadata + LLM (when configured).
                # Best-effort; the dataset import is the primary success.
                try:
                    auto_tags = _auto_tags_for_hf(
                        repo_id, hf_token=hf_token, revision=revision,
                    )
                    if auto_tags:
                        ds.tags = _resolve_tags(', '.join(auto_tags))
                except Exception as e:
                    print(f"auto-tag attach failed: {e}")
                db.session.commit()
        return success, message, ds_id
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.route('/create_lb', methods=['GET'])
@login_required
def create_lb_chooser():
    """Unified entry for "start a new leaderboard". The page presents
    two side-by-side flows:
      - From a BenchHub dataset: pick from your uploaded ZIPs →
        configure name + auto-assign → POST to /create_leaderboard.
      - From a HuggingFace dataset: paste/pick a repo →
        POST to /import_from_hf/preview → schema mapping → auto-LB.

    No Dataset row is ever materialized for HF — the LB attaches to
    the HF repo directly.
    """
    user = g.current_user
    bh_datasets = (
        Dataset.query
        .filter(visible_in_list(Dataset, user))
        .order_by(Dataset.upload_date.desc())
        .all()
    )
    recent_hf = _user_recent_hf_visits(user.id if user else None, limit=8)
    trending_hf = _trending_hf_visits(days=7, limit=8)
    return render_template(
        'create_lb_chooser.html',
        bh_datasets=bh_datasets,
        recent_hf=recent_hf,
        trending_hf=trending_hf,
    )


@app.route('/create_lb/from_hf', methods=['GET'])
@login_required
def create_lb_from_hf():
    # Back-compat: old direct link redirects into the unified chooser.
    return redirect(url_for('create_lb_chooser'))


@app.route('/import_from_hf/preview', methods=['POST'])
@login_required
def import_from_hf_preview():
    """Step 1 of the auto-import flow: paste repo, get back the inferred
    mapping for review. Renders a confirmation page.
    """
    repo_id = (request.form.get('hf_repo_id') or '').strip()
    revision = (request.form.get('hf_revision') or '').strip() or None
    hf_token = _resolve_hf_token(request.form.get('hf_token'))
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
        low = msg.lower()
        if ('401' in msg or 'gated' in low or 'authenticated' in low
                or 'restricted' in low or 'access denied' in low):
            return redirect(url_for(
                'import_from_hf_gated_wizard',
                repo_id=repo_id, dataset_name=dataset_name,
                revision=revision or '', flow='auto',
            ))
        flash(f"Couldn't read schema: {e}", "danger")
        return redirect(url_for('datasets_list'))

    if not features:
        flash(
            f"No `features` schema found in {repo_id}. The dataset may not "
            "be parquet-formatted; try the manual import path instead.",
            "warning",
        )
        return redirect(url_for('datasets_list'))

    # Try LLM-driven inference first; fall back to the rule-based
    # heuristic if the API key isn't set, the call fails, or the
    # response doesn't parse.
    llm_mapping = _llm_infer_mapping(features, dataset_repo=repo_id)
    if llm_mapping:
        mapping = llm_mapping
        inference_source = 'llm'
    else:
        mapping = _infer_mapping(features)
        inference_source = 'rules'

    return render_template(
        'hf_import_preview.html',
        repo_id=repo_id,
        revision=revision,
        hf_token=hf_token,
        dataset_name=dataset_name,
        features=features,
        mapping=mapping,
        inference_source=inference_source,
    )


@app.route('/import_from_hf/auto', methods=['POST'])
@login_required
def import_from_hf_auto():
    """Step 2: form has the (potentially edited) mapping; run the import."""
    repo_id = (request.form.get('hf_repo_id') or '').strip()
    revision = (request.form.get('hf_revision') or '').strip() or None
    hf_token = _resolve_hf_token(request.form.get('hf_token'))
    dataset_name = (request.form.get('dataset_name') or '').strip()

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

    # Re-fetch features so _import_hf_auto can do ClassLabel-aware
    # writes (parallel class-name text column + per-sample tags).
    # Cheap call, same JSON we used in the preview step.
    try:
        features = _hf_fetch_features(repo_id, revision=revision, hf_token=hf_token)
    except Exception:
        features = {}

    # HF datasets no longer get a Dataset row. The "Confirm & import"
    # button on the mapping-preview page lands here; we go STRAIGHT to
    # the auto-LB preview, with the LB's primary attachment configured
    # as an HF ref (repo_id + revision + split + mapping). The actual
    # LB row is created when the user confirms on the auto-LB preview
    # via /create_leaderboard/auto_finalize.
    try:
        metric_props, viz_props = _collect_auto_lb_proposals_for_hf_ref(
            repo_id, mapping, revision=revision, split='train',
            hf_token=hf_token,
        )
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if ('401' in msg or 'gated' in low or 'authenticated' in low
                or 'restricted' in low or 'access denied' in low):
            return redirect(url_for(
                'import_from_hf_gated_wizard',
                repo_id=repo_id, dataset_name=dataset_name,
                revision=revision or '', flow='auto',
            ))
        flash(f"HuggingFace stream failed: {e}", "danger")
        return redirect(url_for('datasets_list'))

    if not metric_props and not viz_props:
        # Don't bounce the user back to /datasets — they came here to
        # build a leaderboard, and the preview page now has explicit
        # "Add a metric" + "Required pred fields" affordances that
        # work without auto-proposals. Just flash a notice and let
        # them continue.
        flash(
            "No metrics were auto-proposed for this HF dataset "
            "(BenchHub looks for scalar / image / depth GT columns). "
            "Use the Add a metric / Required prediction fields "
            "sections below to attach what you want.",
            "info",
        )

    seen_pred = {}
    for p in metric_props + viz_props:
        for pf in (p.get('pred_fields') or []):
            seen_pred.setdefault(pf['name'], pf)

    # Canonicality nudge: if an admin-promoted public LB already exists
    # for this repo, surface a "submit there instead" callout. Doesn't
    # block creation — users can still fork into a personal LB.
    canonical_existing = (
        Leaderboard.query
        .filter(Leaderboard.canonical_for_repo == repo_id,
                Leaderboard.canonicality == 'public')
        .first()
    )

    return render_template(
        'auto_lb_preview.html',
        leaderboard_name=f"{dataset_name}_leaderboard",
        # HF-ref flow (no DB Dataset row); template branches on this.
        hf_ref={
            'repo_id': repo_id, 'revision': revision or '',
            'split': 'train', 'mapping_json': json.dumps(mapping),
        },
        dataset=None,
        metric_proposals=metric_props,
        viz_proposals=viz_props,
        pred_field_schema=list(seen_pred.values()),
        canonical_existing=canonical_existing,
    )


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
def api_hf_datasets():
    """JSON listing of HuggingFace datasets, used by the picker on /datasets.

    Query string: ?sort=likes|downloads|trending  ?q=<keyword>

    Public on purpose: this endpoint just relays huggingface.co's
    public dataset index (server-side cached so we stay under their
    anonymous rate limit). No user-specific data, no quota, no auth.
    Was @login_required by accident — the JS picker breaks on any
    non-JSON response, so a session-expired user would see
    "Failed to load" instead of the dataset list.
    """
    sort = request.args.get('sort', 'likes')
    if sort not in ('likes', 'downloads', 'trending'):
        sort = 'likes'
    q = (request.args.get('q') or '').strip()
    rows = _fetch_hf_datasets(sort=sort, q=q)
    return jsonify({'rows': rows, 'sort': sort, 'q': q})


def _record_hf_visit(user_id, repo_id):
    """Upsert one row of HfDatasetVisit. No-op if user_id is None
    (anonymous browsing — we don't track those)."""
    if not user_id or not repo_id:
        return
    visit = HfDatasetVisit.query.filter_by(
        user_id=user_id, repo_id=repo_id,
    ).first()
    now = datetime.utcnow()
    if visit is None:
        visit = HfDatasetVisit(
            user_id=user_id, repo_id=repo_id,
            last_visited_at=now, visit_count=1,
        )
        db.session.add(visit)
    else:
        visit.last_visited_at = now
        visit.visit_count = (visit.visit_count or 0) + 1
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()


# `/hf/<repo_id>` live-preview surface was removed in the attachment
# refactor (2026-05-08). HF datasets no longer have a BH representation;
# users browse on huggingface.co directly and create LBs via the
# /datasets HF picker → /import_from_hf/preview → auto-LB flow.



def _user_recent_hf_visits(user_id, *, limit=10):
    """Last `limit` HF repos this user has explored, newest first."""
    if not user_id:
        return []
    return (HfDatasetVisit.query
            .filter_by(user_id=user_id)
            .order_by(HfDatasetVisit.last_visited_at.desc())
            .limit(limit)
            .all())


def _trending_hf_visits(*, days=7, limit=10):
    """Most-visited HF repos across the platform in the last `days`.
    Returns list of (repo_id, total_visits, distinct_users)."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (db.session.query(
                HfDatasetVisit.repo_id,
                func.sum(HfDatasetVisit.visit_count).label('total'),
                func.count(HfDatasetVisit.user_id.distinct()).label('users'),
            )
            .filter(HfDatasetVisit.last_visited_at >= cutoff)
            .group_by(HfDatasetVisit.repo_id)
            .order_by(func.sum(HfDatasetVisit.visit_count).desc())
            .limit(limit)
            .all())
    return [
        {'repo_id': r.repo_id, 'visits': int(r.total or 0),
         'users': int(r.users or 0)}
        for r in rows
    ]


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

    # When the user opts into auto-assigned metrics, render the preview
    # page so they can review/edit/skip individual proposals before any
    # GlobalMetric / GlobalVisualization rows are created. The actual
    # commit happens in /create_leaderboard/auto_finalize.
    auto_assign = bool(request.form.get('auto_assign_metrics'))
    if auto_assign:
        dataset_ids = request.form.getlist('dataset_ids')
        if not dataset_ids and 'dataset_id' in request.form:
            dataset_ids = [request.form['dataset_id']]
        ds = (Dataset.query.filter(Dataset.id.in_(dataset_ids)).first()
              if dataset_ids else None)
        if ds is None:
            flash("Auto-assign metrics needs a dataset attached to the LB.", "danger")
            return redirect(url_for('datasets_list'))
        metric_props, viz_props = _collect_auto_lb_proposals(ds)
        if not metric_props and not viz_props:
            flash("No GT scalar fields detected on the dataset — "
                  "nothing to auto-attach.", "warning")
            return redirect(url_for('dataset_view', dataset_id=ds.id))
        # Combined pred-field schema preview so the user sees what
        # submission folders the proposed metrics + viz will require.
        seen_pred = {}
        for p in metric_props + viz_props:
            for pf in (p.get('pred_fields') or []):
                seen_pred.setdefault(pf['name'], pf)
        return render_template(
            'auto_lb_preview.html',
            leaderboard_name=leaderboard_name,
            dataset=ds,
            metric_proposals=metric_props,
            viz_proposals=viz_props,
            pred_field_schema=list(seen_pred.values()),
        )

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
    return redirect(url_for('leaderboard_view', leaderboard_id=new_leaderboard.id))


def _peek_hf_first_virtual_sample(repo_id, mapping, *, revision=None,
                                  split='train', hf_token=None):
    """Stream row 0 of an HF dataset and return a _VirtualSample
    built from the supplied mapping. Used to drive metric/viz
    proposers off an HF ref without materializing a Dataset row."""
    fake_att = type('A', (), {
        'hf_repo_id': repo_id, 'hf_revision': revision,
        'hf_split': split or 'train',
        'hf_mapping_json': json.dumps(mapping or []),
    })()
    for vs in _iter_hf_attachment_samples(fake_att, hf_token=hf_token, cap=1):
        return vs
    return None


def _collect_auto_lb_proposals_for_hf_ref(repo_id, mapping, *, revision=None,
                                          split='train', hf_token=None):
    """Same shape as `_collect_auto_lb_proposals(dataset)` — returns
    (metric_proposals, viz_proposals) — but driven by a peek of the
    HF stream rather than a materialized Dataset. Wraps the virtual
    sample in a single-sample shim so the existing proposers see
    a Dataset-like object."""
    vs = _peek_hf_first_virtual_sample(
        repo_id, mapping, revision=revision, split=split, hf_token=hf_token,
    )
    if vs is None:
        return [], []
    fake_ds = type('FakeDS', (), {'samples': [vs], 'id': None,
                                  'name': repo_id})()
    metric_props = [
        _enrich_metric_proposal_with_existing_code(dict(p))
        for p in _propose_metrics_for_dataset(fake_ds)
    ]
    viz_props = [
        _enrich_viz_proposal_with_existing_code(dict(p))
        for p in _propose_visualizations_for_dataset(fake_ds)
    ]
    return metric_props, viz_props


def _create_lb_with_hf_attachment(lb_name, *, owner_user_id, repo_id,
                                  mapping, revision=None, split='train',
                                  metric_proposals=None, viz_proposals=None,
                                  hf_sample_cap=None):
    """Create a Leaderboard whose primary attachment is an HF ref
    (NOT a BH Dataset). Persists LB + Attachment + the supplied
    metric/viz proposals in one transaction. Returns
    (success, message, lb_id)."""
    if not lb_name:
        return False, "Leaderboard name is required.", None
    if Leaderboard.query.filter_by(name=lb_name).first():
        return False, f'A leaderboard named "{lb_name}" already exists.', None
    metric_proposals = metric_proposals or []
    viz_proposals = viz_proposals or []
    if not metric_proposals and not viz_proposals:
        return False, ("No GT scalar / image / depth fields detected on "
                       "the HF dataset — nothing to auto-attach. Adjust "
                       "the mapping or add metrics manually."), None

    lb = Leaderboard(
        name=lb_name,
        summary_metrics=','.join(p['target_name'] for p in metric_proposals),
        owner_user_id=owner_user_id,
    )
    db.session.add(lb)
    db.session.flush()
    db.session.add(Attachment(
        leaderboard_id=lb.id,
        hf_repo_id=repo_id, hf_revision=revision,
        hf_split=split or 'train',
        hf_mapping_json=json.dumps(mapping or []),
        role='primary',
        hf_sample_cap=hf_sample_cap,
    ))
    db.session.flush()

    for p in metric_proposals:
        gm = GlobalMetric.query.filter_by(name=p['global_name']).first()
        if gm is None:
            code = (p.get('python_code')
                    or _llm_generate_metric_code(p['global_name'], p['llm_hint'])
                    or p['fallback_code'])
            gm = GlobalMetric(
                name=p['global_name'], description=p['description'],
                python_code=code, is_aggregated=False,
                accepts_aggregated_inputs=False,
                owner_user_id=owner_user_id, visibility='public',
            )
            db.session.add(gm); db.session.flush()
        db.session.add(LeaderboardMetric(
            leaderboard_id=lb.id, global_metric_id=gm.id,
            arg_mappings=json.dumps(p['arg_mappings']),
            target_name=p['target_name'],
            pooling_type=p['pooling_type'],
            sort_direction=p['sort_direction'],
        ))

    for p in viz_proposals:
        gv = GlobalVisualization.query.filter_by(name=p['global_name']).first()
        if gv is None:
            code = (p.get('python_code')
                    or _llm_generate_visualization_code(
                        p['global_name'], p['llm_hint'],
                        p['is_aggregated'], p['accepts_aggregated_inputs'],
                    )
                    or p['fallback_code'])
            gv = GlobalVisualization(
                name=p['global_name'], description=p['description'],
                python_code=code,
                is_aggregated=p['is_aggregated'],
                accepts_aggregated_inputs=p['accepts_aggregated_inputs'],
                owner_user_id=owner_user_id, visibility='public',
            )
            db.session.add(gv); db.session.flush()
        db.session.add(LeaderboardVisualization(
            leaderboard_id=lb.id, global_visualization_id=gv.id,
            arg_mappings=json.dumps(p['arg_mappings']),
            target_name=p['target_name'],
        ))

    db.session.commit()
    n_m, n_v = len(metric_proposals), len(viz_proposals)
    parts = []
    if n_m: parts.append(f"{n_m} metric{'' if n_m == 1 else 's'}")
    if n_v: parts.append(f"{n_v} visualization{'' if n_v == 1 else 's'}")
    return True, (f'Created leaderboard "{lb_name}" '
                  f'(HF ref → {repo_id}) with {" and ".join(parts)}.'), lb.id


@app.route('/create_leaderboard/auto_finalize', methods=['POST'])
@login_required
def create_leaderboard_auto_finalize():
    """Consume the auto-LB preview form and persist whichever proposals
    the user kept (with edits). Two source kinds, distinguished by
    which hidden field the form carries:
      - dataset_id  → BH-dataset attachment (uploaded ZIP).
      - hf_repo_id  → HF-ref attachment (no Dataset row).
    """
    leaderboard_name = (request.form.get('leaderboard_name') or '').strip()
    if not leaderboard_name:
        flash("Leaderboard name is required.", "danger")
        return redirect(url_for('datasets_list'))

    hf_repo_id = (request.form.get('hf_repo_id') or '').strip()
    dataset_id = request.form.get('dataset_id')

    # User-added metrics from the inline "Add a metric" UI (library
    # picks + AI-authored). Each entry is a full proposal dict the
    # client edited inline before submit. Merged into kept_metrics
    # alongside the auto-proposals.
    extra_metrics = _parse_extra_metrics(
        request.form.get('extra_metrics_json') or '[]'
    )
    # Pred fields required even without a scoring metric (organizer
    # wants the raw predictions on disk for human review). Persisted
    # to Leaderboard.required_pred_fields_json so the submission-
    # contract widget surfaces them.
    extra_pred_fields = _parse_extra_pred_fields(
        request.form.get('extra_pred_fields_json') or '[]'
    )
    # Edits to the AUTO-derived pred fields (rename / change kind /
    # omit). Renames + omissions rewrite the metric proposals before
    # creation; kind overrides ride on required_pred_fields_json
    # (where _lb_submission_pred_fields merges them onto the derived
    # entry).
    auto_pred_overrides = _parse_auto_pred_fields(
        request.form.get('auto_pred_fields_json') or '[]'
    )
    pred_omits = {o['original_name'] for o in auto_pred_overrides if o['omit']}
    pred_renames = {
        o['original_name']: o['name']
        for o in auto_pred_overrides
        if not o['omit'] and o['name'] and o['name'] != o['original_name']
    }
    pred_kind_overrides = [
        # Final-name-keyed: if a field is renamed, the override targets the new name.
        {'name': o['name'], 'kind': o['kind']}
        for o in auto_pred_overrides
        if not o['omit']
    ]

    if hf_repo_id:
        # ---- HF-ref flow ----
        try:
            mapping = json.loads(request.form.get('hf_mapping_json') or '[]')
        except (TypeError, ValueError):
            mapping = []
        revision = (request.form.get('hf_revision') or '').strip() or None
        split = (request.form.get('hf_split') or 'train').strip()
        hf_token = getattr(g.current_user, 'hf_token', None)
        metric_proposals_all, viz_proposals_all = (
            _collect_auto_lb_proposals_for_hf_ref(
                hf_repo_id, mapping, revision=revision, split=split,
                hf_token=hf_token,
            )
        )
        kept_metrics = [m for m in (_override_proposal(p, 'metric')
                                    for p in metric_proposals_all) if m]
        kept_metrics.extend(extra_metrics)
        kept_viz = [v for v in (_override_proposal(p, 'viz')
                                for p in viz_proposals_all) if v]
        kept_metrics = _apply_pred_field_edits(kept_metrics, pred_omits, pred_renames)
        kept_viz = _apply_pred_field_edits(kept_viz, pred_omits, pred_renames)
        if not kept_metrics and not kept_viz:
            flash("Nothing kept from the proposal — leaderboard not created.", "warning")
            return redirect(url_for('datasets_list'))
        ok, msg, lb_id = _create_lb_with_hf_attachment(
            leaderboard_name, owner_user_id=g.current_user.id,
            repo_id=hf_repo_id, mapping=mapping,
            revision=revision, split=split,
            metric_proposals=kept_metrics, viz_proposals=kept_viz,
        )
        if ok and lb_id:
            combined_extras = _merge_pred_field_extras(
                extra_pred_fields, pred_kind_overrides,
            )
            if combined_extras:
                _persist_required_pred_fields(lb_id, combined_extras)
        flash(msg, "success" if ok else "warning")
        if ok and lb_id:
            return redirect(url_for('leaderboard_view', leaderboard_id=lb_id))
        return redirect(url_for('datasets_list'))

    # ---- BH-dataset flow (legacy uploaded-ZIP path) ----
    if not dataset_id:
        flash("Dataset id or hf_repo_id is required.", "danger")
        return redirect(url_for('datasets_list'))
    ds = Dataset.query.get(int(dataset_id))
    if ds is None:
        flash("Dataset not found.", "danger")
        return redirect(url_for('datasets_list'))

    metric_proposals_all, viz_proposals_all = _collect_auto_lb_proposals(ds)
    kept_metrics = [m for m in (_override_proposal(p, 'metric')
                                for p in metric_proposals_all) if m]
    kept_metrics.extend(extra_metrics)
    kept_viz = [v for v in (_override_proposal(p, 'viz')
                            for p in viz_proposals_all) if v]
    kept_metrics = _apply_pred_field_edits(kept_metrics, pred_omits, pred_renames)
    kept_viz = _apply_pred_field_edits(kept_viz, pred_omits, pred_renames)
    if not kept_metrics and not kept_viz:
        flash("Nothing kept from the proposal — leaderboard not created.", "warning")
        return redirect(url_for('dataset_view', dataset_id=ds.id))
    ok, msg, lb_id = _auto_create_lb_with_metrics(
        ds, leaderboard_name, owner_user_id=g.current_user.id,
        metric_proposals=kept_metrics, viz_proposals=kept_viz,
    )
    if ok and lb_id:
        combined_extras = _merge_pred_field_extras(
            extra_pred_fields, pred_kind_overrides,
        )
        if combined_extras:
            _persist_required_pred_fields(lb_id, combined_extras)
    flash(msg, "success" if ok else "warning")
    if ok and lb_id:
        return redirect(url_for('leaderboard_view', leaderboard_id=lb_id))
    return redirect(url_for('dataset_view', dataset_id=ds.id))


@app.route('/api/lb_preview/library_metrics', methods=['GET'])
@login_required
def api_lb_preview_library_metrics():
    """Return all GlobalMetric rows so the auto-LB preview's
    "Pick from library" dropdown can offer them. Read-only; cheap query."""
    rows = (GlobalMetric.query
            .order_by(GlobalMetric.is_aggregated, GlobalMetric.name)
            .all())
    return jsonify(metrics=[
        {
            'id': r.id,
            'name': r.name,
            'description': r.description,
            'python_code': r.python_code,
            'is_aggregated': bool(r.is_aggregated),
        }
        for r in rows
    ])


@app.route('/api/lb_preview/llm_metric', methods=['POST'])
@login_required
def api_lb_preview_llm_metric():
    """Author a metric on demand from a free-form description.

    Input JSON: {name?: string, description: string}
    Output JSON: full proposal dict ready for the preview to render
    (global_name, target_name, description, python_code, arg_mappings,
    sort_direction). Errors return 4xx/5xx with {error: ...}.

    Falls back to a defensive 503 if no ANTHROPIC_API_KEY is set —
    the user gets a clean error rather than a silent generic stub."""
    body = request.get_json(silent=True) or {}
    description = (body.get('description') or '').strip()
    if not description:
        return jsonify(error='description is required'), 400
    raw_name = (body.get('name') or '').strip()
    # Sanitize the user's name into a valid Python identifier; if they
    # didn't give one, hash the description for a stable global_name.
    if raw_name:
        ident = re.sub(r'[^A-Za-z0-9_]+', '_', raw_name).strip('_').lower()
    else:
        ident = ''
    if not ident or not re.match(r'^[A-Za-z_]', ident):
        import hashlib as _h
        ident = 'custom_' + _h.sha1(description.encode('utf-8')).hexdigest()[:8]

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return jsonify(error=(
            "AI authoring needs ANTHROPIC_API_KEY on the server. "
            "Either pick a metric from the library, or ask an admin "
            "to set the key."
        )), 503

    # Reuse the existing per-sample metric prompt shape — it's good
    # enough for the user-described case. Pass the description through
    # as the `llm_hint`.
    code = _llm_generate_metric_code(ident, description)
    if not code:
        return jsonify(error="Claude couldn't produce a metric for that description. Try rephrasing or be more specific about gt/pred shapes."), 502
    return jsonify(
        global_name=ident,
        target_name=raw_name or ident.replace('_', ' '),
        description=description,
        python_code=code,
        # Default arg-mappings: gt/pred. The user edits these on the
        # preview row if their dataset uses other field names.
        arg_mappings={'gt': 'gt_unknown', 'pred': 'sub_unknown_pred'},
        sort_direction='higher_is_better',
        code_source='llm',
    )


_EXTRA_PRED_KINDS = ('scalar', 'image', 'mask', 'depth')


def _merge_pred_field_extras(extra_pred_fields, kind_overrides):
    """Combine the user-added `extras` list with kind-override entries
    derived from `auto_pred_fields_json`. Overrides for fields already
    present in `extras` are folded in (kind wins); otherwise they
    become extras entries with just `{name, kind}` so the merge inside
    `_lb_submission_pred_fields` can find them. De-duped by name."""
    by_name = {e['name']: dict(e) for e in (extra_pred_fields or []) if e.get('name')}
    for ov in (kind_overrides or []):
        name = (ov.get('name') or '').strip()
        kind = ov.get('kind')
        if not name or not kind:
            continue
        if name in by_name:
            by_name[name]['kind'] = kind
        else:
            by_name[name] = {'name': name, 'kind': kind}
    return list(by_name.values())


def _persist_required_pred_fields(lb_id, fields):
    """Write the validated extra_pred_fields list onto the LB. Best
    effort: a JSON-encode failure is logged but doesn't fail the
    create call — the LB row already exists at this point."""
    if not fields:
        return
    try:
        lb = Leaderboard.query.get(lb_id)
        if lb is None:
            return
        lb.required_pred_fields_json = json.dumps(fields)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"persist required_pred_fields_json failed for lb {lb_id}: {e}")


def _parse_extra_pred_fields(raw_json):
    """Validate the `extra_pred_fields_json` payload from the auto-LB
    preview's "Required prediction fields" UI. Bad JSON / wrong shape
    → []. Each entry must have a non-empty `name` and a `kind` from
    the small allow-list; gt_field defaults to the bare name (strip
    a trailing `_pred` if present)."""
    try:
        items = json.loads(raw_json or '[]')
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    out = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get('name') or '').strip()
        if not name or name in seen:
            continue
        seen.add(name)
        kind = item.get('kind') or 'scalar'
        if kind not in _EXTRA_PRED_KINDS:
            kind = 'scalar'
        gt_field = (item.get('gt_field') or '').strip() or name.removesuffix('_pred')
        out.append({
            'name': name,
            'gt_field': gt_field,
            'kind': kind,
            'description': (item.get('description') or '').strip(),
        })
    return out


def _parse_auto_pred_fields(raw_json):
    """Validate the `auto_pred_fields_json` payload from the auto-LB
    preview's editable derived-pred-field list. Each entry overrides
    one metric-derived pred field: rename, type change, or omit.

    Schema (one entry per derived field):
      {original_name: str,  # the name at preview time; identifies the field
       name: str,           # possibly renamed
       kind: str,           # 'scalar' | 'image' | 'mask' | 'depth'
       omit: bool}          # if True, drop this field AND its metrics

    `original_name` is required (it's the identifier). `name` defaults
    to `original_name` when blank. Bad shapes are silently dropped so
    a missing JS sync doesn't 400 the form."""
    try:
        items = json.loads(raw_json or '[]')
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        orig = (item.get('original_name') or '').strip()
        if not orig:
            continue
        name = (item.get('name') or orig).strip() or orig
        # New name must be a python-identifier-ish folder name AND end
        # in `_pred` so _lb_submission_pred_fields recognizes the
        # arg_mappings value as a submission-side field. If the user
        # dropped the suffix, append it; if the name is otherwise
        # invalid, fall back to the original.
        if not name.endswith('_pred'):
            name = f'{name}_pred'
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', name):
            name = orig
        kind = item.get('kind') or 'scalar'
        if kind not in _EXTRA_PRED_KINDS:
            kind = 'scalar'
        out.append({
            'original_name': orig,
            'name': name,
            'kind': kind,
            'omit': bool(item.get('omit')),
        })
    return out


def _apply_pred_field_edits(proposals, omitted, renames):
    """Filter + rewrite metric/viz proposal dicts based on the user's
    pred-field edits.

    - Drops any proposal whose arg_mappings reference a `sub_<x>_pred`
      where `<x>_pred` is in `omitted`. Without its prediction folder
      the metric/viz can't compute, so we can't keep it.
    - Rewrites `sub_<old>_pred` → `sub_<new>_pred` for surviving
      proposals using the rename map, and updates each proposal's
      `pred_fields` list so downstream preview code sees the new name.

    `omitted` is a set of pred-field names (e.g. {'labels_pred'}).
    `renames` is a dict {old_name: new_name}."""
    out = []
    for p in proposals:
        am = dict(p.get('arg_mappings') or {})
        skip = False
        for k, v in list(am.items()):
            if (isinstance(v, str) and v.startswith('sub_')
                    and v.endswith('_pred')):
                # The pred-field name is the arg-mapping value without
                # the `sub_` prefix — it already includes the `_pred`
                # suffix (e.g. 'sub_labels_pred' → 'labels_pred').
                field = v[len('sub_'):]
                if field in omitted:
                    skip = True
                    break
                if field in renames:
                    am[k] = f'sub_{renames[field]}'
        if skip:
            continue
        p = dict(p)
        p['arg_mappings'] = am
        new_pfs = []
        for pf in (p.get('pred_fields') or []):
            pfn = (pf.get('name') if isinstance(pf, dict) else '') or ''
            if pfn in omitted:
                continue
            if pfn in renames:
                pf = dict(pf)
                pf['name'] = renames[pfn]
            new_pfs.append(pf)
        p['pred_fields'] = new_pfs
        out.append(p)
    return out


def _parse_extra_metrics(raw_json):
    """Validate + normalize the `extra_metrics_json` payload from the
    auto-LB preview's "Add a metric" UI. Defensive parsing: bad JSON
    or a wrong shape returns []. Each accepted entry must have a
    Python-identifier global_name and non-empty python_code; everything
    else has a default."""
    try:
        items = json.loads(raw_json or '[]')
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        gn = (item.get('global_name') or '').strip()
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', gn):
            continue
        code = (item.get('python_code') or '').strip()
        if f"def {gn}(" not in code:
            # Code missing or doesn't define the named function — skip
            # rather than persist a metric that can't run.
            continue
        am = item.get('arg_mappings') if isinstance(item.get('arg_mappings'), dict) else {}
        sd = item.get('sort_direction')
        if sd not in ('higher_is_better', 'lower_is_better'):
            sd = 'higher_is_better'
        out.append({
            'global_name': gn,
            'target_name': (item.get('target_name') or gn).strip(),
            'description': (item.get('description') or '').strip(),
            'python_code': code,
            'arg_mappings': am,
            'sort_direction': sd,
            'pooling_type': item.get('pooling_type') or 'mean',
            'code_source': item.get('code_source') or 'llm',
            'pred_fields': item.get('pred_fields') or [],
        })
    return out


def _override_proposal(p, prefix):
    """Apply form-supplied edits to one proposal dict. Returns the
    edited proposal, or None if the user unchecked it."""
    gn = p['global_name']
    if not request.form.get(f'kept_{prefix}_{gn}'):
        return None
    target_name = (request.form.get(f'{prefix}_target_name_{gn}') or '').strip()
    if target_name:
        p['target_name'] = target_name
    code = (request.form.get(f'{prefix}_code_{gn}') or '').strip()
    if code and f"def {gn}(" in code:
        p['python_code'] = code
    if prefix == 'metric':
        sort_dir = request.form.get(f'metric_sort_direction_{gn}')
        if sort_dir in ('higher_is_better', 'lower_is_better'):
            p['sort_direction'] = sort_dir
    return p


def process_submission_zip(leaderboard_id, submission_name, zip_path,
                           owner_user_id=None, source_colab_url=None):
    """
    Helper function to process a single submission zip file.
    Create DB entry, extract files, and queue processing task.
    owner_user_id (Phase 1 multi-tenancy): the User who uploaded.
    source_colab_url: optional gist URL recorded on the Submission so
    reviewers can re-open the exact notebook that produced the upload.
    """
    try:
        new_submission = Submission(
            name=submission_name,
            leaderboard_id=leaderboard_id,
            processing_status='Queued',
            owner_user_id=owner_user_id,
            source_colab_url=source_colab_url,
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
    
    # `samples_only_mode`: explicit "explore the cached samples"
    # navigation when the user has no submissions to compare yet, or
    # is poking at the dataset before uploading anything. Triggered
    # by the `samples_only=1` query param OR the request landing here
    # with an empty compare_ids parameter AND no submissions on the
    # LB. The template suppresses submission-side columns and shows
    # an empty-state banner.
    samples_only_mode = bool(request.args.get('samples_only'))
    if samples_only_mode:
        # Explicit ?samples_only=1 wins over a stale session.compare_ids.
        # Without this gate, clicking "Explore samples" on a PWC LB while
        # the session still held compare_ids from a previous LB would
        # surface 50 mirrored-submission columns the user did not ask for.
        submissions = []
        compare_ids = []
    elif compare_ids:
        # Filter if user explicitly selected a subset
        submissions = [s for s in leaderboard.submissions if str(s.id) in compare_ids and not s.is_archived]
    else:
        submissions = [s for s in leaderboard.submissions if not s.is_archived]

    # Distinguish "you explicitly want to browse samples only" from
    # "this LB has no submissions yet" — the template renders the
    # same surface but with a slightly different empty-state copy.
    if not submissions and not samples_only_mode:
        samples_only_mode = True  # graceful fall-through

    # Pagination params
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 5, type=int)
    search_query = request.args.get('search_query', '')
    sort_by = request.args.get('sort_by', '') # Default to empty (no sort)
    sort_order = request.args.get('sort_order', 'asc')

    # Base query for samples.
    # Two shapes:
    #   - BH LBs (have lb.datasets): Sample rows live in the DB.
    #   - HF-attached LBs (lb.datasets is empty, an Attachment(kind='hf')
    #     carries the repo): no Sample rows. We synthesize stub objects
    #     from CustomField.sample_name written by the metric-eval task,
    #     so the comparison view can still render per-sample
    #     predictions even though the GT lives on huggingface.co.
    dataset_ids = [ds.id for ds in leaderboard.datasets] if leaderboard.datasets else [leaderboard.dataset_id]
    has_real_dataset = bool(leaderboard.datasets)
    hf_stub_mode = not has_real_dataset and bool(leaderboard.attachments)
    samples_query = Sample.query.filter(Sample.dataset_id.in_(dataset_ids))

    # Apply search filter
    if search_query:
        samples_query = samples_query.filter(Sample.name.ilike(f'%{search_query}%'))

    # Apply tag filters
    samples_query = apply_tag_filters(samples_query, request.args)

    # Collect unique custom field names efficiently
    # Dataset fields — for HF-stub mode, pull from the LB-scoped
    # snapshot rows written by the eval task (sample_id IS NULL,
    # submission_id IS NULL, leaderboard_id = lb.id) so the GT
    # column populates with the streamed HF values.
    if hf_stub_mode:
        dataset_custom_fields_query = db.session.query(
            CustomField.name, CustomField.field_type,
        ).filter(
            CustomField.leaderboard_id == leaderboard.id,
            CustomField.submission_id.is_(None),
            CustomField.sample_id.is_(None),
        ).distinct().all()
    else:
        dataset_custom_fields_query = db.session.query(CustomField.name, CustomField.field_type).join(Sample).filter(
            Sample.dataset_id.in_(dataset_ids),
            CustomField.submission_id == None
        ).distinct().all()

    dataset_custom_fields = {name for name, ftype in dataset_custom_fields_query if ftype in ['image', 'depth', 'mask', 'audio', 'scalar', 'metric', 'text', 'json']}
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

    # HF-attached LBs have no Sample rows — synthesize stubs from the
    # sample_names already written into CustomField by the eval task.
    # The rest of the route operates on these stubs the same way it
    # operates on real Sample objects, since most lookups are by name.
    # `.id` is a stable per-page integer (urls hitting /sample/<int:id>/
    # 404 in this mode — they reference GT bytes we don't have on disk).
    if hf_stub_mode:
        class _SampleStub:
            __slots__ = ('id', 'name', 'tags', 'custom_fields',
                         'signal_shape', 'histogram_data', 'config_data',
                         'dataset')

            def __init__(self, name, idx):
                self.id = idx
                self.name = name
                self.tags = ''
                self.custom_fields = []
                self.signal_shape = None
                self.histogram_data = None
                self.config_data = None
                self.dataset = None

        sub_ids_for_stubs = [s.id for s in submissions]
        # Collect sample names from both the submission rows AND the
        # LB-scoped GT snapshot so a fresh sub with no submission CFs
        # still surfaces GT-only rows.
        sub_name_rows = (
            db.session.query(CustomField.sample_name)
            .filter(CustomField.submission_id.in_(sub_ids_for_stubs))
            .filter(CustomField.sample_name.isnot(None))
            .distinct().all()
        )
        gt_name_rows = (
            db.session.query(CustomField.sample_name)
            .filter(CustomField.leaderboard_id == leaderboard.id)
            .filter(CustomField.submission_id.is_(None))
            .filter(CustomField.sample_name.isnot(None))
            .distinct().all()
        )
        names = sorted({r[0] for r in (sub_name_rows + gt_name_rows) if r[0]})
        if search_query:
            needle = search_query.lower()
            names = [n for n in names if needle in n.lower()]
        if sort_by == 'name' and sort_order == 'desc':
            names.reverse()
        all_stubs = [_SampleStub(n, i) for i, n in enumerate(names)]
        total = len(all_stubs)
        paginated_items = all_stubs[(page-1)*per_page : page*per_page]

        # Pre-fetch GT snapshot rows for the visible page and attach
        # them as .custom_fields on each stub so the existing per-row
        # loop reads GT values without a code branch.
        if paginated_items:
            visible_names = [s.name for s in paginated_items]
            gt_rows = CustomField.query.filter(
                CustomField.leaderboard_id == leaderboard.id,
                CustomField.submission_id.is_(None),
                CustomField.sample_id.is_(None),
                CustomField.sample_name.in_(visible_names),
            ).all()
            gt_by_name = {}
            for cf in gt_rows:
                gt_by_name.setdefault(cf.sample_name, []).append(cf)
            for stub in paginated_items:
                stub.custom_fields = gt_by_name.get(stub.name, [])

    # Sorting (BH path only — HF branch above already paginated)
    if not hf_stub_mode and sort_by:
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
                            context = get_metric_context(s, target_sub, submission_folder=submission_folder, pointer_resolver=_pointer_gt_resolver, paired_gt_provider=_make_paired_gt_provider(target_sub.leaderboard))
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
    elif not hf_stub_mode:
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
        
        # Add GT custom fields for this sample. Non-image kinds (text,
        # json, scalar) get their value surfaced alongside the field_id;
        # the template branches on whichever attribute is non-null.
        for cf in sample.custom_fields:
            if cf.field_type in ['image', 'depth', 'mask', 'audio', 'scalar', 'text', 'json']:
                sample_info['custom_fields'][cf.name] = {
                    'gt_field_id': cf.id if cf.field_type in ['image', 'depth', 'mask', 'audio'] else None,
                    'gt_scalar_value': cf.value_float if cf.field_type == 'scalar' else None,
                    'gt_text_value': cf.value_text if cf.field_type in ('text', 'json') else None,
                    'gt_field_type': cf.field_type,
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
            dynamic_ctx = get_metric_context(sample, sub, pointer_resolver=_pointer_gt_resolver, paired_gt_provider=_make_paired_gt_provider(sub.leaderboard if sub else None))
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
    
    # Inverted-visibility model — default to all available columns,
    # subtract anything the user explicitly hid.
    hidden_set = set(
        c.strip() for c in (leaderboard.hidden_comparison_display_columns or '').split(',')
        if c.strip()
    )
    selected_comparison_display_columns = [
        k for k in available_display_options.keys() if k not in hidden_set
    ]
    
    # Also ensure selected_comparison_display_columns are sorted by priority
    selected_comparison_display_columns = sorted(selected_comparison_display_columns,
                                                  key=lambda x: get_column_priority(x, available_display_options.get(x, {}).get('type'), x in dataset_custom_fields))

    # Explore-samples view: drop ONLY panels that need submission data.
    # Image/depth/mask thumbs render straight from cache; text/scalar/
    # json/audio GT columns are useful for browsing too (e.g. the user
    # wants the `label_class` text column visible on ImageNet-256).
    # `per_sample_metrics` and `per_source_stats` are the two that need
    # actual submissions to populate.
    if samples_only_mode:
        _hide_in_samples_only = {'per_sample_metrics', 'per_source_stats',
                                 'pred_histogram'}
        selected_comparison_display_columns = [
            k for k in selected_comparison_display_columns
            if k not in _hide_in_samples_only
            and not k.startswith('viz_')
        ]

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
                           current_compare_ids=compare_ids_arg,
                           samples_only_mode=samples_only_mode)





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
            
    # Same thumbnail-picking pass /home does — first image-or-depth
    # custom field on any sample, rendered to PNG by the existing
    # /custom_field_image endpoint. None when the dataset is metric-only.
    dataset_thumbs = {ds.id: _dataset_thumb_url(ds) for ds in datasets}
    return render_template('datasets.html', datasets=datasets,
                           dataset_thumbs=dataset_thumbs)

@app.route('/author_avatars/<filename>')
def serve_author_avatar(filename):
    avatar_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'author_avatars')
    return send_from_directory(avatar_dir, filename)

_VALID_FIELD_TYPES = ('image', 'scalar', 'metric', 'histogram', 'depth', 'json', 'text')


def _dataset_field_types(dataset):
    """Return [{name, current_type, sample_count}] for every distinct
    custom-field name on this dataset, used by the settings UI."""
    sample_ids = [s.id for s in dataset.samples]
    if not sample_ids:
        return []
    rows = (
        db.session.query(
            CustomField.name,
            CustomField.field_type,
            func.count(CustomField.id),
        )
        .filter(CustomField.sample_id.in_(sample_ids))
        .group_by(CustomField.name, CustomField.field_type)
        .all()
    )
    # Collapse: a field name with mixed types (rare, mid-edit state) shows
    # the most common one and a "mixed" flag.
    by_name = {}
    for name, ftype, cnt in rows:
        entry = by_name.setdefault(name, {
            'name': name, 'current_type': ftype,
            'sample_count': 0, 'mixed': False,
        })
        entry['sample_count'] += cnt
        if entry.get('current_type') != ftype:
            entry['mixed'] = True
    return sorted(by_name.values(), key=lambda r: r['name'])


@app.route('/dataset/<int:dataset_id>/settings', methods=['GET'])
@login_required
@owner_required(Dataset, 'dataset_id')
def dataset_settings(dataset_id):
    """Dedicated settings page — collects all owner-only controls
    (sharing, danger zone, field types) in one place so they aren't
    scattered across the busy dataset detail view."""
    dataset = Dataset.query.get_or_404(dataset_id)
    field_types = _dataset_field_types(dataset)
    # Source provenance for the HF-import badge.
    hf_meta = None
    if (dataset.source_kind or '').startswith('hf-') and dataset.source_metadata:
        try:
            hf_meta = json.loads(dataset.source_metadata)
        except Exception:
            hf_meta = None
    return render_template(
        'dataset_settings.html',
        dataset=dataset,
        field_types=field_types,
        valid_field_types=_VALID_FIELD_TYPES,
        hf_meta=hf_meta,
    )


@app.route('/dataset/<int:dataset_id>/field/<path:field_name>/type', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def update_dataset_field_type(dataset_id, field_name):
    """Reclassify every CustomField row with this name on this dataset
    to a new field_type. Useful when auto-detection picked the wrong
    type (e.g. a `metric_*` column that landed as 'scalar')."""
    new_type = (request.form.get('field_type') or '').strip()
    if new_type not in _VALID_FIELD_TYPES:
        flash(f"Invalid field type '{new_type}'.", "danger")
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    dataset = Dataset.query.get_or_404(dataset_id)
    sample_ids = [s.id for s in dataset.samples]
    if not sample_ids:
        flash("No samples to update.", "warning")
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    updated = (
        CustomField.query
        .filter(CustomField.sample_id.in_(sample_ids), CustomField.name == field_name)
        .update({'field_type': new_type}, synchronize_session=False)
    )
    db.session.commit()
    flash(f"Reclassified {updated} '{field_name}' rows to '{new_type}'.", "success")
    return redirect(url_for('dataset_settings', dataset_id=dataset_id))


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
    # Sample-name search (case-insensitive substring match), mirrors the
    # comparison view's `search` query param so the URL contract is the
    # same in both places.
    sample_search_query = (request.args.get('search') or '').strip()
    if sample_search_query:
        samples_query = samples_query.filter(Sample.name.ilike(f'%{sample_search_query}%'))
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
        elif field_type == 'text' and field_name != 'tags':
            # Text columns (AG News `text`, NLI `premise`/`hypothesis`,
            # captions, etc.) need their own column too. Skip the
            # reserved `tags` field — that one is already surfaced via
            # the dataset's tag widget.
            available_display_options[field_name] = {'label': field_name, 'type': 'text', 'default_width': '300px'}
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
    
    # Inverted-visibility model: default is "all available columns
    # visible". Only columns the user explicitly hid (saved in
    # hidden_display_columns) are removed from the selection. New custom
    # fields added later automatically appear because they aren't in
    # the hidden list.
    hidden_set = set(
        c.strip() for c in (dataset.hidden_display_columns or '').split(',')
        if c.strip()
    )
    selected_display_columns = [
        k for k in available_display_options.keys() if k not in hidden_set
    ]
    
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
                           custom_scalar_metrics=custom_scalar_metrics,
                           sample_search_query=sample_search_query)


@app.route('/dataset/<int:dataset_id>/update_display_columns', methods=['POST'])
def update_dataset_display_columns(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    chosen = set(request.form.getlist('display_columns'))
    # The form posts back the full set of options it rendered, so we can
    # compute "hidden = rendered - chosen" without re-deriving the
    # available_display_options here. Saves duplicating the heavy
    # column-derivation logic from dataset_view().
    rendered = set(request.form.getlist('display_columns_all'))
    hidden = rendered - chosen
    dataset.hidden_display_columns = ','.join(sorted(hidden)) if hidden else None
    db.session.commit()
    return redirect(request.referrer or url_for('dataset_view', dataset_id=dataset_id))

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
    chosen = set(request.form.getlist('comparison_display_columns'))
    rendered = set(request.form.getlist('comparison_display_columns_all'))
    # Inverted-visibility model: the comparison_view renderer treats
    # `available_display_options - hidden_comparison_display_columns`
    # as the visible set, so all we have to persist is the user's
    # explicit exclusions on this page (rendered checkboxes they didn't
    # tick). Anything they couldn't see (not rendered) is untouched.
    hidden = rendered - chosen
    leaderboard.hidden_comparison_display_columns = ','.join(sorted(hidden)) if hidden else None
    db.session.commit()
    return redirect(request.referrer or url_for('leaderboard_view', leaderboard_id=leaderboard_id))

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

@app.route('/api/submission_viz/<int:submission_id>/<path:rel>')
def serve_submission_viz(submission_id, rel):
    """Serve a per-sample viz thumbnail PNG written by the post-eval
    pipeline (`_generate_submission_viz_assets`). `rel` is `<col>/<sample>.png`.

    No auth: viz assets are derived from prediction outputs and inherit
    the leaderboard's visibility. If the leaderboard is private the
    submission row's owner check is enforced via `visible_in_list`
    on the comparison page that links here; the file path itself is
    not enumerable since it requires submission_id + col + sample_name.
    """
    sub = Submission.query.get_or_404(submission_id)
    # Block path traversal — `..` segments must not climb out of viz/.
    safe_rel = os.path.normpath(rel)
    if safe_rel.startswith('..') or os.path.isabs(safe_rel):
        abort(400)
    base = _submission_viz_dir(sub)
    full = os.path.join(base, safe_rel)
    if not os.path.exists(full):
        abort(404)
    return send_file(full, mimetype='image/png')


@app.route('/api/gt_viz/<int:lb_id>/<col>/<sample_name>')
@visibility_required(Leaderboard, 'lb_id')
def serve_gt_viz(lb_id, col, sample_name):
    """Serve a cached HF GT image/mask/depth thumbnail. Resolves the
    LB's primary HF attachment, parses the row index out of
    `sample_name` (the `_VirtualSample` `s_NNNNNN` convention), looks
    up the bench_cache entry by (repo, revision, split, col, row_idx).

    Inherits LB visibility (the @visibility_required decorator). 404
    when the thumb isn't cached — first eval populates the cache and
    re-eval refreshes it; users browsing before that get 404s, which
    the template falls back to a placeholder icon for."""
    lb = Leaderboard.query.get_or_404(lb_id)
    # Parse row index from `s_NNNNNN`. Any other shape → 404.
    m = re.match(r'^s_(\d+)$', sample_name or '')
    if not m:
        abort(404)
    sample_idx = int(m.group(1))
    # Primary HF attachment carries the repo + split.
    att = next(
        (a for a in (lb.attachments or [])
         if getattr(a, 'kind', None) == 'hf' and a.role == 'primary'
         and a.hf_repo_id),
        None,
    )
    if att is None:
        abort(404)
    cache_root = app.config.get('CACHE_FOLDER')
    if not cache_root:
        abort(404)
    key = _gt_viz_cache_key(
        att.hf_repo_id, att.hf_revision, att.hf_split, col, sample_idx,
    )
    try:
        from bench_cache import cache_get
    except ImportError:
        abort(404)
    path = cache_get(db.session, CacheEntry, cache_root=cache_root, key=key)
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        abort(404)
    # max_age=60 + Flask's default conditional GET (If-Modified-Since +
    # ETag) gives us cheap revalidation: browser caches for a minute,
    # then sends If-Modified-Since. If the cache entry hasn't been
    # regenerated (file mtime unchanged), the server returns 304 with
    # no body. After a wipe-and-repopulate (e.g. fixing a black-depth
    # render), the mtime moves and the next refresh sees the new bytes
    # — instead of being stuck with stale bytes for a full day.
    # Sniff PNG header so audio waveform thumbs serve with the right
    # mimetype (they're PNG; image/mask/depth thumbs are JPEG).
    with open(path, 'rb') as fh:
        magic = fh.read(4)
    if magic == b'\x89PNG':
        return send_file(path, mimetype='image/png', max_age=60)
    return send_file(
        path, mimetype='image/jpeg', max_age=60,
    )


@app.route('/custom_field_image/<int:field_id>')
def serve_custom_field_image(field_id):
    """Serve a custom field image or depth map"""
    custom_field = CustomField.query.get_or_404(field_id)

    # HF-attached LB snapshot rows: bytes live in bench_cache, not on
    # the volume. The marker row has leaderboard_id + sample_name +
    # source_column set; redirect to the dedicated GT-viz route which
    # resolves the cache by (repo, revision, split, col, idx).
    if (custom_field.leaderboard_id is not None
            and custom_field.submission_id is None
            and custom_field.sample_id is None):
        col = custom_field.source_column or custom_field.name
        if not col or not custom_field.sample_name:
            abort(404)
        return redirect(url_for(
            'serve_gt_viz',
            lb_id=custom_field.leaderboard_id,
            col=col,
            sample_name=custom_field.sample_name,
        ))

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

def _fetch_remote_submission_zip(remote_url, *, hf_token=None,
                                 sub_id=None):
    """Resolve a `remote_url` (https:// or hf://owner/repo/path) to a
    local file path via bench_cache. Returns (local_path, sha256_hex).

    Cache key shape: `sub:zip:<remote_url>` (one cache entry per
    URL, NOT per submission, so two submissions referencing the same
    URL share the cached bytes — correct under hash-pin semantics).
    Hash captured + returned so the caller can persist /
    cross-check it on the Submission row.
    """
    import hashlib
    import urllib.parse
    cache_root = app.config.get('CACHE_FOLDER')
    cache_key = f"sub:zip:{remote_url}"

    def _writer(path):
        scheme = urllib.parse.urlparse(remote_url).scheme
        if scheme == 'hf':
            # hf://<repo_id>/<path>[@<revision>]
            spec = remote_url[len('hf://'):]
            revision = None
            if '@' in spec:
                spec, revision = spec.rsplit('@', 1)
            parts = spec.split('/', 2)
            if len(parts) < 3:
                raise ValueError(
                    f"hf:// URL needs `owner/repo/path`, got {remote_url!r}"
                )
            owner, repo, file_path = parts[0], parts[1], parts[2]
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=f"{owner}/{repo}", filename=file_path,
                repo_type='dataset', revision=revision, token=hf_token,
            )
            shutil.copy2(downloaded, path)
        elif scheme in ('http', 'https'):
            import requests as _r
            with _r.get(remote_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
        else:
            raise ValueError(f"unsupported URL scheme: {remote_url!r}")

    cached_path = bench_cache.cache_put(
        db.session, CacheEntry,
        cache_root=cache_root, key=cache_key,
        origin='submission', writer=_writer,
    )
    sha = hashlib.sha256()
    with open(cached_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            sha.update(chunk)
    return cached_path, sha.hexdigest()


@contextlib.contextmanager
def _with_extracted_submission(submission):
    """Yield a directory containing the submission's extracted
    contents — `uploads/submissions/<id>/` for local subs, a transient
    re-extraction for remote subs whose on-disk folder has already
    been evicted.

    Cleanup behavior:
    - Local: never deletes the folder.
    - Remote, on-disk folder still present: yields it without cleanup
      (initial-eval path; the post-eval evictor will tear it down once
      that eval completes).
    - Remote, on-disk folder missing: extracts cached ZIP into a
      tempfile.mkdtemp(), yields, ALWAYS cleans up on exit.

    Raises RuntimeError if the row is `storage_mode='remote'` without
    a `remote_url` (corrupted state)."""
    folder = os.path.join(
        app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id),
    )
    storage = getattr(submission, 'storage_mode', 'local') or 'local'

    if storage != 'remote':
        yield folder
        return

    # On-disk extraction still around (e.g. the initial eval for a
    # freshly-uploaded remote submission) → use it as-is.
    if os.path.isdir(folder) and any(
        name for name in os.listdir(folder)
        if name not in ('__MACOSX',) and not name.startswith('.')
    ):
        yield folder
        return

    if not submission.remote_url:
        raise RuntimeError(
            f"Submission {submission.id} marked remote but has no remote_url"
        )
    owner_token = None
    try:
        owner_token = (submission.owner.hf_token if submission.owner else None)
    except Exception:
        pass
    cached_path, _ = _fetch_remote_submission_zip(
        submission.remote_url, hf_token=owner_token,
    )
    tempdir = tempfile.mkdtemp(prefix=f'benchhub-sub-{submission.id}-')
    try:
        with zipfile.ZipFile(cached_path) as zf:
            zf.extractall(tempdir)
        # If the ZIP contained a single top-level folder (the common
        # case from `make_archive` style packagers), descend into it
        # so the metric scanner sees the same shape it would for a
        # local submission whose ZIP was unwrapped at upload time.
        entries = [
            e for e in os.listdir(tempdir)
            if e != '__MACOSX' and not e.startswith('.')
        ]
        target = tempdir
        if len(entries) == 1 and os.path.isdir(os.path.join(tempdir, entries[0])):
            target = os.path.join(tempdir, entries[0])
        yield target
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


# Cap + target size for the per-sample viz PNGs we keep on disk so the
# comparison page can show dense predictions (depth maps, seg masks)
# without re-loading the raw prediction bytes per request. ImageNet-
# scale runs would be hundreds of GB at full res — at 256×256 colormap
# PNG it's ~30 KB/sample, ~300 MB for a 10k-sample cap.
SUBMISSION_VIZ_MAX_SAMPLES = 10_000
SUBMISSION_VIZ_TARGET_SIZE = 256


def _write_depth_viz_png(arr, dest_path):
    """Normalize → downscale → turbo colormap → PNG. Best-effort: any
    failure is logged and the file simply isn't written, since viz
    assets are non-essential."""
    from PIL import Image
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2 or a.size == 0:
            return False
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return False
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        img = Image.fromarray((norm * 255).astype(np.uint8), mode='L')
        img.thumbnail((SUBMISSION_VIZ_TARGET_SIZE, SUBMISSION_VIZ_TARGET_SIZE))
        try:
            import matplotlib.cm as _cm
            rgba = (_cm.turbo(np.asarray(img) / 255.0) * 255).astype(np.uint8)
            Image.fromarray(rgba[..., :3], 'RGB').save(dest_path, optimize=True)
        except Exception:
            # Matplotlib missing in some envs — fall back to grayscale.
            img.save(dest_path, optimize=True)
        return True
    except Exception as e:
        print(f"DEBUG: depth viz png failed for {dest_path}: {e}")
        return False


def _write_image_viz_png(arr, dest_path):
    """Downscale an RGB/seg-mask prediction to a thumbnail PNG."""
    from PIL import Image
    try:
        a = np.asarray(arr)
        if a.ndim == 2:
            # Single-channel mask: treat as class IDs, give each a
            # deterministic hue via a small LUT so adjacent classes
            # are visually distinct.
            lut = ((np.arange(256, dtype=np.int32) * 31 + 17) % 256).astype(np.uint8)
            idx = a.astype(np.int32) & 0xFF
            r = lut[idx]
            g = lut[(idx + 85) & 0xFF]
            b = lut[(idx + 170) & 0xFF]
            a = np.stack([r, g, b], axis=-1).astype(np.uint8)
        elif a.ndim == 3 and a.shape[-1] == 4:
            a = a[..., :3]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        img = Image.fromarray(a)
        img.thumbnail((SUBMISSION_VIZ_TARGET_SIZE, SUBMISSION_VIZ_TARGET_SIZE))
        img.save(dest_path, optimize=True)
        return True
    except Exception as e:
        print(f"DEBUG: image viz png failed for {dest_path}: {e}")
        return False


def _submission_viz_dir(submission):
    """Persistent path where this submission's viz PNGs live, regardless
    of whether the submission is local (always-on disk) or remote
    (extraction is transient). Survives `_evict_extracted_submission_folder`."""
    return os.path.join(
        app.config['UPLOAD_FOLDER'], 'submissions',
        str(submission.id), 'viz',
    )


# --- HF GT thumbnail cache --------------------------------------------------
# At eval time, we generate small JPEG previews of HF image / mask / depth
# GT columns and cache them through bench_cache (origin='gt' → submissions
# evict first when the cache fills). Cache keys are shared across LBs and
# submissions that point at the same HF repo + column + revision.

# GT cached thumbnails preserve the source's original spatial size.
# We still do the cheap optimizations (JPEG q70 compression, turbo
# colormap for depth) so a 640×480 depth map drops from ~1.2 MB raw
# to ~50 KB; we just don't downscale below the source's pixel grid.
# Reason: the zoom modal lets the user pan + wheel-zoom the cached
# image. With 128×128 thumbs you hit visible pixelation at ~5x,
# which makes detail inspection useless. Original size keeps the
# zoom usable. bench_cache LRU still bounds total disk usage.
GT_VIZ_JPEG_QUALITY = 70


def _turbo_lut():
    """256×3 uint8 LUT for the Google "Turbo" colormap. Precomputed at
    import time via matplotlib so per-thumb rendering is a pair of
    array lookups, not a colormap call per pixel."""
    try:
        from matplotlib import colormaps as _cmaps
        cmap = _cmaps['turbo']
    except Exception:
        # matplotlib.cm fallback for very old matplotlib builds.
        from matplotlib import cm as _cm
        cmap = _cm.get_cmap('turbo')
    return (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


_TURBO_LUT = _turbo_lut()


def _gt_viz_cache_key(repo_id, revision, split, col, sample_idx):
    """Stable key for an HF GT thumbnail. Revision defaults to 'main'
    so the cache survives an unpinned attachment getting later pinned
    to the same content."""
    rev = revision or 'main'
    return f"gt_viz:{repo_id}@{rev}:{split or 'train'}:{col}:{sample_idx}"


def _write_gt_image_thumb(value, dest_path, kind):
    """Render an HF GT cell into a tiny JPEG at `dest_path`. `value` is
    whatever the HF row's column produced (PIL.Image / numpy array /
    bytes); `kind` ∈ {'image', 'mask', 'depth'}.

    Single-channel inputs (mode L / I / I;16 / F numpy 2D etc.) are
    routed through the array path so depth maps and class-id masks
    get normalized + colormapped. Calling `.convert('RGB')` directly
    on a depth-in-meters PIL Image would emit a near-black thumb
    (raw depth values 0..10 mapped 1:1 onto 0..255 RGB).

    Best-effort: any decode error returns False and the caller skips
    caching this thumb."""
    from PIL import Image
    SINGLE_CHANNEL_MODES = {'L', 'I', 'I;16', 'I;16B', 'I;16L', 'F'}
    try:
        # Normalize the input into a numpy array OR a ready-to-thumb
        # RGB PIL Image. Depth/mask kinds always go via the array path
        # so they get proper normalization; image kind takes the fast
        # PIL path unless it's single-channel.
        img = None
        arr = None
        if hasattr(value, 'convert') and hasattr(value, 'mode'):
            if kind in ('depth', 'mask') or value.mode in SINGLE_CHANNEL_MODES:
                arr = np.asarray(value)
            else:
                img = value.convert('RGB')
        elif isinstance(value, (bytes, bytearray)):
            decoded = Image.open(io.BytesIO(bytes(value)))
            if kind in ('depth', 'mask') or decoded.mode in SINGLE_CHANNEL_MODES:
                arr = np.asarray(decoded)
            else:
                img = decoded.convert('RGB')
        else:
            arr = np.asarray(value)

        if img is None:
            if arr is None:
                return False
            # 2D path: normalize for depth, deterministic-hue for masks,
            # grayscale stretch otherwise.
            if kind == 'depth' or (arr.ndim == 2 and kind != 'mask'):
                # Normalize → Google Turbo colormap. Same colormap the
                # zoom-modal's Plotly heatmap uses, so the thumbnail
                # and the zoomed view stay visually consistent.
                a = arr.astype(np.float32)
                finite_mask = np.isfinite(a)
                if finite_mask.any():
                    lo = float(np.nanmin(a[finite_mask]))
                    hi = float(np.nanmax(a[finite_mask]))
                else:
                    lo, hi = 0.0, 1.0
                rng = hi - lo if hi > lo else 1.0
                norm = np.nan_to_num((a - lo) / rng, nan=0.0,
                                     posinf=1.0, neginf=0.0)
                norm = np.clip(norm, 0.0, 1.0)
                idx = np.clip((norm * 255).astype(np.int32), 0, 255)
                rgb = _TURBO_LUT[idx]
            elif kind == 'mask' and arr.ndim == 2:
                lut = ((np.arange(256, dtype=np.int32) * 31 + 17) % 256).astype(np.uint8)
                idx = arr.astype(np.int32) & 0xFF
                r = lut[idx]; g = lut[(idx + 85) & 0xFF]; b = lut[(idx + 170) & 0xFF]
                rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
            elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                rgb = arr[..., :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                # Unknown shape — bail.
                return False
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            img = Image.fromarray(rgb)

        # No thumbnail downscale: preserve original source resolution
        # so the zoom modal can let the user pan + wheel-zoom without
        # hitting pixelation. JPEG q70 + (for depth) the turbo
        # colormap LUT keep the file size sane.
        img.save(dest_path, format='JPEG', quality=GT_VIZ_JPEG_QUALITY,
                 optimize=True)
        return True
    except Exception as e:
        print(f"DEBUG: gt viz thumb failed for {dest_path}: {e}")
        return False


def _write_gt_audio_thumb(value, dest_path):
    """Render an HF Audio cell into a small waveform PNG. `value` is
    typically `{'array': np.ndarray, 'sampling_rate': int, 'path': str}`
    or a numpy array directly. Falls through to False on any error so
    the caller can skip caching this entry."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"DEBUG: matplotlib not available for audio thumb: {e}")
        return False
    try:
        arr = None
        sr = None
        if isinstance(value, dict):
            arr = value.get('array')
            sr = value.get('sampling_rate')
            if arr is None and value.get('path'):
                try:
                    import soundfile as sf
                    arr, sr = sf.read(value['path'])
                except Exception:
                    arr = None
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
        if arr is None:
            return False
        arr = np.asarray(arr)
        if arr.ndim == 2:
            # Mix stereo → mono for the thumb.
            arr = arr.mean(axis=1) if arr.shape[1] in (2, 4) else arr[:, 0]
        if arr.size == 0:
            return False
        # Downsample to ~600 points for a compact waveform.
        target = 600
        if arr.size > target:
            stride = arr.size // target
            arr = arr[: stride * target].reshape(target, stride).mean(axis=1)
        peak = float(np.max(np.abs(arr))) or 1.0
        arr = arr / peak

        fig = plt.figure(figsize=(6, 1.5), dpi=80)
        ax = fig.add_subplot(111)
        ax.fill_between(np.arange(arr.size), arr, -arr, color='#7c3aed', alpha=0.85, linewidth=0)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#fcfcff')
        fig.patch.set_facecolor('#fcfcff')
        fig.tight_layout(pad=0.1)
        fig.savefig(dest_path, format='png', dpi=80,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return True
    except Exception as e:
        print(f"DEBUG: gt audio thumb failed for {dest_path}: {e}")
        return False


def _cache_gt_image_thumb(repo_id, revision, split, col, sample_idx,
                          value, kind):
    """Materialize one HF GT cell as a JPEG in bench_cache. Returns the
    on-disk path or None on failure. Reuses an existing cached entry
    when present (no re-decode, no re-write)."""
    cache_root = app.config.get('CACHE_FOLDER')
    if not cache_root:
        return None
    key = _gt_viz_cache_key(repo_id, revision, split, col, sample_idx)
    # Fast path: already cached.
    try:
        from bench_cache import cache_get, cache_put
    except ImportError:
        return None
    existing = cache_get(db.session, CacheEntry, cache_root=cache_root, key=key)
    if existing:
        return existing

    def _writer(tmp_path):
        if kind == 'audio':
            ok = _write_gt_audio_thumb(value, tmp_path)
        else:
            ok = _write_gt_image_thumb(value, tmp_path, kind)
        if not ok:
            # cache_put expects bytes to land at tmp_path; if the
            # thumb failed, write a placeholder zero-byte file so
            # cache_put doesn't crash. The route serves 404 when
            # size==0.
            with open(tmp_path, 'wb') as f:
                f.write(b'')

    try:
        return cache_put(
            db.session, CacheEntry,
            cache_root=cache_root, key=key,
            writer=_writer, origin='gt',
        )
    except Exception as e:
        print(f"DEBUG: cache_put gt thumb for {key}: {e}")
        return None


def _persist_hf_eval_snapshots(lb, samples_iter):
    """For an HF-attached LB, snapshot streamed GT scalars/text to
    leaderboard-scoped CustomField rows AND cache image/mask/depth
    thumbnails in bench_cache. Called once per eval task.

    `samples_iter` is an iterable of `(virtual_sample, attachment)`
    tuples — typically the same iteration the eval task does for
    metrics. We re-iterate samples here rather than threading the
    persistence through the existing per-metric loop because (1) the
    rows are LB-scoped not metric-scoped, and (2) it keeps the eval
    pipeline readable.

    Idempotent: deletes any existing leaderboard-scoped CustomFields
    for this LB before writing, so re-eval drops stale snapshots.
    Image thumbs use the bench_cache key, so they're auto-deduplicated
    across runs.
    """
    # Reset prior snapshot rows for this LB.
    CustomField.query.filter(
        CustomField.leaderboard_id == lb.id,
        CustomField.submission_id.is_(None),
        CustomField.sample_id.is_(None),
    ).delete(synchronize_session=False)

    for sample, att in samples_iter:
        if att is None or getattr(att, 'kind', None) != 'hf':
            # BH samples already have proper Sample rows + CustomFields.
            continue
        row_idx = None
        if sample.name and sample.name.startswith('s_'):
            try:
                row_idx = int(sample.name[2:])
            except ValueError:
                row_idx = None
        for cf in (sample.custom_fields or []):
            ftype = getattr(cf, 'field_type', None)
            name = getattr(cf, 'name', None)
            if not name:
                continue
            # Keep the bare column name (e.g. 'label', 'image') —
            # mirrors how BH datasets surface GT in the comparison
            # view. The `submission_id IS NULL` flag is what
            # distinguishes GT rows from prediction rows there.
            if ftype == 'scalar':
                db.session.add(CustomField(
                    leaderboard_id=lb.id,
                    sample_name=sample.name,
                    name=name,
                    field_type='scalar',
                    value_float=cf.value_float,
                ))
            elif ftype == 'text':
                db.session.add(CustomField(
                    leaderboard_id=lb.id,
                    sample_name=sample.name,
                    name=name,
                    field_type='text',
                    value_text=cf.value_text,
                ))
            elif ftype in ('image', 'mask', 'depth', 'audio'):
                # Visual/audio: don't persist bytes — write a marker
                # row that points at the served thumbnail URL. The
                # actual JPEG/PNG lives in bench_cache (populated by
                # _populate_hf_gt_thumb_cache).
                db.session.add(CustomField(
                    leaderboard_id=lb.id,
                    sample_name=sample.name,
                    name=name,
                    field_type=ftype,
                    source_column=getattr(cf, 'source_column', None),
                ))
            elif ftype == 'json':
                # JSON-shaped GT (bounding boxes, span offsets, dicts).
                # Stored as text — comparison view detects the field_type
                # and renders it as a key/value card.
                value_text = getattr(cf, 'value_text', None)
                if value_text is None and getattr(cf, 'value_blob', None):
                    try:
                        value_text = cf.value_blob.decode('utf-8', errors='replace')
                    except Exception:
                        value_text = None
                if value_text is not None:
                    db.session.add(CustomField(
                        leaderboard_id=lb.id,
                        sample_name=sample.name,
                        name=name,
                        field_type='json',
                        value_text=value_text,
                    ))
            elif ftype == 'histogram':
                # Histogram bins: persist the raw blob so the comparison
                # view can sparkline it.
                value_blob = getattr(cf, 'value_blob', None)
                if value_blob is not None:
                    db.session.add(CustomField(
                        leaderboard_id=lb.id,
                        sample_name=sample.name,
                        name=name,
                        field_type='histogram',
                        value_blob=value_blob,
                    ))
    db.session.commit()


def _populate_hf_gt_thumb_cache(lb, att, hf_token=None, max_thumbs=10_000,
                                logger=None):
    """Walk the HF attachment a second time to populate the GT image
    thumbnail cache. We do this in a separate pass (not inside the
    metric-eval loop) because re-iteration is cheap on streaming HF
    and it keeps the metric-eval path simple. Returns the count of
    thumbs newly cached.

    Caps:
      - `max_thumbs` per call (default SUBMISSION_VIZ_MAX_SAMPLES,
        but the LB's iteration cap also bounds it).
      - Skips columns whose target_kind is not image/mask/depth.
      - Single attachment at a time (caller iterates attachments).
    """
    try:
        mapping = json.loads(att.hf_mapping_json or '[]')
    except (TypeError, ValueError):
        mapping = []
    image_cols = [
        (m.get('column'), m.get('target_kind'))
        for m in mapping
        if m.get('target_kind') in ('image', 'mask', 'depth', 'audio')
    ]
    if not image_cols:
        return 0
    written = 0
    try:
        from datasets import load_dataset
    except ImportError:
        return 0
    def _load(split):
        return load_dataset(
            att.hf_repo_id, split=split,
            streaming=True, revision=att.hf_revision,
            token=hf_token, trust_remote_code=True,
        )
    ds = _resolve_hf_split_and_load(
        att, _load,
        on_log=(logger.info if logger else None),
    )
    if ds is None:
        return 0
    for i, row in enumerate(ds):
        if i >= max_thumbs:
            break
        for col, kind in image_cols:
            value = row.get(col)
            if value is None:
                continue
            path = _cache_gt_image_thumb(
                att.hf_repo_id, att.hf_revision, att.hf_split,
                col, i, value, kind,
            )
            if path:
                written += 1
    return written


def _persist_pred_scalars_from_disk(submission, leaderboard, submission_folder):
    """For each `<field>/<sample>.txt` under `submission_folder`, parse
    the value as float (or keep as text) and write a CustomField row
    keyed on submission_id + sample_name + field name. Lets the
    comparison view show the actual predicted value (e.g. `label_pred=4`)
    in addition to the metric-output value (`lm_4=0.0` for a miss).

    Idempotent: deletes prior `<field>_pred`-named CustomFields for
    this submission before writing.
    """
    if not submission_folder or not os.path.isdir(submission_folder):
        return
    pred_field_names = [
        p['name'] for p in _lb_submission_pred_fields(leaderboard)
    ]
    if not pred_field_names:
        return
    CustomField.query.filter(
        CustomField.submission_id == submission.id,
        CustomField.name.in_(pred_field_names),
    ).delete(synchronize_session=False)

    sample_pat = re.compile(r'^(s_\d+|.+?)\.txt$')
    for field in pred_field_names:
        d = os.path.join(submission_folder, field)
        if not os.path.isdir(d):
            continue
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for entry in entries:
            m = sample_pat.match(entry)
            if not m:
                continue
            sample_name = m.group(1)
            try:
                with open(os.path.join(d, entry)) as fh:
                    raw = fh.read().strip()
            except OSError:
                continue
            value_float = None
            try:
                value_float = float(raw)
            except (TypeError, ValueError):
                pass
            if value_float is not None:
                db.session.add(CustomField(
                    submission_id=submission.id,
                    sample_name=sample_name,
                    name=field,
                    field_type='scalar',
                    value_float=value_float,
                ))
            elif raw:
                db.session.add(CustomField(
                    submission_id=submission.id,
                    sample_name=sample_name,
                    name=field,
                    field_type='text',
                    value_text=raw,
                ))
    db.session.commit()


def _generate_submission_viz_assets(submission, leaderboard, pred_source_folder):
    """Post-eval pass: write small colormap PNGs for each dense-GT
    prediction so the comparison page can render at-a-glance previews
    without re-decoding the original prediction bytes. Per-submission
    cap of SUBMISSION_VIZ_MAX_SAMPLES (currently 10k) keeps the
    storage footprint bounded on ImageNet-scale runs.

    Reads from `pred_source_folder` (which may be a tempdir for remote
    submissions); writes to `_submission_viz_dir(submission)` (always
    on the persistent volume) so viz survives the post-eval eviction.

    Best-effort throughout — a single bad sample shouldn't poison the
    submission's status. Non-dense fields (scalar, text) are skipped
    because there's nothing visual to thumbnail.
    """
    pred_schema = _lb_submission_pred_fields(leaderboard) or []
    dense_fields = [pf for pf in pred_schema
                    if pf.get('kind') in ('image', 'mask', 'depth')]
    if not dense_fields:
        return 0

    from metric_engine import _load_sub_pred_for_sample
    viz_root = _submission_viz_dir(submission)
    written = 0
    for sample, _att in _iter_lb_eval_samples(leaderboard):
        if written >= SUBMISSION_VIZ_MAX_SAMPLES:
            break
        for pf in dense_fields:
            # Predictions land in `<bare-name>/<sample>.<ext>` where
            # bare-name is the GT field (mirrored on the submission side).
            # `pf['name']` is the arg-mappings shape (`<gt>_pred`); the
            # actual folder on disk is `pf['gt_field']`.
            col = pf.get('gt_field') or pf['name']
            kind = pf['kind']
            arr = _load_sub_pred_for_sample(pred_source_folder, col, sample.name)
            if arr is None:
                continue
            out_dir = os.path.join(viz_root, col)
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{sample.name}.png")
            if kind == 'depth':
                ok = _write_depth_viz_png(arr, dest)
            else:
                ok = _write_image_viz_png(arr, dest)
            if ok:
                written += 1
                if written >= SUBMISSION_VIZ_MAX_SAMPLES:
                    break
    return written


def _evict_extracted_submission_folder(submission):
    """Tear down `uploads/submissions/<id>/` for a remote submission
    once eval has persisted CustomField rows. The cached ZIP under
    bench_cache is the canonical source from here on; recalcs
    re-extract on demand via `_with_extracted_submission`.

    No-op for local submissions (their bytes are immutable on Fly disk
    and there's nowhere else to fetch them from)."""
    if getattr(submission, 'storage_mode', 'local') != 'remote':
        return
    folder = os.path.join(
        app.config['UPLOAD_FOLDER'], 'submissions', str(submission.id),
    )
    if not os.path.isdir(folder):
        return
    # Preserve `viz/` — those are the post-eval thumbnail PNGs the
    # comparison page reads. They're tiny (~30 KB/sample at 256×256)
    # and re-generating them would require re-extracting the entire
    # submission ZIP. Everything else (raw predictions) is gone after
    # this call; the bench_cache ZIP is the canonical source.
    try:
        for entry in os.listdir(folder):
            if entry == 'viz':
                continue
            path = os.path.join(folder, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    except OSError as e:
        # Best-effort: a cleanup failure shouldn't poison the eval.
        # Volume might be busy / a file might be open in another worker.
        print(f"DEBUG: evict extracted folder for sub {submission.id} failed: {e}")


def _verify_remote_submission_hash(submission):
    """Strict hash-pinning: re-fetch the submission's bytes via
    bench_cache and compare SHA-256 to the stored content_hash. The
    cache makes the common case (hot LRU) free; an evicted entry
    forces a re-fetch from upstream which catches any post-submission
    edit on the remote URL.

    Returns (ok: bool, message: str). For local submissions this is
    always ok=True (the bytes are immutable on Fly disk). For remote
    submissions with a populated content_hash, mismatch sets the
    Submission's processing_status to a clear error and the caller
    should bail.

    First re-eval after a hash was captured but never verified is
    treated as a populate, not a mismatch — content_hash NULL is the
    "I haven't seen this submission yet" case.
    """
    if submission is None:
        return False, "Submission missing."
    if getattr(submission, 'storage_mode', 'local') != 'remote':
        return True, ''
    remote_url = submission.remote_url
    if not remote_url:
        # storage_mode='remote' but no URL: corrupted row.
        return False, "Submission marked remote but has no remote_url."
    owner_token = None
    try:
        owner_token = (submission.owner.hf_token if submission.owner else None)
    except Exception:
        pass
    try:
        _path, current_hash = _fetch_remote_submission_zip(
            remote_url, hf_token=owner_token,
        )
    except Exception as e:
        return False, f"Could not refetch submission for hash check: {e}"
    expected = submission.content_hash
    if not expected:
        # First time we're seeing this submission's hash → record + pass.
        submission.content_hash = current_hash
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
        return True, ''
    if current_hash == expected:
        return True, ''
    # Mismatch — refuse to evaluate against drifted bytes.
    submission.processing_status = (
        'Error: submission file changed; please resubmit'
    )
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
    return False, (
        f"Hash mismatch on remote submission {submission.id} "
        f"(expected {expected[:12]}…, got {current_hash[:12]}…)"
    )


@app.route('/api/leaderboard/<int:leaderboard_id>/submission/from_url',
           methods=['POST'])
@require_api_token
def submission_from_url_api(leaderboard_id):
    """Submit by URL — the user's submission ZIP lives at a public
    https:// or hf://owner/repo/path location, BenchHub fetches it
    on-demand and caches via bench_cache. SHA-256 captured at first
    fetch; recalcs verify the hash hasn't drifted.

    JSON body: {url, submission_name?, source_colab_url?}.
    Or form-data with the same fields.
    """
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    payload = request.get_json(silent=True) or request.form
    remote_url = (payload.get('url') or '').strip()
    if not remote_url:
        return jsonify({'error': 'url is required'}), 400

    ok, msg = check_quota(g.current_user, kind='submission')
    if not ok:
        return jsonify({'error': msg}), 429

    submission_name = (payload.get('submission_name') or '').strip() \
        or remote_url.rsplit('/', 1)[-1].replace('.zip', '') or 'remote_submission'
    source_colab_url = (payload.get('source_colab_url') or '').strip() or None

    # Fetch into cache + capture hash. The user's saved hf_token
    # unlocks gated HF Hub repos.
    try:
        cached_path, content_hash = _fetch_remote_submission_zip(
            remote_url, hf_token=getattr(g.current_user, 'hf_token', None),
        )
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        # Surface a clearer hint when the failure is auth-shaped, with a
        # direct link to the token-settings page so the user knows what
        # to do (rather than just "fetch failed: 401").
        if (remote_url.startswith('hf://')
                and ('401' in msg or 'gated' in low or 'unauthorized' in low
                     or 'access denied' in low)):
            return jsonify({
                'error': (
                    f"fetch failed: {e}. This `hf://` URL needs an HF "
                    f"access token. Save one at {url_for('hf_token_settings', _external=True)} "
                    f"and retry."
                ),
                'token_settings_url': url_for('hf_token_settings', _external=True),
            }), 400
        return jsonify({'error': f'fetch failed: {e}'}), 400

    try:
        success, error = process_submission_zip(
            leaderboard.id, submission_name, cached_path,
            owner_user_id=g.current_user.id,
            source_colab_url=source_colab_url,
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    if not success:
        return jsonify({'error': error}), 500

    # process_submission_zip created a Submission row keyed by
    # whichever was the most recent for the user. Stamp it with
    # storage_mode + remote_url + content_hash so re-eval can verify.
    sub = (Submission.query
           .filter_by(leaderboard_id=leaderboard.id,
                      owner_user_id=g.current_user.id,
                      name=submission_name)
           .order_by(Submission.id.desc())
           .first())
    if sub is not None:
        sub.storage_mode = 'remote'
        sub.remote_url = remote_url
        sub.content_hash = content_hash
        db.session.commit()
    return jsonify({
        'success': True, 'message': 'Submission queued',
        'submission_id': sub.id if sub else None,
        'content_hash': content_hash,
    })


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
    # Colab provenance: prefer the URL the form sent (the colab notebook
    # bakes its own gist URL into the upload call). Fall back to looking
    # up the per-user gist for this LB so anyone using the unmodified
    # generated notebook still gets a back-link.
    source_colab_url = (request.form.get('source_colab_url') or '').strip() or None
    if not source_colab_url:
        ucg = UserColabGist.query.filter_by(
            user_id=g.current_user.id, leaderboard_id=leaderboard.id,
        ).first()
        if ucg and ucg.gist_id:
            path = (f'{ucg.gist_owner}/{ucg.gist_id}'
                    if ucg.gist_owner else ucg.gist_id)
            source_colab_url = f'https://colab.research.google.com/gist/{path}'

    temp_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload_zip', secure_filename(file.filename))
    os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
    file.save(temp_zip_path)

    try:
        success, error = process_submission_zip(
            leaderboard.id, submission_name, temp_zip_path,
            owner_user_id=g.current_user.id,
            source_colab_url=source_colab_url,
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

@app.route('/leaderboard/<int:leaderboard_id>/populate_samples', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def populate_lb_samples_route(leaderboard_id):
    """Kick off the sample-cache populate task for an HF-attached LB.
    Lets the owner/admin browse GT samples on the Explore page without
    first uploading a submission (the eval pipeline normally writes
    the cache on first submission-eval, but PWC mirror imports skip
    that pipeline entirely)."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    has_hf = any(
        getattr(a, 'kind', None) == 'hf'
        for a in (lb.attachments or [])
    )
    if not has_hf:
        flash(
            "Sample cache populate is only meaningful for HF-attached "
            "leaderboards. BH datasets already have their Sample rows "
            "in the database.",
            "info",
        )
        return redirect(url_for(
            'comparison_view',
            leaderboard_id=lb.id, samples_only=1,
        ))
    try:
        import tasks as _tasks
        max_samples = int(request.form.get('max_samples') or 200)
        max_samples = max(1, min(max_samples, 10_000))
        _tasks.populate_lb_samples.delay(lb.id, max_samples=max_samples)
        flash(
            f"Sample cache populate started (up to {max_samples} samples). "
            "Refresh in a moment to see them.",
            "success",
        )
    except Exception as e:
        flash(f"Couldn't kick off sample populate: {e}", "warning")
    return redirect(url_for(
        'comparison_view',
        leaderboard_id=lb.id, samples_only=1,
    ))


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
                    'comparison_display_columns': 'TEXT DEFAULT "{}"',
                    # Two-level taxonomy "area/task". Populated for PWC
                    # imports; manual LBs are nullable.
                    'category': 'VARCHAR(120) DEFAULT NULL',
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

                # --- 3b. Backfill LB categories from PWC archive ---
                # Idempotent: only operates on canonical_for_repo LBs
                # whose category is still NULL. Two-stage match:
                #   1. PWC archive HF-repo join (works when the source
                #      eval's links_json carried an HF link)
                #   2. Parse "<task> on <dataset>" out of lb.name and
                #      match against pwc_evaluation.task (catches LBs
                #      imported via suggest_hf_repo where the link
                #      wasn't in PWC's archive)
                try:
                    cursor.execute(
                        "SELECT id, name, canonical_for_repo FROM leaderboard "
                        "WHERE canonical_for_repo IS NOT NULL AND (category IS NULL OR category = '')"
                    )
                    pending = cursor.fetchall()
                    if pending:
                        try:
                            import pwc_client
                            pwc_path = pwc_client._index_path()
                            if os.path.exists(pwc_path):
                                pwc_conn = sqlite3.connect(pwc_path)
                                pcur = pwc_conn.cursor()
                                pwc_tasks = {
                                    t.lower() for (t,) in pcur.execute(
                                        "SELECT DISTINCT task FROM pwc_evaluation WHERE task IS NOT NULL"
                                    )
                                }
                                updated = 0
                                for lb_id, lb_name, repo in pending:
                                    task = None
                                    # Stage 1: HF-repo join
                                    pcur.execute(
                                        "SELECT e.task FROM pwc_evaluation e "
                                        "JOIN pwc_dataset d ON d.id = e.dataset_id "
                                        "WHERE d.name = ? "
                                        "ORDER BY json_array_length(e.results_json) DESC LIMIT 1",
                                        (repo,),
                                    )
                                    row = pcur.fetchone()
                                    if row and row[0]:
                                        task = row[0]
                                    # Stage 2: split "<task> on <dataset>" from lb.name
                                    if not task and lb_name and ' on ' in lb_name:
                                        head = lb_name.split(' on ', 1)[0].strip()
                                        if head.lower() in pwc_tasks:
                                            task = head
                                    if not task:
                                        continue
                                    cat = _pwc_task_to_category(task)
                                    if cat:
                                        cursor.execute(
                                            "UPDATE leaderboard SET category=? WHERE id=?",
                                            (cat, lb_id),
                                        )
                                        updated += 1
                                pwc_conn.close()
                                if updated:
                                    conn.commit()
                                    print(f"Backfilled category for {updated} PWC-imported LB(s).")
                        except Exception as e:
                            print(f"PWC category backfill skipped: {e}")
                except Exception as e:
                    print(f"Category backfill probe failed: {e}")

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

                # --- UserColabGist table (per-user Colab gist mapping) ---
                try:
                    cursor.execute("SELECT user_id FROM user_colab_gist LIMIT 1")
                except sqlite3.OperationalError:
                    print("Migrating DB: Creating 'user_colab_gist' table...")
                    try:
                        cursor.execute('''
                            CREATE TABLE user_colab_gist (
                                user_id INTEGER NOT NULL,
                                leaderboard_id INTEGER NOT NULL,
                                gist_id VARCHAR(64),
                                gist_owner VARCHAR(120),
                                sig VARCHAR(16),
                                PRIMARY KEY (user_id, leaderboard_id)
                            )
                        ''')
                        conn.commit()
                        print("Created 'user_colab_gist' table.")
                    except Exception as e:
                        print(f"Failed to create 'user_colab_gist' table: {e}")

                # --- Attachment table (BH dataset OR HF ref) ---
                try:
                    cursor.execute("SELECT id FROM attachment LIMIT 1")
                except sqlite3.OperationalError:
                    print("Migrating DB: Creating 'attachment' table...")
                    try:
                        cursor.execute('''
                            CREATE TABLE attachment (
                                id INTEGER PRIMARY KEY,
                                leaderboard_id INTEGER NOT NULL,
                                dataset_id INTEGER,
                                hf_repo_id VARCHAR(200),
                                hf_revision VARCHAR(120),
                                hf_split VARCHAR(50),
                                hf_mapping_json TEXT,
                                role VARCHAR(20) NOT NULL DEFAULT 'primary',
                                hf_sample_cap INTEGER
                            )
                        ''')
                        cursor.execute(
                            'CREATE INDEX ix_attachment_leaderboard_id '
                            'ON attachment (leaderboard_id)'
                        )
                        cursor.execute(
                            'CREATE INDEX ix_attachment_hf_repo_id '
                            'ON attachment (hf_repo_id)'
                        )
                        # Backfill from the legacy m2m table so existing
                        # LBs keep working through the new iteration path.
                        cursor.execute('''
                            INSERT INTO attachment
                                (leaderboard_id, dataset_id, role)
                            SELECT leaderboard_id, dataset_id, role
                            FROM leaderboard_datasets
                        ''')
                        conn.commit()
                        print("Created 'attachment' table + backfilled.")
                    except Exception as e:
                        print(f"Failed to create 'attachment' table: {e}")

                # --- HfDatasetVisit (browse-without-import history) ---
                try:
                    cursor.execute("SELECT user_id FROM hf_dataset_visit LIMIT 1")
                except sqlite3.OperationalError:
                    print("Migrating DB: Creating 'hf_dataset_visit' table...")
                    try:
                        cursor.execute('''
                            CREATE TABLE hf_dataset_visit (
                                user_id INTEGER NOT NULL,
                                repo_id VARCHAR(200) NOT NULL,
                                last_visited_at DATETIME NOT NULL,
                                visit_count INTEGER NOT NULL DEFAULT 1,
                                PRIMARY KEY (user_id, repo_id)
                            )
                        ''')
                        cursor.execute(
                            'CREATE INDEX ix_hf_dataset_visit_last_visited_at '
                            'ON hf_dataset_visit (last_visited_at)'
                        )
                        conn.commit()
                        print("Created 'hf_dataset_visit' table.")
                    except Exception as e:
                        print(f"Failed to create 'hf_dataset_visit' table: {e}")

                # --- CacheEntry table (pointer-mode + remote-submission cache) ---
                try:
                    cursor.execute("SELECT cache_key FROM cache_entry LIMIT 1")
                except sqlite3.OperationalError:
                    print("Migrating DB: Creating 'cache_entry' table...")
                    try:
                        cursor.execute('''
                            CREATE TABLE cache_entry (
                                cache_key VARCHAR(512) PRIMARY KEY,
                                size_bytes BIGINT NOT NULL DEFAULT 0,
                                origin VARCHAR(16) NOT NULL,
                                last_accessed_at DATETIME NOT NULL,
                                created_at DATETIME NOT NULL
                            )
                        ''')
                        cursor.execute(
                            'CREATE INDEX ix_cache_entry_last_accessed_at '
                            'ON cache_entry (last_accessed_at)'
                        )
                        conn.commit()
                        print("Created 'cache_entry' table.")
                    except Exception as e:
                        print(f"Failed to create 'cache_entry' table: {e}")

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
                    # Per-user HuggingFace access token.
                    ("user",                 "hf_token",                       "VARCHAR(200)"),
                    # HF auto-import provenance.
                    ("dataset",              "source_kind",                    "VARCHAR(32)"),
                    ("dataset",              "source_metadata",                "TEXT"),
                    # Inverted column-visibility model: track which cols the
                    # user explicitly hid. Empty/NULL = all visible.
                    ("dataset",              "hidden_display_columns",         "TEXT"),
                    ("leaderboard",          "hidden_comparison_display_columns", "TEXT"),
                    # Cached Colab submission notebook (self-invalidating).
                    ("leaderboard",          "colab_notebook_cache",             "TEXT"),
                    # Per-submission Colab gist URL (provenance — lets a
                    # reviewer re-open the exact notebook that produced
                    # the predictions). NULL for non-Colab submissions.
                    ("submission",           "source_colab_url",                 "VARCHAR(500)"),
                    # Pointer-mode storage. NEW (2026-05-08).
                    # storage_mode marks whether a Dataset's samples
                    # carry on-disk files ('local') or get streamed
                    # from HF on-demand ('hf-pointer').
                    ("dataset",              "storage_mode",                     "VARCHAR(20) NOT NULL DEFAULT 'local'"),
                    # source_ref_json on Sample pins the upstream row
                    # (repo_id, revision, split, row_idx) for pointer-
                    # mode samples. NULL for local samples.
                    ("sample",               "source_ref_json",                  "TEXT"),
                    # source_column on CustomField names the HF column
                    # this field pulls from when its parent Sample is
                    # pointer-backed. NULL otherwise.
                    ("custom_field",         "source_column",                    "VARCHAR(120)"),
                    # Remote submissions (NEW 2026-05-08).
                    ("submission",           "storage_mode",                     "VARCHAR(20) NOT NULL DEFAULT 'local'"),
                    ("submission",           "remote_url",                       "VARCHAR(500)"),
                    ("submission",           "content_hash",                     "VARCHAR(64)"),
                    # Paired datasets (NEW 2026-05-08). 'primary' (default)
                    # vs 'gt_source' so an LB can fold a separate GT
                    # repo's same-named samples into the metric context.
                    ("leaderboard_datasets", "role",                             "VARCHAR(20) NOT NULL DEFAULT 'primary'"),
                    # Canonicality (NEW 2026-05-08). 'personal' (default)
                    # vs 'public' (admin-promoted). canonical_for_repo
                    # uniquely binds a public LB to one HF repo.
                    ("leaderboard",          "canonicality",                     "VARCHAR(20) NOT NULL DEFAULT 'personal'"),
                    ("leaderboard",          "canonical_for_repo",               "VARCHAR(200)"),
                    # Phase 9: required pred fields decoupled from metrics
                    # (organizer wants to receive predictions even when no
                    # scoring metric consumes them).
                    ("leaderboard",          "required_pred_fields_json",        "TEXT"),
                    # Phase 15: PWC mirrored submissions. kind='mirrored'
                    # rows skip the eval pipeline; they're score rows
                    # imported from external benchmarks for context.
                    ("submission",           "kind",                             "VARCHAR(20) NOT NULL DEFAULT 'verified'"),
                    ("submission",           "source_attribution",               "VARCHAR(200)"),
                    ("submission",           "source_paper_url",                 "VARCHAR(500)"),
                    ("submission",           "source_paper_title",               "VARCHAR(300)"),
                    ("submission",           "source_external_url",              "VARCHAR(500)"),
                    # Phase 7: cached storage usage on the dataset itself —
                    # cheaper than du'ing the volume on every upload.
                    ("dataset",              "storage_bytes",                  "BIGINT NOT NULL DEFAULT 0"),
                    # HF-attached LB GT/pred scalar snapshots. LB-scoped
                    # CustomField rows (sample_id NULL, submission_id NULL)
                    # carry GT values streamed from HF at eval time so the
                    # comparison view doesn't re-stream on every page load.
                    ("custom_field",         "leaderboard_id",                 "INTEGER"),
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
        # Pointer-mode datasets INTENTIONALLY have no on-disk folder —
        # bytes live on HF and stream through bench_cache. Skip the
        # folder check for them; sample-count > 0 is the only signal
        # of completion.
        is_pointer = (ds.storage_mode == 'hf-pointer')
        sample_count = Sample.query.filter_by(dataset_id=ds.id).count()
        folder_path = os.path.join(datasets_root, secure_filename(ds.name))
        folder_ok = os.path.isdir(folder_path)

        if is_pointer:
            incomplete = sample_count == 0
        else:
            incomplete = sample_count == 0 or not folder_ok

        if incomplete:
            print(
                f"prune_incomplete_datasets: removing dataset {ds.id} "
                f"'{ds.name}' (storage={ds.storage_mode}, "
                f"samples={sample_count}, folder_present={folder_ok})"
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
