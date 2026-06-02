import contextlib
import shutil
import urllib.parse
from urllib.parse import quote
import sys
import re
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, session, send_file, flash, abort, after_this_request, g, make_response, send_from_directory
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
# Google OAuth — same Authlib pattern. Configure with
# GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET via `fly secrets set`.
# Authorized redirect URI on the Google Cloud Console must match
# `https://<host>/login/google/callback`.
oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
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
    # Two-level "Area/Task" string. Set at upload time or via the
    # dataset settings page. LBs created from this dataset inherit
    # the category if the LB doesn't already have one.
    category = db.Column(db.String(120), nullable=True, index=True)
    # Upstream-of-truth URL for datasets we materialised from elsewhere
    # (Zenodo records, EE Zurich, HF mirrors, archive download pages).
    # Surfaced as a "source" link on the dataset + LB pages so users
    # know where the bytes originally came from. NULL for datasets that
    # were uploaded straight as a ZIP without a known external origin.
    source_url = db.Column(db.Text, nullable=True)
    # Long-running import state. `ready` = fully materialised (the
    # default for the existing flow + back-compat for legacy rows).
    # `importing` = a background task is downloading + writing rows;
    # the dataset is listed on /datasets with a progress badge and
    # /dataset/<id> renders a live progress page instead of the
    # normal view. `failed` = task raised; `import_error` carries
    # the message. NULL is treated as `ready` everywhere.
    import_status = db.Column(db.String(20), nullable=True, default='ready')
    import_progress_json = db.Column(db.Text, nullable=True)
    import_error = db.Column(db.Text, nullable=True)
    import_task_id = db.Column(db.String(60), nullable=True)
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
    # Hybrid storage flag. True ⇒ uploads/datasets/<id>/<field>/<sample>.<ext>
    # holds a downscaled / colormapped preview (JPG for vis modalities,
    # PNG for audio), not the original bytes. The HF source URL stays
    # in source_metadata for re-fetch; LeaderboardMaterialization rows
    # carry the full-resolution snapshots needed for submission scoring.
    preview_only = db.Column(
        db.Boolean, nullable=False,
        default=False, server_default='0',
    )
    # Markdown-ish description scraped from the upstream dataset card.
    # For HF imports we strip the YAML frontmatter and stash the body
    # here so /dataset/<id> can show it under a collapsed "Details"
    # section without an extra HTTP call per page load. Backfilled
    # idempotently in check_and_migrate_db for legacy HF rows.
    card_description = db.Column(db.Text, nullable=True)

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

    @property
    def fields(self):
        """All DatasetField schema rows for this dataset, in their
        original insertion order (FK relationship below). Template
        shortcut so we don't have to expose the model class to Jinja."""
        return list(self.dataset_fields or [])

    @property
    def category_list(self) -> list[str]:
        """Multi-category support: Dataset.category is stored as a
        CSV string so a dataset can declare itself relevant to several
        Area/Task buckets (e.g. an image-segmentation dataset that
        ALSO has depth gt → ['Vision/Image Segmentation',
        'Vision/Depth Estimation']). Returns [] when category is
        unset; otherwise the trimmed, deduped list in declared order."""
        raw = (self.category or '').strip()
        if not raw:
            return []
        seen, out = set(), []
        for c in raw.split(','):
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out


class DatasetField(db.Model):
    """One row of typed schema per (dataset, field name).

    This is the source of truth for what data each sample carries —
    `CustomField` rows hold the per-sample VALUES, but the kind +
    params + role come from here. A leaderboard's prediction contract
    is then derivable by mirroring every attached dataset's `role='gt'`
    field as a pred field of the same kind.
    """
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(
        db.Integer, db.ForeignKey('dataset.id'),
        nullable=False, index=True,
    )
    name = db.Column(db.String(100), nullable=False)
    kind = db.Column(db.String(20), nullable=False)  # benchhub.types.DTYPES key
    params = db.Column(db.Text, nullable=True)        # JSON dict; NULL ⇒ {}
    role = db.Column(db.String(10), nullable=False)   # 'input' | 'gt'

    __table_args__ = (
        db.UniqueConstraint('dataset_id', 'name', name='uq_dataset_field_name'),
    )

    dataset = db.relationship('Dataset', backref=db.backref(
        'dataset_fields', lazy=True, cascade='all, delete-orphan',
        order_by='DatasetField.id',
    ))

    def get_params(self) -> dict:
        if not self.params:
            return {}
        try:
            v = json.loads(self.params)
            return v if isinstance(v, dict) else {}
        except (TypeError, ValueError):
            return {}

    def set_params(self, params: dict | None) -> None:
        if not params:
            self.params = None
        else:
            self.params = json.dumps(params, separators=(',', ':'))


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

    custom_fields = db.relationship('CustomField', backref='sample', lazy=True, cascade="all, delete-orphan", foreign_keys='CustomField.sample_id')

class CustomField(db.Model):
    """One field of typed data attached to a Sample (GT) or Submission (pred).

    The `data_type` string is one of `benchhub.types.DTYPES.kind` — the
    same name that resolves to a `DataType` subclass (`Image`, `Depth`,
    `Mask`, `Audio`, `Text`, `BBoxes`, `Label`, `Scalar`, `Json`).
    `data_params` is a JSON-encoded dict carrying per-instance metadata
    that the type class needs to decode the value back (e.g. Depth's
    `unit`, BBoxes' `format`, Mask's `num_classes` + `ignore_index`).
    Empty/NULL = `{}`."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    data_type = db.Column(db.String(20), nullable=False, index=True)  # benchhub.types kind
    data_params = db.Column(db.Text, nullable=True)  # JSON dict; NULL ⇒ {}
    value_text = db.Column(db.Text, nullable=True)  # file path OR inline-stored value (text, label, json)
    value_float = db.Column(db.Float, nullable=True)  # inline-stored scalar
    sample_id = db.Column(db.Integer, db.ForeignKey('sample.id'), nullable=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=True)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=True, index=True)
    sample_name = db.Column(db.String(100), nullable=True)
    source_column = db.Column(db.String(120), nullable=True)

    def get_value(self):
        """Inline value for scalar/metric/label rows; file path for file-backed kinds."""
        if self.data_type in ('scalar', 'metric') and self.value_float is not None:
            return self.value_float
        return self.value_text

    def get_params(self) -> dict:
        """Deserialize `data_params` JSON. Returns `{}` when NULL/blank/invalid."""
        raw = self.data_params
        if not raw:
            return {}
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except (TypeError, ValueError):
            return {}

    def set_params(self, params: dict | None) -> None:
        """Write `data_params`. Stores compact JSON; NULL when params is empty."""
        if not params:
            self.data_params = None
        else:
            self.data_params = json.dumps(params, separators=(',', ':'))


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
    # Was column-level UNIQUE; now relaxed so private metrics can
    # coexist with the same name across users (each user has their
    # own slot). Uniqueness is enforced via two indexes added in
    # check_and_migrate_db: a composite (owner_user_id, name) and a
    # partial unique on name WHERE visibility='public'.
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    python_code = db.Column(db.Text, nullable=False)  # The function definition: def metric_func(...)
    is_aggregated = db.Column(db.Boolean, default=False, nullable=False)
    accepts_aggregated_inputs = db.Column(db.Boolean, default=False)
    # JSON array of `target_kind` strings this metric accepts as inputs,
    # in argument order (`['mask', 'mask']` = IoU on two masks;
    # `['image', 'image']` = FID on two image populations). NULL means
    # "unconstrained" (legacy metrics where we haven't declared the
    # contract yet). The LB editor's metric→field binding filters
    # available GT/pred columns by these kinds.
    input_kinds = db.Column(db.Text, nullable=True)
    # Parallel JSON array (same length / order as `input_kinds`) of
    # the expected dataset-field role per arg: `gt`, `pred`, or
    # `input`. NULL ⇒ unconstrained, the LB picker accepts any role.
    # Drives the per-arg field dropdown filter on the LB-creation
    # form so metric `accuracy(gt, pred)` can only bind `gt` to a
    # role=gt field and `pred` to a role=pred (or auto-derived) field.
    input_roles = db.Column(db.Text, nullable=True)
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

class FeatureRequest(db.Model):
    """User-submitted requests for new features, data types, or
    visualisations the platform doesn't yet support. Admins triage
    these via /admin/feature_requests. Lightweight: free-form
    title/description, no comments thread."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),
                        nullable=False, index=True)
    kind = db.Column(db.String(30), nullable=False, default='feature',
                     server_default='feature')
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), nullable=False, default='open',
                       server_default='open')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    admin_note = db.Column(db.Text, nullable=True)
    user = db.relationship('User', foreign_keys=[user_id])


class LbTemplate(db.Model):
    """Admin-editable LB starter template. Drives the chip picker on
    `/dataset/<id>/create_lb` — clicking a chip pre-fills metrics +
    pred-field schema for the named task. The fallback `LB_TEMPLATES`
    Python dict seeds this table on first boot; admins can edit
    afterwards without a code change."""
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(60), nullable=False, unique=True)
    label = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    # JSON arrays of strings. Required kinds drive the picker
    # filter; metric_names + visualization_names reference catalog
    # rows by GlobalMetric.name / GlobalVisualization.name.
    required_kinds_json = db.Column(db.Text, nullable=False, default='[]',
                                     server_default='[]')
    metric_names_json = db.Column(db.Text, nullable=False, default='[]',
                                   server_default='[]')
    visualization_names_json = db.Column(db.Text, nullable=False, default='[]',
                                          server_default='[]')
    enabled = db.Column(db.Boolean, nullable=False, default=True,
                        server_default='1')
    sort_order = db.Column(db.Integer, nullable=False, default=0,
                           server_default='0')

    def required_kinds(self) -> list[str]:
        try:
            v = json.loads(self.required_kinds_json or '[]')
            return v if isinstance(v, list) else []
        except (TypeError, ValueError):
            return []

    def metric_names(self) -> list[str]:
        try:
            v = json.loads(self.metric_names_json or '[]')
            return v if isinstance(v, list) else []
        except (TypeError, ValueError):
            return []

    def visualization_names(self) -> list[str]:
        try:
            v = json.loads(self.visualization_names_json or '[]')
            return v if isinstance(v, list) else []
        except (TypeError, ValueError):
            return []


class GlobalVisualization(db.Model):
    """Global visualization definition (analogous to GlobalMetric but returns PIL.Image)"""
    id = db.Column(db.Integer, primary_key=True)
    # Same uniqueness semantics as GlobalMetric.name: relaxed so private
    # rows can coexist; partial unique on name WHERE visibility='public'
    # + composite (owner_user_id, name) enforced via the migration block.
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    python_code = db.Column(db.Text, nullable=False)  # Function definition: def viz_func(...) -> PIL.Image
    is_aggregated = db.Column(db.Boolean, default=False, nullable=False)  # True: single image, False: per-sample
    accepts_aggregated_inputs = db.Column(db.Boolean, default=False)
    # Same shape as GlobalMetric.input_kinds — JSON array of accepted
    # `target_kind`s in argument order.
    input_kinds = db.Column(db.Text, nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    # Phase 1 Slice 4 multi-tenancy.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    visibility = db.Column(db.String(20), nullable=False, default='public', server_default='public')
    owner = db.relationship('User', foreign_keys=[owner_user_id])

class LeaderboardMaterialization(db.Model):
    """Frozen full-resolution snapshot taken when an LB is created
    from a preview-only Dataset.

    Each LB picks how many samples + how to sample them; the
    materializer fetches those samples from upstream HF at full
    resolution and stores them at
    `uploads/lb_materializations/<lb_id>/<field>/<sample>.<ext>`.
    Submission scoring reads the full bytes from here; the dataset's
    preview tier stays untouched.

    Quota cost lands on the LB owner.
    """
    id = db.Column(db.Integer, primary_key=True)
    leaderboard_id = db.Column(
        db.Integer, db.ForeignKey('leaderboard.id'),
        nullable=False, unique=True, index=True,
    )
    sample_cap = db.Column(db.Integer, nullable=False)
    # 'head' | 'random' | 'stratified'. 'random' is the default for
    # everything except classification, where 'stratified' is preferred.
    sampling = db.Column(db.String(20), nullable=False, default='random')
    sampling_seed = db.Column(db.Integer, nullable=False, default=42)
    # Field name used for stratification (e.g. 'label'). NULL for
    # non-stratified sampling.
    stratify_field = db.Column(db.String(100), nullable=True)
    materialized_at = db.Column(db.DateTime, nullable=True)
    storage_bytes = db.Column(db.BigInteger, nullable=True)
    status = db.Column(
        db.String(20), nullable=False,
        default='pending', server_default='pending',
    )  # 'pending' | 'running' | 'ready' | 'failed'
    error_message = db.Column(db.Text, nullable=True)
    # Live progress payload; same shape as Dataset.import_progress_json:
    # {'phase': str, 'current': int, 'total': int, 'message': str}.
    # Updated by the Celery worker; polled by the LB page's JS.
    progress_json = db.Column(db.Text, nullable=True)
    leaderboard = db.relationship('Leaderboard', backref=db.backref(
        'materialization', lazy=True, uselist=False, cascade='all, delete-orphan',
    ))


class DatasetRequest(db.Model):
    """Community wishlist row: "please import this HF dataset."

    Anyone signed in can file one (just a repo_id + optional reason);
    anyone signed in can upvote. Decoupled from any user's own
    datasets — this is the queue admins triage when deciding which
    HF repos to import next.
    """
    id = db.Column(db.Integer, primary_key=True)
    repo_id = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    requested_by_user_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=True
    )
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # 'pending' | 'imported' | 'rejected'. Filtering on the list page
    # defaults to pending so the queue stays tractable.
    status = db.Column(db.String(20), nullable=False,
                       default='pending', server_default='pending')
    admin_note = db.Column(db.Text, nullable=True)
    requester = db.relationship('User', foreign_keys=[requested_by_user_id])


class DatasetRequestVote(db.Model):
    """One vote = (request, user) pair. Enforced unique so a user
    can't double-vote. Deleting the row revokes the vote."""
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(
        db.Integer, db.ForeignKey('dataset_request.id'),
        nullable=False, index=True,
    )
    user_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=False, index=True,
    )
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (
        db.UniqueConstraint('request_id', 'user_id',
                             name='uq_dataset_request_vote_user'),
    )
    request = db.relationship('DatasetRequest', backref=db.backref(
        'votes', lazy='dynamic', cascade='all, delete-orphan',
    ))


class DatasetVisualization(db.Model):
    """Link table: Dataset ↔ GlobalVisualization.

    Mirrors LeaderboardVisualization but bound to a Dataset. Drives
    two things:
      1. The visualization list on /dataset/<id>/settings (where the
         owner picks + configures viz attachments + arg mappings).
      2. Inheritance: when a Leaderboard is created from a Dataset,
         each row here is copied into a LeaderboardVisualization for
         the new LB so the LB inherits its dataset's visualisations
         without manual re-attachment.
    """
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(
        db.Integer, db.ForeignKey('dataset.id'),
        nullable=False, index=True,
    )
    global_visualization_id = db.Column(
        db.Integer, db.ForeignKey('global_visualization.id'),
        nullable=False, index=True,
    )
    # JSON object mapping viz function arg name → dataset field key
    # (uses the same `gt_<field>` convention as LeaderboardVisualization
    # so the inheritance copy is verbatim).
    arg_mappings = db.Column(db.Text, nullable=False)
    target_name = db.Column(db.String(100), nullable=True)
    display_order = db.Column(db.Integer, default=0)
    dataset = db.relationship('Dataset', backref=db.backref(
        'dataset_visualizations', lazy=True, cascade='all, delete-orphan',
        order_by='DatasetVisualization.display_order',
    ))
    global_visualization = db.relationship(
        'GlobalVisualization', backref='dataset_usages'
    )


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
    # Per-LB role overrides for attached dataset fields. The Dataset
    # is now role-neutral — the same dataset can back multiple LBs
    # with different role assignments (one LB treats `depth` as gt,
    # another uses it as input for a colorization task). JSON dict:
    # `{dataset_field_name: 'input'|'gt'|'skip'}`. NULL or missing
    # entries fall back to `DatasetField.role` as a sensible default.
    field_roles_json = db.Column(db.Text, nullable=True)
    # Two-level taxonomy "area/task" (e.g. "Vision/Depth Estimation").
    # Auto-populated for PWC imports from the source task name via
    # `_pwc_task_to_category`; manual LBs can override via the
    # Settings page. /explore renders a tree sidebar by splitting on '/'.
    category = db.Column(db.String(120), nullable=True, index=True)
    # Phase 1 multi-tenancy.
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    visibility = db.Column(db.String(20), nullable=False, default='public', server_default='public')
    # GitHub gist path "<owner>/<id>" holding this LB's submission
    # notebook. Created / updated lazily on click of "Open in Colab";
    # reused across clicks so we don't proliferate gists. NULL = never
    # been opened in Colab. Column is varchar(128) — GitHub login is
    # up to 39 chars, gist id is 32 chars, plus the slash.
    colab_gist_id = db.Column(db.String(128), nullable=True)
    owner = db.relationship('User', foreign_keys=[owner_user_id])
    submissions = db.relationship('Submission', backref='leaderboard', lazy=True, cascade="all, delete-orphan")
    datasets = db.relationship('Dataset', secondary=leaderboard_datasets, backref=db.backref('leaderboards', lazy='dynamic'))
    tags = db.relationship('Tag', secondary=leaderboard_tags, lazy='subquery',
                           backref=db.backref('leaderboards', lazy=True))
    collaborators = db.relationship('User', secondary=leaderboard_shares, lazy='subquery',
                                    backref=db.backref('shared_leaderboards', lazy=True))


class SubmissionToken(db.Model):
    """Short-lived, single-LB submission token. Minted server-side when
    a user clicks "Open in Colab" so the generated notebook can submit
    without us having to leak the user's long-lived `User.api_token`
    through the public gist. Behaves identically to `User.api_token` at
    /api/submit/<lb_id> time — the auth shim accepts either — but is
    scoped down: only valid for one leaderboard, expires after
    ~1 hour, and bumps `used_count` per submission so abuse stays
    cheap to revoke."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),
                        nullable=False, index=True)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'),
                               nullable=False, index=True)
    token = db.Column(db.String(64), nullable=False, unique=True, index=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    used_count = db.Column(db.Integer, nullable=False, default=0,
                           server_default='0')
    revoked = db.Column(db.Boolean, nullable=False, default=False,
                        server_default='0')
    user = db.relationship('User', foreign_keys=[user_id])
    leaderboard = db.relationship('Leaderboard', foreign_keys=[leaderboard_id])


def _mint_submission_token(user, lb, *, hours: int = 1) -> 'SubmissionToken':
    """Create a fresh single-use submission token. Caller commits."""
    import secrets
    row = SubmissionToken(
        user_id=user.id,
        leaderboard_id=lb.id,
        token='bhst_' + secrets.token_urlsafe(32),
        expires_at=datetime.utcnow() + timedelta(hours=hours),
    )
    db.session.add(row)
    db.session.commit()
    return row


def _resolve_api_token(raw: str) -> tuple['User | None', 'SubmissionToken | None']:
    """Lookup a bearer token. Returns (user, submission_token_or_None).
    A long-lived User.api_token resolves to (user, None). A short-lived
    SubmissionToken resolves to (user, st) so the caller can also
    enforce per-LB scope and bump used_count."""
    if not raw:
        return None, None
    if raw.startswith('bhst_'):
        st = SubmissionToken.query.filter_by(token=raw, revoked=False).first()
        if st is None or st.expires_at < datetime.utcnow():
            return None, None
        return st.user, st
    user = User.query.filter_by(api_token=raw).first()
    return user, None


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    leaderboard_id = db.Column(db.Integer, db.ForeignKey('leaderboard.id'), nullable=False)
    git_commit = db.Column(db.String(100))
    git_branch = db.Column(db.String(100))
    git_message = db.Column(db.String(200))
    git_author = db.Column(db.String(100))  # Git commit author
    # Free-text blurb the submitter passes via the client (shown in the
    # submission table where the git column used to be; full text on hover).
    description = db.Column(db.Text, nullable=True)
    # Optional external URL the submitter attaches; the submission name
    # in the table hyperlinks to it. http/https only (sanitised server-side).
    link = db.Column(db.String(500), nullable=True)
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
    #
    # Two-bucket quota (Phase 13 split):
    #   * quota_public_max_bytes  — datasets + LB materialisations whose
    #     visibility == 'public'. Higher cap because shared content gets a
    #     bigger budget.
    #   * quota_private_max_bytes — visibility in {'private','unlisted'}.
    #     Tighter cap; flipping a row to public on the Publish button
    #     pre-flights the public bucket.
    # The legacy quota_max_storage_bytes column stays around for backward-
    # compat (old admin tools + the migration's first boot still read it);
    # the split fields are the source of truth for check_quota now.
    quota_max_storage_bytes = db.Column(
        db.BigInteger, nullable=False, default=10 * 1024 ** 3,
        server_default=str(10 * 1024 ** 3),
    )  # 10 GB — legacy total; superseded by the two split caps below
    quota_public_max_bytes = db.Column(
        db.BigInteger, nullable=False, default=100 * 1024 ** 3,
        server_default=str(100 * 1024 ** 3),
    )  # 100 GB default for everything visibility='public'
    quota_private_max_bytes = db.Column(
        db.BigInteger, nullable=False, default=10 * 1024 ** 3,
        server_default=str(10 * 1024 ** 3),
    )  # 10 GB default for private + unlisted
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


class EmailLoginCode(db.Model):
    """Passwordless email sign-in / sign-up. A short-lived 6-digit code is
    emailed to verify the user owns an address; no password is ever stored.
    Brand-new table → created by db.create_all() at boot, no manual
    migration block needed."""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    code_hash = db.Column(db.String(64), nullable=False)  # sha256 of the code
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=0, server_default='0')
    consumed = db.Column(db.Boolean, nullable=False, default=False, server_default='0')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


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


def _gt_stats_custom_fields(sample):
    """Values for the comparison view's GT-stats panel: scalar + metric
    GT fields (value_float) plus label / label_list GT fields (decoded
    from the JSON value_text). Submission-side CustomFields are skipped
    (sample.custom_fields are GT-only when submission_id is None)."""
    out = {}
    for cf in sample.custom_fields:
        if cf.submission_id is not None:
            continue
        if cf.data_type in ('scalar', 'metric'):
            out[cf.name] = cf.value_float
        elif cf.data_type in ('label', 'label_list'):
            try:
                out[cf.name] = json.loads(cf.value_text) if cf.value_text else None
            except Exception:
                out[cf.name] = cf.value_text
    return out


# Canonical folder-name → data_type prefix convention.
# Each known kind has a prefix; a folder named `<prefix>_<field_name>`
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
        if cf.data_type == 'image':
            # Store image path
            pred_data[cf.name] = cf.value_text
        elif cf.data_type in ('label', 'label_list'):
            # Inline-stored label (int/str) or top-K list — decode the
            # JSON so the prediction-stats panel shows the value, not a
            # raw JSON string.
            try:
                pred_data[cf.name] = json.loads(cf.value_text) if cf.value_text else None
            except Exception:
                pred_data[cf.name] = cf.value_text
        elif cf.data_type in ['scalar', 'metric']:
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
            'scalar_width': '120px',
            'image_width': '180px',   # matches the 160px cell + padding
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
        return self.settings.get('scalar_width', '120px')

    @property
    def image_width(self):
        return self.settings.get('image_width', '180px')

    @property
    def theme_mode(self):
        return self.settings.get('theme_mode', 'dark')

global_settings = GlobalSettings()

@app.context_processor
def inject_settings():
    # Per-user theme preference from cookie
    user_theme = request.cookies.get('theme_mode')
    
    # Create a wrapper or use the object directly but inject the user preference
    # NOTE on widths: the per-cell content was tightened to 160×140
    # (images, depth, mask, json, text) and 180×140 (charts). The
    # column widths here are the TH widths — they should be ~20px
    # wider than the cell to leave room for borders/padding. We
    # ignore any pre-existing cookie that pinned a wider value
    # ("user resized older table column"), since the old 300px
    # default left the column visibly wider than the image, which
    # the user flagged as ugly.
    def _clamped(cookie_val, default, max_px):
        try:
            if cookie_val and cookie_val.endswith('px'):
                if int(cookie_val[:-2]) > max_px:
                    return default
            return cookie_val or default
        except Exception:
            return default
    settings_dict = {
        'scalar_width': _clamped(request.cookies.get('scalar_width'),
                                  global_settings.scalar_width, 150),
        'image_width':  _clamped(request.cookies.get('image_width'),
                                  global_settings.image_width, 220),
        'metric_chart_width': _clamped(request.cookies.get('metric_chart_width'),
                                        '200px', 240),
        'tags_width':   _clamped(request.cookies.get('tags_width'),
                                  '140px', 200),
        'config_width': _clamped(request.cookies.get('config_width'),
                                  '140px', 200),
        'name_width':   _clamped(request.cookies.get('name_width'),
                                  '140px', 200),
        'histogram_width': _clamped(request.cookies.get('histogram_width'),
                                     '160px', 220),
        'theme_mode': user_theme if user_theme in ['light', 'dark'] else global_settings.theme_mode
    }
    return {'global_settings': settings_dict}

# --- End Global Settings ---
# (The /app-settings page was dropped — column widths use the cookie-
# clamped defaults wired into the inject_settings context processor
# above, and the theme is toggled via the navbar button which writes
# the `theme_mode` cookie directly. The GlobalSettings class is kept
# because the context processor still pulls defaults from it.)

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
    - **everything for admins** (private + unlisted included) so the
      admin index pages can surface unlisted rows that would otherwise
      be invisible

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

    # Admins see everything — keep them out of the visibility funnel
    # so the catalog reflects the full state of the system.
    if is_admin(user):
        return or_(public_or_legacy,
                   model_cls.owner_user_id.isnot(None))

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

def _visibility_bucket(visibility):
    """Map a row's visibility string onto the two quota buckets.
    'public' → 'public'; everything else (private, unlisted, NULL) →
    'private'. Centralised so call sites don't have to think about it."""
    return 'public' if visibility == 'public' else 'private'


def storage_used_bytes(user, *, visibility=None):
    """Bytes this user currently consumes on the data volume.
    Sums `Dataset.storage_bytes` over the user's datasets plus
    `LeaderboardMaterialization.storage_bytes` over LBs they own (LB
    owners pay for the full-resolution subset they chose). Submission
    ZIPs are still not counted — the LB owner already paid for the
    materialised inputs the submitter is responding to.

    `visibility`:
      - None      → total across both buckets (legacy callers)
      - 'public'  → only rows whose own visibility is 'public'
      - 'private' → rows whose visibility is private/unlisted/NULL
    Treats NULL bytes as 0."""
    if user is None:
        return 0
    bucket = _visibility_bucket(visibility) if visibility else None

    ds_q = db.session.query(func.coalesce(func.sum(Dataset.storage_bytes), 0)).filter(
        Dataset.owner_user_id == user.id
    )
    if bucket == 'public':
        ds_q = ds_q.filter(Dataset.visibility == 'public')
    elif bucket == 'private':
        ds_q = ds_q.filter(
            (Dataset.visibility != 'public') | (Dataset.visibility.is_(None))
        )
    ds_total = int(ds_q.scalar() or 0)

    mat_q = (
        db.session.query(func.coalesce(func.sum(LeaderboardMaterialization.storage_bytes), 0))
        .join(Leaderboard, Leaderboard.id == LeaderboardMaterialization.leaderboard_id)
        .filter(Leaderboard.owner_user_id == user.id)
    )
    if bucket == 'public':
        mat_q = mat_q.filter(Leaderboard.visibility == 'public')
    elif bucket == 'private':
        mat_q = mat_q.filter(
            (Leaderboard.visibility != 'public') | (Leaderboard.visibility.is_(None))
        )
    mat_total = int(mat_q.scalar() or 0)

    return ds_total + mat_total


def quota_cap_for(user, visibility):
    """The byte cap that applies to a write of `visibility` for `user`.
    Falls back to the legacy single-bucket value if the split column is
    NULL/0 (which can happen briefly during a half-applied migration)."""
    bucket = _visibility_bucket(visibility)
    if bucket == 'public':
        v = int(getattr(user, 'quota_public_max_bytes', 0) or 0)
        if v > 0:
            return v
    else:
        v = int(getattr(user, 'quota_private_max_bytes', 0) or 0)
        if v > 0:
            return v
    return int(getattr(user, 'quota_max_storage_bytes', 0) or 0)


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


@app.route('/me/usage')
@login_required
def user_usage():
    """Per-user storage breakdown + quota. Shows the two split caps
    (public bucket vs private/unlisted bucket), per-dataset rows
    ordered by size, and submission-rate state."""
    user = g.current_user
    ds_rows = (
        Dataset.query
        .filter(Dataset.owner_user_id == user.id)
        .order_by(Dataset.storage_bytes.desc().nullslast())
        .all()
    )
    used_public = storage_used_bytes(user, visibility='public')
    used_private = storage_used_bytes(user, visibility='private')
    cap_public = quota_cap_for(user, 'public')
    cap_private = quota_cap_for(user, 'private')
    items = []
    for ds in ds_rows:
        ds_bucket = _visibility_bucket(getattr(ds, 'visibility', None))
        ds_cap = cap_public if ds_bucket == 'public' else cap_private
        items.append({
            'id': ds.id,
            'name': ds.name,
            'category': ds.category,
            'visibility': getattr(ds, 'visibility', 'private'),
            'bytes': int(ds.storage_bytes or 0),
            'pct': (
                float(ds.storage_bytes or 0) * 100.0 / max(int(ds_cap or 1), 1)
            ),
        })
    used = used_public + used_private
    cap = cap_public + cap_private
    # Admins bypass every quota cap in `check_quota` (commit f36fe8f),
    # so the storage / dataset-count / submission-rate columns above
    # are cosmetic for them. Surface that to the template so the
    # progress bar shows "unlimited" instead of the underlying 50 MB
    # row value that confused users by suggesting a cap that
    # doesn't actually apply.
    is_admin_user = is_admin(user)
    return render_template(
        'user_usage.html',
        items=items,
        used_bytes=used,
        cap_bytes=cap,
        used_pct=(used * 100.0 / max(cap, 1)),
        used_human=_format_bytes(used),
        cap_human=_format_bytes(cap),
        used_public_bytes=used_public,
        used_public_human=_format_bytes(used_public),
        cap_public_bytes=cap_public,
        cap_public_human=_format_bytes(cap_public),
        used_public_pct=(used_public * 100.0 / max(cap_public, 1)),
        used_private_bytes=used_private,
        used_private_human=_format_bytes(used_private),
        cap_private_bytes=cap_private,
        cap_private_human=_format_bytes(cap_private),
        used_private_pct=(used_private * 100.0 / max(cap_private, 1)),
        dataset_count=len(items),
        ds_count_cap=user.quota_max_datasets,
        sub_count_24h=daily_submission_count(user),
        sub_count_cap=user.quota_max_submissions_per_day,
        unlimited=is_admin_user,
    )


def check_quota(user, *, kind, incoming_bytes=0, visibility='private'):
    """Return (ok, message). `kind` is one of:
       - 'dataset_create'  : count cap + storage cap (incoming_bytes,
                             charged to the bucket implied by `visibility`)
       - 'submission'      : daily rate cap (visibility is ignored)

    `visibility` is the visibility of the *row being written* — public
    writes charge the public bucket, everything else charges the private
    bucket. Defaults to 'private' so callers that pre-date the split keep
    failing-safe on the smaller bucket.
    """
    if user is None:
        return False, "Sign in required."

    # Admins (BENCHHUB_ADMIN_EMAILS / is_admin) are exempt from
    # quota caps — they're the ones seeding curated datasets +
    # bulk-importing reference benchmarks, which routinely exceed
    # the 50 MB user cap. Regular users still get the limits.
    if is_admin(user):
        return True, None

    if kind == 'dataset_create':
        if dataset_count(user) >= user.quota_max_datasets:
            return False, (
                f"Dataset limit reached ({user.quota_max_datasets}). "
                "Delete an existing dataset or contact us for a higher cap."
            )
        bucket = _visibility_bucket(visibility)
        used = storage_used_bytes(user, visibility=bucket)
        cap = quota_cap_for(user, bucket)
        if used + incoming_bytes > cap:
            bucket_label = 'public' if bucket == 'public' else 'private/unlisted'
            return False, (
                f"{bucket_label.capitalize()} storage limit would be exceeded: "
                f"{_format_bytes(used)} used + {_format_bytes(incoming_bytes)} new "
                f"> {_format_bytes(cap)} cap."
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
    work the same as on cookie-authed routes.

    Accepts either:
      • a long-lived User.api_token (Settings → API tokens)
      • a short-lived SubmissionToken (prefix `bhst_`) minted by the
        Open-in-Colab flow. Stashed on g.submission_token so a wrapped
        view can also enforce per-LB scope (see /api/submit/<lb_id>).
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        token = _bearer_token_from_request()
        if not token:
            return jsonify({'error': 'API token required (Authorization: Bearer <token>)'}), 401
        user, sub_token = _resolve_api_token(token)
        if user is None:
            return jsonify({'error': 'Invalid or expired API token'}), 401
        g.current_user = user
        g.submission_token = sub_token   # None for long-lived tokens
        return view(*args, **kwargs)
    return wrapped


@app.route('/api/whoami')
@require_api_token
def api_whoami():
    """Cheap auth-validation endpoint used by the Colab notebook to
    decide whether the bearer token is still good before submitting.
    A 200 means: token valid, user known, and (for SubmissionTokens)
    not expired and not revoked."""
    st = g.get('submission_token')
    return jsonify({
        'user_id': g.current_user.id,
        'display_name': g.current_user.display_name,
        'token_kind': 'submission' if st else 'api',
        'leaderboard_id': st.leaderboard_id if st else None,
        'expires_at': st.expires_at.isoformat() + 'Z' if st else None,
    })


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
        # Cross-provider account link: if the same email already belongs
        # to a user (e.g. they previously signed in with Google), link
        # the GitHub identity to that row instead of inserting a
        # duplicate (User.email is UNIQUE).
        existing = User.query.filter(func.lower(User.email) == email.lower()).first()
        if existing is not None:
            existing.oauth_provider = 'github'
            existing.oauth_sub = oauth_sub
            existing.display_name = profile.get('name') or profile.get('login') or existing.display_name
            existing.avatar_url = profile.get('avatar_url') or existing.avatar_url
            user = existing
        else:
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


@app.route('/login/google')
def login_google():
    """OIDC-flow login via Google. Configure with GOOGLE_CLIENT_ID +
    GOOGLE_CLIENT_SECRET (Fly secrets in prod). Authorized redirect
    URI on the Google Cloud Console: <site>/oauth/callback/google."""
    if not os.environ.get('GOOGLE_CLIENT_ID') or not os.environ.get('GOOGLE_CLIENT_SECRET'):
        return ("Google OAuth not configured: set GOOGLE_CLIENT_ID and "
                "GOOGLE_CLIENT_SECRET (env vars or Fly secrets)."), 503
    session['oauth_next'] = request.args.get('next') or url_for('home')
    redirect_uri = url_for('oauth_callback_google', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/oauth/callback/google')
def oauth_callback_google():
    try:
        token = oauth.google.authorize_access_token()
    except Exception as e:
        flash(f"Google login failed: {e}", "danger")
        return redirect(url_for('login'))
    # Authlib + Google OIDC parses the id_token into the token dict.
    profile = (token.get('userinfo') if isinstance(token, dict) else None) or {}
    if not profile:
        try:
            profile = oauth.google.parse_id_token(token, nonce=None)
        except Exception:
            profile = {}
    oauth_sub = str(profile.get('sub') or '')
    email = profile.get('email')
    if not oauth_sub or not email:
        flash("Google login succeeded but no email/sub was returned.", "warning")
        return redirect(url_for('login'))

    user = User.query.filter_by(oauth_provider='google', oauth_sub=oauth_sub).first()
    if user is None:
        # Cross-provider account link: if the email already belongs to
        # a user (e.g. GitHub login last time), point THAT row at the
        # Google identity instead of inserting a duplicate row (which
        # would trip the User.email UNIQUE constraint). This is the
        # standard "same email → same account" merge.
        existing = User.query.filter(func.lower(User.email) == email.lower()).first()
        if existing is not None:
            existing.oauth_provider = 'google'
            existing.oauth_sub = oauth_sub
            existing.display_name = profile.get('name') or existing.display_name
            existing.avatar_url = profile.get('picture') or existing.avatar_url
            user = existing
        else:
            user = User(
                email=email,
                display_name=profile.get('name') or email.split('@')[0],
                avatar_url=profile.get('picture'),
                oauth_provider='google',
                oauth_sub=oauth_sub,
            )
            db.session.add(user)
    else:
        user.email = email
        user.display_name = profile.get('name') or user.display_name
        user.avatar_url = profile.get('picture') or user.avatar_url
    user.last_login_at = datetime.utcnow()
    if (user.email or '').strip().lower() in _admin_emails() and not user.is_admin:
        user.is_admin = True
    db.session.commit()

    session['user_id'] = user.id
    flash(f"Logged in as {user.display_name}.", "success")
    return redirect(session.pop('oauth_next', None) or url_for('home'))


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('oauth_next', None)
    flash("Logged out.", "info")
    return redirect(url_for('login'))


# ===================== Passwordless email sign-in (verification code) =====

def _send_email(to_email, subject, body):
    """Send a plain-text email via SMTP. Reads SMTP_HOST / SMTP_PORT /
    SMTP_USER / SMTP_PASS / MAIL_FROM from the environment. Returns True
    on success. When SMTP isn't configured (no SMTP_HOST), logs and
    returns False — the caller decides how to degrade (dev shows the code
    on-page; prod tells the user to retry)."""
    def _clean(name):
        # systemd EnvironmentFile has no inline-comment support — the whole
        # value after `=` is kept. Be forgiving anyway: drop a trailing
        # " # comment" and surrounding quotes/space so a copy-paste typo in
        # .env degrades to a logged failure, not an uncaught 500.
        v = os.environ.get(name)
        if v is None:
            return None
        return v.split(' #', 1)[0].strip().strip('"').strip("'") or None

    host = _clean('SMTP_HOST')
    sender = (_clean('MAIL_FROM') or _clean('SMTP_USER')
              or 'no-reply@runbenchhub.com')
    if not host:
        app.logger.warning("SMTP not configured; email to %s NOT sent (subject=%r)",
                           to_email, subject)
        return False
    raw_port = _clean('SMTP_PORT') or '587'
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        app.logger.warning("Invalid SMTP_PORT %r; falling back to 587", raw_port)
        port = 587
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to_email
    msg.set_content(body)
    user = _clean('SMTP_USER')
    pw = os.environ.get('SMTP_PASS')  # passwords may legitimately contain '#'
    try:
        if port == 465:
            srv = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            srv = smtplib.SMTP(host, port, timeout=15)
            srv.starttls()
        if user and pw:
            srv.login(user, pw)
        srv.send_message(msg)
        srv.quit()
        return True
    except Exception as e:
        app.logger.error("Failed to send email to %s: %s", to_email, e)
        return False


def _hash_login_code(code):
    import hashlib
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


@app.route('/login/email', methods=['POST'])
def login_email_request():
    """Step 1 of passwordless login: user submits an email, we mint a
    6-digit code, email it, and send them to the verify page."""
    email = (request.form.get('email') or '').strip().lower()
    next_url = (request.form.get('next') or '').strip()
    if not email or '@' not in email or '.' not in email.split('@')[-1]:
        flash("Enter a valid email address.", "warning")
        return redirect(url_for('login', next=next_url))

    # Invalidate any outstanding codes for this address (one live code).
    EmailLoginCode.query.filter_by(email=email, consumed=False).update(
        {'consumed': True}, synchronize_session=False)
    code = f"{secrets.randbelow(1_000_000):06d}"
    db.session.add(EmailLoginCode(
        email=email,
        code_hash=_hash_login_code(code),
        expires_at=datetime.utcnow() + timedelta(minutes=10),
    ))
    db.session.commit()

    sent = _send_email(
        email,
        "Your BenchHub sign-in code",
        f"Your BenchHub verification code is: {code}\n\n"
        f"It expires in 10 minutes. If you didn't request this, ignore this email.",
    )
    session['email_login_pending'] = email
    session['oauth_next'] = next_url or url_for('home')

    if not sent:
        if app.debug or app.config.get('TESTING'):
            # Dev / test convenience when SMTP isn't wired up.
            flash(f"Email delivery not configured — your code is {code}", "info")
        else:
            flash("We couldn't send the email right now. Try again shortly, "
                  "or use GitHub / Google.", "danger")
            return redirect(url_for('login', next=next_url))
    else:
        flash(f"We sent a 6-digit code to {email}. It expires in 10 minutes.", "success")
    return redirect(url_for('login_email_verify'))


@app.route('/login/email/verify', methods=['GET'])
def login_email_verify():
    email = session.get('email_login_pending')
    if not email:
        return redirect(url_for('login'))
    return render_template('login_verify.html', email=email,
                           next=session.get('oauth_next', ''))


@app.route('/login/email/verify', methods=['POST'])
def login_email_verify_post():
    """Step 2: validate the submitted code, then upsert + sign in."""
    email = session.get('email_login_pending')
    if not email:
        flash("Your sign-in session expired. Please start again.", "warning")
        return redirect(url_for('login'))
    code = (request.form.get('code') or '').strip()

    row = (EmailLoginCode.query
           .filter_by(email=email, consumed=False)
           .order_by(EmailLoginCode.id.desc())
           .first())
    if row is None or row.expires_at < datetime.utcnow():
        flash("That code expired. Request a new one.", "warning")
        return redirect(url_for('login'))
    if row.attempts >= 5:
        row.consumed = True
        db.session.commit()
        flash("Too many incorrect attempts. Request a new code.", "danger")
        return redirect(url_for('login'))
    if not code or _hash_login_code(code) != row.code_hash:
        row.attempts += 1
        db.session.commit()
        flash("Incorrect code. Please try again.", "danger")
        return redirect(url_for('login_email_verify'))

    # Verified. Upsert the account (email is now proven owned).
    row.consumed = True
    user = User.query.filter(func.lower(User.email) == email).first()
    if user is None:
        user = User(
            email=email,
            display_name=email.split('@')[0],
            oauth_provider='email',
            oauth_sub=email,
        )
        db.session.add(user)
    user.last_login_at = datetime.utcnow()
    if email in _admin_emails() and not user.is_admin:
        user.is_admin = True
    db.session.commit()

    session.pop('email_login_pending', None)
    session['user_id'] = user.id
    flash(f"Logged in as {user.display_name}.", "success")
    return redirect(session.pop('oauth_next', None) or url_for('home'))


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


@app.route('/admin/user_storage', methods=['GET'])
@login_required
def admin_user_storage():
    """Per-user storage breakdown for admins. Groups datasets by owner,
    sums `Dataset.storage_bytes` (the same counter `storage_used_bytes`
    walks for a single user), and lists submission counts so a bad
    actor blasting through the soft cap is visible in one place.

    Sort is bytes-desc by default, so the heaviest users float to the
    top — that's almost always what an admin's looking for here."""
    if not is_admin(g.current_user):
        abort(403)

    # One pass: join User ⋈ Dataset on owner; LEFT join so users with
    # zero datasets still appear (they may still be holding submission
    # rows or just be on the platform).
    rows = (
        db.session.query(
            User.id, User.email, User.display_name,
            User.is_admin, User.quota_max_storage_bytes,
            User.quota_public_max_bytes, User.quota_private_max_bytes,
            User.quota_max_datasets, User.quota_max_submissions_per_day,
            func.count(Dataset.id),
            func.coalesce(func.sum(Dataset.storage_bytes), 0),
        )
        .outerjoin(Dataset, Dataset.owner_user_id == User.id)
        .group_by(User.id)
        .all()
    )

    # Submission counts in the trailing 24h, joined client-side because
    # the query above already has dataset aggregates; mixing them in
    # one GROUP BY produces a cross-product.
    cutoff = datetime.utcnow() - timedelta(days=1)
    sub_24h = dict(
        db.session.query(
            Submission.owner_user_id,
            func.count(Submission.id),
        )
        .filter(Submission.upload_date >= cutoff)
        .group_by(Submission.owner_user_id)
        .all()
    )

    # Per-user public vs private dataset bytes — single grouped query
    # keyed by (owner, bucket) avoids round-tripping per user.
    ds_split = {}
    for uid, vis, total in db.session.query(
        Dataset.owner_user_id,
        Dataset.visibility,
        func.coalesce(func.sum(Dataset.storage_bytes), 0),
    ).group_by(Dataset.owner_user_id, Dataset.visibility).all():
        bucket = 'public' if vis == 'public' else 'private'
        ds_split.setdefault(uid, {'public': 0, 'private': 0})
        ds_split[uid][bucket] += int(total or 0)

    # LB materialisation bytes are charged to the LB owner, and the LB's
    # own visibility decides the bucket.
    mat_split = {}
    for uid, vis, total in db.session.query(
        Leaderboard.owner_user_id,
        Leaderboard.visibility,
        func.coalesce(func.sum(LeaderboardMaterialization.storage_bytes), 0),
    ).join(
        LeaderboardMaterialization,
        LeaderboardMaterialization.leaderboard_id == Leaderboard.id,
    ).group_by(Leaderboard.owner_user_id, Leaderboard.visibility).all():
        bucket = 'public' if vis == 'public' else 'private'
        mat_split.setdefault(uid, {'public': 0, 'private': 0})
        mat_split[uid][bucket] += int(total or 0)

    items = []
    total_bytes = 0
    for (uid, email, name, is_admin_flag, storage_cap,
         pub_cap, priv_cap, ds_cap, sub_cap, ds_count, storage_used) in rows:
        storage_used = int(storage_used or 0)
        total_bytes += storage_used
        used_pub = (ds_split.get(uid, {}).get('public', 0)
                    + mat_split.get(uid, {}).get('public', 0))
        used_priv = (ds_split.get(uid, {}).get('private', 0)
                     + mat_split.get(uid, {}).get('private', 0))
        items.append({
            'id': uid,
            'email': email or '',
            'display_name': name or email or f'user #{uid}',
            'is_admin': bool(is_admin_flag),
            'storage_used': storage_used,
            'storage_cap': int(storage_cap or 0),
            'public_used': used_pub,
            'public_cap': int(pub_cap or 0),
            'private_used': used_priv,
            'private_cap': int(priv_cap or 0),
            'dataset_count': int(ds_count or 0),
            'dataset_cap': int(ds_cap or 0),
            'sub_24h': int(sub_24h.get(uid, 0)),
            'sub_cap': int(sub_cap or 0),
        })

    sort = (request.args.get('sort') or 'storage').strip()
    if sort == 'name':
        items.sort(key=lambda r: r['display_name'].lower())
    elif sort == 'datasets':
        items.sort(key=lambda r: -r['dataset_count'])
    elif sort == 'subs':
        items.sort(key=lambda r: -r['sub_24h'])
    else:
        items.sort(key=lambda r: -r['storage_used'])

    return render_template(
        'admin_user_storage.html',
        items=items,
        total_bytes=total_bytes,
        total_human=_format_bytes(total_bytes),
        sort=sort,
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
            CustomField.data_type,
            func.count(CustomField.id),
        )
        .filter(CustomField.leaderboard_id.isnot(None))
        .filter(CustomField.submission_id.is_(None))
        .filter(CustomField.sample_id.is_(None))
        .group_by(CustomField.leaderboard_id, CustomField.data_type)
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

    # 4. Datasets — Dataset.storage_bytes is the cached counter,
    # joined here with a Sample COUNT so admins can spot e.g.
    # "huge bytes / 0 samples" (broken import) or "tiny bytes / many
    # samples" (something compressed unusually well).
    sample_counts = dict(
        db.session.query(Sample.dataset_id, func.count(Sample.id))
        .group_by(Sample.dataset_id).all()
    )
    dataset_rows = (
        Dataset.query.with_entities(
            Dataset.id, Dataset.name, Dataset.storage_bytes,
        )
        .order_by(Dataset.storage_bytes.desc())
        .all()
    )
    datasets_summary = [
        {
            'id': did, 'name': dname,
            'bytes': int(dsize or 0),
            'samples': int(sample_counts.get(did, 0)),
        }
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


# ===================== Papers With Code import =====================
# Admin-only mirror flow: search PWC benchmarks → preview → confirm to
# create a canonical leaderboard whose primary attachment is the
# benchmark's HuggingFace mirror, plus one mirrored Submission per
# PWC result row. Mirrored submissions skip Celery entirely; their
# scores are inserted directly as MetricResult rows.


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
        ds_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id))
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
    a real homepage.

    Signed-in users are forwarded to /home — their dashboard is the
    relevant entry point. Anonymous visitors see the featured-LB
    pitch below.

    Featured leaderboards: top public leaderboards by submission activity
    in the last 30 days. Visibility filter excludes private + unlisted —
    same rules as /explore (when that lands).
    """
    if getattr(g, 'current_user', None):
        return redirect(url_for('home'))
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

    # Per-LB thumb URL so the landing-page cards can show the same
    # 130px hero image the rest of the site uses (home-card shape).
    landing_thumbs = {}
    for lb, _ in featured_rows:
        lb_datasets = list(lb.datasets)
        landing_thumbs[lb.id] = (
            _dataset_thumb_url(lb_datasets[0]) if lb_datasets else None
        )

    return render_template(
        'landing.html',
        featured=featured_rows,
        leaderboard_thumbs=landing_thumbs,
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
                CustomField.data_type.in_(('image', 'depth')))
        .order_by(CustomField.data_type.desc())  # 'image' before 'depth'
        .first()
    )
    if cf is None:
        return None
    return url_for('serve_custom_field_image', field_id=cf.id)


def _submission_primary_scores(submissions):
    """For each submission, return the value + rank of its leaderboard's
    first (primary) metric. Rank is the position among all verified,
    non-archived submissions on the same LB ordered by that metric
    (respecting its sort_direction; rank 1 = best). Returns
    {sub_id: {'label', 'value', 'rank', 'total'}}; submissions with no
    primary metric or no result are omitted."""
    out = {}
    # Cache per-LB primary metric + its full ranking so several
    # submissions on the same board don't each re-query.
    lb_rankings = {}  # lb_id -> {'primary', 'order': {sub_id: rank}, 'total'}
    for sub in submissions:
        lb = sub.leaderboard
        if lb is None:
            continue
        if lb.id not in lb_rankings:
            lms = list(lb.leaderboard_metrics)
            primary = lms[0] if lms else None
            order, total = {}, 0
            higher_better = True
            vmin = vmax = None
            if primary is not None:
                rows = (
                    db.session.query(MetricResult.submission_id, MetricResult.value)
                    .join(Submission, MetricResult.submission_id == Submission.id)
                    .filter(MetricResult.leaderboard_metric_id == primary.id,
                            MetricResult.value.isnot(None),
                            Submission.is_archived.is_(False),
                            Submission.kind == 'verified')
                    .all()
                )
                higher_better = (primary.sort_direction or 'higher_is_better') != 'lower_is_better'
                ranked = sorted(rows, key=lambda r: r[1], reverse=higher_better)
                total = len(ranked)
                for idx, (sid, _val) in enumerate(ranked, start=1):
                    order[sid] = idx
                if ranked:
                    vals = [v for _s, v in ranked]
                    vmin, vmax = min(vals), max(vals)
            lb_rankings[lb.id] = {'primary': primary, 'order': order,
                                  'total': total, 'higher_better': higher_better,
                                  'vmin': vmin, 'vmax': vmax}

        info = lb_rankings[lb.id]
        primary = info['primary']
        if primary is None:
            continue
        mr = MetricResult.query.filter_by(
            submission_id=sub.id, leaderboard_metric_id=primary.id
        ).first()
        if mr is None or mr.value is None:
            continue
        # Same green gradient the LB table uses: hsl(120,70%,L%), lighter
        # for worse, more saturated for better; flipped for lower-is-better.
        color = None
        vmin, vmax = info['vmin'], info['vmax']
        if vmin is not None and vmax is not None and vmax != vmin:
            normalized = (mr.value - vmin) / (vmax - vmin)
            if not info['higher_better']:
                normalized = 1.0 - normalized
            lightness = int(95 - (normalized * 45))
            color = f'hsl(120, 70%, {lightness}%)'
        elif vmin is not None:
            color = 'hsl(120, 70%, 85%)'  # single value → light green
        out[sub.id] = {
            'label': primary.target_name or primary.global_metric.name,
            'value': mr.value,
            'rank': info['order'].get(sub.id),
            'total': info['total'],
            'color': color,
        }
    return out


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
    # been promoted). /home shows both, public first. After the
    # visibility-driven /explore change, "public" here means
    # visibility=='public' (user has promoted their LB to the public
    # catalog).
    public_lbs = (
        Leaderboard.query
        .filter(Leaderboard.owner_user_id == user.id,
                Leaderboard.visibility == 'public')
        .order_by(Leaderboard.upload_date.desc())
        .limit(24)
        .all()
    )
    personal_lbs = (
        Leaderboard.query
        .filter(Leaderboard.owner_user_id == user.id,
                Leaderboard.visibility != 'public')
        .order_by(Leaderboard.upload_date.desc())
        .limit(24)
        .all()
    )
    leaderboards = public_lbs + personal_lbs  # legacy template var

    dataset_thumbs = {ds.id: _dataset_thumb_url(ds) for ds in datasets}
    # Leaderboards-using-this-dataset count for the dataset cards' trophy
    # stat (ds.leaderboards is a dynamic relationship → .count()).
    dataset_lb_counts = {ds.id: ds.leaderboards.count() for ds in datasets}
    leaderboard_thumbs = {}
    for lb in leaderboards:
        lb_datasets = list(lb.datasets)
        leaderboard_thumbs[lb.id] = (
            _dataset_thumb_url(lb_datasets[0]) if lb_datasets else None
        )

    # Recent-activity rails: the 5 newest verified submissions on public
    # leaderboards (the community feed) and the 5 newest the user made
    # themselves (their own feed). Both skip archived + mirrored rows so
    # only real, live submissions show.
    recent_public_submissions = (
        Submission.query
        .join(Leaderboard, Submission.leaderboard_id == Leaderboard.id)
        .filter(Leaderboard.visibility == 'public',
                Submission.is_archived.is_(False),
                Submission.kind == 'verified')
        .order_by(Submission.upload_date.desc())
        .limit(5)
        .all()
    )
    recent_user_submissions = (
        Submission.query
        .filter(Submission.owner_user_id == user.id,
                Submission.is_archived.is_(False))
        .order_by(Submission.upload_date.desc())
        .limit(5)
        .all()
    )

    # Headline score + rank for each submission: the value of the
    # leaderboard's first (primary) metric — same metric the LB table
    # defaults its best-first sort to — plus the submission's rank among
    # all verified entries on that LB by that metric.
    # {sub_id: {'label', 'value', 'rank', 'total'}}.
    public_submission_scores = _submission_primary_scores(recent_public_submissions)
    user_submission_scores = _submission_primary_scores(recent_user_submissions)

    return render_template(
        'home.html',
        datasets=datasets,
        leaderboards=leaderboards,
        public_lbs=public_lbs,
        personal_lbs=personal_lbs,
        dataset_thumbs=dataset_thumbs,
        dataset_lb_counts=dataset_lb_counts,
        leaderboard_thumbs=leaderboard_thumbs,
        recent_public_submissions=recent_public_submissions,
        recent_user_submissions=recent_user_submissions,
        public_submission_scores=public_submission_scores,
        user_submission_scores=user_submission_scores,
    )


@app.route('/explore')
def explore():
    """Back-compat alias: old `/explore` URLs (external bookmarks,
    inbound links) keep working but redirect to the canonical
    `/leaderboards` endpoint. All in-app callers should use
    `url_for('leaderboards', ...)` directly."""
    return redirect(url_for('leaderboards', **request.args))


@app.route('/leaderboards')
def leaderboards():
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
            func.count(func.distinct(Submission.owner_user_id)).label('user_count'),
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
            func.coalesce(total_activity.c.user_count, 0).label('user_count'),
        )
        .outerjoin(recent_activity, Leaderboard.id == recent_activity.c.lb_id)
        .outerjoin(total_activity, Leaderboard.id == total_activity.c.lb_id)
        # visible_in_list already encodes the right policy: public
        # rows for everyone, plus owner's own non-unlisted rows for
        # signed-in users, plus everything for admins. The old
        # hardcoded `visibility == 'public'` filter we used to stack
        # on top hid owners' own private LBs from their own catalog.
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
        {'lb': lb, 'recent': int(r or 0), 'total': int(t or 0), 'users': int(u or 0)}
        for lb, r, t, u in base.limit(60).all()
    ]
    # Stable secondary sort by category so the template can render
    # Area > Task headers over contiguous runs. The DB-side sort
    # (activity / recent / popular) is preserved within each group
    # because Python's sorted() is stable. Empty/None categories sink
    # to the bottom via the high sentinel.
    rows.sort(key=lambda r: (r['lb'].category or '￿').lower())
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

    # Category tree: per-area / per-task counts over visible public LBs.
    # Counts are computed independent of the current category filter so
    # the tree always shows the full breakdown (clicking a leaf scopes
    # the results panel but the tree stays stable).
    cat_rows = (
        db.session.query(Leaderboard.category, func.count(Leaderboard.id))
        .filter(visible_lb_filter)   # already gates by visibility + ownership
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
        'leaderboards.html',
        rows=rows,
        q=q,
        sort=sort,
        tag_cloud=tag_cloud,
        active_tag=tag_filter,
        leaderboard_thumbs=leaderboard_thumbs,
        category_tree=category_tree,
        active_category=category_filter,
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
        # Category lives as a free-form "Area/Task" string. Owners can
        # type their own, the form also provides a datalist of existing
        # categories used elsewhere on the site. Stripping to None when
        # blank so the explore-tree groups "no category" under
        # Uncategorized rather than the empty-string bucket.
        if 'category' in request.form:
            cat = (request.form.get('category') or '').strip()
            leaderboard.category = cat or None
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
    if any(None for s in samples):
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
        if cf.data_type in ['metric', 'scalar', 'image']:
            dataset_fields_set.add(cf.name)
            
    # Submission Custom Fields
    if sub_ids:
        submission_custom_fields = CustomField.query.filter(CustomField.submission_id.in_(sub_ids)).all()
        for cf in submission_custom_fields:
            if cf.data_type in ['metric', 'scalar', 'image']:
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
    dataset_custom_metrics = CustomField.query.filter(CustomField.sample_id.in_([s.id for s in samples]), CustomField.data_type == 'metric').all()
    for cf in dataset_custom_metrics:
        all_known_metrics.add(f'gt_{cf.name}') # Although leaderboard usually aggregates sub metrics? 
        # Actually leaderboard.html only shows standard, dynamic, and sub-custom metrics. Not GT custom metrics usually (unless dynamic uses them).
        # But let's stick to what's shown in leaderboard table loop.
    
    # Submission custom metrics
    if sub_ids:
        submission_custom_metrics = CustomField.query.filter(
            CustomField.submission_id.in_(sub_ids), 
            CustomField.data_type.in_(['metric', 'scalar'])
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
    # Distinct categories already in use across visible LBs — feeds the
    # datalist suggestions on the Category input so owners pick a
    # consistent value instead of inventing variants.
    known_categories = sorted({
        cat for (cat,) in (
            db.session.query(Leaderboard.category)
            .filter(Leaderboard.category.isnot(None))
            .distinct()
            .all()
        ) if cat
    })
    # Pred-field editor state: current schema (merge of metric-derived +
    # required_pred_fields_json) + a "is the LB editable?" flag (no
    # verified submissions → editor is open).
    pred_fields_schema = _lb_submission_pred_fields(leaderboard)
    has_verified_subs = any(
        (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored'
        for s in (leaderboard.submissions or [])
    )
    # Lifecycle policy:
    #   - ADDING new metrics / pred fields is always allowed (it
    #     only widens the contract for future submissions).
    #   - REMOVING or renaming a metric / pred field, or changing
    #     a kind / role on an existing field, requires the LB to
    #     be empty (no verified submissions).
    # The template renders existing rows read-only and hides the
    # remove buttons when `pred_edit_allowed` is False, but still
    # exposes the add-row button + save when the user is permitted.
    pred_add_allowed = _can_edit_lb_preds(
        leaderboard, getattr(g, 'current_user', None)
    )
    pred_edit_allowed = pred_add_allowed and not has_verified_subs
    # Contract-filtered metric picker — same mechanism as the LB
    # creation page, but built against this LB's effective contract
    # (attached-dataset fields + per-LB role overrides + required
    # pred fields). Unsatisfiable metrics still appear so the
    # picker can show them disabled with a "(needs …)" hint.
    available_metrics_for_lb, gt_field_options, dataset_field_options = (
        _build_lb_picker_context(leaderboard)
    )
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
                           known_categories=known_categories,
                           pred_fields_schema=pred_fields_schema,
                           pred_fields_editable=pred_edit_allowed,
                           pred_add_allowed=pred_add_allowed,
                           pred_edit_allowed=pred_edit_allowed,
                           has_verified_subs=has_verified_subs,
                           available_metrics_for_lb=available_metrics_for_lb,
                           gt_field_options=gt_field_options,
                           dataset_field_options=dataset_field_options,
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
        
        # Display set = summary_metrics. Ensure EVERY bound metric is in
        # it, not just the new one — a metric created via a path that
        # skipped summary_metrics (older LBs, manual seeding) was only
        # visible while summary_metrics stayed empty (the "show all"
        # fallback); adding a second metric populated the list and
        # silently hid the first. Rebuild as: existing entries (order
        # preserved, incl. any non-lm custom columns) + any bound metric
        # not yet listed.
        current_metrics = [m.strip() for m in (leaderboard.summary_metrics or '').split(',') if m.strip()]
        seen = set(current_metrics)
        bound_ids = [f"lm_{m.id}" for m in leaderboard.leaderboard_metrics]
        bound_ids.append(f"lm_{lm.id}")  # newly-added row may not be in the relationship yet
        for bid in bound_ids:
            if bid not in seen:
                current_metrics.append(bid)
                seen.add(bid)
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
    # Removing a metric only drops THIS metric's column + its
    # MetricResult rows — it never reinterprets a submission's
    # prediction bytes (unlike renaming/retyping a pred field, which
    # stays locked when verified submissions exist). Adding metrics is
    # already allowed with submissions present, so removal is too:
    # symmetric, and the only safe way to undo an accidental add.
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

def extract_viz_arg_value(sample, submission, field_key, *, leaderboard_id=None):
    """Helper to extract argument value for visualization from sample/submission.

    When `leaderboard_id` is provided and the LB has a ready
    LeaderboardMaterialization, file-backed gt_<field> lookups
    return the FULL-resolution materialised path instead of the
    dataset's preview path. Falls back to the preview file when no
    materialisation exists. Inline kinds (scalar/label/text/json)
    are not affected — their value_text already holds the content.
    """
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
            # File-backed kinds: swap in the materialised path when
            # this LB has one. Inline kinds (text/json/scalar/label)
            # have value_text = content, not a path — leave alone.
            if (leaderboard_id is not None
                    and cf.data_type in ('image', 'mask', 'depth', 'audio')
                    and isinstance(value, str)):
                from benchhub.lb_materialize import materialized_or_preview_path
                value = materialized_or_preview_path(
                    app.config['UPLOAD_FOLDER'],
                    leaderboard_id, field_name, sample.name, value,
                )
        elif field_name == 'histogram':
            value = None
        elif field_name == 'config':
            value = None
            
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

@app.route('/api/dataset_viz/<int:dv_id>/<int:sample_id>')
def execute_dataset_visualization(dv_id, sample_id):
    """Render a DatasetVisualization for one sample. Same exec model
    as the LB-side execute_visualization (no submission since this is
    dataset-scoped). Cached by content hash so repeat hits hit disk
    not Python."""
    import io as _io, hashlib as _hl, re as _re
    from PIL import Image as _PIL
    dv = DatasetVisualization.query.get_or_404(dv_id)
    gv = dv.global_visualization
    sample = Sample.query.get_or_404(sample_id)
    if sample.dataset_id != dv.dataset_id:
        return abort(400, description='sample/dataset mismatch')

    code_hash = _hl.md5((gv.python_code or '').encode()).hexdigest()
    mapping_hash = _hl.md5((dv.arg_mappings or '').encode()).hexdigest()
    cache_key = f'dsviz_{dv_id}_{sample_id}_{code_hash}_{mapping_hash}'
    cache_hash = _hl.md5(cache_key.encode()).hexdigest()
    cache_dir = os.path.join(os.getcwd(), 'data', 'viz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{cache_hash}.png')
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype='image/png')

    try:
        arg_mappings = json.loads(dv.arg_mappings) if dv.arg_mappings else {}
        kwargs = {a: extract_viz_arg_value(sample, None, k)
                  for a, k in arg_mappings.items()}
        exec_globals = {
            'Image': _PIL, 'PIL': __import__('PIL'),
            'numpy': __import__('numpy'), 'np': __import__('numpy'),
        }
        exec(gv.python_code, exec_globals)
        m = _re.search(r'def\s+(\w+)\s*\(', gv.python_code)
        func_name = m.group(1) if m else None
        if func_name and func_name in exec_globals:
            result = exec_globals[func_name](**kwargs)
            if isinstance(result, _PIL.Image):
                result.save(cache_path, 'PNG')
                buf = _io.BytesIO()
                result.save(buf, 'PNG'); buf.seek(0)
                return send_file(buf, mimetype='image/png')
        return create_error_image('No result')
    except Exception as e:
        import traceback; traceback.print_exc()
        return create_error_image(str(e)[:60])


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
            kwargs[arg_name] = extract_viz_arg_value(
                sample, submission, field_key,
                leaderboard_id=lv.leaderboard_id,
            )
        
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

def _lb_label_vocabs(leaderboard):
    """{field_name: [class names]} for every label / label_list field on
    the LB — GT fields (from dataset field params) + declared pred fields
    (from required_pred_fields_json). Used to label viz axes / stats with
    class names instead of raw indices."""
    vocabs = {}
    for ds in (leaderboard.datasets or []):
        for df in getattr(ds, 'fields', []) or []:
            if df.kind in ('label', 'label_list'):
                names = (df.get_params() or {}).get('names')
                if names:
                    vocabs[df.name] = names
    try:
        for entry in (json.loads(leaderboard.required_pred_fields_json or '[]') or []):
            if isinstance(entry, dict) and entry.get('kind') in ('label', 'label_list'):
                names = (entry.get('params') or {}).get('names')
                if names:
                    vocabs[entry['name']] = names
    except (TypeError, ValueError):
        pass
    return vocabs


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

        # Per-field class-name vocab (label / label_list fields), so a
        # viz can label its axes with `cat` instead of `3`. Keyed by
        # bare field name (e.g. 'label', 'label_pred').
        label_vocabs = _lb_label_vocabs(leaderboard)

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
                # Offer the class vocab as `<arg>_names` for label fields.
                bare = field_key.split('_', 1)[1] if ('_' in field_key and field_key.split('_', 1)[0] in ('gt', 'sub')) else field_key
                if bare in label_vocabs:
                    kwargs[f'{arg_name}_names'] = label_vocabs[bare]

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
            # Pass only kwargs the function accepts (drop injected
            # `<arg>_names` for funcs that don't declare them), unless it
            # takes **kwargs.
            import inspect as _inspect
            try:
                _sig = _inspect.signature(viz_func)
                _accepts_var_kw = any(
                    p.kind == _inspect.Parameter.VAR_KEYWORD
                    for p in _sig.parameters.values()
                )
                if not _accepts_var_kw:
                    _allowed = set(_sig.parameters.keys())
                    kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
            except (TypeError, ValueError):
                pass
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
    """Documents every BenchHub data type. Rows are driven from the live
    `benchhub.types.DTYPES` registry, so the page can't drift from code."""
    import inspect

    from benchhub.types import DTYPES

    descriptions = {
        'image':  'RGB / RGBA photograph or grayscale image.',
        'mask':   'Integer label map for segmentation. Values are class IDs.',
        'depth':  'Float depth map with a declared unit. NaN values are treated as invalid.',
        'audio':  'Waveform (mono or multi-channel) + sample rate.',
        'text':   'UTF-8 string.',
        'bboxes': 'List of bounding boxes, optional labels and scores, with declared coordinate format.',
        'label':  'Single class as an int or string. Vocab is declared at the leaderboard level.',
        'label_list': 'Ordered top-K class predictions. Required `k` is carried alongside the values; metrics use it for top-K accuracy / MRR / Hits@N.',
        'scalar': 'A single float.',
        'json':   "Arbitrary JSON-serialisable structure — escape hatch for shapes that don't fit a typed class yet.",
        'coco_detections': 'COCO-style detection annotations for one image: per-instance category_id, bbox (xywh), optional segmentation polygons, area, iscrowd.',
    }
    examples = {
        'image':  'bh.Image(arr)                                      # arr: (H,W,3) | (H,W,4) | (H,W) uint8',
        'mask':   'bh.Mask(arr, num_classes=21, ignore_index=255)     # arr: (H,W) int',
        'depth':  'bh.Depth(arr, unit="meters")                       # arr: (H,W) float32',
        'audio':  'bh.Audio(waveform, sample_rate=16000)              # waveform: (T,) | (T,C) float32',
        'text':   'bh.Text("a quick brown fox")',
        'bboxes': 'bh.BBoxes([[x1,y1,x2,y2], ...], labels=["cat"], scores=[0.9], format="xyxy")',
        'label':  'bh.Label("cat")                                    # or bh.Label(3)',
        'label_list': 'bh.LabelList([3, 5, 1, 9, 2], k=5)                 # top-K predictions; len(values) must equal k',
        'scalar': 'bh.Scalar(0.91)',
        'json':   'bh.Json({"relations": [...]})',
        'coco_detections':
            'bh.CocoDetections([                                # one record per detection\n'
            '    {"category_id": 17, "category_name": "cat",\n'
            '     "bbox": [x, y, w, h],                         # COCO xywh (NOT xyxy)\n'
            '     "segmentation": [[x0, y0, x1, y1, ...]],      # zero or more polygons\n'
            '     "area": 1234.5, "iscrowd": 0},\n'
            '])',
    }

    types_info = []
    for kind, cls in DTYPES.items():
        sig = str(inspect.signature(cls.__init__))
        # Strip the leading "(self, " / "(self)" so the displayed signature
        # reads as a call site: `Depth(array, *, unit="meters")`.
        if sig.startswith('(self, '):
            sig = '(' + sig[len('(self, '):]
        elif sig.startswith('(self)'):
            sig = '()'
        types_info.append({
            'kind': kind,
            'name': cls.__name__,
            'file_ext': cls.file_ext,
            'storage': cls.file_ext if cls.file_ext else 'inline (SQLite)',
            'signature': cls.__name__ + sig,
            'description': descriptions.get(kind, ''),
            'example': examples.get(kind, ''),
        })

    return render_template('supported_types.html', types_info=types_info)


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
# HF task-category id → (Area, human-readable Task). Drives
# `_hf_tags_to_category(tags)` — when the importer pulls the
# dataset card from HF Hub, the first matching `task_categories:*`
# tag (or `task_ids:*` for sub-task specificity) seeds the
# Dataset.category column so the row lands in the right bucket
# on /datasets and /leaderboards without the admin having to type
# the string by hand. Owners can still override on the dataset
# settings page.
_HF_TASK_TO_AREA_TASK: dict[str, tuple[str, str]] = {
    # --- Vision ---
    'image-classification':        ('Vision', 'Image Classification'),
    'zero-shot-image-classification': ('Vision', 'Zero-Shot Image Classification'),
    'image-segmentation':          ('Vision', 'Image Segmentation'),
    'semantic-segmentation':       ('Vision', 'Semantic Segmentation'),
    'instance-segmentation':       ('Vision', 'Instance Segmentation'),
    'panoptic-segmentation':       ('Vision', 'Panoptic Segmentation'),
    'object-detection':            ('Vision', 'Object Detection'),
    'zero-shot-object-detection':  ('Vision', 'Zero-Shot Object Detection'),
    'face-detection':              ('Vision', 'Face Detection'),
    'depth-estimation':            ('Vision', 'Depth Estimation'),
    'monocular-depth-estimation':  ('Vision', 'Depth Estimation'),
    'image-to-image':              ('Vision', 'Image-to-Image'),
    'image-to-text':               ('Vision', 'Image Captioning'),
    'image-feature-extraction':    ('Vision', 'Feature Extraction'),
    'image-generation':            ('Vision', 'Image Generation'),
    'unconditional-image-generation': ('Vision', 'Image Generation'),
    'text-to-image':               ('Vision', 'Text-to-Image'),
    'text-to-video':               ('Vision', 'Text-to-Video'),
    'video-classification':        ('Vision', 'Video Classification'),
    'pose-estimation':             ('Vision', 'Pose Estimation'),
    'optical-flow':                ('Vision', 'Optical Flow'),
    'super-resolution':            ('Vision', 'Super-Resolution'),
    'denoising':                   ('Vision', 'Denoising'),
    'inpainting':                  ('Vision', 'Inpainting'),
    # --- NLP ---
    'text-classification':         ('NLP', 'Text Classification'),
    'token-classification':        ('NLP', 'Token Classification'),
    'zero-shot-classification':    ('NLP', 'Zero-Shot Classification'),
    'question-answering':          ('NLP', 'Question Answering'),
    'visual-question-answering':   ('NLP', 'Visual Question Answering'),
    'translation':                 ('NLP', 'Translation'),
    'summarization':               ('NLP', 'Summarization'),
    'text-generation':             ('NLP', 'Text Generation'),
    'text2text-generation':        ('NLP', 'Text-to-Text Generation'),
    'fill-mask':                   ('NLP', 'Fill Mask'),
    'sentence-similarity':         ('NLP', 'Sentence Similarity'),
    'feature-extraction':          ('NLP', 'Feature Extraction'),
    'conversational':              ('NLP', 'Conversational'),
    # --- Speech & Audio ---
    'automatic-speech-recognition': ('Speech & Audio', 'Speech Recognition'),
    'text-to-speech':              ('Speech & Audio', 'Text-to-Speech'),
    'audio-classification':        ('Speech & Audio', 'Audio Classification'),
    'audio-to-audio':              ('Speech & Audio', 'Audio-to-Audio'),
    'voice-activity-detection':    ('Speech & Audio', 'Voice Activity Detection'),
    # --- Tabular ---
    'tabular-classification':      ('Tabular', 'Tabular Classification'),
    'tabular-regression':          ('Tabular', 'Tabular Regression'),
    'tabular-to-text':             ('Tabular', 'Tabular-to-Text'),
    # --- Time series ---
    'time-series-forecasting':     ('Time Series', 'Forecasting'),
    # --- Graph ---
    'graph-ml':                    ('Graph', 'Graph ML'),
    # --- RL ---
    'reinforcement-learning':      ('Reinforcement Learning', 'Reinforcement Learning'),
    'robotics':                    ('Reinforcement Learning', 'Robotics'),
}


def _hf_tags_to_category(tags) -> str | None:
    """Map a HuggingFace dataset's tag list onto BH's `Area/Task`
    taxonomy. Walks `task_ids:*` first (more specific) before
    `task_categories:*` (broader). Returns None if nothing matches —
    callers leave `Dataset.category` NULL in that case so the row
    lands in Uncategorized rather than the wrong bucket.
    """
    if not tags:
        return None
    prefixes = ('task_ids:', 'task_categories:', 'task:')
    for prefix in prefixes:
        for tag in tags:
            if not isinstance(tag, str) or not tag.startswith(prefix):
                continue
            key = tag[len(prefix):].strip().lower()
            mapped = _HF_TASK_TO_AREA_TASK.get(key)
            if mapped:
                return f"{mapped[0]}/{mapped[1]}"
    # "medical" as a domain prefix on any task — promote to Medical
    # so e.g. medical-image-segmentation lands under Medical, not
    # Vision. Mirrors the old _DOMAIN_PREFIXES heuristic.
    text = ' '.join(t.lower() for t in tags if isinstance(t, str))
    if 'medical' in text or 'clinical' in text or 'radiograph' in text:
        return 'Medical/Other'
    return None


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


def _extract_metric_arg_names(python_code: str) -> list[str]:
    """Pull the parameter names off the first FunctionDef in a
    GlobalMetric's source. Used by the LB-creation form to render
    one input row per arg so the admin can map each to a dataset
    field. Anything that fails to parse (or has no top-level
    function) returns an empty list — caller treats that as "ask
    for a single `gt` field" by convention.

    Excludes self/cls, *args / **kwargs, and the conventional
    `pred` arg (which the LB derives from the matching pred
    contract field automatically, so the admin shouldn't have to
    map it explicitly).
    """
    import ast as _ast
    try:
        tree = _ast.parse(python_code or '')
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef):
            names: list[str] = []
            for a in node.args.args:
                if a.arg in ('self', 'cls'):
                    continue
                names.append(a.arg)
            return names
    return []


_METRIC_ROLE_GT_TOKENS = {
    'gt', 'target', 'y_true', 'truth', 'gold', 'label_true', 'ground_truth',
}
_METRIC_ROLE_PRED_TOKENS = {
    'pred', 'prediction', 'y_pred', 'output', 'predicted', 'pred_label',
}
_METRIC_ROLE_INPUT_TOKENS = {
    'input', 'x', 'img', 'image', 'inp', 'src',
}


def _kinds_from_signature(python_code: str) -> list[str]:
    """Parse the first FunctionDef in `python_code` and return the
    BH kinds inferred from each arg's type annotation. Examples
    that resolve:

        def acc(gt: bh.Label, pred: bh.Label)         → ['label', 'label']
        def rmse(gt: Depth, pred: Depth)              → ['depth', 'depth']
        def fid(gt: benchhub.Image, pred: Image)      → ['image', 'image']

    Union annotations (`X | Y`, PEP 604) collapse to a `|`-joined
    string so the picker can offer fields matching any branch:

        def acc(gt: bh.Label, pred: bh.Label | bh.LabelList)
            → ['label', 'label|label_list']

    Annotations that don't name a known DTYPES class slot become
    an empty string in the returned list (so the array length
    still matches the arg count). Returns [] when parsing fails
    or no arg has any useful annotation — keeps the caller's
    "input_kinds = NULL ⇒ unconstrained" fallback intact.
    """
    import ast as _ast
    from benchhub.types import DTYPES
    name_to_kind = {cls.__name__: kind for kind, cls in DTYPES.items()}

    def _kind_of_node(ann) -> str:
        """Resolve a single annotation node to a kind, or '' if
        it doesn't name a known DTYPES class. Handles attribute
        access (`bh.Label`), bare names (`Label`), and unions
        (`X | Y`) recursively."""
        if ann is None:
            return ''
        if isinstance(ann, _ast.BinOp) and isinstance(ann.op, _ast.BitOr):
            left = _kind_of_node(ann.left)
            right = _kind_of_node(ann.right)
            parts = [p for p in (left.split('|') if left else []) +
                              (right.split('|') if right else []) if p]
            # Dedupe preserving order so `Label | Label` collapses
            # to one entry.
            seen: set[str] = set()
            uniq: list[str] = []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            return '|'.join(uniq)
        final_name = None
        if isinstance(ann, _ast.Name):
            final_name = ann.id
        elif isinstance(ann, _ast.Attribute):
            final_name = ann.attr
        return name_to_kind.get(final_name) or '' if final_name else ''

    try:
        tree = _ast.parse(python_code or '')
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef):
            kinds: list[str] = []
            saw_any = False
            for a in node.args.args:
                if a.arg in ('self', 'cls'):
                    continue
                k = _kind_of_node(a.annotation)
                kinds.append(k)
                if k:
                    saw_any = True
            return kinds if saw_any else []
    return []


def _apply_metric_typed_contract(metric, *, kinds_raw, roles_raw, python_code):
    """Set `metric.input_kinds` / `metric.input_roles` from the form
    submission, with auto-derivation fallbacks:

      - kinds: explicit comma-sep > annotations on the function args
        (`bh.Label`, `Depth`, …) > NULL.
      - roles: explicit comma-sep > arg-name heuristic (gt / pred /
        input tokens) > NULL.

    Raises ValueError on any explicit-but-invalid value so the route
    can flash a clear error.
    """
    from benchhub.types import DTYPES as _DTYPES_VALID
    _VALID_ROLES = {'gt', 'pred', 'input'}
    args = _extract_metric_arg_names(python_code or '')

    # Kinds --------------------------------------------------------
    if kinds_raw == '':
        derived = _kinds_from_signature(python_code or '')
        # Drop trailing empties to keep the JSON compact; if every
        # entry is empty the column stays NULL.
        while derived and not derived[-1]:
            derived.pop()
        metric.input_kinds = json.dumps(derived) if derived else None
    else:
        kinds = [k.strip() for k in kinds_raw.split(',') if k.strip()]
        # Each entry may be a single kind ('label') or a `|`-joined
        # union ('label|label_list') meaning the arg accepts EITHER
        # type. Every member must be a known DTYPES kind.
        bad = []
        for k in kinds:
            for member in k.split('|'):
                member = member.strip()
                if member and member not in _DTYPES_VALID:
                    bad.append(member)
        if bad:
            raise ValueError(f"unknown kind(s): {bad}")
        metric.input_kinds = json.dumps(kinds)

    # Roles --------------------------------------------------------
    if roles_raw == '':
        derived = _roles_from_arg_names(args)
        while derived and not derived[-1]:
            derived.pop()
        metric.input_roles = json.dumps(derived) if derived else None
    else:
        roles = [r.strip().lower() for r in roles_raw.split(',') if r.strip()]
        bad = [r for r in roles if r not in _VALID_ROLES]
        if bad:
            raise ValueError(f"unknown role(s): {bad}; expected {sorted(_VALID_ROLES)}")
        metric.input_roles = json.dumps(roles)


def _roles_from_arg_names(args: list[str]) -> list[str]:
    """Heuristic role-per-arg from conventional arg names. Returns
    [] when no arg matches a known token, so the caller can keep
    `input_roles = NULL` rather than ship a meaningless half-filled
    array."""
    out: list[str] = []
    any_match = False
    for a in args:
        lower = (a or '').lower()
        if lower in _METRIC_ROLE_GT_TOKENS:
            out.append('gt'); any_match = True
        elif lower in _METRIC_ROLE_PRED_TOKENS:
            out.append('pred'); any_match = True
        elif lower in _METRIC_ROLE_INPUT_TOKENS:
            out.append('input'); any_match = True
        else:
            out.append('')
    return out if any_match else []


# Task-shaped LB templates. Each preselects a sensible default metric
# set + pred-field schema for the named task. `required_kinds` are the
# DTYPES the dataset must expose at role='gt' for the template to be
# offered as a choice on the LB-create form. Adding a template here
# auto-surfaces it; no new route/UI plumbing needed.
LB_TEMPLATES: dict[str, dict] = {
    'image_segmentation': {
        'label': 'Image Segmentation',
        'description': "IoU on the GT mask. Pred contract: one mask "
                       "field per GT mask.",
        'required_kinds': ['mask'],
        'metric_names': ['iou_mask'],
        'visualization_names': [],
    },
    'depth_estimation': {
        'label': 'Depth Estimation',
        'description': "RMSE + MAE against the GT depth. Pred contract: "
                       "one depth field per GT depth.",
        'required_kinds': ['depth'],
        'metric_names': ['rmse_depth', 'mae_depth'],
        'visualization_names': [],
    },
    'image_classification': {
        'label': 'Image Classification',
        'description': "Top-1 + top-5 accuracy. Pred contract: a "
                       "label_list (top-K) per GT label.",
        'required_kinds': ['label'],
        # Optional pred_kinds map: when the required_kind on a GT
        # field maps to a different kind on the pred side, the
        # template applier uses this instead of the same-kind default.
        # For classification the pred is a ranked top-K list, not a
        # single class, so top_5_accuracy can score against it.
        'pred_kinds': {'label': 'label_list'},
        'metric_names': ['top_1_accuracy', 'top_5_accuracy', 'accuracy'],
        'visualization_names': [],
    },
    'text_qa': {
        'label': 'Text QA / Captioning',
        'description': "Exact-match against GT text. Pred contract: "
                       "one text field per GT text.",
        'required_kinds': ['text'],
        'metric_names': ['exact_match_text'],
        'visualization_names': [],
    },
}


def _all_lb_templates() -> list[dict]:
    """Return every enabled LbTemplate row as plain dicts, in
    sort_order. Falls back to the hardcoded LB_TEMPLATES dict if the
    table is empty / unmigrated (defensive for fresh installs)."""
    try:
        rows = (
            LbTemplate.query
            .filter_by(enabled=True)
            .order_by(LbTemplate.sort_order, LbTemplate.id)
            .all()
        )
        if rows:
            return [{
                'id': r.slug,
                'label': r.label,
                'description': r.description or '',
                'required_kinds': r.required_kinds(),
                'metric_names': r.metric_names(),
                'visualization_names': r.visualization_names(),
            } for r in rows]
    except Exception:
        pass
    return [{'id': k, **v} for k, v in LB_TEMPLATES.items()]


def _lb_template_by_slug(slug: str) -> dict:
    for t in _all_lb_templates():
        if t['id'] == slug:
            return t
    return {}


def _applicable_lb_templates(dataset) -> list[dict]:
    """Return the templates that *this* dataset's GT-role fields can
    satisfy, each annotated with the matched GT field names so the
    caller can build a pred-field schema. Templates whose required
    kinds aren't present don't show up — keeps the picker honest."""
    gt_fields_by_kind: dict[str, list[str]] = {}
    for df in dataset.fields:
        if (df.role or 'gt').lower() != 'gt':
            continue
        gt_fields_by_kind.setdefault(df.kind, []).append(df.name)
    out = []
    for tpl in _all_lb_templates():
        gt_names = []
        ok = True
        for k in tpl.get('required_kinds', []):
            picks = gt_fields_by_kind.get(k)
            if not picks:
                ok = False; break
            gt_names.append({'kind': k, 'gt_field': picks[0]})
        if ok:
            entry = dict(tpl)
            entry['matched_gt_fields'] = gt_names
            out.append(entry)
    return out


def _apply_lb_template(dataset, template_id: str) -> dict:
    """Resolve a template against a dataset → {preselected_metric_ids,
    preselected_visualization_ids, preselected_pred_fields,
    preselected_arg_mappings, suggested_name}.

    For each metric in the template, looks it up by name and binds its
    GT-role args to the dataset's first matching GT field. For each
    required-kind, derives a pred-field entry named `<gt>_pred`.
    """
    tpl = _lb_template_by_slug(template_id)
    if not tpl:
        return {}
    # GT fields by kind (gt-role only — matches what _applicable_ above does)
    gt_by_kind: dict[str, list[str]] = {}
    for df in dataset.fields:
        if (df.role or 'gt').lower() != 'gt':
            continue
        gt_by_kind.setdefault(df.kind, []).append(df.name)

    # Metric preselection
    metric_ids: list[int] = []
    arg_mappings: dict[int, dict[str, str]] = {}
    for mname in tpl.get('metric_names', []):
        gm = GlobalMetric.query.filter_by(name=mname).first()
        if not gm:
            continue
        metric_ids.append(gm.id)
        try:
            kinds = json.loads(gm.input_kinds or '[]')
            roles = json.loads(gm.input_roles or '[]')
        except (TypeError, ValueError):
            kinds, roles = [], []
        args = _extract_metric_arg_names(gm.python_code or '')
        amap: dict[str, str] = {}
        for i, arg in enumerate(args):
            role = roles[i] if i < len(roles) else ''
            kind = kinds[i] if i < len(kinds) else ''
            # First member of an `a|b`-joined kind acts as primary.
            k_first = kind.split('|')[0] if kind else ''
            picks = gt_by_kind.get(k_first) or []
            if not picks:
                continue
            gt_name = picks[0]
            if role == 'gt':
                # Bare field name — the form serialises `{arg: field}`
                # and the server converts to `gt_<field>` internally.
                amap[arg] = gt_name
            elif role == 'pred':
                # Pred args bind to the matching pred-field. Naming
                # convention: <gt_name>_pred (also the convention used
                # by the auto pred-row builder).
                amap[arg] = f'{gt_name}_pred'
        if amap:
            arg_mappings[gm.id] = amap

    # Visualization preselection
    viz_ids: list[int] = []
    for vname in tpl.get('visualization_names', []):
        gv = GlobalVisualization.query.filter_by(name=vname).first()
        if gv:
            viz_ids.append(gv.id)

    # Pred-field schema: <gt_name>_pred for every required GT field.
    # Default to same kind on the pred side, but allow the template to
    # override per GT-kind via `pred_kinds` — e.g. classification keeps
    # gt=label but emits pred=label_list so top_K_accuracy can score.
    pred_kinds_map = tpl.get('pred_kinds') or {}
    pred_fields = []
    for kind in tpl.get('required_kinds', []):
        pred_kind = pred_kinds_map.get(kind, kind)
        for gt_name in gt_by_kind.get(kind, []):
            pred_fields.append({
                'name': f'{gt_name}_pred',
                'kind': pred_kind,
                'description': f'{tpl["label"]} prediction for {gt_name}',
            })

    return {
        'template_id': template_id,
        'template_label': tpl['label'],
        'preselected_metric_ids': metric_ids,
        'preselected_visualization_ids': viz_ids,
        'preselected_pred_fields': pred_fields,
        'preselected_arg_mappings': arg_mappings,
    }


def _build_lb_creation_context(dataset):
    """Build the (`available_metrics_for_lb`, `gt_field_options`,
    `dataset_field_options`) triple that the LB-creation form needs.

    Shared between the dataset-view page (for any other widgets that
    consume these) and the dedicated `/dataset/<id>/create_lb` page
    that renders the actual form.

    Returns:
      available_metrics_for_lb : list[dict]
          One entry per visible GlobalMetric. Includes the metric's
          arg list, declared `(kind, role)` per arg, and a
          satisfiability flag (`True` when this dataset can supply
          every arg; `False` plus a `missing_reason` otherwise).
          Unsatisfiable metrics still appear so the picker can
          render them disabled with a "(needs …)" hint.
      gt_field_options : list[str]
          Names of `role=gt` DatasetFields — the default target set
          for unconstrained metric arg dropdowns.
      dataset_field_options : list[dict]
          Per-field `{name, kind, role}` so the picker JS can filter
          per-arg dropdowns by (kind, role).
    """
    dataset_field_options = [
        {'name': df.name, 'kind': df.kind, 'role': (df.role or 'gt').lower()}
        for df in dataset.fields
    ]
    fields_by_kind_role: dict[tuple[str, str], list[str]] = {}
    for f in dataset_field_options:
        fields_by_kind_role.setdefault((f['kind'], f['role']), []).append(f['name'])
    fields_by_kind: dict[str, list[str]] = {}
    for f in dataset_field_options:
        fields_by_kind.setdefault(f['kind'], []).append(f['name'])

    def _arg_specs(args, kinds, roles):
        return [
            {
                'name': name,
                'kind': kinds[i] if i < len(kinds) else None,
                'role': roles[i] if i < len(roles) else None,
            }
            for i, name in enumerate(args)
        ]

    def _metric_satisfiability(arg_specs):
        missing = []
        for spec in arg_specs:
            k = spec.get('kind')
            r = spec.get('role')
            if not k:
                continue
            members = [m for m in k.split('|') if m]
            if r:
                if not any(fields_by_kind_role.get((m, r)) for m in members):
                    missing.append(spec)
            else:
                if not any(fields_by_kind.get(m) for m in members):
                    missing.append(spec)
        return (not missing, missing)

    def _missing_reason(missing_specs):
        parts = []
        for s in missing_specs:
            bits = []
            if s.get('kind'):
                bits.append(f"kind={s['kind']}")
            if s.get('role'):
                bits.append(f"role={s['role']}")
            parts.append(f"{s.get('name') or '?'}: needs " + ' '.join(bits))
        return '; '.join(parts)

    available_metrics_for_lb = []
    for gm in (
        GlobalMetric.query
        .filter(visible_in_list(GlobalMetric, getattr(g, 'current_user', None)))
        .order_by(GlobalMetric.name)
        .all()
    ):
        args = _extract_metric_arg_names(gm.python_code or '')
        try:
            kinds = json.loads(gm.input_kinds) if gm.input_kinds else []
            if not isinstance(kinds, list):
                kinds = []
        except (TypeError, ValueError):
            kinds = []
        try:
            roles = json.loads(gm.input_roles) if gm.input_roles else []
            if not isinstance(roles, list):
                roles = []
        except (TypeError, ValueError):
            roles = []
        specs = _arg_specs(args, kinds, roles)
        ok, missing = _metric_satisfiability(specs)
        available_metrics_for_lb.append({
            'id': gm.id,
            'name': gm.name,
            'description': (gm.description or '').strip(),
            'args': args,
            'arg_specs': specs,
            'satisfiable': ok,
            'missing_reason': '' if ok else _missing_reason(missing),
        })

    gt_field_options = sorted({
        df.name for df in dataset.fields if (df.role or '').lower() == 'gt'
    })
    return available_metrics_for_lb, gt_field_options, dataset_field_options


def _can_edit_lb_preds(leaderboard, user):
    """Return True iff `user` is allowed to edit this LB's
    pred-field schema. Admins always; LB owner always; any
    attached-dataset owner also (the dataset creator authors the
    contract the LB rides on)."""
    if user is None:
        return False
    if is_admin(user):
        return True
    if getattr(leaderboard, 'owner_user_id', None) == user.id:
        return True
    for ds in (leaderboard.datasets or []):
        if getattr(ds, 'owner_user_id', None) == user.id:
            return True
    return False


def _build_lb_picker_context(leaderboard):
    """Same shape as `_build_lb_creation_context` but driven by an
    existing leaderboard's effective contract.

    Aggregates fields from EVERY attached Dataset, then layers on:
      - `Leaderboard.field_roles_json` overrides — per-LB role swap
        (input/gt/skip) for individual dataset fields;
      - `Leaderboard.required_pred_fields_json` — synthesises virtual
        `role=pred` entries so metric arg specs marked `role=pred`
        can be matched against the LB's declared submission contract.

    The result drives the contract-filtered metric picker on the
    LB settings page — same UX as the creation picker, but against
    the LB's actual (multi-dataset, post-override) contract.
    """
    # Step 1: collect fields across every attached Dataset.
    raw_fields: list[dict] = []
    for ds in leaderboard.datasets:
        for df in ds.fields:
            raw_fields.append({
                'name': df.name,
                'kind': df.kind,
                'role': (df.role or 'gt').lower(),
            })
    # If two datasets declare the same field name with conflicting
    # roles, the later one wins — same precedence as runtime context
    # merging, which uses the GT-source dataset's value last.
    fields_by_name: dict[str, dict] = {f['name']: f for f in raw_fields}

    # Step 2: per-LB role overrides.
    try:
        overrides = json.loads(leaderboard.field_roles_json or '{}')
        if not isinstance(overrides, dict):
            overrides = {}
    except (TypeError, ValueError):
        overrides = {}
    for name, role in overrides.items():
        if name in fields_by_name and isinstance(role, str):
            r = role.lower().strip()
            if r in ('input', 'gt', 'skip', 'pred'):
                fields_by_name[name]['role'] = r

    # Step 3: required pred fields → virtual role=pred entries so a
    # metric whose arg spec is (kind=X, role=pred) can resolve.
    try:
        pred_decls = json.loads(leaderboard.required_pred_fields_json or '[]')
        if not isinstance(pred_decls, list):
            pred_decls = []
    except (TypeError, ValueError):
        pred_decls = []
    for entry in pred_decls:
        if not isinstance(entry, dict):
            continue
        name = entry.get('name')
        kind = entry.get('kind')
        if not name or not kind:
            continue
        fields_by_name[name] = {
            'name': name, 'kind': kind, 'role': 'pred',
        }

    dataset_field_options = list(fields_by_name.values())
    # Drop role=skip when computing satisfiability — a skipped field
    # is intentionally not part of this LB's contract.
    live_fields = [f for f in dataset_field_options if f['role'] != 'skip']

    fields_by_kind_role: dict[tuple[str, str], list[str]] = {}
    fields_by_kind: dict[str, list[str]] = {}
    for f in live_fields:
        fields_by_kind_role.setdefault((f['kind'], f['role']), []).append(f['name'])
        fields_by_kind.setdefault(f['kind'], []).append(f['name'])

    def _arg_specs(args, kinds, roles):
        return [
            {
                'name': name,
                'kind': kinds[i] if i < len(kinds) else None,
                'role': roles[i] if i < len(roles) else None,
            }
            for i, name in enumerate(args)
        ]

    def _metric_satisfiability(arg_specs):
        missing = []
        for spec in arg_specs:
            k = spec.get('kind')
            r = spec.get('role')
            if not k:
                continue
            members = [m for m in k.split('|') if m]
            if r:
                if not any(fields_by_kind_role.get((m, r)) for m in members):
                    missing.append(spec)
            else:
                if not any(fields_by_kind.get(m) for m in members):
                    missing.append(spec)
        return (not missing, missing)

    def _missing_reason(missing_specs):
        parts = []
        for s in missing_specs:
            bits = []
            if s.get('kind'):
                bits.append(f"kind={s['kind']}")
            if s.get('role'):
                bits.append(f"role={s['role']}")
            parts.append(f"{s.get('name') or '?'}: needs " + ' '.join(bits))
        return '; '.join(parts)

    available_metrics_for_lb = []
    for gm in (
        GlobalMetric.query
        .filter(visible_in_list(GlobalMetric, getattr(g, 'current_user', None)))
        .order_by(GlobalMetric.name)
        .all()
    ):
        args = _extract_metric_arg_names(gm.python_code or '')
        try:
            kinds = json.loads(gm.input_kinds) if gm.input_kinds else []
            if not isinstance(kinds, list):
                kinds = []
        except (TypeError, ValueError):
            kinds = []
        try:
            roles = json.loads(gm.input_roles) if gm.input_roles else []
            if not isinstance(roles, list):
                roles = []
        except (TypeError, ValueError):
            roles = []
        specs = _arg_specs(args, kinds, roles)
        ok, missing = _metric_satisfiability(specs)
        available_metrics_for_lb.append({
            'id': gm.id,
            'name': gm.name,
            'description': (gm.description or '').strip(),
            'args': args,
            'arg_specs': specs,
            'satisfiable': ok,
            'missing_reason': '' if ok else _missing_reason(missing),
        })

    gt_field_options = sorted({
        f['name'] for f in live_fields if f['role'] == 'gt'
    })
    return available_metrics_for_lb, gt_field_options, dataset_field_options


def _suggest_free_lb_name(base: str) -> str:
    """Return an LB name that doesn't collide with an existing
    Leaderboard. Starts from `<base>_benchmark`; if that's taken,
    appends `_2`, `_3`, … until a free slot is found (capped at
    100 attempts to avoid pathological loops). Used to pre-fill
    the LB-name input on the create forms so the user usually
    doesn't have to think of one.
    """
    base = (base or '').strip() or 'leaderboard'
    candidate = f"{base}_benchmark"
    if not Leaderboard.query.filter_by(name=candidate).first():
        return candidate
    for n in range(2, 102):
        c = f"{base}_benchmark_{n}"
        if not Leaderboard.query.filter_by(name=c).first():
            return c
    # Extremely unlikely fallback — surface the bare base with a
    # numeric suffix.
    return f"{base}_benchmark_{Leaderboard.query.count() + 1}"


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
    # Parse the selected metric's parameter names so the detail
    # panel can show "<arg>: <kind>·<role>" badges instead of bare
    # kinds.
    selected_arg_names = (
        _extract_metric_arg_names(selected.python_code or '') if selected else []
    )
    return render_template(
        'metrics.html',
        metrics=metrics,
        grouped_metrics=grouped_metrics,
        selected=selected,
        selected_arg_names=selected_arg_names,
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

        # User-created metrics default to private. Admins (BENCHHUB_ADMIN_
        # EMAILS / is_admin) default to public so platform-curated
        # metrics ship visible to everyone immediately. The owner can
        # flip visibility from the detail pane anytime.
        default_vis = 'public' if is_admin(g.current_user) else 'private'
        metric = GlobalMetric(
            name=name,
            description=description,
            python_code=python_code,
            is_aggregated=is_aggregated,
            accepts_aggregated_inputs=accepts_aggregated_inputs,
            owner_user_id=g.current_user.id,
            visibility=default_vis,
        )
        _apply_metric_typed_contract(
            metric,
            kinds_raw=(request.form.get('input_kinds') or '').strip(),
            roles_raw=(request.form.get('input_roles') or '').strip(),
            python_code=python_code,
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
        # Typed contract: explicit form fields win, otherwise we
        # derive from the function signature (kinds from `bh.X`
        # annotations) + arg-name heuristic (roles).
        _apply_metric_typed_contract(
            metric,
            kinds_raw=(request.form.get('input_kinds') or '').strip(),
            roles_raw=(request.form.get('input_roles') or '').strip(),
            python_code=python_code,
        )

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

def _suggest_unique_public_name(model_cls, base_name, exclude_id=None):
    """Find a unique name for promoting to public. Tries the base name
    first; on collision, appends _2, _3, … until a free slot is found.
    Used by the metric + viz promote routes when the owner's chosen
    name already belongs to another public entity."""
    candidate = base_name
    n = 2
    while True:
        q = model_cls.query.filter(
            model_cls.name == candidate,
            model_cls.visibility == 'public',
        )
        if exclude_id is not None:
            q = q.filter(model_cls.id != exclude_id)
        if q.first() is None:
            return candidate
        candidate = f"{base_name}_{n}"
        n += 1


def _public_name_in_use(model_cls, name, exclude_id=None):
    q = model_cls.query.filter(
        model_cls.name == name, model_cls.visibility == 'public',
    )
    if exclude_id is not None:
        q = q.filter(model_cls.id != exclude_id)
    return q.first() is not None


@app.route('/leaderboard/<int:leaderboard_id>/dataset_field_roles', methods=['POST'])
@login_required
def edit_lb_dataset_field_roles(leaderboard_id):
    """Persist per-LB role overrides for fields from attached typed
    datasets. Writes to `Leaderboard.field_roles_json`, which the
    picker + runtime context-builder both honour.

    Form payload: one `field_role_<dataset_field_name>` per field,
    with value `input` / `gt` / `skip`. Missing entries fall back to
    the dataset's declared role.

    Gating mirrors pred-field editing: LB owner, admin, or owner of
    any attached Dataset.
    """
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    user = getattr(g, 'current_user', None)
    if not _can_edit_lb_preds(lb, user):
        abort(403)
    has_verified = any(
        (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored'
        for s in (lb.submissions or [])
    )
    if has_verified:
        flash("Can't change field roles — this LB already has verified "
              "submissions.", "warning")
        return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))

    # Build the set of known field names across attached datasets so
    # the form can't smuggle in roles for fields that don't exist.
    known_fields: set[str] = set()
    for ds in (lb.datasets or []):
        for df in ds.fields:
            known_fields.add(df.name)

    overrides: dict[str, str] = {}
    for key, val in request.form.items():
        if not key.startswith('field_role_'):
            continue
        name = key[len('field_role_'):]
        if name not in known_fields:
            continue
        v = (val or '').strip().lower()
        if v in ('input', 'gt', 'skip'):
            overrides[name] = v
    lb.field_roles_json = json.dumps(overrides) if overrides else None
    db.session.commit()
    flash("Saved dataset field roles.", "success")
    return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))


@app.route('/leaderboard/<int:leaderboard_id>/field_roles', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def edit_lb_field_roles(leaderboard_id):
    """Flip per-mapping role between 'input' (given to the submitter at
    inference time, not predicted) and 'gt' (held server-side, the
    target of prediction). Form has one `role_<att_id>_<column>` field
    per mapping entry, value 'input' or 'gt'."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    has_verified = any(
        (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored'
        for s in (lb.submissions or [])
    )
    if has_verified:
        flash("Can't change field roles — this LB already has verified "
              "submissions.", "warning")
        return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))
    for att in (lb.attachments or []):
        if getattr(att, 'kind', None) != 'hf':
            continue
        try:
            mapping = json.loads(att.hf_mapping_json or '[]')
        except Exception:
            mapping = []
        changed = False
        for m in mapping:
            col = m.get('column')
            if not col:
                continue
            target = request.form.get(f'role_{att.id}_{col}')
            if target in ('input', 'gt') and m.get('role', 'gt') != target:
                m['role'] = target
                changed = True
        if changed:
            att.hf_mapping_json = json.dumps(mapping)
    db.session.commit()
    flash("Saved field roles.", "success")
    return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))


@app.route('/leaderboard/<int:leaderboard_id>/pred_fields', methods=['POST'])
@login_required
def edit_lb_pred_fields(leaderboard_id):
    """Edit the LB's prediction-field schema.

    Lifecycle policy:
      - **Adding** new prediction fields is allowed at ANY point in
        the LB's lifetime, even when verified submissions already
        exist. New fields don't reinterpret any existing prediction
        bytes; they only widen the contract for future submissions.
      - **Removing** an existing field, **renaming** it, or
        **changing its kind** is allowed ONLY when the LB has no
        verified submissions. Otherwise those changes would
        silently re-route or invalidate predictions already stored
        on disk.
      - Description text is free to change at any time — it's
        metadata, not contract.

    Gating: admin, OR the LB owner, OR the owner of any attached
    Dataset. The dataset creator is included because they author the
    canonical contract the LB rides on; an LB owner who isn't the
    dataset author can still tune the pred schema for their own LB.

    Form payload shape:
      name_N           — pred-field name (must end `_pred`)
      kind_N           — image|mask|depth|audio|scalar|text|json|histogram
      description_N    — free-form text (optional)
    (N is a per-row index used only to collate; we don't trust ordering.)"""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    user = getattr(g, 'current_user', None)
    if not _can_edit_lb_preds(lb, user):
        abort(403)
    has_verified = any(
        (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored'
        for s in (lb.submissions or [])
    )

    # Source the valid-kinds set from the typed registry — adding a
    # new DataType subclass shouldn't require editing this allow-list
    # by hand (the old hand-rolled set was missing `label` /
    # `label_list` / `bboxes`, which silently coerced existing rows
    # of those kinds to `scalar` on save).
    from benchhub.types import DTYPES as _DTYPES
    valid_kinds = set(_DTYPES.keys()) | {'histogram'}
    # Collate by index. Form keys look like `name_0`, `kind_0`,
    # `description_0`, `name_1`, ...
    rows = {}
    for key, value in request.form.items():
        for tag in ('name', 'kind', 'description'):
            prefix = f'{tag}_'
            if key.startswith(prefix):
                try:
                    idx = int(key[len(prefix):])
                except ValueError:
                    continue
                rows.setdefault(idx, {})[tag] = (value or '').strip()
                break
    saved = []
    for idx in sorted(rows.keys()):
        r = rows[idx]
        name = r.get('name') or ''
        if not name:
            continue
        if not name.endswith('_pred'):
            name = name + '_pred'
        kind = r.get('kind') or 'scalar'
        if kind not in valid_kinds:
            kind = 'scalar'
        desc = r.get('description') or ''
        gt_field = name[:-len('_pred')]
        saved.append({
            'name': name, 'kind': kind, 'gt_field': gt_field,
            'description': desc,
        })

    # When submissions exist, refuse any change that would alter
    # the kind of, or remove, an existing pred field. Adding new
    # fields is fine because they don't reinterpret stored bytes.
    if has_verified:
        try:
            old = json.loads(lb.required_pred_fields_json or '[]')
            if not isinstance(old, list):
                old = []
        except (TypeError, ValueError):
            old = []
        old_by_name = {e.get('name'): e for e in old if isinstance(e, dict)}
        new_by_name = {e['name']: e for e in saved}
        removed = set(old_by_name) - set(new_by_name)
        if removed:
            flash(
                "Can't remove prediction fields — this leaderboard "
                "already has verified submissions. Delete those "
                "submissions first to unlock.",
                "warning",
            )
            return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))
        for name, old_entry in old_by_name.items():
            new_entry = new_by_name.get(name)
            if not new_entry:
                continue
            if (old_entry.get('kind') or '').strip() != (new_entry.get('kind') or '').strip():
                flash(
                    f"Can't change kind of '{name}' — this leaderboard "
                    "already has verified submissions. Delete them "
                    "first to unlock.",
                    "warning",
                )
                return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))

    lb.required_pred_fields_json = json.dumps(saved) if saved else None
    db.session.commit()
    flash(f"Saved {len(saved)} prediction field(s).", "success")
    return redirect(url_for('edit_leaderboard', leaderboard_id=lb.id))


@app.route('/leaderboard/<int:leaderboard_id>/visibility', methods=['POST'])
@login_required
@owner_required(Leaderboard, 'leaderboard_id')
def set_leaderboard_visibility(leaderboard_id):
    """Owner / admin flip of Leaderboard.visibility. Drives the
    'Publish' button on /leaderboard/<id>. Returns to the LB page."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    target = (request.form.get('visibility') or '').strip()
    if target not in ('public', 'private', 'unlisted'):
        flash("Invalid visibility.", "warning")
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    # Publish (→public) pre-flight: a private LB's materialisation
    # bytes are currently charged to the private bucket. Flipping the
    # LB public moves those bytes onto the public bucket, which must
    # have room. Admins bypass.
    if (target == 'public' and lb.visibility != 'public'
            and not is_admin(g.current_user)):
        owner = lb.owner or g.current_user
        mat_total = int(
            db.session.query(func.coalesce(
                func.sum(LeaderboardMaterialization.storage_bytes), 0
            ))
            .filter(LeaderboardMaterialization.leaderboard_id == lb.id)
            .scalar() or 0
        )
        used_pub = storage_used_bytes(owner, visibility='public')
        cap_pub = quota_cap_for(owner, 'public')
        if used_pub + mat_total > cap_pub:
            flash(
                f"Can't publish — public-bucket cap would be exceeded: "
                f"{_format_bytes(used_pub)} used + "
                f"{_format_bytes(mat_total)} moving > "
                f"{_format_bytes(cap_pub)} cap.",
                "warning",
            )
            return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    lb.visibility = target
    db.session.commit()
    if target == 'public':
        flash(f'"{lb.name}" is now public — visible on /leaderboards.', "success")
    else:
        flash(f'"{lb.name}" is now {target}.', "success")
    return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))


@app.route('/dataset/<int:dataset_id>/visibility', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def set_dataset_visibility(dataset_id):
    """Owner-only flip of Dataset.visibility (public / unlisted / private).
    Dataset.name is globally unique so we don't need the name-collision
    dance the GlobalMetric flow has — just commit the new value."""
    ds = Dataset.query.get_or_404(dataset_id)
    target = (request.form.get('visibility') or '').strip()
    if target not in ('public', 'private', 'unlisted'):
        flash("Invalid visibility.", "warning")
        return redirect(url_for('dataset_settings', dataset_id=ds.id))
    # Publish pre-flight: moving a private dataset to public shifts
    # its `storage_bytes` onto the public quota bucket. Admins bypass.
    if (target == 'public' and ds.visibility != 'public'
            and not is_admin(g.current_user)):
        owner = ds.owner or g.current_user
        ds_bytes = int(ds.storage_bytes or 0)
        used_pub = storage_used_bytes(owner, visibility='public')
        cap_pub = quota_cap_for(owner, 'public')
        if used_pub + ds_bytes > cap_pub:
            flash(
                f"Can't publish — public-bucket cap would be exceeded: "
                f"{_format_bytes(used_pub)} used + "
                f"{_format_bytes(ds_bytes)} moving > "
                f"{_format_bytes(cap_pub)} cap.",
                "warning",
            )
            return redirect(url_for('dataset_settings', dataset_id=ds.id))
    ds.visibility = target
    db.session.commit()
    flash(f'"{ds.name}" is now {target}.', "success")
    return redirect(url_for('dataset_settings', dataset_id=ds.id))


@app.route('/global_metric/<int:metric_id>/visibility', methods=['POST'])
@login_required
@owner_required(GlobalMetric, 'metric_id')
def set_global_metric_visibility(metric_id):
    """Flip a user-owned metric between visibility tiers. When the
    target is 'public' and the metric's name collides with an existing
    public metric, redirect to a tiny resolve page that offers a
    non-colliding suggestion and lets the user edit before committing."""
    metric = GlobalMetric.query.get_or_404(metric_id)
    target = (request.form.get('visibility') or '').strip()
    if target not in ('public', 'private', 'unlisted'):
        flash("Invalid visibility.", "warning")
        return redirect(url_for('metrics_view', selected=metric.id))
    if target == 'public' and _public_name_in_use(GlobalMetric, metric.name, exclude_id=metric.id):
        suggestion = _suggest_unique_public_name(GlobalMetric, metric.name, exclude_id=metric.id)
        return render_template(
            'resolve_name_collision.html',
            entity='metric',
            metric_id=metric.id, viz_id=None,
            current_name=metric.name,
            suggested_name=suggestion,
            target='public',
            action_url=url_for('set_global_metric_visibility_confirm', metric_id=metric.id),
        )
    metric.visibility = target
    db.session.commit()
    flash(f'"{metric.name}" is now {target}.', "success")
    return redirect(url_for('metrics_view', selected=metric.id))


@app.route('/global_metric/<int:metric_id>/visibility/confirm', methods=['POST'])
@login_required
@owner_required(GlobalMetric, 'metric_id')
def set_global_metric_visibility_confirm(metric_id):
    """Second hop of promote-with-rename. The user already saw the
    resolve page and confirmed (or edited) the suggested name."""
    metric = GlobalMetric.query.get_or_404(metric_id)
    new_name = (request.form.get('new_name') or '').strip()
    if not new_name:
        flash("Name can't be empty.", "warning")
        return redirect(url_for('metrics_view', selected=metric.id))
    if _public_name_in_use(GlobalMetric, new_name, exclude_id=metric.id):
        suggestion = _suggest_unique_public_name(GlobalMetric, new_name, exclude_id=metric.id)
        return render_template(
            'resolve_name_collision.html',
            entity='metric',
            metric_id=metric.id, viz_id=None,
            current_name=new_name,
            suggested_name=suggestion,
            target='public',
            action_url=url_for('set_global_metric_visibility_confirm', metric_id=metric.id),
            collision_repeat=True,
        )
    old_name = metric.name
    metric.name = new_name
    metric.visibility = 'public'
    db.session.commit()
    if new_name != old_name:
        flash(f'Renamed "{old_name}" → "{new_name}" and promoted to public.', "success")
    else:
        flash(f'"{new_name}" is now public.', "success")
    return redirect(url_for('metrics_view', selected=metric.id))


@app.route('/global_visualization/<int:viz_id>/visibility', methods=['POST'])
@login_required
@owner_required(GlobalVisualization, 'viz_id')
def set_global_visualization_visibility(viz_id):
    viz = GlobalVisualization.query.get_or_404(viz_id)
    target = (request.form.get('visibility') or '').strip()
    if target not in ('public', 'private', 'unlisted'):
        flash("Invalid visibility.", "warning")
        return redirect(url_for('visualizations_view', selected=viz.id))
    if target == 'public' and _public_name_in_use(GlobalVisualization, viz.name, exclude_id=viz.id):
        suggestion = _suggest_unique_public_name(GlobalVisualization, viz.name, exclude_id=viz.id)
        return render_template(
            'resolve_name_collision.html',
            entity='visualization',
            metric_id=None, viz_id=viz.id,
            current_name=viz.name,
            suggested_name=suggestion,
            target='public',
            action_url=url_for('set_global_visualization_visibility_confirm', viz_id=viz.id),
        )
    viz.visibility = target
    db.session.commit()
    flash(f'"{viz.name}" is now {target}.', "success")
    return redirect(url_for('visualizations_view', selected=viz.id))


@app.route('/global_visualization/<int:viz_id>/visibility/confirm', methods=['POST'])
@login_required
@owner_required(GlobalVisualization, 'viz_id')
def set_global_visualization_visibility_confirm(viz_id):
    viz = GlobalVisualization.query.get_or_404(viz_id)
    new_name = (request.form.get('new_name') or '').strip()
    if not new_name:
        flash("Name can't be empty.", "warning")
        return redirect(url_for('visualizations_view', selected=viz.id))
    if _public_name_in_use(GlobalVisualization, new_name, exclude_id=viz.id):
        suggestion = _suggest_unique_public_name(GlobalVisualization, new_name, exclude_id=viz.id)
        return render_template(
            'resolve_name_collision.html',
            entity='visualization',
            metric_id=None, viz_id=viz.id,
            current_name=new_name,
            suggested_name=suggestion,
            target='public',
            action_url=url_for('set_global_visualization_visibility_confirm', viz_id=viz.id),
            collision_repeat=True,
        )
    old_name = viz.name
    viz.name = new_name
    viz.visibility = 'public'
    db.session.commit()
    if new_name != old_name:
        flash(f'Renamed "{old_name}" → "{new_name}" and promoted to public.', "success")
    else:
        flash(f'"{new_name}" is now public.', "success")
    return redirect(url_for('visualizations_view', selected=viz.id))


# ===================== FeatureRequest routes =====================

@app.route('/feature_requests', methods=['GET'])
@login_required
def feature_requests_list():
    """User-facing list of feature requests. Each user sees their own +
    everyone's public ones. Admins see them all via /admin/feature_requests."""
    mine = (
        FeatureRequest.query
        .filter_by(user_id=g.current_user.id)
        .order_by(FeatureRequest.created_at.desc())
        .all()
    )
    return render_template('feature_requests.html', mine=mine)


@app.route('/feature_requests/new', methods=['POST'])
@login_required
def feature_request_new():
    title = (request.form.get('title') or '').strip()
    kind = (request.form.get('kind') or 'feature').strip()
    desc = (request.form.get('description') or '').strip()
    if not title:
        flash("Title is required.", "warning")
        return redirect(url_for('feature_requests_list'))
    if kind not in ('feature', 'data_type', 'metric', 'visualization', 'other'):
        kind = 'other'
    fr = FeatureRequest(
        user_id=g.current_user.id, kind=kind,
        title=title[:200], description=desc or None,
    )
    db.session.add(fr); db.session.commit()
    flash("Request submitted — admins will see it.", "success")
    return redirect(url_for('feature_requests_list'))


def _resolve_lb_input_samples(lb):
    """Shared resolution for the /samples + /inputs.zip endpoints.

    Returns `(ds, input_fields, samples_rows, cfs_by_sample, cache_token)`.
    `ds` is None when the LB has no bound dataset. `cache_token` changes
    whenever the materialised subset is rebuilt (or the dataset sample
    set changes), so the client can key its on-disk input cache on it
    and re-download only when the underlying data actually moved."""
    dss = list(lb.datasets) if lb.datasets else []
    if not dss:
        return None, [], [], {}, f'empty:{lb.id}'
    ds = dss[0]
    # Per-LB role overrides (Leaderboard.field_roles_json) win over the
    # dataset field's own role — that's how an LB marks e.g. `img` as an
    # input even when the dataset declared it gt.
    try:
        _role_overrides = json.loads(lb.field_roles_json or '{}')
        if not isinstance(_role_overrides, dict):
            _role_overrides = {}
    except (TypeError, ValueError):
        _role_overrides = {}

    def _eff_role(df):
        return (_role_overrides.get(df.name) or df.role or 'gt').lower()

    input_fields = [df for df in ds.fields if _eff_role(df) == 'input']
    # Default: when nothing is explicitly an input but the dataset has
    # image field(s), treat those as the inputs (everything else is gt).
    # "An image field present ⇒ it's the input." Only fires when no
    # explicit input exists, so it never overrides a real designation.
    if not input_fields:
        input_fields = [df for df in ds.fields if df.kind == 'image']

    # Materialised subset filter (if the LB has a ready materialisation)
    sample_filter = None
    mat = getattr(lb, 'materialization', None)
    mat_ready = bool(mat and mat.status == 'ready')
    if mat_ready:
        from benchhub.lb_materialize import list_materialized_samples
        mat_names = list_materialized_samples(
            app.config['UPLOAD_FOLDER'], lb.id,
        )
        if mat_names:
            sample_filter = set(mat_names)

    samples_q = (Sample.query
                 .filter_by(dataset_id=ds.id)
                 .order_by(func.length(Sample.name), Sample.name))
    if sample_filter is not None:
        samples_q = samples_q.filter(Sample.name.in_(sample_filter))
    samples_rows = samples_q.all()
    sample_ids = [s.id for s in samples_rows]

    # Bulk-load the CustomField rows for the input fields across these
    # samples in one query — avoids N+1 lookups in the caller's loop.
    input_names = [df.name for df in input_fields]
    cfs_by_sample: dict[int, dict[str, CustomField]] = {}
    if sample_ids and input_names:
        cf_rows = (CustomField.query
                   .filter(CustomField.sample_id.in_(sample_ids),
                           CustomField.name.in_(input_names),
                           CustomField.submission_id.is_(None))
                   .all())
        for cf in cf_rows:
            cfs_by_sample.setdefault(cf.sample_id, {})[cf.name] = cf

    if mat_ready and getattr(mat, 'materialized_at', None):
        cache_token = f'mat:{lb.id}:{mat.materialized_at.isoformat()}'
    else:
        cache_token = f'ds:{ds.id}:{len(samples_rows)}'
    return ds, input_fields, samples_rows, cfs_by_sample, cache_token


@app.route('/api/leaderboard/<int:leaderboard_id>/samples', methods=['GET'])
def api_leaderboard_samples(leaderboard_id):
    """List the samples a submission to this LB should produce
    predictions for, plus URLs / inline values for every `role=input`
    field on the attached dataset.

    Response shape:
      {
        "leaderboard_id": int,
        "cache_token": "ds:<id>:<n>" | "mat:<lb>:<iso>",
        "input_fields": [{name, kind, params}, ...],
        "samples": [
          {"name": "s000000",
           "inputs": {
              "img":   {"kind": "image", "url": "/custom_field_image/<cf_id>"},
              "tags":  {"kind": "text",  "value": "foo,bar"},
           }},
          ...
        ]
      }

    File-backed kinds (image / mask / depth / audio) are returned as
    URLs the client GETs lazily; inline kinds (text / scalar / label /
    json / label_list) embed the decoded value directly. For the bulk
    path the client prefers `/inputs.zip` (one request) and only falls
    back to these per-URL fetches when the archive endpoint is absent.

    When the LB has a ready materialisation the sample list is the
    materialised subset; otherwise it falls back to the bound
    dataset's full sample list. Visibility-gated via cookie OR
    Bearer auth, same as /contract."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    user = getattr(g, 'current_user', None)
    if user is None:
        bearer = _bearer_token_from_request()
        if bearer:
            user, _ = _resolve_api_token(bearer)
    if not _can_view_parent(user, lb):
        abort(404)

    ds, input_fields, samples_rows, cfs_by_sample, cache_token = (
        _resolve_lb_input_samples(lb)
    )
    if ds is None:
        return jsonify({'leaderboard_id': lb.id, 'input_fields': [],
                         'samples': [], 'cache_token': cache_token})

    # Per-field params (Depth.unit, Mask.num_classes, LabelList.k, ...)
    # ride alongside the value/url so the client can reconstruct the
    # canonical bh.<Kind>(...) instance instead of a bare PIL or scalar.
    input_params = {df.name: (df.get_params() or {}) for df in input_fields}

    out_samples = []
    for s in samples_rows:
        bucket = cfs_by_sample.get(s.id, {})
        inputs: dict[str, dict] = {}
        for df in input_fields:
            cf = bucket.get(df.name)
            entry: dict = {'kind': df.kind, 'params': input_params[df.name]}
            if cf is None:
                entry['value'] = None
            elif df.kind in ('image', 'mask', 'depth', 'audio'):
                # File-backed; the client GETs this URL to download.
                entry['url'] = url_for(
                    'serve_custom_field_image', field_id=cf.id,
                    _external=False,
                )
            elif df.kind == 'json':
                try:
                    entry['value'] = json.loads(cf.value_text or 'null')
                except (TypeError, ValueError):
                    entry['value'] = cf.value_text
            elif df.kind in ('scalar', 'metric'):
                entry['value'] = cf.value_float
            elif df.kind == 'label':
                # value_text holds the JSON-encoded label (int or str);
                # decode so callers see the raw value.
                try:
                    entry['value'] = json.loads(cf.value_text or 'null')
                except (TypeError, ValueError):
                    entry['value'] = cf.value_text
            else:
                entry['value'] = cf.value_text
            inputs[df.name] = entry
        out_samples.append({'name': s.name, 'inputs': inputs})

    return jsonify({
        'leaderboard_id': lb.id,
        'cache_token': cache_token,
        'input_fields': [{'name': df.name, 'kind': df.kind,
                          'params': input_params[df.name]}
                          for df in input_fields],
        'samples': out_samples,
    })


@app.route('/api/leaderboard/<int:leaderboard_id>/inputs.zip', methods=['GET'])
def api_leaderboard_inputs_archive(leaderboard_id):
    """Stream ONE ZIP of every file-backed `role=input` field for the
    LB's sample set (materialised subset if present, else the full
    bound dataset). Layout mirrors the on-disk format: `<field>/<sample><ext>`.

    This is the bulk alternative to the per-sample fetch the client
    used to do — one request + one connection instead of N. The client
    extracts the archive into a local cache keyed on the `cache_token`
    from /samples, so a re-run (or a second submission to the same LB)
    skips the download entirely.

    Archive notes:
      * `ZIP_STORED` (no compression) — the payloads are already
        PNG/JPG/NPZ-compressed, so DEFLATE would burn CPU for ~0 gain.
        That makes the server-side pack near-free (essentially a copy).
      * Inline kinds (scalar / label / text / json / label_list) are NOT
        in the ZIP — they're tiny and ride in the /samples JSON.
      * Masks: the raw class-index sidecar (`.classid.png`, mode I;16)
        is packed in preference to the palette-applied display `.jpg`,
        so the client decodes a real `bh.Mask` instead of a `bh.Image`.

    Visibility-gated via cookie OR Bearer auth, same as /samples."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    user = getattr(g, 'current_user', None)
    if user is None:
        bearer = _bearer_token_from_request()
        if bearer:
            user, _ = _resolve_api_token(bearer)
    if not _can_view_parent(user, lb):
        abort(404)

    ds, input_fields, samples_rows, cfs_by_sample, _ = (
        _resolve_lb_input_samples(lb)
    )
    file_fields = [df for df in input_fields
                   if df.kind in ('image', 'mask', 'depth', 'audio')]

    from benchhub.lb_materialize import materialized_or_preview_path
    import tempfile
    import zipfile
    upload_folder = app.config['UPLOAD_FOLDER']

    tmp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    try:
        with zipfile.ZipFile(tmp_path, 'w',
                             compression=zipfile.ZIP_STORED) as zf:
            for s in samples_rows:
                bucket = cfs_by_sample.get(s.id, {})
                for df in file_fields:
                    cf = bucket.get(df.name)
                    if cf is None or not cf.value_text:
                        continue
                    rel = materialized_or_preview_path(
                        upload_folder, lb.id, df.name, s.name, cf.value_text,
                    )
                    abs_path = os.path.join(upload_folder, rel)
                    if df.kind == 'mask':
                        # Prefer the raw class-index sidecar; the display
                        # file is palette-RGB and only round-trips as an
                        # Image. Both BH-uploaded (raw value_text) and
                        # preview (.classid.png sidecar) cases land here.
                        stem = os.path.splitext(abs_path)[0]
                        sidecar = stem + '.classid.png'
                        if os.path.isfile(sidecar):
                            zf.write(sidecar, arcname=f'{df.name}/{s.name}.png')
                            continue
                    if not os.path.isfile(abs_path):
                        continue
                    ext = os.path.splitext(abs_path)[1] or ''
                    zf.write(abs_path, arcname=f'{df.name}/{s.name}{ext}')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        app.logger.error(f'inputs.zip build failed for lb {lb.id}: {e}')
        abort(500)

    @after_this_request
    def _cleanup_inputs_zip(response):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as err:
            app.logger.error(f'Error removing temp inputs.zip: {err}')
        return response

    return send_file(tmp_path, as_attachment=True,
                     download_name=f'lb_{lb.id}_inputs.zip',
                     mimetype='application/zip')


@app.route('/api/leaderboard/<int:leaderboard_id>/contract', methods=['GET'])
def api_leaderboard_contract(leaderboard_id):
    """Live pred contract for an LB.

    Returns the same shape `import_typed_submission` consumes server-
    side: a JSON array of `{name, kind, params, role}` entries. Used
    by `bh.Client.leaderboard_contract(lb_id)` to pre-validate
    predictions locally before any upload, including shape_match
    cross-checks. Visibility-gated through the LB's own visibility
    rules (private LBs 404 to non-owners).

    Accepts either cookie auth (browser sessions) or an Authorization:
    Bearer token (API clients + the Colab notebook). Without the
    bearer fall-through, the notebook hits a 404 on private LBs even
    when it's holding a valid submission token bound to that LB."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    # Cookie-authed users come in with g.current_user pre-populated by
    # load_user. Bearer-authed callers don't; resolve the token if any
    # so they get treated as the owning user for the visibility check.
    user = getattr(g, 'current_user', None)
    if user is None:
        bearer = _bearer_token_from_request()
        if bearer:
            user, _ = _resolve_api_token(bearer)
    if not _can_view_parent(user, lb):
        abort(404)
    return jsonify(_lb_pred_contract_from_dataset_fields(lb))


@app.route('/api/leaderboard/<int:leaderboard_id>/submissions', methods=['GET'])
def api_leaderboard_submissions(leaderboard_id):
    """List submissions on an LB (name + id + upload time). Used by
    `bh.Client.list_submissions(lb_id)` / `submission_exists(lb_id, name)`
    so a submitter can avoid clobbering / duplicating an existing run.

    Optional `?name=<name>` filters to exact-name matches (the client's
    existence check passes this so it doesn't have to pull the whole
    list). Visibility-gated like /contract + /samples; cookie OR Bearer.
    Mirrored (non-verified) submissions are included — they still occupy
    the name on the board."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    user = getattr(g, 'current_user', None)
    if user is None:
        bearer = _bearer_token_from_request()
        if bearer:
            user, _ = _resolve_api_token(bearer)
    if not _can_view_parent(user, lb):
        abort(404)

    name_filter = (request.args.get('name') or '').strip()
    q = Submission.query.filter_by(leaderboard_id=lb.id)
    if name_filter:
        q = q.filter(Submission.name == name_filter)
    rows = q.order_by(Submission.upload_date.desc()).all()
    return jsonify({
        'leaderboard_id': lb.id,
        'count': len(rows),
        'submissions': [
            {'id': s.id, 'name': s.name,
             'upload_date': s.upload_date.isoformat() if s.upload_date else None}
            for s in rows
        ],
    })


@app.route('/api/submit/<int:leaderboard_id>', methods=['POST'])
@require_api_token
def api_submit_typed(leaderboard_id):
    """Programmatic submission endpoint for the Phase B typed-client.

    Expects a multipart upload with a single `submission_zip` field
    holding a ZIP whose root contains a `manifest.json` + one
    sub-directory per declared prediction (`<field>/<sample>.<ext>`).

    The server validates the manifest against `Leaderboard.required_pred_fields_json`
    (every required pred name must be present with a matching kind),
    creates a Submission row + per-prediction CustomField rows, copies
    file-backed kinds into `uploads/submissions/<id>/`, and enqueues
    the Celery `process_submission` task. Returns the submission id
    + a relative view URL.
    """
    import tempfile
    import zipfile

    lb = Leaderboard.query.get_or_404(leaderboard_id)

    # Submission-scoped tokens (Open-in-Colab flow) are bound to a
    # single leaderboard. Reject cross-LB use; bump used_count so
    # admins can spot abuse later.
    st = g.get('submission_token')
    if st is not None:
        if st.leaderboard_id != lb.id:
            return jsonify({
                'error': 'This submission token is scoped to a different '
                         f'leaderboard (#{st.leaderboard_id}).',
            }), 403
        st.used_count = (st.used_count or 0) + 1
        db.session.commit()

    upload = request.files.get('submission_zip')
    if upload is None or not upload.filename:
        return jsonify({'error': 'multipart field "submission_zip" required'}), 400
    name = (request.form.get('name') or '').strip() or None
    description = (request.form.get('description') or '').strip() or None
    # Link is rendered as an href on the submission name — only accept
    # http(s) so a `javascript:`/`data:` URL can't be injected. Anything
    # else is dropped (the name falls back to the comparison-view link).
    link = (request.form.get('link') or '').strip() or None
    if link and not re.match(r'^https?://', link, re.IGNORECASE):
        link = None
    if link and len(link) > 500:
        link = None

    from benchhub.manifest import import_typed_submission
    # Prefer the dataset-derived contract over the legacy
    # `required_pred_fields_json` column. The helper falls back to
    # the override when set.
    contract = _lb_pred_contract_from_dataset_fields(lb)

    # Shape-match resolver: for the manifest's shape-constrained pred
    # fields, look up each sample's input field shape so the
    # importer can cross-check. Indexed by (sample_name, field) on
    # first miss; misses degrade to "couldn't check" (the importer
    # is permissive when either side has no spatial shape).
    upload_folder = app.config['UPLOAD_FOLDER']

    def _get_input_shape(sample_name, input_field_name):
        for ds in (lb.datasets or []):
            sample = next(
                (s for s in (ds.samples or []) if s.name == sample_name),
                None,
            )
            if sample is None:
                continue
            cf = next(
                (c for c in (sample.custom_fields or [])
                 if c.name == input_field_name),
                None,
            )
            if cf is None:
                continue
            try:
                inst = _cf_to_typed_instance(cf, upload_folder)
            except Exception:
                return None
            arr = getattr(inst, 'array', None)
            if arr is None or getattr(arr, 'ndim', 0) < 2:
                return None
            return tuple(arr.shape[:2])
        return None

    with tempfile.TemporaryDirectory(prefix='bh_sub_') as extract_dir:
        try:
            with zipfile.ZipFile(upload.stream) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            return jsonify({'error': 'submission_zip is not a valid ZIP file'}), 400

        # Allow a single top-level folder inside the ZIP (the common
        # shape — `mymodel/manifest.json` rather than bare).
        roots = [p for p in os.listdir(extract_dir) if not p.startswith('.')]
        if (len(roots) == 1
                and os.path.isdir(os.path.join(extract_dir, roots[0]))
                and not os.path.exists(os.path.join(extract_dir, 'manifest.json'))):
            source_root = os.path.join(extract_dir, roots[0])
        else:
            source_root = extract_dir

        try:
            sub_id, summary = import_typed_submission(
                source_root,
                leaderboard=lb,
                submission_name=name,
                db_session=db.session,
                Submission=Submission,
                CustomField=CustomField,
                upload_folder=app.config['UPLOAD_FOLDER'],
                owner_user_id=g.current_user.id,
                contract=contract,
                get_input_shape=_get_input_shape,
            )
            # Attach the submitter-supplied blurb + link (form fields,
            # alongside `name`). Set post-import so manifest.py stays
            # free of presentation metadata.
            if description is not None or link is not None:
                sub_row = Submission.query.get(sub_id)
                if sub_row is not None:
                    if description is not None:
                        sub_row.description = description
                    if link is not None:
                        sub_row.link = link
            db.session.commit()
        except FileNotFoundError as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 400
        except ValueError as e:
            db.session.rollback()
            return jsonify({'error': f'submission invalid: {e}'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'submission ingest failed: {e}'}), 500

    # Enqueue the eval task — lazy import per the circular-import shape
    # documented in CLAUDE.md.
    import tasks
    try:
        tasks.process_submission.delay(sub_id)
    except Exception as e:
        # Don't fail the request — the row is in the DB and the worker
        # can be reaped manually. Just flag it in the response.
        summary['enqueue_warning'] = str(e)

    summary['url'] = url_for(
        'submission_view', submission_id=sub_id, _external=False,
    ) if 'submission_view' in app.view_functions else f'/submission/{sub_id}'
    return jsonify(summary), 201


@app.route('/admin/import_from_hf', methods=['GET'])
@login_required
def admin_import_from_hf():
    """Single-input form: HF repo ID. POST → preview with the parsed
    Croissant schema partially filled in (kinds inferred from
    Croissant's typed schema; roles + params left empty for the
    admin to fill).

    The form's `repo_id` input is wired to two helper endpoints —
    `admin_import_from_hf_search` for type-ahead suggestions and
    `admin_import_from_hf_trending` for the curated by-domain
    grid — so the admin doesn't need to memorise repo IDs."""
    if not is_admin(g.current_user):
        abort(403)
    return render_template('admin_import_from_hf.html')


@app.route('/admin/import_from_hf/search')
@login_required
def admin_import_from_hf_search():
    """JSON proxy over HF Hub's /api/datasets?search=... endpoint.
    Behind an admin gate just so we don't spend our shared IP's
    quota on anonymous traffic — the data itself is public."""
    if not is_admin(g.current_user):
        abort(403)
    from benchhub.hf_search import search_datasets
    q = (request.args.get('q') or '').strip()
    return jsonify(search_datasets(q, limit=10))


@app.route('/admin/import_from_hf/trending')
@login_required
def admin_import_from_hf_trending():
    """JSON: top-downloaded HF datasets per ML domain (Vision / NLP /
    Audio / Tabular). One-hour TTL cache lives in the helper so the
    grid isn't a fresh HF round-trip on every page reload."""
    if not is_admin(g.current_user):
        abort(403)
    from benchhub.hf_search import trending_by_domain
    return jsonify(trending_by_domain(limit_per_domain=5))


@app.route('/admin/import_from_hf/commit', methods=['POST'])
@login_required
def admin_import_from_hf_commit():
    """Read the admin's preview-form selections, download the HF split,
    materialise rows into a typed-manifest staging dir, then call the
    standard `import_typed_dataset` to write Dataset + Sample +
    CustomField rows. Heavy lift — can take minutes for large
    datasets; consider running this in a background task once
    request size grows."""
    if not is_admin(g.current_user):
        abort(403)

    repo_id = (request.form.get('repo_id') or '').strip()
    dataset_name = (request.form.get('dataset_name') or '').strip() or repo_id
    split = (request.form.get('split') or '').strip() or None
    # sample_cap = -1 (or 0) → no cap, import everything. Anything ≥1
    # caps to that count. Storage-quota pre-check below stops a too-big
    # unlimited import before any HF bytes are downloaded.
    raw_cap = (request.form.get('sample_cap') or '').strip()
    try:
        sample_cap = int(raw_cap) if raw_cap else -1
    except ValueError:
        sample_cap = -1
    if 0 < sample_cap:
        pass  # explicit positive cap
    else:
        sample_cap = -1  # canonicalise 0 / negative / missing to -1
    sampling = (request.form.get('sampling') or 'head').strip().lower()
    if sampling not in ('head', 'uniform', 'stratified'):
        sampling = 'head'
    try:
        sampling_seed = int(request.form.get('sampling_seed') or 42)
    except ValueError:
        sampling_seed = 42

    names = request.form.getlist('field_name')
    kinds = request.form.getlist('field_kind')
    params_raw = request.form.getlist('field_params')
    source_columns = request.form.getlist('field_source_column')
    sample_name_from = (request.form.get('sample_name_from') or '').strip() or None

    if not (len(names) == len(kinds) == len(params_raw)):
        flash('Field rows malformed — please retry from the preview page.', 'danger')
        return redirect(url_for('admin_import_from_hf'))

    # Dataset is now role-neutral — every imported field is just
    # data. LBs choose how to use each field (input vs gt vs skip)
    # at creation time. We still set DatasetField.role to a
    # placeholder value (`'gt'`) so the column isn't NULL; downstream
    # LB creation overrides via `field_roles_json`.
    selected: list[dict] = []
    for i, name in enumerate(names):
        source_column = (source_columns[i] if i < len(source_columns) else name) or name
        # If this field's source column is the one chosen as the
        # sample-name source, drop it from the field list — the
        # values are already represented as the sample names on
        # disk + in the Sample.name column, and re-importing them as
        # a separate text field would duplicate the same information
        # in every comparison view.
        if sample_name_from and source_column == sample_name_from:
            continue
        kind = kinds[i] if i < len(kinds) else 'json'
        params_str = (params_raw[i] if i < len(params_raw) else '').strip()
        try:
            params = json.loads(params_str) if params_str else {}
            if not isinstance(params, dict):
                params = {}
        except (TypeError, ValueError):
            params = {}
        selected.append({
            'name': name,
            'source_column': source_column,
            'kind': kind,
            'role': 'gt',  # placeholder; LB-side `field_roles_json` is canonical.
            'params': params,
        })
    if not selected:
        flash('No fields to import.', 'warning')
        return redirect(url_for('admin_import_from_hf'))

    import tempfile
    from benchhub.manifest import import_typed_dataset
    from benchhub.hf_search import fetch_split_byte_sizes, fetch_split_row_counts
    try:
        from benchhub.hf_materialize import materialize_hf_to_typed_dir
    except Exception as e:
        flash(f"HF library unavailable: {e}", 'danger')
        return redirect(url_for('admin_import_from_hf'))

    # Pre-materialize storage-quota check. Estimate the on-disk size
    # of the import by pro-rating the HF parquet-file bytes per row
    # against the requested sample_cap (or the full split when
    # cap=-1). 1.5x headroom covers materialization expansion
    # (parquet → PNG/NPZ). Bail BEFORE downloading anything when the
    # admin would blow past their storage cap; the post-materialize
    # path still recomputes from disk as the authoritative number.
    byte_sizes = fetch_split_byte_sizes(repo_id)
    row_counts = fetch_split_row_counts(repo_id)
    chosen_split = split or next(iter(row_counts), None)
    est_bytes = 0
    if chosen_split and byte_sizes.get(chosen_split) and row_counts.get(chosen_split):
        split_bytes = byte_sizes[chosen_split]
        split_rows = row_counts[chosen_split]
        if sample_cap == -1:
            est_bytes = split_bytes
        else:
            est_bytes = int(split_bytes * min(sample_cap, split_rows) / split_rows)
        est_bytes = int(est_bytes * 1.5)  # materialization headroom
    if est_bytes > 0:
        # Admin HF imports land as visibility='public' (see Dataset row
        # below), so charge the public bucket.
        ok, msg = check_quota(g.current_user, kind='dataset_create',
                              incoming_bytes=est_bytes,
                              visibility='public')
        if not ok:
            flash(f"{msg} (estimated {_format_bytes(est_bytes)} for this import)",
                  'danger')
            return redirect(url_for('admin_import_from_hf'))

    # Pre-create the Dataset row with `import_status='importing'` so
    # the row appears on /datasets immediately and the admin can
    # navigate away. The background task fills in Sample +
    # CustomField + DatasetField rows and flips the status to
    # 'ready' (or 'failed' with an `import_error` payload).
    try:
        ds_row = Dataset(
            name=dataset_name,
            owner_user_id=g.current_user.id,
            visibility='public',
            import_status='importing',
            import_progress_json=json.dumps({
                'phase': 'queued', 'current': 0, 'total': 0,
                'message': 'Waiting for a worker to pick up the import…',
            }),
        )
        db.session.add(ds_row)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Couldn't create dataset row: {e}", 'danger')
        return redirect(url_for('admin_import_from_hf'))

    import tasks as _tasks
    async_result = _tasks.run_hf_import.delay(
        dataset_id=ds_row.id,
        repo_id=repo_id,
        split=split,
        sample_cap=sample_cap,
        sampling=sampling,
        sampling_seed=sampling_seed,
        dataset_name=dataset_name,
        fields=selected,
        sample_name_from=sample_name_from,
        hf_token=getattr(g.current_user, 'hf_token', None),
        owner_user_id=g.current_user.id,
    )
    try:
        ds_row.import_task_id = async_result.id
        db.session.commit()
    except Exception:
        db.session.rollback()
    flash(
        f"Importing {repo_id} in the background. You can leave this "
        f"page — progress lives on the dataset page below and on /datasets.",
        'success',
    )
    return redirect(url_for('dataset_view', dataset_id=ds_row.id))


@app.route('/admin/import_from_hf/preview', methods=['POST'])
@login_required
def admin_import_from_hf_preview():
    """Fetch Croissant, parse it, render the preview form.

    Each detected field becomes a row in a table — the kind is
    pre-filled from the deterministic Croissant type map (admin can
    override e.g. image → mask for segmentation columns). Role
    (input / gt / skip) and per-instance params are empty for the
    admin to fill in before the commit step."""
    if not is_admin(g.current_user):
        abort(403)
    from benchhub.hf_croissant import (
        CroissantFetchError, fetch_croissant, parse_croissant,
    )
    from benchhub.types import DTYPES

    from benchhub.hf_croissant import schema_from_hf_features
    from benchhub.hf_search import fetch_dataset_info

    repo_id = (request.form.get('repo_id') or '').strip()
    if not repo_id:
        flash("Enter an HF dataset repo ID first.", "warning")
        return redirect(url_for('admin_import_from_hf'))

    # Try Croissant first (richer, when present); fall through to the
    # datasets-server /info endpoint which has much broader coverage
    # (HF auto-generates Croissant only for YAML-conformant repos
    # with parquet bytes; /info indexes anything HF can stream).
    schema = None
    croissant_err = None
    try:
        doc = fetch_croissant(repo_id)
        schema = parse_croissant(doc)
    except (CroissantFetchError, ValueError) as e:
        croissant_err = e
    if schema is None:
        info = fetch_dataset_info(repo_id)
        if info is not None:
            schema = schema_from_hf_features(
                info['features'],
                info.get('splits') or [],
                name=repo_id,
            )
    if schema is None:
        # Neither source had anything — surface the original Croissant
        # error since that's the primary path; the /info fallback's
        # silent failure isn't user-actionable.
        flash(
            f"Couldn't read schema for {repo_id!r}: {croissant_err}. "
            "No Croissant document and no /info schema available — "
            "the dataset may need a parquet conversion or a config fix "
            "upstream.",
            "danger",
        )
        return redirect(url_for('admin_import_from_hf'))

    # Per-split row counts so the form can show "500 out of 10,000"
    # next to the max-samples input. Best-effort: an empty dict
    # degrades the UI cleanly to "no count available".
    from benchhub.hf_search import fetch_class_label_vocabs, fetch_split_row_counts
    split_counts = fetch_split_row_counts(repo_id)
    # Pull HF ClassLabel.names off the datasets-server /info endpoint
    # so the label-kind params textarea can pre-fill the vocab — the
    # materializer would lift it anyway at commit time, but showing
    # it on the preview makes the dataset self-explanatory.
    class_label_vocabs = fetch_class_label_vocabs(repo_id)

    # Classification datasets get stratified sampling by default —
    # `kind=label` is the Croissant parser's signal that this looks
    # like a labelled dataset. The admin can still flip back to
    # uniform / head on the form.
    has_label_field = any(f.kind == 'label' for f in schema.fields)

    return render_template(
        'admin_import_from_hf_preview.html',
        repo_id=repo_id,
        schema=schema,
        split_counts=split_counts,
        all_kinds=sorted(DTYPES),
        # Each HF column carries actual data — the role choices here
        # are only "where does the data live in the BH contract":
        # `input` (given to submitters), `gt` (held server-side as
        # the target), or `skip` (not imported). `pred` declarations
        # are schema-only and live in a separate section of the form.
        all_roles=['input', 'gt', 'skip'],
        has_label_field=has_label_field,
        class_label_vocabs=class_label_vocabs,
    )


def _ingest_typed_dataset_zip(upload, *, owner_user, visibility):
    """Common ZIP → typed-import pipeline used by both the bearer-
    token API route and the cookie-auth browser-upload route.

    Returns `(summary_dict, http_status, error_message)`. On success
    `summary_dict` is non-None and `error_message` is None.
    """
    import tempfile
    import zipfile

    if upload is None or not getattr(upload, 'filename', None):
        return None, 400, 'multipart "dataset_zip" field required'
    if visibility not in ('public', 'private', 'unlisted'):
        visibility = 'public'

    from benchhub.manifest import import_typed_dataset
    with tempfile.TemporaryDirectory(prefix='bh_ds_') as extract_dir:
        try:
            with zipfile.ZipFile(upload.stream) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            return None, 400, 'dataset_zip is not a valid ZIP file'

        roots = [p for p in os.listdir(extract_dir) if not p.startswith('.')]
        if (len(roots) == 1
                and os.path.isdir(os.path.join(extract_dir, roots[0]))
                and not os.path.exists(os.path.join(extract_dir, 'manifest.json'))):
            source_root = os.path.join(extract_dir, roots[0])
        else:
            source_root = extract_dir

        # Quota check (per-user storage cap): the import_typed_dataset
        # pass below copies every file into uploads/datasets/<id>/, so
        # the about-to-be-written bytes are exactly the sum of file
        # sizes under `source_root` after extraction. Reject early
        # with 413 if it would push the user over the cap.
        incoming_bytes = 0
        for dirpath, _, filenames in os.walk(source_root):
            for fn in filenames:
                try:
                    incoming_bytes += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    continue
        ok, msg = check_quota(owner_user, kind='dataset_create',
                              incoming_bytes=incoming_bytes,
                              visibility=visibility)
        if not ok:
            return None, 413, msg

        try:
            _, summary = import_typed_dataset(
                source_root,
                db_session=db.session,
                Dataset=Dataset, Sample=Sample, CustomField=CustomField,
                DatasetField=DatasetField,
                upload_folder=app.config['UPLOAD_FOLDER'],
                owner_user_id=owner_user.id,
                visibility=visibility,
            )
            db.session.commit()
        except FileNotFoundError as e:
            db.session.rollback()
            return None, 400, str(e)
        except ValueError as e:
            db.session.rollback()
            return None, 400, f'manifest invalid: {e}'
        except Exception as e:
            db.session.rollback()
            return None, 500, f'import failed: {e}'

    return summary, 201, None


@app.route('/api/datasets', methods=['POST'])
@require_api_token
def api_create_dataset():
    """Programmatic dataset creation for the typed-client (`BHDatasetCreator`).

    Expects a multipart upload with a single `dataset_zip` field
    whose ZIP root contains a `manifest.json` declaring `fields[]`
    (with role=input|gt) and `samples[]`, plus one sub-directory
    per declared field holding `<sample>.<ext>` for each sample.

    Open to any authenticated user — the per-user storage quota
    (`User.quota_max_storage_bytes`, default 50 MB) is what gates
    abuse, not an admin allow-list.
    """
    upload = request.files.get('dataset_zip')
    visibility = (request.form.get('visibility') or 'public').strip()
    summary, status, err = _ingest_typed_dataset_zip(
        upload, owner_user=g.current_user, visibility=visibility,
    )
    if err:
        return jsonify({'error': err}), status
    return jsonify(summary), status


@app.route('/datasets/upload', methods=['POST'])
@login_required
def upload_typed_dataset_zip():
    """Browser-facing companion to /api/datasets — same ZIP shape,
    same per-user quota gate, cookie-authenticated. The /datasets
    page posts to this so non-API users can drop a typed-manifest
    ZIP without juggling tokens.
    """
    upload = request.files.get('dataset_zip')
    # User-uploaded datasets default to private — owners explicitly
    # publish via the "Publish" button on /dataset/<id>. The form can
    # override (`?visibility=public`) for ZIP imports that should land
    # public immediately, but the safe default is private.
    visibility = (request.form.get('visibility') or 'private').strip()
    summary, status, err = _ingest_typed_dataset_zip(
        upload, owner_user=g.current_user, visibility=visibility,
    )
    if err:
        flash(err, 'danger')
        return redirect(url_for('datasets_list'))
    flash(
        f"Uploaded dataset \"{summary['name']}\" "
        f"({summary['samples']} samples × {summary['fields']} fields, "
        f"{summary['bytes_on_disk']} bytes).",
        'success',
    )
    if 'dataset_view' in app.view_functions:
        return redirect(url_for('dataset_view', dataset_id=summary['dataset_id']))
    return redirect(url_for('datasets_list'))


@app.route('/admin/import_typed_dataset', methods=['POST'])
@login_required
def admin_import_typed_dataset():
    """Admin-only: materialise a typed dataset from a server-side path.

    Phase B replacement for the deleted ZIP upload flow. The caller
    (admin running a seed script over SSH, or the on-box CLI) hands
    over an absolute path to a directory containing a `manifest.json`
    + per-field sub-directories. The importer writes Dataset + Sample +
    CustomField rows and copies file-backed kinds under
    `uploads/datasets/<dataset_id>/`. Inline kinds (scalar, label)
    are decoded and stashed directly on the CustomField row.

    Request JSON: `{"source_path": "/abs/path/to/dataset_root"}`.
    Response JSON: the importer's summary dict on success.
    """
    if not is_admin(g.current_user):
        abort(403)
    body = request.get_json(silent=True) or {}
    source_path = (body.get('source_path') or '').strip()
    if not source_path:
        return jsonify({'error': 'source_path required'}), 400
    if not os.path.isabs(source_path):
        return jsonify({'error': 'source_path must be absolute'}), 400
    if not os.path.isdir(source_path):
        return jsonify({'error': f'not a directory: {source_path}'}), 400

    from benchhub.manifest import import_typed_dataset
    try:
        _, summary = import_typed_dataset(
            source_path,
            db_session=db.session,
            Dataset=Dataset, Sample=Sample, CustomField=CustomField,
            DatasetField=DatasetField,
            upload_folder=app.config['UPLOAD_FOLDER'],
            owner_user_id=g.current_user.id,
        )
        db.session.commit()
    except FileNotFoundError as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400
    except ValueError as e:
        db.session.rollback()
        return jsonify({'error': f'manifest invalid: {e}'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'import failed: {e}'}), 500
    return jsonify(summary), 201


@app.route('/admin/feature_requests', methods=['GET'])
@login_required
def admin_feature_requests():
    if not is_admin(g.current_user):
        abort(403)
    rows = (
        FeatureRequest.query
        .order_by(FeatureRequest.status.asc(), FeatureRequest.created_at.desc())
        .all()
    )
    return render_template('admin_feature_requests.html', rows=rows)


@app.route('/admin/feature_requests/<int:req_id>/status', methods=['POST'])
@login_required
def admin_feature_request_status(req_id):
    if not is_admin(g.current_user):
        abort(403)
    fr = FeatureRequest.query.get_or_404(req_id)
    target = (request.form.get('status') or '').strip()
    if target in ('open', 'planned', 'in_progress', 'resolved', 'declined'):
        fr.status = target
    note = (request.form.get('admin_note') or '').strip()
    if note:
        fr.admin_note = note
    db.session.commit()
    flash(f'Updated #{fr.id}.', "success")
    return redirect(url_for('admin_feature_requests'))


@app.route('/admin')
@login_required
def admin_index():
    """Index page listing every admin tool. Linked from the navbar's
    `Admin` chip so admins don't have to dig through the user menu."""
    if not is_admin(g.current_user):
        abort(403)
    return render_template('admin_index.html')


@app.route('/admin/lb_templates')
@login_required
def admin_lb_templates():
    if not is_admin(g.current_user):
        abort(403)
    templates = (
        LbTemplate.query
        .order_by(LbTemplate.sort_order, LbTemplate.id)
        .all()
    )
    # Build kind/metric/viz pickers for the form. Just the catalog
    # names; the admin types arbitrary strings if they want a
    # not-yet-existing entry.
    metric_names = sorted({
        m.name for m in GlobalMetric.query.filter_by(visibility='public').all()
    })
    viz_names = sorted({
        v.name for v in GlobalVisualization.query.filter_by(visibility='public').all()
    })
    kind_names = sorted([
        'image', 'mask', 'depth', 'audio', 'text', 'bboxes',
        'label', 'label_list', 'scalar', 'json',
    ])
    return render_template(
        'admin_lb_templates.html',
        templates=templates,
        metric_names=metric_names,
        viz_names=viz_names,
        kind_names=kind_names,
    )


@app.route('/admin/lb_templates/save', methods=['POST'])
@login_required
def admin_lb_template_save():
    """Upsert a template by slug. Multi-value form fields:
    required_kinds[], metric_names[], visualization_names[].
    Returns to the admin index."""
    if not is_admin(g.current_user):
        abort(403)
    slug = (request.form.get('slug') or '').strip()
    if not slug:
        flash('Slug is required', 'danger')
        return redirect(url_for('admin_lb_templates'))
    row = LbTemplate.query.filter_by(slug=slug).first()
    if row is None:
        row = LbTemplate(slug=slug)
        db.session.add(row)
    row.label = (request.form.get('label') or '').strip() or slug
    row.description = (request.form.get('description') or '').strip() or None
    row.required_kinds_json = json.dumps([
        s.strip() for s in request.form.getlist('required_kinds')
        if s and s.strip()
    ])
    row.metric_names_json = json.dumps([
        s.strip() for s in request.form.getlist('metric_names')
        if s and s.strip()
    ])
    row.visualization_names_json = json.dumps([
        s.strip() for s in request.form.getlist('visualization_names')
        if s and s.strip()
    ])
    row.enabled = request.form.get('enabled') in ('1', 'on', 'true')
    try:
        row.sort_order = int(request.form.get('sort_order') or 0)
    except ValueError:
        row.sort_order = 0
    db.session.commit()
    flash(f'Saved template "{slug}".', 'success')
    return redirect(url_for('admin_lb_templates'))


@app.route('/admin/lb_templates/<int:tid>/delete', methods=['POST'])
@login_required
def admin_lb_template_delete(tid):
    if not is_admin(g.current_user):
        abort(403)
    row = LbTemplate.query.get_or_404(tid)
    slug = row.slug
    db.session.delete(row)
    db.session.commit()
    flash(f'Deleted template "{slug}".', 'success')
    return redirect(url_for('admin_lb_templates'))


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
        
        # User-created visualizations default to private. Admins default
        # to public (curated content). Toggle via the detail pane.
        default_vis = 'public' if is_admin(g.current_user) else 'private'
        new_viz = GlobalVisualization(
            name=name,
            description=description,
            python_code=python_code,
            is_aggregated='is_aggregated' in request.form,
            accepts_aggregated_inputs='accepts_aggregated_inputs' in request.form,
            owner_user_id=g.current_user.id,
            visibility=default_vis,
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


def _pred_field_constructor_lines(lb, *, indent: str = '                  ') -> list[str]:
    """Build the per-pred-field constructor lines for an LB's
    `sub.predict(...)` call. One line per field declared on the
    LB's contract; format mirrors the submitter's actual call so
    they can copy-paste directly.

    Used by both the downloadable submission script + notebook and
    the inline Python snippet on the LB page, so they always show
    the LB's CURRENT contract (e.g. cifar10 with a top-K pred
    surfaces a `bh.LabelList(..., k=5)` line instead of a generic
    `bh.Label`).
    """
    pred_fields = _lb_pred_contract_from_dataset_fields(lb)
    lines: list[str] = []
    for f in pred_fields:
        kind = f.get('kind', 'scalar')
        params = f.get('params') or {}
        name = f['name']
        # Reference the value `my_model` returns in the predictions dict
        # so the generated line is plug-and-play. The user only has to
        # write my_model() — the wrapping is template-generated.
        ref = f"predictions[{name!r}]"
        if kind == 'label':
            line = f"{indent}{name}=bh.Label({ref}),                       # int or str class id"
        elif kind == 'label_list':
            k = params.get('k', 5)
            line = f"{indent}{name}=bh.LabelList({ref}, k={k}),   # top-{k} class ids (descending confidence)"
        elif kind == 'scalar':
            line = f"{indent}{name}=bh.Scalar({ref}),"
        elif kind == 'text':
            line = f"{indent}{name}=bh.Text({ref}),"
        elif kind == 'depth':
            unit = params.get('unit', 'meters')
            line = f"{indent}{name}=bh.Depth({ref}, unit={unit!r}),"
        elif kind == 'mask':
            line = f"{indent}{name}=bh.Mask({ref}),"
        elif kind == 'image':
            line = f"{indent}{name}=bh.Image({ref}),"
        elif kind == 'bboxes':
            fmt = params.get('format', 'xyxy')
            line = f"{indent}{name}=bh.BBoxes({ref}, format={fmt!r}),"
        elif kind == 'audio':
            line = f"{indent}{name}=bh.Audio({ref}[0], {ref}[1]),  # (samples, sample_rate)"
        elif kind == 'json':
            line = f"{indent}{name}=bh.Json({ref}),"
        else:
            line = f"{indent}{name}={ref},  # kind={kind}"
        lines.append(line)
    if not lines:
        lines = [
            f"{indent}# No pred fields declared on this leaderboard yet.",
            f"{indent}# Add them on the LB-creation page (or in the LB settings).",
        ]
    return lines


def _submission_script_source(lb) -> str:
    """Generate a self-contained submission script for an LB. Reads
    BENCHHUB_API_TOKEN from the environment, walks placeholder sample
    names, and POSTs the ZIP to /api/submit/<lb_id>. The user fills
    in the actual prediction logic — sample names + field
    construction — between the marked `# === fill in: ===` blocks."""
    pred_snippets = _pred_field_constructor_lines(lb)

    return f'''#!/usr/bin/env python
"""Submission script for BenchHub leaderboard {lb.id} ({lb.name!r}).

Reads BENCHHUB_API_TOKEN from your environment, runs your model over
every sample on the leaderboard, and uploads the predictions.

Setup:
    pip install benchhub-client    # ships `bh.Client`, types, etc.
    export BENCHHUB_API_TOKEN=<your token from BenchHub Settings>

Then:
    python {lb.name.replace(' ', '_')}_submit.py
"""
import os
from typing import Any
import numpy as np
import benchhub as bh

LEADERBOARD_ID: int = {lb.id}
BASE_URL: str = {(_get_base_url() or 'https://runbenchhub.com')!r}

if not os.environ.get('BENCHHUB_API_TOKEN'):
    raise SystemExit(
        "BENCHHUB_API_TOKEN is not set. Get a token from "
        f"{{BASE_URL}}/settings and run: export BENCHHUB_API_TOKEN=<token>"
    )

client: bh.Client = bh.Client(base_url=BASE_URL)
contract: list[dict[str, Any]] = client.leaderboard_contract(LEADERBOARD_ID)
print(f"LB {{LEADERBOARD_ID}} pred contract: {{contract}}")

# Samples come from the BenchHub server — no need to maintain a local
# copy of the dataset. `client.iter_samples(LB_ID)` yields
# (sample_name, inputs_dict) for every sample on the LB's bound
# dataset (or its materialised subset, if any). Each input field
# arrives decoded: image / mask → PIL.Image, depth → PIL.Image of the
# colormapped preview, scalar/label/text/json → the raw value.

# === fill in: your model ====================================================
def my_model(inputs: dict[str, Any]) -> dict[str, Any]:
    """Take the per-sample inputs dict and return the values BenchHub
    expects for each pred field declared on the LB's contract.

    `inputs` keys are the dataset's role=input fields. Each value is
    the matching `bh.<Kind>` instance — e.g. `bh.Image` (use `.array`
    for a (H,W,3) uint8 ndarray), `bh.Mask`, `bh.Depth`. Inline kinds
    (scalar / label / text / json) arrive as their decoded Python value.

    Return a dict whose keys are the LB's pred-field names; the values
    are unwrapped (plain ints / strings / arrays / etc.), the wrapping
    in bh.<Kind>(...) below splats them into typed instances.

    For this LB the contract is:
      {{contract}}

    Replace this stub. The dict you return is splatted into
    `sub.predict(sample_name, **predictions)`."""
    raise NotImplementedError(
        "Replace my_model() with your inference. Return a dict whose keys "
        "match the LB's pred-field names."
    )
# ===========================================================================

sub = client.submission(LEADERBOARD_ID)
sample_name: str
inputs: dict[str, Any]
for sample_name, inputs in client.iter_samples(LEADERBOARD_ID):
    predictions: dict[str, Any] = my_model(inputs)
    # The generated bh.<Kind>(...) calls below match the LB's current
    # contract — wrap your `predictions` values into the typed objects
    # the server expects.
    sub.predict(sample_name,
{chr(10).join(pred_snippets)}
                )

result: dict[str, Any] = sub.submit(
    name='my submission',
    description='one-line summary (shown in the LB table, full text on hover)',
    link='https://github.com/you/your-model',   # optional; the name links here
)
print(result)  # → {{'submission_id': ..., 'view_url': '...'}}
'''


def _submission_notebook_source(lb, *, inline_token: str | None = None,
                                inline_token_expires: str | None = None) -> str:
    """Generate a Colab-flavoured `.ipynb` for the LB.

    Token resolution at run-time uses three tiers, in order:
      1. Inline short-lived token baked into the cell at generation
         time (Open-in-Colab flow). Expires after ~1 hour.
      2. `BENCHHUB_API_TOKEN` from Colab Secrets — long-lived, set
         once per Google account.
      3. Interactive `getpass` prompt — last-ditch when neither of
         the above is available (e.g. expired inline token + no Colab
         Secret yet).
    The /api/whoami probe validates the chosen token before the body
    cell runs; on 401 the cell drops to tier 2 / tier 3 as needed.
    """
    script = _submission_script_source(lb)
    install_cell = (
        "# Install the BenchHub client + numpy. Run once per fresh runtime.\n"
        "# --upgrade so an older cached copy doesn't shadow a bug fix.\n"
        "!pip install -q --upgrade benchhub-client numpy\n"
    )
    # Embed inline token as a Python string literal (json.dumps gives
    # us a safely-escaped one). Falls back to None.
    inline_lit = json.dumps(inline_token or '')
    inline_expires_lit = json.dumps(inline_token_expires or '')
    base_url = _get_base_url() or 'https://runbenchhub.com'
    base_url_lit = json.dumps(base_url)
    token_cell = (
        "# Three-tier token resolution. The Open-in-Colab flow bakes a\n"
        "# short-lived token into the next line; if that expires (or\n"
        "# wasn't set), we try a Colab Secret named BENCHHUB_API_TOKEN,\n"
        "# then prompt with getpass as a final fallback.\n"
        "import os, urllib.request, urllib.error, json\n"
        f"_INLINE_TOKEN: str = {inline_lit}\n"
        f"_INLINE_EXPIRES: str = {inline_expires_lit}\n"
        f"_BASE_URL: str = {base_url_lit}\n"
        "\n"
        "def _bh_validate_token(tok: str | None) -> bool:\n"
        "    if not tok: return False\n"
        "    req = urllib.request.Request(_BASE_URL + '/api/whoami',\n"
        "        headers={'Authorization': 'Bearer ' + tok})\n"
        "    try:\n"
        "        with urllib.request.urlopen(req, timeout=10) as r:\n"
        "            return r.status == 200\n"
        "    except urllib.error.HTTPError as e:\n"
        "        return False\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "_token: str | None = None\n"
        "if _bh_validate_token(_INLINE_TOKEN):\n"
        "    _token = _INLINE_TOKEN\n"
        "    print(f'\\u2713 Using the short-lived token baked into this notebook (expires {_INLINE_EXPIRES}).')\n"
        "else:\n"
        "    try:\n"
        "        from google.colab import userdata\n"
        "        _ts = userdata.get('BENCHHUB_API_TOKEN')\n"
        "        if _bh_validate_token(_ts):\n"
        "            _token = _ts\n"
        "            print('\\u2713 Using BENCHHUB_API_TOKEN from Colab Secrets.')\n"
        "    except Exception:\n"
        "        pass\n"
        "if _token is None:\n"
        "    import getpass\n"
        "    print('Inline token expired and no Colab Secret found.')\n"
        "    print('Get a token from ' + _BASE_URL + '/settings/account, paste below.')\n"
        "    _token = getpass.getpass('BenchHub API token: ').strip()\n"
        "    if not _bh_validate_token(_token):\n"
        "        raise RuntimeError('Token rejected by ' + _BASE_URL + '/api/whoami.')\n"
        "    print('\\u2713 Token accepted.')\n"
        "os.environ['BENCHHUB_API_TOKEN'] = _token\n"
    )
    body_cell = script

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# BenchHub submission — {lb.name}\n",
                    "\n",
                    f"Leaderboard #{lb.id}. Fill in `iter_samples()` + the per-pred-field constructors, then run all cells.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": install_cell.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": token_cell.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": body_cell.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return json.dumps(nb, indent=2)


def _get_base_url():
    """Best-effort base URL for the running app — used to bake an
    `BASE_URL = ...` into the generated submission scripts. Empty
    string when nothing usable is available (the script's
    `bh.Client()` falls back to the BenchHub-client default)."""
    try:
        return request.url_root.rstrip('/')
    except Exception:
        return ''


@app.route('/leaderboard/<int:leaderboard_id>/submission_script.py')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_submission_script(leaderboard_id):
    """Download a self-contained Python submission script for this LB."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    body = _submission_script_source(lb)
    safe = secure_filename(lb.name) or f'lb_{lb.id}'
    return Response(
        body,
        mimetype='text/x-python; charset=utf-8',
        headers={
            'Content-Disposition': f'attachment; filename="{safe}_submit.py"',
        },
    )


@app.route('/leaderboard/<int:leaderboard_id>/submission_notebook.ipynb')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_submission_notebook(leaderboard_id):
    """Download a Colab-flavoured `.ipynb` for this LB."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    body = _submission_notebook_source(lb)
    safe = secure_filename(lb.name) or f'lb_{lb.id}'
    return Response(
        body,
        mimetype='application/x-ipynb+json',
        headers={
            'Content-Disposition': f'attachment; filename="{safe}_submit.ipynb"',
        },
    )


def _gh_upsert_gist(token: str, gist_id: str | None,
                    description: str, filename: str, content: str) -> str | None:
    """Create or update a single-file public gist via the GitHub REST
    API. Returns `<owner_login>/<gist_id>` (the Colab gist-path
    format), or None on any failure.

    `gist_id` None    → POST (create new gist)
    `gist_id` set     → PATCH existing gist with new content
    """
    import urllib.request, urllib.error
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'benchhub/0.1',
    }
    payload = {
        'description': description,
        'files': {filename: {'content': content}},
    }
    if gist_id is None:
        url = 'https://api.github.com/gists'
        payload['public'] = True
        method = 'POST'
    else:
        url = f'https://api.github.com/gists/{gist_id}'
        method = 'PATCH'
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        method=method, headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            doc = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as e:
        print(f'  ! gist upsert failed: {e}')
        return None
    if not isinstance(doc, dict):
        return None
    new_id = doc.get('id')
    owner = (doc.get('owner') or {}).get('login') if isinstance(doc.get('owner'), dict) else None
    if not new_id or not owner:
        return None
    return f'{owner}/{new_id}'


@app.route('/leaderboard/<int:leaderboard_id>/open_in_colab')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_open_in_colab(leaderboard_id):
    """Push this LB's submission notebook to a GitHub gist (creating it
    on first click, updating it after that) and 302 the browser to
    `colab.research.google.com/gist/<id>`. Colab's URL-import only
    accepts allowlisted hosts (github / gist.github / drive), so the
    gist is the shim that makes 'open in Colab' actually open the
    pre-filled notebook.

    Falls back to opening Colab's homepage with a warning flash if the
    server has no GIST token configured."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    token = os.environ.get('BENCHHUB_GITHUB_GIST_TOKEN', '').strip()
    if not token:
        flash('Server has no GitHub gist token configured — download the '
              '.ipynb and upload manually via Colab → File → Upload Notebook.',
              'warning')
        return redirect('https://colab.research.google.com/')

    # Login-gate: anonymous users can't get a submission token (it's
    # bound to a User row). Send them to /login first; they'll bounce
    # back here after the OAuth dance.
    user = getattr(g, 'current_user', None)
    if user is None:
        return redirect(url_for(
            'login', next=url_for('leaderboard_open_in_colab',
                                  leaderboard_id=lb.id),
        ))

    # Mint a short-lived submission token (1h) scoped to this user +
    # this LB. The notebook reads it from a python literal in the
    # token cell and falls back to Colab Secrets / getpass on expiry.
    st = _mint_submission_token(user, lb, hours=1)
    body = _submission_notebook_source(
        lb, inline_token=st.token,
        inline_token_expires=st.expires_at.isoformat() + 'Z',
    )
    safe = secure_filename(lb.name) or f'lb_{lb.id}'
    filename = f'{safe}_submit.ipynb'
    desc = (f'BenchHub submission notebook for leaderboard '
            f'"{lb.name}" — auto-managed by BenchHub.')

    # lb.colab_gist_id stores "<owner>/<id>"; PATCH wants the bare id.
    prior_path = lb.colab_gist_id or ''
    prior_id = prior_path.split('/', 1)[1] if '/' in prior_path else (prior_path or None)

    new_path = _gh_upsert_gist(token, prior_id, desc, filename, body)
    # If a PATCH to an existing gist failed (e.g. it was deleted on
    # GitHub), try once more as a CREATE so the user still gets a
    # working notebook.
    if new_path is None and prior_id:
        new_path = _gh_upsert_gist(token, None, desc, filename, body)
    if new_path is None:
        flash('Could not push the notebook to GitHub. '
              'Download the .ipynb and upload manually.', 'danger')
        return redirect('https://colab.research.google.com/')

    if new_path != lb.colab_gist_id:
        lb.colab_gist_id = new_path
        db.session.commit()
    return redirect(f'https://colab.research.google.com/gist/{new_path}')


@app.route('/leaderboard/<int:leaderboard_id>')
@visibility_required(Leaderboard, 'leaderboard_id')
def leaderboard_view(leaderboard_id):
    leaderboard = Leaderboard.query.get_or_404(leaderboard_id)
    show_archived = request.args.get('show_archived', 'false').lower() == 'true'
    sort_metric = request.args.get('sort_metric', '')
    sort_order = request.args.get('sort_order', 'asc')
    # Did the viewer explicitly pick a sort? When not, we default to
    # best-first by the leaderboard's first metric (set once all_metrics
    # is known, below).
    _sort_metric_explicit = bool(sort_metric)
    _sort_order_explicit = 'sort_order' in request.args

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
            if cf.data_type in ['metric', 'scalar'] and not cf.name.startswith('lm_'):
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

    # Default sort: when the viewer hasn't picked a column, sort
    # best-first by the leaderboard's FIRST metric. Direction follows
    # that metric's own sort_direction (lower_is_better → ascending,
    # else descending) so the top row is the best submission.
    if not _sort_metric_explicit and all_metrics:
        sort_metric = all_metrics[0]
        if not _sort_order_explicit:
            _lm0 = leaderboard_metrics_map.get(sort_metric)
            _direction = getattr(_lm0, 'sort_direction', None) or 'higher_is_better'
            sort_order = 'asc' if _direction == 'lower_is_better' else 'desc'

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
                        CustomField.data_type.in_(['metric', 'scalar'])
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
    # Split the dataset-side schema into Input fields (handed to the
    # submitter at inference time) and GT fields (held server-side).
    # Both render as a stacked widget on the LB page so submitters see
    # the full data shape, not just the pred contract.
    input_field_schema = []
    gt_field_schema = []
    for att in (leaderboard.attachments or []):
        if not att.hf_repo_id:
            continue
        try:
            _mapping = json.loads(att.hf_mapping_json or '[]')
        except (TypeError, ValueError):
            _mapping = []
        for _m in _mapping:
            if _m.get('target_kind') == 'skip':
                continue
            row = {
                'column': _m.get('column'),
                'target_field': _m.get('target_field') or _m.get('column'),
                'kind': _m.get('target_kind'),
                'reason': _m.get('reason') or '',
            }
            if (_m.get('role') or 'gt') == 'input':
                input_field_schema.append(row)
            else:
                gt_field_schema.append(row)
    # Phase 15: split mirrored (PWC-style) submissions into a separate
    # render bucket. They share the Submission table but show in a
    # second "Reported scores" section so the trust gradient stays
    # visible — verified beats mirrored is a real signal.
    verified_submissions = [s for s in submissions
                            if (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored']
    mirrored_submissions = [s for s in submissions
                            if (getattr(s, 'kind', 'verified') or 'verified') == 'mirrored']
    # Per-LB constructor snippet lines for the inline Submit-predictions
    # snippet — keeps the bh.<Kind>(...) calls in sync with the LB's
    # actual contract (e.g. cifar10 with a top-K pred shows
    # `label_topk_pred=bh.LabelList(..., k=5)` instead of generic
    # `label_pred=bh.Label(...)`).
    submission_predict_lines = _pred_field_constructor_lines(
        leaderboard, indent='                ',
    )
    # Materialization status for the LB (if any). The materialization
    # row was created at LB-create time for preview-only datasets;
    # the page surfaces status + a retry button when failed.
    materialization = leaderboard.materialization
    return render_template('leaderboard.html',
                           leaderboard=leaderboard,
                           materialization=materialization,
                           dataset_thumbs=dataset_thumbs,
                           pred_field_schema=pred_field_schema,
                           submission_predict_lines=submission_predict_lines,
                           input_field_schema=input_field_schema,
                           gt_field_schema=gt_field_schema,
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

# ===================== HuggingFace BYO (Phase 4 — simple) =====================
# Constraint: the HF repo MUST already follow BenchHub's folder convention
# (metric_*/, hist_*/, raw_*/, image_*/ folders with one file per sample).
# We don't translate arbitrary HF Datasets schemas — that's a much bigger
# design problem that needs explicit user input on which datasets to support.
# This path is "snapshot a structured repo + reuse the existing ZIP pipeline."

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
                (cf.name, cf.data_type)
                for cf in CustomField.query.filter(
                    CustomField.sample_id.in_(sample_ids)
                ).all()
            })
    return f"{_zlib.crc32(json.dumps(parts, sort_keys=True, default=str).encode()):08x}"


_HF_SOTA_CACHE = {}


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
    field_types = {cf.name: cf.data_type for cf in (first.custom_fields or [])}
    for cf in (first.custom_fields or []):
        if cf.name.endswith('_class'):
            continue
        if cf.data_type == 'scalar':
            is_classlabel = f"{cf.name}_class" in field_types
            yield cf.name, 'scalar', {'is_classlabel': is_classlabel}
        elif cf.data_type == 'depth':
            yield cf.name, 'depth', {}
        elif cf.data_type == 'image':
            name_lc = cf.name.lower()
            if any(h in name_lc for h in _MASK_NAME_HINTS):
                yield cf.name, 'mask', {}
            else:
                yield cf.name, 'image', {}
        elif cf.data_type == 'text':
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


def _lb_field_roles(lb) -> dict:
    """Parse the LB's per-field role overrides. Returns a dict
    `{field_name: 'input'|'gt'|'skip'}` for any fields the LB has
    explicit roles for. Missing entries fall through to the
    `DatasetField.role` default in callers."""
    raw = (lb.field_roles_json or '').strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                k: v for k, v in parsed.items()
                if isinstance(k, str) and v in ('input', 'gt', 'skip')
            }
    except (TypeError, ValueError):
        pass
    return {}


def _effective_field_role(lb, dataset_field) -> str:
    """LB-side role override > dataset's declared role > 'gt' default.

    The dataset is now role-neutral; LBs decide how to use each field.
    A single Dataset can back multiple LBs with different role
    assignments (depth field used as gt on a depth-estimation LB and
    as input on a colorization LB)."""
    overrides = _lb_field_roles(lb)
    if dataset_field.name in overrides:
        return overrides[dataset_field.name]
    return (dataset_field.role or 'gt').lower()


def _lb_pred_contract_from_dataset_fields(lb):
    """Derive the LB's prediction wire-contract from the
    `DatasetField` rows of every attached dataset.

    Priority order:

      1. `Leaderboard.required_pred_fields_json` — set by an admin
         to fully override the contract (rename pred fields,
         restrict kinds, etc.). Used as-is when non-empty.
      2. Explicit `role='pred'` DatasetField rows. These are the
         contract. If the dataset(s) declare none, the LB has no
         pred contract — submissions are rejected at manifest-
         validation time with a clear "this dataset has no pred
         fields declared" error rather than the engine inventing
         `<gt>_pred` entries the dataset never promised.

    Returns a list of `{name, kind, params, role}` entries
    (`role='pred'`) in the shape `import_typed_submission` consumes.
    """
    # 1. Explicit LB-level override path.
    raw = lb.required_pred_fields_json or ''
    if raw:
        try:
            forced = json.loads(raw)
            if isinstance(forced, list) and forced:
                return [
                    {**e, 'role': e.get('role', 'pred')}
                    for e in forced if isinstance(e, dict)
                ]
        except (TypeError, ValueError):
            pass

    # 2. Explicit `role='pred'` on the dataset. Source of truth.
    # No GT-mirror fallback — preds belong to the dataset's
    # declared schema (the picker enforces the same rule).
    explicit: dict[str, dict] = {}
    for ds in (lb.datasets or []):
        for f in (ds.dataset_fields or []):
            if f.role != 'pred' or f.name in explicit:
                continue
            explicit[f.name] = {
                'name': f.name,
                'kind': f.kind,
                'params': f.get_params(),
                'role': 'pred',
            }
    return list(explicit.values())


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
            gt_field_meta.setdefault(cf.name, cf.data_type)
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

    # Filter out pred fields whose GT column is flagged as `role=input`
    # in the HF mapping — the submitter doesn't predict their own
    # inputs. (For class-conditional image generation: `label` is
    # input, so `label_pred` shouldn't be in the contract.)
    input_gt_fields = set()
    for att in (lb.attachments or []):
        if getattr(att, 'kind', None) != 'hf':
            continue
        try:
            _mapping = json.loads(att.hf_mapping_json or '[]')
        except (TypeError, ValueError):
            _mapping = []
        for _m in _mapping:
            if (_m.get('role') or 'gt') == 'input':
                tf = _m.get('target_field') or _m.get('column')
                if tf:
                    input_gt_fields.add(tf)
    if input_gt_fields:
        for _name in list(seen.keys()):
            if seen[_name].get('gt_field') in input_gt_fields:
                seen.pop(_name, None)

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


_HF_NATURAL_NAME_COLS = (
    'sample_id', 'sample_name', 'sample',
    'file_name', 'filename', 'file', 'image_id',
    'id', 'name', 'uid', 'key',
)


def _natural_sample_name(row, fallback):
    """Pick a human-friendly sample name from the row's natural-key
    columns when present, otherwise fall back to the synthetic
    `s_NNNNNN`. Values are stringified, slugified to `[A-Za-z0-9._-]`,
    and capped at 80 chars so the cache_key + URL stay sane. Returns
    `fallback` when no usable column is found."""
    if not isinstance(row, dict):
        return fallback
    for col in _HF_NATURAL_NAME_COLS:
        if col not in row:
            continue
        val = row[col]
        if val is None:
            continue
        try:
            s = str(val).strip()
        except Exception:
            continue
        if not s:
            continue
        # Strip filesystem-unfriendly characters but keep extension
        # info readable (`xyz.png` stays `xyz.png`).
        import re as _re
        slug = _re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('._-')
        if slug:
            return slug[:80]
    return fallback


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


def _iter_lb_eval_samples(lb):
    """Yield (sample, attachment) tuples for every sample the LB evaluates
    against. Walks Attachment rows for the primary dataset(s), plus any
    legacy m2m `lb.datasets` rows not yet upgraded to an Attachment.

    When the LB has a ready materialisation, the yield is RESTRICTED to
    the materialised sample subset — the same set `/api/leaderboard/<id>/
    samples` exposes to submitters (and that they actually predict).
    Without this filter, eval iterates the full source dataset (e.g.
    cifar10's 10k test rows) while the submission only covers the
    materialised subset (e.g. 1k); the unpredicted samples score 0 and
    tank the aggregate (954/1000 = 95.4% mis-reported as 954/10000 = 9.5%)."""
    mat_names = None
    mat = getattr(lb, 'materialization', None)
    if mat and mat.status == 'ready':
        from benchhub.lb_materialize import list_materialized_samples
        names = list_materialized_samples(app.config['UPLOAD_FOLDER'], lb.id)
        if names:
            mat_names = set(names)

    covered_dataset_ids = set()
    for att in lb.attachments:
        if att.role != 'primary' or att.dataset is None:
            continue
        covered_dataset_ids.add(att.dataset.id)
        for s in att.dataset.samples:
            if mat_names is None or s.name in mat_names:
                yield s, att

    for ds in (lb.datasets or []):
        if ds.id in covered_dataset_ids:
            continue
        for s in (ds.samples or []):
            if mat_names is None or s.name in mat_names:
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
                if cf.data_type in ('scalar', 'image', 'depth', 'text'):
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
        if cf.data_type == 'image':
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
        elif cf.data_type == 'depth':
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
        if cf.data_type == 'image':
            from PIL import Image as _PILImage
            return np.asarray(_PILImage.open(cached_path).convert('RGB'))
        if cf.data_type == 'depth':
            with np.load(cached_path) as data:
                if 'depth' in data:
                    return np.asarray(data['depth'])
                first = next(iter(data.keys()))
                return np.asarray(data[first])
    except Exception as e:
        print(f"DEBUG: pointer read-back failed for {cache_key}: {e}")
    return None


@app.route('/favicon.ico')
def favicon():
    """Browsers fetch /favicon.ico unconditionally regardless of the
    <link rel="icon"> in base.html; without an explicit route every
    page load shows a 404 in the console. Serves the static file
    directly so we don't bounce through 302→/static/favicon.ico."""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon',
    )


@app.route('/dataset/<int:dataset_id>/create_lb', methods=['GET'])
@login_required
@visibility_required(Dataset, 'dataset_id')
def create_lb_for_dataset(dataset_id):
    """Dedicated LB-creation page backed by a specific dataset.

    The form used to live inline on `/dataset/<id>` inside a
    collapse panel, but it grew enough (field roles + pred fields
    + per-metric picker with arg mappings) to deserve its own page.
    POST still goes to `/create_leaderboard` so the route handler
    is unchanged.
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    available_metrics_for_lb, gt_field_options, dataset_field_options = (
        _build_lb_creation_context(dataset)
    )
    # Stratified-sampling default kicks in when the dataset has a
    # label-kind field — classification benchmarks want balanced
    # subsets across classes.
    _has_label_field = any(f.kind == 'label' for f in dataset.dataset_fields)
    # LB templates: `?template=image_segmentation` etc. surface a
    # default metric + pred-field set. `applicable_templates` powers
    # the chip picker; `applied_template` carries the resolved values
    # when one is selected (else falls back to the manual flow).
    applicable_templates = _applicable_lb_templates(dataset)
    template_id = (request.args.get('template') or '').strip()
    applied_template = (
        _apply_lb_template(dataset, template_id) if template_id else {}
    )
    return render_template(
        'create_lb_for_dataset.html',
        dataset=dataset,
        suggested_lb_name=_suggest_free_lb_name(dataset.name),
        available_metrics_for_lb=available_metrics_for_lb,
        gt_field_options=gt_field_options,
        dataset_field_options=dataset_field_options,
        _has_label_field=_has_label_field,
        applicable_templates=applicable_templates,
        applied_template=applied_template,
    )


@app.route('/create_lb', methods=['GET'])
@login_required
def create_lb_chooser():
    """Pick a BenchHub dataset → configure name → POST to /create_leaderboard.

    Used to also have an HF-import half; that's gone with Phase A.
    The form supports attaching one or many datasets (Attachment table)
    and an optional auto-assign-metrics flow that prompts the user
    to confirm individual metric proposals before any LB row is
    actually written."""
    user = g.current_user
    bh_datasets = (
        Dataset.query
        .filter(visible_in_list(Dataset, user))
        .order_by(Dataset.upload_date.desc())
        .all()
    )
    # Suggest a free name based on the first-listed dataset so the
    # name input pre-fills with a usable default. The user can still
    # edit; this just removes the friction for the common case.
    suggested_lb_name = _suggest_free_lb_name(
        bh_datasets[0].name if bh_datasets else 'leaderboard'
    )
    return render_template(
        'create_lb_chooser.html',
        bh_datasets=bh_datasets,
        suggested_lb_name=suggested_lb_name,
    )


# --- HuggingFace dataset listing (Round C) -------------------------------
# Proxy the HF public datasets index so users can pick a repo from a
# searchable list inside BenchHub instead of context-switching to
# huggingface.co. Server-side cache keeps us under HF's anonymous rate
# limit (the result is the same for everyone for an hour).

_hf_datasets_cache = {'fetched_at': 0.0, 'sort': None, 'q': '', 'rows': []}


# `/hf/<repo_id>` live-preview surface was removed in the attachment
# refactor (2026-05-08). HF datasets no longer have a BH representation;
# users browse on huggingface.co directly and create LBs via the
# /datasets HF picker → /import_from_hf/preview → auto-LB flow.



@app.route('/create_leaderboard', methods=['POST'])
@login_required
def create_leaderboard():
    leaderboard_name = request.form['leaderboard_name']

    # Names are global now (no project namespace).
    existing = Leaderboard.query.filter_by(name=leaderboard_name).first()
    if existing:
        flash(
            f'Leaderboard "{leaderboard_name}" already exists. '
            'Choose a different name.', 'danger')
        return redirect(url_for('datasets_list'))

    # Per-field role overrides from the LB-creation form. The form
    # posts `field_role_<dataset_field_name>` = input/gt/skip per
    # field; we collect those into `field_roles_json` so the
    # picker/runtime knows how to treat each attached field for
    # THIS LB. Missing entries fall back to the dataset field's
    # declared role.
    _role_overrides: dict[str, str] = {}
    for key, val in request.form.items(multi=False):
        if not key.startswith('field_role_'):
            continue
        fname = key[len('field_role_'):]
        val = (val or '').strip().lower()
        if val in ('input', 'gt', 'skip') and fname:
            _role_overrides[fname] = val

    # Pred-field declarations from the LB form (mirrors the HF
    # preview's Prediction-fields section). Parallel arrays of
    # `pred_field_name` + `pred_field_kind` + `pred_field_params`.
    # Stored on `required_pred_fields_json` so the existing
    # contract-derivation helper picks them up.
    pred_names = request.form.getlist('pred_field_name')
    pred_kinds = request.form.getlist('pred_field_kind')
    pred_params = request.form.getlist('pred_field_params')
    pred_entries: list[dict] = []
    from benchhub.types import DTYPES as _DTYPES_PRED
    for i, pname in enumerate(pred_names):
        pname = (pname or '').strip()
        if not pname:
            continue
        pkind = (pred_kinds[i] if i < len(pred_kinds) else '').strip()
        if pkind not in _DTYPES_PRED:
            continue
        praw = (pred_params[i] if i < len(pred_params) else '').strip()
        try:
            pp = json.loads(praw) if praw else {}
            if not isinstance(pp, dict):
                pp = {}
        except (TypeError, ValueError):
            pp = {}
        pred_entries.append({
            'name': pname,
            'kind': pkind,
            'params': pp,
            'role': 'pred',
        })

    new_leaderboard = Leaderboard(
        name=leaderboard_name,
        # Legacy CSV; the structured per-metric rows below are the
        # authoritative source. Still populated so any old reader
        # that grovels the column sees a sensible string.
        summary_metrics=','.join(request.form.getlist('summary_metrics')),
        owner_user_id=g.current_user.id,
        category=(request.form.get('category') or '').strip() or None,
        field_roles_json=(json.dumps(_role_overrides) if _role_overrides else None),
        required_pred_fields_json=(json.dumps(pred_entries) if pred_entries else None),
        # User-created LBs land private; the owner publishes via the
        # "Publish" button on the LB page when they're ready to share.
        # Admin/import scripts that need public-by-default pass an
        # explicit `visibility` form field.
        visibility=(request.form.get('visibility') or 'private').strip(),
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

    # Structured per-metric rows from the dataset-view inline form:
    # parallel arrays of `metric_global_id` + `metric_mappings_json`
    # (JSON dict of {arg_name: gt_field_name}). Each surviving pair
    # becomes a LeaderboardMetric row bound to the user's picked
    # GlobalMetric with the chosen arg mappings.
    raw_metric_ids = request.form.getlist('metric_global_id')
    raw_metric_mappings = request.form.getlist('metric_mappings_json')
    # Resolve attached datasets' field schemas once for the type-
    # assertion step below. Only explicit DatasetField rows count
    # — pred bindings must reference a real role=pred field on the
    # dataset, not a synthesized `<gt>_pred` entry.
    _attached_by_name: dict[str, tuple[str, str]] = {}
    for _ds in (new_leaderboard.datasets or []):
        for _df in _ds.fields:
            _attached_by_name[_df.name] = (_df.kind, (_df.role or 'gt').lower())

    validation_errors: list[str] = []
    for i, gm_id_raw in enumerate(raw_metric_ids):
        gm_id_raw = (gm_id_raw or '').strip()
        if not gm_id_raw:
            continue
        try:
            gm_id = int(gm_id_raw)
        except ValueError:
            continue
        gm = GlobalMetric.query.get(gm_id)
        if gm is None:
            continue
        mapping_raw = raw_metric_mappings[i] if i < len(raw_metric_mappings) else ''
        try:
            mapping = json.loads(mapping_raw) if mapping_raw else {}
            if not isinstance(mapping, dict):
                mapping = {}
        except (TypeError, ValueError):
            mapping = {}

        # Assert input types/roles. A metric that declares
        # `input_kinds=['label','label']` + `input_roles=['gt','pred']`
        # must have its `gt` arg bound to a role=gt label field and
        # its `pred` arg bound to a role=gt label field (the engine
        # auto-derives the `<name>_pred` from the pred contract).
        args = _extract_metric_arg_names(gm.python_code or '')
        try:
            kinds = json.loads(gm.input_kinds) if gm.input_kinds else []
            if not isinstance(kinds, list):
                kinds = []
        except (TypeError, ValueError):
            kinds = []
        try:
            roles = json.loads(gm.input_roles) if gm.input_roles else []
            if not isinstance(roles, list):
                roles = []
        except (TypeError, ValueError):
            roles = []

        skip_metric = False
        for idx, arg in enumerate(args):
            picked = mapping.get(arg)
            if not picked:
                continue
            if picked not in _attached_by_name:
                validation_errors.append(
                    f"{gm.name}.{arg}: {picked!r} is not a field on the attached dataset(s)"
                )
                skip_metric = True
                continue
            picked_kind, picked_role = _attached_by_name[picked]
            expected_kind = kinds[idx] if idx < len(kinds) else None
            expected_role = roles[idx] if idx < len(roles) else None
            if expected_kind:
                # Unions: `label|label_list` accepts either member.
                members = [m for m in expected_kind.split('|') if m]
                if picked_kind not in members:
                    validation_errors.append(
                        f"{gm.name}.{arg}: expected kind {expected_kind!r} but "
                        f"{picked!r} is kind {picked_kind!r}"
                    )
                    skip_metric = True
            if expected_role and picked_role != expected_role:
                validation_errors.append(
                    f"{gm.name}.{arg}: expected role {expected_role!r} but "
                    f"{picked!r} has role {picked_role!r}"
                )
                skip_metric = True
        if skip_metric:
            continue

        lm = LeaderboardMetric(
            leaderboard_id=new_leaderboard.id,
            global_metric_id=gm.id,
            arg_mappings=json.dumps(mapping),
        )
        db.session.add(lm)
    if validation_errors:
        # Don't abort the LB creation — the LB is still useful with
        # whatever survived. Just flash the rejected pairs so the
        # admin can re-add them with a correct binding.
        flash(
            'Some metric bindings were rejected: ' + '; '.join(validation_errors),
            'warning',
        )

    # Inherit visualisations from each backing dataset. Each
    # DatasetVisualization row → a LeaderboardVisualization with the
    # same arg_mappings, target_name, and ordering. If two datasets
    # attach the same GlobalVisualization, only the first carries
    # over (idempotent + no duplicate cells in the LB grid).
    seen_viz: set[int] = set()
    for _ds in (new_leaderboard.datasets or []):
        for dv in (DatasetVisualization.query
                   .filter_by(dataset_id=_ds.id)
                   .order_by(DatasetVisualization.display_order,
                              DatasetVisualization.id).all()):
            if dv.global_visualization_id in seen_viz:
                continue
            seen_viz.add(dv.global_visualization_id)
            lv = LeaderboardVisualization(
                leaderboard_id=new_leaderboard.id,
                global_visualization_id=dv.global_visualization_id,
                arg_mappings=dv.arg_mappings,
                target_name=dv.target_name,
                display_order=dv.display_order,
            )
            db.session.add(lv)
    db.session.commit()

    # Stage C: persist a LeaderboardMaterialization row when any
    # backing dataset is preview-only. The wizard fields on
    # /create_lb_for_dataset post sample_cap + sampling + seed +
    # stratify_field. Materialisation runs synchronously after
    # commit so the LB has full bytes ready by the time the user
    # lands on the LB page.
    any_preview = any(_ds.preview_only for _ds in (new_leaderboard.datasets or []))
    if any_preview:
        try:
            sample_cap = int(request.form.get('sample_cap') or 1000)
        except (TypeError, ValueError):
            sample_cap = 1000
        sampling = (request.form.get('sampling') or 'random').strip()
        if sampling not in ('head', 'random', 'stratified'):
            sampling = 'random'
        try:
            sampling_seed = int(request.form.get('sampling_seed') or 42)
        except (TypeError, ValueError):
            sampling_seed = 42
        stratify_field = (request.form.get('stratify_field') or '').strip() or None

        matrow = LeaderboardMaterialization(
            leaderboard_id=new_leaderboard.id,
            sample_cap=sample_cap,
            sampling=sampling,
            sampling_seed=sampling_seed,
            stratify_field=stratify_field,
            status='pending',
        )
        db.session.add(matrow)
        db.session.commit()
        # Kick off the materialisation as a Celery task so the
        # request handler doesn't block on a long HF download. The
        # LB page polls + surfaces the running/ready/failed state.
        # Sync fallback (CELERY_TASK_ALWAYS_EAGER, used in tests) is
        # automatic via Celery's eager mode flag.
        try:
            import tasks as _tasks
            _tasks.materialize_leaderboard.delay(new_leaderboard.id)
            flash(
                f'Materialisation queued ({sample_cap} samples, '
                f'{sampling} sampling). Track progress on the LB page.',
                'info',
            )
        except Exception as e:
            matrow.status = 'failed'
            matrow.error_message = f'enqueue: {e}'
            db.session.commit()
            flash(f'Couldn\'t queue materialisation: {e}', 'warning')

    flash(f'Leaderboard "{leaderboard_name}" created successfully!', 'success')
    return redirect(url_for('leaderboard_view', leaderboard_id=new_leaderboard.id))


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
            category=request.form.get('category'),
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
    # Parse properly: ?samples_only=0/false/no/'' is FALSE. Plain
    # bool(request.args.get(...)) treated the string '0' as truthy.
    samples_only_mode = (request.args.get('samples_only', '') or '').strip().lower() in ('1', 'true', 'yes', 'on')
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

    # Restrict to the materialised subset when the LB has one — those are
    # the only samples submitters can predict (iter_samples streams the
    # subset), so showing the other dataset rows just yields empty
    # prediction columns (e.g. s000002 on cifar, which isn't materialised).
    _mat = getattr(leaderboard, 'materialization', None)
    if _mat and getattr(_mat, 'status', None) == 'ready':
        from benchhub.lb_materialize import list_materialized_samples
        _mat_names = list_materialized_samples(app.config['UPLOAD_FOLDER'], leaderboard.id)
        if _mat_names:
            samples_query = samples_query.filter(Sample.name.in_(_mat_names))

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
            CustomField.name, CustomField.data_type,
        ).filter(
            CustomField.leaderboard_id == leaderboard.id,
            CustomField.submission_id.is_(None),
            CustomField.sample_id.is_(None),
        ).distinct().all()
    else:
        dataset_custom_fields_query = db.session.query(CustomField.name, CustomField.data_type).join(Sample).filter(
            Sample.dataset_id.in_(dataset_ids),
            CustomField.submission_id == None
        ).distinct().all()

    # Kinds that show up as renderable columns on the dataset detail
    # page. Built from `benchhub.types.DTYPES` so new types (label,
    # bboxes, …) appear automatically — the hand-rolled allow-list
    # used to silently drop label columns, which is why CIFAR-10's
    # `label` GT was invisible after import.
    from benchhub.types import DTYPES as _DTYPES
    _RENDERABLE_KINDS = set(_DTYPES) | {'metric'}  # 'metric' is a legacy precomputed-scalar marker
    dataset_custom_fields = {name for name, ftype in dataset_custom_fields_query if ftype in _RENDERABLE_KINDS}
    dataset_field_types = {name: ftype for name, ftype in dataset_custom_fields_query}
    
    # Submission fields
    sub_ids = [s.id for s in submissions]
    submission_custom_fields_query = db.session.query(CustomField.submission_id, CustomField.name, CustomField.data_type).filter(
        CustomField.submission_id.in_(sub_ids)
    ).distinct().all()
    
    submission_custom_fields = {}
    submission_field_types = {}
    for sub_id, name, ftype in submission_custom_fields_query:
        if ftype in ['image', 'depth', 'scalar', 'metric', 'label', 'label_list']:
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
    # Friendly display label only — never expose the internal lm_<id> key
    # to users. Disambiguate same-named metrics with a numeric suffix.
    _label_seen = {}
    for lmid, lm in leaderboard_metrics_map.items():
        base = lm.target_name if lm.target_name else lm.global_metric.name
        _label_seen[base] = _label_seen.get(base, 0) + 1
        metric_labels[lmid] = base if _label_seen[base] == 1 else f"{base} #{_label_seen[base]}"
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
            __slots__ = ('id', 'name', 'display_name', 'tags',
                         'custom_fields', 'signal_shape',
                         'histogram_data', 'config_data', 'dataset')

            def __init__(self, name, idx):
                self.id = idx
                self.name = name
                # Filled in below from the persisted __display_name__
                # marker CF when one exists for this sample_name.
                self.display_name = None
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
            display_by_name = {}
            for cf in gt_rows:
                if cf.name == '__display_name__':
                    display_by_name[cf.sample_name] = cf.value_text
                    continue  # don't surface as a real field
                gt_by_name.setdefault(cf.sample_name, []).append(cf)
            for stub in paginated_items:
                stub.custom_fields = gt_by_name.get(stub.name, [])
                stub.display_name = display_by_name.get(stub.name)

    # Sorting (BH path only — HF branch above already paginated)
    if not hf_stub_mode and sort_by:
        if sort_by == 'name':
            # Natural sort by (length, name) so 1 < 2 < 10 < 100 and
            # s_1 < s_2 < s_10 instead of lex's 1,10,100,2. Lex
            # within equal-length groups handles zero-padded sample
            # names (s_001, s_002) naturally too.
            if sort_order == 'desc':
                samples_query = samples_query.order_by(
                    func.length(Sample.name).desc(), Sample.name.desc())
            else:
                samples_query = samples_query.order_by(
                    func.length(Sample.name).asc(), Sample.name.asc())
            total = samples_query.count()
            paginated_items = samples_query.offset((page-1)*per_page).limit(per_page).all()
        elif sort_by.startswith('disagreement:'):
            # Sort samples by how much the submissions DISAGREE on a
            # per-sample metric: spread = max(value) - min(value) across
            # all compared submissions for that sample. Highest spread =
            # most contentious sample. Samples where <2 submissions have a
            # value get spread 0 (no disagreement to measure).
            metric_key = sort_by.split(':', 1)[1]
            sub_ids = [s.id for s in submissions]
            by_sample = {}
            if sub_ids:
                rows = db.session.query(
                    CustomField.sample_name, CustomField.value_float,
                ).filter(
                    CustomField.submission_id.in_(sub_ids),
                    CustomField.name == metric_key,
                    CustomField.data_type.in_(['scalar', 'metric']),
                    CustomField.value_float.isnot(None),
                ).all()
                for s_name, val in rows:
                    by_sample.setdefault(s_name, []).append(val)

            def _spread(s):
                vals = by_sample.get(s.name, [])
                return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

            all_filtered_samples = samples_query.all()
            # Respect the order dropdown (High→Low = most disagreement first).
            all_filtered_samples.sort(key=_spread, reverse=(sort_order == 'desc'))
            total = len(all_filtered_samples)
            paginated_items = all_filtered_samples[(page-1)*per_page : page*per_page]
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
                    CustomField.data_type.in_(['scalar', 'metric'])
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
            if cf.data_type in ['scalar', 'metric'] and cf.submission_id is None:
                sample_metrics_map[s.id][cf.name] = cf.value_float

    for sample in samples_on_page:
        signal_shape = None.shape_name if None else 'gaussian'
        gt_bins = json.loads(None.bins) if None else []
        gt_counts = json.loads(None.counts) if None else []
        config_json = None.parsed_config if None else {}

        sample_info = {
            'sample_id': sample.id,
            'sample_name': sample.name,
            'display_name': getattr(sample, 'display_name', None) or sample.name,
            'dataset_tags': [t.strip() for t in sample.tags.split(',')] if sample.tags else [],
            'ground_truth': {
                'config': None.parsed_config if None else {},
                'bins': gt_bins,
                'counts': gt_counts,
                # GT stats panel: scalars + metrics (value_float) plus
                # labels / top-K lists (decoded from value_text), so the
                # ground-truth class shows alongside scalar GT.
                'custom_fields': _gt_stats_custom_fields(sample),
            },
            'predictions': [],
            'custom_fields': {},
            'custom_metrics': {}
        }
        
        # Add GT custom fields for this sample. Non-image kinds (text,
        # json, scalar) get their value surfaced alongside the field_id;
        # the template branches on whichever attribute is non-null.
        for cf in sample.custom_fields:
            if cf.data_type in ['image', 'depth', 'mask', 'audio', 'scalar', 'text', 'json']:
                sample_info['custom_fields'][cf.name] = {
                    'gt_field_id': cf.id if cf.data_type in ['image', 'depth', 'mask', 'audio'] else None,
                    'gt_scalar_value': cf.value_float if cf.data_type == 'scalar' else None,
                    'gt_text_value': cf.value_text if cf.data_type in ('text', 'json') else None,
                    'gt_field_type': cf.data_type,
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
                if cf.data_type in ['image', 'depth', 'scalar'] and cf.sample_name == sample.name:
                    if cf.name not in sample_info['custom_fields']:
                        sample_info['custom_fields'][cf.name] = {
                            'gt_field_id': None,
                            'gt_scalar_value': None,
                            'submissions': {},
                            'sub_scalars': {}
                        }
                    if cf.data_type in ['image', 'depth']:
                        sample_info['custom_fields'][cf.name]['submissions'][sub.id] = cf.id
                    elif cf.data_type == 'scalar':
                        sample_info['custom_fields'][cf.name]['sub_scalars'][sub.id] = cf.value_float
                
                # Add submission custom metric fields for this sample
                if cf.data_type == 'metric' and cf.sample_name == sample.name:
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
            if cf.data_type == 'metric':
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

    # Always include every per-sample metric (no manual "Add Metric"
    # selection in the comparison view). The chart + sort options draw
    # from this full set rather than leaderboard.summary_metrics.
    _per_sample_lm_ids = [
        f"lm_{lm.id}" for lm in leaderboard.leaderboard_metrics
        if not lm.global_metric.is_aggregated
    ]
    all_selected_metrics = _per_sample_lm_ids + sorted(list(custom_metrics))
    
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
                CustomField.data_type == 'metric'
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
    
    # Aggregated metrics only appear in the summary chart/table, never
    # in the per-sample comparison detail.
    agg_lm_ids = {f"lm_{lm.id}" for lm in leaderboard.leaderboard_metrics if lm.global_metric.is_aggregated}
    agg_metric_names = {lm.global_metric.name for lm in leaderboard.leaderboard_metrics if lm.global_metric.is_aggregated}

    # The comparison view ALWAYS shows every per-sample metric — there is
    # no manual selection. active_metrics = all non-aggregated LB metrics
    # (lm_<id>) + any custom per-sample metric fields discovered on the
    # submissions. (Ignores leaderboard.selected_metrics entirely.)
    active_metrics = [
        f"lm_{lm.id}" for lm in leaderboard.leaderboard_metrics
        if not lm.global_metric.is_aggregated
    ] + sorted(m for m in custom_metrics if m not in agg_metric_names)
    
 
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
        data_type = all_field_types.get(field_name, 'image')
        if field_name in dataset_custom_fields:
            dataset_fields_dict[field_name] = data_type
        else:
            submission_fields_dict[field_name] = data_type

    # Which field names actually carry a value for some sample on THIS
    # page? Used to drop empty columns (a field declared on the LB but
    # with nothing to show for the current page's samples). Built from
    # the already-assembled comparison_data: GT visual fields land in
    # `custom_fields`; pred values land in each prediction dict.
    fields_with_data = set()
    for sd in comparison_data:
        for k in (sd.get('custom_fields') or {}):
            fields_with_data.add(k)
        for k, v in ((sd.get('ground_truth') or {}).get('custom_fields') or {}).items():
            if v is not None:
                fields_with_data.add(k)
        for pred in (sd.get('predictions') or []):
            if not pred:
                continue
            for k, v in pred.items():
                if v is not None:
                    fields_with_data.add(k)

    # Kinds that DON'T get their own column: `metric` (→ chart panel) and
    # `label` / `label_list` (→ per_source_stats, alongside scalars). A
    # column is also skipped when no sample on this page has a value for
    # it (don't show empty columns).
    _NON_COLUMN_KINDS = ('metric', 'label', 'label_list')

    # Add dataset fields first. Scalar GT (e.g. WN18RR's `head`/`tail`
    # entity IDs) IS the ground truth for many LBs and stays a column.
    for field_name in sorted(dataset_fields_dict.keys()):
        data_type = dataset_fields_dict[field_name]
        if data_type in _NON_COLUMN_KINDS: continue
        if field_name not in fields_with_data: continue
        # Narrower column for scalars since they render as a single
        # numeric/text value, not a 300px-wide image preview.
        default_width = '120px' if data_type == 'scalar' else '300px'
        available_display_options[field_name] = {
            'label': field_name, 'type': data_type, 'default_width': default_width,
        }
        if data_type in ['image', 'mask', 'depth']: custom_image_fields.append(field_name)

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
            data_type = all_field_types.get(field_name, 'image')
            if data_type in _NON_COLUMN_KINDS: continue
            if field_name not in fields_with_data: continue
            default_width = '120px' if data_type == 'scalar' else '300px'
            available_display_options[field_name] = {
                'label': field_name, 'type': data_type, 'default_width': default_width,
            }
            if data_type in ['image', 'mask', 'depth']: custom_image_fields.append(field_name)

    # 1. Check GT fields availability (only for samples on current page for UI rendering hint)
    has_gt_hist = any(None for s in samples_on_page)
    has_gt_config = any(None for s in samples_on_page)
    has_dataset_tags = any(s.tags for s in samples_on_page)
    
    if not has_gt_hist: available_display_options.pop('gt_histogram', None)
    if not has_gt_config: available_display_options.pop('gt_config', None)
    if not has_dataset_tags: available_display_options.pop('dataset_tags', None)

    # GT-side / Submission-side scalar+metric presence drives whether
    # the two per_source_stats columns render in the template. If both
    # sides are empty the whole option goes away from View Options too.
    # The per_source_stats panel now also surfaces label / label_list
    # (GT class + predicted class), so labels alone are enough to show it.
    _STATS_KINDS = ('scalar', 'metric', 'label', 'label_list')
    has_gt_scalars_or_metrics = any(
        all_field_types.get(fn) in _STATS_KINDS
        for fn in dataset_custom_fields
    )
    has_pred_scalars_or_metrics = any(
        all_field_types.get(fn) in _STATS_KINDS
        for sub_id, fns in submission_custom_fields.items()
        for fn in fns
        if fn not in dataset_custom_fields
    )
    if not (has_gt_scalars_or_metrics or has_pred_scalars_or_metrics):
        available_display_options.pop('per_source_stats', None)

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
    # Mirrored-only LBs have no per-sample predictions on disk, so the
    # same submission-needing panels we hide in samples_only_mode are
    # equally useless here. Apply the same filter.
    all_mirrored_local = bool(submissions) and all(
        (getattr(s, 'kind', 'verified') or 'verified') == 'mirrored'
        for s in submissions
    )
    if samples_only_mode or all_mirrored_local:
        _hide_in_samples_only = {'per_sample_metrics', 'per_source_stats',
                                 'pred_histogram'}
        selected_comparison_display_columns = [
            k for k in selected_comparison_display_columns
            if k not in _hide_in_samples_only
            and not k.startswith('viz_')
        ]

    # Pass metric directions for coloring
    metric_directions = json.loads(leaderboard.metric_directions) if leaderboard.metric_directions else {}
    # Merge each bound metric's own sort_direction (keyed by lm_<id>) so
    # the client knows the "good" direction for every metric — used to
    # orient the default sample-sort (worst-first) and disagreement sort.
    for lm in (leaderboard.leaderboard_metrics or []):
        # Default unset metrics to higher_is_better so the client always
        # has a direction to orient the worst-first sample sort.
        metric_directions[f"lm_{lm.id}"] = lm.sort_direction or 'higher_is_better'

    # True iff every submission being rendered is a mirrored PWC row
    # (no per-sample predictions on disk). The template uses this to
    # skip the busy submission legend and the per-sample chart column,
    # both of which are meaningless without per-sample data.
    all_mirrored = bool(submissions) and all(
        (getattr(s, 'kind', 'verified') or 'verified') == 'mirrored'
        for s in submissions
    )

    # Class-name vocab per label field (GT + pred), so the stats panel
    # can render `cat` instead of the raw class index `3`. Pulled from
    # the dataset fields' params and the LB's declared pred fields.
    label_vocabs = {}
    for _ds in (leaderboard.datasets or []):
        for _df in getattr(_ds, 'fields', []) or []:
            if _df.kind in ('label', 'label_list'):
                _names = (_df.get_params() or {}).get('names')
                if _names:
                    label_vocabs[_df.name] = _names
    try:
        for _entry in (json.loads(leaderboard.required_pred_fields_json or '[]') or []):
            if isinstance(_entry, dict) and _entry.get('kind') in ('label', 'label_list'):
                _names = (_entry.get('params') or {}).get('names')
                if _names:
                    label_vocabs[_entry['name']] = _names
    except (TypeError, ValueError):
        pass

    # Human-readable description of the active sort (never expose the
    # internal lm_<id> keys). Used by the "(sorted by …)" caption.
    def _sort_by_label():
        if not sort_by:
            return ''
        if sort_by == 'name':
            return 'Name'
        if sort_by.startswith('disagreement:'):
            k = sort_by.split(':', 1)[1]
            return 'Disagreement: ' + str(metric_labels.get(k, k))
        if ':' in sort_by:
            mk, _, sid = sort_by.partition(':')
            label = metric_labels.get(mk, mk)
            idx = next((i + 1 for i, s in enumerate(submissions) if str(s.id) == sid), None)
            return f"{label} (Sub{idx})" if idx else str(label)
        return str(metric_labels.get(sort_by, sort_by))
    sort_by_label = _sort_by_label()

    return render_template('comparison.html',
                           leaderboard=leaderboard,
                           sort_by_label=sort_by_label,
                           submissions=submissions,
                           all_mirrored=all_mirrored,
                           comparison_data=comparison_data,
                           label_vocabs=label_vocabs,
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
                           has_gt_scalars_or_metrics=has_gt_scalars_or_metrics,
                           has_pred_scalars_or_metrics=has_pred_scalars_or_metrics,
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
    """Serve a depth-map .npz as a pure colormapped PNG — no
    matplotlib axes, no baked-in colorbar. The frontend draws its
    own CSS-gradient colorbar as a sibling so the image's pixel
    grid stays exactly aligned with overlay layers stacked on top.

    ``?cmap=`` selects the palette (turbo / jet / viridis / magma /
    inferno / plasma / gray); default is turbo to match the
    column-header dropdown's pre-selected option.
    """
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)

    if not os.path.exists(full_path):
        return abort(404, description="File not found")

    # Preview-only datasets store depth as a pre-colormapped JPG (turbo
    # baked in at import time). Serve the bytes verbatim — the per-
    # request `?cmap=` switcher only works for raw .npz; for preview
    # the column header dropdown is a no-op.
    ext = full_path.rsplit('.', 1)[-1].lower()
    if ext in ('jpg', 'jpeg', 'png'):
        return send_file(full_path,
                         mimetype=f'image/{"jpeg" if ext != "png" else "png"}',
                         max_age=60)

    try:
        with np.load(full_path) as data:
            arr = None
            for key in data.files:
                if len(data[key].shape) == 2:
                    arr = data[key]
                    break
            if arr is None and data.files:
                arr = data[data.files[0]]
            if arr is None:
                return abort(500, description="Empty npz file")

        # Normalise to 0..255 over the array's own min/max range.
        # Constant arrays land at 0 (avoids divide-by-zero).
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            gray = ((arr - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            gray = np.zeros_like(arr, dtype=np.uint8)

        cmap = (request.args.get('cmap') or 'turbo').strip().lower()
        if cmap == 'gray':
            rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            lut = _matplotlib_lut(cmap)
            rgb = lut[gray]

        from PIL import Image as _PILImage
        out = _PILImage.fromarray(rgb, mode='RGB')
        buf = io.BytesIO()
        out.save(buf, format='PNG', optimize=False)
        buf.seek(0)
        return send_file(buf, mimetype='image/png', max_age=60)

    except Exception as e:
        print(f"Error serving depth image: {e}")
        return abort(500, description=str(e))

@app.route('/api/depth_data/<path:filepath>')
def serve_depth_data(filepath):
    """
    Serve raw depth map data as JSON for interactive visualization.
    filepath should be relative to UPLOAD_FOLDER.

    Preview-only datasets keep depth as a turbo-colormapped JPG; this
    endpoint reverse-turbos it back to a 2D float array in [0, 1] and
    scales by the sidecar .meta.json's (min, max) when available so
    plotly hover still shows metric values.
    """
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)

    if not os.path.exists(full_path):
        return abort(404, description="File not found")

    ext = full_path.rsplit('.', 1)[-1].lower()
    if ext in ('jpg', 'jpeg', 'png'):
        try:
            from PIL import Image as _PIL
            from benchhub.preview import reverse_turbo
            with _PIL.open(full_path) as im:
                rgb = np.asarray(im.convert('RGB'))
            normed = reverse_turbo(rgb)  # (H, W) float32 in [0, 1]
            meta_path = full_path.rsplit('.', 1)[0] + '.meta.json'
            lo, hi, normalized = 0.0, 1.0, True
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        m = json.load(f)
                    lo = float(m.get('min', 0.0))
                    hi = float(m.get('max', 1.0))
                    normalized = False
                except Exception:
                    pass
            arr = (normed * (hi - lo) + lo).astype(float)
            return jsonify({
                'data': arr.tolist(),
                'min': lo,
                'max': hi,
                'shape': list(arr.shape),
                'normalized': normalized,
            })
        except Exception as e:
            print(f"Error serving preview-depth data: {e}")
            return abort(500, description=str(e))

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

        return jsonify({
            'data': arr.tolist(),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'shape': arr.shape,
            'normalized': False,
        })

    except Exception as e:
        print(f"Error serving depth data: {e}")
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

    # HF-attached datasets. These don't live as Dataset rows — they're
    # referenced by Attachment rows pointing at huggingface.co. Group by
    # (hf_repo_id, hf_revision, hf_split) so the same repo at different
    # pins lists once each. Only surface entries whose owning LB has
    # cached GT (_compute_explorable_lb_ids); otherwise the row would
    # link to an empty Explore-samples view.
    viewer = getattr(g, 'current_user', None)
    visible_lb_subq = (
        db.session.query(Leaderboard.id)
        .filter(visible_in_list(Leaderboard, viewer))
        .subquery()
    )
    # NB: `Attachment.kind` is a Python @property, not a DB column —
    # filter on hf_repo_id (the column that defines an HF attachment).
    hf_att_rows = (
        db.session.query(Attachment, Leaderboard)
        .join(Leaderboard, Leaderboard.id == Attachment.leaderboard_id)
        .filter(Attachment.hf_repo_id.isnot(None))
        .filter(Leaderboard.id.in_(db.session.query(visible_lb_subq)))
        .all()
    )
    explorable_lb_ids = _compute_explorable_lb_ids([lb.id for _att, lb in hf_att_rows])
    hf_groups = {}
    for att, lb in hf_att_rows:
        if lb.id not in explorable_lb_ids:
            continue
        key = (att.hf_repo_id, att.hf_revision or '', att.hf_split or '')
        bucket = hf_groups.setdefault(key, {
            'hf_repo_id': att.hf_repo_id,
            'hf_revision': att.hf_revision,
            'hf_split': att.hf_split,
            'leaderboards': [],
        })
        bucket['leaderboards'].append(lb)
    # Per-group cached-sample count (LB-scoped CustomField marker rows).
    lb_id_to_cf_count = dict(
        db.session.query(
            CustomField.leaderboard_id,
            func.count(CustomField.id),
        )
        .filter(
            CustomField.leaderboard_id.in_(explorable_lb_ids or [0]),
            CustomField.submission_id.is_(None),
            CustomField.sample_id.is_(None),
        )
        .group_by(CustomField.leaderboard_id)
        .all()
    )
    hf_datasets = []
    for bucket in hf_groups.values():
        bucket['sample_count'] = max(
            (lb_id_to_cf_count.get(lb.id, 0) for lb in bucket['leaderboards']),
            default=0,
        )
        # First LB's category wins for grouping. Repos used by multiple
        # LBs in different categories are rare; if it happens, surface
        # the alphabetically-first non-null category for stability.
        cats = sorted({lb.category for lb in bucket['leaderboards'] if lb.category})
        bucket['category'] = cats[0] if cats else None
        bucket['area'] = (bucket['category'] or 'Uncategorized').split('/', 1)[0]
        bucket['task'] = (
            (bucket['category'].split('/', 1)[1] if bucket['category'] and '/' in bucket['category'] else None)
        )
        hf_datasets.append(bucket)

    # Unified entries: BH-uploaded Datasets + cached HF entries grouped
    # under one Area/Task hierarchy. Each entry carries a `kind`
    # discriminator so the template can branch on the small handful of
    # kind-specific fields (visibility badge, owner avatar, HF split,
    # etc.) while sharing the Area/Task header rendering.
    entries = []
    for ds in datasets:
        cats = ds.category_list  # multi-cat; primary cat = cats[0]
        primary = cats[0] if cats else None
        area = primary.split('/', 1)[0] if primary else 'Uncategorized'
        task = primary.split('/', 1)[1] if (primary and '/' in primary) else None
        entries.append({
            'kind': 'bh',
            'category': primary,
            'categories': cats,         # full list for filter/match
            'area': area,
            'task': task,
            'bh': ds,
            'thumb_url': dataset_thumbs.get(ds.id),
            'sample_count': len(ds.samples),
            'name': ds.name,
        })
    for bucket in hf_datasets:
        entries.append({
            'kind': 'hf',
            'category': bucket['category'],
            'area': bucket['area'],
            'task': bucket['task'],
            'hf': bucket,
            'thumb_url': None,
            'sample_count': bucket['sample_count'],
            'name': bucket['hf_repo_id'],
        })
    # Uncategorized sinks to the bottom; within a category, BH first
    # then HF (alphabetical on `kind`), tie-broken by name.
    entries.sort(key=lambda e: (
        e['area'] == 'Uncategorized',
        e['area'].lower(),
        (e['task'] or '').lower(),
        e['kind'],
        e['name'].lower(),
    ))

    # Category tree built from the combined entry list. For multi-
    # category BH datasets, the dataset counts under EACH of its
    # declared categories (not just the primary) so the area / task
    # counts on the sidebar match what users will see when they
    # click a filter.
    category_tree_dict = {}
    for e in entries:
        for cat in (e.get('categories') or [e['category']] if e.get('category') else ['']):
            if not cat:
                area = 'Uncategorized'; task = ''
            else:
                area = cat.split('/', 1)[0]
                task = cat.split('/', 1)[1] if '/' in cat else ''
            node = category_tree_dict.setdefault(area, {'count': 0, 'tasks': {}})
            node['count'] += 1
            if task:
                node['tasks'][task] = node['tasks'].get(task, 0) + 1
    category_tree = [
        {
            'area': area,
            'count': v['count'],
            'tasks': sorted(
                [{'name': t, 'count': c} for t, c in v['tasks'].items()],
                key=lambda x: (-x['count'], x['name']),
            ),
        }
        for area, v in sorted(category_tree_dict.items())
    ]

    active_category = (request.args.get('category') or '').strip()
    if active_category:
        if '/' in active_category:
            # Full Area/Task filter — match against any of the dataset's
            # categories (so a multi-cat dataset shows up under every
            # category it declares).
            entries = [
                e for e in entries
                if active_category in (e.get('categories') or [e['category']])
            ]
        else:
            # Area-level filter — match against the area of any cat.
            def _has_area(e):
                cats = e.get('categories') or ([e['category']] if e.get('category') else [])
                return any(c and c.split('/', 1)[0] == active_category for c in cats)
            entries = [e for e in entries if _has_area(e)]

    # Datalist suggestions for the upload form. Categories already in
    # use anywhere on the site (BH dataset OR LB), distinct.
    known_categories = sorted({
        c for (c,) in db.session.execute(db.text(
            "SELECT DISTINCT category FROM dataset WHERE category IS NOT NULL "
            "UNION SELECT DISTINCT category FROM leaderboard WHERE category IS NOT NULL"
        )).all() if c
    })

    # Storage gauge for the upload card. NULL user (anon) and 0-cap
    # users render the gauge in a "logged-out" state. Total cap stays
    # the sum of the two split caps so the legacy gauge keeps showing a
    # meaningful "X / Y" number; the per-bucket breakdown lives on
    # /settings/storage.
    _viewer = g.current_user
    storage_used = storage_used_bytes(_viewer) if _viewer else 0
    storage_cap_public = (
        int(quota_cap_for(_viewer, 'public')) if _viewer else 0
    )
    storage_cap_private = (
        int(quota_cap_for(_viewer, 'private')) if _viewer else 0
    )
    storage_cap = storage_cap_public + storage_cap_private

    # Community dataset-import requests (voting list). Sorted by
    # vote count desc so the most-wanted bubbles up. Pending-only by
    # default; admins can flip statuses via the dedicated routes.
    vote_count_subq = (
        db.session.query(
            DatasetRequestVote.request_id,
            db.func.count(DatasetRequestVote.id).label('n')
        ).group_by(DatasetRequestVote.request_id).subquery()
    )
    pending_reqs = (
        db.session.query(DatasetRequest,
                         db.func.coalesce(vote_count_subq.c.n, 0).label('vote_count'))
        .outerjoin(vote_count_subq, vote_count_subq.c.request_id == DatasetRequest.id)
        .filter(DatasetRequest.status == 'pending')
        .order_by(db.desc('vote_count'), DatasetRequest.created_at.desc())
        .limit(100).all()
    )
    me = getattr(g, 'current_user', None)
    voted_ids = set()
    if me:
        voted_ids = {
            r.request_id for r in
            DatasetRequestVote.query.filter_by(user_id=me.id).all()
        }
    dataset_requests = [
        {'req': r, 'vote_count': int(n), 'i_voted': r.id in voted_ids}
        for r, n in pending_reqs
    ]
    return render_template('datasets.html',
                           datasets=datasets,
                           dataset_thumbs=dataset_thumbs,
                           entries=entries,
                           category_tree=category_tree,
                           active_category=active_category,
                           known_categories=known_categories,
                           storage_used=storage_used,
                           storage_cap=storage_cap,
                           storage_used_public=(
                               storage_used_bytes(_viewer, visibility='public')
                               if _viewer else 0),
                           storage_used_private=(
                               storage_used_bytes(_viewer, visibility='private')
                               if _viewer else 0),
                           storage_cap_public=storage_cap_public,
                           storage_cap_private=storage_cap_private,
                           format_bytes=_format_bytes,
                           dataset_requests=dataset_requests)


# -----------------------------------------------------------------------
# Dataset-request voting routes
# -----------------------------------------------------------------------

_HF_REPO_RE = re.compile(r'^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$')


def _parse_hf_repo_id(raw: str) -> str | None:
    raw = (raw or '').strip()
    # Accept full URLs too — strip the huggingface.co/datasets/ prefix.
    for pre in ('https://huggingface.co/datasets/',
                'http://huggingface.co/datasets/',
                'huggingface.co/datasets/'):
        if raw.startswith(pre):
            raw = raw[len(pre):]
            break
    raw = raw.strip('/').split('/tree/')[0]
    return raw if _HF_REPO_RE.match(raw) else None


@app.route('/leaderboard/<int:leaderboard_id>/materialize/status')
def lb_materialization_status(leaderboard_id: int):
    """JSON status endpoint polled by the LB page while a
    materialization is pending/running. Returns the row's status
    + progress payload + storage size."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    m = lb.materialization
    if m is None:
        return jsonify({'status': 'none'})
    try:
        progress = json.loads(m.progress_json) if m.progress_json else None
    except Exception:
        progress = None
    return jsonify({
        'status': m.status,
        'sample_cap': m.sample_cap,
        'sampling': m.sampling,
        'sampling_seed': m.sampling_seed,
        'stratify_field': m.stratify_field,
        'storage_bytes': m.storage_bytes,
        'error_message': m.error_message,
        'progress': progress,
    })


def _lb_has_real_submissions(lb):
    """True if the LB has any non-mirrored (real, submitter-uploaded)
    submission. Mirrored PWC score rows don't count — they have no
    on-disk predictions tied to the materialised inputs. Same notion of
    'frozen' the pred-field schema editor uses."""
    return any(
        (getattr(s, 'kind', 'verified') or 'verified') != 'mirrored'
        for s in (lb.submissions or [])
    )


@app.route('/leaderboard/<int:leaderboard_id>/materialize/edit', methods=['POST'])
@login_required
def edit_lb_materialization(leaderboard_id: int):
    """Update materialization params (sample_cap, sampling, seed,
    stratify_field) and re-queue. Wipes the existing materialised
    files since the new params will pick a different subset.
    Owner/admin only.
    """
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    if not (g.current_user and (
            g.current_user.id == lb.owner_user_id or is_admin(g.current_user))):
        abort(403)
    matrow = lb.materialization
    if matrow is None:
        flash('No materialization configured for this LB.', 'warning')
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    if _lb_has_real_submissions(lb):
        # Re-materialising would pick a different sample subset, silently
        # changing the inputs existing submissions were scored against.
        flash("Can't re-materialize — this leaderboard already has "
              "submissions. Delete them first to change the sample set.",
              'warning')
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))

    try:
        sample_cap = int(request.form.get('sample_cap') or matrow.sample_cap)
    except (TypeError, ValueError):
        sample_cap = matrow.sample_cap
    sampling = (request.form.get('sampling') or matrow.sampling).strip()
    if sampling not in ('head', 'random', 'stratified'):
        sampling = matrow.sampling
    try:
        sampling_seed = int(request.form.get('sampling_seed') or matrow.sampling_seed)
    except (TypeError, ValueError):
        sampling_seed = matrow.sampling_seed
    stratify_field = (request.form.get('stratify_field') or '').strip() or None

    # Wipe the previous materialisation on disk — new sample set =
    # different files. Storage bytes get reset on re-materialise.
    from benchhub.lb_materialize import materialization_dir
    mat_dir = materialization_dir(app.config['UPLOAD_FOLDER'], lb.id)
    if mat_dir.is_dir():
        shutil.rmtree(mat_dir, ignore_errors=True)

    matrow.sample_cap = sample_cap
    matrow.sampling = sampling
    matrow.sampling_seed = sampling_seed
    matrow.stratify_field = stratify_field
    matrow.status = 'pending'
    matrow.error_message = None
    matrow.progress_json = None
    matrow.storage_bytes = None
    db.session.commit()
    try:
        import tasks as _tasks
        _tasks.materialize_leaderboard.delay(lb.id)
        flash(
            f'Re-materializing with new params '
            f'({sample_cap} samples, {sampling} sampling).',
            'info',
        )
    except Exception as e:
        matrow.status = 'failed'
        matrow.error_message = f'enqueue: {e}'
        db.session.commit()
        flash(f'Couldn\'t queue: {e}', 'warning')
    return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))


@app.route('/leaderboard/<int:leaderboard_id>/materialize/retry', methods=['POST'])
@login_required
def retry_lb_materialization(leaderboard_id: int):
    """Re-enqueue the materialization task. Useful when the initial
    run failed (e.g. HF rate-limited) and the user wants to try again
    without recreating the LB. Owner / admin only."""
    lb = Leaderboard.query.get_or_404(leaderboard_id)
    if not (g.current_user and (
            g.current_user.id == lb.owner_user_id or is_admin(g.current_user))):
        abort(403)
    matrow = lb.materialization
    if matrow is None:
        flash('No materialization configured for this LB.', 'warning')
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    if _lb_has_real_submissions(lb):
        flash("Can't re-materialize — this leaderboard already has "
              "submissions. Delete them first to re-run materialization.",
              'warning')
        return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))
    matrow.status = 'pending'
    matrow.error_message = None
    db.session.commit()
    try:
        import tasks as _tasks
        _tasks.materialize_leaderboard.delay(lb.id)
        flash('Materialization re-queued.', 'info')
    except Exception as e:
        matrow.status = 'failed'
        matrow.error_message = f'enqueue: {e}'
        db.session.commit()
        flash(f'Couldn\'t queue retry: {e}', 'warning')
    return redirect(url_for('leaderboard_view', leaderboard_id=lb.id))


@app.route('/dataset_requests/add', methods=['POST'])
@login_required
def add_dataset_request():
    """Anyone signed in can file a 'please import this HF repo' wish.
    Dedup on repo_id: if a pending row exists, auto-upvote it instead
    of creating a duplicate; if an imported/rejected row exists, allow
    a new pending request (it may have changed upstream)."""
    repo = _parse_hf_repo_id(request.form.get('repo_id') or '')
    if not repo:
        flash('Enter a HuggingFace repo id (user/repo) or full URL.', 'warning')
        return redirect(request.referrer or url_for('datasets_list'))
    description = (request.form.get('description') or '').strip()[:1000] or None

    existing = (DatasetRequest.query
                .filter_by(repo_id=repo, status='pending')
                .first())
    if existing is not None:
        # Treat duplicate submit as an upvote toggle.
        v = DatasetRequestVote.query.filter_by(
            request_id=existing.id, user_id=g.current_user.id
        ).first()
        if v is None:
            db.session.add(DatasetRequestVote(
                request_id=existing.id, user_id=g.current_user.id))
            db.session.commit()
            flash(f'Already requested — upvoted "{repo}".', 'info')
        else:
            flash(f'You\'ve already voted for "{repo}".', 'info')
        return redirect(request.referrer or url_for('datasets_list'))

    req = DatasetRequest(
        repo_id=repo, description=description,
        requested_by_user_id=g.current_user.id,
        status='pending',
    )
    db.session.add(req); db.session.flush()
    # Submitter auto-votes for their own request.
    db.session.add(DatasetRequestVote(
        request_id=req.id, user_id=g.current_user.id))
    db.session.commit()
    flash(f'Requested "{repo}".', 'success')
    return redirect(request.referrer or url_for('datasets_list'))


@app.route('/dataset_requests/<int:req_id>/vote', methods=['POST'])
@login_required
def vote_dataset_request(req_id: int):
    """Toggle vote: add if absent, remove if present."""
    req = DatasetRequest.query.get_or_404(req_id)
    v = DatasetRequestVote.query.filter_by(
        request_id=req.id, user_id=g.current_user.id).first()
    if v is None:
        db.session.add(DatasetRequestVote(
            request_id=req.id, user_id=g.current_user.id))
    else:
        db.session.delete(v)
    db.session.commit()
    return redirect(request.referrer or url_for('datasets_list'))


@app.route('/admin/dataset_requests/<int:req_id>/status', methods=['POST'])
@login_required
def admin_set_dataset_request_status(req_id: int):
    """Admin triage: mark a request as imported or rejected, with a note."""
    if not g.current_user.is_admin:
        abort(403)
    req = DatasetRequest.query.get_or_404(req_id)
    status = request.form.get('status', '').strip()
    if status in ('pending', 'imported', 'rejected'):
        req.status = status
    note = (request.form.get('admin_note') or '').strip()
    if note:
        req.admin_note = note
    db.session.commit()
    return redirect(request.referrer or url_for('datasets_list'))

@app.route('/author_avatars/<filename>')
def serve_author_avatar(filename):
    avatar_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'author_avatars')
    return send_from_directory(avatar_dir, filename)

# Source-of-truth for accepted dataset field kinds. Pulled from the
# typed registry plus 'metric'/'histogram', which are legacy values
# the engine still recognises but aren't full DataType subclasses.
def _valid_field_types() -> set[str]:
    from benchhub.types import DTYPES as _DTYPES
    return set(_DTYPES.keys()) | {'metric', 'histogram'}


def _lbs_using_dataset(dataset):
    """Return the Leaderboards that have this dataset attached. Used
    to gate destructive field-schema edits — once an LB references a
    dataset, changing a field's kind or role on the dataset would
    silently re-interpret data the LB's metrics already scored
    against. Edits remain available while the dataset is unbound."""
    return list(dataset.leaderboards) if dataset else []


def _dataset_field_types(dataset):
    """Return [{name, current_type, role, sample_count, mixed}] for
    every distinct custom-field name on this dataset, used by the
    settings UI.

    `role` comes from the declared `DatasetField.role` so the settings
    page can render a per-field role selector alongside the kind one.
    """
    sample_ids = [s.id for s in dataset.samples]
    if not sample_ids:
        return []
    rows = (
        db.session.query(
            CustomField.name,
            CustomField.data_type,
            func.count(CustomField.id),
        )
        .filter(CustomField.sample_id.in_(sample_ids))
        .group_by(CustomField.name, CustomField.data_type)
        .all()
    )
    # Roles + canonical kinds from the declared schema (DatasetField),
    # which is the source the LB picker reads against.
    df_meta = {
        df.name: {'role': (df.role or 'gt').lower(), 'kind': df.kind}
        for df in dataset.fields
    }
    # Collapse: a field name with mixed types (rare, mid-edit state) shows
    # the most common one and a "mixed" flag.
    by_name = {}
    for name, ftype, cnt in rows:
        entry = by_name.setdefault(name, {
            'name': name, 'current_type': ftype,
            'sample_count': 0, 'mixed': False,
            'role': (df_meta.get(name) or {}).get('role', 'gt'),
        })
        entry['sample_count'] += cnt
        if entry.get('current_type') != ftype:
            entry['mixed'] = True
    return sorted(by_name.values(), key=lambda r: r['name'])


@app.route('/dataset/<int:dataset_id>/settings', methods=['GET', 'POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def dataset_settings(dataset_id):
    """Dedicated settings page — collects all owner-only controls
    (sharing, danger zone, field types, category) in one place so
    they aren't scattered across the busy dataset detail view.

    POST is for inline edits on this same page (currently: the
    Area/Task category). Other settings still have their own
    dedicated POST routes (sharing visibility, field-type
    reclassifier, pred-field add/remove).
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    if request.method == 'POST':
        if 'category' in request.form:
            raw = request.form.get('category') or ''
            # CSV of Area/Task paths. Normalise each entry: trim each
            # path segment, collapse duplicate slashes, drop blank
            # entries; dedupe in declared order. Empty result clears.
            out, seen = [], set()
            for chunk in raw.split(','):
                c = '/'.join(seg.strip() for seg in chunk.split('/') if seg.strip())
                if c and c not in seen:
                    seen.add(c)
                    out.append(c)
            dataset.category = ', '.join(out) if out else None
            db.session.commit()
            flash(f'Saved categories: {dataset.category or "(none)"}', 'success')
        return redirect(url_for('dataset_settings', dataset_id=dataset.id))
    field_types = _dataset_field_types(dataset)
    # Source provenance for the HF-import badge.
    hf_meta = None
    if (dataset.source_kind or '').startswith('hf-') and dataset.source_metadata:
        try:
            hf_meta = json.loads(dataset.source_metadata)
        except Exception:
            hf_meta = None
    # `valid_field_types` is the legacy type list used by the old
    # field-type reclassifier. For the new pred-field editor we
    # want the full DTYPES registry so admins can add e.g. a
    # label_list pred to an existing dataset.
    from benchhub.types import DTYPES as _DTYPES
    # Category datalist — union of existing dataset + LB categories
    # so the user gets type-ahead for buckets in use elsewhere on
    # the site instead of inventing a one-off Area/Task that
    # nobody else will browse to.
    known_categories = sorted({
        row[0] for row in (
            db.session.query(Dataset.category)
            .filter(Dataset.category.isnot(None)).distinct().all()
        ) + (
            db.session.query(Leaderboard.category)
            .filter(Leaderboard.category.isnot(None)).distinct().all()
        ) if row[0]
    })
    # Schema edits (kind / role) are locked once an LB binds metrics
    # against this dataset — changing a field then would silently
    # re-route bytes those metrics scored against. Show the locked
    # state in the UI with the list of leaderboards so the owner can
    # detach them if they really need to edit.
    lbs_using = _lbs_using_dataset(dataset)
    # Candidates for the "use field X as sample name" control —
    # scalar / text / label are the only kinds whose value IS the
    # signal (file-backed kinds would need an extra decode step
    # we don't bother with here).
    name_source_candidates = sorted(
        f.name for f in dataset.fields if f.kind in ('scalar', 'text', 'label')
    )
    # Visualisations the user can pick from in the "Add a viz" form.
    # Include the user's own private viz rows plus every public one.
    _me_id = g.current_user.id if getattr(g, 'current_user', None) else None
    viz_candidates = (
        GlobalVisualization.query.filter(
            (GlobalVisualization.visibility == 'public') |
            (GlobalVisualization.owner_user_id == _me_id)
        ).order_by(GlobalVisualization.name).all()
    )
    # Per-candidate metadata used by the form to render arg pickers
    # (input_kinds → which dataset fields match for each arg).
    viz_specs = []
    dataset_fields_by_kind: dict[str, list[str]] = {}
    for f in dataset.dataset_fields:
        dataset_fields_by_kind.setdefault(f.kind, []).append(f.name)
    for gv in viz_candidates:
        kinds = []
        try:
            kinds = json.loads(gv.input_kinds) if gv.input_kinds else []
        except Exception:
            kinds = []
        # Try to infer arg names from the function signature in
        # python_code. Fallback to arg_0 / arg_1 / …
        arg_names: list[str] = []
        try:
            m = re.search(r'def\s+\w+\s*\(([^)]*)\)', gv.python_code or '')
            if m:
                for tok in m.group(1).split(','):
                    name = tok.split(':')[0].split('=')[0].strip()
                    if name and name != 'self':
                        arg_names.append(name)
        except Exception:
            pass
        if not arg_names:
            arg_names = [f'arg_{i}' for i in range(len(kinds))]
        viz_specs.append({
            'id': gv.id, 'name': gv.name, 'description': gv.description or '',
            'input_kinds': kinds, 'arg_names': arg_names,
        })

    attached_viz = (
        DatasetVisualization.query.filter_by(dataset_id=dataset.id)
        .order_by(DatasetVisualization.display_order, DatasetVisualization.id).all()
    )
    return render_template(
        'dataset_settings.html',
        dataset=dataset,
        field_types=field_types,
        valid_field_types=sorted(_valid_field_types()),
        hf_meta=hf_meta,
        known_categories=known_categories,
        lbs_using=lbs_using,
        schema_editable=(not lbs_using),
        name_source_candidates=name_source_candidates,
        viz_specs=viz_specs,
        attached_viz=attached_viz,
        dataset_fields_by_kind=dataset_fields_by_kind,
    )


@app.route('/dataset/<int:dataset_id>/visualizations/add', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def add_dataset_visualization(dataset_id):
    """Attach a GlobalVisualization to this dataset. Form posts:
      - global_visualization_id
      - arg_<arg_name> field per arg in the viz function signature,
        each holding a dataset field name (mapped through `gt_<field>`).
      - target_name (optional override for display).
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    gv_id = request.form.get('global_visualization_id', type=int)
    gv = GlobalVisualization.query.get_or_404(gv_id) if gv_id else None
    if gv is None:
        flash('Pick a visualisation.', 'warning')
        return redirect(url_for('dataset_settings', dataset_id=dataset.id))

    # Pull arg mappings from arg_<name> form fields.
    mappings = {}
    for key, val in request.form.items():
        if not key.startswith('arg_'): continue
        arg = key[4:]
        val = (val or '').strip()
        if val:
            mappings[arg] = f'gt_{val}' if not val.startswith(('gt_', 'sub_', 'SCALAR:')) else val
    if not mappings:
        flash('Map at least one argument to a dataset field.', 'warning')
        return redirect(url_for('dataset_settings', dataset_id=dataset.id))

    dv = DatasetVisualization(
        dataset_id=dataset.id,
        global_visualization_id=gv.id,
        arg_mappings=json.dumps(mappings),
        target_name=(request.form.get('target_name') or '').strip() or None,
        display_order=(DatasetVisualization.query.filter_by(dataset_id=dataset.id).count()),
    )
    db.session.add(dv); db.session.commit()
    flash(f'Attached visualisation "{gv.name}".', 'success')
    return redirect(url_for('dataset_settings', dataset_id=dataset.id))


@app.route('/dataset/<int:dataset_id>/visualizations/<int:dv_id>/remove', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def remove_dataset_visualization(dataset_id, dv_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    dv = DatasetVisualization.query.get_or_404(dv_id)
    if dv.dataset_id != dataset.id:
        abort(404)
    name = dv.global_visualization.name if dv.global_visualization else 'visualisation'
    db.session.delete(dv); db.session.commit()
    flash(f'Removed {name}.', 'success')
    return redirect(url_for('dataset_settings', dataset_id=dataset.id))


@app.route('/dataset/<int:dataset_id>/pred_fields/add', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def add_dataset_pred_field(dataset_id):
    """Append a `role=pred` DatasetField to a dataset post-import.

    The HF preview's Prediction-fields section is the primary path,
    but a user might realise mid-experiment that they need a top-K
    pred (`label_list`) on an existing dataset. Adding it here
    avoids the re-import → re-bind metrics cycle.

    Validates per the manifest rules (known kind, kind-specific
    params — e.g. label_list needs a positive integer `k`).
    """
    from benchhub.types import DTYPES as _DTYPES
    dataset = Dataset.query.get_or_404(dataset_id)
    name = (request.form.get('name') or '').strip()
    kind = (request.form.get('kind') or '').strip()
    params_raw = (request.form.get('params') or '').strip()

    if not name:
        flash('Pred field name is required.', 'danger')
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))
    if kind not in _DTYPES:
        flash(f"Unknown kind {kind!r}; expected one of {sorted(_DTYPES)}.", 'danger')
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))
    if any(f.name == name for f in dataset.fields):
        flash(f"A field named {name!r} already exists on this dataset.", 'danger')
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))
    try:
        params = json.loads(params_raw) if params_raw else {}
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
    except (TypeError, ValueError) as e:
        flash(f"Params must be a JSON object: {e}", 'danger')
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))
    # Kind-specific params check (same logic as validate_manifest).
    if kind == 'label_list':
        k = params.get('k')
        if not isinstance(k, int) or k < 1:
            flash("label_list requires params.k as a positive integer "
                  "(e.g. {\"k\": 5}).", 'danger')
            return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    df = DatasetField(dataset_id=dataset.id, name=name, kind=kind, role='pred')
    df.set_params(params)
    db.session.add(df)
    db.session.commit()
    flash(f'Added pred field "{name}" (kind={kind}).', 'success')
    return redirect(url_for('dataset_settings', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/pred_fields/<path:field_name>/delete', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def delete_dataset_pred_field(dataset_id, field_name):
    """Drop a role=pred DatasetField from a dataset. Submissions to
    existing LBs that referenced this pred field stay (they're
    already evaluated and stored as MetricResult), but new submissions
    will follow the updated contract."""
    dataset = Dataset.query.get_or_404(dataset_id)
    df = next(
        (f for f in dataset.fields
         if f.name == field_name and (f.role or '').lower() == 'pred'),
        None,
    )
    if df is None:
        flash(f"No pred field named {field_name!r}.", 'warning')
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))
    db.session.delete(df)
    db.session.commit()
    flash(f'Removed pred field "{field_name}".', 'success')
    return redirect(url_for('dataset_settings', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/field/<path:field_name>/type', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def update_dataset_field_type(dataset_id, field_name):
    """Re-classify a dataset field's kind. Updates both
    `DatasetField.kind` (the declared schema) and every CustomField
    row that carries this name on the dataset (so the runtime
    decoder picks the right path).

    Gated on the dataset having no leaderboards attached — once an
    LB binds metrics against a field, changing its kind would
    silently re-interpret bytes those metrics already scored
    against. Detach the LBs (or delete them) to unlock.
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    lbs = _lbs_using_dataset(dataset)
    if lbs:
        flash(
            f"Can't change field types — {len(lbs)} leaderboard"
            f"{'' if len(lbs) == 1 else 's'} use this dataset. Detach "
            "them first to unlock.",
            "warning",
        )
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    new_type = (request.form.get('data_type') or '').strip()
    valid = _valid_field_types()
    if new_type not in valid:
        flash(f"Invalid field type '{new_type}'.", "danger")
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    # Update the declared schema row (DatasetField.kind drives the
    # picker satisfiability checks + role-table rendering on LB edit).
    df = DatasetField.query.filter_by(
        dataset_id=dataset.id, name=field_name,
    ).first()
    if df is not None:
        df.kind = new_type

    sample_ids = [s.id for s in dataset.samples]
    updated = 0
    if sample_ids:
        updated = (
            CustomField.query
            .filter(CustomField.sample_id.in_(sample_ids),
                    CustomField.name == field_name)
            .update({'data_type': new_type}, synchronize_session=False)
        )
    db.session.commit()
    flash(
        f"Reclassified '{field_name}' to '{new_type}' "
        f"({updated} sample row(s) updated).",
        "success",
    )
    return redirect(url_for('dataset_settings', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/field/<path:field_name>/role', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def update_dataset_field_role(dataset_id, field_name):
    """Change a dataset field's declared role (`input` / `gt` /
    `skip`). Gated on no-LBs-using — same reason as the kind editor:
    an LB's metric mappings refer to the field's gt/input
    designation, and flipping it after the fact would silently
    re-route or skip the data those metrics are scoring against.
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    lbs = _lbs_using_dataset(dataset)
    if lbs:
        flash(
            f"Can't change field roles — {len(lbs)} leaderboard"
            f"{'' if len(lbs) == 1 else 's'} use this dataset. Detach "
            "them first to unlock.",
            "warning",
        )
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    new_role = (request.form.get('role') or '').strip().lower()
    if new_role not in ('input', 'gt', 'skip'):
        flash(f"Invalid role '{new_role}'.", "danger")
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    df = DatasetField.query.filter_by(
        dataset_id=dataset.id, name=field_name,
    ).first()
    if df is None:
        flash(f"No field named '{field_name}' on this dataset.", "danger")
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))
    df.role = new_role
    db.session.commit()
    flash(f"'{field_name}' is now role={new_role}.", "success")
    return redirect(url_for('dataset_settings', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/rename_samples_by_field', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def rename_samples_by_field(dataset_id):
    """Rename every Sample on this dataset using one of its scalar /
    text custom-field values. Picks files on disk along for the ride
    — every CustomField stored as a relative path gets its file
    renamed and its `value_text` updated to match.

    The original-name → new-name mapping is built first; collisions
    after sanitisation get `_2`, `_3`, …, so the resulting Sample.name
    set stays unique. Empty / null source values fall back to the
    original sample name (no-op for that row).

    Form: `field_name=<existing field>`. Locked when an LB binds
    this dataset (same gate as schema edits) since file-path
    rewrites would break submissions already scored against the old
    sample names.
    """
    import re as _re
    import shutil as _shutil
    dataset = Dataset.query.get_or_404(dataset_id)
    lbs = _lbs_using_dataset(dataset)
    if lbs:
        flash(
            f"Can't rename samples — {len(lbs)} leaderboard"
            f"{'' if len(lbs) == 1 else 's'} use this dataset. Detach "
            "them first.",
            "warning",
        )
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    field_name = (request.form.get('field_name') or '').strip()
    if not field_name:
        flash("Pick a field first.", "danger")
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    # Source field must be scalar or text — those are the only ones
    # whose value is the entire signal (no file decoding needed).
    src_df = DatasetField.query.filter_by(
        dataset_id=dataset.id, name=field_name,
    ).first()
    if src_df is None or src_df.kind not in ('scalar', 'text', 'label'):
        flash(
            f"'{field_name}' must be scalar / text / label "
            f"(got {src_df.kind if src_df else 'missing'}).",
            "danger",
        )
        return redirect(url_for('dataset_settings', dataset_id=dataset_id))

    samples = Sample.query.filter_by(dataset_id=dataset.id).all()
    # Pull the source values per sample in one query.
    src_cfs = (
        CustomField.query
        .filter(CustomField.name == field_name,
                CustomField.sample_id.in_([s.id for s in samples]))
        .all()
    )
    src_by_sample = {cf.sample_id: cf for cf in src_cfs}

    safe_re = _re.compile(r"[^A-Za-z0-9._-]+")

    def _sanitize(raw) -> str:
        # Drop trailing `.0` when the source is an integer-valued
        # float (HF Value('int32') columns arrive as floats through
        # the CustomField.value_float pathway, so a `139` image_id
        # becomes `139.0` on read). Without this, every numeric-id
        # rename leaves an ugly `.0` suffix.
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        s = safe_re.sub("_", str(raw)).strip("._ \t\r\n")
        return s[:80]

    # Build the desired-name map, deduplicating with `_<n>` suffixes.
    proposed: dict[int, str] = {}
    used: set[str] = set()
    for s in samples:
        cf = src_by_sample.get(s.id)
        if cf is None:
            proposed[s.id] = s.name
            used.add(s.name)
            continue
        raw = cf.value_text if cf.value_text is not None else cf.value_float
        if raw is None or str(raw).strip() == '':
            proposed[s.id] = s.name
            used.add(s.name)
            continue
        cand = _sanitize(raw)
        if not cand:
            proposed[s.id] = s.name
            used.add(s.name)
            continue
        # Dedupe collisions among NEW names. Don't let a renamed
        # sample collide with another sample's old name either —
        # we apply renames in a single batch, but file-system safety
        # means treating any prior name as occupied.
        existing_taken = {x.name for x in samples}
        base = cand; i = 2
        while cand in used or (cand in existing_taken and
                                cand != next((x.name for x in samples if x.id == s.id), None)):
            cand = f"{base}_{i}"; i += 1
        proposed[s.id] = cand
        used.add(cand)

    # Apply: walk every sample, do the rename in two steps —
    # first move files on disk + update CustomField paths, then
    # update Sample.name. Errors short-circuit + rollback the
    # session, but partially-renamed disk files stay where they
    # landed (better than half-committed DB state diverging from
    # disk).
    upload = app.config['UPLOAD_FOLDER']
    renamed = 0
    for s in samples:
        new_name = proposed[s.id]
        if new_name == s.name:
            continue
        # Per-CustomField file rename for file-backed rows on this
        # sample. value_text is `datasets/<id>/<field>/<old>.<ext>`.
        for cf in s.custom_fields:
            v = cf.value_text or ''
            if not (v.startswith('datasets/') or v.startswith('submissions/')):
                continue
            old_full = os.path.join(upload, v)
            if not os.path.isfile(old_full):
                continue
            # Keep the parent dir + extension; swap the basename's stem.
            parent, base = os.path.split(v)
            stem, ext = os.path.splitext(base)
            if stem != s.name:
                # Filename doesn't match Sample.name — this happens
                # for datasets the agent-mode importer namespaced
                # ids differently. Leave that field alone; the
                # mismatch is intentional.
                continue
            new_rel = os.path.join(parent, f"{new_name}{ext}")
            new_full = os.path.join(upload, new_rel)
            try:
                _shutil.move(old_full, new_full)
            except OSError:
                continue
            cf.value_text = new_rel
        s.name = new_name
        renamed += 1

    db.session.commit()
    flash(
        f"Renamed {renamed} sample(s) using '{field_name}'.",
        "success",
    )
    return redirect(url_for('dataset_settings', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/import_status.json')
@visibility_required(Dataset, 'dataset_id')
def dataset_import_status(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    try:
        progress = json.loads(dataset.import_progress_json) if dataset.import_progress_json else {}
    except (TypeError, ValueError):
        progress = {}
    return jsonify({
        'status': dataset.import_status or 'ready',
        'error': dataset.import_error,
        'progress': progress,
    })


@app.route('/dataset/<int:dataset_id>')
@visibility_required(Dataset, 'dataset_id')
def dataset_view(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    # When the dataset is still being imported asynchronously, the
    # rest of the dataset_view assumes Sample / CustomField rows
    # already exist — render a small progress page instead.
    # Failed imports also land here so the admin sees the error
    # message + can clean up.
    if (dataset.import_status or 'ready') in ('importing', 'failed'):
        try:
            progress = json.loads(dataset.import_progress_json) if dataset.import_progress_json else {}
        except (TypeError, ValueError):
            progress = {}
        return render_template(
            'dataset_importing.html',
            dataset=dataset,
            progress=progress,
        )
    page = request.args.get('page', 1, type=int)
    samples_per_page = request.args.get('per_page', 100, type=int)
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
    # ?lb=<id> scopes the view to a single LB's materialised subset.
    # The dataset row itself stays the source of truth (kinds,
    # display columns, viz attachments) — we just filter Sample names
    # to those present on disk under
    # uploads/lb_materializations/<lb_id>/. Surfaces a banner.
    lb_filter_id = request.args.get('lb', type=int)
    lb_filter_obj = None
    lb_filter_names_count = None
    dataset_total_samples = None
    if lb_filter_id:
        lb_filter_obj = Leaderboard.query.get(lb_filter_id)
        if lb_filter_obj:
            from benchhub.lb_materialize import list_materialized_samples
            mat_names = list_materialized_samples(
                app.config['UPLOAD_FOLDER'], lb_filter_id,
            )
            lb_filter_names_count = len(mat_names)
            # Pre-count the full dataset so the banner can show
            # "N of M" without forcing a `.samples` list-load in the
            # template. Done before the .in_() filter is applied.
            dataset_total_samples = Sample.query.filter_by(
                dataset_id=dataset.id,
            ).count()
            if mat_names:
                samples_query = samples_query.filter(Sample.name.in_(mat_names))
            else:
                # LB has no materialisation yet — show empty rather than
                # silently fall back to the full dataset (would mislead).
                samples_query = samples_query.filter(False)
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
    custom_field_query = db.session.query(CustomField.name, CustomField.data_type).join(Sample).filter(
        Sample.dataset_id == dataset.id,
        CustomField.submission_id == None
    ).distinct().all()
    
    custom_field_names = set(custom_field_query)
    custom_scalar_metric_names = [name for name, ftype in custom_field_names if ftype in ('scalar', 'metric')]

    # Class-name vocabularies for label fields. Built up front so
    # the filter logic below can translate `cat` → index 3 before
    # building the CustomField subquery. Mask fields can also carry
    # a names list (e.g. cityscapes' 35 classes), used by the hover
    # tooltip to read "12 road" instead of just "12".
    label_vocabs = {}
    mask_vocabs = {}
    for df in dataset.fields:
        names = (df.get_params() or {}).get('names')
        if not isinstance(names, list) or not names:
            continue
        if df.kind == 'label':
            label_vocabs[df.name] = list(names)
        elif df.kind == 'mask':
            mask_vocabs[df.name] = list(names)

    # Field-filter: narrow the sample list to those whose chosen
    # text / scalar / label / metric custom field matches
    # `filter_value`. The synthetic field `sample_name` filters
    # Sample.name directly. Text-ish kinds do a case-insensitive
    # substring match; numeric kinds accept a bare number or an
    # operator-prefixed form like `>1.5`, `<=10`, `==42`.
    filter_field = (request.args.get('filter_field') or '').strip()
    filter_value = (request.args.get('filter_value') or '').strip()
    field_type_map = {name: ftype for name, ftype in custom_field_names}
    # Build the dropdown source: synthetic sample_name first, then
    # any text/label/scalar/metric CF.
    filterable_fields = ['sample_name'] + sorted(
        name for name, ftype in custom_field_names
        if ftype in ('text', 'label', 'scalar', 'metric')
    )
    filterable_kinds = {'sample_name': 'sample_name'}
    filterable_kinds.update(field_type_map)

    if filter_field and filter_value:
        if filter_field == 'sample_name':
            samples_query = samples_query.filter(
                Sample.name.ilike(f'%{filter_value.strip()}%')
            )
        elif filter_field in field_type_map:
            ftype = field_type_map[filter_field]
            cf_subq = db.session.query(CustomField.sample_id).filter(
                CustomField.name == filter_field,
                CustomField.submission_id.is_(None),
            )
            if ftype in ('text', 'label'):
                needle = filter_value.strip().strip('"').strip("'")
                # Label-with-vocab special case: if the typed string
                # matches a class name (case-insensitive) — including
                # `0 airplane` or `cat` — translate to the integer
                # index stored in value_text. Otherwise fall back to
                # substring match on the raw value_text.
                if ftype == 'label':
                    vocab = label_vocabs.get(filter_field) or []
                    translated = None
                    if vocab:
                        # Accept either the bare class name or the
                        # combined "<idx> <name>" the cell renders.
                        candidate = needle.split(None, 1)
                        if len(candidate) == 2 and candidate[0].isdigit():
                            try:
                                idx = int(candidate[0])
                                if 0 <= idx < len(vocab) and vocab[idx].lower() == candidate[1].lower():
                                    translated = str(idx)
                            except ValueError:
                                pass
                        if translated is None:
                            for i, nm in enumerate(vocab):
                                if nm.lower() == needle.lower():
                                    translated = str(i)
                                    break
                    if translated is not None:
                        cf_subq = cf_subq.filter(CustomField.value_text == translated)
                    else:
                        cf_subq = cf_subq.filter(CustomField.value_text.ilike(f'%{needle}%'))
                else:
                    cf_subq = cf_subq.filter(CustomField.value_text.ilike(f'%{needle}%'))
            else:
                # scalar / metric. Parse an optional comparison operator;
                # default is equality.
                import re as _re
                m = _re.match(r'^\s*(==|=|!=|<=|>=|<|>)?\s*(-?\d+(?:\.\d+)?)\s*$', filter_value)
                if m:
                    op = m.group(1) or '='
                    try:
                        num = float(m.group(2))
                    except ValueError:
                        num = None
                    if num is not None:
                        col = CustomField.value_float
                        if op in ('=', '=='):
                            cf_subq = cf_subq.filter(col == num)
                        elif op == '!=':
                            cf_subq = cf_subq.filter(col != num)
                        elif op == '<':
                            cf_subq = cf_subq.filter(col < num)
                        elif op == '<=':
                            cf_subq = cf_subq.filter(col <= num)
                        elif op == '>':
                            cf_subq = cf_subq.filter(col > num)
                        elif op == '>=':
                            cf_subq = cf_subq.filter(col >= num)
            samples_query = samples_query.filter(Sample.id.in_(cf_subq))

    # Autocomplete suggestions for the active filter field. Capped
    # at 50 entries to keep the rendered <datalist> tractable on
    # large datasets. Numeric kinds skip suggestions (the operator
    # syntax is more useful than a flat list of numbers).
    filter_suggestions: list[str] = []
    if filter_field == 'sample_name':
        rows = db.session.query(Sample.name).filter(
            Sample.dataset_id == dataset.id
        ).distinct().limit(50).all()
        filter_suggestions = [r[0] for r in rows if r[0]]
    elif filter_field in field_type_map:
        ftype = field_type_map[filter_field]
        if ftype == 'label':
            vocab = label_vocabs.get(filter_field) or []
            filter_suggestions = [f"{i} {nm}" for i, nm in enumerate(vocab)]
        elif ftype == 'text':
            rows = (
                db.session.query(CustomField.value_text)
                .join(Sample, Sample.id == CustomField.sample_id)
                .filter(
                    Sample.dataset_id == dataset.id,
                    CustomField.name == filter_field,
                    CustomField.submission_id.is_(None),
                )
                .distinct()
                .limit(50)
                .all()
            )
            filter_suggestions = [r[0] for r in rows if r[0]]

    # Sorting
    if sort_by == 'name':
        # Natural sort: shorter strings before longer ones, lex
        # within equal lengths. So 1, 2, 10, 100 and s_1, s_2, s_10
        # come out in numeric / human order instead of lex's
        # 1, 10, 100, 2.
        if sort_order == 'desc':
            samples_query = samples_query.order_by(
                func.length(Sample.name).desc(), Sample.name.desc())
        else:
            samples_query = samples_query.order_by(
                func.length(Sample.name).asc(), Sample.name.asc())
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
    hist_map = {}
    shape_map = {}

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
            hist_cf = next((cf for cf in sample.custom_fields if cf.name == 'hist' and cf.data_type == 'histogram'), None)
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
            shape_cf = next((cf for cf in sample.custom_fields if cf.name == 'wave_shape' and cf.data_type == 'scalar'), None)
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
            
            if cf.data_type in ('scalar', 'metric'):
                metrics[cf.name] = cf.value_float
                cf_vals[cf.name] = {'type': cf.data_type, 'value': cf.value_float, 'field_id': cf.id}
            elif cf.data_type == 'label':
                # Label values land in value_text as JSON-encoded primitives
                # ("cat" or 3). Strip the JSON wrap so the template shows
                # the bare value instead of a quoted string. If the field
                # carries a class-name vocab (DatasetField.data_params.names,
                # populated from HF ClassLabel.names), render as
                # "<idx> <name>" so the cell carries both pieces of
                # information (matches the legend panel's format).
                raw = cf.value_text
                display = raw
                if raw:
                    try:
                        display = json.loads(raw)
                    except (TypeError, ValueError):
                        display = raw
                names = label_vocabs.get(cf.name)
                if names and isinstance(display, int) and 0 <= display < len(names):
                    display = f"{display} {names[display]}"
                cf_vals[cf.name] = {'type': 'label', 'value': display, 'field_id': cf.id}
            else:
                cf_vals[cf.name] = {'type': cf.data_type, 'value': cf.value_text, 'field_id': cf.id}

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
    
    # Inject detected custom fields. Scalars stay out of this list —
    # they surface in the per-source-stats panel — but every other
    # typed kind (image/mask/depth/json/text/label/audio/bboxes)
    # gets its own column.
    for field_name, data_type in custom_field_names:
        if data_type == 'image':
            available_display_options[field_name] = {'label': field_name, 'type': 'image', 'default_width': '150px'}
        elif data_type == 'mask':
            available_display_options[field_name] = {'label': field_name, 'type': 'mask', 'default_width': '150px'}
        elif data_type == 'depth':
            available_display_options[field_name] = {'label': field_name, 'type': 'depth', 'default_width': '150px'}
        elif data_type == 'audio':
            available_display_options[field_name] = {'label': field_name, 'type': 'audio', 'default_width': '200px'}
        elif data_type == 'json':
            available_display_options[field_name] = {'label': field_name, 'type': 'json', 'default_width': '150px'}
        elif data_type == 'bboxes':
            available_display_options[field_name] = {'label': field_name, 'type': 'bboxes', 'default_width': '150px'}
        elif data_type == 'coco_detections':
            # The raw json column itself — same renderer as 'json'.
            # The image-overlay viz gets its own column below.
            available_display_options[field_name] = {'label': field_name, 'type': 'json', 'default_width': '150px'}
        elif data_type == 'label':
            # Classification GT, vocab-keyed. Narrow text-style column.
            available_display_options[field_name] = {'label': field_name, 'type': 'label', 'default_width': '100px'}
        elif data_type == 'text' and field_name != 'tags':
            # Text columns (AG News `text`, NLI `premise`/`hypothesis`,
            # captions, etc.) need their own column too. Skip the
            # reserved `tags` field — that one is already surfaced via
            # the dataset's tag widget.
            available_display_options[field_name] = {'label': field_name, 'type': 'text', 'default_width': '300px'}
        # Scalars are not added as individual columns - they appear in per_source_stats
    
    # Check if any sample has data for these fields
    has_tags = bool(all_dataset_tags)
    
    # Efficient existence checks
    has_hist = any(ft == 'histogram' for fn, ft in custom_field_names)
    has_shape = any(fn == 'wave_shape' for fn, ft in custom_field_names)

    if not has_hist: available_display_options.pop('histogram', None)
    if not has_shape: available_display_options.pop('signal_shape', None)
    if not has_tags: available_display_options.pop('tags', None)

    # GT Stats panel surfaces scalar + metric GT custom fields. No
    # such fields on this dataset → no panel to render → drop the
    # column option entirely so it doesn't show up in View Options.
    has_gt_scalars_or_metrics = any(
        ft in ('scalar', 'metric') for fn, ft in custom_field_names
    )
    if not has_gt_scalars_or_metrics:
        available_display_options.pop('per_source_stats', None)

    # Dataset-attached visualisations → per-sample image columns.
    # Each DatasetVisualization gets a key like `dsviz_<dv_id>`; the
    # template hits /api/dataset_viz/<dv_id>/<sample_id> to render.
    for dv in DatasetVisualization.query.filter_by(dataset_id=dataset.id) \
            .order_by(DatasetVisualization.display_order,
                      DatasetVisualization.id).all():
        gv = dv.global_visualization
        if gv is None:
            continue
        label = dv.target_name or gv.name
        available_display_options[f'dsviz_{dv.id}'] = {
            'label': label, 'type': 'dataset_viz',
            'dv_id': dv.id, 'default_width': '180px',
        }

    # Filter selected columns to ensure they exist in available options
    selected_display_columns = [col for col in selected_display_columns if col in available_display_options]

    active_visualizations = [v for v in dataset.visualizations.split(',') if v.strip()]
    # Dynamic sample metric options (no custom metrics - they auto-appear in charts)
    sample_metric_options_dynamic = SAMPLE_METRIC_OPTIONS.copy()
    
    # Extract custom scalar metric names for auto-inclusion in charts
    custom_scalar_metrics = [field_name for field_name, data_type in custom_field_names if data_type == 'scalar']
    
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

    # LB-creation form context — shared with the standalone
    # /dataset/<id>/create_lb page below.
    available_metrics_for_lb, gt_field_options, dataset_field_options = (
        _build_lb_creation_context(dataset)
    )

    # Category datalist used by the inline category editor in the
    # dataset_view header — union of existing dataset + LB
    # categories already in use elsewhere on the site, exploded over
    # the CSV separator so multi-category values surface each one.
    _known_cats_raw = sorted({
        row[0] for row in (
            db.session.query(Dataset.category)
            .filter(Dataset.category.isnot(None)).distinct().all()
        ) + (
            db.session.query(Leaderboard.category)
            .filter(Leaderboard.category.isnot(None)).distinct().all()
        ) if row[0]
    })
    known_categories = sorted({
        c.strip() for raw in _known_cats_raw for c in raw.split(',') if c.strip()
    })

    return render_template('dataset_view.html',
                           known_categories=known_categories,
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
                           label_vocabs=label_vocabs,
                           mask_vocabs=mask_vocabs,
                           lb_filter=lb_filter_obj,
                           lb_filter_names_count=lb_filter_names_count,
                           dataset_total_samples=dataset_total_samples,
                           custom_scalar_metrics=custom_scalar_metrics,
                           sample_search_query=sample_search_query,
                           filter_field=filter_field,
                           filter_value=filter_value,
                           filterable_fields=filterable_fields,
                           filterable_field_types=filterable_kinds,
                           filter_suggestions=filter_suggestions,
                           available_metrics_for_lb=available_metrics_for_lb,
                           gt_field_options=gt_field_options,
                           dataset_field_options=dataset_field_options,
                           suggested_lb_name=_suggest_free_lb_name(dataset.name))


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


@app.route('/dataset/<int:dataset_id>/update_categories', methods=['POST'])
@login_required
@owner_required(Dataset, 'dataset_id')
def update_dataset_categories(dataset_id):
    """Inline edit of a dataset's category list from /dataset/<id>.

    Accepts a comma-separated list of `Area/Task` paths. Each entry
    is normalised (strip whitespace, collapse internal double-slashes,
    drop blanks); dupes removed. Stored as a CSV in Dataset.category;
    the `category_list` property splits it back to a Python list at
    read time.
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    raw = request.form.get('categories', '')
    out, seen = [], set()
    for chunk in raw.split(','):
        c = '/'.join(seg.strip() for seg in chunk.split('/') if seg.strip())
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    dataset.category = ', '.join(out) if out else None
    db.session.commit()
    flash(f'Categories: {dataset.category or "(none)"}', 'success')
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
    # Cascade-delete attached leaderboards (and their submissions).
    # The same dataset can back many LBs now; dropping the dataset
    # without dropping the LBs would leave them with a broken
    # contract. The confirm prompt in the template surfaces the
    # count so the admin sees what they're about to lose.
    attached_lbs = list(dataset.leaderboards)
    for lb in attached_lbs:
        db.session.delete(lb)
    shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', str(dataset.id)), ignore_errors=True)
    db.session.delete(dataset)
    db.session.commit()
    if attached_lbs:
        flash(
            f"Deleted dataset {dataset.name!r} and {len(attached_lbs)} attached "
            f"leaderboard(s): {', '.join(lb.name for lb in attached_lbs)}",
            'success',
        )
    else:
        flash(f"Deleted dataset {dataset.name!r}.", 'success')
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
    # Depth cells now cache as grayscale PNG; pick colormap at view time.
    # When ?cmap= is supplied (or the file is gray PNG), render via the
    # depth-recolor helper below.
    requested_cmap = (request.args.get('cmap') or '').strip().lower()
    if magic == b'\x89PNG' and requested_cmap:
        return _render_depth_png(path, requested_cmap)
    if magic == b'\x89PNG':
        return send_file(path, mimetype='image/png', max_age=60)
    return send_file(
        path, mimetype='image/jpeg', max_age=60,
    )


def _cf_to_typed_instance(cf, upload_folder):
    """Reconstruct a `benchhub.types.DataType` instance from a
    CustomField row.

    File-backed kinds (image/mask/depth/audio/text/bboxes/json) read
    bytes off the volume at `<upload_folder>/<cf.value_text>`; inline
    kinds (scalar, label) pull the primitive value directly off the
    column and feed it to the type's constructor.
    """
    from benchhub.types import DTYPES
    kind = cf.data_type
    if kind not in DTYPES:
        raise ValueError(f"unknown data_type {kind!r}")
    cls = DTYPES[kind]
    params = cf.get_params()

    # Inline kinds: synthesize directly from the column values.
    if cls.file_ext is None:
        if kind == "scalar":
            if cf.value_float is None:
                raise ValueError(f"scalar CustomField {cf.id} has NULL value_float")
            return cls(float(cf.value_float))
        if kind == "label":
            raw = cf.value_text or ""
            try:
                v = json.loads(raw)
            except (TypeError, ValueError):
                v = raw
            return cls(v)
        # Other inline kinds (none yet) — decode via class.
        return cls.decode((cf.value_text or "").encode("utf-8"), params)

    # File-backed kinds: read bytes off the volume.
    rel = cf.value_text or ""
    if not rel:
        raise FileNotFoundError(f"CustomField {cf.id} has empty value_text")
    abs_path = os.path.join(upload_folder, rel)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"CustomField {cf.id} file missing: {rel}")
    with open(abs_path, "rb") as f:
        blob = f.read()
    return cls.decode(blob, params)


def _can_view_parent(user, parent):
    """Visibility gate shared by the typed-viz dispatch route. Mirrors
    the existing @visibility_required decorator's policy: anonymous +
    non-owner can see public/unlisted; private requires owner or
    admin."""
    if parent is None:
        return False
    vis = getattr(parent, "visibility", "public") or "public"
    if vis in ("public", "unlisted"):
        return True
    if vis == "private":
        if user is None:
            return False
        owner_id = getattr(parent, "owner_user_id", None)
        return (owner_id is not None and owner_id == user.id) or is_admin(user)
    return False


@app.route('/api/viz/<int:cf_id>')
def api_viz(cf_id):
    """Unified renderer: load the CustomField, reconstruct the typed
    DataType instance via DTYPES[cf.data_type].decode(...), call
    `.visualize(**query_params)`, and return a Response with the
    type's `viz_mime`. Visibility-gated via the parent dataset /
    submission / leaderboard.
    """
    cf = CustomField.query.get_or_404(cf_id)

    # Resolve the parent row for the visibility check.
    parent = None
    if cf.sample_id:
        sample = Sample.query.get(cf.sample_id)
        parent = sample.dataset if sample else None
    elif cf.submission_id:
        sub = Submission.query.get(cf.submission_id)
        parent = sub.leaderboard if sub else None
    elif cf.leaderboard_id:
        parent = Leaderboard.query.get(cf.leaderboard_id)
    if not _can_view_parent(g.current_user, parent):
        abort(404)

    try:
        inst = _cf_to_typed_instance(cf, app.config['UPLOAD_FOLDER'])
    except FileNotFoundError:
        abort(404)
    except ValueError:
        abort(400)

    # Pass query-string args as renderer opts. Subclasses that don't
    # know a given opt just ignore it via **_; we don't want a stray
    # ?foo=bar to 500 the response, so swallow TypeError as a fallback.
    opts = {k: v for k, v in request.args.items()}
    try:
        body, mime = inst.visualize(**opts)
    except TypeError:
        body, mime = inst.visualize()
    return Response(body, content_type=mime)


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
        # Forward any ?cmap=<name> param (used for depth colormap
        # selection) through the redirect — Flask doesn't carry query
        # args across `redirect()` automatically.
        url_kwargs = {
            'lb_id': custom_field.leaderboard_id,
            'col': col,
            'sample_name': custom_field.sample_name,
        }
        if request.args.get('cmap'):
            url_kwargs['cmap'] = request.args['cmap']
        return redirect(url_for('serve_gt_viz', **url_kwargs))

    if custom_field.data_type == 'depth':
        return serve_depth_image(custom_field.value_text)

    if custom_field.data_type not in ('image', 'mask', 'audio'):
        return "Not an image/mask/depth/audio field", 400

    # value_text contains the relative path from uploads folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], custom_field.value_text)

    if not os.path.exists(image_path):
        return "Image not found", 404

    if custom_field.data_type == 'mask':
        # `?raw=1` serves the underlying class-index PNG unmodified so
        # the frontend can read pixel values via a Canvas getImageData
        # for hover tooltips. The default still applies the palette
        # for visual rendering.
        if request.args.get('raw') in ('1', 'true'):
            # Preview-only masks ship a `.classid.png` sidecar — a
            # single-channel mode-I;16 PNG with class ids. Use that if
            # present (the display .jpg is palette-applied and would
            # give wrong hover values).
            stem, _ = os.path.splitext(image_path)
            sidecar = stem + '.classid.png'
            if os.path.isfile(sidecar):
                return send_file(sidecar)
            return send_file(image_path)
        # BH masks land on disk as raw class-index PNGs (values like
        # 0/1 for binary masks, 0/1/2/4 for multi-class). Rendered as
        # grayscale those are visually black. Apply the deterministic-
        # hue LUT (same one used for HF GT mask thumbs in
        # _cache_gt_image_thumb) at serve time so each class index
        # picks up a distinct colour.
        return _render_mask_png_with_palette(image_path)

    return send_file(image_path)


def _render_mask_png_with_palette(mask_path):
    """Recolour a class-index mask PNG using the deterministic-hue LUT.
    Returns a Flask response with the recoloured PNG bytes."""
    from PIL import Image
    import io
    try:
        with Image.open(mask_path) as src:
            if src.mode in ('P', 'L'):
                # P-mode: np.asarray returns the index array directly,
                # not the RGB-converted view (which would lose class
                # info). L-mode is already a single-channel index map.
                arr = np.asarray(src, dtype=np.int32)
            elif src.mode in ('I', 'I;16', 'I;16B', 'I;16L'):
                arr = np.asarray(src, dtype=np.int32)
            elif src.mode in ('RGB', 'RGBA'):
                # Already RGB — probably user-pre-coloured. Just pass it through.
                return send_file(mask_path)
            else:
                return send_file(mask_path)
    except Exception:
        abort(404)
    lut = ((np.arange(256, dtype=np.int32) * 31 + 17) % 256).astype(np.uint8)
    idx = arr & 0xFF
    r = lut[idx]
    g = lut[(idx + 85) & 0xFF]
    b = lut[(idx + 170) & 0xFF]
    # Background (class 0) stays black for visual contrast.
    bg = (arr == 0)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[bg] = 0
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    return send_file(buf, mimetype='image/png', max_age=300)

@app.route('/api/custom_field_depth_data/<int:field_id>')
def serve_custom_field_depth_data(field_id):
    """Serve raw depth data for a custom field as JSON."""
    custom_field = CustomField.query.get_or_404(field_id)
    
    if custom_field.data_type != 'depth':
        return abort(400, description="Not a depth field")
        
    return serve_depth_data(custom_field.value_text)

@app.route('/api/custom_field_depth_meta/<int:field_id>')
def serve_custom_field_depth_meta(field_id):
    """Return just {min, max, shape} for a depth field. The full
    array endpoint loads ~MB per request; this one is for the
    inline colorbar labels in the comparison + dataset views,
    where rendering 50 cells × full arrays is wasteful.

    Preview-only datasets store depth as a turbo-colormapped JPG.
    A sidecar .meta.json (written by manifest.import_typed_dataset)
    carries the original min/max so colorbar labels stay metric.
    Without a sidecar we return normalised [0, 1] and flag it."""
    custom_field = CustomField.query.get_or_404(field_id)
    if custom_field.data_type != 'depth':
        return abort(400, description="Not a depth field")
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], custom_field.value_text or '')
    if not os.path.exists(full_path):
        return abort(404, description="File not found")
    ext = full_path.rsplit('.', 1)[-1].lower()
    if ext in ('jpg', 'jpeg', 'png'):
        meta_path = full_path.rsplit('.', 1)[0] + '.meta.json'
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                return jsonify({
                    'min': float(m.get('min', 0.0)),
                    'max': float(m.get('max', 1.0)),
                    'shape': m.get('shape') or [],
                    'normalized': False,
                })
            except Exception:
                pass
        # No sidecar — preview is normalised [0, 1]; tell the client.
        from PIL import Image as _PIL
        try:
            with _PIL.open(full_path) as im:
                shp = [im.height, im.width]
        except Exception:
            shp = []
        return jsonify({'min': 0.0, 'max': 1.0, 'shape': shp, 'normalized': True})
    try:
        with np.load(full_path) as data:
            arr = None
            for key in data.files:
                if len(data[key].shape) == 2:
                    arr = data[key]; break
            if arr is None and data.files:
                arr = data[data.files[0]]
            if arr is None:
                return abort(500, description="Empty npz")
            return jsonify({
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'shape': list(arr.shape),
                'normalized': False,
            })
    except Exception as e:
        return abort(500, description=str(e))


@app.route('/api/custom_field_json/<int:field_id>')
def serve_custom_field_json(field_id):
    """Serve JSON data for a custom field.

    Two storage shapes are accepted, in order:
      1. Inline JSON content in `value_text` (what the typed-manifest
         importer writes since commit 4f15907 so the comparison
         text/json scrollbox renders content directly).
      2. Relative path to a `.json` file under UPLOAD_FOLDER
         (legacy + back-compat).
    """
    custom_field = CustomField.query.get_or_404(field_id)
    # coco_detections is a structured JSON kind too — the wire format
    # is a JSON list of detection records, so the same endpoint
    # serves it.
    if custom_field.data_type not in ('json', 'coco_detections'):
        return abort(400, description="Not a JSON field")

    raw = custom_field.value_text or ''
    if not raw:
        return abort(404, description="No JSON content")

    # Try inline first — value_text holds the JSON content directly.
    stripped = raw.lstrip()
    if stripped[:1] in ('{', '[', '"'):
        try:
            return jsonify(json.loads(raw))
        except (TypeError, ValueError):
            pass  # fall through to path lookup

    # Fall back to the legacy path-in-value_text shape.
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], raw)
    if not os.path.exists(json_path):
        return abort(404, description="JSON file not found")
    try:
        with open(json_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return abort(500, description=f"Error reading JSON file: {str(e)}")


@app.route('/sample/<int:sample_id>/download')
def download_sample(sample_id):
    sample = Sample.query.get_or_404(sample_id)
    # Optional submission IDs to include in the zip
    submission_ids = request.args.getlist('submission_id', type=int)
    # Optional leaderboard id → pull the full-resolution MATERIALISED
    # bytes for file-backed fields instead of the preview tier. Falls
    # back to the preview when a field wasn't materialised.
    lb_id = request.args.get('lb', type=int)
    _materialized = bool(lb_id)
    if lb_id:
        from benchhub.lb_materialize import materialized_or_preview_path

    # The typed registry tells us how each kind is persisted:
    # `file_ext is None` → inline value (stash in value_float /
    # value_text); otherwise the bytes live on disk and value_text
    # is the relative path under UPLOAD_FOLDER. Driving the export
    # from the same source as the importer means new kinds get
    # bundled automatically without touching this function.
    from benchhub.types import DTYPES as _DTYPES
    _INLINE_EXT_BY_KIND = {
        'scalar': '.txt',
        'label': '.txt',
        'label_list': '.json',
    }

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # Tags travel as a flat CSV file alongside the per-field
        # folders so users get the same layout they uploaded with.
        if sample.tags:
            zf.writestr(f'ground_truth/tags/{sample.name}.txt', sample.tags)

        for cf in sample.custom_fields:
            kind = cf.data_type
            # Metric outputs are bookkeeping (per-sample scores written
            # back by the engine), not user-facing data — skip them.
            if kind == 'metric' or (cf.name or '').startswith('lm_'):
                continue

            type_cls = _DTYPES.get(kind)
            is_inline = (type_cls is not None and type_cls.file_ext is None)

            # Legacy `histogram` rows have no entry in DTYPES; round-trip
            # the cached JSON back to .npz to match the original layout.
            if kind == 'histogram':
                try:
                    h = json.loads(cf.value_text or '{}')
                    buf = io.BytesIO()
                    np.savez_compressed(buf,
                                        bins=np.array(h.get('bins', [])),
                                        counts=np.array(h.get('counts', [])))
                    zf.writestr(f'ground_truth/{cf.name}/{sample.name}.npz',
                                buf.getvalue())
                except Exception:
                    pass
                continue

            if is_inline:
                ext = _INLINE_EXT_BY_KIND.get(kind, '.txt')
                if kind == 'scalar':
                    payload = '' if cf.value_float is None else str(cf.value_float)
                else:
                    payload = cf.value_text or ''
                zf.writestr(f'ground_truth/{cf.name}/{sample.name}{ext}', payload)
                continue

            # File-backed: copy the on-disk file into the bundle.
            # Depth files carry a `_<W>x<H>` suffix the importer needs
            # on round-trip, so keep the original filename.
            rel = cf.value_text or ''
            if _materialized and rel:
                # Prefer the LB's full-resolution materialised copy.
                rel = materialized_or_preview_path(
                    app.config['UPLOAD_FOLDER'], lb_id, cf.name, sample.name, rel,
                )
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], rel)
            if rel and os.path.exists(src_path):
                arc_filename = os.path.basename(rel)
                zf.write(src_path, f'ground_truth/{cf.name}/{arc_filename}')
                # Masks: include the raw class-index sidecar (the display
                # file is palette-RGB and only round-trips as an Image).
                if kind == 'mask':
                    sidecar = os.path.splitext(src_path)[0] + '.classid.png'
                    if os.path.isfile(sidecar):
                        zf.write(sidecar,
                                 f'ground_truth/{cf.name}/'
                                 f'{os.path.splitext(arc_filename)[0]}.classid.png')


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
    suffix = '_materialized' if _materialized else ''
    return send_file(memory_file,
                     download_name=f'{sample.name}{suffix}_data.zip',
                     as_attachment=True)

    
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
    # process_dataset_zip below creates a Dataset row with the default
    # visibility (private for regular users, public for admins). Charge
    # the corresponding bucket.
    api_default_vis = 'public' if is_admin(g.current_user) else 'private'
    ok, msg = check_quota(
        g.current_user, kind='dataset_create',
        incoming_bytes=incoming, visibility=api_default_vis,
    )
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


def _matplotlib_lut(name):
    """256×3 uint8 LUT for a named matplotlib colormap. Cached in
    `_MPL_LUT_CACHE` so the second hit is a dict lookup. Falls back
    to the precomputed turbo LUT for unknown names."""
    try:
        global _MPL_LUT_CACHE
    except NameError:
        pass
    if not hasattr(_matplotlib_lut, '_cache'):
        _matplotlib_lut._cache = {}
    cache = _matplotlib_lut._cache
    if name in cache:
        return cache[name]
    try:
        from matplotlib import colormaps as _cmaps
        cmap = _cmaps[name]
    except Exception:
        cache[name] = _TURBO_LUT
        return _TURBO_LUT
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)
    cache[name] = lut
    return lut


def _render_depth_png(gray_path, cmap):
    """Load a grayscale depth PNG from `gray_path`, apply the requested
    colormap (or normal-map projection) on the fly, and return a Flask
    response with the recolored PNG bytes. Used by serve_gt_viz when
    the request asks for a specific cmap."""
    from PIL import Image
    try:
        with Image.open(gray_path) as src:
            arr = np.asarray(src.convert('L'), dtype=np.uint8)
    except Exception:
        abort(404)
    if cmap == 'gray':
        # Echo the grayscale PNG verbatim — caller asked for raw depth.
        return send_file(gray_path, mimetype='image/png', max_age=60)
    if cmap == 'normal':
        # Surface normals from height. The naive `np.gradient * 8` on a
        # 0..1 normalized depth map gives near-flat normals (mostly
        # (0,0,1) → purple); we want enough z-axis tilt that surface
        # orientation is actually visible.
        #
        # Approach:
        # 1. Light Gaussian smooth to suppress single-pixel noise.
        # 2. 3x3 Sobel for the gradient (better than np.gradient for
        #    visualizing surface orientation — central-difference is
        #    too soft on sharp edges).
        # 3. Auto-scale the gradient so the 98th-percentile of the
        #    magnitude lands at ~1.0. Keeps the normal map well-
        #    saturated regardless of how "rough" the depth is.
        from scipy.ndimage import gaussian_filter
        z = arr.astype(np.float32) / 255.0
        z = gaussian_filter(z, sigma=1.0)
        # Sobel kernels via convolution. scipy.signal.convolve2d would
        # add another import; we use simple np slicing instead.
        gx = np.zeros_like(z)
        gy = np.zeros_like(z)
        gx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) * 0.5
        gy[1:-1, :] = (z[2:, :] - z[:-2, :]) * 0.5
        # Auto-scale to land the 98th-percentile magnitude near 1.
        mag = np.sqrt(gx * gx + gy * gy)
        p98 = float(np.percentile(mag, 98)) or 1e-6
        scale = 1.0 / max(p98, 1e-3)
        nx = -gx * scale
        ny = gy * scale  # +Y points UP in tangent-space convention
        nz = np.ones_like(z) * 0.5  # base flatness; clamp later
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        length[length == 0] = 1.0
        nx /= length; ny /= length; nz /= length
        # Map [-1, 1] → [0, 255] in standard tangent-space normal-map
        # encoding (X=right, Y=up, Z=out → RGB).
        rgb = np.stack([
            np.clip(((nx + 1.0) * 127.5), 0, 255).astype(np.uint8),
            np.clip(((ny + 1.0) * 127.5), 0, 255).astype(np.uint8),
            np.clip(((nz + 1.0) * 127.5), 0, 255).astype(np.uint8),
        ], axis=-1)
    else:
        lut = _matplotlib_lut(cmap)
        rgb = lut[arr]
    out = Image.fromarray(rgb)
    buf = io.BytesIO()
    out.save(buf, format='PNG', optimize=False)
    buf.seek(0)
    resp = send_file(buf, mimetype='image/png', max_age=60)
    return resp


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
            if kind == 'depth':
                # Store depth as 8-bit GRAYSCALE PNG. Users pick the
                # colormap (or normal-map projection) at view time via
                # /api/gt_depth_render?cmap=... — the cache holds the
                # canonical normalized signal so future viewers don't
                # have to round-trip a colormapped JPEG.
                a = arr.astype(np.float32)
                if a.ndim == 3 and a.shape[-1] in (3, 4):
                    a = a[..., 0]  # squeeze single-channel out of RGB-ish wrappers
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
                gray = (norm * 255).astype(np.uint8)
                img = Image.fromarray(gray, mode='L')
                # PNG keeps the gray exact (lossless); cheap to recolour
                # client-side or via /api/gt_depth_render. Skip JPEG path.
                img.save(dest_path, format='PNG', optimize=True)
                return True
            if arr.ndim == 2 and kind != 'mask':
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
        # hitting pixelation.
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
                CustomField.data_type == 'metric'
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
                        CustomField.data_type == 'metric',
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
            if cf.data_type == 'metric':
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


# Stable per-card accent palette. The list / leaderboard catalogs
# used to be a wall of identical cream rows — eyes glazed over once
# you scrolled past five. Each card now gets one color from this
# palette, picked by hashing its name so the assignment is the same
# across page loads (and stable when the row is filtered in/out).
# Palette colors are chosen to look at home on the warm-cream theme
# (`#fcf9f4` body, `#f5efe2` cards) and to remain readable as a
# left-edge stripe + a very faint background tint.
_CARD_ACCENT_PALETTE = (
    '#7c3aed',  # violet (brand primary)
    '#d97706',  # amber
    '#059669',  # emerald
    '#dc2626',  # red
    '#0891b2',  # cyan
    '#db2777',  # pink
    '#65a30d',  # lime
    '#9333ea',  # purple
    '#2563eb',  # blue
    '#ea580c',  # orange
)


def _accent_color_for(name) -> str:
    """Return a stable accent color for a string. Hash → index into
    `_CARD_ACCENT_PALETTE`. Always returns a non-empty hex — empty /
    None input lands on the first slot, which is fine for the
    "Uncategorized" / placeholder rows that share that fallback.
    """
    if not name:
        return _CARD_ACCENT_PALETTE[0]
    import hashlib
    h = hashlib.md5(str(name).encode('utf-8', 'replace')).digest()
    return _CARD_ACCENT_PALETTE[h[0] % len(_CARD_ACCENT_PALETTE)]


def _accent_bg_for(name) -> str:
    """Mix the accent for `name` with the warm-cream card background
    so the whole card can take the accent tint without sacrificing
    text contrast. Returns an `rgb(...)` string ready for inline
    style.

    Mixed at ~14% accent + 86% cream — strong enough that the
    catalog stops looking like one beige wall, soft enough that the
    body text (dark warm-brown) stays comfortably readable.
    """
    accent = _accent_color_for(name)
    # `#f5efe2` is the card base in the warm-cream theme.
    base = (0xf5, 0xef, 0xe2)
    try:
        ar = int(accent[1:3], 16)
        ag = int(accent[3:5], 16)
        ab = int(accent[5:7], 16)
    except (ValueError, IndexError):
        return f"rgb({base[0]},{base[1]},{base[2]})"
    mix = 0.14
    r = int(round(ar * mix + base[0] * (1 - mix)))
    g = int(round(ag * mix + base[1] * (1 - mix)))
    b = int(round(ab * mix + base[2] * (1 - mix)))
    return f"rgb({r},{g},{b})"


@app.template_filter('label_name')
def label_name_filter(value, names):
    """Map a label index to its class name via the field's `names`
    vocab. `value` is an int (class index); `names` the vocab list.
    Falls back to the raw value when there's no vocab, the value isn't a
    valid index, or it's already a string class name."""
    try:
        if names and isinstance(value, bool) is False and isinstance(value, int) \
                and 0 <= value < len(names):
            return names[value]
    except Exception:
        pass
    return value


@app.template_filter('label_display')
def label_display_filter(value, names):
    """Like `label_name`, but keeps the index too: `cat (3)`. Falls back
    to the bare value when there's no vocab / it's not a valid index."""
    try:
        if names and isinstance(value, bool) is False and isinstance(value, int) \
                and 0 <= value < len(names):
            return f"{names[value]} ({value})"
    except Exception:
        pass
    return value


@app.template_filter('accent_color')
def accent_color_filter(s):
    return _accent_color_for(s)


@app.template_filter('accent_bg')
def accent_bg_filter(s):
    return _accent_bg_for(s)


@app.template_filter('display_name')
def display_name_filter(s):
    """Render a dataset name for display: strip the owner/org prefix
    (everything before the last '__'), replace both underscores AND
    hyphens with spaces, and title-case each word. The DB-canonical
    name stays `<owner>__<slug>`; this is purely cosmetic.

    Examples:
      'tanganke__eurosat'                    → 'Eurosat'
      'COIN-Research-Group__sawhill-dataset' → 'Sawhill Dataset'
      'Lin-Chen__MMStar'                     → 'Mmstar'
      'gksriharsha__chitralekha'             → 'Chitralekha'
      'wahlinski__handwritten_cross-outs'    → 'Handwritten Cross Outs'
      'cifar10'                              → 'Cifar10'
    """
    if not s:
        return s
    raw = s.rsplit('__', 1)[-1] if '__' in s else s
    return raw.replace('_', ' ').replace('-', ' ').title()

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

                if 'category' not in dataset_columns:
                    print("Migrating DB: Adding 'category' to 'dataset' table...")
                    try:
                        cursor.execute("ALTER TABLE dataset ADD COLUMN category VARCHAR(120) DEFAULT NULL")
                        conn.commit()
                        print("Migration successful: Added 'category' column to dataset.")
                    except Exception as e:
                        print(f"Migration error (dataset.category): {e}")

                if 'source_url' not in dataset_columns:
                    print("Migrating DB: Adding 'source_url' to 'dataset' table...")
                    try:
                        cursor.execute("ALTER TABLE dataset ADD COLUMN source_url TEXT DEFAULT NULL")
                        conn.commit()
                        print("Migration successful: Added 'source_url' column to dataset.")
                    except Exception as e:
                        print(f"Migration error (dataset.source_url): {e}")

                # --- Relax global UNIQUE on GlobalMetric.name + GlobalVisualization.name ---
                # Private rows now coexist with the same name across users; only
                # visibility='public' rows must have a globally unique name.
                # SQLite can't `ALTER TABLE DROP CONSTRAINT`, so for each table:
                #   1. detect the auto-created UNIQUE index on `name`
                #   2. rebuild the table (CREATE ... AS SELECT) if found
                #   3. add the new composite + partial-public indexes
                for _tbl in ('global_metric', 'global_visualization'):
                    try:
                        cursor.execute(f"PRAGMA index_list({_tbl})")
                        indexes = cursor.fetchall()
                        auto_unique_on_name = None
                        for _i in indexes:
                            # row shape: (seq, name, unique, origin, partial)
                            iname = _i[1]; uniq = _i[2]; origin = _i[3] if len(_i) > 3 else 'c'
                            partial = _i[4] if len(_i) > 4 else 0
                            if not uniq:
                                continue
                            # Only target the implicit SQLAlchemy `unique=True`
                            # autoindex (origin='u', non-partial). My own
                            # `CREATE UNIQUE INDEX … WHERE visibility='public'`
                            # has origin='c' / partial=1 — must NOT rebuild
                            # the table on every boot.
                            if origin != 'u' or partial:
                                continue
                            cursor.execute(f"PRAGMA index_info({iname})")
                            cols = [r[2] for r in cursor.fetchall()]
                            if cols == ['name']:
                                auto_unique_on_name = iname
                                break
                        if auto_unique_on_name:
                            # Rebuild. Pull column list from PRAGMA so we don't
                            # have to hardcode every column in the model.
                            cursor.execute(f"PRAGMA table_info({_tbl})")
                            col_rows = cursor.fetchall()
                            col_names = [r[1] for r in col_rows]
                            cols_csv = ', '.join(col_names)
                            new_tbl = f"{_tbl}__rebuild"
                            cursor.execute(f"DROP TABLE IF EXISTS {new_tbl}")
                            # Recreate via CTAS then swap; UNIQUE constraints get lost
                            # but the PK INTEGER PRIMARY KEY does too — so we add it
                            # back explicitly via a temp rename + structured rebuild.
                            cursor.execute(
                                f"CREATE TABLE {new_tbl} AS SELECT {cols_csv} FROM {_tbl} WHERE 0"
                            )
                            # Insert preserving ids; SQLite keeps NOT NULL types
                            # from AS SELECT but loses the PK constraint, so
                            # rebuild with the original schema sans UNIQUE.
                            cursor.execute(f"DROP TABLE {new_tbl}")
                            # Read original schema:
                            cursor.execute(
                                f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                                (_tbl,),
                            )
                            ddl_row = cursor.fetchone()
                            if ddl_row and ddl_row[0]:
                                ddl = ddl_row[0]
                                # SQLAlchemy emits the UNIQUE as a table-level
                                # constraint by default (`UNIQUE (name)`) — not
                                # inline on the column. Strip both forms.
                                import re as _re
                                # Inline form (rare).
                                ddl_new = _re.sub(
                                    r'("?name"?\s+VARCHAR\(\d+\)\s+NOT NULL)\s+UNIQUE',
                                    r'\1',
                                    ddl,
                                )
                                # Table-level form: `, UNIQUE ("name")` or
                                # `UNIQUE (name)` somewhere in the body.
                                ddl_new = _re.sub(
                                    r',\s*UNIQUE\s*\(\s*"?name"?\s*\)',
                                    '',
                                    ddl_new,
                                )
                                ddl_new = _re.sub(
                                    r'UNIQUE\s*\(\s*"?name"?\s*\)\s*,?',
                                    '',
                                    ddl_new,
                                )
                                # SQLAlchemy emits the table name quoted
                                # (`CREATE TABLE "global_metric"`). Cover
                                # both quoted and unquoted variants.
                                ddl_new = _re.sub(
                                    rf'CREATE TABLE\s+"?{_tbl}"?',
                                    f'CREATE TABLE "{new_tbl}"',
                                    ddl_new, count=1,
                                )
                                cursor.execute(ddl_new)
                                cursor.execute(
                                    f"INSERT INTO {new_tbl} ({cols_csv}) SELECT {cols_csv} FROM {_tbl}"
                                )
                                cursor.execute(f"DROP TABLE {_tbl}")
                                cursor.execute(f"ALTER TABLE {new_tbl} RENAME TO {_tbl}")
                                print(f"Migration: dropped column-level UNIQUE(name) on {_tbl}")
                        # Add the new indexes (idempotent via IF NOT EXISTS).
                        cursor.execute(
                            f"CREATE UNIQUE INDEX IF NOT EXISTS "
                            f"uq_{_tbl}_name_per_owner ON {_tbl} (owner_user_id, name)"
                        )
                        cursor.execute(
                            f"CREATE UNIQUE INDEX IF NOT EXISTS "
                            f"uq_{_tbl}_name_public ON {_tbl} (name) WHERE visibility = 'public'"
                        )
                        conn.commit()
                    except Exception as e:
                        print(f"Migration error ({_tbl} unique relax): {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                # --- GlobalMetric.input_kinds / GlobalVisualization.input_kinds ---
                for _tbl in ('global_metric', 'global_visualization'):
                    try:
                        cursor.execute(f"PRAGMA table_info({_tbl})")
                        cols = [r[1] for r in cursor.fetchall()]
                        if 'input_kinds' not in cols:
                            cursor.execute(
                                f"ALTER TABLE {_tbl} ADD COLUMN input_kinds TEXT DEFAULT NULL"
                            )
                            conn.commit()
                            print(f"Migration: added {_tbl}.input_kinds")
                    except Exception as e:
                        print(f"Migration error ({_tbl}.input_kinds): {e}")

                # --- Dataset.import_status + import_progress_json + import_error + import_task_id ---
                for _col, _spec in (
                    ('import_status',        "TEXT DEFAULT 'ready'"),
                    ('import_progress_json', 'TEXT DEFAULT NULL'),
                    ('import_error',         'TEXT DEFAULT NULL'),
                    ('import_task_id',       'TEXT DEFAULT NULL'),
                ):
                    try:
                        cursor.execute("PRAGMA table_info(dataset)")
                        cols = [r[1] for r in cursor.fetchall()]
                        if _col not in cols:
                            cursor.execute(
                                f"ALTER TABLE dataset ADD COLUMN {_col} {_spec}"
                            )
                            conn.commit()
                            print(f"Migration: added dataset.{_col}")
                    except Exception as e:
                        print(f"Migration error (dataset.{_col}): {e}")

                # --- Leaderboard.field_roles_json (LB-level role overrides) ---
                try:
                    cursor.execute("PRAGMA table_info(leaderboard)")
                    cols = [r[1] for r in cursor.fetchall()]
                    if 'field_roles_json' not in cols:
                        cursor.execute(
                            "ALTER TABLE leaderboard ADD COLUMN field_roles_json TEXT DEFAULT NULL"
                        )
                        conn.commit()
                        print("Migration: added leaderboard.field_roles_json")
                except Exception as e:
                    print(f"Migration error (leaderboard.field_roles_json): {e}")

                # --- GlobalMetric.input_roles (parallel role-per-arg array) ---
                try:
                    cursor.execute("PRAGMA table_info(global_metric)")
                    cols = [r[1] for r in cursor.fetchall()]
                    if 'input_roles' not in cols:
                        cursor.execute(
                            "ALTER TABLE global_metric ADD COLUMN input_roles TEXT DEFAULT NULL"
                        )
                        conn.commit()
                        print("Migration: added global_metric.input_roles")
                except Exception as e:
                    print(f"Migration error (global_metric.input_roles): {e}")

                # --- Lower per-user storage quota 200 MB → 50 MB ---
                # Existing users were created with the old 200 MB default;
                # the new default is 50 MB. Only touch rows that still
                # carry the old default value — preserves any admin-set
                # custom caps and the NULL ("unlimited") system user.
                try:
                    cursor.execute(
                        "UPDATE user SET quota_max_storage_bytes = ? "
                        "WHERE quota_max_storage_bytes = ?",
                        (50 * 1024 * 1024, 200 * 1024 * 1024),
                    )
                    if cursor.rowcount:
                        print(
                            f"Migrating DB: lowered storage cap to 50 MB "
                            f"on {cursor.rowcount} user(s)."
                        )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (user storage cap 50 MB): {e}")

                # --- DatasetField table (Phase B+: dataset-level schema) ---
                # `db.create_all()` runs in every gunicorn worker on boot;
                # without an explicit IF NOT EXISTS guarded migration here,
                # the workers race each other and the loser gets
                # `OperationalError: table dataset_field already exists`,
                # killing the boot. Doing it here under the same try/except
                # as the other tables keeps the prod restart safe.
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS dataset_field ("
                        "  id INTEGER PRIMARY KEY,"
                        "  dataset_id INTEGER NOT NULL REFERENCES dataset(id),"
                        "  name VARCHAR(100) NOT NULL,"
                        "  kind VARCHAR(20) NOT NULL,"
                        "  params TEXT,"
                        "  role VARCHAR(10) NOT NULL,"
                        "  UNIQUE (dataset_id, name)"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_dataset_field_dataset_id "
                        "ON dataset_field (dataset_id)"
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (dataset_field): {e}")

                # --- FeatureRequest table ---
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS feature_request ("
                        "  id INTEGER PRIMARY KEY,"
                        "  user_id INTEGER NOT NULL REFERENCES user(id),"
                        "  kind VARCHAR(30) NOT NULL DEFAULT 'feature',"
                        "  title VARCHAR(200) NOT NULL,"
                        "  description TEXT,"
                        "  status VARCHAR(20) NOT NULL DEFAULT 'open',"
                        "  created_at DATETIME NOT NULL,"
                        "  admin_note TEXT"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_feature_request_user_id "
                        "ON feature_request (user_id)"
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (feature_request): {e}")

                # --- LbTemplate (admin-editable task templates) ---
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS lb_template ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "slug VARCHAR(60) NOT NULL UNIQUE, "
                        "label VARCHAR(120) NOT NULL, "
                        "description TEXT, "
                        "required_kinds_json TEXT NOT NULL DEFAULT '[]', "
                        "metric_names_json TEXT NOT NULL DEFAULT '[]', "
                        "visualization_names_json TEXT NOT NULL DEFAULT '[]', "
                        "enabled BOOLEAN NOT NULL DEFAULT 1, "
                        "sort_order INTEGER NOT NULL DEFAULT 0"
                        ")"
                    )
                    conn.commit()
                    # Seed from LB_TEMPLATES on first boot only — admin
                    # edits in the DB win on later boots.
                    cursor.execute("SELECT COUNT(*) FROM lb_template")
                    if cursor.fetchone()[0] == 0:
                        print("Migrating DB: Seeding lb_template from LB_TEMPLATES dict...")
                        for i, (slug, tpl) in enumerate(LB_TEMPLATES.items()):
                            cursor.execute(
                                "INSERT INTO lb_template "
                                "(slug, label, description, "
                                " required_kinds_json, metric_names_json, "
                                " visualization_names_json, sort_order) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (slug, tpl['label'], tpl.get('description', ''),
                                 json.dumps(tpl.get('required_kinds', [])),
                                 json.dumps(tpl.get('metric_names', [])),
                                 json.dumps(tpl.get('visualization_names', [])),
                                 i),
                            )
                        conn.commit()
                except Exception as e:
                    print(f"Migration error (lb_template): {e}")

                # --- Dataset.card_description column + backfill ---
                try:
                    cursor.execute("PRAGMA table_info(dataset)")
                    cols = [row[1] for row in cursor.fetchall()]
                    if 'card_description' not in cols:
                        print("Migrating DB: Adding 'card_description' to 'dataset'...")
                        cursor.execute(
                            "ALTER TABLE dataset ADD COLUMN card_description TEXT"
                        )
                        conn.commit()
                    # Idempotent retroactive backfill — fetch the HF
                    # README for every HF row whose card_description is
                    # still NULL. Run only when we find at least one
                    # candidate so this doesn't re-hit HF on every boot.
                    cursor.execute(
                        "SELECT id, source_url FROM dataset "
                        "WHERE card_description IS NULL "
                        "AND source_kind = 'hf' "
                        "AND source_url LIKE 'https://huggingface.co/datasets/%'"
                    )
                    targets = cursor.fetchall()
                    if targets:
                        from benchhub.hf_search import fetch_hf_card_description
                        print(f"Migrating DB: Backfilling card_description "
                              f"for {len(targets)} HF dataset(s)...")
                        for ds_id, url in targets:
                            repo = (url or '').split(
                                'huggingface.co/datasets/', 1
                            )[-1].strip('/')
                            if not repo:
                                continue
                            desc = fetch_hf_card_description(repo)
                            if desc:
                                cursor.execute(
                                    "UPDATE dataset SET card_description = ? "
                                    "WHERE id = ?",
                                    (desc, ds_id),
                                )
                        conn.commit()
                except Exception as e:
                    print(f"Migration error (card_description): {e}")

                # --- Leaderboard.colab_gist_id column ---
                try:
                    cursor.execute("PRAGMA table_info(leaderboard)")
                    cols = [row[1] for row in cursor.fetchall()]
                    if 'colab_gist_id' not in cols:
                        print("Migrating DB: Adding 'colab_gist_id' to 'leaderboard'...")
                        cursor.execute(
                            "ALTER TABLE leaderboard ADD COLUMN colab_gist_id VARCHAR(64)"
                        )
                        conn.commit()
                except Exception as e:
                    print(f"Migration error (colab_gist_id): {e}")

                # --- SubmissionToken table ---
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS submission_token ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "user_id INTEGER NOT NULL REFERENCES user(id), "
                        "leaderboard_id INTEGER NOT NULL REFERENCES leaderboard(id), "
                        "token VARCHAR(64) NOT NULL UNIQUE, "
                        "expires_at DATETIME NOT NULL, "
                        "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                        "used_count INTEGER NOT NULL DEFAULT 0, "
                        "revoked BOOLEAN NOT NULL DEFAULT 0"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_submission_token_user_id "
                        "ON submission_token (user_id)"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_submission_token_lb_id "
                        "ON submission_token (leaderboard_id)"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_submission_token_token "
                        "ON submission_token (token)"
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (submission_token): {e}")

                # --- Dataset.preview_only column ---
                try:
                    cursor.execute("PRAGMA table_info(dataset)")
                    cols = [row[1] for row in cursor.fetchall()]
                    if 'preview_only' not in cols:
                        print("Migrating DB: Adding 'preview_only' to 'dataset'...")
                        cursor.execute(
                            "ALTER TABLE dataset ADD COLUMN preview_only "
                            "BOOLEAN NOT NULL DEFAULT 0"
                        )
                        conn.commit()
                except Exception as e:
                    print(f"Migration error (dataset.preview_only): {e}")

                # --- LeaderboardMaterialization table ---
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS leaderboard_materialization ("
                        "  id INTEGER PRIMARY KEY,"
                        "  leaderboard_id INTEGER NOT NULL UNIQUE "
                        "    REFERENCES leaderboard(id),"
                        "  sample_cap INTEGER NOT NULL,"
                        "  sampling VARCHAR(20) NOT NULL DEFAULT 'random',"
                        "  sampling_seed INTEGER NOT NULL DEFAULT 42,"
                        "  stratify_field VARCHAR(100),"
                        "  materialized_at DATETIME,"
                        "  storage_bytes BIGINT,"
                        "  status VARCHAR(20) NOT NULL DEFAULT 'pending',"
                        "  error_message TEXT,"
                        "  progress_json TEXT"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_leaderboard_materialization_leaderboard_id "
                        "ON leaderboard_materialization (leaderboard_id)"
                    )
                    # progress_json was added after the initial table —
                    # back-fill on existing rows for the ALTER path.
                    cursor.execute("PRAGMA table_info(leaderboard_materialization)")
                    cols = [r[1] for r in cursor.fetchall()]
                    if 'progress_json' not in cols:
                        cursor.execute(
                            "ALTER TABLE leaderboard_materialization "
                            "ADD COLUMN progress_json TEXT"
                        )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (leaderboard_materialization): {e}")

                # --- Bump default user storage quota to 10 GB ---
                try:
                    # Existing rows still at the old 50 MB default get
                    # bumped to the new free-tier quota (10 GB). Admins
                    # are unaffected (they bypass via is_admin), and
                    # any user already on a custom cap > old default
                    # keeps their custom value.
                    OLD = 50 * 1024 * 1024
                    NEW = 10 * 1024 ** 3
                    cursor.execute(
                        'UPDATE user SET quota_max_storage_bytes = ? '
                        'WHERE quota_max_storage_bytes <= ?',
                        (NEW, OLD),
                    )
                    if cursor.rowcount:
                        print(f"Migrated {cursor.rowcount} user(s) to 10 GB quota")
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (quota bump): {e}")

                # --- Phase 13 split-quota backfill ---
                # ALTER ADD COLUMN above already gives every user the new
                # 100 GB / 10 GB defaults; this block only fires if an
                # earlier partial migration left someone at 0.
                try:
                    PUB = 100 * 1024 ** 3
                    PRIV = 10 * 1024 ** 3
                    cursor.execute(
                        'UPDATE user SET quota_public_max_bytes = ? '
                        'WHERE quota_public_max_bytes IS NULL OR quota_public_max_bytes <= 0',
                        (PUB,),
                    )
                    if cursor.rowcount:
                        print(f"Backfilled {cursor.rowcount} user(s) with 100 GB public quota")
                    cursor.execute(
                        'UPDATE user SET quota_private_max_bytes = ? '
                        'WHERE quota_private_max_bytes IS NULL OR quota_private_max_bytes <= 0',
                        (PRIV,),
                    )
                    if cursor.rowcount:
                        print(f"Backfilled {cursor.rowcount} user(s) with 10 GB private quota")
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (split quota backfill): {e}")

                # --- DatasetRequest + DatasetRequestVote tables ---
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS dataset_request ("
                        "  id INTEGER PRIMARY KEY,"
                        "  repo_id VARCHAR(200) NOT NULL,"
                        "  description TEXT,"
                        "  requested_by_user_id INTEGER REFERENCES user(id),"
                        "  created_at DATETIME NOT NULL,"
                        "  status VARCHAR(20) NOT NULL DEFAULT 'pending',"
                        "  admin_note TEXT"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_dataset_request_repo_id "
                        "ON dataset_request (repo_id)"
                    )
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS dataset_request_vote ("
                        "  id INTEGER PRIMARY KEY,"
                        "  request_id INTEGER NOT NULL REFERENCES dataset_request(id),"
                        "  user_id INTEGER NOT NULL REFERENCES user(id),"
                        "  created_at DATETIME NOT NULL,"
                        "  UNIQUE(request_id, user_id)"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_dataset_request_vote_request_id "
                        "ON dataset_request_vote (request_id)"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_dataset_request_vote_user_id "
                        "ON dataset_request_vote (user_id)"
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (dataset_request): {e}")

                # --- DatasetVisualization table ---
                try:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS dataset_visualization ("
                        "  id INTEGER PRIMARY KEY,"
                        "  dataset_id INTEGER NOT NULL REFERENCES dataset(id),"
                        "  global_visualization_id INTEGER NOT NULL "
                        "    REFERENCES global_visualization(id),"
                        "  arg_mappings TEXT NOT NULL,"
                        "  target_name VARCHAR(100),"
                        "  display_order INTEGER DEFAULT 0"
                        ")"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_dataset_visualization_dataset_id "
                        "ON dataset_visualization (dataset_id)"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_dataset_visualization_global_visualization_id "
                        "ON dataset_visualization (global_visualization_id)"
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (dataset_visualization): {e}")

                # custom_field.sample_id was historically un-indexed, making
                # any per-sample CF lookup a full table scan — fatal on
                # 100k+ row datasets (dataset_view page took ~50s).
                try:
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_custom_field_sample_id_submission_id "
                        "ON custom_field (sample_id, submission_id)"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS ix_custom_field_submission_id "
                        "ON custom_field (submission_id) WHERE submission_id IS NOT NULL"
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Migration error (custom_field indexes): {e}")

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

                # --- 3c. Backfill custom_field.data_type from field_type ---
                # Pre-Phase-A installs name the typed-contract column
                # `field_type`. The ALTER above adds `data_type`; copy
                # the per-row value across so existing GT/pred rows
                # remain readable through the model's `data_type`
                # attribute. Idempotent: skipped when the legacy column
                # is absent (fresh DBs) or when every row has already
                # been backfilled.
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(custom_field)")
                    cf_cols = {row[1] for row in cursor.fetchall()}
                    if 'field_type' in cf_cols and 'data_type' in cf_cols:
                        cursor.execute(
                            "UPDATE custom_field SET data_type = field_type "
                            "WHERE data_type IS NULL AND field_type IS NOT NULL"
                        )
                        if cursor.rowcount:
                            print(
                                f"Migrating DB: backfilled data_type on "
                                f"{cursor.rowcount} custom_field rows."
                            )
                        conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"custom_field data_type backfill failed (non-fatal): {e}")

                # --- 3d. Drop legacy custom_field.field_type column ---
                # The data has been backfilled into `data_type` for
                # ages. The legacy column was NOT NULL, so once the
                # ORM started inserting only `data_type` rows, every
                # CustomField INSERT (HF import, /api/submit,
                # /api/datasets, …) blew up with `NOT NULL constraint
                # failed: custom_field.field_type`. Drop the column.
                # SQLite 3.35+ supports `ALTER TABLE DROP COLUMN`.
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(custom_field)")
                    cf_cols = {row[1] for row in cursor.fetchall()}
                    if 'field_type' in cf_cols:
                        cursor.execute(
                            "ALTER TABLE custom_field DROP COLUMN field_type"
                        )
                        conn.commit()
                        print(
                            "Migrating DB: dropped legacy "
                            "custom_field.field_type column."
                        )
                    conn.close()
                except Exception as e:
                    print(f"Migration error dropping custom_field.field_type: {e}")

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
                    # Phase 13: split storage budget. 100 GB for content the
                    # user has chosen to publish, 10 GB for private/unlisted
                    # working space. Existing rows pick up these defaults via
                    # the ALTER ADD COLUMN; the legacy single-bucket cap above
                    # is retained for back-compat but no longer enforced.
                    ("user",                 "quota_public_max_bytes",         f"BIGINT NOT NULL DEFAULT {100 * 1024 ** 3}"),
                    ("user",                 "quota_private_max_bytes",        f"BIGINT NOT NULL DEFAULT {10 * 1024 ** 3}"),
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
                    # Submitter-supplied blurb + external link (client submit()).
                    ("submission",           "description",                      "TEXT"),
                    ("submission",           "link",                             "VARCHAR(500)"),
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
                    # Phase A typed contract: per-instance metadata for the
                    # DataType subclass that decodes this field's value
                    # (e.g. {"unit": "meters"} for Depth, {"format": "xyxy"}
                    # for BBoxes). JSON-encoded string; NULL ⇒ {}.
                    ("custom_field",         "data_params",                      "TEXT"),
                    # Phase A typed contract: renamed from `field_type`.
                    # On pre-rename installs the old column remains; this
                    # ALTER adds `data_type` and the backfill below copies
                    # the value across.
                    ("custom_field",         "data_type",                        "VARCHAR(20)"),
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
            CustomField.data_type.in_(['scalar', 'metric'])
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
                        CustomField.data_type.in_(['scalar', 'metric'])
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
                    cfs = CustomField.query.filter_by(submission_id=sub.id, name=m_id_name, data_type='scalar').all()
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
    on-disk folder under uploads/datasets/<id>/ exists. If either is
    missing — typical signature of an interrupted typed-manifest
    import or a deleted volume snapshot — the row is just orphaned
    bookkeeping. Cascade-delete it (and any leftover folder bytes)
    so the datasets list is honest about what's actually available.

    Runs at boot from run_migrations(); also exposed for tests.

    Returns the number of datasets removed.
    """
    upload_folder = app.config['UPLOAD_FOLDER']
    datasets_root = os.path.join(upload_folder, 'datasets')
    removed = 0

    for ds in Dataset.query.all():
        # Async HF imports look "incomplete" by all the usual signals
        # (no samples yet, no folder) right up until the Celery task
        # finishes. Skip those — the task itself flips status to
        # 'failed' + cleans up, or 'ready' on success.
        if (ds.import_status or 'ready') == 'importing':
            continue
        # Pointer-mode datasets INTENTIONALLY have no on-disk folder —
        # bytes live on HF and stream through bench_cache. Skip the
        # folder check for them; sample-count > 0 is the only signal
        # of completion.
        is_pointer = (ds.storage_mode == 'hf-pointer')
        sample_count = Sample.query.filter_by(dataset_id=ds.id).count()
        folder_path = os.path.join(datasets_root, str(ds.id))
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
        # Idempotent reference-metric upsert. Every boot the curated
        # set (accuracy / rmse_depth / mae_depth / iou_mask /
        # exact_match_text / top_1_accuracy / top_5_accuracy) lands
        # on the live DB with the current source code + input_kinds
        # + input_roles, so a `git pull && systemctl restart` ships
        # new reference metrics or fixes to existing ones without a
        # manual seed-script step. Best-effort: any failure (HF API
        # check, etc.) is non-fatal.
        try:
            from scripts.seed_reference_metrics import seed_reference_metrics
            seed_reference_metrics()
        except Exception as e:
            print(f"seed_reference_metrics failed (non-fatal): {e}")

        # Same idempotent upsert for the curated reference visualizations
        # (currently: confusion_matrix). Appears in the /visualizations
        # catalog for everyone; bind per-LB on the LB settings page.
        try:
            from scripts.seed_reference_visualizations import seed_reference_visualizations
            seed_reference_visualizations()
        except Exception as e:
            print(f"seed_reference_visualizations failed (non-fatal): {e}")


# Auto-run migrations at module import time when the env var is set. On the
# home-box deploy this is what makes a schema ALTER apply on `systemctl
# restart` (the systemd unit's EnvironmentFile sets BENCHHUB_AUTO_MIGRATE=1).
# Historical reason it exists: the old Fly deploy's `release_command` ran in
# a temp VM that didn't mount the persistent volume, so it couldn't migrate
# the live SQLite DB by itself.
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
