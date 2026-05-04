"""Pytest fixtures for BenchHub.

The data dir is redirected via BENCHHUB_DATA_DIR (set before any `app` import)
so tests never touch ~/.dtofbenchmarking. Each test gets a fresh schema.
"""
import io
import json as _json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pytest

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="benchhub-tests-")
os.environ["BENCHHUB_DATA_DIR"] = _TEST_DATA_DIR

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def app():
    from app import app as flask_app, db

    flask_app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
    )

    # The app sets old-style Celery keys (CELERY_BROKER_URL etc.) via
    # make_celery → celery.conf.update(app.config). Old-style keys do NOT
    # propagate transparently to their new-style counterparts on Celery 5.6
    # (verified: `CELERY_TASK_ALWAYS_EAGER=True` leaves `task_always_eager=False`).
    # Migrate everything to new-style in a single update so the mix-check
    # doesn't fire and so the runtime actually picks up eager mode.
    from app import celery
    broker = celery.conf.get("CELERY_BROKER_URL") or "memory://"
    celery.conf.update(
        broker_url=broker,
        result_backend="cache+memory://",
        task_always_eager=True,
        task_eager_propagates=True,
    )

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def db_session(app):
    """Wipe and recreate tables before each test. Simple and slow-but-fine for SQLite."""
    from app import db

    db.session.remove()
    db.drop_all()
    db.create_all()

    upload_folder = app.config["UPLOAD_FOLDER"]
    if os.path.isdir(upload_folder):
        shutil.rmtree(upload_folder, ignore_errors=True)
    os.makedirs(upload_folder, exist_ok=True)

    yield db.session

    db.session.remove()


@pytest.fixture
def client(app, db_session):
    return app.test_client()


@pytest.fixture
def project_ctx(app, db_session, client):
    """Create a Project and pin it as the active one via cookie.

    Most non-API routes require a project context (the `load_project_context`
    middleware redirects to /projects otherwise). Tests that don't care which
    project they target can use this fixture to satisfy the middleware.
    """
    from app import Project, db

    p = Project(name="ctx_proj")
    db.session.add(p)
    db.session.commit()
    client.set_cookie("active_project_id", str(p.id))
    return p


# ---------------------------------------------------------------------------
# ZIP factory — used by Phase 2-4
# ---------------------------------------------------------------------------


def _encode_file(value):
    """Convert a layout entry into bytes for zipfile.

    - bytes: written as-is
    - str: utf-8 encoded
    - dict: assumed to be {"npz": {"key": np.array(...)}} or a json-encodable dict
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, dict) and "npz" in value:
        buf = io.BytesIO()
        np.savez(buf, **value["npz"])
        return buf.getvalue()
    if isinstance(value, dict):
        return _json.dumps(value).encode("utf-8")
    raise TypeError(f"Unsupported layout value type: {type(value)!r}")


def build_zip(target_zip_path, layout, root_folder=None):
    """Create a ZIP file at `target_zip_path` from a flat path→content mapping.

    layout: dict mapping a relative POSIX path to one of:
        - str (utf-8 written verbatim)
        - bytes (written verbatim)
        - dict with "npz" key → np.savez(**value["npz"])
        - other dict → JSON-serialized

    root_folder: if set, every entry is nested under this folder (simulates the
                 "single root folder inside the ZIP" pattern that triggers the
                 dataset/submission rename path).
    """
    with zipfile.ZipFile(target_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path, value in layout.items():
            arcname = f"{root_folder}/{rel_path}" if root_folder else rel_path
            zf.writestr(arcname, _encode_file(value))
    return target_zip_path


@pytest.fixture
def make_zip(tmp_path):
    """Returns a callable that builds a ZIP for a test and yields its path.

    Usage:
        zip_path = make_zip("my.zip", {"config/s1.json": '{"k": 1}'})
        zip_path = make_zip("d.zip", {"hist/s1.npz": {"npz": {"bins": [...], "counts": [...]}}}, root_folder="ds_v1")
    """
    counter = {"n": 0}

    def _make(name, layout, root_folder=None):
        counter["n"] += 1
        path = tmp_path / f"{counter['n']}_{name}"
        return str(build_zip(str(path), layout, root_folder=root_folder))

    return _make
