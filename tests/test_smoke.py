"""Smoke tests for the conftest fixtures.

Confirms the app loads with BENCHHUB_DATA_DIR redirected, the DB schema is
created fresh per test, and the test client can hit a basic route.
"""
import os


def test_app_uses_test_data_dir(app):
    expected = os.environ["BENCHHUB_DATA_DIR"]
    assert app.config["UPLOAD_FOLDER"].startswith(expected)
    assert expected in app.config["SQLALCHEMY_DATABASE_URI"]


def test_db_session_starts_empty(app, db_session):
    from app import Dataset

    assert Dataset.query.count() == 0


def test_db_session_isolated_between_tests_part_1(app, db_session):
    from app import Dataset

    db_session.add(Dataset(name="leaks-if-not-isolated"))
    db_session.commit()
    assert Dataset.query.count() == 1


def test_db_session_isolated_between_tests_part_2(app, db_session):
    from app import Dataset

    # If part_1 leaked, this would be 1.
    assert Dataset.query.count() == 0


def test_root_renders_landing_page(client):
    """Replaces the old `/projects` redirect — Phase 6 made `/` a real
    public marketing page that anonymous visitors can hit."""
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert b"Benchmark your model" in resp.data


def test_celery_is_eager(app):
    from app import celery

    # The conftest migrates Celery config to new-style keys; assert the runtime
    # flag actually used by Celery's worker dispatch.
    assert celery.conf.task_always_eager is True
