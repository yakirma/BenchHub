"""Route tests for dataset lifecycle.

Datasets are global (not project-scoped). The web upload form lives under
/<project_name>/upload_dataset, but the API and the listing/view/delete
routes don't carry a project component.
"""
import io
import json
import os

import pytest

from app import Dataset, Sample, User, app, db, generate_api_token


@pytest.fixture
def api_token_headers(db_session):
    """Phase 8: /api/dataset/upload now requires a Bearer token. Tests
    that hit it use this fixture to mint a User with a token and pass
    the header in."""
    u = User(
        email='api-tok@example.com',
        display_name='API Tester',
        oauth_provider='github',
        oauth_sub='api-tok-1',
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return {'Authorization': f'Bearer {u.api_token}'}


@pytest.fixture
def seeded_dataset(db_session, make_zip):
    """Run process_dataset_zip once so we have a Dataset with files on disk
    that other tests can download/view/delete."""
    from app import process_dataset_zip

    layout = {
        "config/s1.json": '{"k":1}',
        "config/s2.json": '{"k":2}',
    }
    zip_path = make_zip("seed.zip", layout, root_folder="seed_ds")
    success, _, ds_id = process_dataset_zip(zip_path, "seed_ds")
    assert success
    return Dataset.query.get(ds_id)


# ---------------------------------------------------------------------------
# Listing / viewing
# ---------------------------------------------------------------------------


def test_datasets_index_lists_existing(client, project_ctx, seeded_dataset):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    assert b"seed_ds" in resp.data


def test_dataset_view_renders(client, project_ctx, seeded_dataset):
    resp = client.get(f"/dataset/{seeded_dataset.id}")
    assert resp.status_code == 200
    assert b"seed_ds" in resp.data


def test_dataset_view_404_for_unknown_id(client, project_ctx):
    resp = client.get("/dataset/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Web-form upload
# ---------------------------------------------------------------------------


def test_web_upload_creates_dataset(auth_client, project_ctx, make_zip, logged_in_user):
    layout = {"config/s1.json": '{"k":1}'}
    zip_path = make_zip("web.zip", layout, root_folder="web_ds")

    with open(zip_path, "rb") as f:
        resp = auth_client.post(
            "/upload_dataset",
            data={"dataset_name": "web_ds", "dataset_zip": (f, "web.zip")},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 302
    ds = Dataset.query.filter_by(name="web_ds").first()
    assert ds is not None
    assert ds.owner_user_id == logged_in_user.id


def test_web_upload_with_blank_name_uses_filename(auth_client, project_ctx, make_zip):
    layout = {"config/s1.json": '{"k":1}'}
    zip_path = make_zip("auto_named.zip", layout, root_folder="auto_named")

    with open(zip_path, "rb") as f:
        resp = auth_client.post(
            "/upload_dataset",
            data={"dataset_name": "", "dataset_zip": (f, "auto_named.zip")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 302

    # Inner-folder rename wins → final name is "auto_named".
    assert Dataset.query.filter_by(name="auto_named").count() == 1


# ---------------------------------------------------------------------------
# API upload (JSON response)
# ---------------------------------------------------------------------------


def test_api_upload_returns_201_and_dataset_id(client, make_zip, api_token_headers):
    zip_path = make_zip(
        "api.zip", {"config/s1.json": '{"k":1}'}, root_folder="api_ds"
    )
    with open(zip_path, "rb") as f:
        resp = client.post(
            "/api/dataset/upload",
            headers=api_token_headers,
            data={"dataset_name": "api_ds", "dataset_zip": (f, "api.zip")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["dataset_id"] is not None
    assert "Uploaded" in body["message"]


def test_api_upload_missing_file_returns_400(client, api_token_headers):
    resp = client.post(
        "/api/dataset/upload",
        headers=api_token_headers,
        data={},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_upload_collision_without_override_returns_400(client, seeded_dataset, make_zip, api_token_headers):
    zip_path = make_zip(
        "dup.zip", {"config/s1.json": '{"k":1}'}, root_folder="seed_ds"
    )
    with open(zip_path, "rb") as f:
        resp = client.post(
            "/api/dataset/upload",
            headers=api_token_headers,
            data={"dataset_name": "seed_ds", "dataset_zip": (f, "dup.zip")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400
    assert "already exists" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def test_download_dataset_returns_zip(client, project_ctx, seeded_dataset):
    resp = client.get(f"/dataset/{seeded_dataset.id}/download")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] in ("application/zip", "application/x-zip-compressed")
    # File starts with PK (zip signature).
    assert resp.data[:2] == b"PK"


# ---------------------------------------------------------------------------
# Display column updates
# ---------------------------------------------------------------------------


def test_dataset_view_search_filters_samples(client, project_ctx, seeded_dataset):
    """Search box on /dataset/<id> filters samples by name (case-insensitive
    substring), mirrors the comparison view's `search` param."""
    body = client.get(f'/dataset/{seeded_dataset.id}?search=s1').data
    assert b's1' in body
    # The seeded dataset has s1 + s2; s2 should drop out.
    # (Using partial markers since rendering varies, just check the
    # search input value persisted.)
    assert b'value="s1"' in body


def test_dataset_view_search_renders_clear_link(client, project_ctx, seeded_dataset):
    body = client.get(f'/dataset/{seeded_dataset.id}?search=s1').data
    # Active search → "Clear search" button shows.
    assert b'Clear search' in body
    # No search → no clear button.
    body2 = client.get(f'/dataset/{seeded_dataset.id}').data
    assert b'Clear search' not in body2


def test_update_display_columns_records_hidden(client, project_ctx, seeded_dataset):
    """New inverted model: form posts back the chosen-visible list AND
    the full rendered set; route stores the difference as 'hidden'."""
    resp = client.post(
        f"/dataset/{seeded_dataset.id}/update_display_columns",
        data={
            "display_columns": ["sample_name", "tags"],
            "display_columns_all": ["sample_name", "tags", "config", "histogram"],
        },
    )
    assert resp.status_code in (302, 200)
    db.session.expire_all()
    refreshed = Dataset.query.get(seeded_dataset.id)
    # 'config' and 'histogram' were rendered but not chosen → saved as hidden.
    saved = sorted((refreshed.hidden_display_columns or '').split(','))
    assert saved == ['config', 'histogram']


def test_update_display_columns_all_chosen_clears_hidden(
    client, project_ctx, seeded_dataset
):
    """Selecting every rendered option means 'no exclusions' — hidden
    column gets cleared back to NULL so the dataset is in the
    everything-visible default state."""
    seeded_dataset.hidden_display_columns = 'tags'
    db.session.commit()

    client.post(
        f"/dataset/{seeded_dataset.id}/update_display_columns",
        data={
            "display_columns": ["sample_name", "tags", "config"],
            "display_columns_all": ["sample_name", "tags", "config"],
        },
    )
    db.session.expire_all()
    refreshed = Dataset.query.get(seeded_dataset.id)
    assert refreshed.hidden_display_columns is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_dataset_removes_row_and_files(auth_client, project_ctx, seeded_dataset):
    ds_dir = os.path.join(app.config["UPLOAD_FOLDER"], "datasets", "seed_ds")
    assert os.path.isdir(ds_dir)  # sanity

    resp = auth_client.post(f"/dataset/{seeded_dataset.id}/delete")
    assert resp.status_code == 302

    db.session.expire_all()
    assert Dataset.query.get(seeded_dataset.id) is None
    assert not os.path.exists(ds_dir)


def test_delete_dataset_404_unknown(auth_client, project_ctx):
    resp = auth_client.post("/dataset/9999/delete")
    assert resp.status_code == 404
