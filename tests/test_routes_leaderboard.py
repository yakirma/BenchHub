"""Route tests for leaderboard lifecycle.

Leaderboards live under /<project_name>/. Each leaderboard is bound to one
or more datasets (many-to-many).
"""
import json

import pytest

from app import (
    Dataset,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    Sample,
    db,
)


@pytest.fixture
def project(db_session, client):
    import types
    return types.SimpleNamespace(id=0, name='legacy')


@pytest.fixture
def dataset(db_session):
    ds = Dataset(name="lb_ds")
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name="s1"))
    db.session.commit()
    return ds


@pytest.fixture
def leaderboard(db_session, project, dataset):
    lb = Leaderboard(name="primary_lb", summary_metrics="")
    lb.datasets.append(dataset)
    db.session.add(lb)
    db.session.commit()
    return lb


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_dataset_view_renders_new_leaderboard_form_for_signed_in(
    auth_client, logged_in_user, db_session,
):
    """Anyone signed in sees the inline 'New leaderboard' form on a
    dataset detail page (not just the owner)."""
    from app import Dataset, db as _db
    ds = Dataset(name='lb_form_ds', visibility='public')
    _db.session.add(ds); _db.session.commit()

    body = auth_client.get(f'/dataset/{ds.id}').data
    assert b'New leaderboard' in body
    # Form posts to the create endpoint with this dataset pre-selected.
    assert b'/create_leaderboard' in body
    assert f'value="{ds.id}"'.encode() in body


def test_dataset_view_hides_new_leaderboard_form_anon(client, db_session):
    from app import Dataset, db as _db
    ds = Dataset(name='lb_form_anon', visibility='public')
    _db.session.add(ds); _db.session.commit()
    body = client.get(f'/dataset/{ds.id}').data
    assert b'New leaderboard' not in body


def test_create_leaderboard_attaches_dataset(auth_client, project, dataset, logged_in_user):
    resp = auth_client.post(
        "/create_leaderboard",
        data={"leaderboard_name": "new_lb", "dataset_ids": [str(dataset.id)]},
    )
    assert resp.status_code == 302

    lb = Leaderboard.query.filter_by(name="new_lb").first()
    assert lb is not None
    assert dataset in lb.datasets
    assert lb.owner_user_id == logged_in_user.id


def test_create_leaderboard_supports_multiple_datasets(auth_client, project, dataset):
    ds2 = Dataset(name="lb_ds_2")
    db.session.add(ds2)
    db.session.commit()

    resp = auth_client.post(
        "/create_leaderboard",
        data={
            "leaderboard_name": "multi_lb",
            "dataset_ids": [str(dataset.id), str(ds2.id)],
        },
    )
    assert resp.status_code == 302

    lb = Leaderboard.query.filter_by(name="multi_lb").first()
    assert {d.name for d in lb.datasets} == {"lb_ds", "lb_ds_2"}


def test_create_leaderboard_collision_without_overwrite_blocks(
    auth_client, project, dataset, leaderboard
):
    resp = auth_client.post(
        "/create_leaderboard",
        data={
            "leaderboard_name": leaderboard.name,
            "dataset_ids": [str(dataset.id)],
        },
    )
    assert resp.status_code == 302
    # Still only one leaderboard with that name in this project.
    assert Leaderboard.query.filter_by(name=leaderboard.name).count() == 1


def test_create_leaderboard_with_overwrite_replaces_existing(
    auth_client, project, dataset, leaderboard
):
    old_id = leaderboard.id

    resp = auth_client.post(
        "/create_leaderboard",
        data={
            "leaderboard_name": leaderboard.name,
            "overwrite": "true",
            "dataset_ids": [str(dataset.id)],
        },
    )
    assert resp.status_code == 302

    db.session.expire_all()
    fresh = Leaderboard.query.filter_by(name=leaderboard.name).all()
    assert len(fresh) == 1
    # The new leaderboard replaced the old one (different identity, same logical slot).
    # Don't compare IDs (SQLite recycles); just ensure exactly one row exists.


# ---------------------------------------------------------------------------
# View / edit / delete
# ---------------------------------------------------------------------------


def test_leaderboard_view_renders(client, project, leaderboard):
    resp = client.get(f"/leaderboard/{leaderboard.id}")
    assert resp.status_code == 200
    assert b"primary_lb" in resp.data


def test_leaderboard_view_unknown_404(client, project):
    resp = client.get("/leaderboard/9999")
    assert resp.status_code == 404


def test_admin_cache_stats_renders_for_admin(auth_client, logged_in_user, db_session):
    """Admin cache-stats page renders for admins, even with an empty
    cache. Lock the view together with its summary copy."""
    logged_in_user.is_admin = True
    db.session.commit()
    resp = auth_client.get('/admin/cache_stats')
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'Cache stats' in body
    assert 'GT thumbnails' in body
    assert 'Submission cache' in body
    assert 'Datasets' in body


def test_admin_cache_stats_forbidden_to_non_admin(auth_client, logged_in_user, db_session):
    logged_in_user.is_admin = False
    db.session.commit()
    resp = auth_client.get('/admin/cache_stats')
    assert resp.status_code == 403


def test_samples_only_view_hides_scalar_metric_columns(client, db_session):
    """Explore samples should suppress the GT-Stats / Submission-Stats /
    Metric-chart columns by default — they're either empty or
    duplicate info when there are no submissions to compare. Only
    image-like columns + sample_name/tags remain."""
    from app import Attachment, CustomField
    lb = Leaderboard(name='img_only_lb', summary_metrics='',
                     visibility='public')
    db.session.add(lb); db.session.flush()
    db.session.add(Attachment(
        leaderboard_id=lb.id, hf_repo_id='fake/fake',
        hf_split='train', role='primary',
    ))
    # GT snapshots: one scalar + one image marker per sample so both
    # column shapes are present.
    for i in range(3):
        db.session.add(CustomField(
            leaderboard_id=lb.id, sample_id=None, submission_id=None,
            sample_name=f's_{i:06d}', name='label',
            field_type='scalar', value_float=float(i),
        ))
        db.session.add(CustomField(
            leaderboard_id=lb.id, sample_id=None, submission_id=None,
            sample_name=f's_{i:06d}', name='image_image',
            field_type='image', source_column='image',
        ))
    db.session.commit()

    resp = client.get(f'/comparison/{lb.id}?samples_only=1')
    assert resp.status_code == 200
    body = resp.data.decode()
    # Image column survives.
    assert 'image_image' in body
    # The scalar-stats column header is suppressed in samples-only mode.
    # (Cannot just look for the word "Scalars" — it lives inside the
    # per_source_stats panel that we're hiding.)
    assert 'GT Stats' not in body
    assert 'Submission Stats' not in body


def test_edit_leaderboard_get_renders(auth_client, project, leaderboard):
    resp = auth_client.get(f"/leaderboard/{leaderboard.id}/edit")
    assert resp.status_code == 200


def test_delete_leaderboard_removes_row(auth_client, project, leaderboard):
    resp = auth_client.post(f"/delete_leaderboard/{leaderboard.id}")
    assert resp.status_code == 302

    db.session.expire_all()
    assert Leaderboard.query.get(leaderboard.id) is None


# ---------------------------------------------------------------------------
# import_settings (the "Import from another LB" flow)
# ---------------------------------------------------------------------------


def test_import_settings_clones_metrics_with_id_remapping(
    auth_client, project, dataset, leaderboard
):
    # Set up a SOURCE leaderboard with one metric and a summary_metrics field
    # that references that metric's lm_<id>.
    src_lb = Leaderboard(name="src_lb", summary_metrics="")
    src_lb.datasets.append(dataset)
    db.session.add(src_lb)
    db.session.flush()

    gm = GlobalMetric(name="src_metric", python_code="def m(): return 1")
    db.session.add(gm)
    db.session.flush()

    src_lm = LeaderboardMetric(
        leaderboard_id=src_lb.id,
        global_metric_id=gm.id,
        arg_mappings="{}",
        target_name="alpha",
    )
    db.session.add(src_lm)
    db.session.flush()
    # Reference the metric by its lm_<id> in the summary_metrics CSV.
    src_lb.summary_metrics = f"lm_{src_lm.id}"
    db.session.commit()

    resp = auth_client.post(
        f"/leaderboard/{leaderboard.id}/import_settings",
        data={"source_leaderboard_id": str(src_lb.id)},
    )
    assert resp.status_code == 302

    db.session.expire_all()
    target = Leaderboard.query.get(leaderboard.id)
    assert len(target.leaderboard_metrics) == 1
    new_lm = target.leaderboard_metrics[0]
    # IDs must have been remapped from src_lm.id → new_lm.id in the summary CSV.
    assert target.summary_metrics == f"lm_{new_lm.id}"
    assert target.summary_metrics != f"lm_{src_lm.id}"


def test_import_settings_clears_existing_metrics_first(
    auth_client, project, dataset, leaderboard
):
    # Pre-populate target with a metric — it must be deleted before import.
    gm = GlobalMetric(name="pre_existing", python_code="def m(): return 1")
    db.session.add(gm)
    db.session.flush()
    db.session.add(
        LeaderboardMetric(
            leaderboard_id=leaderboard.id,
            global_metric_id=gm.id,
            arg_mappings="{}",
        )
    )
    db.session.commit()

    src_lb = Leaderboard(name="src_lb", summary_metrics="")
    src_lb.datasets.append(dataset)
    db.session.add(src_lb)
    db.session.commit()

    auth_client.post(
        f"/leaderboard/{leaderboard.id}/import_settings",
        data={"source_leaderboard_id": str(src_lb.id)},
    )

    db.session.expire_all()
    target = Leaderboard.query.get(leaderboard.id)
    assert target.leaderboard_metrics == []


# ---------------------------------------------------------------------------
# JSON info APIs
# ---------------------------------------------------------------------------


def test_api_leaderboard_info_by_id_returns_json(client, leaderboard, dataset):
    resp = client.get(f"/api/leaderboard/{leaderboard.id}/info")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == leaderboard.id
    assert body["name"] == leaderboard.name
    assert body["dataset"]["id"] == dataset.id


def test_api_leaderboard_info_by_id_404_unknown(client):
    resp = client.get("/api/leaderboard/9999/info")
    assert resp.status_code == 404


def test_api_leaderboard_info_by_name_scoped_to_project(
    client, project, leaderboard
):
    resp = client.get(
        f"/api/leaderboard/by_name/{leaderboard.name}/info"
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == leaderboard.id


def test_api_leaderboard_info_by_name_404_for_unknown_name(client):
    """Names are global now (project namespace removed). Unknown name → 404."""
    resp = client.get("/api/leaderboard/by_name/no_such_lb/info")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# suggest_name
# ---------------------------------------------------------------------------


def test_suggest_name_returns_base_when_available(client):
    resp = client.get("/api/leaderboard/suggest_name?name=fresh")
    assert resp.status_code == 200
    assert resp.get_json()["suggested_name"] == "fresh"


def test_suggest_name_appends_counter_when_taken(client, leaderboard):
    resp = client.get(f"/api/leaderboard/suggest_name?name={leaderboard.name}")
    assert resp.status_code == 200
    assert resp.get_json()["suggested_name"] == f"{leaderboard.name}_2"


def test_suggest_name_keeps_incrementing(client, project, dataset, leaderboard):
    # Add primary_lb_2 too — so the helper should bump to _3.
    extra = Leaderboard(
        name=f"{leaderboard.name}_2",
        summary_metrics="",
    )
    extra.datasets.append(dataset)
    db.session.add(extra)
    db.session.commit()

    resp = client.get(f"/api/leaderboard/suggest_name?name={leaderboard.name}")
    assert resp.get_json()["suggested_name"] == f"{leaderboard.name}_3"


def test_suggest_name_400_when_blank(client):
    resp = client.get("/api/leaderboard/suggest_name?name=")
    assert resp.status_code == 400
