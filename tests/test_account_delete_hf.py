"""Account deletion (GDPR) + HuggingFace BYO importer tests."""
import os
import sys
import types
from unittest.mock import patch

import pytest

from app import (
    Dataset,
    GlobalMetric,
    GlobalVisualization,
    Leaderboard,
    Sample,
    Submission,
    User,
    db,
)


# ===========================================================================
# Account deletion
# ===========================================================================


def test_account_delete_requires_email_match(auth_client, logged_in_user, db_session):
    """Wrong confirmation text → no deletion, flash message."""
    resp = auth_client.post(
        '/settings/account/delete',
        data={'confirm_email': 'wrong@example.com'},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    # User row still exists.
    assert User.query.get(logged_in_user.id) is not None


def test_account_delete_with_correct_email_removes_user(auth_client, logged_in_user, db_session):
    user_id = logged_in_user.id
    resp = auth_client.post(
        '/settings/account/delete',
        data={'confirm_email': logged_in_user.email},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert User.query.get(user_id) is None


def test_account_delete_cascades_owned_content(auth_client, logged_in_user, db_session):
    """Owned project + dataset + leaderboard + submission + global metric
    all go away. Files-on-disk for the dataset folder also go."""
    ds = Dataset(name='my_ds', owner_user_id=logged_in_user.id)
    db.session.add_all([ds]); db.session.flush()

    lb = Leaderboard(name='my_lb', summary_metrics='',
                     owner_user_id=logged_in_user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()

    sub = Submission(name='my_sub', leaderboard_id=lb.id,
                     owner_user_id=logged_in_user.id)
    gm = GlobalMetric(name='my_metric', python_code='x=1', is_aggregated=False,
                      owner_user_id=logged_in_user.id)
    gv = GlobalVisualization(name='my_viz', python_code='x=1',
                             owner_user_id=logged_in_user.id)
    db.session.add_all([sub, gm, gv]); db.session.commit()

    ds_id, lb_id, sub_id, gm_id, gv_id = ds.id, lb.id, sub.id, gm.id, gv.id

    resp = auth_client.post(
        '/settings/account/delete',
        data={'confirm_email': logged_in_user.email},
        follow_redirects=False,
    )
    assert resp.status_code == 302

    assert Dataset.query.get(ds_id) is None
    assert Leaderboard.query.get(lb_id) is None
    assert Submission.query.get(sub_id) is None
    assert GlobalMetric.query.get(gm_id) is None
    assert GlobalVisualization.query.get(gv_id) is None


def test_account_delete_detaches_submissions_in_other_users_leaderboards(
    auth_client, logged_in_user, db_session,
):
    """A submission to someone else's public leaderboard should NOT be
    deleted — that would corrupt their benchmark history. The owner_user_id
    link is severed instead."""
    other = User(
        email='other@example.com', display_name='Other',
        oauth_provider='github', oauth_sub='other-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(name='others_ds', owner_user_id=other.id)
    db.session.add_all([ds]); db.session.flush()

    lb = Leaderboard(name='others_lb', summary_metrics='',
                     owner_user_id=other.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()

    # Logged-in user submits to someone else's leaderboard.
    sub = Submission(name='guest_sub', leaderboard_id=lb.id,
                     owner_user_id=logged_in_user.id)
    db.session.add(sub); db.session.commit()
    sub_id = sub.id

    resp = auth_client.post(
        '/settings/account/delete',
        data={'confirm_email': logged_in_user.email},
        follow_redirects=False,
    )
    assert resp.status_code == 302

    # Submission still exists, but disowned.
    surviving = Submission.query.get(sub_id)
    assert surviving is not None
    assert surviving.owner_user_id is None


def test_account_settings_page_requires_login(client):
    resp = client.get('/settings/account', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


# ===========================================================================
# HuggingFace BYO
# ===========================================================================


def test_hf_import_route_requires_login(client, project_ctx):
    resp = client.post(
        '/import_from_hf',
        data={'hf_repo_id': 'user/repo'},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_hf_import_with_blank_repo_flashes_error(auth_client, project_ctx, db_session):
    resp = auth_client.post(
        '/import_from_hf',
        data={'hf_repo_id': '', 'dataset_name': 'whatever'},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b'Missing HuggingFace repo' in resp.data


def test_hf_import_calls_snapshot_download_and_creates_dataset(
    auth_client, project_ctx, logged_in_user, db_session, tmp_path,
):
    """Patch snapshot_download to lay down a BenchHub-conventional folder
    structure on disk; assert the dataset is ingested as if it were a ZIP."""
    # Build a fake snapshot directory mimicking a structured HF repo.
    # README at the root prevents the single-folder unwrap heuristic in
    # process_dataset_zip from treating metric_score/ as the dataset root.
    snap_root = tmp_path / 'fake_snap'
    snap_root.mkdir()
    (snap_root / 'README.md').write_text('Fake HF dataset.')
    metric_dir = snap_root / 'metric_score'
    metric_dir.mkdir()
    (metric_dir / 's1.txt').write_text('0.5')
    (metric_dir / 's2.txt').write_text('0.7')

    def fake_snapshot_download(repo_id, repo_type, revision, token, local_dir):
        # Mimic the real function: copy the fake snapshot into local_dir.
        import shutil as _sh
        _sh.copytree(snap_root, local_dir)
        return local_dir

    # Inject a fake huggingface_hub module — the real package isn't a
    # test dependency. The lazy `from huggingface_hub import ...` inside
    # the importer picks this up via sys.modules.
    fake_hub = types.ModuleType('huggingface_hub')
    fake_hub.snapshot_download = fake_snapshot_download
    with patch.dict(sys.modules, {'huggingface_hub': fake_hub}):
        resp = auth_client.post(
            '/import_from_hf',
            data={
                'hf_repo_id': 'fake-org/fake-dataset',
                'dataset_name': 'hf_imported_ds',
            },
            follow_redirects=True,
        )

    assert resp.status_code == 200
    ds = Dataset.query.filter_by(name='hf_imported_ds').first()
    assert ds is not None
    assert ds.owner_user_id == logged_in_user.id
    # Samples derived from the metric files.
    sample_names = {s.name for s in Sample.query.filter_by(dataset_id=ds.id).all()}
    assert sample_names == {'s1', 's2'}


def test_hf_import_quota_blocks_when_dataset_count_at_cap(
    auth_client, project_ctx, logged_in_user, db_session,
):
    logged_in_user.quota_max_datasets = 0
    db.session.commit()

    resp = auth_client.post(
        '/import_from_hf',
        data={'hf_repo_id': 'org/ds', 'dataset_name': 'over_cap'},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    # Should NOT have created the dataset.
    assert Dataset.query.filter_by(name='over_cap').first() is None
    assert b'limit' in resp.data.lower() or b'reached' in resp.data.lower()
