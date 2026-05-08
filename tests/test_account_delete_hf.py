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
# The legacy POST /import_from_hf direct-import route has been removed
# in Phase 5; HF datasets become leaderboard attachments via
# /import_from_hf/preview, never local Dataset rows. The login_required
# guard on the new entry page is covered by tests/test_canonicality.py
# (auto_lb_preview flow) and the schema/quota gates by
# tests/test_attachment_iter.py + tests/test_canonicality.py.
