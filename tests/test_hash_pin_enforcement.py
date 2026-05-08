"""Strict hash-pinning for remote submissions.

Pin: when a remote submission's bytes drift on the upstream URL
(user edited their `predictions.zip` post-submission), recalc must
refuse to evaluate against the new bytes and surface a clear
"submission file changed; please resubmit" status.

Local submissions are immutable on Fly disk → verifier always passes.
"""
import hashlib
import io
import sys
import zipfile
from unittest.mock import patch

import pytest

from app import (
    Dataset, Leaderboard, Submission, db,
    _verify_remote_submission_hash,
)


def _zip_bytes(payload):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('metric_dummy/s00000.txt', str(payload))
        zf.writestr('README.md', f'submission-{payload}')
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def remote_sub(db_session, logged_in_user):
    """A bare-bones remote Submission (no LB metrics needed for the
    verifier — it only inspects storage_mode + remote_url + content_hash)."""
    ds = Dataset(name='hash_pin_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='hash_pin_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    sub = Submission(
        name='hash_pin_sub', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        storage_mode='remote',
        remote_url='https://example.test/sub.zip',
    )
    db.session.add(sub); db.session.commit()
    return sub


def _patch_fetcher(returned_hash):
    """Stand-in for `_fetch_remote_submission_zip` that returns a
    fixed (path, hash). Avoids real HTTP / cache plumbing in
    verifier-only tests."""
    return patch(
        'app._fetch_remote_submission_zip',
        return_value=('/tmp/whatever', returned_hash),
    )


# ---------------------------------------------------------------------------
# Local submissions: verifier no-ops.
# ---------------------------------------------------------------------------


def test_local_submission_always_passes(db_session, logged_in_user):
    ds = Dataset(name='local_pass_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='local_pass_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(
        name='local_pass', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        storage_mode='local',
    )
    db.session.add(sub); db.session.commit()
    ok, msg = _verify_remote_submission_hash(sub)
    assert ok and msg == ''


# ---------------------------------------------------------------------------
# Remote submissions: hash captured / matches / mismatches.
# ---------------------------------------------------------------------------


def test_remote_first_eval_populates_hash_when_unset(remote_sub, db_session):
    """If content_hash is NULL (first re-eval after submission),
    populate it with whatever the fetcher returns and pass."""
    assert remote_sub.content_hash is None
    fresh_hash = hashlib.sha256(b'whatever').hexdigest()
    with _patch_fetcher(fresh_hash):
        ok, msg = _verify_remote_submission_hash(remote_sub)
    assert ok and msg == ''
    db.session.refresh(remote_sub)
    assert remote_sub.content_hash == fresh_hash


def test_remote_matching_hash_passes(remote_sub, db_session):
    pinned = hashlib.sha256(b'first').hexdigest()
    remote_sub.content_hash = pinned
    db.session.commit()
    with _patch_fetcher(pinned):
        ok, msg = _verify_remote_submission_hash(remote_sub)
    assert ok and msg == ''
    db.session.refresh(remote_sub)
    # processing_status untouched.
    assert not (remote_sub.processing_status or '').startswith('Error')


def test_remote_mismatch_sets_error_status_and_returns_false(remote_sub, db_session):
    pinned = hashlib.sha256(b'first').hexdigest()
    drifted = hashlib.sha256(b'second').hexdigest()
    remote_sub.content_hash = pinned
    db.session.commit()
    with _patch_fetcher(drifted):
        ok, msg = _verify_remote_submission_hash(remote_sub)
    assert not ok
    # First 12 chars of each hash surfaced in the message for debugging.
    assert pinned[:12] in msg
    assert drifted[:12] in msg
    db.session.refresh(remote_sub)
    assert remote_sub.processing_status == (
        'Error: submission file changed; please resubmit'
    )
    # The stored hash stays pinned to the original — recalc fails
    # closed. User must explicitly resubmit to record the new hash.
    assert remote_sub.content_hash == pinned


def test_remote_fetch_failure_returns_false_with_message(remote_sub, db_session):
    """Network drop / 404 on the upstream URL surfaces as a verifier
    failure with the underlying error text — caller's recalc bails."""
    remote_sub.content_hash = hashlib.sha256(b'pinned').hexdigest()
    db.session.commit()
    with patch(
        'app._fetch_remote_submission_zip',
        side_effect=RuntimeError('upstream 503'),
    ):
        ok, msg = _verify_remote_submission_hash(remote_sub)
    assert not ok
    assert 'upstream 503' in msg


def test_remote_marked_remote_without_url_surfaces_as_error(
    remote_sub, db_session,
):
    remote_sub.remote_url = None
    db.session.commit()
    ok, msg = _verify_remote_submission_hash(remote_sub)
    assert not ok
    assert 'no remote_url' in msg.lower()


# ---------------------------------------------------------------------------
# UI: drift status renders with a danger badge + tooltip.
# ---------------------------------------------------------------------------


def test_lb_page_renders_drift_badge_for_remote_mismatch(
    client, db_session, logged_in_user,
):
    ds = Dataset(name='lb_drift_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='lb_drift_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    sub = Submission(
        name='drifted', leaderboard_id=lb.id,
        owner_user_id=logged_in_user.id,
        storage_mode='remote',
        remote_url='https://example.test/sub.zip',
        content_hash=hashlib.sha256(b'old').hexdigest(),
        processing_status='Error: submission file changed; please resubmit',
    )
    db.session.add(sub); db.session.commit()
    resp = client.get(f'/leaderboard/{lb.id}')
    body = resp.data.decode()
    # Danger badge styling applied; shield-exclamation icon present.
    assert 'bg-danger' in body
    assert 'bi-shield-exclamation' in body
    # Tooltip mentions the SHA-256 contract so the user knows what changed.
    assert 'SHA-256' in body or 'submission time' in body
