"""Phase 6 Slice 1 — landing page tests.

Covers:
- `/` renders the landing template (anonymous + logged-in).
- CTA differs based on auth state.
- Featured leaderboards section honors visibility (public + legacy NULL
  owner ok; private + unlisted hidden).
- Featured ordering is by recent submission count.
- Empty-state when there are no public leaderboards.
"""
from datetime import datetime, timedelta

import pytest

from app import (
    Dataset,
    Leaderboard,
    Sample,
    Submission,
    User,
    db,
)


def _mk_project(*args, **kwargs):
    return None



def _mk_dataset(name='ds'):
    ds = Dataset(name=name)
    db.session.add(ds)
    db.session.flush()
    return ds


def _mk_leaderboard(project, dataset, name, *, visibility='public', owner_user_id=None):
    lb = Leaderboard(
        name=name,
        summary_metrics='',
        owner_user_id=owner_user_id,
        visibility=visibility,
    )
    lb.datasets.append(dataset)
    db.session.add(lb)
    db.session.flush()
    return lb


def _mk_submission(lb, *, days_ago=1, archived=False):
    sub = Submission(
        name=f"sub_{lb.id}_{days_ago}",
        leaderboard_id=lb.id,
        upload_date=datetime.utcnow() - timedelta(days=days_ago),
        is_archived=archived,
    )
    db.session.add(sub)
    db.session.flush()
    return sub


# ---------------------------------------------------------------------------
# Basic rendering + CTA
# ---------------------------------------------------------------------------


def test_landing_renders_for_anonymous(client):
    resp = client.get("/")
    assert resp.status_code == 200
    # Hero headline — tracks the current marketing pitch.
    assert b"Build, run, and" in resp.data
    assert b"How it works" in resp.data


def test_landing_shows_login_cta_when_anonymous(client):
    resp = client.get("/")
    # The bottom CTA now collapses to a single button to /login (which
    # hosts GitHub / Google / email-code options).
    assert b"Log in / Sign up" in resp.data
    # No "dashboard" CTA when logged out.
    assert b"Go to your dashboard" not in resp.data


def test_landing_redirects_to_home_when_logged_in(auth_client):
    """Signed-in visitors shouldn't see the marketing landing —
    they jump straight to their dashboard. Was a 200 with a
    'Go to your dashboard' CTA before; now it's a 302."""
    resp = auth_client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/home")


# ---------------------------------------------------------------------------
# Featured leaderboards
# ---------------------------------------------------------------------------


def test_featured_includes_public_leaderboards(client, db_session):
    p = _mk_project("featured_proj")
    ds = _mk_dataset("featured_ds")
    lb = _mk_leaderboard(p, ds, "lb_public", visibility="public")
    _mk_submission(lb)
    db_session.commit()

    resp = client.get("/")
    assert b"lb_public" in resp.data


def _stranger(db_session):
    """Owner for non-public leaderboards. visible_in_list treats NULL
    owner as legacy-public — so to test the private/unlisted branch we
    need a real user behind the row."""
    u = User(
        email="lb_owner@example.com", display_name="Owner",
        oauth_provider="github", oauth_sub="featured-owner-1",
    )
    db_session.add(u)
    db_session.commit()
    return u


def test_featured_excludes_private_leaderboards(client, db_session):
    """Private leaderboards stay off the public landing — even if they had
    activity in the window."""
    owner = _stranger(db_session)
    p = _mk_project("priv_proj", owner_user_id=owner.id)
    ds = _mk_dataset("priv_ds")
    lb = _mk_leaderboard(p, ds, "lb_private", visibility="private", owner_user_id=owner.id)
    for _ in range(5):
        _mk_submission(lb)  # ensure it'd otherwise rank
    db_session.commit()

    resp = client.get("/")
    assert b"lb_private" not in resp.data


def test_featured_excludes_unlisted_leaderboards(client, db_session):
    """Unlisted means URL-only — no list pages, including the public
    landing's featured grid."""
    owner = _stranger(db_session)
    p = _mk_project("unl_proj", owner_user_id=owner.id)
    ds = _mk_dataset("unl_ds")
    lb = _mk_leaderboard(p, ds, "lb_unlisted", visibility="unlisted", owner_user_id=owner.id)
    _mk_submission(lb)
    db_session.commit()

    resp = client.get("/")
    assert b"lb_unlisted" not in resp.data


def test_featured_includes_legacy_null_owner(client, db_session):
    """Pre-Phase-1 leaderboards have owner_user_id IS NULL. The visibility
    filter treats them as public until the backfill assigns owners."""
    p = _mk_project("legacy_proj", owner_user_id=None)
    ds = _mk_dataset("legacy_ds")
    lb = _mk_leaderboard(p, ds, "lb_legacy", visibility="public", owner_user_id=None)
    _mk_submission(lb)
    db_session.commit()

    resp = client.get("/")
    assert b"lb_legacy" in resp.data


def test_featured_orders_by_recent_activity(client, db_session):
    """Three public LBs with 1 / 5 / 2 recent submissions. Top-3 should
    have the busiest first; less-active ones still listed but lower."""
    p = _mk_project("rank_proj")
    ds = _mk_dataset("rank_ds")
    quiet = _mk_leaderboard(p, ds, "lb_quiet")
    busy = _mk_leaderboard(p, ds, "lb_busy")
    medium = _mk_leaderboard(p, ds, "lb_medium")

    _mk_submission(quiet)
    for _ in range(5):
        _mk_submission(busy)
    for _ in range(2):
        _mk_submission(medium)
    db_session.commit()

    body = client.get("/").data.decode("utf-8", errors="ignore")
    # All three should appear.
    assert "lb_busy" in body
    assert "lb_medium" in body
    assert "lb_quiet" in body
    # Busy comes before medium, which comes before quiet.
    assert body.index("lb_busy") < body.index("lb_medium") < body.index("lb_quiet")


def test_featured_ignores_old_activity(client, db_session):
    """A leaderboard whose only submissions are >30 days old shouldn't out-rank
    a leaderboard with one recent submission."""
    p = _mk_project("age_proj")
    ds = _mk_dataset("age_ds")
    stale = _mk_leaderboard(p, ds, "lb_stale")
    recent = _mk_leaderboard(p, ds, "lb_recent")

    for _ in range(10):
        _mk_submission(stale, days_ago=60)  # outside the window
    _mk_submission(recent, days_ago=1)
    db_session.commit()

    body = client.get("/").data.decode("utf-8", errors="ignore")
    # Both still listed (we don't FILTER by activity, only ORDER); recent
    # should rank above stale because recent_count(stale)=0 in the window.
    assert body.index("lb_recent") < body.index("lb_stale")


def test_featured_ignores_archived_submissions(client, db_session):
    """Archived submissions don't count toward the activity window — they're
    hidden from leaderboard views by default and shouldn't pump the ranking."""
    p = _mk_project("arch_proj")
    ds = _mk_dataset("arch_ds")
    archived_only = _mk_leaderboard(p, ds, "lb_archived_only")
    live = _mk_leaderboard(p, ds, "lb_live")

    for _ in range(10):
        _mk_submission(archived_only, archived=True)
    _mk_submission(live)
    db_session.commit()

    body = client.get("/").data.decode("utf-8", errors="ignore")
    assert body.index("lb_live") < body.index("lb_archived_only")


def test_landing_empty_state_when_no_public_leaderboards(client, db_session):
    """No leaderboards at all → friendly empty state, not a broken grid."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No public leaderboards yet" in resp.data
