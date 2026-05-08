"""Phase 6 Slice 2 — discoverability tests.

Covers /explore (public leaderboard browse), /u/<user_id> (public profile),
the anon-friendly /dataset/<id> view, and the SEO meta block on those
pages plus the existing leaderboard view.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stranger(db_session):
    u = User(
        email="stranger@example.com",
        display_name="Stranger",
        oauth_provider="github",
        oauth_sub="stranger-1",
    )
    db_session.add(u)
    db_session.commit()
    return u


def _mk_proj(*args, **kwargs):
    return None



def _mk_dataset(name, *, owner_user_id=None, visibility='public'):
    ds = Dataset(name=name, owner_user_id=owner_user_id, visibility=visibility)
    db.session.add(ds)
    db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    db.session.commit()
    return ds


def _mk_lb(project, dataset, name, *, owner_user_id=None, visibility='public',
           canonicality='public'):
    # Tests in this module assert /explore behavior, so default to
    # canonicality='public' — /explore filters out personal LBs by
    # design (see admin_promote_leaderboard).
    lb = Leaderboard(
        name=name, summary_metrics='',
        owner_user_id=owner_user_id, visibility=visibility,
        canonicality=canonicality,
    )
    lb.datasets.append(dataset)
    db.session.add(lb)
    db.session.flush()
    return lb


def _mk_sub(lb, *, owner_user_id=None, days_ago=1, archived=False):
    s = Submission(
        name=f"sub_{lb.id}_{days_ago}",
        leaderboard_id=lb.id,
        owner_user_id=owner_user_id,
        upload_date=datetime.utcnow() - timedelta(days=days_ago),
        is_archived=archived,
    )
    db.session.add(s)
    db.session.flush()
    return s


# ===========================================================================
# /explore
# ===========================================================================


def test_explore_renders_with_no_leaderboards(client, db_session):
    resp = client.get("/explore")
    assert resp.status_code == 200
    assert b"Explore leaderboards" in resp.data
    assert b"Nothing matches" in resp.data or b"No public leaderboards" in resp.data


def test_explore_lists_public_leaderboards(client, db_session):
    p = _mk_proj("ep")
    ds = _mk_dataset("eds")
    _mk_lb(p, ds, "lb_alpha")
    _mk_lb(p, ds, "lb_beta")
    db_session.commit()

    body = client.get("/explore").data
    assert b"lb_alpha" in body
    assert b"lb_beta" in body


def test_explore_excludes_private_leaderboards(client, db_session, stranger):
    p = _mk_proj("ep_priv", owner_user_id=stranger.id)
    ds = _mk_dataset("eds_priv", owner_user_id=stranger.id)
    _mk_lb(p, ds, "lb_pub")
    _mk_lb(p, ds, "lb_priv", owner_user_id=stranger.id, visibility='private')
    _mk_lb(p, ds, "lb_unl", owner_user_id=stranger.id, visibility='unlisted')
    db_session.commit()

    body = client.get("/explore").data
    assert b"lb_pub" in body
    assert b"lb_priv" not in body
    assert b"lb_unl" not in body


def test_explore_q_param_filters_by_name(client, db_session):
    p = _mk_proj("ep_q")
    ds = _mk_dataset("eds_q")
    _mk_lb(p, ds, "image_classification_2024")
    _mk_lb(p, ds, "depth_estimation_dtof")
    db_session.commit()

    body = client.get("/explore?q=depth").data
    assert b"depth_estimation_dtof" in body
    assert b"image_classification_2024" not in body


def test_explore_sort_by_recent(client, db_session):
    p = _mk_proj("ep_recent")
    ds = _mk_dataset("eds_recent")
    # Create LBs in known order so newest is unambiguous.
    older = _mk_lb(p, ds, "lb_older")
    newer = _mk_lb(p, ds, "lb_newer")
    older.upload_date = datetime.utcnow() - timedelta(days=10)
    newer.upload_date = datetime.utcnow()
    db_session.commit()

    body = client.get("/explore?sort=recent").data.decode()
    assert body.index("lb_newer") < body.index("lb_older")


def test_explore_sort_by_popular(client, db_session):
    p = _mk_proj("ep_pop")
    ds = _mk_dataset("eds_pop")
    quiet = _mk_lb(p, ds, "lb_quiet")
    loud = _mk_lb(p, ds, "lb_loud")
    _mk_sub(quiet)
    for _ in range(5):
        _mk_sub(loud)
    db_session.commit()

    body = client.get("/explore?sort=popular").data.decode()
    assert body.index("lb_loud") < body.index("lb_quiet")


def test_explore_invalid_sort_falls_back_to_default(client, db_session):
    """Anything not in {activity, recent, popular} reverts to activity —
    pin so a typo or hostile query string can't break the page."""
    p = _mk_proj("ep_bad_sort")
    ds = _mk_dataset("eds_bad_sort")
    _mk_lb(p, ds, "lb_x")
    db_session.commit()

    resp = client.get("/explore?sort=DROP+TABLE")
    assert resp.status_code == 200
    assert b"lb_x" in resp.data


# ===========================================================================
# /u/<user_id>
# ===========================================================================


def test_user_profile_404_for_unknown_id(client):
    resp = client.get("/u/9999")
    assert resp.status_code == 404


def test_user_profile_renders_with_basic_info(client, db_session, stranger):
    resp = client.get(f"/u/{stranger.id}")
    assert resp.status_code == 200
    body = resp.data
    assert stranger.display_name.encode() in body
    # Bio chrome is present even with no content yet.
    assert b"Public datasets" in body
    assert b"Public leaderboards" in body
    assert b"Recent submissions" in body


def test_user_profile_lists_users_public_assets(client, db_session, stranger):
    p = _mk_proj("up_proj", owner_user_id=stranger.id)
    ds_pub = _mk_dataset("ds_user_pub", owner_user_id=stranger.id)
    ds_priv = _mk_dataset("ds_user_priv", owner_user_id=stranger.id, visibility='private')
    lb_pub = _mk_lb(p, ds_pub, "lb_user_pub", owner_user_id=stranger.id)
    lb_priv = _mk_lb(p, ds_pub, "lb_user_priv", owner_user_id=stranger.id, visibility='private')
    db_session.commit()

    body = client.get(f"/u/{stranger.id}").data
    # Public stuff shows up.
    assert b"ds_user_pub" in body
    assert b"lb_user_pub" in body
    # Private stuff stays hidden, even on the user's own profile (the
    # public profile is for public stuff; private lives on the dashboard).
    assert b"ds_user_priv" not in body
    assert b"lb_user_priv" not in body


def test_user_profile_recent_submissions_only_to_public_leaderboards(
    client, db_session, stranger
):
    """Don't leak that the user submitted to a private leaderboard, even
    if the submission itself isn't directly access-gated."""
    p = _mk_proj("up_subs", owner_user_id=stranger.id)
    ds = _mk_dataset("up_subs_ds", owner_user_id=stranger.id)
    lb_pub = _mk_lb(p, ds, "pub_lb", owner_user_id=stranger.id)
    lb_priv = _mk_lb(p, ds, "priv_lb", owner_user_id=stranger.id, visibility='private')

    _mk_sub(lb_pub, owner_user_id=stranger.id)
    _mk_sub(lb_priv, owner_user_id=stranger.id)
    db_session.commit()

    body = client.get(f"/u/{stranger.id}").data
    assert b"pub_lb" in body
    assert b"priv_lb" not in body


def test_user_profile_excludes_archived_submissions(client, db_session, stranger):
    p = _mk_proj("arch_p", owner_user_id=stranger.id)
    ds = _mk_dataset("arch_ds", owner_user_id=stranger.id)
    lb = _mk_lb(p, ds, "arch_lb", owner_user_id=stranger.id)
    live = _mk_sub(lb, owner_user_id=stranger.id, archived=False)
    archived = _mk_sub(lb, owner_user_id=stranger.id, archived=True, days_ago=2)
    db_session.commit()

    body = client.get(f"/u/{stranger.id}").data.decode()
    assert live.name in body
    assert archived.name not in body


# ===========================================================================
# /dataset/<id> — anon access
# ===========================================================================


def test_anon_can_view_public_dataset(client, db_session):
    """Was crashing pre-fix because the template builds project-scoped URLs
    (download_sample, custom_field_image) that need g.current_project."""
    _mk_proj("anon_p")  # provides the fallback project context
    ds = _mk_dataset("anon_ds_pub")
    db_session.commit()

    resp = client.get(f"/dataset/{ds.id}")
    assert resp.status_code == 200
    assert b"anon_ds_pub" in resp.data


def test_anon_cannot_view_private_dataset(client, db_session, stranger):
    _mk_proj("anon_p2")
    ds = _mk_dataset("private_ds_secret", owner_user_id=stranger.id, visibility='private')
    db_session.commit()

    resp = client.get(f"/dataset/{ds.id}")
    # @visibility_required → 404 not 403 (don't leak existence).
    assert resp.status_code == 404


# ===========================================================================
# SEO meta block
# ===========================================================================


def test_landing_emits_og_meta(client):
    body = client.get("/").data
    assert b'<meta property="og:title"' in body
    assert b'<meta name="description"' in body
    # Title content reflects the marketing pitch, not the generic default.
    assert b'Benchmark your model in 60 seconds' in body


def test_explore_emits_og_meta(client, db_session):
    body = client.get("/explore").data
    assert b'<meta property="og:title"' in body
    assert b'Explore leaderboards' in body


def test_user_profile_emits_og_meta(client, db_session, stranger):
    body = client.get(f"/u/{stranger.id}").data
    assert b'<meta property="og:title"' in body
    # Title should mention the user's display name.
    assert stranger.display_name.encode() in body


def test_dataset_view_emits_og_meta(client, db_session):
    _mk_proj("seo_p")
    ds = _mk_dataset("seo_meta_ds")
    db_session.commit()

    body = client.get(f"/dataset/{ds.id}").data
    assert b'<meta property="og:title"' in body
    assert b'seo_meta_ds' in body
