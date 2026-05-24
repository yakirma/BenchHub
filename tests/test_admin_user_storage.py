"""Coverage for /admin/user_storage — the per-user storage view
admins use to spot a single user hoarding bytes."""
import pytest

from app import Dataset, Submission, User, db


@pytest.fixture
def admin(db_session):
    u = User(email='ux_admin@bench.local', display_name='ux_admin',
             oauth_provider='github', oauth_sub='ux-1', is_admin=True)
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def regular(db_session):
    u = User(email='ux_reg@bench.local', display_name='ux_reg',
             oauth_provider='github', oauth_sub='ux-2')
    db.session.add(u); db.session.commit()
    return u


def test_user_storage_requires_admin(client, db_session, regular):
    """Non-admin users get 403 — this is admin-only intel."""
    with client.session_transaction() as sess:
        sess['user_id'] = regular.id
    r = client.get('/admin/user_storage')
    assert r.status_code == 403


def test_user_storage_anon_redirects(client, db_session):
    """Anonymous users get bounced to /login (the @login_required gate
    fires before the admin check)."""
    r = client.get('/admin/user_storage', follow_redirects=False)
    assert r.status_code == 302
    assert '/login' in r.headers['Location']


def test_user_storage_lists_users_with_bytes(client, db_session, admin, regular):
    """Admin sees a row per user with their summed dataset bytes."""
    db.session.add(Dataset(
        name='big', owner_user_id=regular.id, storage_bytes=5_000_000,
    ))
    db.session.add(Dataset(
        name='tiny', owner_user_id=regular.id, storage_bytes=2_000_000,
    ))
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id
    body = client.get('/admin/user_storage').data.decode('utf-8')
    assert 'ux_reg@bench.local' in body
    # 7 MB total for `regular` should land near 6.7 MB depending on
    # the fmt_bytes precision — assert the integer-prefix part.
    assert '6.7 MB' in body or '7.0 MB' in body or '6.68 MB' in body or '7 MB' in body.replace('\n', ' ')
    # `regular` owns 2 datasets — the row should somewhere render "2 / <cap>".
    import re
    assert re.search(r'>\s*2\s*/\s*\d+', body), \
        "expected the regular user's dataset count to render as '2 / <cap>'"


def test_user_storage_includes_users_with_zero_datasets(client, db_session, admin, regular):
    """LEFT JOIN means a user with no datasets still appears, with 0
    bytes — useful for admin auditing of who's just signed up."""
    # `regular` has no datasets.
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id
    body = client.get('/admin/user_storage').data.decode('utf-8')
    assert 'ux_reg@bench.local' in body
    assert 'ux_admin@bench.local' in body


def test_user_storage_marks_admins_unlimited(client, db_session, admin):
    """Admins bypass the storage cap inside check_quota; the per-user
    table should reflect that — show ∞ instead of the cap row value
    (which would mislead the viewer into thinking the cap applies)."""
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id
    body = client.get('/admin/user_storage').data.decode('utf-8')
    # The infinity icon row sits in the cap column for admin rows.
    assert 'bi-infinity' in body


def test_user_storage_counts_submissions_24h(client, db_session, admin, regular):
    """Trailing-24h submission count surfaces per user so an admin can
    spot a runaway uploader. Submissions older than 24h must not count."""
    from datetime import datetime, timedelta
    from app import Leaderboard
    # Submissions need a leaderboard FK; spin up a throwaway one.
    lb = Leaderboard(name='sub_count_lb', summary_metrics='',
                     owner_user_id=admin.id)
    db.session.add(lb); db.session.flush()
    db.session.add(Submission(
        owner_user_id=regular.id, leaderboard_id=lb.id, name='fresh',
        upload_date=datetime.utcnow() - timedelta(hours=2),
    ))
    db.session.add(Submission(
        owner_user_id=regular.id, leaderboard_id=lb.id, name='stale',
        upload_date=datetime.utcnow() - timedelta(days=3),
    ))
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = admin.id
    body = client.get('/admin/user_storage?sort=subs').data.decode('utf-8')
    # Fresh sub counts (1) and stale doesn't — assert the row carries
    # the count somewhere as "1 / <cap>".
    import re
    assert re.search(r'>\s*1\s*/\s*\d+', body), (
        "expected the regular user's 24h sub count to render as '1 / <cap>'"
    )
