"""User-thumbnail rendering on dataset / LB / comparison surfaces.

Pin the move from the legacy git_author + AuthorProfile flow to
OAuth-authenticated User identity (display_name + avatar_url) so a
future refactor can't silently re-introduce the git-author pathway.
"""
from app import (
    Dataset, Leaderboard, Sample, Submission, User, db,
)


def _user_with_avatar(db_session, **overrides):
    base = dict(
        email='avatar@example.com', display_name='Avi Avatar',
        oauth_provider='github', oauth_sub='avatar-1',
        avatar_url='https://example.test/avatar.png',
    )
    base.update(overrides)
    u = User(**base)
    db.session.add(u); db.session.commit()
    return u


def test_dataset_view_renders_oauth_avatar_image_for_owner(
    client, db_session,
):
    """Owner has an OAuth avatar_url → the page embeds it as an <img>
    pointing at that URL, NOT the legacy initials placeholder."""
    owner = _user_with_avatar(db_session)
    ds = Dataset(name='oauth_ds', visibility='public', owner_user_id=owner.id)
    db.session.add(ds); db.session.commit()
    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    body = resp.data
    assert b'https://example.test/avatar.png' in body
    # Legacy git-author wiring is gone.
    assert b'author-avatar-init' not in body
    assert b'AUTHOR_PROFILES' not in body


def test_dataset_view_renders_initials_when_owner_has_no_avatar_url(
    client, db_session,
):
    """No avatar_url → fall back to display-name initials in a coloured
    circle. Server-rendered, no JS."""
    owner = _user_with_avatar(db_session,
                              email='ni@example.com', oauth_sub='ni-1',
                              display_name='Nora Initials',
                              avatar_url=None)
    ds = Dataset(name='no_avatar_ds', visibility='public',
                 owner_user_id=owner.id)
    db.session.add(ds); db.session.commit()
    resp = client.get(f'/dataset/{ds.id}')
    body = resp.data.decode()
    # Initials of "Nora Initials" → NI.
    assert '>NI<' in body or 'NI</a>' in body or 'NI</span>' in body
    # No legacy hook left over.
    assert 'author-avatar-init' not in body


def test_dataset_view_renders_unknown_placeholder_when_no_owner(
    client, db_session,
):
    """owner_user_id is NULL → render a neutral '?' chip rather than
    leak git_author into the avatar."""
    ds = Dataset(
        name='legacy_ds', visibility='public',
        # Old-style row: git_author present, no owner.
        git_author='Some <git@author>',
    )
    db.session.add(ds); db.session.commit()
    resp = client.get(f'/dataset/{ds.id}')
    body = resp.data.decode()
    assert '>?<' in body or '?</span>' in body
    # The git-author string MUST NOT have been used to build the avatar.
    # (It still appears as inline commit-metadata text — that's commit
    # provenance, not identity. Just check the avatar circle markup.)
    assert 'data-author=' not in body


def test_leaderboard_view_uses_owner_for_submission_avatars(
    client, db_session,
):
    """Submission rows show the uploader's OAuth avatar, not a hash
    of git_author."""
    owner = _user_with_avatar(db_session,
                              email='subowner@example.com',
                              oauth_sub='sub-1',
                              avatar_url='https://example.test/sub.png')
    ds = Dataset(name='lb_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='lb_for_avatar', summary_metrics='',
                     visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    sub = Submission(
        name='s1', leaderboard_id=lb.id,
        git_author='Old Git Author',  # legacy field still on the row
        owner_user_id=owner.id,
    )
    db.session.add(sub); db.session.commit()
    resp = client.get(f'/leaderboard/{lb.id}')
    body = resp.data
    assert b'https://example.test/sub.png' in body
    assert b'author-avatar-init' not in body


def test_base_template_no_longer_injects_legacy_author_profiles(
    client, db_session,
):
    """The window.AUTHOR_PROFILES JSON dump is gone — all callers were
    wired off the legacy avatar JS."""
    resp = client.get('/explore')
    assert b'AUTHOR_PROFILES' not in resp.data
