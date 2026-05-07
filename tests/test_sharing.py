"""Per-row collaborator sharing on Dataset and Leaderboard."""
import pytest

from app import Dataset, Leaderboard, User, db


@pytest.fixture
def collaborator(db_session):
    u = User(
        email='friend@example.com', display_name='Friend',
        oauth_provider='github', oauth_sub='fr-1',
    )
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def stranger(db_session):
    u = User(
        email='stranger-share@example.com', display_name='Stranger',
        oauth_provider='github', oauth_sub='st-share-1',
    )
    db.session.add(u); db.session.commit()
    return u


def test_owner_shares_dataset_by_email(auth_client, logged_in_user, collaborator, db_session):
    ds = Dataset(name='priv_ds', owner_user_id=logged_in_user.id, visibility='private')
    db.session.add(ds); db.session.commit()

    resp = auth_client.post(
        f'/dataset/{ds.id}/share',
        data={'email': 'friend@example.com'},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    db.session.refresh(ds)
    assert collaborator in ds.collaborators


def test_share_unknown_email_warns_no_create(auth_client, logged_in_user, db_session):
    ds = Dataset(name='priv_ds_2', owner_user_id=logged_in_user.id, visibility='private')
    db.session.add(ds); db.session.commit()

    resp = auth_client.post(
        f'/dataset/{ds.id}/share',
        data={'email': 'never-signed-in@example.com'},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b'No BenchHub user' in resp.data
    db.session.refresh(ds)
    assert ds.collaborators == []


def test_collaborator_can_view_private_dataset(client, db_session, logged_in_user, collaborator):
    """The whole point of sharing: a private dataset becomes visible to
    the listed collaborator (logged in as them)."""
    ds = Dataset(
        name='shared_priv', owner_user_id=logged_in_user.id,
        visibility='private',
    )
    ds.collaborators.append(collaborator)
    db.session.add(ds); db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = collaborator.id
    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    assert b'shared_priv' in resp.data


def test_non_collaborator_still_404s_on_private(client, db_session, logged_in_user, stranger):
    """Sanity: sharing with one user doesn't open the door to everyone."""
    ds = Dataset(
        name='still_priv', owner_user_id=logged_in_user.id,
        visibility='private',
    )
    db.session.add(ds); db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = stranger.id
    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 404


def test_visible_in_list_includes_shared_dataset(client, db_session, logged_in_user, collaborator, app):
    """Collaborator should see the shared private dataset on /datasets.
    Add a Sample + on-disk folder so the inline prune-orphans pass on
    /datasets doesn't sweep the row before the list renders."""
    import os
    from app import Sample
    ds = Dataset(
        name='shows_in_list', owner_user_id=logged_in_user.id,
        visibility='private',
    )
    ds.collaborators.append(collaborator)
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    db.session.commit()
    folder = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', 'shows_in_list')
    os.makedirs(folder, exist_ok=True)

    with client.session_transaction() as sess:
        sess['user_id'] = collaborator.id
    body = client.get('/datasets').data
    assert b'shows_in_list' in body


def test_unshare_removes_collaborator(auth_client, logged_in_user, collaborator, db_session):
    ds = Dataset(name='unshare_me', owner_user_id=logged_in_user.id, visibility='private')
    ds.collaborators.append(collaborator)
    db.session.add(ds); db.session.commit()
    assert collaborator in ds.collaborators

    resp = auth_client.post(
        f'/dataset/{ds.id}/unshare/{collaborator.id}',
        follow_redirects=False,
    )
    assert resp.status_code == 302
    db.session.refresh(ds)
    assert collaborator not in ds.collaborators


def test_share_route_owner_only(client, db_session, logged_in_user, stranger, collaborator):
    """A non-owner non-admin can't share someone else's dataset."""
    ds = Dataset(name='not_yours_to_share', owner_user_id=logged_in_user.id,
                 visibility='private')
    db.session.add(ds); db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = stranger.id
    resp = client.post(
        f'/dataset/{ds.id}/share',
        data={'email': 'friend@example.com'},
    )
    assert resp.status_code == 403


def test_leaderboard_share_round_trip(auth_client, logged_in_user, collaborator, db_session):
    ds = Dataset(name='lb_share_ds', owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(
        name='priv_lb', summary_metrics='',
        owner_user_id=logged_in_user.id, visibility='private',
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    auth_client.post(
        f'/leaderboard/{lb.id}/share',
        data={'email': 'friend@example.com'},
    )
    db.session.refresh(lb)
    assert collaborator in lb.collaborators
