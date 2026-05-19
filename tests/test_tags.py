"""Discovery tags on Dataset + Leaderboard, and the /explore tag cloud."""
import pytest

from app import Dataset, Leaderboard, Tag, User, db, _resolve_tags


def test_resolve_tags_creates_new_and_reuses_existing(client, db_session):
    db.session.add(Tag(name='depth'))
    db.session.commit()

    tags = _resolve_tags("Depth, Segmentation, depth")
    assert sorted(t.name for t in tags) == ['depth', 'segmentation']
    # Existing 'depth' was reused, not duplicated.
    assert Tag.query.filter_by(name='depth').count() == 1


def test_update_dataset_tags_owner_only(auth_client, logged_in_user, db_session):
    ds = Dataset(name='tagged_ds', owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.commit()

    resp = auth_client.post(
        f'/dataset/{ds.id}/update_tags',
        data={'tags': 'depth, indoor'},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    db.session.refresh(ds)
    assert sorted(t.name for t in ds.tags) == ['depth', 'indoor']


def test_update_dataset_tags_blocks_non_owner(auth_client, logged_in_user, db_session):
    other = User(
        email='nonowner@example.com', display_name='NO',
        oauth_provider='github', oauth_sub='no-1',
    )
    db.session.add(other); db.session.flush()
    ds = Dataset(name='owned_by_other', owner_user_id=other.id)
    db.session.add(ds); db.session.commit()

    resp = auth_client.post(
        f'/dataset/{ds.id}/update_tags',
        data={'tags': 'rude'},
    )
    assert resp.status_code == 403


def test_dataset_tags_render_on_dataset_list(client, db_session):
    """Tags must show up on /datasets, not just on the detail page."""
    ds = Dataset(name='listed_with_tags', visibility='public')
    db.session.add(ds); db.session.flush()
    ds.samples.append(__import__('app').Sample(dataset_id=ds.id, name='s1'))
    seg_tag = Tag(name='segmentation')
    db.session.add(seg_tag); db.session.flush()
    ds.tags.append(seg_tag)
    db.session.commit()
    # Need a folder so the inline prune doesn't sweep it.
    import os
    from app import app as flask_app
    folder = os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', 'listed_with_tags')
    os.makedirs(folder, exist_ok=True)

    body = client.get('/datasets').data
    assert b'listed_with_tags' in body
    assert b'segmentation' in body


def test_dataset_tags_render_on_home_card(auth_client, logged_in_user, db_session):
    """Owned-dataset cards on /home show tag chips."""
    ds = Dataset(name='home_with_tags', owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    depth_tag = Tag(name='depth')
    db.session.add(depth_tag); db.session.flush()
    ds.tags.append(depth_tag)
    db.session.commit()

    body = auth_client.get('/home').data
    assert b'home_with_tags' in body
    assert b'depth' in body


