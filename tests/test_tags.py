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


def test_explore_tag_filter_narrows_to_matching_lbs(client, db_session):
    ds = Dataset(name='tags_ds', visibility='public')
    db.session.add(ds); db.session.flush()

    depth_tag = Tag(name='depth')
    db.session.add(depth_tag); db.session.flush()

    matched = Leaderboard(name='depth_lb', summary_metrics='', visibility='public')
    matched.datasets.append(ds)
    matched.tags.append(depth_tag)
    other = Leaderboard(name='untagged_lb', summary_metrics='', visibility='public')
    other.datasets.append(ds)
    db.session.add_all([matched, other]); db.session.commit()

    body = client.get('/explore?tag=depth').data
    assert b'depth_lb' in body
    assert b'untagged_lb' not in body


def test_explore_tag_cloud_renders_with_counts(client, db_session):
    ds = Dataset(name='cloud_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    depth_tag = Tag(name='depth')
    seg_tag = Tag(name='segmentation')
    db.session.add_all([depth_tag, seg_tag]); db.session.flush()

    lb1 = Leaderboard(name='lb_d1', summary_metrics='', visibility='public')
    lb2 = Leaderboard(name='lb_d2', summary_metrics='', visibility='public')
    lb3 = Leaderboard(name='lb_seg', summary_metrics='', visibility='public')
    for lb in (lb1, lb2, lb3):
        lb.datasets.append(ds)
    lb1.tags.append(depth_tag)
    lb2.tags.append(depth_tag)
    lb3.tags.append(seg_tag)
    db.session.add_all([lb1, lb2, lb3]); db.session.commit()

    body = client.get('/explore').data.decode()
    # Both tags surface; depth has higher count → bigger SIZE tier.
    assert 'depth' in body
    assert 'segmentation' in body
    # depth (count=2) gets the top tier; segmentation (count=1) gets the bottom.
    # `hue-N` is independent of count (deterministic per name) — don't pin.
    import re as _re
    depth_anchor = _re.search(r'<a [^>]*class="tier-(\d+)[^"]*"[^>]*>\s*depth', body)
    seg_anchor = _re.search(r'<a [^>]*class="tier-(\d+)[^"]*"[^>]*>\s*segmentation', body)
    assert depth_anchor is not None and seg_anchor is not None
    assert int(depth_anchor.group(1)) > int(seg_anchor.group(1))


def test_explore_tag_cloud_assigns_distinct_colors_by_name(client, db_session):
    """Two tags at the SAME count should get different hue classes
    (color is per-name, not per-count) so they don't look identical."""
    ds = Dataset(name='hue_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    a = Tag(name='alpha-tag')
    b = Tag(name='beta-tag')
    db.session.add_all([a, b]); db.session.flush()
    lb_a = Leaderboard(name='lb_alpha_only', summary_metrics='', visibility='public')
    lb_a.datasets.append(ds); lb_a.tags.append(a)
    lb_b = Leaderboard(name='lb_beta_only', summary_metrics='', visibility='public')
    lb_b.datasets.append(ds); lb_b.tags.append(b)
    db.session.add_all([lb_a, lb_b]); db.session.commit()

    body = client.get('/explore').data.decode()
    import re as _re
    a_match = _re.search(r'class="tier-\d+ hue-(\d+)[^"]*"[^>]*>\s*alpha-tag', body)
    b_match = _re.search(r'class="tier-\d+ hue-(\d+)[^"]*"[^>]*>\s*beta-tag', body)
    assert a_match is not None and b_match is not None
    # Same count, different name → different hue (very high probability;
    # a CRC32 collision among two specific names mod 12 is unlikely
    # but not impossible — these two specific names don't collide).
    assert a_match.group(1) != b_match.group(1)


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


def test_explore_tag_filter_empty_when_no_matches(client, db_session):
    """Filtering on a tag with zero matches still renders the page,
    just with the empty-state copy."""
    resp = client.get('/explore?tag=nonexistent')
    assert resp.status_code == 200
    assert b'Nothing matches your filters yet' in resp.data
