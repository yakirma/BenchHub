"""Split-quota behaviour: every user has two storage budgets — a 100 GB
public bucket (datasets + LB materialisations whose visibility=='public')
and a 10 GB private bucket (everything else). Writes are charged to the
bucket implied by the row's visibility; the publish-flip routes
pre-flight the public bucket before allowing private→public."""
from __future__ import annotations

import io
import zipfile

from app import (
    Dataset,
    Leaderboard,
    LeaderboardMaterialization,
    User,
    check_quota,
    db,
    quota_cap_for,
    storage_used_bytes,
)


def _mk_user(email, *, public_cap=100 * 1024 ** 3, private_cap=10 * 1024 ** 3):
    u = User(email=email, display_name=email.split('@')[0],
             oauth_provider='github', oauth_sub=email,
             is_admin=False,
             quota_max_storage_bytes=10 * 1024 ** 3,
             quota_public_max_bytes=public_cap,
             quota_private_max_bytes=private_cap,
             quota_max_datasets=50)
    db.session.add(u); db.session.commit()
    return u


def test_defaults_are_100gb_public_10gb_private(db_session):
    u = _mk_user('defaults@bench.local')
    assert quota_cap_for(u, 'public') == 100 * 1024 ** 3
    assert quota_cap_for(u, 'private') == 10 * 1024 ** 3


def test_unlisted_counts_against_private_bucket(db_session):
    u = _mk_user('unl@bench.local')
    assert quota_cap_for(u, 'unlisted') == 10 * 1024 ** 3
    assert quota_cap_for(u, 'private') == 10 * 1024 ** 3
    assert quota_cap_for(u, 'public') == 100 * 1024 ** 3


def test_storage_used_bytes_partitions_by_visibility(db_session):
    u = _mk_user('partition@bench.local')
    db.session.add_all([
        Dataset(name='pub_ds_a', visibility='public',
                owner_user_id=u.id, storage_bytes=200),
        Dataset(name='pub_ds_b', visibility='public',
                owner_user_id=u.id, storage_bytes=300),
        Dataset(name='priv_ds',  visibility='private',
                owner_user_id=u.id, storage_bytes=50),
        Dataset(name='unl_ds',   visibility='unlisted',
                owner_user_id=u.id, storage_bytes=10),
    ])
    db.session.commit()
    assert storage_used_bytes(u, visibility='public') == 500
    assert storage_used_bytes(u, visibility='private') == 60  # unlisted ⊕ private
    assert storage_used_bytes(u) == 560  # legacy callers see the sum


def test_lb_materialization_charges_lb_owner_by_lb_visibility(db_session):
    u = _mk_user('mat@bench.local')
    ds = Dataset(name='mat_ds', visibility='public',
                 owner_user_id=u.id, storage_bytes=100)
    db.session.add(ds); db.session.flush()
    pub_lb = Leaderboard(name='pub_lb', visibility='public',
                         owner_user_id=u.id)
    priv_lb = Leaderboard(name='priv_lb', visibility='private',
                          owner_user_id=u.id)
    pub_lb.datasets.append(ds); priv_lb.datasets.append(ds)
    db.session.add_all([pub_lb, priv_lb]); db.session.flush()
    db.session.add_all([
        LeaderboardMaterialization(
            leaderboard_id=pub_lb.id, status='ready', storage_bytes=900,
            sample_cap=-1, sampling='random', sampling_seed=42),
        LeaderboardMaterialization(
            leaderboard_id=priv_lb.id, status='ready', storage_bytes=400,
            sample_cap=-1, sampling='random', sampling_seed=42),
    ])
    db.session.commit()
    # public: 100 (dataset) + 900 (pub LB mat); private: 0 + 400 (priv LB mat)
    assert storage_used_bytes(u, visibility='public') == 1000
    assert storage_used_bytes(u, visibility='private') == 400


def test_check_quota_charges_public_bucket_for_public_write(db_session):
    # 1 GB public cap, 10 GB private cap. 2 GB write to public → reject;
    # the same 2 GB write to private → ok.
    u = _mk_user('ck@bench.local',
                 public_cap=1 * 1024 ** 3,
                 private_cap=10 * 1024 ** 3)
    big = 2 * 1024 ** 3
    ok, msg = check_quota(u, kind='dataset_create',
                          incoming_bytes=big, visibility='public')
    assert ok is False
    assert 'public' in msg.lower()
    ok, _ = check_quota(u, kind='dataset_create',
                       incoming_bytes=big, visibility='private')
    assert ok is True


def test_check_quota_charges_private_bucket_for_unlisted(db_session):
    # 50 GB private cap, 100 MB public cap. unlisted 30 GB write → ok (private).
    u = _mk_user('unl_quota@bench.local',
                 public_cap=100 * 1024 * 1024,
                 private_cap=50 * 1024 ** 3)
    ok, _ = check_quota(u, kind='dataset_create',
                       incoming_bytes=30 * 1024 ** 3, visibility='unlisted')
    assert ok is True


def test_publish_flip_preflight_rejects_when_public_bucket_full(
        client, db_session):
    """A private→public flip on a dataset gets blocked if the public
    bucket can't absorb the dataset's bytes."""
    # 10 MB public cap, big private cap. 50 MB private dataset → can't
    # be published.
    u = _mk_user('flip@bench.local',
                 public_cap=10 * 1024 * 1024,
                 private_cap=10 * 1024 ** 3)
    ds = Dataset(name='flip_ds', visibility='private',
                 owner_user_id=u.id, storage_bytes=50 * 1024 * 1024)
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = u.id
    r = client.post(f'/dataset/{ds.id}/visibility',
                    data={'visibility': 'public'},
                    follow_redirects=True)
    assert r.status_code == 200
    body = r.data.decode()
    assert 'Can&#39;t publish' in body or "Can't publish" in body
    # Visibility is unchanged in the DB.
    db.session.refresh(ds)
    assert ds.visibility == 'private'


def test_publish_flip_succeeds_when_public_bucket_has_room(
        client, db_session):
    u = _mk_user('flipok@bench.local')
    ds = Dataset(name='flipok_ds', visibility='private',
                 owner_user_id=u.id, storage_bytes=1024)
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = u.id
    r = client.post(f'/dataset/{ds.id}/visibility',
                    data={'visibility': 'public'},
                    follow_redirects=False)
    assert r.status_code == 302
    db.session.refresh(ds)
    assert ds.visibility == 'public'
