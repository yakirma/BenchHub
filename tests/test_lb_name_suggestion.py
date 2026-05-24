"""LB-creation forms pre-fill the leaderboard_name input with a
guaranteed-free suggestion so the common case is a one-click submit.
"""
from __future__ import annotations

import os

from app import (
    Dataset,
    DatasetField,
    Leaderboard,
    User,
    _suggest_free_lb_name,
    app as flask_app,
    db,
)


def _seed_dataset(name='cifar10'):
    user = User(email='lbname@bench.local', display_name='lbname',
                oauth_provider='github', oauth_sub='lbname-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name=name, visibility='public', owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='label',
                                kind='label', role='gt'))
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    db.session.commit()
    return user, ds


def test_suggest_free_lb_name_simple_case(db_session):
    """No existing LBs → returns `<base>_benchmark` unchanged."""
    assert _suggest_free_lb_name('cifar10') == 'cifar10_benchmark'


def test_suggest_free_lb_name_handles_collision(db_session):
    """When `<base>_benchmark` is taken, suggest `<base>_benchmark_2`."""
    db.session.add(Leaderboard(name='cifar10_benchmark', visibility='public'))
    db.session.commit()
    assert _suggest_free_lb_name('cifar10') == 'cifar10_benchmark_2'


def test_suggest_free_lb_name_walks_through_multiple_collisions(db_session):
    """Sequential collisions land on the next free integer suffix."""
    for n in ('cifar10_benchmark', 'cifar10_benchmark_2', 'cifar10_benchmark_3'):
        db.session.add(Leaderboard(name=n, visibility='public'))
    db.session.commit()
    assert _suggest_free_lb_name('cifar10') == 'cifar10_benchmark_4'


def test_suggest_free_lb_name_handles_blank_base(db_session):
    """Empty/whitespace base falls back to `leaderboard_benchmark`."""
    assert _suggest_free_lb_name('') == 'leaderboard_benchmark'
    assert _suggest_free_lb_name('   ') == 'leaderboard_benchmark'


def test_dataset_view_prefills_lb_name_input(client, db_session):
    """The inline LB-creation form on /dataset/<id> ships an
    initial `value=` so the user doesn't need to type a name."""
    user, ds = _seed_dataset('mydata')
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    body = client.get(f'/dataset/{ds.id}/create_lb').data.decode()
    # The pre-fill is a name derived from the dataset name.
    assert 'value="mydata_benchmark"' in body


def test_create_lb_chooser_prefills_lb_name_input(client, db_session):
    """/create_lb pre-fills with a free name derived from the
    first-listed dataset (most-recently-uploaded)."""
    user, ds = _seed_dataset('nyu_depth_v2')
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
    body = client.get('/create_lb').data.decode()
    assert 'value="nyu_depth_v2_benchmark"' in body
