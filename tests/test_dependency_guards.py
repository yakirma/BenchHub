"""Cross-user dependency guards: an object others depend on can't be
downgraded to private or deleted by its owner (admins bypass)."""
from app import Dataset, Leaderboard, Submission, User, db


def _user(email, sub, *, admin=False):
    u = User(email=email, display_name=email.split('@')[0],
             oauth_provider='github', oauth_sub=sub, is_admin=admin)
    db.session.add(u); db.session.commit()
    return u


def _login(client, user):
    with client.session_transaction() as s:
        s['user_id'] = user.id


def _ds(owner, name='dep-ds', vis='public'):
    d = Dataset(name=name, owner_user_id=owner.id, visibility=vis)
    db.session.add(d); db.session.commit()
    return d


def _lb(owner, name='dep-lb', vis='public', datasets=()):
    lb = Leaderboard(name=name, summary_metrics='', owner_user_id=owner.id,
                     visibility=vis)
    for d in datasets:
        lb.datasets.append(d)
    db.session.add(lb); db.session.commit()
    return lb


# --- dataset depended on by another user's leaderboard --------------------

def test_dataset_downgrade_blocked_when_foreign_lb_binds_it(client, db_session):
    a, b = _user('a@x.io', 'a'), _user('b@x.io', 'b')
    ds = _ds(a)
    _lb(b, datasets=[ds])                       # B's LB binds A's dataset
    _login(client, a)
    r = client.post(f'/dataset/{ds.id}/visibility', data={'visibility': 'private'})
    assert r.status_code == 302
    assert db.session.get(Dataset, ds.id).visibility == 'public'   # unchanged


def test_dataset_delete_blocked_when_foreign_lb_binds_it(client, db_session):
    a, b = _user('a2@x.io', 'a2'), _user('b2@x.io', 'b2')
    ds = _ds(a, name='dep-ds2')
    _lb(b, name='dep-lb2', datasets=[ds])
    _login(client, a)
    r = client.post(f'/dataset/{ds.id}/delete')
    assert r.status_code == 302
    assert db.session.get(Dataset, ds.id) is not None              # not deleted


def test_dataset_downgrade_allowed_when_only_own_lb(client, db_session):
    a = _user('a3@x.io', 'a3')
    ds = _ds(a, name='own-ds')
    _lb(a, name='own-lb', datasets=[ds])         # owner's own LB → fine
    _login(client, a)
    client.post(f'/dataset/{ds.id}/visibility', data={'visibility': 'private'})
    assert db.session.get(Dataset, ds.id).visibility == 'private'


def test_admin_bypasses_dataset_guard(client, db_session):
    a = _user('adm@x.io', 'adm', admin=True)
    b = _user('b3@x.io', 'b3')
    ds = _ds(a, name='adm-ds')
    _lb(b, name='adm-lb', datasets=[ds])
    _login(client, a)
    client.post(f'/dataset/{ds.id}/visibility', data={'visibility': 'private'})
    assert db.session.get(Dataset, ds.id).visibility == 'private'  # admin allowed


# --- leaderboard depended on by another user's submission -----------------

def test_lb_downgrade_blocked_when_foreign_submission(client, db_session):
    a, b = _user('la@x.io', 'la'), _user('lb@x.io', 'lb')
    lb = _lb(a, name='sub-lb')
    db.session.add(Submission(name='s', leaderboard_id=lb.id, owner_user_id=b.id))
    db.session.commit()
    _login(client, a)
    r = client.post(f'/leaderboard/{lb.id}/visibility', data={'visibility': 'private'})
    assert r.status_code == 302
    assert db.session.get(Leaderboard, lb.id).visibility == 'public'


def test_lb_delete_blocked_when_foreign_submission(client, db_session):
    a, b = _user('la2@x.io', 'la2'), _user('lb2@x.io', 'lb2')
    lb = _lb(a, name='sub-lb2')
    db.session.add(Submission(name='s', leaderboard_id=lb.id, owner_user_id=b.id))
    db.session.commit()
    _login(client, a)
    client.post(f'/delete_leaderboard/{lb.id}')
    assert db.session.get(Leaderboard, lb.id) is not None


def test_lb_downgrade_allowed_when_only_own_submission(client, db_session):
    a = _user('la3@x.io', 'la3')
    lb = _lb(a, name='own-sub-lb')
    db.session.add(Submission(name='s', leaderboard_id=lb.id, owner_user_id=a.id))
    db.session.commit()
    _login(client, a)
    client.post(f'/leaderboard/{lb.id}/visibility', data={'visibility': 'private'})
    assert db.session.get(Leaderboard, lb.id).visibility == 'private'
