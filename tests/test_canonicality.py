"""Phase 3 of the LB-attachment refactor: canonicality.

A leaderboard is either 'personal' (default; visible only to owner +
collaborators on /home) or 'public' (admin-promoted; appears on
/explore). At most one public LB per HF repo via canonical_for_repo,
enforced at promote time.
"""
import pytest

from app import Dataset, Leaderboard, Sample, User, db


@pytest.fixture
def login_as(client):
    """Tiny test helper: stuff a user_id into the Flask session."""
    def _go(user):
        with client.session_transaction() as sess:
            sess['user_id'] = user.id
    return _go


def _mk_admin(email='admin@bench.local'):
    u = User(email=email, display_name='admin', is_admin=True,
             oauth_provider='github', oauth_sub=email)
    db.session.add(u); db.session.commit()
    return u


def _mk_user(email='someone@bench.local'):
    u = User(email=email, display_name='someone', is_admin=False,
             oauth_provider='github', oauth_sub=email)
    db.session.add(u); db.session.commit()
    return u


def _seed_lb(name, *, canonicality='personal', canonical_for_repo=None,
             owner=None, visibility='public'):
    ds = Dataset(name=f'ds_{name}', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    lb = Leaderboard(
        name=name, summary_metrics='', visibility=visibility,
        canonicality=canonicality, canonical_for_repo=canonical_for_repo,
        owner_user_id=owner.id if owner else None,
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb


# ---------------------------------------------------------------------------
# /explore filters to canonicality=public
# ---------------------------------------------------------------------------


def test_explore_hides_personal_leaderboards(client, db_session):
    _seed_lb('lb_personal', canonicality='personal')
    _seed_lb('lb_public', canonicality='public')
    body = client.get('/explore').data
    assert b'lb_public' in body
    assert b'lb_personal' not in body


# ---------------------------------------------------------------------------
# /home shows owner's personal + public LBs in separate sections
# ---------------------------------------------------------------------------


def test_home_splits_public_and_personal_for_owner(client, db_session, login_as):
    owner = _mk_user('owner@bench.local')
    _seed_lb('lb_owned_personal', canonicality='personal', owner=owner)
    _seed_lb('lb_owned_public', canonicality='public', owner=owner)
    login_as(owner)
    body = client.get('/home').data
    # Both visible to owner.
    assert b'lb_owned_personal' in body
    assert b'lb_owned_public' in body
    # The "Your public leaderboards" section header appears only when
    # the owner has at least one public LB.
    assert b'Your public leaderboards' in body


# ---------------------------------------------------------------------------
# Admin promote endpoint
# ---------------------------------------------------------------------------


def test_promote_endpoint_requires_admin(client, db_session, login_as):
    rando = _mk_user('rando@bench.local')
    lb = _seed_lb('lb_to_promote', canonicality='personal')
    login_as(rando)
    r = client.post(
        f'/admin/leaderboard/{lb.id}/promote',
        data={'canonicality': 'public', 'canonical_for_repo': 'cifar10'},
    )
    assert r.status_code == 403
    db.session.expire_all()
    assert lb.canonicality == 'personal'


def test_promote_sets_canonical_for_repo(client, db_session, login_as):
    admin = _mk_admin()
    lb = _seed_lb('lb_to_promote', canonicality='personal')
    login_as(admin)
    r = client.post(
        f'/admin/leaderboard/{lb.id}/promote',
        data={'canonicality': 'public', 'canonical_for_repo': 'cifar10'},
        follow_redirects=False,
    )
    # 302 back to LB view with flash.
    assert r.status_code in (302, 303)
    db.session.expire_all()
    assert lb.canonicality == 'public'
    assert lb.canonical_for_repo == 'cifar10'


def test_promote_demote_clears_canonical_for_repo(client, db_session, login_as):
    admin = _mk_admin()
    lb = _seed_lb(
        'lb_already_canonical', canonicality='public',
        canonical_for_repo='nyu_depth_v2',
    )
    login_as(admin)
    client.post(
        f'/admin/leaderboard/{lb.id}/promote',
        data={'canonicality': 'personal'},
    )
    db.session.expire_all()
    assert lb.canonicality == 'personal'
    assert lb.canonical_for_repo is None


def test_promote_rejects_repo_that_already_has_canonical(client, db_session, login_as):
    """Two LBs cannot both claim canonicality for the same HF repo —
    /explore would show duplicates and "submit there instead" loses
    its meaning."""
    admin = _mk_admin()
    _seed_lb(
        'lb_first', canonicality='public', canonical_for_repo='cifar10',
    )
    contender = _seed_lb('lb_contender', canonicality='personal')
    login_as(admin)
    r = client.post(
        f'/admin/leaderboard/{contender.id}/promote',
        data={'canonicality': 'public', 'canonical_for_repo': 'cifar10'},
        follow_redirects=True,
    )
    db.session.expire_all()
    assert contender.canonicality == 'personal'
    # Flash surfaced on the redirect target.
    assert b'already canonicalized' in r.data


# ---------------------------------------------------------------------------
# Auto-LB preview surfaces the "submit there" callout
# ---------------------------------------------------------------------------


def test_auto_lb_preview_warns_when_canonical_exists_for_repo(
    client, db_session, monkeypatch, login_as,
):
    user = _mk_user('newcomer@bench.local')
    _seed_lb(
        'lb_canon_for_cifar', canonicality='public',
        canonical_for_repo='cifar10',
    )

    # Stub the proposers so we don't reach out to HF.
    import app as app_mod

    def fake_proposals(repo_id, mapping, **_):
        return (
            [{
                'global_name': 'mae_score',
                'target_name': 'mae',
                'description': 'Mean abs error',
                'sort_direction': 'lower_is_better',
                'arg_mappings': {},
                'python_code': '',
                'code_source': 'static',
                'pred_fields': [],
                'is_aggregated': False,
            }],
            [],
        )
    monkeypatch.setattr(app_mod, '_collect_auto_lb_proposals_for_hf_ref',
                        fake_proposals)

    login_as(user)
    r = client.post('/import_from_hf/auto', data={
        'hf_repo_id': 'cifar10',
        'hf_revision': '',
        'hf_token': '',
        'dataset_name': 'cifar10_local',
        'sample_cap': 200,
        'mapping_column[]': ['image', 'label'],
        'mapping_target_kind[]': ['image', 'scalar'],
        'mapping_target_field[]': ['image_image', 'label'],
    })
    assert r.status_code == 200
    body = r.data
    assert b'A canonical leaderboard already exists' in body
    assert b'lb_canon_for_cifar' in body
