"""Activation breakdown on the admin acquisition dashboard.

'Activated' = a signup that did something real after creating the account
(owns content, minted a token, returned, or browsed >2 min). Drive-by ad
signups that bounce must count as NOT activated.
"""
import json
from datetime import datetime, timedelta

import pytest

from app import (
    Dataset,
    User,
    _activation_cache,
    _activation_stats,
    db,
    generate_api_token,
)


@pytest.fixture(autouse=True)
def _reset_activation_cache():
    """The stats are cached 5 min (keyed only on `days`); tests run faster than
    that, so clear it before each test or new signups won't show."""
    _activation_cache.update(ts=0.0, days=None, data=None)
    yield
    _activation_cache.update(ts=0.0, days=None, data=None)


@pytest.fixture
def admin_client(client, db_session):
    u = User(email='admin@example.com', display_name='Admin',
             oauth_provider='github', oauth_sub='adm-1', is_admin=True)
    db.session.add(u)
    db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = u.id
    return client


def _mk(email, sub, *, attribution=None, token=False, browsed=0, returned=0):
    """Create a user with a controllable activation profile."""
    now = datetime.utcnow()
    u = User(email=email, display_name=email.split('@')[0],
             oauth_provider='google', oauth_sub=sub,
             created_at=now,
             last_login_at=now + timedelta(seconds=returned),
             last_seen_at=now + timedelta(seconds=max(browsed, returned)),
             api_token=generate_api_token() if token else None,
             signup_attribution=json.dumps(attribution) if attribution else None)
    db.session.add(u)
    db.session.commit()
    return u


def test_drive_by_ad_signup_is_not_activated(db_session):
    # Signed up via an ad, bounced immediately: no content, no token, 0s session.
    _mk('driveby@x.io', 'g-driveby',
        attribution={'source': 'google_ads', 'gad_campaignid': '99999'})
    stats = _activation_stats(days=30)
    assert stats['signups'] >= 1
    row = next(u for u in stats['recent'] if u['email'].startswith('driveby@'))
    assert row['activated'] is False
    assert 'never engaged' in row['reason']
    # the campaign shows up with a 0% rate
    camp = next(c for c in stats['by_campaign'] if c['name'] == '99999')
    assert camp['signups'] == 1 and camp['activated'] == 0 and camp['rate'] == 0


def test_content_owner_is_activated(db_session):
    u = _mk('builder@x.io', 'g-builder',
            attribution={'source': 'google_ads', 'gad_campaignid': '111'})
    db.session.add(Dataset(name='builder_ds', visibility='public', owner_user_id=u.id))
    db.session.commit()
    stats = _activation_stats(days=30)
    row = next(r for r in stats['recent'] if r['email'].startswith('builder@'))
    assert row['activated'] is True
    assert row['reason'] == 'created content'


def test_token_and_browse_signals_activate(db_session):
    _mk('tok@x.io', 'g-tok', token=True)
    _mk('browser@x.io', 'g-br', browsed=300)   # >120s session
    stats = _activation_stats(days=30)
    reasons = {r['email'].split('@')[0]: r for r in stats['recent']}
    assert reasons['tok']['activated'] and reasons['tok']['reason'] == 'API token'
    assert reasons['browser']['activated'] and reasons['browser']['reason'] == 'browsed'


def test_rate_and_source_rollup(db_session):
    # 1 activated + 1 not, both google_ads → 50% for that source.
    _mk('a1@x.io', 'g-a1', token=True, attribution={'source': 'google_ads'})
    _mk('a2@x.io', 'g-a2', attribution={'source': 'google_ads'})
    stats = _activation_stats(days=30)
    src = next(s for s in stats['by_source'] if s['name'] == 'google_ads')
    assert src['signups'] == 2 and src['activated'] == 1 and src['rate'] == 50


def test_acquisition_page_renders_activation_section(admin_client, db_session):
    _mk('seen@x.io', 'g-seen', browsed=300)
    resp = admin_client.get('/admin/acquisition?days=7')
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'Activation' in body
    assert 'Activation rate' in body
    assert 'By ad campaign' in body or 'By traffic source' in body


def test_acquisition_page_forbidden_for_non_admin(client, logged_in_user):
    with client.session_transaction() as sess:
        sess['user_id'] = logged_in_user.id
    assert client.get('/admin/acquisition').status_code == 403
