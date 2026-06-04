"""Per-user HuggingFace token settings — the page that lets a user
authorise gated/private HF dataset imports."""
from app import User, db


def test_requires_login(client, db_session):
    r = client.get('/settings/hf_token', follow_redirects=False)
    assert r.status_code == 302 and '/login' in r.headers['Location']


def test_get_renders_no_token_state(auth_client):
    body = auth_client.get('/settings/hf_token').data.decode()
    assert 'HuggingFace access token' in body
    assert 'No token saved' in body


def test_post_saves_token_then_shows_masked(auth_client, logged_in_user):
    r = auth_client.post('/settings/hf_token',
                         data={'hf_token': 'hf_abcdef=ghijklmnop12345'},
                         follow_redirects=True)
    assert r.status_code == 200
    # Persisted on the user row…
    assert db.session.get(User, logged_in_user.id).hf_token == 'hf_abcdef=ghijklmnop12345'
    # …and shown masked, never in full.
    body = r.data.decode()
    assert 'hf_abc' in body and '2345' in body
    assert 'hf_abcdef=ghijklmnop12345' not in body


def test_post_clear_removes_token(auth_client, logged_in_user):
    u = db.session.get(User, logged_in_user.id)
    u.hf_token = 'hf_tokeniztokeniz'
    db.session.commit()
    auth_client.post('/settings/hf_token', data={'action': 'clear'},
                     follow_redirects=True)
    assert db.session.get(User, logged_in_user.id).hf_token is None
