"""Phase 6: unified LB-creation chooser at /create_lb.

Replaces /create_lb/from_hf as the single entry point. Both the
BenchHub-dataset side and the HuggingFace-dataset side render in one
page; the user picks their flow there.
"""
from app import Dataset, Sample, db


def test_chooser_requires_login(client, db_session):
    r = client.get('/create_lb', follow_redirects=False)
    assert r.status_code == 302
    assert '/login' in r.headers['Location']


def test_old_from_hf_url_redirects_into_chooser(auth_client, db_session):
    r = auth_client.get('/create_lb/from_hf', follow_redirects=False)
    assert r.status_code == 302
    assert '/create_lb' in r.headers['Location']


def test_chooser_lists_visible_bh_datasets(auth_client, logged_in_user, db_session):
    """The BH side dropdown surfaces only datasets the user can see."""
    own = Dataset(name='own_ds', visibility='private',
                  owner_user_id=logged_in_user.id)
    pub = Dataset(name='public_ds', visibility='public')
    other_priv = Dataset(name='hidden_priv', visibility='private',
                         owner_user_id=999)
    db.session.add_all([own, pub, other_priv]); db.session.flush()
    for d in (own, pub, other_priv):
        db.session.add(Sample(dataset_id=d.id, name='s1'))
    db.session.commit()

    body = auth_client.get('/create_lb').data
    assert b'own_ds' in body
    assert b'public_ds' in body
    assert b'hidden_priv' not in body


def test_chooser_renders_hf_picker_form(auth_client, db_session):
    body = auth_client.get('/create_lb').data
    # BH side
    assert b'From a BenchHub dataset' in body
    # HF side + picker form (action goes to /import_from_hf/preview)
    assert b'From a HuggingFace dataset' in body
    assert b'/import_from_hf/preview' in body


def test_chooser_empty_state_when_no_bh_datasets(auth_client, db_session):
    """When the user has no visible BH datasets, the BH side surfaces
    an "upload a ZIP" CTA instead of an empty dropdown."""
    body = auth_client.get('/create_lb').data
    assert b'No BenchHub datasets yet' in body
    assert b'Upload a ZIP' in body
