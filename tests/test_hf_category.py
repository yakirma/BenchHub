"""Coverage for HF-derived Dataset.category:

1. `_hf_tags_to_category(tags)` — maps HF Hub `task_categories:*` /
   `task_ids:*` tags onto BH's Area/Task taxonomy.
2. `/dataset/<id>/settings` POST — owner can edit category later.
"""
import json

import pytest

from app import Dataset, User, _hf_tags_to_category, db


# ---------------------------------------------------------------------------
# _hf_tags_to_category
# ---------------------------------------------------------------------------


def test_maps_image_classification_to_vision():
    cat = _hf_tags_to_category(['task_categories:image-classification'])
    assert cat == 'Vision/Image Classification'


def test_task_ids_beats_task_categories_when_both_present():
    """A more-specific task_ids tag wins over the broad task_categories
    one so e.g. semantic-segmentation isn't downgraded to "Image
    Segmentation" when the uploader spelled it out."""
    cat = _hf_tags_to_category([
        'task_categories:image-segmentation',
        'task_ids:semantic-segmentation',
    ])
    assert cat == 'Vision/Semantic Segmentation'


def test_translation_tag_lands_in_nlp():
    cat = _hf_tags_to_category(['task_categories:translation', 'language:en'])
    assert cat == 'NLP/Translation'


def test_audio_classification_lands_in_speech_audio():
    cat = _hf_tags_to_category(['task_categories:audio-classification'])
    assert cat == 'Speech & Audio/Audio Classification'


def test_medical_domain_overrides_otherwise_vision_task():
    """A tag list mentioning `medical` falls through to the Medical
    bucket even when no recognised task tag is present — matches the
    old _DOMAIN_PREFIXES heuristic."""
    cat = _hf_tags_to_category(['medical', 'license:cc-by-4.0'])
    assert cat == 'Medical/Other'


def test_returns_none_when_no_recognised_tag():
    """Unknown / non-task tags → None so the dataset row lands in
    Uncategorized rather than the wrong bucket."""
    assert _hf_tags_to_category(['language:en', 'size_categories:10K<n<100K']) is None
    assert _hf_tags_to_category([]) is None
    assert _hf_tags_to_category(None) is None


def test_non_string_tags_dont_crash():
    assert _hf_tags_to_category([42, None, 'task_categories:translation']) == 'NLP/Translation'


# ---------------------------------------------------------------------------
# /dataset/<id>/settings POST — owner edits category later
# ---------------------------------------------------------------------------


@pytest.fixture
def owner(db_session):
    u = User(email='dsowner@bench.local', display_name='ds',
             oauth_provider='github', oauth_sub='dsown-1')
    db.session.add(u); db.session.commit()
    return u


def test_settings_post_saves_category(client, db_session, owner):
    """POSTing a category string on /dataset/<id>/settings writes it
    to Dataset.category and redirects back to the same page."""
    ds = Dataset(name='cat_ds', owner_user_id=owner.id)
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(f'/dataset/{ds.id}/settings',
                    data={'category': 'Vision/Custom Task'})
    assert r.status_code == 302
    db.session.refresh(ds)
    assert ds.category == 'Vision/Custom Task'


def test_settings_post_blank_category_clears(client, db_session, owner):
    """Empty string clears the category back to NULL — the dataset
    falls into the Uncategorized bucket on /datasets."""
    ds = Dataset(name='blank_ds', owner_user_id=owner.id,
                 category='Vision/Old Task')
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    client.post(f'/dataset/{ds.id}/settings', data={'category': '   '})
    db.session.refresh(ds)
    assert ds.category is None


def test_settings_post_normalises_extra_slashes(client, db_session, owner):
    """`Vision //  Depth Estimation` should normalise to
    `Vision/Depth Estimation` so equality filters elsewhere find it."""
    ds = Dataset(name='slash_ds', owner_user_id=owner.id)
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    client.post(f'/dataset/{ds.id}/settings',
                data={'category': 'Vision //  Depth Estimation'})
    db.session.refresh(ds)
    assert ds.category == 'Vision/Depth Estimation'


def test_settings_post_forbidden_for_non_owner(client, db_session, owner):
    """Random users get 403 — owner_required is the gate."""
    stranger = User(email='nope@bench.local', display_name='n',
                    oauth_provider='github', oauth_sub='nope-1')
    db.session.add(stranger); db.session.flush()
    ds = Dataset(name='locked', owner_user_id=owner.id)
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = stranger.id
    r = client.post(f'/dataset/{ds.id}/settings',
                    data={'category': 'Vision/Whatever'})
    assert r.status_code == 403


def test_settings_get_renders_category_input(client, db_session, owner):
    """The Category card must appear on the page with the dataset's
    current value pre-filled."""
    ds = Dataset(name='render_ds', owner_user_id=owner.id,
                 category='Vision/Image Segmentation')
    db.session.add(ds); db.session.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    body = client.get(f'/dataset/{ds.id}/settings').data.decode('utf-8')
    assert 'name="category"' in body
    assert 'Vision/Image Segmentation' in body
