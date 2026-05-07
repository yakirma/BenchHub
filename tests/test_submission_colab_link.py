"""Submission ↔ Colab provenance: the gist URL that produced a
submission is recorded on the row and surfaced on the LB page."""
import io
import zipfile
from unittest.mock import patch

import pytest

from app import (
    Dataset, Leaderboard, Submission, UserColabGist, db,
)


def _build_minimal_zip():
    """A submission ZIP that has at least one metric_* file so the
    upload path doesn't reject it. Two top-level entries so
    process_submission_zip doesn't auto-rename the submission to a
    nested folder name."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('metric_dummy/s00000.txt', '0.5')
        zf.writestr('README.md', 'submission')
    buf.seek(0)
    return buf


@pytest.fixture
def lb_and_token(auth_client, logged_in_user, db_session):
    """Public LB + the logged-in user holding a fresh API token."""
    from app import generate_api_token
    logged_in_user.api_token = generate_api_token()
    db.session.commit()
    ds = Dataset(name='colab_link_ds', visibility='public')
    db.session.add(ds); db.session.commit()
    lb = Leaderboard(name='colab_link_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb, logged_in_user


def test_api_upload_records_form_supplied_colab_url(
    client, lb_and_token, db_session,
):
    """The notebook bakes its own gist URL into the upload call;
    we persist it on the Submission row."""
    lb, user = lb_and_token
    resp = client.post(
        f'/api/leaderboard/{lb.id}/submission/upload',
        data={
            'submission_name': 'from_colab',
            'source_colab_url': 'https://colab.research.google.com/gist/me/abc123',
            'submission_zip': (_build_minimal_zip(), 'submission.zip'),
        },
        headers={'Authorization': f'Bearer {user.api_token}'},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 200, resp.data
    sub = Submission.query.filter_by(name='from_colab').first()
    assert sub is not None
    assert sub.source_colab_url == 'https://colab.research.google.com/gist/me/abc123'


def test_api_upload_falls_back_to_user_colab_gist_when_url_missing(
    client, lb_and_token, db_session,
):
    """Older notebooks (or scripted uploads) won't supply
    source_colab_url. If the user has a UserColabGist for this LB,
    we infer the URL from there so the back-link still works."""
    lb, user = lb_and_token
    db.session.add(UserColabGist(
        user_id=user.id, leaderboard_id=lb.id,
        gist_id='legacygist1', gist_owner='someone', sig='x',
    ))
    db.session.commit()

    resp = client.post(
        f'/api/leaderboard/{lb.id}/submission/upload',
        data={
            'submission_name': 'fallback_colab',
            'submission_zip': (_build_minimal_zip(), 'submission.zip'),
        },
        headers={'Authorization': f'Bearer {user.api_token}'},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 200, resp.data
    sub = Submission.query.filter_by(name='fallback_colab').first()
    assert sub is not None
    assert sub.source_colab_url == (
        'https://colab.research.google.com/gist/someone/legacygist1'
    )


def test_api_upload_leaves_url_null_when_neither_form_nor_gist_present(
    client, lb_and_token, db_session,
):
    """No form URL + no UserColabGist for this LB → don't invent one."""
    lb, user = lb_and_token
    resp = client.post(
        f'/api/leaderboard/{lb.id}/submission/upload',
        data={
            'submission_name': 'no_provenance',
            'submission_zip': (_build_minimal_zip(), 'submission.zip'),
        },
        headers={'Authorization': f'Bearer {user.api_token}'},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 200, resp.data
    sub = Submission.query.filter_by(name='no_provenance').first()
    assert sub is not None
    assert sub.source_colab_url is None


def test_lb_page_renders_colab_link_when_submission_has_url(
    client, lb_and_token, db_session,
):
    lb, user = lb_and_token
    sub = Submission(
        name='with_link', leaderboard_id=lb.id,
        owner_user_id=user.id,
        source_colab_url='https://colab.research.google.com/gist/u/xyz',
    )
    db.session.add(sub); db.session.commit()
    resp = client.get(f'/leaderboard/{lb.id}')
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'colab.research.google.com/gist/u/xyz' in body
    # Rocket icon used as the click target.
    assert 'bi-rocket-takeoff' in body


def test_lb_page_does_not_render_colab_link_when_url_missing(
    client, lb_and_token, db_session,
):
    lb, user = lb_and_token
    sub = Submission(
        name='no_link', leaderboard_id=lb.id,
        owner_user_id=user.id, source_colab_url=None,
    )
    db.session.add(sub); db.session.commit()
    resp = client.get(f'/leaderboard/{lb.id}')
    assert resp.status_code == 200
    body = resp.data.decode()
    # The "Open the Colab notebook that produced this submission"
    # tooltip is the unique fingerprint for the per-row icon (the
    # 'Submit via Colab' header button doesn't share this title).
    assert 'Colab notebook that produced this submission' not in body


# ---------------------------------------------------------------------------
# Personalization: per-user gist substitutes SOURCE_COLAB_URL placeholder
# when a gist record already exists.
# ---------------------------------------------------------------------------


def test_personalize_substitutes_source_colab_url_when_known(monkeypatch):
    """The personalize helper rewrites the SOURCE_COLAB_URL
    placeholder so the cell carries the gist URL verbatim."""
    from app import _personalize_notebook_for_user
    from types import SimpleNamespace
    nb = (
        '{"cells":[{"cell_type":"code","source":['
        '"API_TOKEN = \'\'\\n",'
        '"SOURCE_COLAB_URL = \'\'\\n"'
        ']}],"nbformat":4,"nbformat_minor":5,"metadata":{}}'
    )
    user = SimpleNamespace(api_token='tok')
    out = _personalize_notebook_for_user(
        nb, user, source_colab_url='https://colab.research.google.com/gist/u/g1',
    )
    assert "API_TOKEN = 'tok'" in out
    assert "SOURCE_COLAB_URL = 'https://colab.research.google.com/gist/u/g1'" in out


def test_personalize_leaves_source_url_blank_when_unknown():
    """First-time gist creation has no URL yet — leave the placeholder
    empty so the API endpoint's UserColabGist fallback fills it in."""
    from app import _personalize_notebook_for_user
    from types import SimpleNamespace
    nb = (
        '{"cells":[{"cell_type":"code","source":['
        '"API_TOKEN = \'\'\\n",'
        '"SOURCE_COLAB_URL = \'\'\\n"'
        ']}],"nbformat":4,"nbformat_minor":5,"metadata":{}}'
    )
    user = SimpleNamespace(api_token='tok')
    out = _personalize_notebook_for_user(nb, user)  # no URL
    assert "API_TOKEN = 'tok'" in out
    assert "SOURCE_COLAB_URL = ''" in out
