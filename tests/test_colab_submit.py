"""Per-LB Colab submission notebook generation + caching."""
import json
from unittest.mock import patch

import pytest

from app import (
    Dataset, Leaderboard, Sample, db,
    _lb_structure_signature,
    _static_colab_notebook,
    _get_or_generate_colab_notebook,
)


@pytest.fixture
def lb_with_one_dataset(db_session):
    ds = Dataset(name='colab_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    db.session.flush()
    lb = Leaderboard(name='colab_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb


# ---------------------------------------------------------------------------
# Signature stability and drift
# ---------------------------------------------------------------------------


def test_signature_is_stable_for_same_lb(lb_with_one_dataset):
    sig_a = _lb_structure_signature(lb_with_one_dataset)
    sig_b = _lb_structure_signature(lb_with_one_dataset)
    assert sig_a == sig_b


def test_signature_changes_when_dataset_added(lb_with_one_dataset, db_session):
    lb = lb_with_one_dataset
    sig_before = _lb_structure_signature(lb)
    other = Dataset(name='colab_ds_2', visibility='public')
    db.session.add(other); db.session.flush()
    lb.datasets.append(other); db.session.commit()
    sig_after = _lb_structure_signature(lb)
    assert sig_before != sig_after


# ---------------------------------------------------------------------------
# Static fallback (no API key)
# ---------------------------------------------------------------------------


def test_static_notebook_is_valid_ipynb(lb_with_one_dataset):
    raw = _static_colab_notebook(lb_with_one_dataset)
    nb = json.loads(raw)
    assert nb['nbformat'] == 4
    assert isinstance(nb['cells'], list) and len(nb['cells']) >= 4
    # Mentions the LB name + dataset name + submission upload route.
    assert 'colab_lb' in raw
    assert 'colab_ds' in raw
    assert '/api/leaderboard/' in raw
    assert 'my_model' in raw


# ---------------------------------------------------------------------------
# get_or_generate: cache hit + miss + LLM path + signature drift
# ---------------------------------------------------------------------------


def test_first_call_without_api_key_uses_static(monkeypatch, lb_with_one_dataset, db_session):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    nb1, src1 = _get_or_generate_colab_notebook(lb_with_one_dataset)
    assert src1 == 'static'
    # And the result was cached.
    db.session.refresh(lb_with_one_dataset)
    assert lb_with_one_dataset.colab_notebook_cache is not None


def test_second_call_hits_cache(monkeypatch, lb_with_one_dataset, db_session):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    _get_or_generate_colab_notebook(lb_with_one_dataset)
    nb2, src2 = _get_or_generate_colab_notebook(lb_with_one_dataset)
    assert src2 == 'cache'


def test_signature_drift_invalidates_cache(monkeypatch, lb_with_one_dataset, db_session):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    _get_or_generate_colab_notebook(lb_with_one_dataset)
    # Change the LB structure → next call should regen, not cache-hit.
    other = Dataset(name='extra_ds', visibility='public')
    db.session.add(other); db.session.flush()
    lb_with_one_dataset.datasets.append(other); db.session.commit()
    _, src = _get_or_generate_colab_notebook(lb_with_one_dataset)
    assert src == 'static'  # regenerated, not cached


def test_llm_path_when_api_key_set(monkeypatch, lb_with_one_dataset, db_session):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    fake_nb_text = json.dumps({
        'cells': [{'cell_type': 'markdown', 'source': ['hi']}],
        'nbformat': 4, 'nbformat_minor': 5, 'metadata': {},
    })

    class _Ok:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': fake_nb_text}]}

    with patch('requests.post', return_value=_Ok()):
        nb, src = _get_or_generate_colab_notebook(lb_with_one_dataset)
    assert src == 'llm'
    parsed = json.loads(nb)
    assert parsed['cells'][0]['source'] == ['hi']


def test_llm_failure_falls_back_to_static(monkeypatch, lb_with_one_dataset, db_session):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    with patch('requests.post', side_effect=RuntimeError('rate limited')):
        nb, src = _get_or_generate_colab_notebook(lb_with_one_dataset)
    assert src == 'static'


# ---------------------------------------------------------------------------
# Route serves the notebook
# ---------------------------------------------------------------------------


def test_colab_notebook_route_returns_ipynb_json(client, db_session, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    ds = Dataset(name='route_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='route_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    resp = client.get(f'/leaderboard/{lb.id}/colab_notebook.ipynb')
    assert resp.status_code == 200
    assert resp.headers['Content-Type'] == 'application/x-ipynb+json'
    # Filename hint preserved (sanitized).
    assert 'route_lb' in resp.headers.get('Content-Disposition', '')
    # Body parses as a notebook.
    nb = json.loads(resp.data)
    assert nb['nbformat'] == 4
    assert any('my_model' in ''.join(c.get('source', [])) for c in nb['cells'])


# ---------------------------------------------------------------------------
# /colab_open route — gist creation when token configured, fallback otherwise
# ---------------------------------------------------------------------------


def test_colab_open_redirects_to_gist_when_token_set(
    client, db_session, monkeypatch, lb_with_one_dataset,
):
    monkeypatch.setenv('BENCHHUB_GITHUB_GIST_TOKEN', 'ghp_test')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    class _GistOk:
        status_code = 201
        def raise_for_status(self): pass
        def json(self):
            return {
                'id': 'gistabc123',
                'html_url': 'https://gist.github.com/testuser/gistabc123',
                'owner': {'login': 'testuser'},
            }

    with patch('requests.post', return_value=_GistOk()):
        resp = client.get(
            f'/leaderboard/{lb_with_one_dataset.id}/colab_open',
            follow_redirects=False,
        )
    assert resp.status_code == 302
    # Colab requires `<owner>/<gist_id>` — bare id is rejected.
    assert 'colab.research.google.com/gist/testuser/gistabc123' in resp.headers['Location']

    # gist_id + owner persisted to cache so the next request can PATCH and reuse the URL.
    db.session.refresh(lb_with_one_dataset)
    wrapped = json.loads(lb_with_one_dataset.colab_notebook_cache)
    assert wrapped.get('gist_id') == 'gistabc123'
    assert wrapped.get('gist_owner') == 'testuser'


def test_colab_open_falls_back_when_no_token(
    client, db_session, monkeypatch, lb_with_one_dataset,
):
    monkeypatch.delenv('BENCHHUB_GITHUB_GIST_TOKEN', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    resp = client.get(
        f'/leaderboard/{lb_with_one_dataset.id}/colab_open',
        follow_redirects=False,
    )
    # No token → bounce to the LB page with a warning flash.
    assert resp.status_code == 302
    assert f'/leaderboard/{lb_with_one_dataset.id}' in resp.headers['Location']
    assert 'colab.research.google.com' not in resp.headers['Location']


def test_colab_open_patches_existing_gist_on_second_call(
    client, db_session, monkeypatch, lb_with_one_dataset,
):
    """Second visit reuses the cached gist_id and PATCHes instead of
    creating a fresh gist — no orphans on subsequent regenerations."""
    monkeypatch.setenv('BENCHHUB_GITHUB_GIST_TOKEN', 'ghp_test')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    # Seed cache with a gist_id+owner and a stale signature so the notebook regens.
    lb_with_one_dataset.colab_notebook_cache = json.dumps({
        'sig': 'stale', 'notebook': '{}',
        'gist_id': 'oldgist', 'gist_owner': 'testuser',
    })
    db.session.commit()

    class _PatchOk:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                'id': 'oldgist',
                'html_url': 'https://gist.github.com/testuser/oldgist',
                'owner': {'login': 'testuser'},
            }

    with patch('requests.patch', return_value=_PatchOk()) as patch_mock, \
         patch('requests.post') as post_mock:
        resp = client.get(f'/leaderboard/{lb_with_one_dataset.id}/colab_open')

    assert resp.status_code == 302
    patch_mock.assert_called_once()
    post_mock.assert_not_called()
    assert 'colab.research.google.com/gist/testuser/oldgist' in resp.headers['Location']


def test_colab_notebook_route_404s_for_private_to_anon(client, db_session, monkeypatch):
    """Notebook respects LB visibility — private LB → 404 to anon."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    from app import User
    owner = User(email='priv-colab@example.com', display_name='Owner',
                 oauth_provider='github', oauth_sub='pc-1')
    db.session.add(owner); db.session.flush()
    lb = Leaderboard(name='priv_lb', summary_metrics='',
                     owner_user_id=owner.id, visibility='private')
    db.session.add(lb); db.session.commit()
    resp = client.get(f'/leaderboard/{lb.id}/colab_notebook.ipynb')
    assert resp.status_code == 404
