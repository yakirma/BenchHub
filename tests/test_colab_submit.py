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
