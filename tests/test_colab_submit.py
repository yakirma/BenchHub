"""Per-LB Colab submission notebook generation + caching."""
import json
from unittest.mock import patch

import pytest

from app import (
    Dataset, Leaderboard, Sample, UserColabGist, db,
    _lb_structure_signature,
    _static_colab_notebook,
    _get_or_generate_colab_notebook,
    _personalize_notebook_for_user,
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


def test_static_notebook_lets_user_pick_runtime_and_offers_local_bootstrap(
    lb_with_one_dataset,
):
    """The notebook must NOT force a GPU runtime (some users only need
    CPU; some have their own accelerator preference) — but it must
    surface the choice. It must also include a no-op-on-Colab
    `>>> RUN-LOCALLY BOOTSTRAP <<<` cell so users who'd rather work
    locally can fetch the notebook + dataset to their own machine."""
    raw = _static_colab_notebook(lb_with_one_dataset)
    nb = json.loads(raw)

    # No accelerator pin in metadata — user decides.
    md = nb.get('metadata', {})
    assert 'accelerator' not in md
    assert 'gpuClass' not in (md.get('colab') or {})

    # GPU choice surfaced in the leading markdown.
    first_md = ''.join(nb['cells'][0]['source'])
    assert 'Change runtime type' in first_md
    assert 'GPU' in first_md

    # Bootstrap script lives near the top, runs only outside Colab,
    # and downloads both the notebook and the dataset ZIP.
    bootstrap_src = next(
        (''.join(c['source']) for c in nb['cells']
         if c.get('cell_type') == 'code'
         and 'RUN-LOCALLY BOOTSTRAP' in ''.join(c['source'])),
        None,
    )
    assert bootstrap_src is not None, "bootstrap cell missing"
    assert "'google.colab' not in sys.modules" in bootstrap_src
    assert 'urllib.request' in bootstrap_src
    assert '/colab_notebook.ipynb' in bootstrap_src
    assert '/dataset/' in bootstrap_src
    # Bootstrap precedes the model-definition cell.
    bootstrap_idx = next(
        i for i, c in enumerate(nb['cells'])
        if c.get('cell_type') == 'code'
        and 'RUN-LOCALLY BOOTSTRAP' in ''.join(c['source'])
    )
    model_idx = next(
        i for i, c in enumerate(nb['cells'])
        if 'def my_model' in ''.join(c.get('source', []))
    )
    assert bootstrap_idx < model_idx


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


# ---------------------------------------------------------------------------
# Per-user token autofill: generic notebook → personalized per logged-in user
# ---------------------------------------------------------------------------


def test_personalize_substitutes_empty_token_placeholder():
    """Generic `API_TOKEN = ''` is replaced with the user's actual token."""
    from types import SimpleNamespace
    nb = json.dumps({
        'cells': [{
            'cell_type': 'code', 'source': ["API_TOKEN = ''  # paste here\n"],
        }],
        'nbformat': 4, 'nbformat_minor': 5, 'metadata': {},
    })
    user = SimpleNamespace(api_token='bh_secret_42')
    out = _personalize_notebook_for_user(nb, user)
    assert "API_TOKEN = 'bh_secret_42'" in out
    assert "API_TOKEN = ''" not in out


def test_personalize_is_noop_for_anonymous_user():
    """No user / no token → leave the placeholder untouched."""
    nb = json.dumps({'cells': [{'cell_type': 'code',
                                'source': ["API_TOKEN = ''\n"]}],
                     'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}})
    assert _personalize_notebook_for_user(nb, None) == nb
    from types import SimpleNamespace
    assert _personalize_notebook_for_user(nb, SimpleNamespace(api_token=None)) == nb
    assert _personalize_notebook_for_user(nb, SimpleNamespace(api_token='')) == nb


def test_personalize_handles_double_quoted_placeholder():
    """LLM-generated cells may use double quotes; we still substitute."""
    from types import SimpleNamespace
    nb = json.dumps({'cells': [{'cell_type': 'code',
                                'source': ['API_TOKEN = ""\n']}],
                     'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}})
    out = _personalize_notebook_for_user(nb, SimpleNamespace(api_token='tok123'))
    assert "API_TOKEN = 'tok123'" in out


def test_static_notebook_includes_token_placeholder():
    """The static template uses the empty single-quoted placeholder so
    the personalize step has something to find."""
    ds = Dataset(name='ph_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='ph_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()
    raw = _static_colab_notebook(lb)
    assert "API_TOKEN = ''" in raw


def test_colab_open_uses_per_user_gist_when_logged_in_with_token(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """Authed user with an api_token → personalized notebook on a
    per-user gist (not the LB-level shared one)."""
    monkeypatch.setenv('BENCHHUB_GITHUB_GIST_TOKEN', 'ghp_test')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    logged_in_user.api_token = 'mytoken_xyz'
    db.session.commit()

    ds = Dataset(name='per_user_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    lb = Leaderboard(name='per_user_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    captured = {}

    class _Created:
        status_code = 201
        def raise_for_status(self): pass
        def json(self):
            return {
                'id': 'gistuser1',
                'html_url': 'https://gist.github.com/personalowner/gistuser1',
                'owner': {'login': 'personalowner'},
            }

    def _capture(url, **kw):
        captured['url'] = url
        captured['payload'] = kw.get('json')
        return _Created()

    with patch('requests.post', side_effect=_capture):
        resp = auth_client.get(f'/leaderboard/{lb.id}/colab_open',
                               follow_redirects=False)

    assert resp.status_code == 302
    assert 'gist/personalowner/gistuser1' in resp.headers['Location']

    # The gist content carries the user's actual token.
    files = captured['payload']['files']
    nb_text = next(iter(files.values()))['content']
    assert "API_TOKEN = 'mytoken_xyz'" in nb_text
    assert "API_TOKEN = ''" not in nb_text

    # Per-user mapping is persisted, not the LB-level cache.
    record = UserColabGist.query.filter_by(
        user_id=logged_in_user.id, leaderboard_id=lb.id,
    ).first()
    assert record is not None
    assert record.gist_id == 'gistuser1'
    assert record.gist_owner == 'personalowner'

    # And the LB-level cache is untouched (no anonymous gist created).
    db.session.refresh(lb)
    lb_cache = json.loads(lb.colab_notebook_cache or '{}')
    assert 'gist_id' not in lb_cache or lb_cache.get('gist_id') is None


def test_colab_open_authed_without_token_uses_anonymous_gist(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """Logged-in user without an api_token → fall back to the generic
    LB-level gist (token placeholder stays empty)."""
    monkeypatch.setenv('BENCHHUB_GITHUB_GIST_TOKEN', 'ghp_test')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    logged_in_user.api_token = None
    db.session.commit()

    ds = Dataset(name='no_tok_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='no_tok_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    class _Ok:
        status_code = 201
        def raise_for_status(self): pass
        def json(self):
            return {
                'id': 'gistanon1',
                'html_url': 'https://gist.github.com/sharedowner/gistanon1',
                'owner': {'login': 'sharedowner'},
            }

    captured = {}
    def _capture(url, **kw):
        captured['payload'] = kw.get('json')
        return _Ok()

    with patch('requests.post', side_effect=_capture):
        resp = auth_client.get(f'/leaderboard/{lb.id}/colab_open',
                               follow_redirects=False)

    assert resp.status_code == 302
    assert 'gist/sharedowner/gistanon1' in resp.headers['Location']
    nb_text = next(iter(captured['payload']['files'].values()))['content']
    # Placeholder stays empty for the shared/anonymous gist.
    assert "API_TOKEN = ''" in nb_text
    # No per-user record.
    assert UserColabGist.query.filter_by(
        user_id=logged_in_user.id, leaderboard_id=lb.id,
    ).first() is None


def test_colab_open_per_user_patches_existing_gist_on_second_call(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """Returning user → PATCH their existing per-user gist instead of
    creating a fresh one."""
    monkeypatch.setenv('BENCHHUB_GITHUB_GIST_TOKEN', 'ghp_test')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    logged_in_user.api_token = 'tok_returning'
    db.session.commit()

    ds = Dataset(name='returning_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='returning_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    db.session.add(UserColabGist(
        user_id=logged_in_user.id, leaderboard_id=lb.id,
        gist_id='oldpergist', gist_owner='personalowner', sig='stale',
    ))
    db.session.commit()

    class _PatchOk:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                'id': 'oldpergist',
                'html_url': 'https://gist.github.com/personalowner/oldpergist',
                'owner': {'login': 'personalowner'},
            }

    with patch('requests.patch', return_value=_PatchOk()) as patch_mock, \
         patch('requests.post') as post_mock:
        resp = auth_client.get(f'/leaderboard/{lb.id}/colab_open')

    assert resp.status_code == 302
    patch_mock.assert_called_once()
    post_mock.assert_not_called()
    assert 'gist/personalowner/oldpergist' in resp.headers['Location']


def test_colab_notebook_direct_download_personalizes_for_logged_in_user(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """Direct .ipynb download also embeds the user's token."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    logged_in_user.api_token = 'direct_dl_tok'
    db.session.commit()

    ds = Dataset(name='dl_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='dl_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    resp = auth_client.get(f'/leaderboard/{lb.id}/colab_notebook.ipynb')
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "API_TOKEN = 'direct_dl_tok'" in body
    assert "API_TOKEN = ''" not in body


def test_colab_notebook_direct_download_anon_keeps_placeholder(
    client, db_session, monkeypatch,
):
    """Anonymous direct download → placeholder stays empty so the user
    can paste in a token themselves."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    ds = Dataset(name='anon_dl_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    lb = Leaderboard(name='anon_dl_lb', summary_metrics='', visibility='public')
    lb.datasets.append(ds); db.session.add(lb); db.session.commit()

    resp = client.get(f'/leaderboard/{lb.id}/colab_notebook.ipynb')
    assert resp.status_code == 200
    assert "API_TOKEN = ''" in resp.data.decode()


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
