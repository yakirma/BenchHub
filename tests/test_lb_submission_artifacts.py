"""Generated submission artifacts on the LB page.

Two server routes deliver the artifacts for the "Submit
predictions" card on /leaderboard/<id>:

  - /leaderboard/<id>/submission_script.py
  - /leaderboard/<id>/submission_notebook.ipynb

Both bake the LB id into the file, list one constructor per
declared pred field (so the LB's actual contract is visible at the
top of the file), and instruct the user to set BENCHHUB_TOKEN.
"""
from __future__ import annotations

import json
import os

from app import (
    Dataset,
    DatasetField,
    Leaderboard,
    User,
    app as flask_app,
    db,
)


def _seed_lb_with_pred(kind='label', extra_params=None):
    user = User(email='subart@bench.local', display_name='subart',
                oauth_provider='github', oauth_sub='subart-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='subart_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name='label',
                                kind='label', role='gt'))
    pred = DatasetField(dataset_id=ds.id, name='label_pred',
                        kind=kind, role='pred')
    if extra_params:
        pred.set_params(extra_params)
    db.session.add(pred)
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    lb = Leaderboard(name='subart_lb', summary_metrics='',
                     visibility='public', owner_user_id=user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return user, lb


def test_submission_script_route_returns_python(client, db_session):
    _, lb = _seed_lb_with_pred(kind='label')
    r = client.get(f'/leaderboard/{lb.id}/submission_script.py')
    assert r.status_code == 200
    assert r.mimetype.startswith('text/x-python')
    body = r.data.decode('utf-8')
    assert 'BENCHHUB_TOKEN' in body
    assert f'LEADERBOARD_ID = {lb.id}' in body
    assert 'bh.Client' in body
    # No `token="YOUR_API_TOKEN"` placeholder — env var is the contract.
    assert 'YOUR_API_TOKEN' not in body
    # Pred-field constructor matches the dataset's declared kind.
    assert 'label_pred=bh.Label(' in body


def test_submission_script_uses_label_list_constructor_with_correct_k(client, db_session):
    """label_list pred with k=5 generates `bh.LabelList([0] * 5, k=5)`."""
    _, lb = _seed_lb_with_pred(kind='label_list', extra_params={'k': 5})
    r = client.get(f'/leaderboard/{lb.id}/submission_script.py')
    body = r.data.decode('utf-8')
    assert 'bh.LabelList([0] * 5, k=5)' in body


def test_submission_notebook_route_returns_ipynb_json(client, db_session):
    _, lb = _seed_lb_with_pred(kind='label')
    r = client.get(f'/leaderboard/{lb.id}/submission_notebook.ipynb')
    assert r.status_code == 200
    assert r.mimetype.startswith('application/x-ipynb+json')
    nb = json.loads(r.data.decode('utf-8'))
    assert nb['nbformat'] == 4
    # One install cell + one Colab-secrets cell + one body cell + markdown.
    all_src = ''.join(
        ''.join(c.get('source', []))
        for c in nb['cells']
    )
    assert 'pip install -q benchhub-client' in all_src
    assert 'BENCHHUB_TOKEN' in all_src
    assert 'google.colab' in all_src
    assert 'userdata' in all_src
    assert f'LEADERBOARD_ID = {lb.id}' in all_src


def test_lb_page_renders_submit_action_buttons(client, db_session):
    """The LB page now has Colab + .ipynb + .py buttons in the Submit
    block, and the old top-toolbar Submit button is gone."""
    _, lb = _seed_lb_with_pred()
    body = client.get(f'/leaderboard/{lb.id}').data.decode('utf-8')
    # Three new action buttons.
    assert 'Submit via Colab' in body
    assert 'Download .ipynb' in body
    assert 'Download script (.py)' in body
    # Code snippet uses the env-var path, not the literal placeholder.
    assert 'YOUR_API_TOKEN' not in body
    assert 'BENCHHUB_TOKEN' in body
    # The old toolbar Submit button is no longer in the action row.
    # (The text "Submit" still appears in headings; check the
    # specific old button class instead.)
    assert 'btn btn-success' in body  # the new Colab button has this
    assert 'bi-rocket-takeoff me-1"></i>Submit\n' not in body  # old text-only "Submit" button


def test_bh_client_reads_BENCHHUB_TOKEN_env(monkeypatch):
    """bh.Client() with no explicit token reads BENCHHUB_TOKEN first."""
    import benchhub as bh
    monkeypatch.setenv('BENCHHUB_TOKEN', 'tok_new')
    monkeypatch.delenv('BENCHHUB_API_TOKEN', raising=False)
    c = bh.Client()
    assert c.token == 'tok_new'


def test_bh_client_falls_back_to_BENCHHUB_API_TOKEN_env(monkeypatch):
    """Backwards compat: BENCHHUB_API_TOKEN still works."""
    import benchhub as bh
    monkeypatch.delenv('BENCHHUB_TOKEN', raising=False)
    monkeypatch.setenv('BENCHHUB_API_TOKEN', 'tok_legacy')
    c = bh.Client()
    assert c.token == 'tok_legacy'


def test_bh_client_prefers_new_env_var_over_legacy(monkeypatch):
    import benchhub as bh
    monkeypatch.setenv('BENCHHUB_TOKEN', 'tok_new')
    monkeypatch.setenv('BENCHHUB_API_TOKEN', 'tok_legacy')
    assert bh.Client().token == 'tok_new'
