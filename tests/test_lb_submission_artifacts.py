"""Generated submission artifacts on the LB page.

Two server routes deliver the artifacts for the "Submit
predictions" card on /leaderboard/<id>:

  - /leaderboard/<id>/submission_script.py
  - /leaderboard/<id>/submission_notebook.ipynb

Both bake the LB id into the file, list one constructor per
declared pred field (so the LB's actual contract is visible at the
top of the file), and instruct the user to set BENCHHUB_API_TOKEN.
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
    assert 'BENCHHUB_API_TOKEN' in body
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
    assert 'pip install -q --upgrade benchhub-client' in all_src
    assert 'BENCHHUB_API_TOKEN' in all_src
    assert 'google.colab' in all_src
    assert 'userdata' in all_src
    assert f'LEADERBOARD_ID = {lb.id}' in all_src


def test_lb_page_renders_submit_action_buttons(client, db_session):
    """The LB page now has Colab + .ipynb + .py buttons in the Submit
    block, and the old top-toolbar Submit button is gone."""
    _, lb = _seed_lb_with_pred()
    body = client.get(f'/leaderboard/{lb.id}').data.decode('utf-8')
    # Three new action buttons.
    assert 'Open in Colab' in body
    assert 'Download notebook' in body
    assert 'Download script (.py)' in body
    # Code snippet uses the env-var path, not the literal placeholder.
    assert 'YOUR_API_TOKEN' not in body
    assert 'BENCHHUB_API_TOKEN' in body
    # The old toolbar Submit button is no longer in the action row.
    # (The text "Submit" still appears in headings; check the
    # specific old button class instead.)
    assert 'btn btn-success' in body  # the new Colab button has this
    assert 'bi-rocket-takeoff me-1"></i>Submit\n' not in body  # old text-only "Submit" button


def test_inline_snippet_on_lb_page_reflects_pred_contract(client, db_session):
    """The Python snippet in the Submit-predictions card emits one
    bh.<Kind>(...) line per declared pred field on the LB, instead
    of a hardcoded `label_pred=bh.Label(...)`. label_list preds
    show `bh.LabelList(..., k=K)` with the declared K."""
    user = User(email='inline-snip@bench.local', display_name='inline',
                oauth_provider='github', oauth_sub='inline-1')
    db.session.add(user); db.session.commit()
    ds = Dataset(name='inline_snip_ds', visibility='public',
                 owner_user_id=user.id)
    db.session.add(ds); db.session.flush()
    db.session.add_all([
        DatasetField(dataset_id=ds.id, name='label', kind='label', role='gt'),
    ])
    db.session.commit()
    # Two pred fields on the LB's contract — one single-class, one top-K.
    lb = Leaderboard(
        name='inline_snip_lb', visibility='public', owner_user_id=user.id,
        required_pred_fields_json=json.dumps([
            {'name': 'label_pred',      'kind': 'label',      'role': 'pred', 'params': {}},
            {'name': 'label_topk_pred', 'kind': 'label_list', 'role': 'pred',
             'params': {'k': 7}},
        ]),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    os.makedirs(os.path.join(flask_app.config['UPLOAD_FOLDER'], 'datasets', str(ds.id)),
                exist_ok=True)
    body = client.get(f'/leaderboard/{lb.id}').data.decode()
    # Both pred fields rendered with the right constructors.
    assert 'label_pred=bh.Label(0)' in body
    assert 'label_topk_pred=bh.LabelList([0] * 7, k=7)' in body
    # Old hardcoded line is gone.
    assert 'label_pred=bh.Label(predicted_class)' not in body


def test_bh_client_reads_BENCHHUB_API_TOKEN_env(monkeypatch):
    """bh.Client() with no explicit token reads BENCHHUB_API_TOKEN
    from the environment."""
    import benchhub as bh
    monkeypatch.setenv('BENCHHUB_API_TOKEN', 'tok_from_env')
    c = bh.Client()
    assert c.token == 'tok_from_env'


def test_bh_client_explicit_token_beats_env(monkeypatch):
    """Explicit token= arg wins over the env var."""
    import benchhub as bh
    monkeypatch.setenv('BENCHHUB_API_TOKEN', 'tok_env')
    c = bh.Client(token='tok_explicit')
    assert c.token == 'tok_explicit'
