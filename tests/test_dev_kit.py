"""bh-client dev kit: programmatic metric/visualization authoring +
local-test helpers."""
import pytest

import benchhub as bh
from app import GlobalMetric, GlobalVisualization, User, db, generate_api_token


@pytest.fixture
def api_user(db_session):
    u = User(email='dev@bench.local', display_name='dev',
             oauth_provider='github', oauth_sub='dev-1',
             is_admin=False, api_token=generate_api_token())
    db.session.add(u); db.session.commit()
    return u


def _client(client, user):
    return bh.Client(token=user.api_token, base_url="http://test",
                     transport=bh.FlaskTestClientTransport(client))


# --- create_metric / create_visualization (server round-trip) -------------

def test_create_metric_from_source_derives_input_kinds(client, api_user):
    c = _client(client, api_user)
    code = ("def acc(gt: bh.Label, pred: bh.Label):\n"
            "    return 1.0 if gt.value == pred.value else 0.0\n")
    res = c.create_metric("my_acc", code, description="exact label match")
    assert res['name'] == 'my_acc' and res['visibility'] == 'private'
    assert res['input_kinds'] == ['label', 'label']        # from annotations
    m = db.session.get(GlobalMetric, res['id'])
    assert m.owner_user_id == api_user.id
    assert m.python_code.strip() == code.strip()


def test_create_metric_from_function_object(client, api_user):
    c = _client(client, api_user)

    def rmse(gt: bh.Depth, pred: bh.Depth):
        import numpy as np
        return float(np.sqrt(np.mean((gt.array - pred.array) ** 2)))

    res = c.create_metric("my_rmse", rmse)
    assert res['input_kinds'] == ['depth', 'depth']
    assert 'def rmse' in db.session.get(GlobalMetric, res['id']).python_code


def test_create_metric_duplicate_name_conflicts(client, api_user):
    c = _client(client, api_user)
    c.create_metric("dup", "def m(x):\n    return float(x)\n")
    with pytest.raises(bh.BenchHubAPIError) as ei:
        c.create_metric("dup", "def m(x):\n    return float(x)\n")
    assert ei.value.status_code == 409


def test_create_visualization(client, api_user):
    c = _client(client, api_user)
    code = ("def v(gt):\n    from PIL import Image\n"
            "    return Image.new('RGB', (4, 4))\n")
    res = c.create_visualization("my_viz", code)
    assert res['kind'] == 'visualization'
    assert db.session.get(GlobalVisualization, res['id']).owner_user_id == api_user.id


def test_create_metric_requires_token(monkeypatch):
    monkeypatch.delenv("BENCHHUB_API_TOKEN", raising=False)
    c = bh.Client(token="", base_url="http://x")
    with pytest.raises(ValueError, match="API token"):
        c.create_metric("x", "def m(x):\n    return 1.0\n")


# --- local-test helpers (no server) ---------------------------------------

def test_author_test_metric_runs_locally():
    def m(gt, pred):
        return 1.0 if gt == pred else 0.0
    assert bh.author.test_metric(m, gt=1, pred=1) == 1.0
    assert bh.author.test_metric_batch(
        m, [{'gt': 1, 'pred': 1}, {'gt': 1, 'pred': 2}]) == [1.0, 0.0]


def test_author_test_visualization_returns_image_and_guards_type():
    from PIL import Image

    def v(x):
        return Image.new('RGB', (8, 8), (1, 2, 3))
    img = bh.author.test_visualization(v, x=1)
    assert img.size == (8, 8)

    with pytest.raises(TypeError, match="PIL.Image"):
        bh.author.test_visualization(lambda x: 42, x=1)
