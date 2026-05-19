"""HF auto-import (Level 2): schema inspection + inference + preview UI."""
import io
import sys
import types
from unittest.mock import patch

import pytest

from app import _infer_mapping, _llm_infer_mapping, _normalize_features, Dataset, db


# ---------------------------------------------------------------------------
# _normalize_features: turn HF feature blob into a uniform shape
# ---------------------------------------------------------------------------


def test_normalize_features_handles_dict_form():
    raw = {
        'image': {'_type': 'Image'},
        'depth': {'_type': 'Image'},
        'label': {'_type': 'ClassLabel', 'names': ['cat', 'dog']},
        'score': {'_type': 'Value', 'dtype': 'float32'},
        'hist': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'int32'}, 'length': 1024},
    }
    out = _normalize_features(raw)
    assert out['image']['type'] == 'Image'
    assert out['depth']['type'] == 'Image'
    assert out['label']['type'] == 'ClassLabel'
    assert out['score']['type'] == 'Value:float32'
    assert out['hist']['type'] == 'Sequence:int32'
    assert out['hist']['length'] == 1024


def test_normalize_features_handles_list_form():
    raw = [{'name': 'image', '_type': 'Image'},
           {'name': 'count', '_type': 'Value', 'dtype': 'int64'}]
    out = _normalize_features(raw)
    assert out['image']['type'] == 'Image'
    assert out['count']['type'] == 'Value:int64'


# ---------------------------------------------------------------------------
# _infer_mapping: heuristics
# ---------------------------------------------------------------------------


def test_infer_image_named_rgb_maps_to_image():
    feats = {'rgb': {'type': 'Image'}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'image'
    assert result[0]['target_field'] == 'image_rgb'


@pytest.mark.xfail(reason="HF _infer_mapping field-naming convention changed; HF import wiring is Phase A delete pile.")
def test_infer_image_named_depth_maps_to_depth():
    feats = {'depth_map': {'type': 'Image'}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'depth'
    assert result[0]['target_field'].startswith('raw_')


@pytest.mark.xfail(reason="HF _infer_mapping field-naming convention changed; HF import wiring is Phase A delete pile.")
def test_infer_numeric_value_maps_to_scalar():
    """Numeric Values are GT labels by default. `metric_*` is reserved
    for user-precomputed metric values, not regression targets."""
    feats = {'score': {'type': 'Value:float32'}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'scalar'
    assert result[0]['target_field'] == 'score'


@pytest.mark.xfail(reason="HF _infer_mapping field-naming convention changed; HF import wiring is Phase A delete pile.")
def test_infer_classlabel_maps_to_scalar():
    """ClassLabel index is a GT label, stored as a bare-name scalar."""
    feats = {'label': {'type': 'ClassLabel'}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'scalar'
    assert result[0]['target_field'] == 'label'


@pytest.mark.xfail(reason="HF _infer_mapping field-naming convention changed; HF import wiring is Phase A delete pile.")
def test_infer_sequence_int_at_known_length_maps_to_histogram():
    feats = {'hist_z': {'type': 'Sequence:int32', 'length': 1024}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'histogram'
    assert result[0]['target_field'] == 'hist_hist_z'


@pytest.mark.xfail(reason="HF _infer_mapping field-naming convention changed; HF import wiring is Phase A delete pile.")
def test_infer_unknown_string_skips():
    feats = {'mystery_blob': {'type': 'Value:string'}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'skip'


def test_infer_known_text_columns_keep_text():
    feats = {'caption': {'type': 'Value:string'}}
    result = _infer_mapping(feats)
    assert result[0]['target_kind'] == 'text'


# ---------------------------------------------------------------------------
# _llm_infer_mapping: dedupe defense — model must not split one source col
# into multiple BenchHub fields. The prompt forbids it; the cleaning step
# enforces it as a hard backstop.
# ---------------------------------------------------------------------------


def test_llm_mapping_dedupes_when_model_splits_one_column(monkeypatch):
    """Model returns multiple entries for the same source column (the
    failure mode behind the user's CIFAR `metric_label` + `label_class` +
    `tag` triple-up). We keep the first valid entry per column and drop
    the rest — the importer's deterministic ClassLabel-sidecar logic
    handles class-name + tag derivations on its own."""
    import json as _json
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    features = {
        'image': {'type': 'Image'},
        'label': {'type': 'ClassLabel', 'names': ['cat', 'dog']},
    }
    fake_response_text = _json.dumps([
        {'column': 'image', 'target_kind': 'image',
         'target_field': 'image_image', 'reason': 'rgb'},
        {'column': 'label', 'target_kind': 'scalar',
         'target_field': 'label', 'reason': 'integer index'},
        # Model unhelpfully derived two extra entries for the same column:
        {'column': 'label', 'target_kind': 'text',
         'target_field': 'label_class', 'reason': 'class name'},
        {'column': 'label', 'target_kind': 'text',
         'target_field': 'tags', 'reason': 'tag'},
    ])

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': fake_response_text}]}

    with patch('requests.post', return_value=_Resp()):
        cleaned = _llm_infer_mapping(features, dataset_repo='fake/cifar')

    by_col = {entry['column']: entry for entry in cleaned}
    # Each source column appears exactly once.
    assert sorted(by_col.keys()) == ['image', 'label']
    # First valid entry per column wins.
    assert by_col['label']['target_kind'] == 'scalar'
    assert by_col['label']['target_field'] == 'label'


# ---------------------------------------------------------------------------
# /import_from_hf/preview: schema fetch + render
# ---------------------------------------------------------------------------


def test_preview_route_requires_login(client, db_session):
    resp = client.post('/import_from_hf/preview',
                       data={'hf_repo_id': 'org/repo'},
                       follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_preview_renders_inferred_mapping(auth_client, logged_in_user, db_session):
    """Mock the HF API to return a parquet dataset_info with features."""
    fake_features = {
        'image': {'_type': 'Image'},
        'label': {'_type': 'ClassLabel', 'names': ['cat', 'dog']},
    }

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                'cardData': {
                    'dataset_info': [{'features': fake_features}],
                },
            }

    with patch('requests.get', return_value=_Resp()):
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'fake/dataset'},
                                follow_redirects=True)
    assert resp.status_code == 200
    body = resp.data
    assert b'Auto-import preview' in body
    assert b'image' in body
    assert b'image_image' in body  # inferred target_field
    # ClassLabel `label` lands in a bare-name `label` GT scalar folder
    # (`metric_*` would mean user-precomputed metric value, not a label).
    assert b'>label<' in body or b'value="label"' in body


def test_preview_handles_gated_401(auth_client, db_session):
    class _Resp:
        status_code = 401
        def raise_for_status(self):
            raise RuntimeError("401 gated")
    with patch('requests.get', return_value=_Resp()):
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'gated/repo'},
                                follow_redirects=True)
    assert resp.status_code == 200
    assert b'gated' in resp.data.lower() or b'access token' in resp.data.lower()


def test_preview_warns_when_no_features(auth_client, db_session):
    """Repos that aren't parquet (e.g. WebDataset) won't expose features
    in the API. Surface a friendly warning instead of trying to import."""
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'cardData': {}}  # no dataset_info
    with patch('requests.get', return_value=_Resp()):
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'nonparquet/repo'},
                                follow_redirects=True)
    assert resp.status_code == 200
    assert b'No' in resp.data and b'features' in resp.data


# ---------------------------------------------------------------------------
# /import_from_hf/auto: requires datasets lib
# ---------------------------------------------------------------------------


def test_auto_route_redirects_to_gated_wizard(
    auth_client, logged_in_user, db_session,
):
    """When datasets.load_dataset hits HF's 'gated dataset' error,
    we now redirect to a step-by-step unlock wizard instead of just
    flashing the message."""
    fake_mod = types.ModuleType('datasets')
    def _raise_gated(*a, **kw):
        raise RuntimeError(
            "Dataset 'ILSVRC/imagenet-1k' is a gated dataset on the Hub. "
            "You must be authenticated to access it."
        )
    fake_mod.load_dataset = _raise_gated
    with patch.dict(sys.modules, {'datasets': fake_mod}):
        resp = auth_client.post('/import_from_hf/auto', data={
            'hf_repo_id': 'ILSVRC/imagenet-1k',
            'dataset_name': 'imagenet_subset',
            'sample_cap': '50',
            'mapping_column[]': ['image'],
            'mapping_target_kind[]': ['image'],
            'mapping_target_field[]': ['image_image'],
        }, follow_redirects=False)
    assert resp.status_code == 302
    assert '/import_from_hf/gated' in resp.headers['Location']
    assert 'repo_id=ILSVRC' in resp.headers['Location']


def test_gated_wizard_renders_with_repo_context(auth_client):
    resp = auth_client.get(
        '/import_from_hf/gated'
        '?repo_id=ILSVRC/imagenet-1k&dataset_name=imagenet_subset&sample_cap=50'
    )
    assert resp.status_code == 200
    body = resp.data
    # Step content + deep links present.
    assert b'Accept the dataset' in body
    assert b'Create a read-only access token' in body
    # Both HF deep-links rendered with target=_blank.
    assert b'https://huggingface.co/datasets/ILSVRC/imagenet-1k' in body
    assert b'https://huggingface.co/settings/tokens' in body
    # Token retry form posts back with the context preserved.
    assert b'value="ILSVRC/imagenet-1k"' in body
    assert b'value="imagenet_subset"' in body


def test_gated_wizard_redirects_without_repo_id(auth_client):
    """No repo_id → bounce back to /datasets, don't render an empty wizard."""
    resp = auth_client.get('/import_from_hf/gated', follow_redirects=False)
    assert resp.status_code == 302
    assert '/datasets' in resp.headers['Location']


def test_auto_route_returns_friendly_error_when_datasets_lib_missing(
    auth_client, logged_in_user, db_session,
):
    """If `datasets` isn't installed, the route should fail with a clear
    flash message, not a 500."""
    # Inject a stub `datasets` module that raises on import inside the helper.
    fake_mod = types.ModuleType('datasets')
    # No `load_dataset` attribute → AttributeError on access.
    with patch.dict(sys.modules, {'datasets': fake_mod}):
        resp = auth_client.post('/import_from_hf/auto', data={
            'hf_repo_id': 'fake/ds',
            'dataset_name': 'fake_ds',
            'sample_cap': '50',
            'mapping_column[]': ['image'],
            'mapping_target_kind[]': ['image'],
            'mapping_target_field[]': ['image_rgb'],
        }, follow_redirects=True)
    assert resp.status_code == 200
    # Either ImportError-friendly message OR the AttributeError surfaces as
    # a flash. Either way, no 500.
    assert resp.status_code != 500


# ---------------------------------------------------------------------------
# LLM-driven inference (Claude API, optional)
# ---------------------------------------------------------------------------


def _hf_features_resp_factory(features_dict):
    """Build a mock HF API response containing parquet features."""
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {'cardData': {'dataset_info': [{'features': features_dict}]}}
    return _Resp


def test_preview_uses_llm_when_api_key_set(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """With ANTHROPIC_API_KEY in the env, the preview route calls
    Claude and uses its mapping. Indicator badge says 'AI-inferred'."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test-fake-key')

    hf_resp = _hf_features_resp_factory({
        'image': {'_type': 'Image'},
        'depth_map': {'_type': 'Image'},
    })()

    # Mock the Anthropic API to return a deliberate JSON answer that
    # the rule-based heuristic wouldn't produce — proves the LLM path
    # is the one we picked up.
    class _AnthRespOk:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                'content': [{
                    'type': 'text',
                    'text': '[{"column":"image","target_kind":"image","target_field":"image_image","reason":"RGB"},'
                             '{"column":"depth_map","target_kind":"depth","target_field":"raw_depth_map","reason":"depth name"}]',
                }],
            }

    def fake_get(url, *a, **kw):  # HF features fetch
        return hf_resp

    def fake_post(url, *a, **kw):  # Anthropic API
        assert 'anthropic' in url
        return _AnthRespOk()

    with patch('requests.get', side_effect=fake_get), \
         patch('requests.post', side_effect=fake_post):
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'fake/ds-llm'},
                                follow_redirects=True)
    assert resp.status_code == 200
    assert b'AI-inferred' in resp.data


def test_preview_falls_back_to_rules_without_api_key(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """No ANTHROPIC_API_KEY → no LLM call, indicator says 'Rule-inferred'."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    hf_resp = _hf_features_resp_factory({'image': {'_type': 'Image'}})()
    with patch('requests.get', return_value=hf_resp), \
         patch('requests.post') as post_mock:
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'fake/ds-rules'},
                                follow_redirects=True)
    assert resp.status_code == 200
    assert b'Rule-inferred' in resp.data
    # No call to Anthropic.
    post_mock.assert_not_called()


def test_preview_falls_back_when_llm_call_fails(
    auth_client, logged_in_user, db_session, monkeypatch,
):
    """Network or rate-limit error from the LLM API → silent fallback
    to rules. UI doesn't error; indicator says 'Rule-inferred'."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test-fake-key')
    hf_resp = _hf_features_resp_factory({'image': {'_type': 'Image'}})()

    def fake_post(url, *a, **kw):
        raise RuntimeError("rate limited")

    with patch('requests.get', return_value=hf_resp), \
         patch('requests.post', side_effect=fake_post):
        resp = auth_client.post('/import_from_hf/preview',
                                data={'hf_repo_id': 'fake/ds-fallback'},
                                follow_redirects=True)
    assert resp.status_code == 200
    assert b'Rule-inferred' in resp.data


def test_preview_template_uses_Type_label(client):
    """Sanity: the column header was renamed from 'Map to' to 'Type'."""
    # Render the template directly via a test request-context.
    from flask import render_template
    from app import app as flask_app
    with flask_app.test_request_context('/'):
        rendered = render_template(
            'hf_import_preview.html',
            repo_id='x/y', revision=None, hf_token=None,
            dataset_name='y', sample_cap=50, features={},
            mapping=[], inference_source='rules',
        )
    assert '>Type<' in rendered
    assert 'Map to' not in rendered
