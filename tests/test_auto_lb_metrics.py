"""Auto-create-leaderboard-with-metrics-and-visualizations.

Tests the proposer + LB-creation helper:
- ClassLabel-shaped GT scalar (sibling `<col>_class`) → top-1 accuracy
  metric (higher_is_better) + confusion-matrix visualization.
- Plain numeric GT scalar → MAE metric (lower_is_better).
- Strict name-match dedupe against the GlobalMetric / GlobalVisualization
  libraries.
- LLM code generation when missing, with deterministic fallbacks when
  the LLM is unavailable or returns code that fails the safety check.
"""
import json
from unittest.mock import patch

import pytest

from app import (
    CustomField, Dataset, GlobalMetric, GlobalVisualization,
    Leaderboard, LeaderboardMetric, LeaderboardVisualization,
    Sample, db,
    _propose_metrics_for_dataset,
    _propose_visualizations_for_dataset,
    _llm_generate_metric_code,
    _llm_generate_visualization_code,
    _auto_create_lb_with_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures — GT side mirrors what the HF auto-importer now writes:
# `<col>/<sample>.txt` (scalar), NOT `metric_<col>/`.
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_classlabel_shape(db_session):
    """Dataset that mirrors a HF ClassLabel auto-import: a `label` GT
    scalar plus a sibling `label_class` text field signaling the
    ClassLabel name list."""
    ds = Dataset(name='cls_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='label', field_type='scalar',
        value_float=0.0,
    ))
    db.session.add(CustomField(
        sample_id=s.id, name='label_class', field_type='text',
        value_text='cat',
    ))
    db.session.commit()
    return ds


@pytest.fixture
def dataset_numeric_shape(db_session):
    """Dataset with a plain numeric GT scalar and no class sidecar."""
    ds = Dataset(name='num_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='score', field_type='scalar',
        value_float=0.42,
    ))
    db.session.commit()
    return ds


# ---------------------------------------------------------------------------
# _propose_metrics_for_dataset
# ---------------------------------------------------------------------------


def test_propose_classlabel_picks_top1_accuracy(dataset_classlabel_shape):
    proposals = _propose_metrics_for_dataset(dataset_classlabel_shape)
    assert len(proposals) == 1
    p = proposals[0]
    assert p['global_name'] == 'top1_label'
    assert p['sort_direction'] == 'higher_is_better'
    # GT scalar `label` is compared against submission's `label_pred`.
    assert p['arg_mappings'] == {'gt': 'gt_label', 'pred': 'sub_label_pred'}
    assert 'def top1_label' in p['fallback_code']
    # The proposer surfaces the submission contract so the colab notebook
    # + flash message can tell the user what folders to ship.
    pf = p['pred_fields']
    assert len(pf) == 1
    assert pf[0]['name'] == 'label_pred'
    assert pf[0]['kind'] == 'scalar'
    assert pf[0]['gt_field'] == 'label'
    assert 'class index' in pf[0]['description']


def test_propose_numeric_picks_mae(dataset_numeric_shape):
    proposals = _propose_metrics_for_dataset(dataset_numeric_shape)
    assert len(proposals) == 1
    p = proposals[0]
    assert p['global_name'] == 'mae_score'
    assert p['sort_direction'] == 'lower_is_better'
    assert p['arg_mappings'] == {'gt': 'gt_score', 'pred': 'sub_score_pred'}
    assert 'def mae_score' in p['fallback_code']


def test_propose_skips_unhandled_field_types(db_session):
    """Datasets with only JSON / text / histogram GT yield no
    proposals — nothing the structured-GT proposer can metric."""
    ds = Dataset(name='json_only', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='meta', field_type='json',
        value_text='/path/to/m.json',
    ))
    db.session.commit()
    assert _propose_metrics_for_dataset(ds) == []


def test_propose_image_field_picks_psnr(db_session):
    """RGB image GT (denoising / super-res shape) → PSNR proposal,
    higher_is_better, pred field kind = 'image'."""
    ds = Dataset(name='img_psnr', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='clean', field_type='image',
        value_text='clean/s00000.png',
    ))
    db.session.commit()
    proposals = _propose_metrics_for_dataset(ds)
    assert len(proposals) == 1
    p = proposals[0]
    assert p['global_name'] == 'psnr_clean'
    assert p['sort_direction'] == 'higher_is_better'
    assert p['arg_mappings'] == {'gt': 'gt_clean', 'pred': 'sub_clean_pred'}
    assert p['pred_fields'][0]['kind'] == 'image'


def test_propose_mask_named_image_picks_miou(db_session):
    """Image GT whose name contains 'mask' / 'seg' / 'label' → mIoU
    proposal (not PSNR), pred field kind = 'mask'."""
    ds = Dataset(name='seg_miou', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='annotation_mask', field_type='image',
        value_text='masks/s00000.png',
    ))
    db.session.commit()
    proposals = _propose_metrics_for_dataset(ds)
    names = [p['global_name'] for p in proposals]
    assert names == ['miou_annotation_mask']
    assert proposals[0]['sort_direction'] == 'higher_is_better'
    assert proposals[0]['pred_fields'][0]['kind'] == 'mask'


def test_propose_depth_field_picks_three_metrics(db_session):
    """Depth GT → RMSE + abs-rel + a1 (three metrics from one column).
    Sort directions split: RMSE / abs-rel are lower-better, a1 is
    higher-better. All three pred-fields agree on kind='depth'."""
    ds = Dataset(name='depth_three', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='depth_map', field_type='depth',
        value_text='depth_maps/s00000_4x4.npz',
    ))
    db.session.commit()
    proposals = _propose_metrics_for_dataset(ds)
    names = sorted(p['global_name'] for p in proposals)
    assert names == ['a1_depth_map', 'abs_rel_depth_map', 'rmse_depth_map']
    sort_dirs = {p['global_name']: p['sort_direction'] for p in proposals}
    assert sort_dirs['rmse_depth_map'] == 'lower_is_better'
    assert sort_dirs['abs_rel_depth_map'] == 'lower_is_better'
    assert sort_dirs['a1_depth_map'] == 'higher_is_better'
    for p in proposals:
        assert p['pred_fields'][0]['kind'] == 'depth'
        assert p['pred_fields'][0]['name'] == 'depth_map_pred'


def test_propose_depth_visualization_emits_error_heatmap(db_session):
    from app import _propose_visualizations_for_dataset
    ds = Dataset(name='depth_viz', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='depth_map', field_type='depth',
        value_text='depth_maps/s00000_4x4.npz',
    ))
    db.session.commit()
    viz = _propose_visualizations_for_dataset(ds)
    assert len(viz) == 1
    v = viz[0]
    assert v['global_name'] == 'depth_error_heatmap_depth_map'
    assert v['is_aggregated'] is True
    assert v['accepts_aggregated_inputs'] is True


def test_static_depth_metric_runs_on_real_arrays():
    """Smoke-test the deterministic fallback code: exec the source
    and call it on a couple of arrays to confirm the math is sane."""
    from app import _proposal_rmse_depth, _proposal_abs_rel_depth, _proposal_a1_depth
    import numpy as _np
    scope = {'np': _np}
    for proposal_fn in (_proposal_rmse_depth, _proposal_abs_rel_depth, _proposal_a1_depth):
        p = proposal_fn('depth_map')
        exec(p['fallback_code'], scope)
    rmse = scope['rmse_depth_map']
    abs_rel = scope['abs_rel_depth_map']
    a1 = scope['a1_depth_map']
    gt = _np.array([[1.0, 2.0], [4.0, 5.0]])
    pred = gt.copy()  # perfect prediction
    assert rmse(gt, pred) == 0.0
    assert abs_rel(gt, pred) == 0.0
    assert a1(gt, pred) == 1.0
    pred_off = gt + 1.0
    assert rmse(gt, pred_off) == pytest.approx(1.0)
    assert rmse(None, pred) != rmse(None, pred)  # NaN propagation


def test_static_psnr_runs_on_real_images():
    from app import _proposal_psnr_image
    import numpy as _np
    scope = {'np': _np}
    p = _proposal_psnr_image('clean')
    exec(p['fallback_code'], scope)
    psnr = scope['psnr_clean']
    gt = _np.full((4, 4, 3), 100, dtype=_np.uint8)
    assert psnr(gt, gt) == 100.0  # identical → clamp
    pred = gt.astype(_np.int32) + 25
    assert psnr(gt, pred.clip(0, 255).astype(_np.uint8)) > 10  # finite + reasonable


def test_static_miou_runs_on_real_masks():
    from app import _proposal_miou_mask
    import numpy as _np
    scope = {'np': _np}
    p = _proposal_miou_mask('annotation_mask')
    exec(p['fallback_code'], scope)
    miou = scope['miou_annotation_mask']
    # Two-class mask, perfect overlap.
    gt = _np.array([[0, 0], [1, 1]])
    assert miou(gt, gt) == 1.0
    # Disjoint prediction → IoU = 0 for both classes.
    pred = _np.array([[1, 1], [0, 0]])
    assert miou(gt, pred) == 0.0


def test_propose_numeric_pred_fields_describe_regression_target(dataset_numeric_shape):
    proposals = _propose_metrics_for_dataset(dataset_numeric_shape)
    pf = proposals[0]['pred_fields']
    assert len(pf) == 1
    assert pf[0]['name'] == 'score_pred'
    assert pf[0]['kind'] == 'scalar'
    assert pf[0]['gt_field'] == 'score'
    assert 'numeric' in pf[0]['description'] or 'value' in pf[0]['description']


def test_propose_skips_class_name_sidecar(dataset_classlabel_shape):
    """`<col>_class` text columns are sidecars to ClassLabel scalars,
    not standalone GT scalars — don't propose a metric for them."""
    proposals = _propose_metrics_for_dataset(dataset_classlabel_shape)
    assert {p['global_name'] for p in proposals} == {'top1_label'}


# ---------------------------------------------------------------------------
# _propose_visualizations_for_dataset
# ---------------------------------------------------------------------------


def test_propose_visualization_for_classlabel_is_confusion_matrix(
    dataset_classlabel_shape,
):
    viz = _propose_visualizations_for_dataset(dataset_classlabel_shape)
    assert len(viz) == 1
    v = viz[0]
    assert v['global_name'] == 'confusion_matrix_label'
    assert v['is_aggregated'] is True
    assert v['accepts_aggregated_inputs'] is True
    assert v['arg_mappings'] == {'gt': 'gt_label', 'pred': 'sub_label_pred'}
    assert 'def confusion_matrix_label' in v['fallback_code']
    assert 'PIL' in v['fallback_code'] or 'Image' in v['fallback_code']


def test_propose_visualization_skips_numeric_only_datasets(
    dataset_numeric_shape,
):
    """Regression-style scalars get no canned visualization (users add
    scatter plots, etc. manually). Out of scope for the heuristic."""
    assert _propose_visualizations_for_dataset(dataset_numeric_shape) == []


def test_static_confusion_matrix_actually_builds_an_image(
    dataset_classlabel_shape,
):
    """Eval the deterministic fallback to make sure it parses + runs +
    returns a real PIL image. Belt-and-braces against typos in the
    code-string template."""
    viz = _propose_visualizations_for_dataset(dataset_classlabel_shape)
    assert viz, "expected one viz proposal"
    code = viz[0]['fallback_code']
    scope = {}
    exec(code, scope)
    fn = scope['confusion_matrix_label']
    img = fn([0, 1, 1, 0], [0, 1, 0, 0])  # 4 samples, mixed
    assert hasattr(img, 'size'), "expected a PIL.Image-like object"


# ---------------------------------------------------------------------------
# _llm_generate_metric_code (existing) + _llm_generate_visualization_code
# ---------------------------------------------------------------------------


def test_metric_llm_generate_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert _llm_generate_metric_code('top1_label', 'classlabel') is None


def test_metric_llm_generate_rejects_response_with_wrong_function_name(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': 'def wrong_name(gt, pred): return 0.0'}]}

    with patch('requests.post', return_value=_Resp()):
        out = _llm_generate_metric_code('top1_label', 'classlabel')
    assert out is None


def test_viz_llm_generate_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    out = _llm_generate_visualization_code(
        'confusion_matrix_label', 'classlabel',
        is_aggregated=True, accepts_aggregated_inputs=True,
    )
    assert out is None


def test_viz_llm_generate_returns_python_when_api_key_set(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    fake_code = (
        "def confusion_matrix_label(gt, pred):\n"
        "    from PIL import Image\n"
        "    return Image.new('L', (32, 32), 0)\n"
    )

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': fake_code}]}

    with patch('requests.post', return_value=_Resp()):
        out = _llm_generate_visualization_code(
            'confusion_matrix_label', 'cm',
            is_aggregated=True, accepts_aggregated_inputs=True,
        )
    assert out is not None
    assert 'def confusion_matrix_label(' in out


def test_viz_llm_generate_rejects_response_with_wrong_function_name(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {'content': [{'type': 'text', 'text': 'def wrong(gt, pred): pass'}]}

    with patch('requests.post', return_value=_Resp()):
        out = _llm_generate_visualization_code(
            'confusion_matrix_label', 'cm',
            is_aggregated=True, accepts_aggregated_inputs=True,
        )
    assert out is None


# ---------------------------------------------------------------------------
# _auto_create_lb_with_metrics — attaches BOTH metrics and visualizations
# ---------------------------------------------------------------------------


def test_auto_create_lb_attaches_metric_and_visualization_for_classlabel(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    ok, msg, lb_id = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'cls_lb', owner_user_id=logged_in_user.id,
    )
    assert ok, msg
    # Flash message surfaces the prediction contract so the user knows
    # what their submissions need to ship — once per pred field, even
    # though the metric and viz both reference `label_pred`.
    assert 'label_pred' in msg
    assert msg.count('label_pred') == 1
    lb = Leaderboard.query.get(lb_id)
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).all()
    lvs = LeaderboardVisualization.query.filter_by(leaderboard_id=lb.id).all()
    assert [lm.global_metric.name for lm in lms] == ['top1_label']
    assert [lv.global_visualization.name for lv in lvs] == ['confusion_matrix_label']
    # The visualization is aggregated end-to-end.
    gv = lvs[0].global_visualization
    assert gv.is_aggregated is True
    assert gv.accepts_aggregated_inputs is True


# ---------------------------------------------------------------------------
# _lb_submission_pred_fields: derive submission contract from arg_mappings
# ---------------------------------------------------------------------------


def test_lb_submission_pred_fields_dedupes_across_metrics_and_viz(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    """The metric and the viz both reference `sub_label_pred` —
    derived schema lists it once."""
    from app import _lb_submission_pred_fields
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    _, _, lb_id = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'pred_dedupe_lb',
        owner_user_id=logged_in_user.id,
    )
    lb = Leaderboard.query.get(lb_id)
    schema = _lb_submission_pred_fields(lb)
    assert len(schema) == 1
    entry = schema[0]
    assert entry['name'] == 'label_pred'
    assert entry['gt_field'] == 'label'
    assert entry['kind'] == 'scalar'
    # Used-by is the union of metric + viz target names that consume
    # this pred field — both reference `sub_label_pred`.
    assert sorted(entry['used_by']) == sorted([
        'top-1 accuracy (label)', 'confusion matrix (label)',
    ])
    # Description differentiates ClassLabel from regression-style.
    assert 'class index' in entry['description'].lower()


def test_lb_submission_pred_fields_ignores_non_pred_sub_keys(db_session, logged_in_user):
    """Bare `sub_<x>` (precomputed metric value, no `_pred` suffix)
    isn't a submission-side prediction — it's user-precomputed metric
    input. The helper filters those out so the colab notebook only
    enumerates fields the user actually has to author."""
    from app import _lb_submission_pred_fields
    ds = Dataset(name='ignore_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s1'))
    lb = Leaderboard(name='ignore_lb', summary_metrics='',
                     owner_user_id=logged_in_user.id)
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    gm = GlobalMetric(
        name='precomputed_thing', description='id',
        python_code='def precomputed_thing(value): return value\n',
        owner_user_id=logged_in_user.id,
    )
    db.session.add(gm); db.session.flush()
    lm = LeaderboardMetric(
        leaderboard_id=lb.id, global_metric_id=gm.id,
        # `sub_metric_score` — no `_pred` suffix → not a prediction field.
        arg_mappings=json.dumps({'value': 'sub_metric_score'}),
        target_name='precomputed', pooling_type='mean',
    )
    db.session.add(lm); db.session.commit()
    assert _lb_submission_pred_fields(lb) == []


# ---------------------------------------------------------------------------
# Colab notebook surfaces the pred-field schema in its leading markdown
# ---------------------------------------------------------------------------


def test_static_colab_notebook_lists_pred_fields_for_auto_lb(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    from app import _static_colab_notebook
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    _, _, lb_id = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'colab_pred_lb',
        owner_user_id=logged_in_user.id,
    )
    lb = Leaderboard.query.get(lb_id)
    raw = _static_colab_notebook(lb)
    nb = json.loads(raw)
    intro = ''.join(nb['cells'][0]['source'])
    # The top markdown lists the required submission folder.
    assert 'Required submission folders' in intro
    assert 'label_pred' in intro
    # The model-stub cell uses PRED_FIELDS so the loop writes the
    # right folder names — no longer baked to `metric_<key>`.
    model_cell = next(
        ''.join(c['source']) for c in nb['cells']
        if c.get('cell_type') == 'code' and 'def my_model' in ''.join(c['source'])
    )
    assert 'PRED_FIELDS' in model_cell
    assert 'label_pred' in model_cell


def test_auto_create_lb_uses_existing_global_metric_when_named_match(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    pre = GlobalMetric(
        name='top1_label',
        description='Pre-existing — should be reused.',
        python_code='def top1_label(gt, pred):\n    return 1.0\n',
        is_aggregated=False,
        owner_user_id=logged_in_user.id,
    )
    db.session.add(pre); db.session.commit()
    pre_id = pre.id

    ok, _msg, lb_id = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'lb_reuse_metric',
        owner_user_id=logged_in_user.id,
    )
    assert ok and lb_id
    lms = LeaderboardMetric.query.filter_by(leaderboard_id=lb_id).all()
    assert len(lms) == 1 and lms[0].global_metric_id == pre_id
    assert GlobalMetric.query.filter_by(name='top1_label').count() == 1


def test_auto_create_lb_uses_existing_visualization_when_named_match(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    pre = GlobalVisualization(
        name='confusion_matrix_label',
        description='Pre-existing — should be reused.',
        python_code=(
            "def confusion_matrix_label(gt, pred):\n"
            "    from PIL import Image\n"
            "    return Image.new('L', (8, 8), 0)\n"
        ),
        is_aggregated=True,
        accepts_aggregated_inputs=True,
        owner_user_id=logged_in_user.id,
    )
    db.session.add(pre); db.session.commit()
    pre_id = pre.id

    ok, _msg, lb_id = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'lb_reuse_viz',
        owner_user_id=logged_in_user.id,
    )
    assert ok and lb_id
    lvs = LeaderboardVisualization.query.filter_by(leaderboard_id=lb_id).all()
    assert len(lvs) == 1 and lvs[0].global_visualization_id == pre_id
    assert GlobalVisualization.query.filter_by(name='confusion_matrix_label').count() == 1


def test_auto_create_lb_falls_back_to_static_when_llm_unavailable(
    dataset_numeric_shape, logged_in_user, db_session, monkeypatch,
):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    ok, _msg, lb_id = _auto_create_lb_with_metrics(
        dataset_numeric_shape, 'lb_static_metric',
        owner_user_id=logged_in_user.id,
    )
    assert ok and lb_id
    gm = GlobalMetric.query.filter_by(name='mae_score').first()
    assert gm is not None
    assert 'def mae_score' in gm.python_code
    assert 'abs(' in gm.python_code


def test_auto_create_lb_uses_llm_metric_code_when_api_key_set(
    dataset_classlabel_shape, logged_in_user, db_session, monkeypatch,
):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    fake_metric = (
        "def top1_label(gt, pred):\n"
        "    # LLM-authored\n"
        "    try:\n"
        "        return 1.0 if int(gt) == int(pred) else 0.0\n"
        "    except Exception:\n"
        "        return 0.0\n"
    )
    fake_viz = (
        "def confusion_matrix_label(gt, pred):\n"
        "    from PIL import Image  # LLM-authored\n"
        "    return Image.new('L', (16, 16), 0)\n"
    )

    def _post(url, **kw):
        # First request is for the metric, second for the viz. Discriminate
        # by what the system prompt asks for.
        sys_text = (kw.get('json') or {}).get('system', [{}])[0].get('text', '')
        is_viz = 'visualization' in sys_text.lower()
        text = fake_viz if is_viz else fake_metric

        class _Resp:
            def raise_for_status(self): pass
            def json(self):
                return {'content': [{'type': 'text', 'text': text}]}

        return _Resp()

    with patch('requests.post', side_effect=_post):
        ok, _msg, lb_id = _auto_create_lb_with_metrics(
            dataset_classlabel_shape, 'lb_llm_pair',
            owner_user_id=logged_in_user.id,
        )
    assert ok and lb_id
    gm = GlobalMetric.query.filter_by(name='top1_label').first()
    gv = GlobalVisualization.query.filter_by(name='confusion_matrix_label').first()
    assert gm is not None and '# LLM-authored' in gm.python_code
    assert gv is not None and '# LLM-authored' in gv.python_code


def test_auto_create_lb_refuses_duplicate_name(
    dataset_classlabel_shape, logged_in_user, db_session,
):
    db.session.add(Leaderboard(
        name='dup', summary_metrics='', owner_user_id=logged_in_user.id,
    ))
    db.session.commit()
    ok, msg, _ = _auto_create_lb_with_metrics(
        dataset_classlabel_shape, 'dup', owner_user_id=logged_in_user.id,
    )
    assert not ok
    assert 'already exists' in msg.lower()


def test_auto_create_lb_returns_clear_error_when_no_evaluable_gt(
    db_session, logged_in_user,
):
    """Dataset with only JSON / text GT (nothing the proposer recognizes)
    yields no metrics — surface that instead of creating an empty LB."""
    ds = Dataset(name='json_only_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='meta', field_type='json',
        value_text='/m.json',
    ))
    db.session.commit()
    ok, msg, lb_id = _auto_create_lb_with_metrics(
        ds, 'empty_lb', owner_user_id=logged_in_user.id,
    )
    assert not ok
    assert lb_id is None
    assert 'scalar' in msg.lower() or 'no gt' in msg.lower()


# ---------------------------------------------------------------------------
# /create_leaderboard end-to-end with auto_assign_metrics=1
# ---------------------------------------------------------------------------


def test_create_leaderboard_with_auto_assign_renders_preview(
    auth_client, logged_in_user, db_session, monkeypatch,
    dataset_classlabel_shape,
):
    """auto_assign_metrics now opens the review-and-edit preview page
    instead of creating the LB immediately. The user has to confirm
    via /create_leaderboard/auto_finalize for any rows to land."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    resp = auth_client.post(
        '/create_leaderboard',
        data={
            'leaderboard_name': 'auto_lb_preview',
            'dataset_ids': str(dataset_classlabel_shape.id),
            'auto_assign_metrics': '1',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'Review auto-proposed' in body
    # Both kinds of proposals surfaced.
    assert 'top1_label' in body
    assert 'confusion_matrix_label' in body
    # Submission contract preview is included.
    assert 'label_pred' in body
    # No LB was created yet — the user has to confirm.
    assert Leaderboard.query.filter_by(name='auto_lb_preview').first() is None


def test_auto_finalize_persists_only_kept_proposals(
    auth_client, logged_in_user, db_session, monkeypatch,
    dataset_classlabel_shape,
):
    """User unchecks the visualization → only the metric is attached."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    resp = auth_client.post(
        '/create_leaderboard/auto_finalize',
        data={
            'leaderboard_name': 'kept_metric_only',
            'dataset_id': str(dataset_classlabel_shape.id),
            'kept_metric_top1_label': '1',
            'metric_target_name_top1_label': 'top-1 accuracy (label)',
            'metric_sort_direction_top1_label': 'higher_is_better',
            # Note: kept_viz_confusion_matrix_label intentionally absent.
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert '/leaderboard/' in resp.headers['Location']
    lb = Leaderboard.query.filter_by(name='kept_metric_only').first()
    assert lb is not None
    assert [lm.global_metric.name for lm in lb.leaderboard_metrics] == ['top1_label']
    assert lb.leaderboard_visualizations == []


def test_auto_finalize_honors_user_edits_to_target_name_and_sort(
    auth_client, logged_in_user, db_session, monkeypatch,
    dataset_classlabel_shape,
):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    resp = auth_client.post(
        '/create_leaderboard/auto_finalize',
        data={
            'leaderboard_name': 'edited_lb',
            'dataset_id': str(dataset_classlabel_shape.id),
            'kept_metric_top1_label': '1',
            'metric_target_name_top1_label': 'My renamed top1',
            'metric_sort_direction_top1_label': 'lower_is_better',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    lb = Leaderboard.query.filter_by(name='edited_lb').first()
    lm = lb.leaderboard_metrics[0]
    assert lm.target_name == 'My renamed top1'
    assert lm.sort_direction == 'lower_is_better'


def test_auto_finalize_uses_user_edited_python_code(
    auth_client, logged_in_user, db_session, monkeypatch,
    dataset_classlabel_shape,
):
    """User overrides the proposed python_code → the new GlobalMetric
    persists with the override (only when the function name still
    matches; otherwise the form-supplied code is silently ignored
    and the proposal's default lands)."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    user_code = (
        "def top1_label(gt, pred):\n"
        "    # USER-EDITED\n"
        "    return 1.0 if int(gt) == int(pred) else 0.0\n"
    )
    resp = auth_client.post(
        '/create_leaderboard/auto_finalize',
        data={
            'leaderboard_name': 'edited_code_lb',
            'dataset_id': str(dataset_classlabel_shape.id),
            'kept_metric_top1_label': '1',
            'metric_code_top1_label': user_code,
            'metric_sort_direction_top1_label': 'higher_is_better',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    gm = GlobalMetric.query.filter_by(name='top1_label').first()
    assert gm is not None
    assert '# USER-EDITED' in gm.python_code


def test_auto_finalize_with_nothing_kept_creates_no_lb(
    auth_client, logged_in_user, db_session, monkeypatch,
    dataset_classlabel_shape,
):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    resp = auth_client.post(
        '/create_leaderboard/auto_finalize',
        data={
            'leaderboard_name': 'nothing_kept',
            'dataset_id': str(dataset_classlabel_shape.id),
            # No kept_metric_* / kept_viz_* checkboxes.
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert Leaderboard.query.filter_by(name='nothing_kept').first() is None


def test_create_leaderboard_without_auto_assign_metrics_path_unchanged(
    auth_client, logged_in_user, db_session, dataset_classlabel_shape,
):
    resp = auth_client.post(
        '/create_leaderboard',
        data={
            'leaderboard_name': 'manual_lb',
            'dataset_ids': str(dataset_classlabel_shape.id),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert '/leaderboard/' in resp.headers['Location']
    lb = Leaderboard.query.filter_by(name='manual_lb').first()
    assert lb is not None
    assert lb.leaderboard_metrics == []
    assert lb.leaderboard_visualizations == []


def test_create_leaderboard_auto_assign_requires_a_dataset(
    auth_client, logged_in_user, db_session,
):
    resp = auth_client.post(
        '/create_leaderboard',
        data={
            'leaderboard_name': 'no_ds',
            'auto_assign_metrics': '1',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert Leaderboard.query.filter_by(name='no_ds').first() is None
