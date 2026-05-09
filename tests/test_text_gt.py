"""Phase 11: text-typed GT columns (e.g. sentiment 'neg'/'pos') get
auto-proposed an exact-match metric. Pred submissions for text fields
ship as `.txt` and survive the engine's float-parse fallback as raw
strings so exact_match can consume them."""
import json

from app import (
    Dataset, Sample, CustomField, db,
    _gt_columns, _propose_metrics_for_dataset,
    _propose_visualizations_for_dataset,
    _virtual_sample_from_hf_row,
    _proposal_top1_text_classlabel,
    _proposal_macro_f1_text_classlabel,
    _viz_text_confusion_matrix,
)


# ---------------------------------------------------------------------------
# _gt_columns yields text kind for short text fields
# ---------------------------------------------------------------------------


def _seed_ds_with_field(field_type, value_text=None, value_float=None,
                        name='label'):
    ds = Dataset(name=f'tg_{field_type}_{name}', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name=name,
        field_type=field_type,
        value_text=value_text, value_float=value_float,
    ))
    db.session.commit()
    return ds


def test_gt_columns_yields_text_for_short_text(client, db_session):
    ds = _seed_ds_with_field('text', value_text='neg')
    cols = list(_gt_columns(ds))
    assert ('label', 'text', {}) in cols


def test_gt_columns_skips_long_text_captions(client, db_session):
    ds = _seed_ds_with_field('text', value_text='x' * 200, name='caption')
    cols = list(_gt_columns(ds))
    # Long text → not yielded (treated as free-form, no exact-match
    # metric proposed).
    assert all(c[0] != 'caption' for c in cols)


# ---------------------------------------------------------------------------
# _propose_metrics_for_dataset dispatches text → exact_match
# ---------------------------------------------------------------------------


def test_proposes_top1_and_macro_f1_for_text_gt(client, db_session):
    """Text classes get full classification treatment: top-1 accuracy
    (per-sample) + macro F1 (aggregated)."""
    ds = _seed_ds_with_field('text', value_text='pos', name='sentiment')
    proposals = _propose_metrics_for_dataset(ds)
    by_name = {p['global_name']: p for p in proposals}
    assert 'top1_text' in by_name
    assert 'macro_f1_text' in by_name

    top1 = by_name['top1_text']
    assert top1['arg_mappings'] == {'gt': 'gt_sentiment',
                                     'pred': 'sub_sentiment_pred'}
    assert top1['sort_direction'] == 'higher_is_better'
    assert 'def top1_text(gt, pred)' in top1['fallback_code']
    assert 'top-1 accuracy' in top1['target_name']

    macro = by_name['macro_f1_text']
    assert macro['is_aggregated'] is True
    assert macro['accepts_aggregated_inputs'] is True
    assert 'def macro_f1_text(gt, pred)' in macro['fallback_code']


def test_proposes_text_confusion_matrix_viz(client, db_session):
    """Text classes also get a confusion-matrix viz, mirroring the
    int-ClassLabel auto-proposal."""
    ds = _seed_ds_with_field('text', value_text='cat', name='species')
    vizes = _propose_visualizations_for_dataset(ds)
    names = [v['global_name'] for v in vizes]
    assert 'confusion_matrix_text' in names
    cm = next(v for v in vizes if v['global_name'] == 'confusion_matrix_text')
    assert cm['is_aggregated'] is True
    assert cm['arg_mappings'] == {'gt': 'gt_species',
                                   'pred': 'sub_species_pred'}


def test_macro_f1_handles_imbalanced_two_class():
    """Quick sanity check on the generated F1 code: 80% of class A
    predicted correctly + 0/2 class B predicted should give F1 well
    below the accuracy."""
    code = _proposal_macro_f1_text_classlabel('mood')['fallback_code']
    ns = {}
    exec(code, ns)
    fn = ns['macro_f1_text']
    gt   = ['pos', 'pos', 'pos', 'pos', 'neg', 'neg']
    pred = ['pos', 'pos', 'pos', 'pos', 'pos', 'pos']
    f1 = fn(gt, pred)
    # pos: tp=4 fp=2 fn=0 → prec=4/6, rec=1, F1 = 2*0.667/1.667 = 0.8
    # neg: tp=0 → F1=0 (short-circuit)
    # Macro = (0.8 + 0) / 2 = 0.4
    assert abs(f1 - 0.4) < 1e-6
    # Sanity: accuracy is 4/6 ≈ 0.667 — strictly higher than F1 here,
    # which is the whole point of also surfacing macro F1.
    assert f1 < 4 / 6


def test_top1_text_classlabel_proposal_shape():
    p = _proposal_top1_text_classlabel('mood')
    assert p['global_name'] == 'top1_text'
    assert p['pred_fields'][0]['kind'] == 'scalar'
    assert p['pred_fields'][0]['name'] == 'mood_pred'
    assert p['pooling_type'] == 'mean'


# ---------------------------------------------------------------------------
# Lenient match (case-insensitive + punctuation-stripped + prefix-aware)
# ---------------------------------------------------------------------------


def _build_top1():
    code = _proposal_top1_text_classlabel('mood')['fallback_code']
    ns = {}
    exec(code, ns)
    return ns['top1_text']


def test_top1_text_case_insensitive():
    fn = _build_top1()
    assert fn('pos', 'Pos') == 1.0
    assert fn('pos', 'POS') == 1.0


def test_top1_text_strips_punctuation_and_whitespace():
    fn = _build_top1()
    assert fn('pos', '  pos!  ') == 1.0
    assert fn('pos', 'pos.') == 1.0


def test_top1_text_prefix_aware_pos_vs_positive():
    fn = _build_top1()
    # Common case: GT says 'pos', model emits 'positive' or vice versa.
    assert fn('pos', 'positive') == 1.0
    assert fn('positive', 'pos') == 1.0


def test_top1_text_token_match_for_phrasing():
    fn = _build_top1()
    # Prediction wraps the class label in a sentence — token match.
    assert fn('cat', 'this is a cat') == 1.0


def test_top1_text_distinct_classes_dont_collide():
    fn = _build_top1()
    # 'neg' vs 'pos' don't share a prefix or token, so we don't accidentally
    # call them equal.
    assert fn('pos', 'neg') == 0.0
    assert fn('positive', 'negative') == 0.0


def test_top1_text_empty_or_none():
    fn = _build_top1()
    assert fn('', 'pos') == 0.0
    assert fn('pos', '') == 0.0


def test_macro_f1_text_collapses_lenient_predictions():
    """Macro F1 canonicalizes pred 'positive' / 'POS!' / '  positive  '
    onto class 'pos' so a model that just uses different surface forms
    isn't penalized."""
    code = _proposal_macro_f1_text_classlabel('mood')['fallback_code']
    ns = {}; exec(code, ns)
    fn = ns['macro_f1_text']
    gt   = ['pos', 'pos', 'neg', 'neg']
    pred = ['positive', 'POS!', 'negative', 'NEGATIVE']
    f1 = fn(gt, pred)
    assert abs(f1 - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _virtual_sample_from_hf_row: scalar-mapped string falls back to text
# ---------------------------------------------------------------------------


def test_virtual_sample_scalar_string_fallback_to_text():
    """User maps 'label' as scalar but the HF column emits strings
    (e.g. 'neg'/'pos'). We previously dropped these silently — now we
    store them as text so the proposer's text dispatch picks them up."""
    att = type('A', (), {
        'hf_repo_id': 'fake/repo', 'hf_revision': None, 'hf_split': 'train',
        'hf_mapping_json': json.dumps([
            {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
        ]),
    })()
    vs = _virtual_sample_from_hf_row(att, {'label': 'neg'}, 0, {})
    by_name = {cf.name: cf for cf in vs.custom_fields}
    assert 'label' in by_name
    cf = by_name['label']
    assert cf.field_type == 'text'
    assert cf.value_text == 'neg'


def test_virtual_sample_classlabel_path_unaffected():
    """ClassLabel path still wins for int-encoded labels with names."""
    att = type('A', (), {
        'hf_repo_id': 'fake/repo', 'hf_revision': None, 'hf_split': 'train',
        'hf_mapping_json': json.dumps([
            {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
        ]),
    })()
    vs = _virtual_sample_from_hf_row(att, {'label': 1}, 0, {'label': ['neg', 'pos']})
    by_name = {cf.name: cf for cf in vs.custom_fields}
    # Scalar with float index + a _class sidecar — same as before.
    assert by_name['label'].field_type == 'scalar'
    assert by_name['label'].value_float == 1.0
    assert by_name['label_class'].value_text == 'pos'


# ---------------------------------------------------------------------------
# _load_sub_pred_for_sample returns a string when float() fails
# ---------------------------------------------------------------------------


def test_load_sub_pred_returns_string_for_non_numeric_txt(tmp_path):
    from metric_engine import _load_sub_pred_for_sample
    folder = tmp_path / 'sub'
    pred_dir = folder / 'sentiment_pred'
    pred_dir.mkdir(parents=True)
    (pred_dir / 's0.txt').write_text('pos\n')
    val = _load_sub_pred_for_sample(str(folder), 'sentiment_pred', 's0')
    assert val == 'pos'


def test_load_sub_pred_still_returns_float_for_numeric_txt(tmp_path):
    from metric_engine import _load_sub_pred_for_sample
    folder = tmp_path / 'sub'
    pred_dir = folder / 'score_pred'
    pred_dir.mkdir(parents=True)
    (pred_dir / 's0.txt').write_text('0.85\n')
    val = _load_sub_pred_for_sample(str(folder), 'score_pred', 's0')
    assert val == 0.85
