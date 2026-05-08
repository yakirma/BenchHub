"""Phase 11: text-typed GT columns (e.g. sentiment 'neg'/'pos') get
auto-proposed an exact-match metric. Pred submissions for text fields
ship as `.txt` and survive the engine's float-parse fallback as raw
strings so exact_match can consume them."""
import json

from app import (
    Dataset, Sample, CustomField, db,
    _gt_columns, _propose_metrics_for_dataset,
    _virtual_sample_from_hf_row, _proposal_exact_match_text,
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


def test_proposes_exact_match_for_text_gt(client, db_session):
    ds = _seed_ds_with_field('text', value_text='pos', name='sentiment')
    proposals = _propose_metrics_for_dataset(ds)
    by_name = {p['global_name']: p for p in proposals}
    assert 'exact_match' in by_name
    p = by_name['exact_match']
    assert p['arg_mappings'] == {'gt': 'gt_sentiment',
                                  'pred': 'sub_sentiment_pred'}
    assert p['sort_direction'] == 'higher_is_better'
    assert 'def exact_match(gt, pred)' in p['fallback_code']


def test_exact_match_proposal_shape():
    """Sanity-check the proposal dict shape so it slots into the same
    pipeline the other proposals use."""
    p = _proposal_exact_match_text('mood')
    assert p['global_name'] == 'exact_match'
    assert p['pred_fields'][0]['kind'] == 'scalar'
    assert p['pred_fields'][0]['name'] == 'mood_pred'
    assert p['pooling_type'] == 'mean'


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
