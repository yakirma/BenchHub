"""Phase 9: pred fields can be required by an LB even without a metric."""
import json

from app import (
    Dataset, Leaderboard, Sample, db,
    _lb_submission_pred_fields, _parse_extra_pred_fields,
)


# ---------------------------------------------------------------------------
# _parse_extra_pred_fields — defensive normalization
# ---------------------------------------------------------------------------


def test_parser_rejects_bad_json():
    assert _parse_extra_pred_fields("not-json") == []
    assert _parse_extra_pred_fields("{}") == []  # not a list


def test_parser_skips_empty_names():
    raw = json.dumps([{'name': '', 'kind': 'scalar'}, {'name': 'good_pred'}])
    out = _parse_extra_pred_fields(raw)
    assert len(out) == 1
    assert out[0]['name'] == 'good_pred'


def test_parser_normalizes_kind_and_gt_field():
    raw = json.dumps([
        {'name': 'caption_pred', 'kind': 'bogus'},
        {'name': 'mask_pred', 'kind': 'mask'},
    ])
    out = _parse_extra_pred_fields(raw)
    by_name = {p['name']: p for p in out}
    # bogus kind falls back to scalar
    assert by_name['caption_pred']['kind'] == 'scalar'
    # gt_field strips the trailing _pred
    assert by_name['caption_pred']['gt_field'] == 'caption'
    assert by_name['mask_pred']['kind'] == 'mask'


def test_parser_dedupes_by_name():
    raw = json.dumps([
        {'name': 'dup', 'kind': 'scalar'},
        {'name': 'dup', 'kind': 'depth'},
    ])
    out = _parse_extra_pred_fields(raw)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# _lb_submission_pred_fields merges required fields with metric-derived
# ---------------------------------------------------------------------------


def _seed_lb_with_required(required, lb_name='req_lb'):
    ds = Dataset(name=f'{lb_name}_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    lb = Leaderboard(
        name=lb_name, summary_metrics='', visibility='public',
        required_pred_fields_json=json.dumps(required),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()
    return lb


def test_required_pred_fields_appear_on_submission_contract(client, db_session):
    lb = _seed_lb_with_required([{
        'name': 'caption_pred',
        'gt_field': 'caption',
        'kind': 'scalar',
        'description': 'Free-form caption per sample.',
    }])
    fields = _lb_submission_pred_fields(lb)
    by_name = {f['name']: f for f in fields}
    assert 'caption_pred' in by_name
    entry = by_name['caption_pred']
    assert entry['kind'] == 'scalar'
    assert entry['used_by'] == ['(no metric — required by LB)']
    assert 'caption' in entry['description'].lower()


def test_required_pred_fields_dont_override_metric_derived(client, db_session):
    """If a metric already declares the same pred field, the metric's
    used_by + description win — required-only doesn't shadow it."""
    from app import GlobalMetric, LeaderboardMetric
    ds = Dataset(name='clash_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()

    gm = GlobalMetric(
        name='dummy_metric',
        python_code='def dummy_metric(gt, pred): return 0.0',
    )
    db.session.add(gm); db.session.flush()
    lb = Leaderboard(
        name='clash_lb', summary_metrics='', visibility='public',
        required_pred_fields_json=json.dumps([
            {'name': 'overlap_pred', 'kind': 'depth', 'description': 'should-not-win'},
        ]),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    db.session.add(LeaderboardMetric(
        leaderboard_id=lb.id, global_metric_id=gm.id,
        target_name='dummy',
        arg_mappings=json.dumps({'gt': 'gt_overlap', 'pred': 'sub_overlap_pred'}),
        sort_direction='lower_is_better',
    ))
    db.session.commit()

    fields = _lb_submission_pred_fields(lb)
    by_name = {f['name']: f for f in fields}
    assert 'overlap_pred' in by_name
    # used_by reflects the metric, not the "(no metric — required by LB)" sentinel.
    assert by_name['overlap_pred']['used_by'] == ['dummy']


def test_required_pred_fields_handle_bad_json_gracefully(client, db_session):
    lb = _seed_lb_with_required([])
    lb.required_pred_fields_json = '{not-json'
    db.session.commit()
    # Should not raise.
    fields = _lb_submission_pred_fields(lb)
    assert isinstance(fields, list)


# ---------------------------------------------------------------------------
# auto_finalize persists required pred fields onto the new LB
# ---------------------------------------------------------------------------


def test_auto_finalize_persists_required_pred_fields(
    auth_client, logged_in_user, db_session,
):
    ds = Dataset(name='reqf_ds', visibility='public',
                 owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.commit()

    extras = json.dumps([
        {'name': 'caption_pred', 'kind': 'scalar',
         'description': 'free-form caption'},
        {'name': 'thumb_pred', 'kind': 'image'},
    ])
    metric_extra = json.dumps([{
        'global_name': 'unused_metric',
        'target_name': 'unused',
        'description': '',
        'python_code': 'def unused_metric(gt, pred):\n    return 0.0',
        'arg_mappings': {},
        'sort_direction': 'higher_is_better',
        'code_source': 'llm',
    }])

    r = auth_client.post('/create_leaderboard/auto_finalize', data={
        'leaderboard_name': 'reqf_lb',
        'dataset_id': str(ds.id),
        'extra_metrics_json': metric_extra,
        'extra_pred_fields_json': extras,
    }, follow_redirects=False)
    assert r.status_code in (302, 303)

    lb = Leaderboard.query.filter_by(name='reqf_lb').first()
    assert lb is not None
    saved = json.loads(lb.required_pred_fields_json or '[]')
    names = {row['name'] for row in saved}
    assert names == {'caption_pred', 'thumb_pred'}

    fields = _lb_submission_pred_fields(lb)
    by_name = {f['name']: f for f in fields}
    assert 'caption_pred' in by_name
    assert 'thumb_pred' in by_name
    assert by_name['thumb_pred']['kind'] == 'image'
