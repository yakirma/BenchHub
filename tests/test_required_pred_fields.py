"""Phase 9: pred fields can be required by an LB even without a metric."""
import json

from app import (
    Dataset, Leaderboard, Sample, db,
    _lb_submission_pred_fields, _parse_extra_pred_fields,
    _parse_auto_pred_fields, _apply_pred_field_edits,
    _merge_pred_field_extras,
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


def test_pred_field_kind_inferred_from_hf_mapping_for_hf_lb(client, db_session):
    """HF-attached LBs have no BH Dataset rows. _lb_submission_pred_fields
    must read the GT field kinds off the primary HF Attachment's
    hf_mapping_json — otherwise every pred field defaulted to 'scalar'
    (user-reported: NYU `raw_depth_map_pred` showed Type=scalar instead
    of depth)."""
    import json as _json
    from app import (
        Attachment, GlobalMetric, LeaderboardMetric,
        _lb_submission_pred_fields,
    )
    lb = Leaderboard(name='nyu_kind_lb', summary_metrics='',
                     visibility='public')
    db.session.add(lb); db.session.flush()
    db.session.add(Attachment(
        leaderboard_id=lb.id, hf_repo_id='sayakpaul/nyu_depth_v2',
        hf_split='train', role='primary',
        hf_mapping_json=_json.dumps([
            {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
            {'column': 'depth_map', 'target_kind': 'depth', 'target_field': 'raw_depth_map'},
        ]),
    ))
    gm = GlobalMetric(name='rms', python_code='def rms(gt, pred): return 0.0')
    db.session.add(gm); db.session.flush()
    db.session.add(LeaderboardMetric(
        leaderboard_id=lb.id, global_metric_id=gm.id,
        target_name='RMS',
        arg_mappings=_json.dumps({'gt': 'gt_raw_depth_map', 'pred': 'sub_raw_depth_map_pred'}),
        sort_direction='lower_is_better',
    ))
    db.session.commit()

    fields = _lb_submission_pred_fields(lb)
    by_name = {f['name']: f for f in fields}
    assert 'raw_depth_map_pred' in by_name
    # Depth GT in the HF mapping → pred field kind is depth, not scalar.
    assert by_name['raw_depth_map_pred']['kind'] == 'depth'
    # Description reflects the depth contract (.npz, depth map).
    assert 'depth' in by_name['raw_depth_map_pred']['description'].lower()


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


# ---------------------------------------------------------------------------
# Auto-derived pred fields: rename / kind override / omit
# ---------------------------------------------------------------------------


def test_parse_auto_pred_fields_validates_and_normalizes():
    raw = json.dumps([
        {'original_name': 'labels_pred', 'name': 'class_pred', 'kind': 'scalar'},
        {'original_name': 'depth_pred', 'name': '!!bad!!', 'kind': 'depth'},
        {'original_name': 'mask_pred', 'name': '', 'kind': 'mask'},
        {'original_name': '', 'name': 'orphan', 'kind': 'scalar'},  # dropped
        {'original_name': 'bogus_pred', 'name': 'ok_pred', 'kind': 'nonsense'},
        {'original_name': 'gone_pred', 'name': 'gone_pred', 'kind': 'scalar',
         'omit': True},
    ])
    out = _parse_auto_pred_fields(raw)
    by_orig = {e['original_name']: e for e in out}
    assert 'labels_pred' in by_orig and by_orig['labels_pred']['name'] == 'class_pred'
    # invalid identifier falls back to original_name (no silent rename).
    assert by_orig['depth_pred']['name'] == 'depth_pred'
    # blank name falls back to original_name.
    assert by_orig['mask_pred']['name'] == 'mask_pred'
    # missing original_name is rejected (the field has no anchor to apply edits to).
    assert 'orphan' not in {e.get('name') for e in out}
    # bogus kind falls back to scalar.
    assert by_orig['bogus_pred']['kind'] == 'scalar'
    # omit flag preserved.
    assert by_orig['gone_pred']['omit'] is True


def test_apply_pred_field_edits_renames_arg_mappings():
    proposals = [{
        'global_name': 'top1', 'target_name': 'top-1',
        'arg_mappings': {'gt': 'gt_labels', 'pred': 'sub_labels_pred'},
        'pred_fields': [{'name': 'labels_pred', 'kind': 'scalar'}],
    }]
    out = _apply_pred_field_edits(
        proposals,
        omitted=set(),
        renames={'labels_pred': 'class_idx_pred'},
    )
    assert len(out) == 1
    assert out[0]['arg_mappings']['pred'] == 'sub_class_idx_pred'
    assert out[0]['arg_mappings']['gt'] == 'gt_labels'  # GT side untouched
    assert out[0]['pred_fields'][0]['name'] == 'class_idx_pred'


def test_apply_pred_field_edits_omits_metric_that_depends_on_omitted_field():
    proposals = [
        {
            'global_name': 'top1',
            'arg_mappings': {'gt': 'gt_labels', 'pred': 'sub_labels_pred'},
            'pred_fields': [{'name': 'labels_pred', 'kind': 'scalar'}],
        },
        {
            'global_name': 'psnr',
            'arg_mappings': {'gt': 'gt_image', 'pred': 'sub_image_pred'},
            'pred_fields': [{'name': 'image_pred', 'kind': 'image'}],
        },
    ]
    out = _apply_pred_field_edits(
        proposals,
        omitted={'labels_pred'},
        renames={},
    )
    # top1 dropped (its pred is omitted); psnr survives.
    assert [p['global_name'] for p in out] == ['psnr']


def test_apply_pred_field_edits_leaves_proposal_dict_immutable():
    """The route mutates kept_metrics in place between proposal-collection
    and LB creation. Make sure _apply_pred_field_edits doesn't trash the
    caller's input — return copies."""
    original = {
        'global_name': 'top1',
        'arg_mappings': {'gt': 'gt_labels', 'pred': 'sub_labels_pred'},
        'pred_fields': [{'name': 'labels_pred', 'kind': 'scalar'}],
    }
    out = _apply_pred_field_edits(
        [original], omitted=set(), renames={'labels_pred': 'cls_pred'},
    )
    assert original['arg_mappings']['pred'] == 'sub_labels_pred'
    assert out[0]['arg_mappings']['pred'] == 'sub_cls_pred'


def test_merge_pred_field_extras_combines_kind_overrides_with_extras():
    extras = [{'name': 'caption_pred', 'kind': 'scalar', 'description': 'free text'}]
    overrides = [
        {'name': 'labels_pred', 'kind': 'depth'},
        {'name': 'caption_pred', 'kind': 'image'},  # override on an extras entry
    ]
    out = _merge_pred_field_extras(extras, overrides)
    by_name = {e['name']: e for e in out}
    # New override entry created for the metric-derived field.
    assert by_name['labels_pred']['kind'] == 'depth'
    # Override on a pre-existing extras row wins on kind, keeps description.
    assert by_name['caption_pred']['kind'] == 'image'
    assert by_name['caption_pred'].get('description') == 'free text'


def test_lb_submission_pred_fields_applies_kind_override_from_extras(
    client, db_session,
):
    """When required_pred_fields_json names a field that's ALSO derived
    from a metric, the extras entry's kind/description override the
    derived defaults — but `used_by` keeps pointing at the metric."""
    from app import GlobalMetric, LeaderboardMetric
    ds = Dataset(name='ovr_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.flush()
    gm = GlobalMetric(name='dm', python_code='def dm(gt, pred): return 0.0')
    db.session.add(gm); db.session.flush()
    lb = Leaderboard(
        name='ovr_lb', summary_metrics='', visibility='public',
        required_pred_fields_json=json.dumps([
            {'name': 'class_pred', 'kind': 'depth',
             'description': 'special-cased depth-encoded class'},
        ]),
    )
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    db.session.add(LeaderboardMetric(
        leaderboard_id=lb.id, global_metric_id=gm.id,
        target_name='top-1',
        arg_mappings=json.dumps({'gt': 'gt_class', 'pred': 'sub_class_pred'}),
        sort_direction='higher_is_better',
    ))
    db.session.commit()

    fields = _lb_submission_pred_fields(lb)
    by_name = {f['name']: f for f in fields}
    assert 'class_pred' in by_name
    # Kind from extras wins.
    assert by_name['class_pred']['kind'] == 'depth'
    # Description from extras wins.
    assert 'special-cased' in by_name['class_pred']['description']
    # used_by still reflects the metric.
    assert by_name['class_pred']['used_by'] == ['top-1']


def test_auto_finalize_renames_pred_field_in_metric_arg_mappings(
    auth_client, logged_in_user, db_session,
):
    """End-to-end: posting auto_pred_fields_json with a rename
    rewrites the resulting LeaderboardMetric.arg_mappings on creation."""
    from app import LeaderboardMetric
    ds = Dataset(name='rn_ds', visibility='public',
                 owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.commit()

    extra_metric = json.dumps([{
        'global_name': 'top1',
        'target_name': 'top-1',
        'description': '',
        'python_code': 'def top1(gt, pred): return 1.0 if gt == pred else 0.0',
        'arg_mappings': {'gt': 'gt_labels', 'pred': 'sub_labels_pred'},
        'sort_direction': 'higher_is_better',
        'code_source': 'llm',
        'pred_fields': [{'name': 'labels_pred', 'kind': 'scalar',
                         'gt_field': 'labels'}],
    }])
    auto_pred = json.dumps([
        {'original_name': 'labels_pred', 'name': 'class_idx_pred',
         'kind': 'scalar'},
    ])

    r = auth_client.post('/create_leaderboard/auto_finalize', data={
        'leaderboard_name': 'rn_lb',
        'dataset_id': str(ds.id),
        'extra_metrics_json': extra_metric,
        'auto_pred_fields_json': auto_pred,
    }, follow_redirects=False)
    assert r.status_code in (302, 303)
    lb = Leaderboard.query.filter_by(name='rn_lb').first()
    assert lb is not None
    lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id).first()
    am = json.loads(lm.arg_mappings or '{}')
    assert am['pred'] == 'sub_class_idx_pred'  # renamed
    assert am['gt'] == 'gt_labels'  # GT untouched
    # The submission contract now reports the renamed folder name.
    fields = _lb_submission_pred_fields(lb)
    names = {f['name'] for f in fields}
    assert 'class_idx_pred' in names
    assert 'labels_pred' not in names


def test_auto_finalize_omits_pred_field_drops_metric_too(
    auth_client, logged_in_user, db_session,
):
    """Omitting a metric-derived pred field also drops the metric that
    depends on it — the metric can't score without its prediction."""
    from app import LeaderboardMetric
    ds = Dataset(name='om_ds', visibility='public',
                 owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.commit()

    extra_metric = json.dumps([
        {
            'global_name': 'top1',
            'target_name': 'top-1',
            'description': '',
            'python_code': 'def top1(gt, pred): return 1.0',
            'arg_mappings': {'gt': 'gt_labels', 'pred': 'sub_labels_pred'},
            'sort_direction': 'higher_is_better',
            'code_source': 'llm',
            'pred_fields': [{'name': 'labels_pred', 'kind': 'scalar'}],
        },
        {
            'global_name': 'mae',
            'target_name': 'mae',
            'description': '',
            'python_code': 'def mae(gt, pred): return 0.0',
            'arg_mappings': {'gt': 'gt_value', 'pred': 'sub_value_pred'},
            'sort_direction': 'lower_is_better',
            'code_source': 'llm',
            'pred_fields': [{'name': 'value_pred', 'kind': 'scalar'}],
        },
    ])
    auto_pred = json.dumps([
        {'original_name': 'labels_pred', 'name': 'labels_pred',
         'kind': 'scalar', 'omit': True},
        {'original_name': 'value_pred', 'name': 'value_pred',
         'kind': 'scalar'},
    ])

    r = auth_client.post('/create_leaderboard/auto_finalize', data={
        'leaderboard_name': 'om_lb',
        'dataset_id': str(ds.id),
        'extra_metrics_json': extra_metric,
        'auto_pred_fields_json': auto_pred,
    }, follow_redirects=False)
    assert r.status_code in (302, 303)
    lb = Leaderboard.query.filter_by(name='om_lb').first()
    assert lb is not None
    metric_names = {lm.target_name for lm in lb.leaderboard_metrics}
    # top-1 dropped (its pred was omitted); mae survives.
    assert metric_names == {'mae'}


def test_auto_finalize_kind_override_persists_to_required_pred_fields(
    auth_client, logged_in_user, db_session,
):
    """Editing the kind of a metric-derived pred field writes an
    override entry into required_pred_fields_json so the submission
    contract + downloaded notebook reflect the new serialization."""
    ds = Dataset(name='ko_ds', visibility='public',
                 owner_user_id=logged_in_user.id)
    db.session.add(ds); db.session.flush()
    db.session.add(Sample(dataset_id=ds.id, name='s0'))
    db.session.commit()

    extra_metric = json.dumps([{
        'global_name': 'top1',
        'target_name': 'top-1',
        'description': '',
        'python_code': 'def top1(gt, pred): return 1.0',
        'arg_mappings': {'gt': 'gt_labels', 'pred': 'sub_labels_pred'},
        'sort_direction': 'higher_is_better',
        'code_source': 'llm',
        'pred_fields': [{'name': 'labels_pred', 'kind': 'scalar'}],
    }])
    auto_pred = json.dumps([
        {'original_name': 'labels_pred', 'name': 'labels_pred',
         'kind': 'depth'},  # override
    ])

    r = auth_client.post('/create_leaderboard/auto_finalize', data={
        'leaderboard_name': 'ko_lb',
        'dataset_id': str(ds.id),
        'extra_metrics_json': extra_metric,
        'auto_pred_fields_json': auto_pred,
    }, follow_redirects=False)
    assert r.status_code in (302, 303)
    lb = Leaderboard.query.filter_by(name='ko_lb').first()
    assert lb is not None
    saved = json.loads(lb.required_pred_fields_json or '[]')
    by_name = {e['name']: e for e in saved}
    assert by_name['labels_pred']['kind'] == 'depth'
    fields = _lb_submission_pred_fields(lb)
    by_name = {f['name']: f for f in fields}
    assert by_name['labels_pred']['kind'] == 'depth'
