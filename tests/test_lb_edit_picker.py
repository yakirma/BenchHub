"""Coverage for the contract-filtered metric picker on the LB
settings (edit) page.

Mirrors `_build_lb_creation_context` for an existing leaderboard:
the picker enumerates GlobalMetrics, marks each one satisfiable
when the LB's effective contract (attached-dataset fields + role
overrides + required pred fields) can supply every declared
arg, and emits a `missing_reason` otherwise. Unsatisfiable
metrics still appear so the picker can render them disabled with
a hint about why.
"""
import json

from app import (
    Dataset, DatasetField, GlobalMetric, Leaderboard, User,
    _build_lb_picker_context, db,
)


def _mk_label_metric(name='label_acc', python='''import benchhub as bh
def label_acc(gt: bh.Label, pred: bh.Label):
    return float(gt.value == pred.value)
'''):
    gm = GlobalMetric(
        name=name,
        description='',
        python_code=python,
        is_aggregated=False,
        input_kinds=json.dumps(['label', 'label']),
        input_roles=json.dumps(['gt', 'pred']),
        visibility='public',
    )
    db.session.add(gm); db.session.flush()
    return gm


def _attach_dataset(lb, ds):
    lb.datasets.append(ds)
    db.session.flush()


def _add_field(ds, name, kind, role='gt'):
    df = DatasetField(dataset_id=ds.id, name=name, kind=kind, role=role)
    db.session.add(df); db.session.flush()
    return df


def test_picker_marks_metric_satisfiable_when_lb_has_required_fields(db_session):
    """A label/label metric with arg_specs (gt: label, pred: label)
    is satisfiable when the LB has a gt-label field AND a pred-label
    field in its contract."""
    ds = Dataset(name='ds_a'); db.session.add(ds); db.session.flush()
    _add_field(ds, 'label', kind='label', role='gt')

    lb = Leaderboard(
        name='lb_a',
        summary_metrics='',
        required_pred_fields_json=json.dumps([
            {'name': 'label_pred', 'kind': 'label'},
        ]),
    )
    db.session.add(lb); db.session.flush()
    _attach_dataset(lb, ds)
    _mk_label_metric()
    db.session.commit()

    metrics, gt_opts, field_opts = _build_lb_picker_context(lb)
    m = next(x for x in metrics if x['name'] == 'label_acc')
    assert m['satisfiable'] is True, m
    assert m['missing_reason'] == ''
    assert 'label' in gt_opts


def test_picker_marks_metric_unsatisfiable_with_missing_reason(db_session):
    """No `pred:label` field in the contract → metric still appears
    in the picker, but flagged unsatisfiable with a reason that names
    which arg is missing what."""
    ds = Dataset(name='ds_b'); db.session.add(ds); db.session.flush()
    _add_field(ds, 'label', kind='label', role='gt')

    lb = Leaderboard(name='lb_b', summary_metrics='')
    db.session.add(lb); db.session.flush()
    _attach_dataset(lb, ds)
    _mk_label_metric()
    db.session.commit()

    metrics, _, _ = _build_lb_picker_context(lb)
    m = next(x for x in metrics if x['name'] == 'label_acc')
    assert m['satisfiable'] is False
    assert 'pred' in m['missing_reason'].lower()


def test_picker_honours_per_lb_role_overrides(db_session):
    """A dataset field declared role=input but overridden to role=gt
    on THIS LB via `field_roles_json` must count as gt for
    satisfiability checks."""
    ds = Dataset(name='ds_c'); db.session.add(ds); db.session.flush()
    # Declared as input on the dataset; the LB swaps it to gt.
    _add_field(ds, 'label', kind='label', role='input')

    lb = Leaderboard(
        name='lb_c',
        summary_metrics='',
        field_roles_json=json.dumps({'label': 'gt'}),
        required_pred_fields_json=json.dumps([
            {'name': 'label_pred', 'kind': 'label'},
        ]),
    )
    db.session.add(lb); db.session.flush()
    _attach_dataset(lb, ds)
    _mk_label_metric()
    db.session.commit()

    metrics, gt_opts, _ = _build_lb_picker_context(lb)
    m = next(x for x in metrics if x['name'] == 'label_acc')
    assert m['satisfiable'] is True, m
    assert 'label' in gt_opts


def test_edit_lb_page_renders_picker_options_with_satisfiable_flag(
    client, db_session,
):
    """The settings page renders the picker with `data-satisfiable`
    on every option + disabled rows for unsatisfiable metrics."""
    owner = User(email='lbowner@bench.local', display_name='o',
                 oauth_provider='github', oauth_sub='lb-o-1')
    db.session.add(owner); db.session.flush()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id

    ds = Dataset(name='ds_view'); db.session.add(ds); db.session.flush()
    _add_field(ds, 'label', kind='label', role='gt')

    lb = Leaderboard(
        name='lb_view',
        summary_metrics='',
        owner_user_id=owner.id,
        required_pred_fields_json=json.dumps([
            {'name': 'label_pred', 'kind': 'label'},
        ]),
    )
    db.session.add(lb); db.session.flush()
    _attach_dataset(lb, ds)
    _mk_label_metric(name='label_acc_ok')
    # A second metric the LB CAN'T satisfy — needs a kind=depth GT
    # field which this LB doesn't have.
    _mk_label_metric(
        name='depth_metric_bad',
        python='''import benchhub as bh
def depth_metric_bad(gt: bh.Depth, pred: bh.Depth):
    return 0.0
''',
    )
    GlobalMetric.query.filter_by(name='depth_metric_bad').update({
        'input_kinds': json.dumps(['depth', 'depth']),
        'input_roles': json.dumps(['gt', 'pred']),
    })
    db.session.commit()

    body = client.get(f'/leaderboard/{lb.id}/edit').data.decode('utf-8')
    # Every option carries the satisfiable flag for the JS picker.
    assert 'data-satisfiable="1"' in body
    assert 'data-satisfiable="0"' in body
    # Unsatisfiable rows are visually disabled in the dropdown.
    assert 'depth_metric_bad' in body
    assert 'disabled' in body
    # Auto-mapping wiring (same UX as the creation form): options carry
    # arg specs, and the page ships the field-options data + the
    # auto-suggest helper that maps each arg to a matching field.
    assert 'data-arg-specs=' in body
    assert 'autoSuggestMapping' in body
    assert 'lbFieldOptions' in body
    # The pred field is exposed to the client so a role=pred arg can
    # auto-bind to it (regression: pred args used to default to a blank
    # gt row, leaving accuracy's `pred` unmapped → 0.0 scores).
    assert 'label_pred' in body


# ---------------------------------------------------------------------------
# Pred-field edit gating: admin / LB owner / attached-dataset owner
# ---------------------------------------------------------------------------


def test_pred_edit_allowed_for_attached_dataset_owner(client, db_session):
    """Owner of any attached Dataset must be allowed to edit the
    LB's pred-field schema, even if they don't own the LB itself.
    The dataset author writes the canonical contract."""
    ds_owner = User(email='ds@bench.local', display_name='ds_owner',
                    oauth_provider='github', oauth_sub='ds-1')
    lb_owner = User(email='lb@bench.local', display_name='lb_owner',
                    oauth_provider='github', oauth_sub='lb-1')
    db.session.add_all([ds_owner, lb_owner]); db.session.flush()

    ds = Dataset(name='gated_ds', owner_user_id=ds_owner.id)
    db.session.add(ds); db.session.flush()

    lb = Leaderboard(name='gated_lb', summary_metrics='',
                     owner_user_id=lb_owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = ds_owner.id
    r = client.post(f'/leaderboard/{lb.id}/pred_fields', data={
        'name_0': 'thing_pred', 'kind_0': 'scalar',
    })
    assert r.status_code == 302, r.data[:200]


def test_pred_edit_forbidden_for_unrelated_user(client, db_session):
    """A logged-in user who's neither admin, nor LB owner, nor an
    attached-dataset owner gets 403."""
    owner = User(email='owns@bench.local', display_name='o',
                 oauth_provider='github', oauth_sub='own-1')
    stranger = User(email='stranger@bench.local', display_name='s',
                    oauth_provider='github', oauth_sub='str-1')
    db.session.add_all([owner, stranger]); db.session.flush()

    ds = Dataset(name='locked_ds', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()

    lb = Leaderboard(name='locked_lb', summary_metrics='',
                     owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = stranger.id
    r = client.post(f'/leaderboard/{lb.id}/pred_fields', data={
        'name_0': 'thing_pred', 'kind_0': 'scalar',
    })
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Dataset-fields tab — per-LB role overrides
# ---------------------------------------------------------------------------


def test_dataset_field_roles_tab_renders(client, db_session):
    """The settings page exposes a "Dataset fields" tab with one row
    per attached-dataset field; each row carries the field's kind +
    a role selector."""
    owner = User(email='df_owner@bench.local', display_name='o',
                 oauth_provider='github', oauth_sub='df-1')
    db.session.add(owner); db.session.flush()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id

    ds = Dataset(name='df_ds', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    _add_field(ds, 'rgb', kind='image', role='input')
    _add_field(ds, 'depth', kind='depth', role='gt')

    lb = Leaderboard(name='df_lb', summary_metrics='', owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()

    body = client.get(f'/leaderboard/{lb.id}/edit').data.decode('utf-8')
    assert 'id="dsf-tab"' in body
    assert 'Dataset fields' in body
    # The role selector exists per field, name keyed by field name.
    assert 'name="field_role_rgb"' in body
    assert 'name="field_role_depth"' in body


def test_dataset_field_roles_post_writes_field_roles_json(client, db_session):
    """POSTing the role table writes the per-LB overrides to
    `Leaderboard.field_roles_json` so the picker + runtime context
    builder both see the swap."""
    owner = User(email='df_owner2@bench.local', display_name='o2',
                 oauth_provider='github', oauth_sub='df-2')
    db.session.add(owner); db.session.flush()
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id

    ds = Dataset(name='df_ds2', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    _add_field(ds, 'rgb', kind='image', role='gt')   # default gt
    _add_field(ds, 'depth', kind='depth', role='gt')

    lb = Leaderboard(name='df_lb2', summary_metrics='', owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()

    r = client.post(f'/leaderboard/{lb.id}/dataset_field_roles', data={
        'field_role_rgb':   'input',
        'field_role_depth': 'gt',
        # Unknown field name must be silently ignored (defence-in-depth).
        'field_role_imaginary': 'gt',
    })
    assert r.status_code == 302
    db.session.refresh(lb)
    stored = json.loads(lb.field_roles_json or '{}')
    assert stored == {'rgb': 'input', 'depth': 'gt'}


# ---------------------------------------------------------------------------
# Lifecycle policy: add-anytime, remove-only-when-empty
# ---------------------------------------------------------------------------


def _mk_lb_with_subs(db_session, owner, pred_fields=None, with_submission=True):
    """Helper: returns an LB owned by `owner`, attached to one dataset,
    optionally seeded with a verified Submission so `has_verified_subs`
    is true."""
    from app import Submission
    ds = Dataset(name=f'life_ds_{owner.id}', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    _add_field(ds, 'label', kind='label', role='gt')

    lb = Leaderboard(
        name=f'life_lb_{owner.id}',
        summary_metrics='',
        owner_user_id=owner.id,
        required_pred_fields_json=(json.dumps(pred_fields) if pred_fields else None),
    )
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    if with_submission:
        sub = Submission(leaderboard_id=lb.id, name='s1',
                         processing_status='Processed')
        db.session.add(sub)
    db.session.commit()
    return lb


def test_pred_add_allowed_when_lb_has_submissions(client, db_session):
    """Adding a NEW pred field is allowed even with verified
    submissions — it only widens the contract for future ones."""
    owner = User(email='add_ok@bench.local', display_name='a',
                 oauth_provider='github', oauth_sub='add-1')
    db.session.add(owner); db.session.flush()
    lb = _mk_lb_with_subs(
        db_session, owner,
        pred_fields=[{'name': 'label_pred', 'kind': 'label',
                      'gt_field': 'label', 'description': ''}],
    )
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(f'/leaderboard/{lb.id}/pred_fields', data={
        # Preserve the existing row …
        'name_0': 'label_pred', 'kind_0': 'label', 'description_0': '',
        # … and add a brand-new one.
        'name_1': 'confidence_pred', 'kind_1': 'scalar', 'description_1': '',
    })
    assert r.status_code == 302
    db.session.refresh(lb)
    schema = json.loads(lb.required_pred_fields_json)
    names = {e['name'] for e in schema}
    assert {'label_pred', 'confidence_pred'} <= names


def test_pred_remove_blocked_when_lb_has_submissions(client, db_session):
    """Removing an existing pred field with verified subs present
    must fail — the LB keeps its old schema."""
    owner = User(email='rem_no@bench.local', display_name='r',
                 oauth_provider='github', oauth_sub='rem-1')
    db.session.add(owner); db.session.flush()
    lb = _mk_lb_with_subs(
        db_session, owner,
        pred_fields=[
            {'name': 'label_pred', 'kind': 'label',
             'gt_field': 'label', 'description': ''},
            {'name': 'extra_pred', 'kind': 'scalar',
             'gt_field': 'extra', 'description': ''},
        ],
    )
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(f'/leaderboard/{lb.id}/pred_fields', data={
        # Drop `extra_pred`.
        'name_0': 'label_pred', 'kind_0': 'label', 'description_0': '',
    })
    assert r.status_code == 302
    db.session.refresh(lb)
    names = {e['name'] for e in json.loads(lb.required_pred_fields_json)}
    assert names == {'label_pred', 'extra_pred'}, (
        "the remove attempt must NOT take effect when subs exist"
    )


def test_pred_kind_change_blocked_when_lb_has_submissions(client, db_session):
    """Changing the kind of an existing field with subs present must
    fail — the LB keeps the old kind."""
    owner = User(email='kind_no@bench.local', display_name='k',
                 oauth_provider='github', oauth_sub='kind-1')
    db.session.add(owner); db.session.flush()
    lb = _mk_lb_with_subs(
        db_session, owner,
        pred_fields=[{'name': 'label_pred', 'kind': 'label',
                      'gt_field': 'label', 'description': ''}],
    )
    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(f'/leaderboard/{lb.id}/pred_fields', data={
        'name_0': 'label_pred', 'kind_0': 'scalar', 'description_0': '',
    })
    assert r.status_code == 302
    db.session.refresh(lb)
    schema = json.loads(lb.required_pred_fields_json)
    assert schema[0]['kind'] == 'label'


def test_metric_remove_blocked_when_lb_has_submissions(client, db_session):
    """Removing a LeaderboardMetric drops its column + every MetricResult
    that powered it — only legal on an empty LB."""
    from app import LeaderboardMetric, Submission
    owner = User(email='mrem@bench.local', display_name='m',
                 oauth_provider='github', oauth_sub='mrem-1')
    db.session.add(owner); db.session.flush()
    ds = Dataset(name='mrem_ds', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    _add_field(ds, 'label', kind='label', role='gt')

    lb = Leaderboard(name='mrem_lb', summary_metrics='',
                     owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    gm = _mk_label_metric(name='metric_to_keep')
    lm = LeaderboardMetric(leaderboard_id=lb.id, global_metric_id=gm.id,
                           target_name='metric_to_keep',
                           arg_mappings='{}', pooling_type='mean',
                           sort_direction='higher_is_better')
    db.session.add(lm); db.session.flush()
    db.session.add(Submission(leaderboard_id=lb.id, name='s1',
                              processing_status='Processed'))
    db.session.commit()
    lm_id = lm.id

    with client.session_transaction() as sess:
        sess['user_id'] = owner.id
    r = client.post(
        f'/leaderboard/{lb.id}/leaderboard_metric/{lm_id}/delete',
    )
    assert r.status_code == 302
    # The metric is still there — the policy blocked the removal.
    assert LeaderboardMetric.query.get(lm_id) is not None


def test_dataset_field_roles_post_forbidden_for_unrelated_user(client, db_session):
    """Same gate as pred-fields: random users get 403."""
    owner = User(email='df_o3@bench.local', display_name='o3',
                 oauth_provider='github', oauth_sub='df-3')
    stranger = User(email='df_str@bench.local', display_name='s',
                    oauth_provider='github', oauth_sub='df-str-1')
    db.session.add_all([owner, stranger]); db.session.flush()

    ds = Dataset(name='df_ds3', owner_user_id=owner.id)
    db.session.add(ds); db.session.flush()
    _add_field(ds, 'rgb', kind='image', role='gt')

    lb = Leaderboard(name='df_lb3', summary_metrics='', owner_user_id=owner.id)
    db.session.add(lb); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = stranger.id
    r = client.post(f'/leaderboard/{lb.id}/dataset_field_roles', data={
        'field_role_rgb': 'input',
    })
    assert r.status_code == 403
