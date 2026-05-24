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
