"""Paired-dataset support via `leaderboard_datasets.role`.

A leaderboard can pair an 'input' dataset with one or more 'gt_source'
datasets. The metric engine folds gt_source CustomFields into the
context for each primary sample whose name matches a gt_source sample.

This solves the dirty-docs `-train` (noisy) / `-cleaned` (GT) split
across two HF repos without merging the upstream repos.
"""
import pytest

from app import (
    CustomField, Dataset, Leaderboard, Sample, db,
    leaderboard_datasets,
    _lb_dataset_role, _set_lb_dataset_role, _gt_source_datasets_for_lb,
    _make_paired_gt_provider,
)


# ---------------------------------------------------------------------------
# Fixtures: a denoising-shape LB pairing an input dataset + a GT dataset.
# Sample names match across the two datasets.
# ---------------------------------------------------------------------------


@pytest.fixture
def paired_lb(db_session):
    """LB with two datasets: noisy (primary) + clean (gt_source).
    Both have a sample s00000."""
    noisy = Dataset(name='dirty_docs_noisy', visibility='public')
    clean = Dataset(name='dirty_docs_clean', visibility='public')
    db.session.add_all([noisy, clean]); db.session.flush()

    s_noisy = Sample(dataset_id=noisy.id, name='s00000')
    s_clean = Sample(dataset_id=clean.id, name='s00000')
    db.session.add_all([s_noisy, s_clean]); db.session.flush()

    # Noisy side has the input image (would be shipped by submission
    # for evaluation; the metric needs the GT side too).
    db.session.add(CustomField(
        sample_id=s_noisy.id, name='image_input', data_type='image',
        value_text='inputs/s00000.png',
    ))
    # Clean side has the GT image and a per-sample text caption.
    db.session.add(CustomField(
        sample_id=s_clean.id, name='clean', data_type='image',
        value_text='clean/s00000.png',
    ))
    db.session.add(CustomField(
        sample_id=s_clean.id, name='caption', data_type='text',
        value_text='document #1',
    ))
    db.session.add(CustomField(
        sample_id=s_clean.id, name='quality_score', data_type='scalar',
        value_float=0.95,
    ))

    lb = Leaderboard(name='dirty_docs_lb', summary_metrics='', visibility='public')
    lb.datasets = [noisy, clean]
    db.session.add(lb); db.session.commit()

    # Tag the second attachment as the GT source.
    _set_lb_dataset_role(lb.id, clean.id, 'gt_source')

    return {
        'lb': lb,
        'noisy': noisy, 'clean': clean,
        'noisy_sample': s_noisy, 'clean_sample': s_clean,
    }


# ---------------------------------------------------------------------------
# Role helpers — read / write / lookup.
# ---------------------------------------------------------------------------


def test_default_role_is_primary(db_session):
    lb = Leaderboard(name='solo_lb', summary_metrics='', visibility='public')
    ds = Dataset(name='solo_ds', visibility='public')
    db.session.add_all([lb, ds]); db.session.flush()
    lb.datasets.append(ds)
    db.session.commit()
    # Newly attached → 'primary' by default (server_default).
    assert _lb_dataset_role(lb.id, ds.id) == 'primary'


def test_set_role_persists(db_session):
    lb = Leaderboard(name='r_lb', summary_metrics='', visibility='public')
    ds = Dataset(name='r_ds', visibility='public')
    db.session.add_all([lb, ds]); db.session.flush()
    lb.datasets.append(ds); db.session.commit()
    changed = _set_lb_dataset_role(lb.id, ds.id, 'gt_source')
    assert changed is True
    assert _lb_dataset_role(lb.id, ds.id) == 'gt_source'
    # Idempotent: calling again with same role returns False.
    assert _set_lb_dataset_role(lb.id, ds.id, 'gt_source') is False


def test_set_role_rejects_unknown_value(db_session):
    lb = Leaderboard(name='bad_role', summary_metrics='', visibility='public')
    ds = Dataset(name='bad_role_ds', visibility='public')
    db.session.add_all([lb, ds]); db.session.flush()
    lb.datasets.append(ds); db.session.commit()
    with pytest.raises(ValueError, match='unknown role'):
        _set_lb_dataset_role(lb.id, ds.id, 'auxiliary')


def test_lb_role_returns_none_for_unattached_pair(db_session):
    lb = Leaderboard(name='unpaired_lb', summary_metrics='', visibility='public')
    ds = Dataset(name='unpaired_ds', visibility='public')
    db.session.add_all([lb, ds]); db.session.commit()
    assert _lb_dataset_role(lb.id, ds.id) is None


def test_gt_source_datasets_lists_only_gt_role(paired_lb):
    out = _gt_source_datasets_for_lb(paired_lb['lb'])
    assert [d.name for d in out] == ['dirty_docs_clean']


# ---------------------------------------------------------------------------
# _make_paired_gt_provider — yields fields from gt_source samples
# whose names match the primary sample.
# ---------------------------------------------------------------------------


def test_paired_gt_provider_yields_gt_fields_for_matching_sample(paired_lb):
    provider = _make_paired_gt_provider(paired_lb['lb'])
    assert provider is not None
    yielded = list(provider(paired_lb['noisy_sample']))
    field_names = {cf.name for cf, _ in yielded}
    assert 'clean' in field_names         # image GT
    assert 'caption' in field_names       # text
    assert 'quality_score' in field_names  # scalar


def test_paired_gt_provider_silent_for_non_matching_sample(paired_lb, db_session):
    """Sample whose name doesn't appear in the gt_source dataset
    should yield nothing (rather than raise)."""
    other = Sample(dataset_id=paired_lb['noisy'].id, name='no_partner')
    db.session.add(other); db.session.commit()
    provider = _make_paired_gt_provider(paired_lb['lb'])
    assert list(provider(other)) == []


def test_paired_gt_provider_returns_none_when_no_gt_source(db_session):
    """LB with only primary datasets → no provider needed."""
    lb = Leaderboard(name='primary_only', summary_metrics='', visibility='public')
    ds = Dataset(name='primary_only_ds', visibility='public')
    db.session.add_all([lb, ds]); db.session.flush()
    lb.datasets.append(ds); db.session.commit()
    assert _make_paired_gt_provider(lb) is None


# ---------------------------------------------------------------------------
# get_metric_context end-to-end with a paired_gt_provider wired in.
# ---------------------------------------------------------------------------


def test_get_metric_context_folds_in_paired_scalar_and_text(paired_lb):
    """Pin: a metric context built for the noisy primary sample
    surfaces the clean dataset's scalar + text fields under
    `gt_<name>` AND bare `<name>`."""
    from metric_engine import get_metric_context
    provider = _make_paired_gt_provider(paired_lb['lb'])
    ctx = get_metric_context(
        paired_lb['noisy_sample'], paired_gt_provider=provider,
    )
    assert ctx['gt_quality_score'] == 0.95
    assert ctx['quality_score'] == 0.95
    assert ctx['gt_caption'] == 'document #1'
    assert ctx['caption'] == 'document #1'
    # Image field comes through as None (file path doesn't exist on
    # disk in this unit test) — the field key still appears so a
    # metric can detect "missing GT" via context.get().
    assert 'gt_clean' in ctx
    assert ctx['gt_clean'] is None


def test_get_metric_context_without_provider_skips_paired_fields(paired_lb):
    """Sanity check: when get_metric_context isn't given the
    provider, the gt_source dataset's fields don't leak into the
    context. Backwards-compat for callers that don't yet pass the
    new arg."""
    from metric_engine import get_metric_context
    ctx = get_metric_context(paired_lb['noisy_sample'])
    assert 'gt_quality_score' not in ctx
    assert 'gt_caption' not in ctx
    assert 'gt_clean' not in ctx


# ---------------------------------------------------------------------------
# /leaderboard/<id>/edit POST — form-based role updates persist.
# ---------------------------------------------------------------------------


def test_edit_leaderboard_form_updates_dataset_role(
    auth_client, logged_in_user, db_session,
):
    """Posting `dataset_role_<id>=gt_source` flips the attachment."""
    noisy = Dataset(name='ui_noisy', visibility='public', owner_user_id=logged_in_user.id)
    clean = Dataset(name='ui_clean', visibility='public', owner_user_id=logged_in_user.id)
    db.session.add_all([noisy, clean]); db.session.flush()
    lb = Leaderboard(name='ui_paired_lb', summary_metrics='',
                     visibility='public', owner_user_id=logged_in_user.id)
    lb.datasets = [noisy, clean]
    db.session.add(lb); db.session.commit()

    resp = auth_client.post(
        f'/leaderboard/{lb.id}/edit',
        data={
            'name': lb.name,
            f'dataset_role_{noisy.id}': 'primary',
            f'dataset_role_{clean.id}': 'gt_source',
        },
        follow_redirects=False,
    )
    # Form-submit redirects on success.
    assert resp.status_code in (200, 302)
    assert _lb_dataset_role(lb.id, noisy.id) == 'primary'
    assert _lb_dataset_role(lb.id, clean.id) == 'gt_source'


def test_edit_leaderboard_renders_role_dropdown(
    auth_client, logged_in_user, db_session, paired_lb,
):
    resp = auth_client.get(f'/leaderboard/{paired_lb["lb"].id}/edit')
    assert resp.status_code == 200
    body = resp.data.decode()
    # Role dropdown markup shows up for each attached dataset.
    assert f'dataset_role_{paired_lb["noisy"].id}' in body
    assert f'dataset_role_{paired_lb["clean"].id}' in body
    # The pre-selected gt_source for the clean dataset is reflected
    # in the rendered HTML (the option carries `selected`).
    assert 'gt_source' in body
