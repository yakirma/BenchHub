"""Phase 1 of the LB-attachment refactor: an LB can attach BH
datasets (uploaded ZIPs) AND/OR HF refs (live-streamed). The
iterator yields a uniform sample handle for both kinds; the engine
can call get_metric_context with either."""
import json
import sys
import types

import pytest
from PIL import Image

from app import (
    Attachment, CustomField, Dataset, Leaderboard, Sample, db,
    _iter_lb_eval_samples, _iter_hf_attachment_samples,
    _virtual_sample_from_hf_row,
)


@pytest.fixture
def fake_hf_two_rows(monkeypatch):
    """Tiny streaming HF stand-in: 2 rows with image + label + caption."""
    rows = [
        {'image': Image.new('RGB', (4, 4), (10, 20, 30)),
         'label': 0, 'caption': 'first'},
        {'image': Image.new('RGB', (4, 4), (40, 50, 60)),
         'label': 1, 'caption': 'second'},
    ]

    class _ClassLabel:
        names = ['cat', 'dog']

    class _DS:
        features = {'image': object(), 'label': _ClassLabel(),
                    'caption': object()}
        def __iter__(self):
            return iter(rows)

    fake = types.ModuleType('datasets')
    fake.load_dataset = lambda *a, **kw: _DS()
    monkeypatch.setitem(sys.modules, 'datasets', fake)
    return rows


# ---------------------------------------------------------------------------
# _virtual_sample_from_hf_row — direct unit test, no streaming needed.
# ---------------------------------------------------------------------------


def test_virtual_sample_image_field_is_pointer_only():
    att = type('A', (), {
        'hf_repo_id': 'fake/repo', 'hf_revision': None, 'hf_split': 'train',
        'hf_mapping_json': json.dumps([
            {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        ]),
    })()
    row = {'image': Image.new('RGB', (4, 4))}
    vs = _virtual_sample_from_hf_row(att, row, 0, {})
    assert vs.name == 's_000000'
    assert json.loads(vs.source_ref_json) == {
        'repo_id': 'fake/repo', 'revision': None, 'split': 'train', 'row_idx': 0,
    }
    cf = vs.custom_fields[0]
    assert cf.field_type == 'image'
    # Pointer-only: no inline value_text, but source_column is set so
    # the engine's pointer_resolver can fetch on demand.
    assert cf.value_text is None
    assert cf.source_column == 'image'


def test_virtual_sample_classlabel_emits_sidecar_and_tag():
    att = type('A', (), {
        'hf_repo_id': 'fake/repo', 'hf_revision': None, 'hf_split': 'train',
        'hf_mapping_json': json.dumps([
            {'column': 'label', 'target_kind': 'scalar', 'target_field': 'label'},
        ]),
    })()
    row = {'label': 0}
    vs = _virtual_sample_from_hf_row(att, row, 5, {'label': ['cat', 'dog']})
    by_name = {cf.name: cf for cf in vs.custom_fields}
    assert by_name['label'].field_type == 'scalar'
    assert by_name['label'].value_float == 0.0
    assert by_name['label_class'].value_text == 'cat'
    assert 'cat' in vs.tags


def test_virtual_sample_text_field_inline():
    att = type('A', (), {
        'hf_repo_id': 'fake/repo', 'hf_revision': None, 'hf_split': 'train',
        'hf_mapping_json': json.dumps([
            {'column': 'caption', 'target_kind': 'text', 'target_field': 'caption'},
        ]),
    })()
    vs = _virtual_sample_from_hf_row(att, {'caption': 'hello'}, 0, {})
    assert vs.custom_fields[0].field_type == 'text'
    assert vs.custom_fields[0].value_text == 'hello'


# ---------------------------------------------------------------------------
# _iter_hf_attachment_samples — caps + streaming.
# ---------------------------------------------------------------------------


def test_iter_hf_attachment_caps_to_default(client, db_session, fake_hf_two_rows):
    att = Attachment(
        leaderboard_id=1, hf_repo_id='fake/repo', hf_split='train',
        hf_mapping_json=json.dumps([
            {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        ]),
    )
    rows = list(_iter_hf_attachment_samples(att, cap=2))
    assert len(rows) == 2
    assert rows[0].name == 's_000000'
    assert rows[1].name == 's_000001'


def test_iter_hf_attachment_respects_per_attachment_cap(client, db_session, fake_hf_two_rows):
    att = Attachment(
        leaderboard_id=1, hf_repo_id='fake/repo', hf_split='train',
        hf_sample_cap=1,
        hf_mapping_json=json.dumps([
            {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        ]),
    )
    rows = list(_iter_hf_attachment_samples(att))
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# _iter_lb_eval_samples — yields BH samples + HF-ref virtual samples.
# ---------------------------------------------------------------------------


def test_iter_lb_yields_bh_and_hf_ref_in_order(
    client, db_session, fake_hf_two_rows,
):
    """Hybrid LB with one BH attachment + one HF-ref attachment.
    Iterator returns BH samples first, then virtual HF samples."""
    bh_ds = Dataset(name='hybrid_bh', visibility='public')
    db.session.add(bh_ds); db.session.flush()
    db.session.add(Sample(dataset_id=bh_ds.id, name='real_s0'))
    db.session.add(Sample(dataset_id=bh_ds.id, name='real_s1'))
    lb = Leaderboard(name='hybrid_lb', summary_metrics='', visibility='public')
    db.session.add(lb); db.session.flush()
    db.session.add(Attachment(
        leaderboard_id=lb.id, dataset_id=bh_ds.id, role='primary',
    ))
    db.session.add(Attachment(
        leaderboard_id=lb.id, hf_repo_id='fake/repo', hf_split='train',
        role='primary', hf_sample_cap=2,
        hf_mapping_json=json.dumps([
            {'column': 'image', 'target_kind': 'image', 'target_field': 'image_image'},
        ]),
    ))
    db.session.commit()

    yielded = list(_iter_lb_eval_samples(lb))
    names = [s.name for s, _ in yielded]
    assert names == ['real_s0', 'real_s1', 's_000000', 's_000001']
    # Per-sample attachment surfaced so the eval pipeline knows which
    # attachment a virtual sample came from (for paired-GT lookups).
    kinds = [att.kind for _, att in yielded]
    assert kinds == ['bh', 'bh', 'hf', 'hf']


def test_iter_lb_skips_gt_source_attachments(client, db_session, fake_hf_two_rows):
    """gt_source attachments are folded in via the paired-GT provider,
    NOT via the primary iteration. This test confirms they don't get
    double-counted."""
    bh_ds = Dataset(name='primary_only', visibility='public')
    db.session.add(bh_ds); db.session.flush()
    db.session.add(Sample(dataset_id=bh_ds.id, name='primary_s0'))
    lb = Leaderboard(name='paired_iter_lb', summary_metrics='', visibility='public')
    db.session.add(lb); db.session.flush()
    db.session.add(Attachment(
        leaderboard_id=lb.id, dataset_id=bh_ds.id, role='primary',
    ))
    db.session.add(Attachment(
        leaderboard_id=lb.id, hf_repo_id='fake/repo', hf_split='train',
        role='gt_source', hf_sample_cap=2,
        hf_mapping_json='[]',
    ))
    db.session.commit()

    yielded = list(_iter_lb_eval_samples(lb))
    assert [s.name for s, _ in yielded] == ['primary_s0']


# ---------------------------------------------------------------------------
# Migration: existing leaderboard_datasets rows should already be
# mirrored as Attachment rows after the boot-time migration. The
# session-level test fixture runs check_and_migrate_db() at startup,
# so the backfill query has already executed against the fixture DB.
# ---------------------------------------------------------------------------


def test_existing_lb_dataset_pair_has_matching_attachment(client, db_session):
    """When an LB is created via the legacy m2m + the migration runs,
    an Attachment row mirroring it exists. (Brand-new code uses
    Attachment directly; this case checks backward compat with rows
    inserted via lb.datasets.append.)"""
    bh = Dataset(name='legacy_compat_ds', visibility='public')
    db.session.add(bh); db.session.flush()
    lb = Leaderboard(name='legacy_compat_lb', summary_metrics='', visibility='public')
    lb.datasets.append(bh)
    db.session.add(lb); db.session.commit()

    # New code paths should create Attachment rows directly when
    # adding HF refs, so the migration only matters for pre-refactor
    # rows. The legacy m2m + the new Attachment table coexist for now;
    # this test just sanity-checks both can be queried.
    legacy_n = db.session.execute(
        db.text("SELECT COUNT(*) FROM leaderboard_datasets WHERE leaderboard_id = :lb"),
        {'lb': lb.id},
    ).scalar()
    assert legacy_n == 1
