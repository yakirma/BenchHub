"""`_iter_lb_eval_samples` must restrict to the materialised subset when
the LB has one — otherwise a submission (which only predicts the subset)
is scored over the full source dataset and the unpredicted samples drag
the metric toward zero.

Regression: a 95.4%-accurate cifar10 submission reported 9.5% because
the eval iterated all 10k dataset rows while the LB materialised 1k.
"""
from __future__ import annotations

import os

from app import (
    Dataset,
    Leaderboard,
    LeaderboardMaterialization,
    Sample,
    _iter_lb_eval_samples,
    app as flask_app,
    db,
)
from benchhub.lb_materialize import materialization_dir


def _seed(n_total, materialized_names):
    ds = Dataset(name='scope_ds', visibility='public')
    db.session.add(ds); db.session.flush()
    for i in range(n_total):
        db.session.add(Sample(dataset_id=ds.id, name=f's{i:06d}'))
    lb = Leaderboard(name='scope_lb', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    if materialized_names is not None:
        db.session.add(LeaderboardMaterialization(
            leaderboard_id=lb.id, status='ready',
            sample_cap=len(materialized_names), sampling='random',
            sampling_seed=42,
        ))
        # list_materialized_samples reads the on-disk subset.
        field_dir = materialization_dir(flask_app.config['UPLOAD_FOLDER'], lb.id) / 'img'
        os.makedirs(field_dir, exist_ok=True)
        for name in materialized_names:
            (field_dir / f'{name}.jpg').write_bytes(b'x')
    db.session.commit()
    return lb


def test_eval_scope_restricted_to_materialized_subset(client, db_session):
    # 100 dataset samples, LB materialises 10 of them.
    subset = [f's{i:06d}' for i in range(10)]
    lb = _seed(100, subset)
    names = sorted(s.name for s, _ in _iter_lb_eval_samples(lb))
    assert names == sorted(subset)
    assert len(names) == 10  # NOT 100


def test_eval_scope_full_dataset_when_no_materialization(client, db_session):
    lb = _seed(25, None)  # no materialization → eval over everything
    names = [s.name for s, _ in _iter_lb_eval_samples(lb)]
    assert len(names) == 25


def test_eval_scope_full_dataset_when_materialization_not_ready(client, db_session):
    """A pending/failed materialisation must NOT restrict the scope
    (the subset isn't on disk yet)."""
    ds = Dataset(name='scope_ds2', visibility='public')
    db.session.add(ds); db.session.flush()
    for i in range(8):
        db.session.add(Sample(dataset_id=ds.id, name=f's{i:06d}'))
    lb = Leaderboard(name='scope_lb2', visibility='public')
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    db.session.add(LeaderboardMaterialization(
        leaderboard_id=lb.id, status='pending',
        sample_cap=4, sampling='random', sampling_seed=42,
    ))
    db.session.commit()
    names = [s.name for s, _ in _iter_lb_eval_samples(lb)]
    assert len(names) == 8  # pending → no restriction
