"""End-to-end smoke for the Phase B typed pipeline.

Stitches together every layer in one test: the typed-manifest importer
seeds a synthetic image-classification dataset; the reference
`accuracy` metric is registered against the LB with input_kinds=
['label', 'label']; the client constructs typed predictions and
submits them through the Flask test client; the Celery
process_submission task runs eagerly; the resulting MetricResult is
asserted equal to the analytical accuracy.

If this test passes, a real submitter using `benchhub-client` against
the live site can score predictions on a typed LB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import benchhub as bh
from app import (
    CustomField,
    Dataset,
    GlobalMetric,
    Leaderboard,
    LeaderboardMetric,
    MetricResult,
    Sample,
    User,
    db,
    generate_api_token,
)
from benchhub.manifest import import_typed_dataset


N_SAMPLES = 8
NUM_CLASSES = 4


def _gen_synth_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)


def _gt_label_for_sample(i: int) -> int:
    return i % NUM_CLASSES


def _write_synth_dataset(root: Path) -> None:
    """Lay out a synthetic image-classification dataset on disk in the
    typed manifest format."""
    manifest = {
        "name": "synth_imgcls",
        "version": "1.0",
        "fields": [
            {"name": "image", "kind": "image", "role": "input", "params": {}},
            {"name": "label", "kind": "label", "role": "gt", "params": {}},
        ],
        "samples": [f"s{i}" for i in range(N_SAMPLES)],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "image").mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    for i in range(N_SAMPLES):
        bh.Image(_gen_synth_image(i)).encode  # validate shape
        (root / "image" / f"s{i}.png").write_bytes(
            bh.Image(_gen_synth_image(i)).encode()
        )
        (root / "label" / f"s{i}.txt").write_bytes(
            bh.Label(_gt_label_for_sample(i)).encode()
        )


@pytest.fixture
def seeded_lb_with_accuracy_metric(db_session, tmp_path, monkeypatch):
    """Seed a typed dataset + an LB + the typed `accuracy` reference
    metric, returned ready for submissions."""
    # 1. Materialise the typed dataset (this is what /admin/import_typed_dataset does).
    src = tmp_path / "src"
    src.mkdir()
    _write_synth_dataset(src)
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    ds_id, _ = import_typed_dataset(
        src,
        db_session=db.session,
        Dataset=Dataset, Sample=Sample, CustomField=CustomField,
        upload_folder=str(uploads),
    )
    # Redirect the running app at this test's uploads dir so the
    # eval pipeline (which reads files via app.config['UPLOAD_FOLDER'])
    # finds the synthetic data we just wrote. monkeypatch.setitem
    # restores the previous value at teardown so the smoke test below
    # us doesn't see a stale path.
    from app import app as flask_app
    monkeypatch.setitem(flask_app.config, "UPLOAD_FOLDER", str(uploads))

    # 2. Create a Leaderboard that declares a `label_pred` (kind=label) contract.
    lb = Leaderboard(
        name="synth_imgcls_lb",
        summary_metrics="accuracy",
        visibility="public",
        required_pred_fields_json=json.dumps([
            {"name": "label_pred", "kind": "label", "params": {}, "role": "pred"},
        ]),
    )
    lb.datasets.append(Dataset.query.get(ds_id))
    db.session.add(lb)
    db.session.flush()

    # 3. Register the `accuracy` GlobalMetric with input_kinds=['label','label']
    #    so the metric engine hands it bh.Label instances.
    gm = GlobalMetric(
        name="accuracy",
        python_code=(
            "def accuracy(gt, pred):\n"
            "    if gt is None or pred is None:\n"
            "        return 0.0\n"
            "    return 1.0 if gt.value == pred.value else 0.0\n"
        ),
        input_kinds=json.dumps(["label", "label"]),
        is_aggregated=False,
        visibility="public",
    )
    db.session.add(gm)
    db.session.flush()

    # 4. Bind it to the LB. arg_mappings tells the engine which context
    #    keys feed each kwarg — gt → the dataset's `label` field,
    #    pred → the submission's `label_pred` field.
    lm = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        target_name="accuracy",
        arg_mappings=json.dumps({"gt": "gt_label", "pred": "sub_label_pred"}),
        sort_direction="higher_is_better",
        pooling_type="mean",
    )
    db.session.add(lm)
    db.session.commit()
    return lb, lm


@pytest.fixture
def api_user(db_session):
    u = User(
        email="phaseb@bench.local", display_name="phaseb",
        oauth_provider="github", oauth_sub="phaseb-1",
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _submit_via_client(client, api_user, lb, predict_fn):
    """Drive a submission through the public client + Flask transport."""
    bh_client = bh.Client(
        token=api_user.api_token,
        base_url="http://test",
        transport=bh.FlaskTestClientTransport(client),
    )
    sub = bh_client.submission(lb.id, name="under-test")
    for i in range(N_SAMPLES):
        sub.predict(f"s{i}", label_pred=bh.Label(predict_fn(i)))
    return sub.submit()


def test_perfect_model_scores_one(client, seeded_lb_with_accuracy_metric, api_user):
    """A model that returns the GT label for every sample should score 1.0."""
    lb, lm = seeded_lb_with_accuracy_metric
    payload = _submit_via_client(
        client, api_user, lb,
        predict_fn=_gt_label_for_sample,  # perfect oracle
    )
    sub_id = payload["submission_id"]
    mr = MetricResult.query.filter_by(submission_id=sub_id, leaderboard_metric_id=lm.id).one()
    assert mr.value == pytest.approx(1.0)


def test_always_wrong_model_scores_zero(client, seeded_lb_with_accuracy_metric, api_user):
    """A model that always picks an obviously-wrong label scores 0."""
    lb, lm = seeded_lb_with_accuracy_metric

    def always_wrong(i):
        # Pick something that can never collide with the GT.
        return (_gt_label_for_sample(i) + 1) % NUM_CLASSES

    payload = _submit_via_client(client, api_user, lb, predict_fn=always_wrong)
    mr = MetricResult.query.filter_by(
        submission_id=payload["submission_id"],
        leaderboard_metric_id=lm.id,
    ).one()
    assert mr.value == pytest.approx(0.0)


def test_half_right_model_scores_half(client, seeded_lb_with_accuracy_metric, api_user):
    """Half of the samples predicted correctly → score ≈ 0.5."""
    lb, lm = seeded_lb_with_accuracy_metric

    def half_right(i):
        # Even indices: correct. Odd: shifted by 1.
        if i % 2 == 0:
            return _gt_label_for_sample(i)
        return (_gt_label_for_sample(i) + 1) % NUM_CLASSES

    payload = _submit_via_client(client, api_user, lb, predict_fn=half_right)
    mr = MetricResult.query.filter_by(
        submission_id=payload["submission_id"],
        leaderboard_metric_id=lm.id,
    ).one()
    assert mr.value == pytest.approx(0.5)


def test_metric_added_after_submission_shows_not_computed(
    client, seeded_lb_with_accuracy_metric, api_user,
):
    """Binding a metric whose required pred field the submission never
    produced must record NULL + a 'Not computed' message — NOT a
    misleading 0.0. (The submission has label_pred; the new metric
    needs label_topk_pred.)"""
    import tasks
    lb, _ = seeded_lb_with_accuracy_metric
    payload = _submit_via_client(client, api_user, lb, predict_fn=_gt_label_for_sample)
    sub_id = payload["submission_id"]

    # New metric requiring a pred field the submission lacks.
    gm = GlobalMetric(
        name="top5_needs_topk",
        python_code=(
            "def top5_needs_topk(gt, pred):\n"
            "    if gt is None or pred is None:\n"
            "        return 0.0\n"
            "    return 1.0\n"
        ),
        input_kinds=json.dumps(["label", "label_list"]),
        is_aggregated=False,
        visibility="public",
    )
    db.session.add(gm); db.session.flush()
    lm2 = LeaderboardMetric(
        leaderboard_id=lb.id,
        global_metric_id=gm.id,
        target_name="top5",
        arg_mappings=json.dumps({"gt": "gt_label", "pred": "sub_label_topk_pred"}),
        sort_direction="higher_is_better",
        pooling_type="mean",
    )
    db.session.add(lm2); db.session.commit()

    # Re-run eval for the existing submission (eager).
    tasks.process_submission.delay(sub_id)

    mr = MetricResult.query.filter_by(
        submission_id=sub_id, leaderboard_metric_id=lm2.id,
    ).one()
    assert mr.value is None
    assert mr.error_message and "Not computed" in mr.error_message
    assert "label_topk_pred" in mr.error_message
    # No per-sample lm_<id> CFs were written for the uncomputed metric.
    assert CustomField.query.filter_by(
        submission_id=sub_id, name=f"lm_{lm2.id}",
    ).count() == 0
