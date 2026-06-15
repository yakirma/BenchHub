import json
from pathlib import Path

from app import (
    CustomField,
    Dataset,
    DatasetField,
    Leaderboard,
    LeaderboardMaterialization,
    db,
)
from benchhub.lb_materialize import materialize_for_lb


def test_materialize_for_lb_writes_lb_scoped_gt_and_uses_lb_params(
    db_session, monkeypatch
):
    calls = []

    def fake_materialize_hf_to_typed_dir(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["staging_dir"])
        (root / "label").mkdir(parents=True)
        samples = ["full_s000010", "full_s000020"]
        (root / "manifest.json").write_text(json.dumps({
            "name": "fake",
            "fields": [{"name": "label", "kind": "label", "role": "gt"}],
            "samples": samples,
        }))
        (root / "label" / "full_s000010.txt").write_text("1")
        (root / "label" / "full_s000020.txt").write_text("0")
        return {"samples": 2}

    monkeypatch.setattr(
        "benchhub.hf_materialize.materialize_hf_to_typed_dir",
        fake_materialize_hf_to_typed_dir,
    )

    ds = Dataset(
        name="decoupled_ds",
        source_metadata=json.dumps({
            "repo_id": "org/repo",
            "split": "preview_split",
            "config_name": "preview_cfg",
            "sample_name_from": "image_id",
        }),
    )
    db.session.add(ds); db.session.flush()
    db.session.add(DatasetField(dataset_id=ds.id, name="label",
                                kind="label", role="gt"))
    lb = Leaderboard(name="decoupled_lb", summary_metrics="")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.flush()
    db.session.add(LeaderboardMaterialization(
        leaderboard_id=lb.id, sample_cap=2, sampling="random",
        sampling_seed=123, shard_cap=-1, split="test",
        config_name="core", status="pending",
    ))
    db.session.commit()

    summary = materialize_for_lb(
        leaderboard=lb,
        dataset=ds,
        db_session=db.session,
        upload_folder="/tmp/uploads",
        CustomField=CustomField,
        LeaderboardMaterialization=LeaderboardMaterialization,
    )

    assert summary["samples_picked"] == 2
    assert summary["gt_rows"] == 2
    assert calls[0]["repo_id"] == "org/repo"
    assert calls[0]["split"] == "test"
    assert calls[0]["config_name"] == "core"
    assert calls[0]["sample_cap"] == 2
    assert calls[0]["shard_cap"] == -1
    # LB-level 'random' maps to the HF materializer's 'uniform' strategy name.
    assert calls[0]["sampling"] == "uniform"
    assert calls[0]["seed"] == 123
    assert calls[0]["sample_name_from"] == "image_id"

    rows = CustomField.query.filter_by(leaderboard_id=lb.id).order_by(
        CustomField.sample_name,
    ).all()
    assert [(r.sample_name, r.name, r.data_type, r.value_text) for r in rows] == [
        ("full_s000010", "label", "label", "1"),
        ("full_s000020", "label", "label", "0"),
    ]
