"""role='pred' on a dataset — schema-only fields with no per-sample data.

Locks the new behaviour end-to-end:
  - BHDatasetCreator.add_field accepts role='pred'.
  - add_sample rejects values for pred fields (would corrupt the
    contract: predictions come from submissions, not the dataset).
  - The dataset ZIP carries the DatasetField row but no
    `<field>/<sample>.<ext>` files for pred fields.
  - The LB pred contract picks up explicit pred fields and skips
    GT mirroring when present.
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

import benchhub as bh
from app import (
    CustomField,
    Dataset,
    DatasetField,
    Leaderboard,
    Sample,
    User,
    _lb_pred_contract_from_dataset_fields,
    db,
    generate_api_token,
)


# ---------------------------------------------------------------------------
# BHDatasetCreator side
# ---------------------------------------------------------------------------

def _local_client() -> bh.Client:
    return bh.Client(token="t", base_url="http://x")


def test_add_field_accepts_pred_role():
    creator = _local_client().create_dataset("d")
    creator.add_field("label_pred", bh.Label, role="pred")
    # Need at least one data-bearing field to build a manifest.
    creator.add_field("label", bh.Label, role="gt")
    creator.add_sample("s0", label=bh.Label("cat"))
    manifest = creator.build_manifest()
    by_name = {f["name"]: f for f in manifest["fields"]}
    assert by_name["label_pred"]["role"] == "pred"
    assert by_name["label_pred"]["kind"] == "label"


def test_add_sample_rejects_pred_field_values():
    creator = _local_client().create_dataset("d")
    creator.add_field("label",      bh.Label, role="gt")
    creator.add_field("label_pred", bh.Label, role="pred")
    with pytest.raises(ValueError, match="schema-only"):
        creator.add_sample(
            "s0",
            label=bh.Label("cat"),
            label_pred=bh.Label("dog"),  # schema-only — should bounce
        )


def test_build_zip_skips_per_sample_files_for_pred_role():
    creator = _local_client().create_dataset("with-pred")
    creator.add_field("image",      bh.Image, role="input")
    creator.add_field("label",      bh.Label, role="gt")
    creator.add_field("label_pred", bh.Label, role="pred")

    import numpy as np
    for s in ("s0", "s1"):
        creator.add_sample(
            s,
            image=bh.Image(np.zeros((4, 4, 3), dtype=np.uint8)),
            label=bh.Label("cat" if s == "s0" else "dog"),
        )
    blob = creator.build_zip()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = sorted(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))

    # Image + label files exist; nothing under label_pred/.
    assert "image/s0.png" in names and "image/s1.png" in names
    assert "label/s0.txt" in names and "label/s1.txt" in names
    assert not any(n.startswith("label_pred/") for n in names)
    # But the manifest still declares the pred field — it's the contract.
    by_name = {f["name"]: f for f in manifest["fields"]}
    assert by_name["label_pred"]["role"] == "pred"


def test_build_manifest_requires_data_only_for_input_and_gt():
    """A sample missing a pred-field value is fine; missing a GT or
    input value is not."""
    creator = _local_client().create_dataset("d")
    creator.add_field("a",       bh.Scalar, role="gt")
    creator.add_field("a_pred",  bh.Scalar, role="pred")
    creator.add_sample("s0", a=bh.Scalar(0.5))   # no a_pred — OK
    # No missing-field error:
    manifest = creator.build_manifest()
    assert manifest["samples"] == ["s0"]
    assert {f["name"] for f in manifest["fields"]} == {"a", "a_pred"}


# ---------------------------------------------------------------------------
# Server-side: end-to-end through /api/datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_user(db_session):
    u = User(
        email='predrole-admin@bench.local', display_name='pra',
        oauth_provider='github', oauth_sub='pra-1',
        is_admin=True,
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def test_end_to_end_dataset_with_pred_field_creates_dataset_field_only(
    client, admin_user,
):
    """The server-side importer writes the DatasetField schema row but
    no per-sample CustomField rows for pred fields."""
    bh_client = bh.Client(
        token=admin_user.api_token, base_url='http://test',
        transport=bh.FlaskTestClientTransport(client),
    )
    creator = bh_client.create_dataset("pred-on-dataset")
    creator.add_field("label",      bh.Label, role="gt")
    creator.add_field("label_pred", bh.Label, role="pred")
    creator.add_sample("s0", label=bh.Label("cat"))
    creator.add_sample("s1", label=bh.Label("dog"))
    result = creator.create()

    ds = Dataset.query.get(result["dataset_id"])
    # DatasetField rows: both fields land in the schema.
    schema = {f.name: f for f in DatasetField.query.filter_by(dataset_id=ds.id).all()}
    assert schema["label"].role == "gt"
    assert schema["label_pred"].role == "pred"

    # CustomField rows: only for the data-bearing GT field.
    n_label_cf = CustomField.query.filter_by(name="label").count()
    n_pred_cf = CustomField.query.filter_by(name="label_pred").count()
    assert n_label_cf == 2  # one per sample
    assert n_pred_cf == 0   # schema-only; no per-sample values stored


# ---------------------------------------------------------------------------
# LB pred-contract derivation
# ---------------------------------------------------------------------------

def _seed_dataset_with_schema(name: str, fields: list[dict]) -> Dataset:
    ds = Dataset(name=name, visibility="public")
    db.session.add(ds); db.session.flush()
    for f in fields:
        df = DatasetField(
            dataset_id=ds.id, name=f["name"], kind=f["kind"],
            role=f.get("role", "gt"),
        )
        if f.get("params"):
            df.set_params(f["params"])
        db.session.add(df)
    db.session.commit()
    return ds


def test_lb_contract_uses_explicit_pred_fields_when_present(db_session):
    """If the dataset declares pred fields, those are the contract;
    GT mirroring is skipped to avoid duplicate listings."""
    ds = _seed_dataset_with_schema("explicit_pred_ds", [
        {"name": "image",         "kind": "image", "role": "input"},
        {"name": "label",         "kind": "label", "role": "gt"},
        {"name": "label_logits",  "kind": "json",  "role": "pred"},
    ])
    lb = Leaderboard(name="explicit_pred_lb", summary_metrics="",
                     visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    contract = _lb_pred_contract_from_dataset_fields(lb)
    names = {e["name"] for e in contract}
    # Only the explicit pred field, NOT label_pred (which would have
    # come from GT mirroring).
    assert names == {"label_logits"}
    assert contract[0]["kind"] == "json"


def test_lb_contract_empty_when_dataset_has_no_pred_fields(db_session):
    """Pred fields are part of the dataset's declared schema. A
    dataset with only GT fields and no explicit role=pred field
    has an EMPTY pred contract — the engine no longer invents
    `<name>_pred` mirror entries. Submissions against such an LB
    fail at manifest validation time with a clear "no pred fields"
    error rather than the runtime guessing what the user meant."""
    ds = _seed_dataset_with_schema("only_gt_ds", [
        {"name": "label", "kind": "label", "role": "gt"},
    ])
    lb = Leaderboard(name="only_gt_lb", summary_metrics="",
                     visibility="public")
    lb.datasets.append(ds)
    db.session.add(lb); db.session.commit()

    contract = _lb_pred_contract_from_dataset_fields(lb)
    assert contract == []


def test_lb_contract_unions_explicit_pred_across_datasets(db_session):
    """When multiple attached datasets each declare pred fields, the
    contract is the union (first occurrence wins on name collision)."""
    ds_a = _seed_dataset_with_schema("pred_a", [
        {"name": "out_a", "kind": "scalar", "role": "pred"},
    ])
    ds_b = _seed_dataset_with_schema("pred_b", [
        {"name": "out_b", "kind": "scalar", "role": "pred"},
    ])
    lb = Leaderboard(name="union_pred_lb", summary_metrics="",
                     visibility="public")
    lb.datasets.extend([ds_a, ds_b])
    db.session.add(lb); db.session.commit()

    names = {e["name"] for e in _lb_pred_contract_from_dataset_fields(lb)}
    assert names == {"out_a", "out_b"}
