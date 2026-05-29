"""BHDatasetCreator + POST /api/datasets — typed dataset upload from Python."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import benchhub as bh
from app import (
    CustomField,
    Dataset,
    DatasetField,
    Sample,
    User,
    db,
    generate_api_token,
)


# ---------------------------------------------------------------------------
# Construction + add_field
# ---------------------------------------------------------------------------

def _local_client() -> bh.Client:
    return bh.Client(token="t", base_url="http://x")


def test_create_dataset_requires_non_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        _local_client().create_dataset("   ")


def test_add_field_rejects_unknown_kind_string():
    creator = _local_client().create_dataset("d")
    with pytest.raises(ValueError, match="unknown kind"):
        creator.add_field("x", "not-a-kind")


def test_add_field_accepts_datatype_class():
    creator = _local_client().create_dataset("d")
    creator.add_field("d", bh.Depth, role="gt", params={"unit": "meters"})
    # Stage one sample so the manifest is buildable.
    creator.add_sample("s0",
                       d=bh.Depth(np.zeros((4, 4), dtype=np.float32), unit="meters"))
    schema = creator.build_manifest()["fields"]
    assert schema[0] == {
        "name": "d", "kind": "depth", "role": "gt",
        "params": {"unit": "meters"},
    }


def test_add_field_rejects_bad_role():
    creator = _local_client().create_dataset("d")
    with pytest.raises(ValueError, match="role"):
        creator.add_field("x", bh.Scalar, role="weird")  # not input/gt/pred


def test_add_field_rejects_kind_redeclaration():
    creator = _local_client().create_dataset("d")
    creator.add_field("x", bh.Scalar)
    with pytest.raises(ValueError, match="kind"):
        creator.add_field("x", bh.Label)  # same name, different kind → boom


# ---------------------------------------------------------------------------
# add_sample
# ---------------------------------------------------------------------------

def test_add_sample_rejects_non_datatype_value():
    creator = _local_client().create_dataset("d")
    with pytest.raises(TypeError, match="DataType"):
        creator.add_sample("s0", x=0.5)  # bare float


def test_add_sample_validates_each_instance():
    creator = _local_client().create_dataset("d")
    bad_image = bh.Image(np.zeros((4, 4, 5), dtype=np.uint8))  # 5 channels: invalid
    with pytest.raises(ValueError):
        creator.add_sample("s0", img=bad_image)


def test_add_sample_infers_schema_when_no_add_field_called():
    creator = _local_client().create_dataset("d")
    creator.add_sample("s0",
                       img=bh.Image(np.zeros((4, 4, 3), dtype=np.uint8)),
                       label=bh.Label("cat"))
    m = creator.build_manifest()
    by_name = {f["name"]: f for f in m["fields"]}
    assert by_name["img"]["kind"] == "image"
    assert by_name["label"]["kind"] == "label"
    # Default role when not declared.
    assert {f["role"] for f in m["fields"]} == {"gt"}


def test_add_sample_cross_checks_kind_against_declared_schema():
    creator = _local_client().create_dataset("d")
    creator.add_field("out", bh.Scalar, role="gt")
    with pytest.raises(ValueError, match="kind"):
        creator.add_sample("s0", out=bh.Label("cat"))  # declared scalar


def test_build_manifest_requires_each_sample_to_have_each_field():
    creator = _local_client().create_dataset("d")
    creator.add_field("a", bh.Scalar)
    creator.add_field("b", bh.Scalar)
    creator.add_sample("s0", a=bh.Scalar(0.5), b=bh.Scalar(0.7))
    creator.add_sample("s1", a=bh.Scalar(0.6))  # missing 'b'
    with pytest.raises(ValueError, match="missing field"):
        creator.build_manifest()


def test_build_manifest_empty_raises():
    creator = _local_client().create_dataset("d")
    with pytest.raises(ValueError, match="no samples"):
        creator.build_manifest()


# ---------------------------------------------------------------------------
# ZIP layout — matches the server's import_typed_dataset shape
# ---------------------------------------------------------------------------

def test_build_zip_writes_manifest_and_per_field_files():
    creator = _local_client().create_dataset("nyu-tiny")
    creator.add_field("image",    bh.Image, role="input")
    creator.add_field("depth_gt", bh.Depth, role="gt", params={"unit": "meters"})
    creator.add_field("label",    bh.Label, role="gt")

    img = bh.Image(np.zeros((4, 4, 3), dtype=np.uint8))
    depth = bh.Depth(np.ones((4, 4), dtype=np.float32), unit="meters")
    for s in ("s0", "s1"):
        creator.add_sample(s, image=img, depth_gt=depth, label=bh.Label("cat"))

    blob = creator.build_zip()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = sorted(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    # Expect: manifest + 2 samples × 3 fields, with the right extensions
    # per kind (.png for image, .npz for depth, .txt for inline label).
    assert names == sorted([
        "manifest.json",
        "image/s0.png", "image/s1.png",
        "depth_gt/s0.npz", "depth_gt/s1.npz",
        "label/s0.txt", "label/s1.txt",
    ])
    fields = {f["name"]: f for f in manifest["fields"]}
    assert fields["depth_gt"]["params"] == {"unit": "meters"}
    assert fields["image"]["role"] == "input"
    assert fields["label"]["role"] == "gt"


# ---------------------------------------------------------------------------
# End-to-end via FlaskTestClientTransport
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_user(db_session):
    u = User(
        email='ds-creator@bench.local', display_name='dsc',
        oauth_provider='github', oauth_sub='dsc-1',
        is_admin=True,
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


@pytest.fixture
def non_admin_user(db_session):
    u = User(
        email='regular@bench.local', display_name='reg',
        oauth_provider='github', oauth_sub='reg-1',
        is_admin=False,
        api_token=generate_api_token(),
    )
    db.session.add(u); db.session.commit()
    return u


def _client_with_transport(client, user):
    return bh.Client(
        token=user.api_token,
        base_url="http://test",
        transport=bh.FlaskTestClientTransport(client),
    )


def test_end_to_end_create_dataset_happy_path(client, admin_user):
    bh_client = _client_with_transport(client, admin_user)
    creator = bh_client.create_dataset("e2e-tiny")
    creator.add_field("image", bh.Image, role="input")
    creator.add_field("label", bh.Label, role="gt", params={"vocab": ["cat", "dog"]})

    for i in range(3):
        creator.add_sample(
            f"s{i}",
            image=bh.Image(np.zeros((4, 4, 3), dtype=np.uint8)),
            label=bh.Label("cat" if i % 2 == 0 else "dog"),
        )

    result = creator.create()
    assert result["samples"] == 3
    assert result["fields"] == 2
    ds = Dataset.query.get(result["dataset_id"])
    assert ds is not None
    assert ds.name == "e2e-tiny"
    # Schema landed in DatasetField as the source of truth.
    schema = {f.name: f for f in DatasetField.query.filter_by(dataset_id=ds.id).all()}
    assert schema["image"].role == "input"
    assert schema["label"].role == "gt"
    assert schema["label"].get_params() == {"vocab": ["cat", "dog"]}
    # Per-sample data is in CustomField rows.
    n_cf = CustomField.query.join(Sample).filter(Sample.dataset_id == ds.id).count()
    assert n_cf == 3 * 2  # samples × fields


def test_end_to_end_non_admin_token_accepted_within_quota(client, non_admin_user):
    """Dataset upload is open to any authenticated user — the per-user
    storage quota is the only gate. A tiny upload from a non-admin
    sails through."""
    bh_client = _client_with_transport(client, non_admin_user)
    creator = bh_client.create_dataset("regular-user-upload")
    creator.add_field("x", bh.Scalar)
    creator.add_sample("s0", x=bh.Scalar(0.5))
    result = creator.create()
    assert result["samples"] == 1
    ds = Dataset.query.get(result["dataset_id"])
    assert ds.owner_user_id == non_admin_user.id


def test_end_to_end_over_quota_returns_413(client, db_session):
    """A user already at their storage cap can't upload another byte."""
    u = User(
        email='maxed@bench.local', display_name='maxed',
        oauth_provider='github', oauth_sub='maxed-1',
        api_token=generate_api_token(),
        quota_max_storage_bytes=10,  # legacy cap; not enforced
        quota_public_max_bytes=10,
        quota_private_max_bytes=10,  # both buckets at 10 bytes
    )
    db.session.add(u); db.session.commit()
    bh_client = _client_with_transport(client, u)
    creator = bh_client.create_dataset("too-big")
    # A 32×32 RGB image is ~3KB encoded; well over the 10-byte cap.
    creator.add_field("img", bh.Image, role="input")
    creator.add_sample("s0", img=bh.Image(np.zeros((32, 32, 3), dtype=np.uint8)))

    with pytest.raises(bh.BenchHubAPIError) as excinfo:
        creator.create()
    assert excinfo.value.status_code == 413
    assert "storage limit" in excinfo.value.payload["error"].lower()


def test_end_to_end_missing_api_token_gets_401(client, db_session):
    # No bearer header — server should 401 before doing any work.
    resp = client.post(
        "/api/datasets",
        data={"dataset_zip": (io.BytesIO(b""), "x.zip")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401


def test_browser_upload_route_accepts_cookie_auth(client, admin_user, db_session):
    """The /datasets/upload route is the browser companion to
    /api/datasets — same ZIP, cookie auth, same quota gate."""
    creator = bh.Client(token="t", base_url="http://x").create_dataset("browser-up")
    creator.add_field("x", bh.Scalar)
    creator.add_sample("s0", x=bh.Scalar(0.5))
    zip_bytes = creator.build_zip()

    with client.session_transaction() as sess:
        sess['user_id'] = admin_user.id

    resp = client.post(
        '/datasets/upload',
        data={'dataset_zip': (io.BytesIO(zip_bytes), 'browser.zip'), 'visibility': 'public'},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    # Successful upload → 302 to /dataset/<id> (or /datasets if the
    # endpoint isn't around). Either way: redirect, not 4xx.
    assert resp.status_code == 302
    ds = Dataset.query.filter_by(name='browser-up').one()
    assert ds.owner_user_id == admin_user.id


def test_create_propagates_bad_zip_as_api_error(client, admin_user):
    bh_client = _client_with_transport(client, admin_user)
    # Build a malformed ZIP — random bytes.
    bh_client.token = admin_user.api_token
    resp = client.post(
        "/api/datasets",
        data={"dataset_zip": (io.BytesIO(b"definitely not a zip"), "broken.zip")},
        headers={"Authorization": f"Bearer {admin_user.api_token}"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"ZIP" in resp.data
