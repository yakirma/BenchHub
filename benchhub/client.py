"""Minimal Phase-B client for submitting typed predictions to BenchHub.

Usage:

    import benchhub as bh
    client = bh.Client()  # reads BENCHHUB_API_TOKEN + BENCHHUB_BASE_URL

    sub = client.submission(leaderboard_id=42, name="my-resnet50")
    for sample_name in samples_to_score:
        pred = my_model.predict(load_input(sample_name))
        sub.predict(sample_name, depth_pred=bh.Depth(pred, unit="meters"))
    result = sub.submit()
    print(result["url"])  # → https://runbenchhub.com/submission/123

The HTTP transport is swappable so tests can hand in a Flask `test_client`
instead of going over the network.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from benchhub.types import DTYPES, DataType


_DEFAULT_BASE_URL = "https://runbenchhub.com"


class _RequestsTransport:
    """Production transport — thin wrapper around `requests`."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post_submission_zip(self, leaderboard_id: int, name: str | None,
                            zip_bytes: bytes, token: str) -> dict:
        # Lazy import so the dependency only matters at network time.
        import requests
        files = {"submission_zip": ("submission.zip", zip_bytes, "application/zip")}
        data = {"name": name} if name else {}
        resp = requests.post(
            f"{self.base_url}/api/submit/{leaderboard_id}",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
            data=data,
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()

    def post_dataset_zip(self, zip_bytes: bytes, token: str,
                         *, visibility: str = "public") -> dict:
        """Upload a typed dataset ZIP. Returns the server's import
        summary (`dataset_id`, `samples`, `fields`, …)."""
        import requests
        resp = requests.post(
            f"{self.base_url}/api/datasets",
            headers={"Authorization": f"Bearer {token}"},
            files={"dataset_zip": ("dataset.zip", zip_bytes, "application/zip")},
            data={"visibility": visibility},
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()

    def get_leaderboard_contract(self, leaderboard_id: int) -> list[dict]:
        """Fetch the LB's pred wire-contract. Token optional — the
        server-side route is visibility-gated, but public LBs respond
        to anonymous requests too."""
        import requests
        resp = requests.get(
            f"{self.base_url}/api/leaderboard/{leaderboard_id}/contract",
            headers=({"Authorization": f"Bearer {self.token}"}
                     if self.token else {}),  # type: ignore[attr-defined]
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()


class BenchHubAPIError(Exception):
    """Raised when the server returns a non-2xx response."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"BenchHub API {status_code}: {payload.get('error', payload)!r}")


class Client:
    """Entry point for programmatic submissions.

    Configuration precedence is constructor args > environment.
    Token comes from `BENCHHUB_TOKEN` (preferred) with a
    `BENCHHUB_API_TOKEN` fallback for back-compat; base URL from
    `BENCHHUB_BASE_URL`. The `transport` arg is the HTTP plumbing;
    tests pass a Flask-test-client wrapper, production gets the default
    `requests`-backed one.
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        *,
        transport: Any = None,
    ):
        self.token = (
            token
            or os.environ.get("BENCHHUB_TOKEN")
            or os.environ.get("BENCHHUB_API_TOKEN")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("BENCHHUB_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.transport = transport or _RequestsTransport(self.base_url)

    def submission(self, leaderboard_id: int, name: str | None = None) -> "SubmissionBuilder":
        """Open a new in-memory builder. Call `.predict()` per sample,
        then `.submit()` to send the whole package."""
        return SubmissionBuilder(self, leaderboard_id, name)

    def leaderboard_contract(self, leaderboard_id: int) -> list[dict]:
        """Fetch the LB's pred wire-contract (kinds + params, including
        any `shape_match` constraints). Hand the result to
        `SubmissionBuilder.set_contract()` so client-side validation
        catches shape mismatches before any ZIP upload."""
        return self.transport.get_leaderboard_contract(leaderboard_id)

    def create_dataset(
        self,
        name: str,
        *,
        visibility: str = "public",
    ) -> "BHDatasetCreator":
        """Open an in-memory dataset builder. Declare fields with
        `.add_field()` (optional — schema can also be inferred from
        the first `.add_sample()` call), then stage typed values per
        sample, then `.create()` to send the whole package up."""
        return BHDatasetCreator(self, name, visibility=visibility)

    def submit_directory(
        self,
        leaderboard_id: int,
        directory: str | os.PathLike,
        *,
        name: str | None = None,
    ) -> dict:
        """Submit a pre-built predictions directory (`manifest.json` +
        `<field>/<sample>.<ext>`). Convenient for non-Python workflows
        that write predictions out to disk before submitting."""
        directory = Path(directory)
        if not (directory / "manifest.json").exists():
            raise FileNotFoundError(f"manifest.json missing under {directory}")
        zip_bytes = _zip_directory(directory)
        return self._post_zip(leaderboard_id, name, zip_bytes)

    # internal -----------------------------------------------------------------
    def _post_zip(self, leaderboard_id: int, name: str | None, zip_bytes: bytes) -> dict:
        if not self.token:
            raise ValueError(
                "BenchHub Client has no API token — pass `token=...` or "
                "set BENCHHUB_TOKEN in your environment."
            )
        return self.transport.post_submission_zip(
            leaderboard_id, name, zip_bytes, self.token,
        )

    def _post_dataset_zip(self, zip_bytes: bytes, *, visibility: str) -> dict:
        if not self.token:
            raise ValueError(
                "BenchHub Client has no API token — pass `token=...` or "
                "set BENCHHUB_TOKEN in your environment."
            )
        return self.transport.post_dataset_zip(
            zip_bytes, self.token, visibility=visibility,
        )


class SubmissionBuilder:
    """Staging area for a typed submission.

    Predictions are accumulated in memory and packaged into a
    manifest.json + on-disk-layout ZIP at submit time. The kinds + params
    declared on each `DataType` instance flow straight into the manifest,
    so the server can validate against the LB contract without separate
    metadata.
    """

    def __init__(self, client: Client, leaderboard_id: int, name: str | None):
        self.client = client
        self.leaderboard_id = leaderboard_id
        self.name = name
        # sample_name -> field_name -> DataType instance
        self._preds: dict[str, dict[str, DataType]] = {}
        # Optional LB contract (set via `set_contract` or
        # `fetch_contract`). When present + paired with per-sample
        # input shapes, build_zip enforces shape_match locally before
        # any upload.
        self._contract: list[dict] | None = None
        # sample_name -> input_field_name -> (H, W)
        self._input_shapes: dict[str, dict[str, tuple[int, int]]] = {}

    def predict(self, sample_name: str, **typed_predictions: DataType) -> None:
        """Stage one or more typed prediction values for a sample."""
        if not typed_predictions:
            raise ValueError("predict() needs at least one field=instance kwarg")
        for field_name, inst in typed_predictions.items():
            if not isinstance(inst, DataType):
                raise TypeError(
                    f"prediction {field_name!r} must be a benchhub.DataType "
                    f"instance; got {type(inst).__name__}"
                )
            inst.validate()
        bucket = self._preds.setdefault(sample_name, {})
        bucket.update(typed_predictions)

    def set_contract(self, contract: list[dict]) -> None:
        """Tell the builder what pred shapes the LB expects. Pair with
        `set_input_shape(...)` calls to get client-side `shape_match`
        cross-checks before the upload. Usually obtained via
        `bh.Client(...).leaderboard_contract(lb_id)`."""
        if not isinstance(contract, list):
            raise TypeError("contract must be a list of {name,kind,params,role} dicts")
        self._contract = contract

    def fetch_contract(self) -> list[dict]:
        """Pull the LB's contract from the server + remember it on
        this builder. Returns the contract so the caller can also
        inspect it (e.g. to size predictions to the right kind)."""
        contract = self.client.leaderboard_contract(self.leaderboard_id)
        self.set_contract(contract)
        return contract

    def set_input_shape(self, sample_name: str, **shapes_by_field: tuple[int, int]) -> None:
        """Record per-sample spatial (H, W) shapes for input fields.

        Used together with `set_contract(...)` to enforce a pred
        field's `params.shape_match=<input_field_name>` locally —
        when the user iterates over a dataset they already know the
        input's shape, so handing it in here costs nothing and saves
        a round-trip to discover a shape mismatch."""
        if not isinstance(sample_name, str) or not sample_name:
            raise ValueError("sample_name must be a non-empty string")
        bucket = self._input_shapes.setdefault(sample_name, {})
        for field, shape in shapes_by_field.items():
            try:
                bucket[field] = (int(shape[0]), int(shape[1]))
            except (TypeError, ValueError, IndexError) as e:
                raise ValueError(
                    f"sample {sample_name!r} input {field!r}: shape "
                    f"must be a 2-tuple (H, W); got {shape!r}"
                ) from e

    @property
    def samples(self) -> list[str]:
        return sorted(self._preds.keys())

    @property
    def fields(self) -> list[str]:
        seen: dict[str, None] = {}  # preserve first-seen order
        for prediction_map in self._preds.values():
            for f in prediction_map:
                seen.setdefault(f, None)
        return list(seen)

    def build_manifest(self) -> dict:
        """Mirror what the server's `validate_submission_manifest` expects."""
        if not self._preds:
            raise ValueError("no predictions staged; call .predict(...) first")
        # Field schema is the union across samples — every sample must
        # supply every field (server enforces this by treating missing
        # files as missing-prediction errors).
        all_fields = self.fields
        type_by_field: dict[str, type[DataType]] = {}
        params_by_field: dict[str, dict] = {}
        for sample_preds in self._preds.values():
            for field_name, inst in sample_preds.items():
                cls = type(inst)
                if field_name in type_by_field and type_by_field[field_name] is not cls:
                    raise ValueError(
                        f"prediction {field_name!r} has mixed types across "
                        f"samples: {type_by_field[field_name].__name__} vs "
                        f"{cls.__name__}"
                    )
                type_by_field[field_name] = cls
                # First-seen params wins — DataType params are LB-level
                # in spirit, so they should be identical across samples.
                params_by_field.setdefault(field_name, inst.params)
        predictions = [
            {
                "name": field_name,
                "kind": type_by_field[field_name].kind,
                "params": params_by_field[field_name],
            }
            for field_name in all_fields
        ]
        return {
            "name": self.name or "submission",
            "version": "1.0",
            "predictions": predictions,
            "samples": self.samples,
        }

    def _validate_shape_constraint(self, manifest: dict) -> None:
        """Pre-upload shape check.

        Two contract forms (mutually exclusive per pred field):
          * ``params.shape_match=<input_field>`` — needs a matching
            entry in `set_input_shape(...)` to enforce. Otherwise
            falls through to the server's check.
          * ``params.shape=[H, W]`` — fixed shape; enforced for
            every staged sample without needing any input-shape
            registration.
        Both set on the same field → ValueError (raised here too so
        the user catches the contract author's mistake locally)."""
        if not self._contract:
            return
        by_name = {c["name"]: c for c in self._contract if isinstance(c, dict)}
        for p in manifest["predictions"]:
            spec = by_name.get(p["name"])
            if not spec:
                continue
            params = spec.get("params") or {}
            shape_match = params.get("shape_match")
            raw_fixed = params.get("shape")
            fixed_shape = None
            if raw_fixed is not None:
                if (not isinstance(raw_fixed, (list, tuple))
                        or len(raw_fixed) != 2):
                    raise ValueError(
                        f"contract pred {p['name']!r}: params.shape must "
                        f"be a 2-element [H, W]; got {raw_fixed!r}"
                    )
                try:
                    fixed_shape = (int(raw_fixed[0]), int(raw_fixed[1]))
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"contract pred {p['name']!r}: params.shape "
                        f"entries must be ints; got {raw_fixed!r}"
                    ) from e
            if shape_match and fixed_shape is not None:
                raise ValueError(
                    f"contract pred {p['name']!r}: params.shape and "
                    f"params.shape_match can't be set together; pick one."
                )
            if not (shape_match or fixed_shape):
                continue
            for sample_name in manifest["samples"]:
                inst = self._preds[sample_name].get(p["name"])
                if inst is None:
                    continue
                arr = getattr(inst, "array", None)
                if arr is None or getattr(arr, "ndim", 0) < 2:
                    continue
                pred_shape = tuple(arr.shape[:2])
                if fixed_shape is not None:
                    if pred_shape != fixed_shape:
                        raise ValueError(
                            f"sample {sample_name!r} pred {p['name']!r}: "
                            f"shape {pred_shape} != contract shape "
                            f"{fixed_shape}"
                        )
                    continue
                expected = (
                    self._input_shapes.get(sample_name, {}).get(shape_match)
                )
                if expected is None:
                    continue
                if pred_shape != tuple(expected):
                    raise ValueError(
                        f"sample {sample_name!r} pred {p['name']!r}: "
                        f"shape {pred_shape} != input {shape_match!r} shape "
                        f"{tuple(expected)} (contract requires shape_match)"
                    )

    # Earlier code/tests call the old name; keep it working.
    _validate_shape_match = _validate_shape_constraint

    def build_zip(self) -> bytes:
        """Produce the submission ZIP bytes the server consumes."""
        manifest = self.build_manifest()
        self._validate_shape_constraint(manifest)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            for p in manifest["predictions"]:
                cls = DTYPES[p["kind"]]
                ext = cls.file_ext or ".txt"
                for sample_name in manifest["samples"]:
                    inst = self._preds[sample_name].get(p["name"])
                    if inst is None:
                        raise ValueError(
                            f"sample {sample_name!r} missing prediction "
                            f"for field {p['name']!r}"
                        )
                    zf.writestr(f"{p['name']}/{sample_name}{ext}", inst.encode())
        return buf.getvalue()

    def submit(self) -> dict:
        """Build the ZIP and POST it. Returns the server's response payload."""
        return self.client._post_zip(self.leaderboard_id, self.name, self.build_zip())


# ---------------------------------------------------------------------------
# BHDatasetCreator — client-side dataset builder
# ---------------------------------------------------------------------------

_VALID_ROLES = {"input", "gt", "pred"}
# Roles that carry per-sample values; `pred` is schema-only.
_DATA_BEARING_ROLES = {"input", "gt"}


class BHDatasetCreator:
    """Stage a typed dataset locally, then upload it as one ZIP.

    Mirrors `SubmissionBuilder` for the dataset side. The builder
    accumulates schema (`add_field`) + per-sample values (`add_sample`)
    in memory, validates every `DataType` instance + checks that
    every sample has every declared field, packs everything into the
    typed-manifest layout the server already knows how to ingest, and
    POSTs it as a multipart upload.

    Example:

        creator = client.create_dataset("nyu-tiny")
        creator.add_field("image",    bh.Image, role="input")
        creator.add_field("depth_gt", bh.Depth, role="gt",
                          params={"unit": "meters"})

        for sample_id, img_arr, depth_arr in iter_samples():
            creator.add_sample(
                sample_id,
                image=bh.Image(img_arr),
                depth_gt=bh.Depth(depth_arr, unit="meters"),
            )

        result = creator.create()
        print(result["dataset_id"], result["samples"])
    """

    def __init__(self, client: Client, name: str, *, visibility: str = "public"):
        if not name or not name.strip():
            raise ValueError("dataset name must be a non-empty string")
        self.client = client
        self.name = name.strip()
        self.visibility = visibility
        # Schema: field_name → {kind, role, params}
        self._schema: dict[str, dict] = {}
        # Data: sample_name → field_name → DataType instance
        self._samples: dict[str, dict[str, DataType]] = {}

    def add_field(
        self,
        name: str,
        kind: str | type[DataType] | None = None,
        *,
        role: str = "gt",
        params: dict | None = None,
    ) -> None:
        """Declare one field of the dataset's schema.

        `kind` accepts either the wire-kind string (`"depth"`) or the
        `DataType` subclass (`bh.Depth`). If omitted, the kind is
        inferred from the first `add_sample()` instance for this name.
        `role` is `"input"` or `"gt"`. `params` is the per-instance
        metadata the type needs at decode time (e.g.
        `{"unit": "meters"}` for `Depth`).
        """
        if not name or not isinstance(name, str):
            raise ValueError("field name must be a non-empty string")
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}; got {role!r}")
        kind_str: str | None = None
        if kind is not None:
            if isinstance(kind, type) and issubclass(kind, DataType):
                kind_str = kind.kind
            elif isinstance(kind, str):
                if kind not in DTYPES:
                    raise ValueError(f"unknown kind {kind!r}; known: {sorted(DTYPES)}")
                kind_str = kind
            else:
                raise TypeError(
                    f"`kind` must be a benchhub.DataType subclass, a wire-kind "
                    f"string, or None; got {type(kind).__name__}"
                )
        if name in self._schema and kind_str and self._schema[name]["kind"] != kind_str:
            raise ValueError(
                f"field {name!r} already declared with kind "
                f"{self._schema[name]['kind']!r}; can't redeclare as {kind_str!r}"
            )
        self._schema[name] = {
            "kind": kind_str,
            "role": role,
            "params": dict(params or {}),
        }

    def add_sample(self, sample_name: str, **typed_values: DataType) -> None:
        """Stage typed values for one sample. Each kwarg is
        `field_name=DataType_instance`. Each instance is validated
        on the spot — bad shapes / dtypes raise immediately so you
        catch problems while you still have the surrounding context.

        Pred fields (declared via `add_field(..., role="pred")`) are
        schema-only — passing a value for one raises ValueError, since
        prediction data comes from submissions, not the dataset.
        """
        if not sample_name or not isinstance(sample_name, str):
            raise ValueError("sample_name must be a non-empty string")
        if not typed_values:
            raise ValueError("add_sample() needs at least one field=instance kwarg")
        for field_name, inst in typed_values.items():
            if not isinstance(inst, DataType):
                raise TypeError(
                    f"sample {sample_name!r} field {field_name!r}: expected a "
                    f"benchhub.DataType instance; got {type(inst).__name__}"
                )
            inst.validate()
            # Infer / cross-check schema.
            declared = self._schema.get(field_name)
            if declared is None:
                self._schema[field_name] = {
                    "kind": inst.kind,
                    "role": "gt",
                    "params": dict(inst.params),
                }
            else:
                if declared["role"] not in _DATA_BEARING_ROLES:
                    raise ValueError(
                        f"sample {sample_name!r} field {field_name!r}: "
                        f"role={declared['role']!r} is schema-only — "
                        f"prediction values come from submissions, not "
                        f"dataset uploads."
                    )
                if declared["kind"] is None:
                    declared["kind"] = inst.kind
                elif declared["kind"] != inst.kind:
                    raise ValueError(
                        f"sample {sample_name!r} field {field_name!r}: "
                        f"declared kind {declared['kind']!r} != instance kind {inst.kind!r}"
                    )
        bucket = self._samples.setdefault(sample_name, {})
        bucket.update(typed_values)

    @property
    def fields(self) -> list[str]:
        # Sorted so the manifest order is deterministic across runs.
        return sorted(self._schema)

    @property
    def samples(self) -> list[str]:
        return sorted(self._samples)

    def build_manifest(self) -> dict:
        """Mirror the on-disk typed-manifest format the server expects."""
        if not self._samples:
            raise ValueError("no samples staged; call .add_sample(...) first")
        # Every sample must supply every DATA-BEARING declared field
        # (input + gt). Pred fields are schema-only — they declare
        # the wire contract for submissions but carry no per-sample
        # values at the dataset level.
        data_fields = {
            n for n, s in self._schema.items()
            if s["role"] in _DATA_BEARING_ROLES
        }
        for name in self.samples:
            present = set(self._samples[name])
            missing = sorted(data_fields - present)
            if missing:
                raise ValueError(
                    f"sample {name!r} missing field values for: {missing}"
                )
        return {
            "name": self.name,
            "version": "1.0",
            "fields": [
                {
                    "name": fname,
                    "kind": self._schema[fname]["kind"],
                    "role": self._schema[fname]["role"],
                    "params": self._schema[fname]["params"],
                }
                for fname in self.fields
            ],
            "samples": self.samples,
        }

    def build_zip(self) -> bytes:
        """Produce the dataset ZIP bytes the server consumes."""
        manifest = self.build_manifest()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            for f in manifest["fields"]:
                # Pred fields are schema-only — no per-sample files.
                if f["role"] not in _DATA_BEARING_ROLES:
                    continue
                cls = DTYPES[f["kind"]]
                ext = cls.file_ext or ".txt"
                for sample_name in manifest["samples"]:
                    inst = self._samples[sample_name][f["name"]]
                    zf.writestr(f"{f['name']}/{sample_name}{ext}", inst.encode())
        return buf.getvalue()

    def create(self) -> dict:
        """Build the ZIP and POST it. Returns the server's import summary."""
        return self.client._post_dataset_zip(self.build_zip(), visibility=self.visibility)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zip_directory(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in directory.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(directory)))
    return buf.getvalue()


class FlaskTestClientTransport:
    """Test-only transport that wraps a Flask `app.test_client()` so the
    client code can be exercised end-to-end without standing up a real
    HTTP server."""

    def __init__(self, test_client):
        self.test_client = test_client

    def get_leaderboard_contract(self, leaderboard_id: int) -> list[dict]:
        resp = self.test_client.get(f"/api/leaderboard/{leaderboard_id}/contract")
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload or {})
        return payload if isinstance(payload, list) else []

    def post_submission_zip(self, leaderboard_id: int, name: str | None,
                            zip_bytes: bytes, token: str) -> dict:
        from io import BytesIO
        resp = self.test_client.post(
            f"/api/submit/{leaderboard_id}",
            data={
                "submission_zip": (BytesIO(zip_bytes), "submission.zip"),
                **({"name": name} if name else {}),
            },
            headers={"Authorization": f"Bearer {token}"},
            content_type="multipart/form-data",
        )
        try:
            payload = resp.get_json() or {}
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload)
        return payload

    def post_dataset_zip(self, zip_bytes: bytes, token: str,
                         *, visibility: str = "public") -> dict:
        from io import BytesIO
        resp = self.test_client.post(
            "/api/datasets",
            data={
                "dataset_zip": (BytesIO(zip_bytes), "dataset.zip"),
                "visibility": visibility,
            },
            headers={"Authorization": f"Bearer {token}"},
            content_type="multipart/form-data",
        )
        try:
            payload = resp.get_json() or {}
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload)
        return payload
