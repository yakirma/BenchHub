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


class BenchHubAPIError(Exception):
    """Raised when the server returns a non-2xx response."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"BenchHub API {status_code}: {payload.get('error', payload)!r}")


class Client:
    """Entry point for programmatic submissions.

    Configuration precedence is constructor args > environment.
    Token comes from `BENCHHUB_API_TOKEN`; base URL from
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
        self.token = token or os.environ.get("BENCHHUB_API_TOKEN") or ""
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
                "set BENCHHUB_API_TOKEN."
            )
        return self.transport.post_submission_zip(
            leaderboard_id, name, zip_bytes, self.token,
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

    def build_zip(self) -> bytes:
        """Produce the submission ZIP bytes the server consumes."""
        manifest = self.build_manifest()
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
