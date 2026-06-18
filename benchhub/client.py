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


def _decode_input_bytes(kind: str, blob: bytes, params: dict | None = None):
    """Decode a per-sample input blob into the corresponding `bh.<Kind>`
    instance — e.g. an image field arrives as a `bh.Image` carrying a
    `(H, W, 3)` uint8 ndarray, not a bare `PIL.JpegImageFile`. That way
    `iter_samples()` outputs the same type system the predict side uses.

    The DataType subclasses' `decode()` methods expect canonical
    encodings (PNG / NPZ / WAV); preview-tier blobs (colormapped JPG
    depth, waveform PNG audio) aren't canonical, so we fall back to a
    PIL-wrapped `bh.Image` for those — the user can fetch
    full-resolution canonical bytes by materialising the LB."""
    params = params or {}
    cls = DTYPES.get(kind)
    if cls is None:
        return blob
    try:
        return cls.decode(blob, params)
    except Exception:
        # Preview-tier depth/audio don't round-trip through the
        # canonical decoder. Fall back to a generic PIL+ndarray wrap
        # so the caller still gets a `bh.Image`-shaped instance
        # rather than raw bytes.
        try:
            import io as _io
            import numpy as _np
            from PIL import Image as _PILImage
            img = _PILImage.open(_io.BytesIO(blob))
            if img.mode == "P":
                img = img.convert("RGB")
            return DTYPES["image"](_np.asarray(img))
        except Exception:
            return blob


class _RequestsTransport:
    """Production transport — thin wrapper around `requests`."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post_submission_zip(self, leaderboard_id: int, name: str | None,
                            zip_bytes: bytes, token: str,
                            *, description: str | None = None,
                            link: str | None = None) -> dict:
        # Lazy import so the dependency only matters at network time.
        import requests
        files = {"submission_zip": ("submission.zip", zip_bytes, "application/zip")}
        data = {}
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        if link:
            data["link"] = link
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

    def post_json(self, path: str, payload: dict, token: str) -> dict:
        """POST a JSON body to `path` with bearer auth; return the JSON
        response. Used by the metric/visualization authoring helpers."""
        import requests
        resp = requests.post(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, body)
        return resp.json()

    def resolve_leaderboard(self, ref: str,
                            token: str | None = None) -> dict:
        """Resolve a leaderboard handle (`<owner>/<slug>`, a bare slug, or
        an id) to `{id, name, slug, owner, ref}`."""
        import requests
        resp = requests.get(
            f"{self.base_url}/api/leaderboard/resolve",
            params={"ref": ref},
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()

    def get_leaderboard_contract(self, leaderboard_id: int,
                                 token: str | None = None) -> list[dict]:
        """Fetch the LB's pred wire-contract. Token optional — the
        server-side route is visibility-gated, but public LBs respond
        to anonymous requests too."""
        import requests
        resp = requests.get(
            f"{self.base_url}/api/leaderboard/{leaderboard_id}/contract",
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()

    def get_leaderboard_samples(self, leaderboard_id: int,
                                token: str | None = None) -> dict:
        """List samples + their `role=input` field URLs / inline
        values. File-backed kinds (image/depth/mask/audio) come back
        as URLs the caller fetches lazily."""
        import requests
        resp = requests.get(
            f"{self.base_url}/api/leaderboard/{leaderboard_id}/samples",
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()

    def get_leaderboard_submissions(self, leaderboard_id: int,
                                    name: str | None = None,
                                    token: str | None = None) -> dict:
        """List submissions on an LB (optionally filtered to an exact
        `name`). Returns the server's `{count, submissions:[...]}` shape."""
        import requests
        resp = requests.get(
            f"{self.base_url}/api/leaderboard/{leaderboard_id}/submissions",
            params=({"name": name} if name else None),
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": resp.text}
            raise BenchHubAPIError(resp.status_code, payload)
        return resp.json()

    def fetch_bytes(self, url: str, token: str | None = None) -> bytes:
        """Pull a per-sample file from the BH server. Relative URLs
        are resolved against self.base_url so the listing endpoint
        can return short paths."""
        import requests
        full = url if url.startswith("http") else self.base_url + url
        resp = requests.get(
            full,
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
        )
        resp.raise_for_status()
        return resp.content

    def download_inputs_archive(self, leaderboard_id: int, dest_path: str,
                                token: str | None = None) -> None:
        """Stream the LB's bulk inputs ZIP straight to `dest_path` on
        disk — never holds the whole archive in memory, so a multi-GB
        LB is bounded by disk, not RAM. Raises BenchHubAPIError on a
        non-2xx (e.g. 404 from an older server with no archive route)
        so the caller can fall back to the per-sample path."""
        import requests
        with requests.get(
            f"{self.base_url}/api/leaderboard/{leaderboard_id}/inputs.zip",
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
            stream=True,
        ) as resp:
            if resp.status_code >= 400:
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"error": resp.text}
                raise BenchHubAPIError(resp.status_code, payload)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)


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
        self._ref_cache: dict[str, int] = {}

    def _resolve_ref(self, ref) -> int:
        """Turn a leaderboard reference into its integer id. Integers (and
        numeric strings) pass straight through; an `<owner>/<slug>` handle or
        a bare slug is resolved via the server once and cached."""
        if isinstance(ref, int):
            return ref
        s = str(ref).strip()
        if s.isdigit():
            return int(s)
        if s in self._ref_cache:
            return self._ref_cache[s]
        info = self.transport.resolve_leaderboard(s, token=self.token or None)
        lb_id = int(info["id"])
        self._ref_cache[s] = lb_id
        return lb_id

    def leaderboard(self, ref) -> "LeaderboardHandle":
        """Look up a leaderboard by handle and return an object whose methods
        need no id — the recommended, readable entry point:

            lb = client.leaderboard("john-smith/ade20k-scene-parse-150")
            print(lb.contract())
            for name, inputs in lb.iter_samples():
                sub = lb.submission("my-model")
                sub.predict(name, label_pred=bh.Label(...))
            sub.submit()

        `ref` may be a `<owner>/<slug>` handle, a bare unambiguous slug, or
        an integer id (which still works everywhere for back-compat)."""
        info = self.transport.resolve_leaderboard(
            str(ref).strip(), token=self.token or None)
        self._ref_cache[str(ref).strip()] = int(info["id"])
        return LeaderboardHandle(self, info)

    def submission(self, leaderboard_id, name: str | None = None,
                   *, description: str | None = None,
                   link: str | None = None) -> "SubmissionBuilder":
        """Open a new in-memory builder. Call `.predict()` per sample,
        then `.submit()` to send the whole package.

        `leaderboard_id` accepts an id or an `<owner>/<slug>` handle.
        `description` is a free-text blurb shown in the leaderboard's
        submission table (full text on hover). `link` is an optional
        http(s) URL the submission name will hyperlink to (e.g. a repo
        or model card). Both can also be passed at `.submit()` time."""
        return SubmissionBuilder(self, self._resolve_ref(leaderboard_id), name,
                                 description=description, link=link)

    def leaderboard_contract(self, leaderboard_id) -> list[dict]:
        """Fetch the LB's pred wire-contract (kinds + params, including
        any `shape_match` constraints). Hand the result to
        `SubmissionBuilder.set_contract()` so client-side validation
        catches shape mismatches before any ZIP upload."""
        return self.transport.get_leaderboard_contract(
            self._resolve_ref(leaderboard_id), token=self.token or None,
        )

    def list_submissions(self, leaderboard_id) -> list[dict]:
        """Return the leaderboard's submissions as a list of
        `{id, name, upload_date}` dicts (most recent first)."""
        payload = self.transport.get_leaderboard_submissions(
            self._resolve_ref(leaderboard_id), token=self.token or None,
        )
        return payload.get("submissions", [])

    def submission_exists(self, leaderboard_id, name: str) -> bool:
        """True if a submission named exactly `name` already exists on
        the leaderboard. Handy to guard a submit() against clobbering or
        duplicating a previous run:

            if client.submission_exists(lb_ref, "resnet50-v1"):
                raise SystemExit("already submitted — pick a new name")
        """
        payload = self.transport.get_leaderboard_submissions(
            self._resolve_ref(leaderboard_id), name=name, token=self.token or None,
        )
        return int(payload.get("count", 0)) > 0

    def _inputs_cache_dir(self, leaderboard_id: int) -> Path:
        """Local dir holding this LB's extracted input files. Defaults
        to ~/.cache/benchhub; override the root with BENCHHUB_CACHE_DIR
        (tests point it at a per-run tempdir for isolation)."""
        root = os.environ.get("BENCHHUB_CACHE_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "benchhub"
        )
        # Namespace by base_url host so two servers (prod vs local) with
        # colliding LB ids don't share a cache.
        host = self.base_url.replace("://", "_").replace("/", "_").replace(":", "_")
        return Path(root) / host / f"lb_{leaderboard_id}"

    def _ensure_inputs_cached(self, leaderboard_id: int, samples: list[dict],
                              cache_token: str, *,
                              force_download: bool) -> Path | None:
        """Make sure every file-backed input for `samples` is extracted
        under the cache dir. Downloads the bulk ZIP once when the cache
        is missing/stale/forced; returns the cache dir, or None if the
        bulk archive route is unavailable (caller falls back per-sample).

        Staleness is decided by a manifest carrying the server's
        `cache_token` (busts when a materialisation is rebuilt) plus the
        sorted `<field>/<sample>` entry list (busts if the subset
        changed). `force_download=True` always re-fetches."""
        cache_dir = self._inputs_cache_dir(leaderboard_id)
        expected = sorted(
            f"{fname}/{s['name']}"
            for s in samples
            for fname, entry in (s.get("inputs") or {}).items()
            if "url" in entry
        )
        manifest_path = cache_dir / ".manifest.json"
        if not force_download and manifest_path.is_file():
            try:
                have = json.loads(manifest_path.read_text())
                if (have.get("cache_token") == cache_token
                        and have.get("entries") == expected):
                    return cache_dir  # warm + valid
            except Exception:
                pass  # corrupt manifest → re-download

        # (Re)download. Wipe first so a stale subset can't linger.
        import shutil
        import tempfile
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_zip = tempfile.mkstemp(suffix=".zip", dir=str(cache_dir))
        os.close(tmp_fd)
        try:
            self.transport.download_inputs_archive(
                leaderboard_id, tmp_zip, token=self.token or None,
            )
        except BenchHubAPIError:
            # Older server with no /inputs.zip route, or a build error —
            # signal the caller to use the per-sample fallback.
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
            shutil.rmtree(cache_dir, ignore_errors=True)
            return None
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(cache_dir)
        try:
            os.remove(tmp_zip)
        except OSError:
            pass
        manifest_path.write_text(
            json.dumps({"cache_token": cache_token, "entries": expected})
        )
        return cache_dir

    @staticmethod
    def _index_cached_inputs(cache_dir: Path,
                             field_names: list[str]) -> dict[str, dict[str, Path]]:
        """Walk each field subdir once → {field: {sample_stem: path}}.
        O(files) up front so the per-sample lookup in the loop is O(1)
        instead of re-globbing 10k times."""
        index: dict[str, dict[str, Path]] = {}
        for fname in field_names:
            fdir = cache_dir / fname
            bucket: dict[str, Path] = {}
            if fdir.is_dir():
                for p in fdir.iterdir():
                    if p.is_file():
                        bucket[p.stem] = p
            index[fname] = bucket
        return index

    def iter_samples(self, leaderboard_id, *, force_download: bool = False):
        """Yield `(sample_name, inputs_dict)` for every sample on the
        leaderboard's bound dataset (or its materialised subset, if
        any). The dict is keyed by input-field name; values are
        already-decoded `bh.<Kind>` instances — the same type system
        used on the predict side, so a model that ingests
        `inputs['img'].array` (uint8 ndarray) round-trips cleanly into
        `bh.Image(arr)` outputs.

          * **image / mask / depth**:  bh.Image / bh.Mask / bh.Depth
          * **audio**:                  bh.Audio
          * **sequence**:               bh.Sequence — an iterable clip
            container; `for frame in inputs['clip']: frame.array …`
          * **bboxes / coco_detections**: bh.BBoxes / bh.CocoDetections
          * **scalar**:                 float
          * **label**:                  int / str (raw value)
          * **text / json / label_list**: decoded value (str / dict / list)

        File-backed inputs are downloaded ONCE as a single bulk ZIP and
        cached on disk (under ~/.cache/benchhub, or $BENCHHUB_CACHE_DIR),
        so a re-run — or a second submission to the same LB — reads from
        disk with no network. Pass `force_download=True` to bypass the
        cache and re-fetch (e.g. you suspect the local copy is corrupt).

        Designed for the generated submission notebook: the user
        plugs in their model, calls `sub.predict(name, …)`. No
        torchvision / HF datasets dependency.

        `leaderboard_id` accepts an id or an `<owner>/<slug>` handle.
        """
        leaderboard_id = self._resolve_ref(leaderboard_id)
        payload = self.transport.get_leaderboard_samples(
            leaderboard_id, token=self.token or None,
        )
        samples = payload.get("samples", [])
        cache_token = payload.get("cache_token") or ""

        # Which fields are file-backed (carry a 'url')? If none, every
        # input is inline — skip the archive download entirely.
        file_field_names = sorted({
            fname
            for s in samples
            for fname, entry in (s.get("inputs") or {}).items()
            if "url" in entry
        })

        cache_dir = None
        index: dict[str, dict[str, Path]] = {}
        if file_field_names:
            cache_dir = self._ensure_inputs_cached(
                leaderboard_id, samples, cache_token,
                force_download=force_download,
            )
            if cache_dir is not None:
                index = self._index_cached_inputs(cache_dir, file_field_names)

        for s in samples:
            inputs: dict[str, Any] = {}
            for field_name, entry in (s.get("inputs") or {}).items():
                kind = entry.get("kind") or ""
                params = entry.get("params") or {}
                if "url" not in entry:
                    inputs[field_name] = entry.get("value")
                    continue
                blob: bytes | None = None
                cached = index.get(field_name, {}).get(s["name"])
                if cached is not None:
                    blob = cached.read_bytes()
                elif cache_dir is None:
                    # Bulk archive unavailable → legacy per-sample fetch.
                    blob = self.transport.fetch_bytes(
                        entry["url"], token=self.token or None,
                    )
                inputs[field_name] = (
                    _decode_input_bytes(kind, blob, params)
                    if blob is not None else None
                )
            yield s["name"], inputs

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

    def create_metric(
        self,
        name: str,
        code,
        *,
        description: str | None = None,
        is_aggregated: bool = False,
        accepts_aggregated_inputs: bool = False,
        input_kinds=None,
        input_roles=None,
    ) -> dict:
        """Upload a metric to your library. `code` is either Python source
        (a string) or a function object (its source is read via
        `inspect.getsource`). `input_kinds` / `input_roles` auto-derive from
        the function's type annotations + arg names when omitted. Returns
        `{id, name, kind, visibility, input_kinds}`.

        Iterate locally first with `benchhub.author.test_metric(fn, ...)`."""
        return self._post_library_asset(
            "/api/metrics", name, code,
            description=description, is_aggregated=is_aggregated,
            accepts_aggregated_inputs=accepts_aggregated_inputs,
            input_kinds=input_kinds, input_roles=input_roles)

    def create_visualization(
        self,
        name: str,
        code,
        *,
        description: str | None = None,
        is_aggregated: bool = False,
        input_kinds=None,
    ) -> dict:
        """Upload a visualization to your library. `code` is source or a
        function returning a `PIL.Image`. Preview locally first with
        `benchhub.author.test_visualization(fn, ...)`."""
        return self._post_library_asset(
            "/api/visualizations", name, code,
            description=description, is_aggregated=is_aggregated,
            input_kinds=input_kinds)

    def create_datatype(
        self,
        name: str,
        *,
        file_ext: str | None = None,
        visualize_code=None,
        decode_code=None,
        viz_mime: str = "image/png",
        description: str | None = None,
    ) -> dict:
        """Register a new data type (kind). `visualize_code` is source or a
        ``def visualize(blob, params) -> PIL.Image`` function — it runs only
        in the sandbox. Optional `decode_code` is source or a
        ``def decode(blob, params) -> object`` function; when given, metrics
        on this kind receive the decoded object instead of the raw stored
        bytes (the deserialize side of the contract, also sandboxed).
        `file_ext` is the on-disk extension (e.g. ``'.nii.gz'``); None means
        inline. The kind name joins the global namespace (lowercase, unique).
        Returns ``{id, name, file_ext, visibility}``."""
        if not self.token:
            raise ValueError(
                "BenchHub Client has no API token — pass `token=...` or "
                "set BENCHHUB_API_TOKEN.")
        import inspect
        import textwrap
        if callable(visualize_code):
            visualize_code = textwrap.dedent(inspect.getsource(visualize_code))
        if callable(decode_code):
            decode_code = textwrap.dedent(inspect.getsource(decode_code))
        payload = {"name": name, "viz_mime": viz_mime}
        for k, v in {"file_ext": file_ext, "visualize_code": visualize_code,
                     "decode_code": decode_code,
                     "description": description}.items():
            if v is not None:
                payload[k] = v
        return self.transport.post_json("/api/datatypes", payload, self.token)

    def _post_library_asset(self, path, name, code, **fields) -> dict:
        if not self.token:
            raise ValueError(
                "BenchHub Client has no API token — pass `token=...` or "
                "set BENCHHUB_API_TOKEN.")
        if callable(code):
            import inspect
            import textwrap
            code = textwrap.dedent(inspect.getsource(code))
        payload = {"name": name, "python_code": code}
        for k, v in fields.items():
            if v is not None:
                payload[k] = v
        return self.transport.post_json(path, payload, self.token)

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
    def _post_zip(self, leaderboard_id: int, name: str | None, zip_bytes: bytes,
                  *, description: str | None = None,
                  link: str | None = None) -> dict:
        if not self.token:
            raise ValueError(
                "BenchHub Client has no API token — pass `token=...` or "
                "set BENCHHUB_API_TOKEN."
            )
        return self.transport.post_submission_zip(
            leaderboard_id, name, zip_bytes, self.token,
            description=description, link=link,
        )

    def _post_dataset_zip(self, zip_bytes: bytes, *, visibility: str) -> dict:
        if not self.token:
            raise ValueError(
                "BenchHub Client has no API token — pass `token=...` or "
                "set BENCHHUB_API_TOKEN."
            )
        return self.transport.post_dataset_zip(
            zip_bytes, self.token, visibility=visibility,
        )


class LeaderboardHandle:
    """A resolved leaderboard, returned by `Client.leaderboard(ref)`. Its
    methods mirror the id-taking `Client` methods but carry the id for you,
    so a script reads in terms of a human handle, never a raw number:

        lb = client.leaderboard("john-smith/ade20k-scene-parse-150")
        lb.id        # the numeric id, if you ever need it
        lb.ref       # "john-smith/ade20k-scene-parse-150"
    """

    def __init__(self, client: "Client", info: dict):
        self._client = client
        self.id = int(info["id"])
        self.name = info.get("name")
        self.slug = info.get("slug")
        self.owner = info.get("owner")
        self.ref = info.get("ref") or str(self.id)

    def __repr__(self) -> str:
        return f"<Leaderboard {self.ref!r} (id={self.id})>"

    def submission(self, name: str | None = None, *,
                   description: str | None = None,
                   link: str | None = None) -> "SubmissionBuilder":
        return self._client.submission(
            self.id, name, description=description, link=link)

    def iter_samples(self, *, force_download: bool = False):
        return self._client.iter_samples(self.id, force_download=force_download)

    def contract(self) -> list[dict]:
        return self._client.leaderboard_contract(self.id)

    def submissions(self) -> list[dict]:
        return self._client.list_submissions(self.id)

    def submission_exists(self, name: str) -> bool:
        return self._client.submission_exists(self.id, name)


class RawPrediction:
    """A prediction for a server-**registered** kind the standalone client
    has no `DataType` class for. You serialize your model output to bytes
    yourself (the producer owns serialization — exactly as a dataset author
    produces the GT file); the client packs those bytes **verbatim** under
    the kind's on-disk extension, and the server stores them as-is. A metric
    consuming the kind gets them back through the kind's `decode(blob,
    params)` hook (or raw, if none).

    `data` is bytes (or a file-like with `.read()`); use `from_file(...)` for
    a path. `file_ext` is the on-disk extension (e.g. ``'.nii.gz'``); when
    omitted the builder fills it from the LB contract entry for the field, so
    it matches what the server looks for.
    """

    def __init__(self, kind: str, data, *, file_ext: str | None = None,
                 params: dict | None = None):
        if hasattr(data, "read"):
            data = data.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                "RawPrediction data must be bytes (or a file-like / use "
                f"RawPrediction.from_file); got {type(data).__name__}")
        self.kind = str(kind)
        self.data = bytes(data)
        self.file_ext = file_ext
        self.params = params or {}

    @classmethod
    def from_file(cls, kind: str, path, *, file_ext: str | None = None,
                  params: dict | None = None) -> "RawPrediction":
        import os as _os
        with open(path, "rb") as fh:
            blob = fh.read()
        # Default the extension to the source file's own (e.g. .nii.gz → use
        # the full multi-suffix when present).
        if file_ext is None:
            name = _os.path.basename(str(path))
            dot = name.find(".")
            if dot > 0:
                file_ext = name[dot:]
        return cls(kind, blob, file_ext=file_ext, params=params)


class SubmissionBuilder:
    """Staging area for a typed submission.

    Predictions are accumulated in memory and packaged into a
    manifest.json + on-disk-layout ZIP at submit time. The kinds + params
    declared on each `DataType` instance flow straight into the manifest,
    so the server can validate against the LB contract without separate
    metadata.
    """

    def __init__(self, client: Client, leaderboard_id: int, name: str | None,
                 *, description: str | None = None, link: str | None = None):
        self.client = client
        self.leaderboard_id = leaderboard_id
        self.name = name
        self.description = description
        self.link = link
        # sample_name -> field_name -> DataType instance
        self._preds: dict[str, dict[str, DataType]] = {}
        # Optional LB contract (set via `set_contract` or
        # `fetch_contract`). When present + paired with per-sample
        # input shapes, build_zip enforces shape_match locally before
        # any upload.
        self._contract: list[dict] | None = None
        # sample_name -> input_field_name -> (H, W)
        self._input_shapes: dict[str, dict[str, tuple[int, int]]] = {}

    def predict(self, sample_name: str, **typed_predictions) -> None:
        """Stage one or more prediction values for a sample.

        Each value is either a `benchhub.DataType` (built-in kind, serialized
        via its `.encode()`) or a `benchhub.RawPrediction` (a registered kind
        whose bytes you serialized yourself, packed verbatim)."""
        if not typed_predictions:
            raise ValueError("predict() needs at least one field=instance kwarg")
        for field_name, inst in typed_predictions.items():
            if isinstance(inst, DataType):
                inst.validate()
            elif isinstance(inst, RawPrediction):
                pass  # opaque bytes; the server validates against the contract
            else:
                raise TypeError(
                    f"prediction {field_name!r} must be a benchhub.DataType or "
                    f"benchhub.RawPrediction; got {type(inst).__name__}"
                )
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
        kind_by_field: dict[str, str] = {}
        params_by_field: dict[str, dict] = {}
        for sample_preds in self._preds.values():
            for field_name, inst in sample_preds.items():
                # Both DataType subclasses and RawPrediction expose `.kind`.
                kind = inst.kind
                if field_name in kind_by_field and kind_by_field[field_name] != kind:
                    raise ValueError(
                        f"prediction {field_name!r} has mixed kinds across "
                        f"samples: {kind_by_field[field_name]} vs {kind}"
                    )
                kind_by_field[field_name] = kind
                # First-seen params wins — kind params are LB-level in
                # spirit, so they should be identical across samples.
                params_by_field.setdefault(field_name, inst.params)
        predictions = [
            {
                "name": field_name,
                "kind": kind_by_field[field_name],
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
        # Contract carries file_ext per pred field — authoritative for
        # registered kinds (the server looks for that exact extension).
        contract_ext = {}
        if self._contract:
            contract_ext = {c["name"]: c.get("file_ext")
                            for c in self._contract if isinstance(c, dict)}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            for p in manifest["predictions"]:
                cls = DTYPES.get(p["kind"])
                rep = next(
                    (self._preds[s].get(p["name"]) for s in manifest["samples"]
                     if self._preds[s].get(p["name"]) is not None), None)
                if cls is not None:
                    ext = cls.file_ext or ".txt"
                else:
                    # Registered kind: prefer the contract's ext, then a
                    # RawPrediction-supplied one. Without either we can't be
                    # sure the server will find the file.
                    ext = (contract_ext.get(p["name"])
                           or (rep.file_ext if isinstance(rep, RawPrediction) else None))
                    if not ext:
                        raise ValueError(
                            f"prediction {p['name']!r} is a registered kind "
                            f"{p['kind']!r} with no file extension — call "
                            f"fetch_contract()/set_contract(...) or pass "
                            f"RawPrediction(..., file_ext='.ext')."
                        )
                for sample_name in manifest["samples"]:
                    inst = self._preds[sample_name].get(p["name"])
                    if inst is None:
                        raise ValueError(
                            f"sample {sample_name!r} missing prediction "
                            f"for field {p['name']!r}"
                        )
                    payload = (inst.data if isinstance(inst, RawPrediction)
                               else inst.encode())
                    zf.writestr(f"{p['name']}/{sample_name}{ext}", payload)
        return buf.getvalue()

    def submit(self, name: str | None = None, *,
               description: str | None = None,
               link: str | None = None) -> dict:
        """Build the ZIP and POST it. Returns the server's response payload.

        `name` is the submission's display name. `description` is a
        free-text blurb (shown in the LB submission table, full text on
        hover) and `link` an optional http(s) URL the submission name
        links to. Each overrides the value passed to
        `client.submission(lb_id, ...)`; pass them here if you opened
        the builder without them (which is what the generated submission
        script + LB-page snippet do)."""
        return self.client._post_zip(
            self.leaderboard_id, name or self.name, self.build_zip(),
            description=description or self.description,
            link=link or self.link,
        )


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

    def get_leaderboard_contract(self, leaderboard_id: int,
                                 token: str | None = None) -> list[dict]:
        # Test transport ignores token — Flask-test-client routes
        # don't run @require_api_token in the same way; tests stub
        # auth directly.
        resp = self.test_client.get(f"/api/leaderboard/{leaderboard_id}/contract")
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload or {})
        return payload if isinstance(payload, list) else []

    def resolve_leaderboard(self, ref: str,
                            token: str | None = None) -> dict:
        from urllib.parse import quote
        resp = self.test_client.get(f"/api/leaderboard/resolve?ref={quote(str(ref))}")
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload or {})
        return payload if isinstance(payload, dict) else {}

    def get_leaderboard_samples(self, leaderboard_id: int,
                                token: str | None = None) -> dict:
        resp = self.test_client.get(f"/api/leaderboard/{leaderboard_id}/samples")
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload or {})
        return payload if isinstance(payload, dict) else {}

    def get_leaderboard_submissions(self, leaderboard_id: int,
                                    name: str | None = None,
                                    token: str | None = None) -> dict:
        url = f"/api/leaderboard/{leaderboard_id}/submissions"
        if name:
            from urllib.parse import quote
            url += f"?name={quote(name)}"
        resp = self.test_client.get(url)
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, payload or {})
        return payload if isinstance(payload, dict) else {}

    def fetch_bytes(self, url: str, token: str | None = None) -> bytes:
        # Forward the token (mirrors the real requests transport) so auth-gated
        # data endpoints accept the in-process test client too.
        resp = self.test_client.get(
            url, headers=({"Authorization": f"Bearer {token}"} if token else {}))
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code,
                                    {"error": resp.data.decode("utf-8", "replace")})
        return resp.data

    def download_inputs_archive(self, leaderboard_id: int, dest_path: str,
                                token: str | None = None) -> None:
        resp = self.test_client.get(
            f"/api/leaderboard/{leaderboard_id}/inputs.zip",
            headers=({"Authorization": f"Bearer {token}"} if token else {}),
        )
        if resp.status_code >= 400:
            raise BenchHubAPIError(
                resp.status_code,
                {"error": resp.data.decode("utf-8", "replace")},
            )
        with open(dest_path, "wb") as f:
            f.write(resp.data)

    def post_submission_zip(self, leaderboard_id: int, name: str | None,
                            zip_bytes: bytes, token: str,
                            *, description: str | None = None,
                            link: str | None = None) -> dict:
        from io import BytesIO
        resp = self.test_client.post(
            f"/api/submit/{leaderboard_id}",
            data={
                "submission_zip": (BytesIO(zip_bytes), "submission.zip"),
                **({"name": name} if name else {}),
                **({"description": description} if description else {}),
                **({"link": link} if link else {}),
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

    def post_json(self, path: str, payload: dict, token: str) -> dict:
        resp = self.test_client.post(
            path, json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            body = resp.get_json() or {}
        except Exception:
            body = {"error": resp.data.decode("utf-8", "replace")}
        if resp.status_code >= 400:
            raise BenchHubAPIError(resp.status_code, body)
        return body

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
