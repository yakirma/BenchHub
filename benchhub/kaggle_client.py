"""Kaggle dataset adapter — a thin REST wrapper over the public Kaggle API.

We deliberately do NOT depend on the `kaggle` pip package (it is often
absent, needs creds at import time, and its surface drifts). Instead this
talks to `https://www.kaggle.com/api/v1` with `requests` (already a dep).
Two auth schemes, preferred in this order: a Bearer **access token** (the
newer `KGAT…` token the kaggle CLI v1.8.0+ writes to ~/.kaggle/access_token
or $KAGGLE_API_TOKEN), else HTTP-Basic with a username + key (kaggle.json /
$KAGGLE_USERNAME+$KAGGLE_KEY). Auth is resolved lazily so importing this
module never fails without creds — only a *call* that needs the network.

The source-specific surface BenchHub's import engine needs is tiny (see
`docs/KAGGLE_IMPORT_PLAN.md`): `list_files`, a whole-zip `download` + a
`fetch(relpath) -> localpath` factory, dataset `view` metadata, and a pure
license classifier. Everything downstream (`file_tree_import`,
`manifest.import_typed_dataset`, the preview tier) is source-agnostic.

REST paths were VERIFIED against the live API (Bearer token, 2026-06-08):
`/datasets/list`, `/datasets/view/{o}/{s}`, `/datasets/list/{o}/{s}` (files,
returns `datasetFiles`), `/datasets/download/{o}/{s}` all return 200; the
files endpoint is `/datasets/list/...` NOT `/datasets/files/...` (404). Kept
as module constants so any future swagger drift is a one-place fix.
Everything (auth, backoff, parsing, license logic) is also exercised by
tests/test_kaggle_client.py with an injected fake session — no live calls.
"""
from __future__ import annotations

import json
import os
import re
import time
import zipfile

API_BASE = "https://www.kaggle.com/api/v1"
USER_AGENT = "benchhub/0.1"

# Endpoint templates (verified live 2026-06-08 — see module docstring).
EP_LIST = "/datasets/list"
EP_VIEW = "/datasets/view/{owner}/{slug}"
EP_FILES = "/datasets/list/{owner}/{slug}"
EP_DOWNLOAD_ALL = "/datasets/download/{owner}/{slug}"
EP_DOWNLOAD_FILE = "/datasets/download/{owner}/{slug}/{path}"


class KaggleAuthError(RuntimeError):
    """Raised when a network call needs credentials and none resolve."""


class KaggleAPIError(RuntimeError):
    """A non-retryable HTTP error from the Kaggle API."""


# --------------------------------------------------------------------------
# Credentials + ref parsing (pure)
# --------------------------------------------------------------------------

def resolve_credentials():
    """(username, key) from the environment or ~/.kaggle/kaggle.json, or
    None. Order: KAGGLE_USERNAME/KAGGLE_KEY env, then a kaggle.json under
    $KAGGLE_CONFIG_DIR or ~/.kaggle. Never raises."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    for d in (os.environ.get("KAGGLE_CONFIG_DIR"),
              os.path.join(os.path.expanduser("~"), ".kaggle")):
        if not d:
            continue
        path = os.path.join(d, "kaggle.json")
        try:
            with open(path) as fh:
                doc = json.load(fh)
            if doc.get("username") and doc.get("key"):
                return doc["username"], doc["key"]
        except (OSError, ValueError, KeyError):
            continue
    return None


def resolve_access_token():
    """A Kaggle API *access token* (the newer `KGAT…` Bearer token), or
    None. Order: KAGGLE_API_TOKEN / KAGGLE_ACCESS_TOKEN env, then an
    `access_token` file under $KAGGLE_CONFIG_DIR or ~/.kaggle. The kaggle
    CLI (v1.8.0+) writes the file; it's sent as `Authorization: Bearer`,
    not Basic auth. Never raises."""
    tok = (os.environ.get("KAGGLE_API_TOKEN")
           or os.environ.get("KAGGLE_ACCESS_TOKEN"))
    if tok and tok.strip():
        return tok.strip()
    for d in (os.environ.get("KAGGLE_CONFIG_DIR"),
              os.path.join(os.path.expanduser("~"), ".kaggle")):
        if not d:
            continue
        try:
            with open(os.path.join(d, "access_token")) as fh:
                tok = fh.read().strip()
            if tok:
                return tok
        except OSError:
            continue
    return None


_REF_RE = re.compile(
    r"^(?:https?://(?:www\.)?kaggle\.com/(?:datasets/)?)?"
    r"(?P<owner>[^/\s]+)/(?P<slug>[^/\s?]+)"
    r"(?:/versions/(?P<version>\d+))?")


def split_ref(ref):
    """A Kaggle dataset ref → (owner, slug, version|None). Accepts
    'owner/slug', 'owner/slug/versions/3', or a kaggle.com URL."""
    m = _REF_RE.match((ref or "").strip())
    if not m:
        raise ValueError(f"not a Kaggle dataset ref: {ref!r}")
    v = m.group("version")
    return m.group("owner"), m.group("slug"), (int(v) if v else None)


# --------------------------------------------------------------------------
# License classification (pure) — the legal gate (plan §8, STRICT policy)
# --------------------------------------------------------------------------

# Substrings that, alone, make a license redistributable (we may cache the
# bytes and re-serve them to other users). Checked AFTER the NC/ND/copyleft
# vetoes below so e.g. 'cc-by-nc' isn't caught by the 'cc-by' rule.
_REDISTRIBUTABLE_HINTS = (
    "cc0", "publicdomain", "public domain", "pddl",
    "cc-by-sa", "ccbysa", "cc-by", "ccby",
    "odbl", "open database", "odc-by", "odc by",
    "dbcl", "database contents",
    "cdla",                                   # CDLA permissive & sharing
    "apache", "mit", "bsd",
    "u.s. government", "us government", "government works",
)
# Vetoes — never redistributable under the STRICT policy.
_RESTRICTED_HINTS = (
    "reddit", "world bank", "original author", "©", "(c) original",
    "other", "unknown", "proprietary", "all rights reserved",
)


def classify_license(name):
    """Map a Kaggle license name → a redistribution verdict.

    Returns {'name', 'redistributable': bool, 'category'}. Policy is STRICT
    (plan §8): only clearly-redistributable licenses pass; non-commercial
    (NC), no-derivatives (ND), GPL-family copyleft, 'Other', 'Unknown', and
    anything unrecognised are restricted (fail-safe → False)."""
    raw = (name or "").strip()
    low = raw.lower()
    norm = re.sub(r"[^a-z0-9]+", "-", low).strip("-")
    toks = set(norm.split("-"))

    def _verdict(cat, ok):
        return {"name": raw, "redistributable": bool(ok), "category": cat}

    if not raw:
        return _verdict("unknown", False)
    # Non-commercial / no-derivatives vetoes (check before attribution).
    if "nc" in toks or "noncommercial" in low or "non-commercial" in low:
        return _verdict("non-commercial", False)
    if "nd" in toks or "noderivatives" in low or "no-derivatives" in low:
        return _verdict("no-derivatives", False)
    # GPL-family copyleft carries source/use conditions → restricted.
    if any(t in toks for t in ("gpl", "agpl", "lgpl", "gplv2", "gplv3")):
        return _verdict("copyleft", False)
    # Explicit restricted markers.
    if any(h in low for h in _RESTRICTED_HINTS):
        return _verdict("restricted", False)
    # Redistributable families.
    if any(h in low for h in _REDISTRIBUTABLE_HINTS):
        cat = ("public-domain" if ("cc0" in low or "pddl" in low
                                   or "public" in low) else "permissive")
        return _verdict(cat, True)
    return _verdict("unknown", False)       # fail-safe


def license_name_from_view(view):
    """Extract a license-name string from a /datasets/view document, which
    may spell it as `licenseName`, a `licenses[].name`, or `licenseShortName`."""
    if not isinstance(view, dict):
        return ""
    for k in ("licenseName", "licenseShortName", "license_name"):
        if view.get(k):
            return str(view[k])
    lic = view.get("licenses")
    if isinstance(lic, list) and lic and isinstance(lic[0], dict):
        return str(lic[0].get("name") or lic[0].get("nameNullable") or "")
    return ""


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------

class KaggleClient:
    """REST client for the Kaggle dataset API. Inject a `session`
    (requests-like: `.request(method, url, **kw) -> resp`) for tests; in
    prod it lazily builds a `requests.Session`."""

    def __init__(self, *, username=None, key=None, token=None, session=None,
                 base=API_BASE, timeout=60, max_retries=4,
                 backoff_initial=1.0, sleeper=time.sleep):
        self._username = username
        self._key = key
        self._token = token
        # Explicit constructor creds take precedence over ambient
        # env/files, so don't overwrite them during lazy resolution.
        self._explicit_basic = bool(username and key)
        self._auth_resolved = bool(token) or self._explicit_basic
        self._session = session
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self._sleep = sleeper

    @classmethod
    def from_env(cls, **kw):
        # Prefer a Bearer access token; fall back to a username/key pair.
        if "token" not in kw:
            tok = resolve_access_token()
            if tok:
                kw["token"] = tok
        if not kw.get("token"):
            creds = resolve_credentials()
            if creds:
                kw.setdefault("username", creds[0])
                kw.setdefault("key", creds[1])
        return cls(**kw)

    @property
    def has_credentials(self):
        self._ensure_resolved()
        return bool(self._token or (self._username and self._key))

    def _ensure_resolved(self):
        """Lazily fill in ambient credentials once. A Bearer access token
        wins over a Basic username/key pair; explicit constructor values
        win over both."""
        if self._auth_resolved:
            return
        if not self._token and not self._explicit_basic:
            self._token = resolve_access_token()
        if not self._token and not (self._username and self._key):
            creds = resolve_credentials()
            if creds:
                self._username, self._key = creds
        self._auth_resolved = True

    def _auth(self):
        """Basic-auth (username, key) tuple for `requests`, or None. When a
        Bearer token is in play it's applied as a header in `_request`, so
        this returns None to keep the two schemes from colliding."""
        self._ensure_resolved()
        if self._token:
            return None
        return (self._username, self._key) if (self._username and self._key) else None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _request(self, path, *, params=None, stream=False, require_auth=True):
        self._ensure_resolved()
        auth = self._auth()  # Basic tuple, or None when a token is used
        headers = {"User-Agent": USER_AGENT}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if require_auth and not (self._token or auth):
            raise KaggleAuthError(
                "No Kaggle credentials — set KAGGLE_API_TOKEN (or place an "
                "access_token / kaggle.json under ~/.kaggle), or set "
                "KAGGLE_USERNAME/KAGGLE_KEY.")
        url = self.base + path
        delay = self.backoff_initial
        resp = None
        for attempt in range(self.max_retries + 1):
            resp = self._get_session().request(
                "GET", url, params=params, auth=auth,
                headers=headers, timeout=self.timeout,
                stream=stream)
            code = getattr(resp, "status_code", 200)
            if code == 429 or 500 <= code < 600:
                if attempt < self.max_retries:
                    ra = (resp.headers or {}).get("Retry-After")
                    try:
                        wait = float(ra) if ra else delay
                    except (TypeError, ValueError):
                        wait = delay
                    self._sleep(wait)
                    delay *= 2
                    continue
            return resp
        return resp

    def _get_json(self, path, *, params=None):
        resp = self._request(path, params=params)
        code = getattr(resp, "status_code", 200)
        if code == 401 or code == 403:
            raise KaggleAuthError(f"Kaggle API {code} for {path} — bad or "
                                  f"missing credentials.")
        if code >= 400:
            raise KaggleAPIError(f"Kaggle API {code} for {path}")
        return resp.json()

    # -- metadata ----------------------------------------------------------

    def view(self, ref):
        """Dataset metadata document (title, subtitle, description,
        license, usabilityRating, totalBytes, version count, …)."""
        owner, slug, _v = split_ref(ref)
        return self._get_json(EP_VIEW.format(owner=owner, slug=slug))

    def list_files(self, ref):
        """All files in a dataset → list of {name, totalBytes, ...}.
        Follows `nextPageToken` pagination so huge datasets list fully."""
        owner, slug, version = split_ref(ref)
        out, token, guard = [], None, 0
        while True:
            params = {}
            if version:
                params["datasetVersionNumber"] = version
            if token:
                params["pageToken"] = token
            doc = self._get_json(EP_FILES.format(owner=owner, slug=slug),
                                 params=params or None)
            files = (doc.get("datasetFiles") or doc.get("files")
                     or doc.get("datasetFilesList") or [])
            out.extend(f for f in files if isinstance(f, dict))
            token = doc.get("nextPageToken") or doc.get("nextPageTokenNullable")
            guard += 1
            if not token or guard > 1000:
                break
        return out

    def file_names(self, ref):
        """Just the repo-relative file-name strings (feeds inspect_repo /
        detect_shape)."""
        return [str(f.get("name") or f.get("nameNullable") or "")
                for f in self.list_files(ref)
                if (f.get("name") or f.get("nameNullable"))]

    # -- download ----------------------------------------------------------

    def download(self, ref, cache_root, *, progress_cb=None, chunk=1 << 20):
        """Whole-zip-once: download the dataset archive and extract it into
        a cache dir keyed by owner__slug__vN (deduped — a populated cache
        dir is reused). Returns the extracted directory path."""
        owner, slug, version = split_ref(ref)
        key = f"{owner}__{slug}__v{version if version else 'latest'}"
        cache_dir = os.path.join(cache_root, key)
        if os.path.isdir(cache_dir) and os.listdir(cache_dir):
            return cache_dir
        os.makedirs(cache_root, exist_ok=True)
        params = {"datasetVersionNumber": version} if version else None
        resp = self._request(EP_DOWNLOAD_ALL.format(owner=owner, slug=slug),
                             params=params, stream=True)
        code = getattr(resp, "status_code", 200)
        if code >= 400:
            raise KaggleAPIError(f"Kaggle download {code} for {ref}")
        tmp_zip = cache_dir + ".download.zip"
        n = 0
        with open(tmp_zip, "wb") as fh:
            for piece in resp.iter_content(chunk_size=chunk):
                if piece:
                    fh.write(piece)
                    n += len(piece)
                    if progress_cb:
                        progress_cb(n)
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(tmp_zip) as z:
                z.extractall(cache_dir)
        finally:
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
        # Some Kaggle single-file datasets ship as <name>.csv.zip members;
        # leave them — file_tree's gz/zip loaders / detect handle nesting.
        return cache_dir

    @staticmethod
    def fetch_factory(cache_dir):
        """`fetch(relpath) -> absolute local path` over an extracted cache
        dir — the only source-specific callable materialize_file_tree needs."""
        def _fetch(rel):
            return os.path.join(cache_dir, rel)
        return _fetch

    # -- search ------------------------------------------------------------

    def search(self, query="", *, sort_by=None, file_type=None,
               license_name=None, page=1, page_size=20, tags=None,
               min_size=None, max_size=None, user=None):
        """Raw /datasets/list search → list of dataset dicts. Filters mirror
        the Kaggle API params; all optional."""
        params = {"page": int(page), "pageSize": int(page_size)}
        if query:
            params["search"] = query
        if sort_by:
            params["sortBy"] = sort_by          # hottest|votes|updated|active|published
        if file_type:
            params["fileType"] = file_type      # csv|json|sqlite|bigQuery|parquet|all
        if license_name:
            params["license"] = license_name     # cc|gpl|odb|other|all
        if tags:
            params["tagids"] = tags
        if min_size is not None:
            params["minSize"] = int(min_size)
        if max_size is not None:
            params["maxSize"] = int(max_size)
        if user:
            params["user"] = user
        doc = self._get_json(EP_LIST, params=params)
        if isinstance(doc, dict):
            doc = doc.get("datasets") or doc.get("results") or []
        return [d for d in doc if isinstance(d, dict)]
