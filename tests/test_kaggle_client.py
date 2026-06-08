"""Offline tests for benchhub.kaggle_client + benchhub.kaggle_search.

A fake requests-like session is injected, so nothing hits the network. The
license classifier (the legal gate) is tested hard — it decides what may be
re-served to other users."""
import io
import json
import zipfile

import pytest

from benchhub import kaggle_client as kc
from benchhub import kaggle_search as ks
from benchhub.kaggle_client import (
    KaggleClient, KaggleAuthError, KaggleAPIError, classify_license,
    license_name_from_view, split_ref, resolve_credentials,
    resolve_access_token,
)


# --------------------------------------------------------------------------
# Fake transport
# --------------------------------------------------------------------------

class FakeResp:
    def __init__(self, *, status_code=200, json_data=None, headers=None,
                 content=b""):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self._content = content

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]


class FakeSession:
    def __init__(self, responses):
        # responses: a list (popped in order) or a callable(method,url,kw)->resp
        self.responses = responses
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        if callable(self.responses):
            return self.responses(method, url, kw)
        return self.responses.pop(0)


def _client(responses, **kw):
    kw.setdefault("username", "u")
    kw.setdefault("key", "k")
    kw.setdefault("sleeper", lambda s: None)        # never really sleep
    return KaggleClient(session=FakeSession(responses), **kw)


# --------------------------------------------------------------------------
# ref parsing
# --------------------------------------------------------------------------

def test_split_ref_plain():
    assert split_ref("owner/slug") == ("owner", "slug", None)


def test_split_ref_version():
    assert split_ref("owner/slug/versions/3") == ("owner", "slug", 3)


def test_split_ref_url():
    assert split_ref("https://www.kaggle.com/datasets/owner/slug") == \
        ("owner", "slug", None)


def test_split_ref_bad():
    with pytest.raises(ValueError):
        split_ref("not-a-ref")


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def test_resolve_credentials_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    assert resolve_credentials() == ("alice", "secret")


def test_resolve_credentials_json(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "bob",
                                                       "key": "tok"}))
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert resolve_credentials() == ("bob", "tok")


def test_no_credentials_raises(monkeypatch):
    # Stub BOTH resolvers — otherwise an ambient ~/.kaggle/access_token on
    # the dev box would make this client look authenticated.
    monkeypatch.setattr(kc, "resolve_credentials", lambda: None)
    monkeypatch.setattr(kc, "resolve_access_token", lambda: None)
    client = KaggleClient(session=FakeSession([]))
    assert client.has_credentials is False
    with pytest.raises(KaggleAuthError):
        client.view("owner/slug")


# --------------------------------------------------------------------------
# access-token (KGAT Bearer) auth
# --------------------------------------------------------------------------

def test_resolve_access_token_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_envtoken")
    assert resolve_access_token() == "KGAT_envtoken"


def test_resolve_access_token_file(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_ACCESS_TOKEN", raising=False)
    (tmp_path / "access_token").write_text("KGAT_filetoken\n")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert resolve_access_token() == "KGAT_filetoken"


def test_token_sends_bearer_header_not_basic(monkeypatch):
    monkeypatch.setattr(kc, "resolve_credentials", lambda: None)
    monkeypatch.setattr(kc, "resolve_access_token", lambda: None)
    captured = {}

    def _respond(method, url, kw):
        captured.update(kw)
        return FakeResp(json_data={"ok": True})

    client = KaggleClient(session=FakeSession(_respond), token="KGAT_abc",
                          sleeper=lambda s: None)
    assert client.has_credentials is True
    client.view("owner/slug")
    assert captured["headers"]["Authorization"] == "Bearer KGAT_abc"
    assert captured.get("auth") is None  # Bearer, not Basic


def test_from_env_prefers_token_over_basic(monkeypatch):
    monkeypatch.setattr(kc, "resolve_access_token", lambda: "KGAT_xyz")
    monkeypatch.setattr(kc, "resolve_credentials", lambda: ("u", "k"))
    client = KaggleClient.from_env(session=FakeSession([]))
    assert client._token == "KGAT_xyz"
    assert client._auth() is None  # token suppresses Basic


# --------------------------------------------------------------------------
# license classification — the legal gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "CC0-1.0", "CC0: Public Domain", "CC-BY-4.0", "CC-BY-SA-4.0",
    "ODbL-1.0", "ODC-BY-1.0", "PDDL", "DbCL-1.0", "Apache 2.0", "MIT",
    "BSD-3-Clause", "CDLA-Permissive-1.0", "U.S. Government Works",
    "Database: Open Database, Contents: Database Contents",
])
def test_license_redistributable(name):
    assert classify_license(name)["redistributable"] is True


@pytest.mark.parametrize("name", [
    "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-ND-4.0", "CC-BY-NC-ND-4.0",
    "GPL-2.0", "GPL 3.0", "AGPL-3.0", "Other (specified in description)",
    "Other", "Unknown", "Data files © Original Authors",
    "World Bank Dataset Terms of Use", "Subject to Reddit API Terms", "",
])
def test_license_restricted(name):
    assert classify_license(name)["redistributable"] is False


def test_license_nc_beats_attribution():
    # 'CC-BY-NC' must NOT be caught by the 'cc-by' redistributable rule.
    v = classify_license("CC-BY-NC-2.0")
    assert v["redistributable"] is False and v["category"] == "non-commercial"


def test_license_name_from_view_variants():
    assert license_name_from_view({"licenseName": "MIT"}) == "MIT"
    assert license_name_from_view(
        {"licenses": [{"name": "CC0-1.0"}]}) == "CC0-1.0"
    assert license_name_from_view({}) == ""


# --------------------------------------------------------------------------
# client HTTP behaviours
# --------------------------------------------------------------------------

def test_view_parses_json():
    client = _client([FakeResp(json_data={"title": "T", "licenseName": "MIT"})])
    assert client.view("o/s")["title"] == "T"


def test_list_files_paginates():
    client = _client([
        FakeResp(json_data={"datasetFiles": [{"name": "a.csv"}],
                            "nextPageToken": "p2"}),
        FakeResp(json_data={"datasetFiles": [{"name": "b.csv"}]}),
    ])
    names = client.file_names("o/s")
    assert names == ["a.csv", "b.csv"]


def test_429_then_success_backs_off():
    slept = []
    client = _client(
        [FakeResp(status_code=429, headers={"Retry-After": "2"}),
         FakeResp(json_data={"ok": 1})],
        sleeper=lambda s: slept.append(s))
    assert client._get_json("/x") == {"ok": 1}
    assert slept == [2.0]                       # honoured Retry-After


def test_401_raises_auth():
    client = _client([FakeResp(status_code=401)])
    with pytest.raises(KaggleAuthError):
        client.view("o/s")


def test_500_exhausts_retries_then_errors():
    client = _client([FakeResp(status_code=500)] * 10, max_retries=2)
    with pytest.raises(KaggleAPIError):
        client._get_json("/x")


def test_download_extracts_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("train.csv", "a,b\n1,2\n")
        z.writestr("imgs/0.png", b"\x89PNG fake")
    client = _client([FakeResp(content=buf.getvalue())])
    cache_dir = client.download("owner/slug", str(tmp_path))
    assert cache_dir.endswith("owner__slug__vlatest")
    fetch = KaggleClient.fetch_factory(cache_dir)
    with open(fetch("train.csv")) as fh:
        assert fh.read() == "a,b\n1,2\n"


def test_download_reuses_cache(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "1")
    client = _client([FakeResp(content=buf.getvalue())])
    d1 = client.download("o/s", str(tmp_path))
    # Second call: no responses left → if it tried to download it'd IndexError.
    d2 = client.download("o/s", str(tmp_path))
    assert d1 == d2


def test_search_unwraps_list():
    client = _client([FakeResp(json_data=[{"ref": "o/s", "title": "T"}])])
    rows = client.search("cats")
    assert rows[0]["ref"] == "o/s"


# --------------------------------------------------------------------------
# kaggle_search
# --------------------------------------------------------------------------

def test_search_datasets_normalizes():
    ks._clear_cache()
    client = _client([FakeResp(json_data=[
        {"ref": "o/s", "title": "Cats", "totalBytes": 1000,
         "voteCount": 5, "licenseName": "CC-BY-NC-4.0"}])])
    rows = ks.search_datasets(client, "cats")
    assert rows[0]["ref"] == "o/s"
    assert rows[0]["total_bytes"] == 1000
    assert rows[0]["redistributable"] is False     # NC

def test_card_summary_caches():
    ks._clear_cache()
    session = FakeSession([FakeResp(json_data={
        "title": "T", "subtitle": "sub", "totalBytes": 10,
        "licenseName": "MIT", "currentVersionNumber": 2})])
    client = KaggleClient(session=session, username="u", key="k",
                          sleeper=lambda s: None)
    c1 = ks.card_summary(client, "o/s")
    c2 = ks.card_summary(client, "o/s")           # served from cache
    assert c1 == c2
    assert c1["redistributable"] is True and c1["version"] == 2
    assert len(session.calls) == 1                # only one network call


def test_trending_degrades_on_error():
    ks._clear_cache()

    def boom(method, url, kw):
        return FakeResp(status_code=500)

    client = KaggleClient(session=FakeSession(boom), username="u", key="k",
                          sleeper=lambda s: None, max_retries=0)
    out = ks.trending_by_domain(client, limit_per_domain=3)
    assert set(out.keys()) == {"Vision", "NLP", "Audio", "Tabular"}
    assert all(v == [] for v in out.values())     # all failed → empty, no raise
