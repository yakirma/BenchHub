"""Tests for app.handle_dlp_safe_code.

Decodes user metric/visualization code that was Base64-encoded client-side to
bypass DLP filters that block .py uploads. The contract is loose: bad input
should return-as-is, never raise.
"""
import base64

import pytest

from app import handle_dlp_safe_code


def b64(s: str) -> str:
    return "BASE64:" + base64.b64encode(s.encode("utf-8")).decode("ascii")


def test_passthrough_when_no_prefix_but_strips_outer_whitespace():
    # Quirk: handle_dlp_safe_code() unconditionally calls .strip() on the input,
    # so even non-prefixed code loses leading/trailing whitespace.
    code = "  def m(x):\n    return x\n"
    assert handle_dlp_safe_code(code) == code.strip()


def test_decodes_base64_prefix():
    original = "def metric(a, b):\n    return a + b\n"
    assert handle_dlp_safe_code(b64(original)) == original


def test_decodes_with_surrounding_whitespace():
    # Frontend may add trailing newlines; the function strips before checking the prefix.
    original = "def m():\n    return 1\n"
    assert handle_dlp_safe_code("\n  " + b64(original) + "  \n") == original


def test_falls_back_to_latin1_for_non_utf8():
    # Build a payload that's invalid UTF-8 but valid latin-1 (a stray 0xff byte).
    raw_bytes = b"def m():\n    return '\xff'\n"
    encoded = "BASE64:" + base64.b64encode(raw_bytes).decode("ascii")
    out = handle_dlp_safe_code(encoded)
    # latin-1 decoding of 0xff is "ÿ"; assert we got a string back, not an exception.
    assert "def m()" in out
    assert "\xff" in out or "ÿ" in out


def test_malformed_base64_returns_original_string():
    # Garbage after BASE64: prefix → b64decode raises → function logs and returns input as-is.
    bad = "BASE64:!!!not-valid-base64!!!"
    assert handle_dlp_safe_code(bad) == bad


def test_empty_string_returns_empty():
    assert handle_dlp_safe_code("") == ""


def test_none_returned_unchanged():
    # Guard at top: `if not code_str: return code_str`.
    assert handle_dlp_safe_code(None) is None


def test_prefix_alone_with_no_payload_treated_as_empty():
    # "BASE64:" with nothing after → b64decode("") returns b"" → decodes to "".
    assert handle_dlp_safe_code("BASE64:") == ""


@pytest.mark.parametrize(
    "fake_prefix",
    ["base64:", "Base64:", "BASE_64:", "BASE-64:"],
)
def test_only_exact_prefix_triggers_decode(fake_prefix):
    # Case-sensitive, exact match. Anything else passes through verbatim.
    payload = fake_prefix + "ZGVmIG0oKTogcmV0dXJuIDE="  # b64("def m(): return 1")
    assert handle_dlp_safe_code(payload) == payload
