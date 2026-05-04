"""Tests for app.apply_tag_filters.

The CSV-tag string format is fragile: tags live as a single comma-separated
string in `Sample.tags`. The filter must match exact tags regardless of
position (only, first, middle, last) and not be confused by tag prefixes
that share substrings.
"""
import pytest

from app import Dataset, Sample, apply_tag_filters, db


@pytest.fixture
def samples(app, db_session):
    """Build a small dataset with samples that exercise every tag-position case."""
    ds = Dataset(name="tag_test_ds")
    db_session.add(ds)
    db_session.flush()

    # Tag positions: only / first / middle / last / none / substring trap.
    fixtures = [
        ("s_only_a", "tag_a"),
        ("s_first_a", "tag_a,tag_b"),
        ("s_middle_a", "tag_b,tag_a,tag_c"),
        ("s_last_a", "tag_b,tag_a"),
        ("s_no_a", "tag_b,tag_c"),
        ("s_substring_trap", "tag_aa"),  # must NOT match "tag_a"
        ("s_empty", ""),
        ("s_prefix_x", "x_one"),
        ("s_prefix_x_first", "x_two,other"),
        ("s_no_prefix_x", "other,unrelated"),
    ]
    for name, tags in fixtures:
        db_session.add(Sample(dataset_id=ds.id, name=name, tags=tags))
    db_session.commit()
    return ds


def names(query):
    return sorted(s.name for s in query.all())


# ---------------------------------------------------------------------------
# Pass-through: filters disabled
# ---------------------------------------------------------------------------


def test_no_filters_returns_all(samples):
    q = apply_tag_filters(Sample.query, {})
    assert len(names(q)) == 10


def test_disabled_flags_ignore_provided_tags(samples):
    args = {"enable_include": "false", "include_tags": "tag_a"}
    q = apply_tag_filters(Sample.query, args)
    assert len(names(q)) == 10


# ---------------------------------------------------------------------------
# Include (AND)
# ---------------------------------------------------------------------------


def test_include_single_tag_matches_all_positions(samples):
    args = {"enable_include": "true", "include_tags": "tag_a"}
    assert names(apply_tag_filters(Sample.query, args)) == [
        "s_first_a",
        "s_last_a",
        "s_middle_a",
        "s_only_a",
    ]


def test_include_does_not_substring_match(samples):
    # s_substring_trap has "tag_aa" — must not match include "tag_a".
    args = {"enable_include": "true", "include_tags": "tag_a"}
    assert "s_substring_trap" not in names(apply_tag_filters(Sample.query, args))


def test_include_multiple_tags_uses_AND(samples):
    args = {"enable_include": "true", "include_tags": "tag_a,tag_b"}
    # Only samples that contain BOTH tag_a AND tag_b.
    assert names(apply_tag_filters(Sample.query, args)) == [
        "s_first_a",
        "s_last_a",
        "s_middle_a",
    ]


def test_include_lowercases_input(samples):
    args = {"enable_include": "true", "include_tags": "TAG_A"}
    # Inputs are lowercased before matching; stored tags are already lowercase.
    assert "s_only_a" in names(apply_tag_filters(Sample.query, args))


def test_include_strips_whitespace_and_drops_empty(samples):
    args = {"enable_include": "true", "include_tags": " tag_a , , "}
    assert names(apply_tag_filters(Sample.query, args)) == [
        "s_first_a",
        "s_last_a",
        "s_middle_a",
        "s_only_a",
    ]


# ---------------------------------------------------------------------------
# Exclude (OR)
# ---------------------------------------------------------------------------


def test_exclude_single_tag(samples):
    args = {"enable_exclude": "true", "exclude_tags": "tag_a"}
    out = names(apply_tag_filters(Sample.query, args))
    assert "s_only_a" not in out
    assert "s_first_a" not in out
    assert "s_no_a" in out  # has tag_b but not tag_a


def test_exclude_multiple_tags_uses_OR(samples):
    # Exclude if sample has ANY of these.
    args = {"enable_exclude": "true", "exclude_tags": "tag_a,tag_b"}
    out = names(apply_tag_filters(Sample.query, args))
    # All samples with either tag_a or tag_b are gone.
    for excluded in ["s_only_a", "s_first_a", "s_middle_a", "s_last_a", "s_no_a"]:
        assert excluded not in out
    # Substring-trap sample (tag_aa) should survive — neither tag_a nor tag_b matches.
    assert "s_substring_trap" in out


# ---------------------------------------------------------------------------
# Prefix (AND)
# ---------------------------------------------------------------------------


def test_prefix_matches_first_position(samples):
    args = {"enable_prefix": "true", "prefix_tags": "x_"}
    out = names(apply_tag_filters(Sample.query, args))
    assert "s_prefix_x" in out
    assert "s_prefix_x_first" in out
    assert "s_no_prefix_x" not in out


def test_prefix_multiple_uses_AND(samples):
    # Must satisfy ALL prefixes simultaneously.
    args = {"enable_prefix": "true", "prefix_tags": "x_,other"}
    out = names(apply_tag_filters(Sample.query, args))
    # Only s_prefix_x_first has both an x_-prefixed tag AND an "other"-prefixed tag.
    assert out == ["s_prefix_x_first"]


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def test_include_and_exclude_compose(samples):
    args = {
        "enable_include": "true",
        "include_tags": "tag_a",
        "enable_exclude": "true",
        "exclude_tags": "tag_c",
    }
    out = names(apply_tag_filters(Sample.query, args))
    # Has tag_a AND not tag_c.
    assert "s_middle_a" not in out  # has tag_c, excluded
    assert "s_only_a" in out
    assert "s_first_a" in out
    assert "s_last_a" in out
