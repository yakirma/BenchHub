"""Tests for small pure helpers in app.py.

format_tag_value, get_distinguishable_metric_name, get_column_priority.
"""
from types import SimpleNamespace

import pytest

from app import format_tag_value, get_column_priority, get_distinguishable_metric_name


# ---------------------------------------------------------------------------
# format_tag_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "N/A"),
        ("true", 1),
        ("True", 1),
        (" YES ", 1),
        ("false", 0),
        ("No", 0),
        ("3", 3),
        ("3.0", 3),         # float that's exactly an int → returned as int
        ("3.14", "3.140000"),  # non-int float → 6-decimal string
        (0.5, "0.500000"),
        (42, 42),
        ("hello", "hello"),
        ("", ""),           # empty string → not numeric, not bool → returned as-is
    ],
)
def test_format_tag_value_table(raw, expected):
    assert format_tag_value(raw) == expected


def test_format_tag_value_inf_returns_original():
    # math.isfinite(inf) is False → numeric branch falls through → returns input.
    assert format_tag_value("inf") == "inf"


def test_format_tag_value_nan_returns_original():
    assert format_tag_value("nan") == "nan"


# ---------------------------------------------------------------------------
# get_distinguishable_metric_name
# ---------------------------------------------------------------------------


def test_distinguishable_name_uses_target_name_when_set():
    lm = SimpleNamespace(
        id=7,
        target_name="L1 Loss",
        global_metric=SimpleNamespace(name="l1", label=None),
    )
    assert get_distinguishable_metric_name(lm) == "L1 Loss_7"


def test_distinguishable_name_falls_back_through_label_to_name():
    # When target_name is None and label exists → label wins.
    lm = SimpleNamespace(
        id=3,
        target_name=None,
        global_metric=SimpleNamespace(label="DisplayLabel", name="raw_name"),
    )
    assert get_distinguishable_metric_name(lm) == "DisplayLabel_3"


def test_distinguishable_name_falls_back_to_global_name_when_label_falsy():
    lm = SimpleNamespace(
        id=11,
        target_name=None,
        global_metric=SimpleNamespace(label=None, name="raw_name"),
    )
    assert get_distinguishable_metric_name(lm) == "raw_name_11"


def test_distinguishable_name_real_global_metric_object_raises():
    """REAL BUG: get_distinguishable_metric_name accesses
    lm.global_metric.label, but the GlobalMetric SQLAlchemy model has no
    `label` column. With a real GlobalMetric and no target_name, this raises
    AttributeError. Pin this — when the helper is fixed (likely by removing
    the `or .label` branch), update this test."""
    from app import GlobalMetric, LeaderboardMetric

    gm = GlobalMetric(name="real_metric", python_code="def m(): return 1")
    lm = LeaderboardMetric(global_metric=gm, arg_mappings="{}", target_name=None)
    lm.id = 5

    with pytest.raises(AttributeError, match="label"):
        get_distinguishable_metric_name(lm)


# ---------------------------------------------------------------------------
# get_column_priority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,kwargs,expected",
    [
        # Metadata
        ("sample_name", {}, 0),
        # Tags
        ("dataset_tags", {}, 5),
        ("tags", {}, 5),
        # Charts / stats
        ("per_source_stats", {}, 12),
        # Config / signal shape
        ("gt_config", {}, 30),
        ("signal_shape", {}, 30),
        ("config", {}, 31),
        # Histograms
        ("gt_histogram", {}, 40),
        ("histogram", {}, 41),
        ("histogram_filtered", {}, 41),
        # Unknown key falls through to 100
        ("zzz", {}, 100),
    ],
)
def test_column_priority_named_keys(key, kwargs, expected):
    assert get_column_priority(key, **kwargs) == expected


@pytest.mark.parametrize(
    "column_type,is_dataset_field,expected",
    [
        ("json", True, 35),
        ("json", False, 36),
        ("image", True, 50),
        ("image", False, 51),
        ("depth", True, 50),
        ("depth", False, 51),
        ("scalar", True, 60),
        ("scalar", False, 61),
    ],
)
def test_column_priority_typed_fields(column_type, is_dataset_field, expected):
    # Use an arbitrary key not matched by named-key branches.
    assert (
        get_column_priority("any_custom_name", column_type=column_type, is_dataset_field=is_dataset_field)
        == expected
    )


def test_column_priority_named_key_beats_column_type():
    # 'sample_name' should win over column_type rules.
    assert get_column_priority("sample_name", column_type="scalar", is_dataset_field=False) == 0


def test_column_priority_orders_groups_correctly():
    # Verify the documented ordering: name < tags < stats < config < histograms < images < scalars
    name = get_column_priority("sample_name")
    tags = get_column_priority("tags")
    stats = get_column_priority("per_source_stats")
    config = get_column_priority("gt_config")
    hist = get_column_priority("gt_histogram")
    image = get_column_priority("any", column_type="image", is_dataset_field=True)
    scalar = get_column_priority("any", column_type="scalar", is_dataset_field=True)
    assert name < tags < stats < config < hist < image < scalar
