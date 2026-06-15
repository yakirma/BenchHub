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


# Current layout (see get_column_priority docstring): fixed framing columns
# name(0) → metric(10) → stats(20) → tags(30); then data fields keyed by role
# (GT=100s, pred=200s) + modality (text0 image1 mask2 depth3 audio4 sequence5
# json6 hist7 scalar8 label9). gt_config/signal_shape are GT-json (106),
# config is pred-json (206), gt_histogram GT-hist (107), histogram pred-hist
# (207); an unmatched key is pred-block, default modality 20 → 220.
@pytest.mark.parametrize(
    "key,kwargs,expected",
    [
        # Metadata
        ("sample_name", {}, 0),
        # Tags (right of stats now)
        ("dataset_tags", {}, 30),
        ("tags", {}, 30),
        # Stats (left of tags + all data fields)
        ("per_source_stats", {}, 20),
        # Config / signal shape → GT-block json
        ("gt_config", {}, 106),
        ("signal_shape", {}, 106),
        ("config", {}, 206),
        # Histograms → hist modality in their role block
        ("gt_histogram", {}, 107),
        ("histogram", {}, 207),
        ("histogram_filtered", {}, 207),
        # Unknown key → pred block, default modality
        ("zzz", {}, 220),
    ],
)
def test_column_priority_named_keys(key, kwargs, expected):
    assert get_column_priority(key, **kwargs) == expected


# Data fields by role + modality: GT block = 100 + modality, pred block =
# 200 + modality (image=1, depth=3, json=6, scalar=8).
@pytest.mark.parametrize(
    "column_type,is_dataset_field,expected",
    [
        ("json", True, 106),
        ("json", False, 206),
        ("image", True, 101),
        ("image", False, 201),
        ("depth", True, 103),
        ("depth", False, 203),
        ("scalar", True, 108),
        ("scalar", False, 208),
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
    # Documented ordering: sample_name → metric chart → stats → tags →
    # GT data fields → pred data fields. Within a role block, modality order
    # places image before json/histogram/scalar.
    name = get_column_priority("sample_name")
    metric = get_column_priority("per_sample_metrics")
    stats = get_column_priority("per_source_stats")
    tags = get_column_priority("tags")
    gt_image = get_column_priority("any", column_type="image", is_dataset_field=True)
    gt_scalar = get_column_priority("any", column_type="scalar", is_dataset_field=True)
    pred_image = get_column_priority("any", column_type="image", is_dataset_field=False)
    # Framing columns precede every data field; GT block precedes pred block.
    assert name < metric < stats < tags < gt_image < gt_scalar < pred_image
