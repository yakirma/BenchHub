"""Unit tests for metric_engine.py.

These intentionally avoid any DB / Flask setup — metric_engine only needs
duck-typed objects with the right attributes.
"""
import json
from types import SimpleNamespace

import pytest

from metric_engine import evaluate_dynamic_metric, sort_metrics_by_dependency


def make_metric(code, name="m"):
    return SimpleNamespace(name=name, python_code=code)


# ---------------------------------------------------------------------------
# evaluate_dynamic_metric
# ---------------------------------------------------------------------------


class TestEvaluateDynamicMetric:
    def test_happy_path_two_args(self):
        gm = make_metric("def m(x, y):\n    return x + y\n")
        ctx = {"a": 2, "b": 3}
        mappings = json.dumps({"x": "a", "y": "b"})

        value, error = evaluate_dynamic_metric(gm, ctx, mappings)

        assert value == 5.0
        assert error is None

    def test_returns_float_even_for_int_result(self):
        gm = make_metric("def m():\n    return 7\n")
        value, error = evaluate_dynamic_metric(gm, {}, "{}")
        assert value == 7.0
        assert isinstance(value, float)
        assert error is None

    def test_nan_result_is_rejected(self):
        gm = make_metric("def m():\n    return float('nan')\n")
        value, error = evaluate_dynamic_metric(gm, {}, "{}")
        assert value is None
        assert error == "Result is NaN or Inf"

    def test_inf_result_is_rejected(self):
        gm = make_metric("def m():\n    return float('inf')\n")
        value, error = evaluate_dynamic_metric(gm, {}, "{}")
        assert value is None
        assert error == "Result is NaN or Inf"

    def test_scalar_literal_int(self):
        gm = make_metric("def m(x):\n    assert isinstance(x, int)\n    return x * 2\n")
        mappings = json.dumps({"x": "SCALAR:5"})
        value, error = evaluate_dynamic_metric(gm, {}, mappings)
        assert value == 10.0
        assert error is None

    def test_scalar_literal_float(self):
        gm = make_metric("def m(x):\n    return x\n")
        mappings = json.dumps({"x": "SCALAR:3.14"})
        value, error = evaluate_dynamic_metric(gm, {}, mappings)
        assert value == pytest.approx(3.14)
        assert error is None

    def test_scalar_literal_string_fallback(self):
        gm = make_metric("def m(label):\n    return len(label)\n")
        mappings = json.dumps({"label": "SCALAR:hello"})
        value, error = evaluate_dynamic_metric(gm, {}, mappings)
        assert value == 5.0
        assert error is None

    def test_missing_context_key_passes_none(self):
        gm = make_metric(
            "def m(x):\n"
            "    return 0 if x is None else 1\n"
        )
        mappings = json.dumps({"x": "not_in_ctx"})
        value, error = evaluate_dynamic_metric(gm, {}, mappings)
        assert value == 0.0
        assert error is None

    def test_syntax_error_returns_traceback(self):
        gm = make_metric("def m(:\n    return 1\n")
        value, error = evaluate_dynamic_metric(gm, {}, "{}")
        assert value is None
        assert error is not None
        assert "SyntaxError" in error

    def test_runtime_error_returns_traceback(self):
        gm = make_metric("def m():\n    return 1 / 0\n")
        value, error = evaluate_dynamic_metric(gm, {}, "{}")
        assert value is None
        assert "ZeroDivisionError" in error

    def test_no_callable_defined(self):
        gm = make_metric("x = 5\n")
        value, error = evaluate_dynamic_metric(gm, {}, "{}")
        assert value is None
        assert error == "No callable function found in code."

    def test_numpy_is_injected(self):
        gm = make_metric(
            "def m(arr):\n"
            "    return float(np.mean(arr))\n"
        )
        mappings = json.dumps({"arr": "values"})
        value, error = evaluate_dynamic_metric(gm, {"values": [1.0, 2.0, 3.0]}, mappings)
        assert value == 2.0
        assert error is None

    def test_invalid_arg_mappings_json_treated_as_empty(self):
        gm = make_metric("def m():\n    return 42\n")
        # Bad JSON should not crash; mappings become {}
        value, error = evaluate_dynamic_metric(gm, {}, "not-valid-json")
        assert value == 42.0
        assert error is None


# ---------------------------------------------------------------------------
# sort_metrics_by_dependency
# ---------------------------------------------------------------------------


def make_lm(metric_id, output_name, deps=None, target_name=None):
    """Build a duck-typed LeaderboardMetric.

    `output_name` is the name this metric exposes (used by other metrics' deps).
    If `target_name` is provided it's the override; otherwise `output_name` doubles
    as `global_metric.name`.
    """
    deps = deps or []
    arg_mappings = json.dumps({f"arg{i}": dep for i, dep in enumerate(deps)})
    gm = SimpleNamespace(name=output_name)
    return SimpleNamespace(
        id=metric_id,
        target_name=target_name,
        global_metric=gm,
        arg_mappings=arg_mappings,
    )


def order_ids(metrics):
    return [m.id for m in metrics]


class TestSortMetricsByDependency:
    def test_independent_metrics_all_returned(self):
        a = make_lm(1, "a")
        b = make_lm(2, "b")
        c = make_lm(3, "c")
        out = sort_metrics_by_dependency([a, b, c])
        assert sorted(order_ids(out)) == [1, 2, 3]
        assert len(out) == 3

    def test_linear_chain(self):
        # A -> B -> C
        a = make_lm(1, "a")
        b = make_lm(2, "b", deps=["a"])
        c = make_lm(3, "c", deps=["b"])
        out = order_ids(sort_metrics_by_dependency([c, b, a]))  # input order shuffled
        assert out.index(1) < out.index(2) < out.index(3)

    def test_diamond(self):
        # A -> B, A -> C, {B, C} -> D
        a = make_lm(1, "a")
        b = make_lm(2, "b", deps=["a"])
        c = make_lm(3, "c", deps=["a"])
        d = make_lm(4, "d", deps=["b", "c"])
        out = order_ids(sort_metrics_by_dependency([d, c, b, a]))
        assert out.index(1) < out.index(2) < out.index(4)
        assert out.index(1) < out.index(3) < out.index(4)

    def test_target_name_takes_precedence_over_global_name(self):
        # If a metric has target_name "renamed", another metric depending on
        # "renamed" should resolve to it (not the underlying global_metric.name).
        a = make_lm(1, "underlying_a", target_name="renamed")
        b = make_lm(2, "b", deps=["renamed"])
        out = order_ids(sort_metrics_by_dependency([b, a]))
        assert out.index(1) < out.index(2)

    def test_cycle_returns_all_metrics(self):
        # A -> B -> A. No valid topo order, but the function must not drop metrics.
        a = make_lm(1, "a", deps=["b"])
        b = make_lm(2, "b", deps=["a"])
        out = sort_metrics_by_dependency([a, b])
        assert sorted(order_ids(out)) == [1, 2]

    def test_self_reference_is_ignored(self):
        # Metric depending on its own name shouldn't deadlock.
        a = make_lm(1, "a", deps=["a"])
        b = make_lm(2, "b")
        out = sort_metrics_by_dependency([a, b])
        assert sorted(order_ids(out)) == [1, 2]

    def test_dep_on_external_name_is_ignored(self):
        # If arg_mappings references a name that isn't another metric in the
        # list (e.g. a dataset field like "gt_peak"), it must not affect order.
        a = make_lm(1, "a", deps=["gt_peak"])
        b = make_lm(2, "b")
        out = sort_metrics_by_dependency([a, b])
        assert sorted(order_ids(out)) == [1, 2]

    def test_malformed_arg_mappings_ignored(self):
        a = SimpleNamespace(
            id=1,
            target_name=None,
            global_metric=SimpleNamespace(name="a"),
            arg_mappings="this is not json",
        )
        b = make_lm(2, "b")
        out = sort_metrics_by_dependency([a, b])
        assert sorted(order_ids(out)) == [1, 2]
