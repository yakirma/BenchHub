
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_engine import evaluate_dynamic_metric

# Mock Global Metric
class MockGlobalMetric:
    def __init__(self, code, name):
        self.python_code = code
        self.name = name

# 1. Test No-Arg Metric
code_no_arg = """
def constant_metric():
    return 42.0
"""
gm_no_arg = MockGlobalMetric(code_no_arg, "Constant Metric")
result, error = evaluate_dynamic_metric(gm_no_arg, {}, "{}")

print(f"Test No-Arg Metric: Result={result}, Error={error}")
if result == 42.0 and error is None:
    print("PASS: No-Arg Metric")
else:
    print("FAIL: No-Arg Metric")

# 2. Test Scalar Argument
code_scalar = """
def scale_metric(val, factor):
    return val * factor
"""
gm_scalar = MockGlobalMetric(code_scalar, "Scalar Metric")
# Mapping: val -> context['gt_val'], factor -> SCALAR:2.5
context = {'gt_val': 10.0}
mappings = json.dumps({'val': 'gt_val', 'factor': 'SCALAR:2.5'})

result, error = evaluate_dynamic_metric(gm_scalar, context, mappings)

print(f"Test Scalar Metric: Result={result}, Error={error}")
if result == 25.0 and error is None:
    print("PASS: Scalar Metric")
else:
    print("FAIL: Scalar Metric")

