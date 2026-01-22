import re

with open('templates/comparison.html', 'r') as f:
    content = f.read()

# Fix the multi-line hasSubmissions tag
content = re.sub(
    r'const hasSubmissions = \{\{ \(submissions \| length > 0\) \| tojson \| safe\n    \}\};',
    'const hasSubmissions = {{ (submissions | length > 0) | tojson | safe }};',
    content
)

# Fix the unspaced or[] and other variable inits
content = re.sub(
    r'let submissions = \{\{ \(submissions_json or\[\]\) \| tojson \| safe \}\};',
    '        let submissions = {{ (submissions_json or []) | tojson | safe }};',
    content
)
content = re.sub(
    r'let selectedMetrics = \{\{ \(selected_metrics or\[\]\) \| tojson \| safe \}\};',
    '        let selectedMetrics = {{ (selected_metrics or []) | tojson | safe }};',
    content
)
content = re.sub(
    r'let activeMetrics = \{\{ \(active_metrics or\[\]\) \| tojson \| safe \}\};',
    '        let activeMetrics = {{ (active_metrics or []) | tojson | safe }};',
    content
)
content = re.sub(
    r'let customMetrics = \{\{ \(custom_metrics or\[\]\) \| tojson \| safe \}\};',
    '        let customMetrics = {{ (custom_metrics or []) | tojson | safe }};',
    content
)
content = re.sub(
    r'let sampleMetricOptions = \{\{ \(sample_metric_options or \{ \}\) \| tojson \| safe \}\};',
    '        let sampleMetricOptions = {{ (sample_metric_options or {}) | tojson | safe }};',
    content
)

with open('templates/comparison.html', 'w') as f:
    f.write(content)
