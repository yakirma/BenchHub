"""HTTP wrapper around runner/harness.py.

Deployed as a separate Fly app (see runner/fly.toml). The web tier POSTs
metric jobs here; this server runs them in-process via harness.run_job and
returns the result. The "isolation" comes from being on a different VM
with no database, no /data volume, and no network egress to anywhere
inside the user's account — NOT from per-request fresh interpreters.

The trade-off: a leaky user metric can leave threads / sys.modules state /
file handles around for the next request in the same gunicorn worker. We
mitigate by:

- gunicorn --max-requests=100 (worker recycles after 100 jobs, see
  runner/fly.toml CMD).
- A short request timeout (60s, set on the WSGI app and enforced by
  gunicorn --timeout).

This is the "soft boundary, revisit later" path from DEPLOY.md § 9.
Strong boundary (machine-per-job) is Option B in that table.
"""
import os

from flask import Flask, jsonify, request

import harness


app = Flask(__name__)

# Reject huge job payloads at the WSGI layer so a malformed POST can't
# blow out worker memory before harness gets a chance to refuse it.
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('RUNNER_MAX_BODY_BYTES', 4 * 1024 * 1024))


@app.get('/health')
def health():
    """Fly's HTTP healthcheck hits this. Cheap; no work."""
    return jsonify({'ok': True}), 200


@app.post('/run')
def run():
    """Same JSON contract as harness.main: input is the job dict, output
    is the result dict. Errors that happen *inside* the metric land in
    the per-call .error field; errors that happen here (bad JSON, oversize
    body) are HTTP 4xx so the caller can distinguish them."""
    job = request.get_json(silent=True)
    if not isinstance(job, dict):
        return jsonify({'fatal': 'request body must be a JSON object', 'results': []}), 400

    result = harness.run_job(job)
    return jsonify(result), 200


@app.errorhandler(413)
def too_large(_e):
    return jsonify({'fatal': 'job payload too large', 'results': []}), 413


if __name__ == '__main__':  # pragma: no cover
    # Dev convenience only. Production uses gunicorn (see Dockerfile CMD).
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
