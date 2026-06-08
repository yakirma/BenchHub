# tests/ — pytest suite

Run `pytest tests/` (NOT bare `pytest` — that tries to collect the ad-hoc root-level `test_chain.py`/`test_celery_chain.py`/`test_chain_app.py` Celery experiments, which aren't part of the suite).

## Fixtures (`conftest.py`)
- Per-session `app` — wires Flask + Celery in TEST mode (`task_always_eager=True`) so submission/eval flows run inline.
- Per-test `db_session` — drops + recreates all tables, so tests are independent.
- `auth_client` — a `client` with `session['user_id']` set to a fresh `logged_in_user`.
- `make_zip(name, layout, root_folder=...)` — builds a fake submission/dataset ZIP for upload-path tests.
- `BENCHHUB_DATA_DIR` → per-session tempdir so tests never touch `~/.dtofbenchmarking`.
- `$BENCHHUB_CACHE_DIR` → session tmp, **wiped per test**, so `Client.iter_samples`' cache doesn't leak across tests (LB ids repeat across tests).

## Add a test next to the closest existing one when you fix a bug
| Touched code | Test file |
|---|---|
| `_pwc_task_to_category`, `_PWC_AREA_RULES`, `_DOMAIN_PREFIXES` | `test_pwc_category.py` |
| `_resolve_hf_split_and_load`, `_HF_SPLIT_PREFERENCE`, `_persist_resolved_split` | `test_hf_split_resolver.py` |
| `_infer_mapping` (`Value:unknown`→json, Audio kind, Sequence-of-* fallback) | `test_pwc_import.py` / `test_hf_features_fallback.py` |
| `_compute_explorable_lb_ids` | `test_explorable.py` |
| `_VirtualSample`/`_VirtualCustomField` json/topk_list/audio dispatch | `test_attachment_iter.py` |
| `get_metric_context` text/json/topk_list deserialization | `test_metric_context_arrays.py` |
| samples-only `comparison_view` (pagination + form param threading) | `test_routes_comparison.py` |

E2E: `test_phase_b_end_to_end.py` runs the full typed loop (import → client → submit → typed metric eval → MetricResult). Sandbox: `test_sandbox_*` reference `runner/`.
