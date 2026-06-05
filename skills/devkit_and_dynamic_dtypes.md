---
name: devkit-and-dynamic-dtypes
description: "The bh-client authoring dev kit (metrics/visualizations) and user-registered data types (dynamic dtypes) — how they work + where the code is. Read before touching the authoring API, DataTypeDef, or registered-kind rendering."
metadata:
  type: project
  originSessionId: 5c7cb6c2-c6d2-4a9f-9ba4-8a84e62b77b0
---

Shipped (public-launch session). Both ride on the live metric/viz sandbox
([sandbox-typed-state](sandbox_typed_state.md)).

**Dev kit (authoring metrics + visualizations programmatically):**
- API: `POST /api/metrics` + `POST /api/visualizations` (`@require_api_token`,
  shared `_api_create_library_asset` in app.py) — reuse
  `_apply_metric_typed_contract` (signature-derived `input_kinds`/`roles`),
  visibility/ownership defaults, per-owner + public name-uniqueness (409 on
  conflict).
- Client: `Client.create_metric` / `create_visualization` (accept Python
  source OR a function object → `inspect.getsource`); generic
  `transport.post_json` on both `_RequestsTransport` + `FlaskTestClientTransport`.
- `benchhub/author.py` (`bh.author`): `test_metric` / `test_metric_batch` /
  `test_visualization` — run a metric/viz locally against sample data before
  upload. Tests: `tests/test_dev_kit.py`.

**Dynamic dtypes (user-registered kinds):** model = registered kind =
declared storage (file_ext or inline) + a sandboxed `visualize(blob, params)
-> PIL.Image`. The kind name joins the GLOBAL namespace (so `DataTypeDef.name`
is globally unique); metrics on the kind get the raw bytes (sandboxed).
- Model `DataTypeDef` (app.py): name (unique), file_ext, viz_mime,
  visualize_code, owner, visibility. Auto-created by db.create_all().
- Sandbox: raw bytes cross JSON as `{"__bytes__": b64}` (metric_engine
  `_jsonify_kwarg` + harness `_decode_arg`); `metric_engine.
  visualize_dtype_in_sandbox(code, blob, params)` reuses the viz job.
- Author: `POST /api/datatypes` + `Client.create_datatype(name, *,
  file_ext, visualize_code, viz_mime, description)`. Web form: `/datatypes`
  has a "Register a data type" form → `POST /datatypes/create`
  (`create_datatype_web`); both share `_register_datatype()`. Manage:
  `/datatypes/<id>/delete` + `/visibility`. Nav link "Data types" in base.html.
- **Shipped to PyPI: `benchhub-client` 0.1.8** (was 0.1.7) — `pip install
  -U benchhub-client` gets create_metric/visualization/datatype + bh.author.
  Build: `python -m build` (pyproject `include=["benchhub*"]` → only the
  benchhub package, no app.py); `twine upload` via ~/.pypirc (never echo it).
- Discovery: registered kinds appear on `/supported_types` and in the
  file-tree importer kind dropdowns (`_all_kind_names()`).
- Render end-to-end: `_all_kind_names()` feeds the importer;
  `serve_custom_field_image` has a registered-kind branch
  (`_serve_registered_dtype_image` → sandboxed visualize, disk-cached);
  `dataset_view` emits the `<img>` via `image_render_kinds` (built-ins +
  registered kinds with a visualize). The WEB process needs docker-group
  access for this serve route (it has it post-restart, like celery).
- **Guard:** a dtype used by a PUBLIC leaderboard can't be deleted OR made
  private (`_datatype_used_by_public_lb`) — same dependency-freeze family as
  the dataset/LB guards (another user's LB binds your dataset / submits to
  your LB → frozen; see set_dataset_visibility / set_leaderboard_visibility /
  delete_dataset / delete_leaderboard). Admins bypass all.
- Tests: `tests/test_dynamic_dtypes.py`, `tests/test_dependency_guards.py`.

NOT YET: registered dtypes aren't threaded through the *comparison* view or
the typed-manifest importer's preview pre-render (the file-tree path stores
bytes + renders on demand). Metrics on a registered kind receive raw bytes,
not a typed instance.
