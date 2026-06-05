# BenchHub dev-knowledge notes

Durable, granular notes accumulated while building BenchHub — the non-obvious
decisions, gotchas, and subsystem maps that aren't derivable from the code or
git history alone. They complement [`../CLAUDE.md`](../CLAUDE.md) (the
top-level architecture + conventions) and the in-app docs at `/docs`. Read
the relevant note before touching the matching subsystem; build on them when
you add new things.

Each file carries a short `name` / `description` header and links related
notes inline.

## Index

| Note | What it covers |
|---|---|
| [sandbox_typed_state](sandbox_typed_state.md) | The metric/viz sandbox (live in prod): hardened container, typed-arg + bytes JSON round-trip, image vendoring, the `BENCHHUB_SANDBOX_METRICS` flag. Read before touching `metric_engine` sandbox / `runner/`. |
| [devkit_and_dynamic_dtypes](devkit_and_dynamic_dtypes.md) | bh-client metric/viz authoring (`create_metric`/`visualization`, `benchhub.author`) + user-registered data types (`DataTypeDef`, `create_datatype`, sandboxed `visualize`, render route, public-LB guard). |
| [launch_hardening](launch_hardening.md) | Public-launch changes: 50/10 GB split quotas + usage UI, 200-user signup cap, feature-requests→GitHub, cross-user dependency guards, local-upload bypasses materialize. |
| [file_tree_importer](file_tree_importer.md) | The self-service file-tree HF importer: spec/loaders, bounded listing for huge repos, the split chooser. Prefer it over bespoke per-dataset scripts. |
| [hf_agent_mode_imports](hf_agent_mode_imports.md) | Lessons from HF import scripts: file-tree layouts, canonical-ext conversions, role guessing, junk filtering. |
| [agent_mode_mandate](agent_mode_mandate.md) | Every HF import: read the card + list the tree + probe one artifact before writing import code. |
| [category_modality_contract](category_modality_contract.md) | A dataset's task tag dictates the required output kinds (depth→depth field, segmentation→mask, …). |
| [vision_shards_mandate](vision_shards_mandate.md) | WebDataset-shard vision datasets: import shards + merged parquet metadata as one dataset. |
| [hf_sequence_of_masks](hf_sequence_of_masks.md) | `Sequence(Image)` mask stacks must be composited into one instance-id mask, not dropped. |
| [sampling_policy_no_cap](sampling_policy_no_cap.md) | HF imports take the full eval split; no sample cap. |
| [storage_pricing_reference](storage_pricing_reference.md) | Storage cost framing (Fly / B2 / self-host) for sizing the catalog. |
| [data_dir_isolation](data_dir_isolation.md) | Never import `app` without `BENCHHUB_DATA_DIR` set — it binds the prod DB; `drop_all`/`create_all` would nuke it. |
| [deploy_sudo_nopasswd](deploy_sudo_nopasswd.md) | Restart each `benchhub-*` service in its own `systemctl` call (sudo NOPASSWD quirk). |
