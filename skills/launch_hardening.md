---
name: launch-hardening
description: "Public-launch changes: quotas (50GB public / 10GB private + account usage bars), 200-user signup cap, feature-requests→GitHub, cross-user dependency guards (freeze downgrade+delete when others depend), local-upload bypasses materialize. Read before touching quotas, signup, visibility/delete routes, or feature requests."
metadata:
  type: project
  originSessionId: 5c7cb6c2-c6d2-4a9f-9ba4-8a84e62b77b0
---

Shipped for the public launch (all live in prod):

- **Quotas:** public default dropped **100→50 GB** (private stays 10 GB).
  `User.quota_public_max_bytes` default + the ALTER default + a
  `check_and_migrate_db` backfill (old-100GB rows → 50GB, ran on 3 users).
  `/settings/account` shows both buckets as used/cap/free progress bars
  (route `account_settings` computes via `storage_used_bytes(visibility=)` +
  `quota_cap_for`).
- **Signup cap:** new registrations refused once `BENCHHUB_MAX_USERS`
  (default 200) accounts exist. Guarded at all 3 user-creation sites
  (GitHub/Google OAuth callbacks + email-login verify) via
  `_signup_blocked(email)`; existing users + admin emails always get in.
- **Feature requests → GitHub:** `/feature_requests` 302s to
  `github.com/yakirma/BenchHub/issues`; in-app form retired; nav/docs links
  repointed. (`GITHUB_ISSUES_URL` in app.py.)
- **Cross-user dependency guards** ([devkit-and-dynamic-dtypes](devkit_and_dynamic_dtypes.md) covers the
  dtype one): freeze visibility-downgrade-off-public AND delete when others
  depend — dataset bound to another user's LB
  (`_dataset_foreign_leaderboards`), LB another user submitted to
  (`_leaderboard_foreign_submissions`, mirrored subs don't count), dtype used
  by a public LB (`_datatype_used_by_public_lb`). On
  set_dataset_visibility / set_leaderboard_visibility / delete_dataset /
  delete_leaderboard / the datatype routes. Admins bypass. Tests:
  `tests/test_dependency_guards.py`.
- **Local python uploads bypass materialize** (verified): API/client dataset
  uploads are `preview_only=False` (full-res); the per-LB materialize trigger
  is gated on `any_preview`, and `materialize_for_lb` skips non-HF datasets.
- **Docs:** README/USER_GUIDE/docs updated to this state; `docs/ARCHITECTURE.md`
  + editable `docs/diagrams/architecture.drawio` (+ rendered `.svg`, served
  in-app at `/docs/diagram/<name>`, embedded on the Core concepts page).
- Sandbox + dev kit + dynamic dtypes: see [sandbox-typed-state](sandbox_typed_state.md) +
  [devkit-and-dynamic-dtypes](devkit_and_dynamic_dtypes.md).
