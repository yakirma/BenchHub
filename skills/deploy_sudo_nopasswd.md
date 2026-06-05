---
name: deploy-sudo-nopasswd
description: Restarting benchhub services without a password requires one systemctl invocation per unit — combining them in a single command breaks NOPASSWD matching.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f5a8564-be84-43d2-bf54-da49e47cc27b
---

When deploying BenchHub to prod (on the box itself), restart each
service in its own `sudo systemctl restart …` call:

```
sudo -n systemctl restart benchhub-web
sudo -n systemctl restart benchhub-celery
```

**Why:** sudoers grants `(root) NOPASSWD:` for the exact commands
`/usr/bin/systemctl restart benchhub-web` and
`/usr/bin/systemctl restart benchhub-celery` (each unit listed
separately, also `reload`/`start`/`stop` variants).
Combining them — `sudo systemctl restart benchhub-web benchhub-celery` —
is a different argv that doesn't match any single rule, so sudo
falls back to the `(ALL : ALL) ALL` rule and prompts for a password,
which fails non-interactively.

**How to apply:** during deploy (the `git pull && systemctl restart …`
step in [deploy-runbook](../docs/SELFHOST_RUNBOOK.md)), split the restart into two sequential
sudo calls. The `&&` chain is fine; what breaks NOPASSWD is putting
multiple unit names in one systemctl invocation.

**Reload doesn't work — use restart.** `sudo systemctl reload benchhub-web`
fails with `Job type reload is not applicable for unit benchhub-web.service`
because the gunicorn unit has no `ExecReload=` directive. Despite what
CLAUDE.md's runbook suggests ("graceful HUP, no dropped requests"), the
only working command is `sudo systemctl restart benchhub-web`. Same for
benchhub-celery. **Also blocked:** `sudo systemctl status …` — not in
the NOPASSWD allowlist; use `journalctl -u <unit> -n 20 --no-pager`
(no sudo) to verify the service came back up.
