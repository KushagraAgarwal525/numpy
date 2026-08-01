# Agent notes (fork)

## Cursor Cloud specific instructions

This fork is configured for Cursor Cloud Agents via `.cursor/environment.json`.

### Staying current with upstream (SHA-gated)
- On every boot, `.cursor/install.sh` fetches `upstream/main` (`numpy/numpy`).
- If the working tree is behind, it merges `upstream/main` (keeps fork-only `.cursor/` + `AGENTS.md`).
- It rebuilds with `spin build` only when:
  - upstream was just merged, or
  - `.cursor/build-state.json` `upstream_sha` != current `upstream/main`, or
  - the smoke check fails.
- Otherwise it reuses the cached build and continues.
- After a successful build it writes `.cursor/build-state.json`.
- If the build/smoke check fails, **stop** — do not work on issues that run.

### Smoke / import rules
- Always activate: `source .venv/bin/activate`
- Use `spin python`, `spin test`, `spin build`.
- Never bare `python -c "import numpy"` or `pytest` from the repo root.
- One-liners: `spin python -- -c "import numpy as np; print(np.__version__)"` (note the `--`).

### Build / test
- Rebuild after C/Cython edits: `spin build`
- During development, narrow tests are fine for iteration:
  - `spin test -v -t path/to/test_file.py::test_name`
- **Before opening any draft PR, you MUST run the full test suite and it MUST pass:**
  - `spin test -v`
  - Do not open a PR if the full suite fails, is incomplete, or was skipped.
- Lint touched Python before PR: `spin lint`

### Issue tracking
- Read/update `.cursor/issue-state.json` (`completed` / `attempted` / `skipped`) so later runs do not repeat work.

### Contribution rules
- One upstream `numpy/numpy` issue per run.
- Branch from up-to-date `main` after install finishes.
- Minimal diffs; match local style; no drive-by refactors.
- **Draft PRs only** against this fork's `main` (never against upstream).
- No PR unless **full** `spin test -v` passed for that branch.
