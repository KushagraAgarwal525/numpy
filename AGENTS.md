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
- Use `spin python`, `spin test`, `spin build` — never bare `python -c "import numpy"` or `pytest` from the repo root (imports the source tree and breaks).

### Build / test
- Rebuild after C/Cython edits: `spin build`
- Narrow tests preferred:
  - `spin test -v -t numpy/_core/tests/test_multiarray.py`
  - `spin test -v -t path/to/test_file.py::test_name`
- Lint: `spin lint`

### Issue tracking
- Read/update `.cursor/issue-state.json` (`completed` / `attempted` / `skipped`) so later runs do not repeat work.

### Contribution rules
- One upstream `numpy/numpy` issue per run.
- Branch from up-to-date `main` after install finishes.
- Minimal diffs; match local style; no drive-by refactors.
- **Draft PRs only** against this fork's `main` (never against upstream).
- No PR unless verification for the changed area passed.
