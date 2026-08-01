# Agent notes (fork)

## Cursor Cloud specific instructions

This fork is configured for Cursor Cloud Agents via `.cursor/environment.json`.

### Staying current with upstream
- Fork `main` is synced from `numpy/numpy` daily by `.github/workflows/sync-upstream.yml` (also runnable via Actions → Sync upstream main → Run workflow).
- Every Cloud Agent boot also merges `upstream/main` in `.cursor/install.sh` before `spin build`, so the build matches the latest upstream tip even if the daily sync has not run yet.
- Fork-only files (`.cursor/`, `AGENTS.md`) are kept across merges.

### Environment
- Base image installs compilers, `gfortran`, OpenBLAS, Python 3, and ccache.
- `install` syncs upstream, creates `.venv`, installs build/test/lint requirements, runs `spin build`, and verifies `import numpy`.
- Always activate the venv before commands: `source .venv/bin/activate` (from repo root).

### Build / test
- Rebuild after C/Cython changes: `spin build`
- Full suite (slow): `spin test -v`
- Preferred for issue fixes — narrowest relevant tests, e.g.:
  - `spin test -v -t numpy/_core/tests/test_multiarray.py`
  - `spin test -v -t path/to/test_file.py::test_name`
- Lint touched Python: `spin lint`
- Do **not** run `pytest` from repo root without `spin` (strange import errors).

### Contribution rules for this fork automation
- Work one upstream `numpy/numpy` issue at a time.
- Branch from up-to-date `main` after install sync has completed.
- Minimal diffs; match local style; no drive-by refactors.
- Open **draft PRs only** against this fork's `main` (never against upstream).
- No PR unless verification for the changed area passed.
