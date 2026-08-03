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
- Also treat existing fork PR titles mentioning `gh-<n>` as completed/claimed.

### Issue selection (hard gate — BEFORE any code changes)
Recent runs over-indexed on small, old `numpy.ma` edge cases. That is **not** acceptable.

**Prefer (high priority):**
- `numpy/_core` C/Cython, dtype/ufunc/casting, NDArray protocol, SIMD, linalg, FFT, random
- Correctness bugs with wide impact or many reactions/comments
- Nontrivial API consistency across dtypes / array-likes
- Issues where a real fix needs design + substantial tests

**Reject / skip immediately (record in `issue-state.json` → `skipped`):**
- Docs, typos, wording-only, comment cleanup
- One-line or ~5–15 minute Python tweaks
- Tiny single-method `numpy.ma` patches (fill_value / flags / copy edge cases) unless they clearly require deep C/`_core` work
- Pure “good first issue” unless also high-impact and technically deep
- RFCs / design debates with no agreed fix
- Hardware / platform you cannot run in this environment

**Minimum complexity bar:**
The chosen issue’s honest fix must require at least one of:
1. C or Cython changes, or
2. Multi-file / cross-module design, or
3. Nontrivial algorithm or public-API work with substantial new tests

If the smallest correct fix is a small Python one-liner, **skip and pick another**. Prefer **zero PRs** over a trivial PR.

**Selection memo (required):** Before editing, write issue number, why it is meaningful, and expected complexity. Then implement.

### Contribution rules
- One upstream `numpy/numpy` issue per run.
- Branch from up-to-date `main` after install finishes.
- Minimal diffs; match local style; no drive-by refactors.
- **Draft PRs only** against this fork's `main` (never against upstream).
- No PR unless **full** `spin test -v` passed for that branch.
