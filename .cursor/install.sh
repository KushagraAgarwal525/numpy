#!/usr/bin/env bash
# Idempotent Cloud Agent install: sync latest upstream, refresh deps, rebuild NumPy.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

ensure_upstream() {
  if git remote get-url upstream >/dev/null 2>&1; then
    git remote set-url upstream https://github.com/numpy/numpy.git
  else
    git remote add upstream https://github.com/numpy/numpy.git
  fi
  git fetch upstream main --tags --prune
}

# Keep the working tree on the newest upstream code while preserving fork-only
# files (.cursor/, AGENTS.md). Safe on main; on topic branches, merge upstream
# tip so issue fixes are based on current NumPy.
sync_latest_upstream() {
  ensure_upstream
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  echo "Syncing upstream/main into ${branch}..."

  if git merge-base --is-ancestor upstream/main HEAD; then
    echo "Already contains upstream/main ($(git rev-parse --short upstream/main))"
    return 0
  fi

  if ! git merge upstream/main -m "chore: sync upstream/main for cloud agent $(date -u +%Y-%m-%dT%H:%MZ)"; then
    echo "ERROR: merge conflict while syncing upstream/main" >&2
    git merge --abort || true
    exit 1
  fi
  echo "Synced to upstream/main ($(git rev-parse --short upstream/main))"
}

sync_latest_upstream

git submodule update --init --recursive

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements/build_requirements.txt
python -m pip install -r requirements/test_requirements.txt
python -m pip install -r requirements/hypothesis_requirements.txt
python -m pip install -r requirements/linter_requirements.txt

# Ensure spin/venv tools win over system PATH in later agent shells.
if ! grep -q 'source .venv/bin/activate' ~/.bashrc 2>/dev/null; then
  {
    echo ''
    echo '# NumPy Cloud Agent venv'
    echo "if [[ -f \"$PWD/.venv/bin/activate\" ]]; then source \"$PWD/.venv/bin/activate\"; fi"
  } >> ~/.bashrc
fi

# Rebuild against the synced tree. Cursor checkpoints after install.
spin build

python - <<'PY'
import numpy as np
print("numpy_ok", np.__version__, np.__file__)
PY
