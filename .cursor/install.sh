#!/usr/bin/env bash
# Idempotent Cloud Agent install:
# - fetch upstream/main
# - merge only if behind
# - rebuild only if upstream SHA != last successful build (or smoke check fails)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

STATE_FILE=".cursor/build-state.json"

ensure_upstream() {
  if git remote get-url upstream >/dev/null 2>&1; then
    git remote set-url upstream https://github.com/numpy/numpy.git
  else
    git remote add upstream https://github.com/numpy/numpy.git
  fi
  git fetch upstream main --tags --prune
}

read_build_state_field() {
  local field="$1"
  if [[ ! -f "$STATE_FILE" ]]; then
    echo ""
    return
  fi
  python3 - <<PY
import json
from pathlib import Path
try:
    print(json.loads(Path(".cursor/build-state.json").read_text()).get("$field", "") or "")
except Exception:
    print("")
PY
}

write_build_state() {
  local sha="$1"
  local status="$2"
  python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
Path(".cursor").mkdir(exist_ok=True)
Path(".cursor/build-state.json").write_text(json.dumps({
    "upstream_sha": "$sha",
    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "$status",
}, indent=2) + "\n")
print("Wrote $STATE_FILE -> $sha ($status)")
PY
}

smoke_numpy() {
  # `spin python -c` fails (click eats -c). Use -- to forward args to Python.
  # Also prefer a temp script outside the repo so cwd import traps cannot win.
  local script
  script="$(mktemp /tmp/numpy-smoke.XXXXXX.py)"
  cat > "$script" <<'PY'
import numpy as np
print("numpy_ok", np.__version__, np.__file__)
PY
  # shellcheck disable=SC2064
  trap "rm -f '$script'" RETURN
  if spin python -- "$script"; then
    return 0
  fi
  # Fallback: explicit -c after --
  spin python -- -c "import numpy as np; print('numpy_ok', np.__version__, np.__file__)"
}

ensure_venv_and_deps() {
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

  if ! grep -q 'source .venv/bin/activate' ~/.bashrc 2>/dev/null; then
    {
      echo ''
      echo '# NumPy Cloud Agent venv'
      echo "if [[ -f \"$PWD/.venv/bin/activate\" ]]; then source \"$PWD/.venv/bin/activate\"; fi"
    } >> ~/.bashrc
  fi
}

sync_if_needed() {
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if git merge-base --is-ancestor upstream/main HEAD; then
    echo "Already contains upstream/main ($(git rev-parse --short upstream/main))"
    return 1
  fi
  echo "Merging upstream/main into ${branch}..."
  if ! git merge upstream/main -m "chore: sync upstream/main for cloud agent $(date -u +%Y-%m-%dT%H:%MZ)"; then
    echo "ERROR: merge conflict while syncing upstream/main" >&2
    git merge --abort || true
    exit 1
  fi
  echo "Synced to upstream/main ($(git rev-parse --short upstream/main))"
  return 0
}

ensure_upstream
UPSTREAM_SHA="$(git rev-parse upstream/main)"
LAST_BUILT_SHA="$(read_build_state_field upstream_sha)"
LAST_STATUS="$(read_build_state_field status)"
echo "upstream/main=$UPSTREAM_SHA last_built=${LAST_BUILT_SHA:-<none>} status=${LAST_STATUS:-<none>}"

SYNCED=0
if sync_if_needed; then
  SYNCED=1
fi

ensure_venv_and_deps

NEED_BUILD=0
if [[ "$SYNCED" -eq 1 ]]; then
  NEED_BUILD=1
elif [[ "$LAST_STATUS" != "ok" || "$LAST_BUILT_SHA" != "$UPSTREAM_SHA" ]]; then
  NEED_BUILD=1
elif ! smoke_numpy; then
  echo "Smoke check failed against cached build; rebuilding."
  NEED_BUILD=1
fi

if [[ "$NEED_BUILD" -eq 1 ]]; then
  echo "Building NumPy for $UPSTREAM_SHA ..."
  spin build
  if ! smoke_numpy; then
    write_build_state "$UPSTREAM_SHA" "failed"
    echo "ERROR: NumPy build smoke check failed after spin build" >&2
    exit 1
  fi
  write_build_state "$UPSTREAM_SHA" "ok"
  echo "Build complete for $UPSTREAM_SHA"
else
  echo "Skipping rebuild; already built for $UPSTREAM_SHA"
  smoke_numpy
fi
