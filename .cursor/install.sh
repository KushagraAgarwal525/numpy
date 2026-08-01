#!/usr/bin/env bash
# Idempotent Cloud Agent install: refresh deps and keep an in-tree NumPy build ready.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

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
if ! grep -q 'Projects/numpy/.venv\|numpy/.venv\|source .venv/bin/activate' ~/.bashrc 2>/dev/null; then
  {
    echo ''
    echo '# NumPy Cloud Agent venv'
    echo "if [[ -f \"$PWD/.venv/bin/activate\" ]]; then source \"$PWD/.venv/bin/activate\"; fi"
  } >> ~/.bashrc
fi

# First run is slow; Cursor checkpoints after install so later starts reuse the build.
spin build

python - <<'PY'
import numpy as np
print("numpy_ok", np.__version__, np.__file__)
PY
