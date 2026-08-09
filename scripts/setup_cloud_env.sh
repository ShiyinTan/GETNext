#!/usr/bin/env bash
# Idempotent install for Cursor Cloud Builds (CPU only).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Prefer a project venv so we don't fight system Python / PEP 668.
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch, then project deps.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.1,<2.5"

python -m pip install -r requirements-cpu.txt

# Unpack NYC dataset if needed (zip already includes graph_A/X).
if [ ! -f dataset/NYC/NYC_train.csv ]; then
  mkdir -p dataset
  unzip -o -q dataset/NYC.zip -d dataset
  rm -rf dataset/__MACOSX || true
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "device", "cpu")
PY

# Persist venv activation for later agent shells when possible.
if [ -n "${CURSOR_AGENT:-}" ] || [ -f /etc/profile.d/cursor.sh ] || true; then
  ACTIVATE_LINE="source ${ROOT}/.venv/bin/activate"
  for rc in "${HOME}/.bashrc" "${HOME}/.profile"; do
    if [ -f "$rc" ] && ! grep -Fq "$ACTIVATE_LINE" "$rc"; then
      echo "$ACTIVATE_LINE" >> "$rc"
    fi
  done
fi

echo "Cloud CPU environment ready at ${ROOT} (venv: ${ROOT}/.venv)"
