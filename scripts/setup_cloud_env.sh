#!/usr/bin/env bash
# Idempotent install for Cursor Cloud Builds (CPU only).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

ensure_venv_capable() {
  # Default Cursor images may lack ensurepip / python3-venv.
  if python3 -c "import ensurepip" >/dev/null 2>&1; then
    return 0
  fi
  echo "ensurepip missing; installing python3-venv via apt..."
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip unzip
  else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip unzip
  fi
}

venv_usable() {
  [ -x .venv/bin/python ] && .venv/bin/python -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null
}

ensure_venv_capable

# Prefer a project venv so we don't fight system Python / PEP 668.
if ! venv_usable; then
  rm -rf .venv
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
  if ! command -v unzip >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip
    else
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip
    fi
  fi
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
