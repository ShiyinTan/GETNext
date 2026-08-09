#!/usr/bin/env bash
# Idempotent install for Cursor Cloud Builds (CPU only).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

python -m pip install --upgrade pip setuptools wheel --break-system-packages

# Install CPU-only PyTorch first, then the rest.
python -m pip install --break-system-packages \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.1,<2.5"

python -m pip install --break-system-packages -r requirements-cpu.txt

# Unpack NYC dataset if needed (zip already includes graph_A/X).
if [ ! -f dataset/NYC/NYC_train.csv ]; then
  mkdir -p dataset
  unzip -o -q dataset/NYC.zip -d dataset
  # Drop macOS junk if present
  rm -rf dataset/__MACOSX || true
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "device", "cpu")
PY

echo "Cloud CPU environment ready at ${ROOT}"
