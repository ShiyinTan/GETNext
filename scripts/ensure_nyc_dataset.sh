#!/usr/bin/env bash
# Per-boot restore for NYC files unpacked during install.
# Builds skip install on later pods; git checkout can drop untracked dataset/NYC/.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f dataset/NYC/NYC_train.csv ]; then
  exit 0
fi

mkdir -p dataset
if ! command -v unzip >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip
  else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip
  fi
fi

unzip -o -q dataset/NYC.zip -d dataset
rm -rf dataset/__MACOSX || true

echo "Restored dataset/NYC from dataset/NYC.zip"
