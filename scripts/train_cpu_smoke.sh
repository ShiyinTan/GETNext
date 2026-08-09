#!/usr/bin/env bash
# Short CPU smoke train + predict for GETNext.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f dataset/NYC/NYC_train.csv ]; then
  bash scripts/setup_cloud_env.sh
fi

EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-8}"
NAME="${NAME:-cpu-smoke}"

echo "=== GETNext CPU smoke train (${EPOCHS} epochs, batch=${BATCH}) ==="
python train.py \
  --data-train dataset/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --time-units 48 \
  --time-feature norm_in_day_time \
  --poi-embed-dim 64 \
  --user-embed-dim 64 \
  --time-embed-dim 16 \
  --cat-embed-dim 16 \
  --node-attn-nhid 64 \
  --transformer-nhid 256 \
  --transformer-nlayers 2 \
  --transformer-nhead 2 \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --workers 0 \
  --no-cuda \
  --name "${NAME}" \
  --exist-ok

# Resolve latest/matching run dir
RUN_DIR="$(ls -d runs/train/${NAME}* 2>/dev/null | sort | tail -n 1)"
if [ -z "${RUN_DIR}" ]; then
  echo "No run directory found under runs/train/${NAME}*"
  exit 1
fi

CKPT="${RUN_DIR}/checkpoints/best_epoch.state.pt"
echo "=== Predict with ${CKPT} ==="
python predict.py \
  --checkpoint "${CKPT}" \
  --data-test dataset/NYC/NYC_test.csv \
  --data-adj-mtx dataset/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --batch "${BATCH}" \
  --no-cuda \
  --output-dir "${RUN_DIR}/predictions"

echo "Done. Metrics: ${RUN_DIR}/metrics-val.txt"
echo "Predictions: ${RUN_DIR}/predictions/"
