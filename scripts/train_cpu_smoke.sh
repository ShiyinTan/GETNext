#!/usr/bin/env bash
# Short CPU smoke train + predict for GETNext.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! python -c "import torch" >/dev/null 2>&1 || [ ! -f dataset/NYC/NYC_train.csv ]; then
  bash scripts/setup_cloud_env.sh
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-8}"
NAME="${NAME:-cpu-smoke}"
# Smaller dims keep CPU smoke tractable on Cloud VMs.
POI_EMBED_DIM="${POI_EMBED_DIM:-32}"
USER_EMBED_DIM="${USER_EMBED_DIM:-32}"
TIME_EMBED_DIM="${TIME_EMBED_DIM:-16}"
CAT_EMBED_DIM="${CAT_EMBED_DIM:-16}"
NODE_ATTN_NHID="${NODE_ATTN_NHID:-32}"
TRANSFORMER_NHID="${TRANSFORMER_NHID:-128}"
TRANSFORMER_NLAYERS="${TRANSFORMER_NLAYERS:-2}"
TRANSFORMER_NHEAD="${TRANSFORMER_NHEAD:-2}"

echo "=== GETNext CPU smoke train (${EPOCHS} epochs, batch=${BATCH}) ==="
python train.py \
  --data-train dataset/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --time-units 48 \
  --time-feature norm_in_day_time \
  --poi-embed-dim "${POI_EMBED_DIM}" \
  --user-embed-dim "${USER_EMBED_DIM}" \
  --time-embed-dim "${TIME_EMBED_DIM}" \
  --cat-embed-dim "${CAT_EMBED_DIM}" \
  --node-attn-nhid "${NODE_ATTN_NHID}" \
  --transformer-nhid "${TRANSFORMER_NHID}" \
  --transformer-nlayers "${TRANSFORMER_NLAYERS}" \
  --transformer-nhead "${TRANSFORMER_NHEAD}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --workers 0 \
  --no-cuda \
  --name "${NAME}" \
  --exist-ok

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
