#!/usr/bin/env bash
# CPU 冒烟：只跑很少几步，确认因果代码能在没有 GPU 的机器上跑通。
# 不是完整训练，Acc@k 接近 0 是正常的（随机初始化 + 只更新几个 batch）。
#
# 用法（在仓库根目录）：
#   bash causal/run_cpu_smoke.sh
# 可改环境变量：
#   EPOCHS=1 BATCH=4 NAME=causal-debug MAX_BATCHES=8 bash causal/run_cpu_smoke.sh
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

EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-4}"
NAME="${NAME:-causal-cpu-smoke}"
MAX_BATCHES="${MAX_BATCHES:-6}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}"

echo "=== Causal next-POI CPU smoke (${EPOCHS} epoch, batch=${BATCH}, max_batches=${MAX_BATCHES}) ==="
# 模型比论文配置小一号，和 GETNext 的 CPU smoke 同一量级，才能在 CPU 上几秒跑完
python causal/train.py \
  --data-train dataset/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC_val.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --time-units 48 \
  --time-feature norm_in_day_time \
  --poi-embed-dim 64 \
  --user-embed-dim 64 \
  --time-embed-dim 16 \
  --cat-embed-dim 16 \
  --hc-dim 32 \
  --transformer-nhid 256 \
  --transformer-nlayers 2 \
  --transformer-nhead 2 \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --workers 0 \
  --no-cuda \
  --max-batches "${MAX_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --name "${NAME}" \
  --exist-ok

RUN_DIR="$(ls -d runs/causal/${NAME}* 2>/dev/null | sort | tail -n 1)"
if [ -z "${RUN_DIR}" ]; then
  echo "No run directory found under runs/causal/${NAME}*"
  exit 1
fi

CKPT="${RUN_DIR}/checkpoints/best_epoch.state.pt"
echo "=== Predict with ${CKPT} ==="
python causal/predict.py \
  --checkpoint "${CKPT}" \
  --data-test dataset/NYC/NYC_test.csv \
  --data-train dataset/NYC/NYC_train.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --batch "${BATCH}" \
  --no-cuda \
  --max-batches "${MAX_VAL_BATCHES}" \
  --output-dir "${RUN_DIR}/predictions"

echo "Done. Metrics: ${RUN_DIR}/metrics-val.txt"
echo "Predictions: ${RUN_DIR}/predictions/"
