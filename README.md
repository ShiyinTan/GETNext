# GETNext

This is the pytorch implementation of paper "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation"

![model-structure](figures/model-structure.png)

## Quick start (copy & run)

**CPU (Cursor Cloud / no GPU):**

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
bash scripts/train_cpu_smoke.sh
```

**GPU (local machine with CUDA):**

```bash
pip install -r requirements.txt
unzip -o dataset/NYC.zip -d dataset/
python train.py --batch 16 --epochs 200 --name nyc-gpu --exist-ok
python predict.py \
  --checkpoint runs/train/nyc-gpu/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv
```

---

## Installation

### GPU

```bash
pip install -r requirements.txt
```

### CPU-only

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

Always activate the venv in a new shell:

```bash
source .venv/bin/activate
```

## Requirements

See `requirements.txt` (GPU) or `requirements-cpu.txt` (CPU). Key packages: PyTorch, NumPy, pandas, scikit-learn, PyYAML, tqdm.

## Prepare data

### NYC (default)

```bash
unzip -o dataset/NYC.zip -d dataset/
ls dataset/NYC/
```

Expected files:

- `dataset/NYC/NYC_train.csv`
- `dataset/NYC/NYC_val.csv`
- `dataset/NYC/NYC_test.csv`
- `dataset/NYC/graph_A.csv`
- `dataset/NYC/graph_X.csv`

If `graph_*.csv` are missing, rebuild from train:

```bash
python build_graph.py
```

### Tokyo (TKY)

```bash
mkdir -p dataset/TKY
unzip -o dataset/TKY.zip -d dataset/TKY
ls dataset/TKY/
```

### Gowalla-CA

```bash
mkdir -p dataset/Gowalla-CA
unzip -o dataset/Gowalla-CA.zip -d dataset/Gowalla-CA
ls dataset/Gowalla-CA/
```

---

## Train examples

All flags live in `param_parser.py`. Console output is compact by default (`tqdm` + per-epoch summary). Use `--verbose` for per-batch sample dumps. Full logs: `runs/train/<name>/log_training.txt`.

### 1) Minimal NYC train (defaults + short run)

Uses default data paths under `dataset/NYC/`. Auto-picks CUDA if available.

```bash
source .venv/bin/activate   # if needed
python train.py --epochs 5 --batch 16 --name nyc-mini --exist-ok
```

### 2) Force CPU

```bash
python train.py \
  --epochs 3 \
  --batch 8 \
  --no-cuda \
  --name nyc-cpu \
  --exist-ok
```

### 3) Force a specific GPU

```bash
python train.py \
  --epochs 200 \
  --batch 16 \
  --device cuda:0 \
  --name nyc-cuda0 \
  --exist-ok
```

### 4) Paper-scale hyper-parameters (NYC)

```bash
python train.py \
  --data-train dataset/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --time-units 48 \
  --time-feature norm_in_day_time \
  --poi-embed-dim 128 \
  --user-embed-dim 128 \
  --time-embed-dim 32 \
  --cat-embed-dim 32 \
  --node-attn-nhid 128 \
  --transformer-nhid 1024 \
  --transformer-nlayers 2 \
  --transformer-nhead 2 \
  --batch 16 \
  --epochs 200 \
  --lr 0.001 \
  --name nyc-paper \
  --exist-ok
```

### 5) CPU smoke (smaller model, few epochs)

```bash
python train.py \
  --data-train dataset/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --poi-embed-dim 64 \
  --user-embed-dim 64 \
  --time-embed-dim 16 \
  --cat-embed-dim 16 \
  --node-attn-nhid 64 \
  --transformer-nhid 256 \
  --transformer-nlayers 2 \
  --transformer-nhead 2 \
  --batch 8 \
  --epochs 3 \
  --workers 0 \
  --no-cuda \
  --name cpu-smoke \
  --exist-ok
```

Or the helper (train + predict):

```bash
EPOCHS=3 BATCH=8 NAME=cpu-smoke bash scripts/train_cpu_smoke.sh
```

Override only epochs/batch/name:

```bash
EPOCHS=1 BATCH=4 NAME=debug bash scripts/train_cpu_smoke.sh
```

### 6) Custom learning rate / batch / epochs

```bash
python train.py \
  --epochs 50 \
  --batch 32 \
  --lr 0.0005 \
  --name nyc-lr5e4-b32 \
  --exist-ok
```

### 7) Verbose training (detailed batch dumps)

```bash
python train.py \
  --epochs 3 \
  --batch 8 \
  --no-cuda \
  --verbose \
  --name nyc-verbose \
  --exist-ok
```

### 8) Train on TKY

```bash
mkdir -p dataset/TKY
unzip -o dataset/TKY.zip -d dataset/TKY

python train.py \
  --data-train dataset/TKY/TKY_train.csv \
  --data-val dataset/TKY/TKY_val.csv \
  --data-adj-mtx dataset/TKY/graph_A.csv \
  --data-node-feats dataset/TKY/graph_X.csv \
  --batch 16 \
  --epochs 200 \
  --name tky-exp \
  --exist-ok
```

### 9) Train on Gowalla-CA

```bash
mkdir -p dataset/Gowalla-CA
unzip -o dataset/Gowalla-CA.zip -d dataset/Gowalla-CA

python train.py \
  --data-train dataset/Gowalla-CA/gowalla_train.csv \
  --data-val dataset/Gowalla-CA/gowalla_val.csv \
  --data-adj-mtx dataset/Gowalla-CA/graph_A.csv \
  --data-node-feats dataset/Gowalla-CA/graph_X.csv \
  --batch 16 \
  --epochs 200 \
  --name gowalla-exp \
  --exist-ok
```

### Common flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--epochs` | Training epochs | `200` |
| `--batch` | Batch size | `20` |
| `--lr` | Learning rate | `0.001` |
| `--data-train` | Train CSV | `dataset/NYC/NYC_train.csv` |
| `--data-val` | Val CSV | `dataset/NYC/NYC_val.csv` |
| `--data-adj-mtx` | Graph adjacency | `dataset/NYC/graph_A.csv` |
| `--data-node-feats` | Graph node features | `dataset/NYC/graph_X.csv` |
| `--device` | Device (`cuda` / `cuda:0` / `cpu`) | auto |
| `--no-cuda` | Force CPU | off |
| `--name` | Run folder under `runs/train/` | `exp` |
| `--exist-ok` | Do not auto-increment run name | off |
| `--verbose` | Noisy per-batch dumps | off |
| `--workers` | DataLoader workers | `0` |

### Outputs

```text
runs/train/<name>/
  ├── args.yaml
  ├── log_training.txt
  ├── metrics-train.txt
  ├── metrics-val.txt
  ├── checkpoints/best_epoch.state.pt
  └── checkpoints/best_epoch.txt
```

---

## Predict examples

### NYC (CPU)

```bash
python predict.py \
  --checkpoint runs/train/cpu-smoke/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv \
  --data-adj-mtx dataset/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/graph_X.csv \
  --batch 8 \
  --no-cuda \
  --output-dir runs/train/cpu-smoke/predictions
```

### NYC (GPU)

```bash
python predict.py \
  --checkpoint runs/train/nyc-paper/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv \
  --batch 16 \
  --output-dir runs/train/nyc-paper/predictions
```

### TKY

```bash
python predict.py \
  --checkpoint runs/train/tky-exp/checkpoints/best_epoch.state.pt \
  --data-test dataset/TKY/TKY_test.csv \
  --data-adj-mtx dataset/TKY/graph_A.csv \
  --data-node-feats dataset/TKY/graph_X.csv \
  --batch 16 \
  --output-dir runs/train/tky-exp/predictions
```

### Gowalla-CA

```bash
python predict.py \
  --checkpoint runs/train/gowalla-exp/checkpoints/best_epoch.state.pt \
  --data-test dataset/Gowalla-CA/gowalla_test.csv \
  --data-adj-mtx dataset/Gowalla-CA/graph_A.csv \
  --data-node-feats dataset/Gowalla-CA/graph_X.csv \
  --batch 16 \
  --output-dir runs/train/gowalla-exp/predictions
```

Prediction outputs (when `--output-dir` is set):

- `metrics.json`
- `predictions.jsonl`

---

## End-to-end recipes

### A) NYC CPU: setup → train → predict

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
EPOCHS=3 BATCH=8 NAME=nyc-cpu-e2e bash scripts/train_cpu_smoke.sh
cat runs/train/nyc-cpu-e2e/predictions/metrics.json
```

### B) NYC GPU: unpack → train 200 epochs → predict

```bash
pip install -r requirements.txt
unzip -o dataset/NYC.zip -d dataset/

python train.py \
  --batch 16 \
  --epochs 200 \
  --name nyc-gpu-e2e \
  --exist-ok

python predict.py \
  --checkpoint runs/train/nyc-gpu-e2e/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv \
  --batch 16 \
  --output-dir runs/train/nyc-gpu-e2e/predictions

cat runs/train/nyc-gpu-e2e/predictions/metrics.json
```

### C) Change only epochs / batch / device

```bash
# 10 epochs, batch 4, CPU
python train.py --epochs 10 --batch 4 --no-cuda --name run-e10-b4 --exist-ok

# 100 epochs, batch 20, default device
python train.py --epochs 100 --batch 20 --name run-e100-b20 --exist-ok

# 200 epochs on cuda:0
python train.py --epochs 200 --batch 16 --device cuda:0 --name run-cuda0 --exist-ok
```

---

## Citation

```
@inproceedings{10.1145/3477495.3531983,
  author = {Yang, Song and Liu, Jiamou and Zhao, Kaiqi},
  title = {GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {1144–1153},
  series = {SIGIR '22'}
}
```
