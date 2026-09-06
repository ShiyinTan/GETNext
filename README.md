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
# Same `python3 -m venv` as CPU. Do not reuse a CPU .venv — pip will skip the
# CUDA wheel (2.4.1+cpu and 2.4.1+cu121 look like the same version).
python3 -m venv .venv            # or .venv-gpu if .venv is already CPU
source .venv/bin/activate
python -m pip install --upgrade pip
# Example CUDA 12.1; pick cu118/cu121/cu124 at https://pytorch.org/get-started/locally/
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.1,<2.5"
python -m pip install -r requirements.txt
unzip -o dataset/NYC.zip -d dataset/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect +cu121 True
python train.py --batch 16 --epochs 200 --name nyc-gpu --exist-ok
python predict.py \
  --checkpoint runs/train/nyc-gpu/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv
```

---

## Installation

**Python:** 3.10, 3.11, or 3.12 (3.12 verified). Do not use 3.9- or 3.13+.

`torch` is **not** listed in `requirements.txt` because CPU and CUDA wheels come from different indexes. Install torch first, then the requirements file.

CPU and GPU both start with `python3 -m venv`. They must not share one venv: a venv can hold only one torch build. If `.venv` already has `2.4.1+cpu`, installing `2.4.1+cu121` into it is a no-op. Use `.venv` for whichever this machine is, or `.venv-cpu` and `.venv-gpu` if you need both.

### CPU-only

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<2.5"
python -m pip install -r requirements-cpu.txt
```

### GPU (CUDA)

```bash
python3 -m venv .venv            # use .venv-gpu if .venv is already a CPU env
source .venv/bin/activate
python -m pip install --upgrade pip
# Example: CUDA 12.1. Change cu121 to match your driver.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.1,<2.5"
python -m pip install -r requirements.txt
```

Always activate the venv in a new shell:

```bash
source .venv/bin/activate
```

Check the install:

```bash
python -c "import torch, numpy, pandas; print(torch.__version__, torch.cuda.is_available())"
# CPU venv:  2.4.1+cpu   False
# GPU venv:  2.4.1+cu121 True
```

Pick **either** venv or conda, not both in the same shell.

### Conda

Same Python range (3.10–3.12). CPU and GPU are **two env names**, same rule as venv: do not install both torch builds into one env.

**CPU:**

```bash
conda env create -f environment-cpu.yml
conda activate getnext-cpu
unzip -o dataset/NYC.zip -d dataset/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect False
```

Or: `bash scripts/setup_conda_cpu.sh` then `conda activate getnext-cpu`.

**GPU (CUDA 12.1):**

```bash
conda env create -f environment-gpu.yml
conda activate getnext-gpu
unzip -o dataset/NYC.zip -d dataset/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect True
```

For CUDA 11.8, edit `pytorch-cuda=12.1` in `environment-gpu.yml` to `11.8`. `mamba` / `micromamba` work with the same `-f` files.

Need both on one machine:

```bash
conda activate getnext-cpu    # CPU
conda activate getnext-gpu    # GPU
```

## Requirements

See `requirements.txt` (shared packages) and `requirements-cpu.txt` (includes the former). Key packages: NumPy, pandas, scikit-learn, SciPy, PyYAML, tqdm, networkx. Install PyTorch separately as above.

Conda users: `environment-cpu.yml` / `environment-gpu.yml` (same packages; do not mix with `.venv`).

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
