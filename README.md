# GETNext

This is the pytorch implementation of paper "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation"

![model-structure](figures/model-structure.png)

## Installation

GPU / original paper setup:

```bash
pip install -r requirements.txt
```

CPU-only (e.g. Cursor Cloud):

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
# or: pip install -r requirements-cpu.txt
```

## Requirements

See `requirements.txt` (GPU) or `requirements-cpu.txt` (CPU). Key packages include PyTorch, NumPy, pandas, scikit-learn, PyYAML, and tqdm.

## Data

1. Unzip a dataset under `dataset/` (NYC ships as `dataset/NYC.zip`):

   ```bash
   unzip -o dataset/NYC.zip -d dataset/
   ```

   Expected files for NYC:
   - `dataset/NYC/NYC_train.csv`
   - `dataset/NYC/NYC_val.csv`
   - `dataset/NYC/NYC_test.csv`
   - `dataset/NYC/graph_A.csv` / `graph_X.csv` (included in the zip)

2. If graph files are missing, build them from the training split:

   ```bash
   python build_graph.py
   ```

Other zips in this repo: `dataset/TKY.zip`, `dataset/Gowalla-CA.zip`. Point the `--data-*` flags at the corresponding paths after unpacking.

## Train

All hyper-parameters are defined in `param_parser.py`.

### Common flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--epochs` | Number of training epochs | `200` |
| `--batch` | Batch size | `20` |
| `--lr` | Learning rate | `0.001` |
| `--data-train` | Train CSV | `dataset/NYC/NYC_train.csv` |
| `--data-val` | Val CSV | `dataset/NYC/NYC_val.csv` |
| `--data-adj-mtx` | Graph adjacency | `dataset/NYC/graph_A.csv` |
| `--data-node-feats` | Graph node features | `dataset/NYC/graph_X.csv` |
| `--device` | Device string (`cuda` / `cpu`) | auto (`cuda` if available) |
| `--no-cuda` | Force CPU | off |
| `--name` | Run name under `runs/train/` | `exp` |
| `--exist-ok` | Reuse `project/name` without incrementing | off |
| `--verbose` | Print detailed per-batch sample dumps | off |

Console output is kept short by default: startup config, `tqdm` progress bars (train/val with live `loss` / `avg` / `top1`), and an aligned per-epoch metrics summary. Full debug logs go to `runs/train/<name>/log_training.txt`. Pass `--verbose` for the old noisy per-batch dumps.

### Example (paper-scale, GPU if available)

```bash
python train.py \
  --data-train dataset/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC_val.csv \
  --time-units 48 --time-feature norm_in_day_time \
  --poi-embed-dim 128 --user-embed-dim 128 \
  --time-embed-dim 32 --cat-embed-dim 32 \
  --node-attn-nhid 128 \
  --transformer-nhid 1024 \
  --transformer-nlayers 2 --transformer-nhead 2 \
  --batch 16 --epochs 200 --name exp1
```

### Example (CPU smoke)

```bash
python train.py \
  --epochs 3 --batch 8 --no-cuda \
  --name cpu-smoke --exist-ok \
  --poi-embed-dim 64 --user-embed-dim 64 \
  --time-embed-dim 16 --cat-embed-dim 16 \
  --node-attn-nhid 64 \
  --transformer-nhid 256 \
  --transformer-nlayers 2 --transformer-nhead 2
```

Or use the helper script:

```bash
EPOCHS=3 BATCH=8 NAME=cpu-smoke bash scripts/train_cpu_smoke.sh
```

Outputs:
- Training logs / checkpoints: `runs/train/<name>/`
- Best checkpoint: `runs/train/<name>/checkpoints/best_epoch.state.pt`
- Metrics history: `metrics-train.txt`, `metrics-val.txt`

## Predict

```bash
python predict.py \
  --checkpoint runs/train/cpu-smoke/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv \
  --no-cuda
```

Prediction metrics and top-k trajectories are written under the run’s `predictions/` directory (or `--output-dir` if set).

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
