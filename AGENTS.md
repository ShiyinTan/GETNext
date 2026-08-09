# GETNext

## Cursor Cloud specific instructions

This repo trains GETNext (next-POI recommendation) on the NYC check-in dataset.

### Environment
- Cloud VM is **CPU-only**. Use `requirements-cpu.txt` and the CPU PyTorch wheel.
- Setup is defined in `.cursor/environment.json` (default Cursor image + install script). No custom Dockerfile.
- `scripts/setup_cloud_env.sh` creates `.venv`, installs CPU torch, and unpacks NYC.
- Always `source .venv/bin/activate` before running Python if the shell is fresh.
- NYC data ships as `dataset/NYC.zip` and already includes `graph_A.csv` / `graph_X.csv`.

### Smoke train + predict (few epochs)
```bash
source .venv/bin/activate  # if needed
bash scripts/train_cpu_smoke.sh
```

Optional overrides:
```bash
EPOCHS=3 BATCH=8 NAME=cpu-smoke bash scripts/train_cpu_smoke.sh
```

Outputs:
- Training logs / checkpoints: `runs/train/<name>/`
- Prediction metrics: `runs/train/<name>/predictions/metrics.json`
- Per-trajectory top-k: `runs/train/<name>/predictions/predictions.jsonl`

### Manual commands
```bash
# Train a few epochs on CPU
python train.py --epochs 3 --batch 8 --no-cuda --name cpu-smoke --exist-ok \
  --poi-embed-dim 64 --user-embed-dim 64 --time-embed-dim 16 --cat-embed-dim 16 \
  --node-attn-nhid 64 --transformer-nhid 256 --transformer-nlayers 2 --transformer-nhead 2

# Predict on test split
python predict.py --checkpoint runs/train/cpu-smoke/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv --no-cuda
```

Do not attempt CUDA/`nvidia-smi` checks in this environment.
