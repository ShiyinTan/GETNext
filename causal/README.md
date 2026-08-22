# Causal next-POI (Appendix D)

This folder is a **standalone** implementation of
`docs/causal-nextpoi-thinking.md` **Appendix D**
(score decomposition §5.1 + dual representation / back-door training §5.2).

It does **not** modify the original GETNext files (`train.py`, `model.py`, …).
Run it next to GETNext, using the same NYC / TKY / Gowalla CSVs.

| GETNext | Causal (this folder) |
|---------|----------------------|
| GCN(`checkin_cnt`, trajectory-flow `A`) as `e_p` | `nn.Embedding` as `e_p` (Appendix A / D.8) |
| `NodeAttnMap` added to POI logits | no graph prior on `h_z` |
| tokens = user/POI/time/cat only | same tokens; **C is not concatenated into H** (D.3.4) |
| single CE over a fused logit | `s = s_pref + s_conf` with channel losses |
| one ranking | **factual** and **deconfounded** rankings (D.5, §7) |

Pipeline (same shape as GETNext):

```text
CSV trajectories
    → padded batch (B, T)
    → Transformer encoder (causal mask)
    → split h → (h_z, h_c)
    → s_pref = <h_z, e_p>
    → s_conf = g_acc + g_pop + g_area + <W_c h_c, ψ(p)>
    → L = L_main + λ_pref L_pref + λ_conf L_conf + λ_adv L_adv + λ_recon L_recon
```

`L_main` is CE on **total** `s` (fits `P(Y|H,C)`).
`L_pref` is same-distance-band CE on `s_pref` only.
`L_conf` aligns `s_conf` with a stop-grad hand-crafted `g̃` (near / popular / same-area).
`L_adv` / `L_recon` use a GRL so `h_z ≁ C` while `h_c` reconstructs discrete `C`.

---

## Setup

Same environment as GETNext.

**CPU (Cursor Cloud):**

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
```

**GPU:**

```bash
pip install -r requirements.txt
unzip -o dataset/NYC.zip -d dataset/
```

Always run from the **repo root**.

---

## Quick start

### CPU smoke (few batches, small model)

```bash
source .venv/bin/activate
bash causal/run_cpu_smoke.sh
```

Overrides:

```bash
EPOCHS=1 BATCH=4 NAME=causal-debug MAX_BATCHES=8 bash causal/run_cpu_smoke.sh
```

### CPU full-epoch mini run

```bash
python causal/train.py \
  --epochs 3 --batch 8 --no-cuda --name nyc-causal-cpu --exist-ok \
  --poi-embed-dim 64 --user-embed-dim 64 --time-embed-dim 16 --cat-embed-dim 16 \
  --hc-dim 32 --transformer-nhid 256 --transformer-nlayers 2 --transformer-nhead 2
```

### GPU (paper-scale-ish)

```bash
python causal/train.py \
  --device cuda --batch 16 --epochs 200 --name nyc-causal-gpu --exist-ok \
  --poi-embed-dim 128 --user-embed-dim 128 --time-embed-dim 32 --cat-embed-dim 32 \
  --hc-dim 64 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2
```

Force a specific GPU:

```bash
python causal/train.py --device cuda:0 --batch 16 --epochs 200 --name nyc-cuda0 --exist-ok
```

`graph_A.csv` is **not** an input. Only the POI table `graph_X.csv` is used (coords / category / fallback counts). Popularity `C_pop` is counted from **train** check-ins.

---

## Predict

```bash
python causal/predict.py \
  --checkpoint runs/causal/nyc-causal-cpu/checkpoints/best_epoch.state.pt \
  --data-test dataset/NYC/NYC_test.csv \
  --no-cuda \
  --modes factual,deconf_pref,deconf_do,deconf_sum
```

GPU: drop `--no-cuda`.

`--modes` (Appendix D.5):

| Mode | Score | Question |
|------|--------|----------|
| `factual` | `s_pref + s_conf` with real `C(p)` | next hop under real constraints |
| `deconf_pref` | `s_pref` only | preferred `do(C)` interest ranking |
| `deconf_do` | `s_pref + s_conf(φ̄)` | access/pop replaced by training-mode buckets |
| `deconf_sum` | mix `g_acc` / `g_pop` over `P̂(c)` | cheap back-door marginalisation |

Outputs (when `--output-dir` is omitted, written next to the run):

- `predictions/metrics.json` — overall + distance / pop / area slices for **each** mode
- `predictions/predictions.jsonl` — per-trajectory top-k (factual and `deconf_pref`)

Do not pick checkpoints with deconfounded Acc (spec §7). Training monitors **factual** Acc@1 / Acc@20.

---

## Important flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--lambda-pref` | ring-contrastive on `s_pref` | `0.05` |
| `--lambda-conf` | `s_conf` ↔ `g̃` MSE | `0.05` |
| `--lambda-adv` | GRL adversarial CE on `h_z` | `0.05` |
| `--lambda-recon` | `h_c` reconstructs `C` | `0.05` |
| `--lambda-cat` | category aux from `h_z` | `0.05` |
| `--lambda-time` | GETNext-style time MSE (off by default) | `0.0` |
| `--dist-bins` | km edges for `c_acc` | `0.5,1,2,5,10` |
| `--pop-bins` | pop quantiles | `4` |
| `--area-grid-deg` | lat/lon grid | `0.02` |
| `--conf-aux-ce` | optional weak CE on `s_conf` | off |
| `--max-batches` | smoke cap on train batches / epoch | `0` (all) |
| `--project` | run root | `runs/causal` |
| `--no-cuda` | force CPU | off |

---

## Outputs

```text
runs/causal/<name>/
  ├── args.yaml
  ├── log_training.txt
  ├── metrics-train.txt
  ├── metrics-val.txt          # factual and deconf_pref Acc@k
  ├── poi_table_meta.pkl
  ├── checkpoints/best_epoch.state.pt
  └── predictions/
        ├── metrics.json
        └── predictions.jsonl
```

---

## What this spec does *not* include

Appendix D.8: IPS (§5.4) and Group DRO (§5.5) are orthogonal and not required.
Ring contrastive is already `L_pref`. Graph-edge reweighting is irrelevant because GCN is not used.

This is a deconfounded **training bias / dual inference** implementation, not a claim of identified causal effects (§3.3 / D.7).
