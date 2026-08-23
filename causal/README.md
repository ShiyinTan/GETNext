# Causal next-POI (Appendix D)

## 不懂代码时怎么读（建议顺序）

这个文件夹是「因果版 next-POI」，和原来的 GETNext **并排存在**，不会改 `train.py` / `model.py`。
可以把它想成一条流水线，而不是一堆神秘公式：

1. **用户走过的地点序列**（历史 H）送进 Transformer，得到一个向量 `h`（“读完这条轨迹后的摘要”）。
2. 把 `h` **拆成两份**：`h_z` 尽量只表示兴趣，`h_c` 尽量表示近/热/区等混杂。
3. 对每个候选地点打两个分再相加：`s = 兴趣分 s_pref + 混杂分 s_conf`。
4. 训练时：总分要能猜对下一站；兴趣分要在「一样远」的地点里比出偏好；混杂分要去学距离/热度；并且 `h_z` 不该轻易猜出混杂。
5. 预测时出两套榜：**factual**（真实约束下下一站）和 **deconf**（把近/热拿掉后兴趣指向谁）。

对应文件（从上到下读即可）：

| 文件 | 人话 |
|------|------|
| `param_parser.py` | 命令行开关：学多久、batch 多大、几项损失的权重 |
| `features.py` | 事先算好每个地点的距离桶、热度桶、区域 id（混杂 C） |
| `model.py` | 网络结构：编码 → 拆 `h_z/h_c` → 两个分数 |
| `train.py` | 训练循环，和 GETNext 很像：读 CSV → 按 batch 更新 → 存最好的模型 |
| `predict.py` | 加载模型，打出 factual / deconf 两套 top-k |
| `metrics.py` | Acc@k、MRR，以及按距离/热度/是否跨区切开的指标 |
| `run_cpu_smoke.sh` | CPU 上跑几步，确认能跑通 |

代码里每个函数的 docstring 都有 **输入 / 输出** 两节，形状用下面这套记号（不要当成具体数字）：

| 记号 | 含义 | NYC 大约 |
| --- | --- | --- |
| `B` | 一个 batch 有几条轨迹 | `--batch`，默认 20 |
| `T` | pad 后的轨迹长度 | 每条轨迹不同，batch 内取最长 |
| `N` | POI 词表大小 | ~4980 |
| `d` / `d_model` | Transformer 宽度 = poi+user+time+cat embed | 默认 128+128+32+32=320 |
| `d_z` | 兴趣向量 `h_z` 宽度，等于 `poi_embed_dim` | 默认 128 |
| `d_c` | 混杂向量 `h_c` 宽度，`--hc-dim` | 默认 64 |
| `K` | 距离桶数 `num_acc_bins` | 切分点 `0.5,1,2,5,10` → 通常 6 |
| `P` | 热度档数 `num_pop_bins` | `--pop-bins`，默认 4 |
| `A` | 区域数 `num_areas` | 网格压缩后，远小于棋盘格数 |
| `H` | 时刻桶数 `--time-units` | 默认 48 |
| `n_cat` / `n_user` | 类别数 / 用户数 | 由 CSV 统计 |
| `k` | top-k | 预测默认 20 |

常见 tensor（训练一个 batch 时）：

| 变量 | shape | 说明 |
| --- | --- | --- |
| `poi` / `cat` | `(B, T)` long | pad 位置填 0 |
| `time` | `(B, T)` float | 一天内归一化时刻 `[0,1]` |
| `user` | `(B,)` long | 整条轨迹同一个用户 |
| `pad` | `(B, T)` bool | True = pad，注意力忽略 |
| `y_poi` / `y_cat` | `(B, T)` long | 下一步真值，pad = `-1` |
| `h` | `(B, T, d)` | Transformer 输出 |
| `h_z` | `(B, T, d_z)` | 兴趣代理 |
| `h_c` | `(B, T, d_c)` | 混杂摘要 |
| `s` / `s_pref` / `s_conf` | `(B, T, N)` | 对每个候选 POI 的分 |
| `C` 表 `dist_bin` / `dist_km` | `(N, N)` | 起点→终点距离桶 / 公里 |
| `log_pop` / `pop_bin` / `area_id` | `(N,)` | 每个 POI 的热度/区域 |
| `e_p` 权重 | `(N, d_z)` | POI embedding，和输入查表绑定 |

符号对照：

- `H` / `poi`：历史轨迹（当前已访问的地点序列）
- `Y` / `y_poi`：下一站真值
- `C`：混杂（距离 `c_acc`、热度 `c_pop`、区域 `c_area`、时刻 `c_hour`）
- `h_z`：兴趣代理；`h_c`：混杂摘要
- `s_pref`：兴趣通道；`s_conf`：混杂通道；`s = s_pref + s_conf`

---

## English / 运行说明

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
