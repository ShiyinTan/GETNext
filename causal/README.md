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
- `s_pref`：兴趣通道；`s_conf`：混杂通道
- 总分：`s = w_pref * s_pref + w_conf * s_conf`（默认两个权重都是 1，等于附录 D 的直接相加）
- `s_conf` 内部：`w_acc * 距离 + w_pop * 热度 + w_area * 区域 + w_ctx * 情境`（默认也全是 1）

### 分数权重：加了超参，但不要六个一起搜

现在可以调，但 **默认全部 = 1，和改之前一模一样**。

| 开关 | 作用 | 建议 |
|------|------|------|
| `--w-pref` | 兴趣通道在总分里的音量 | 保持 1 |
| `--w-conf` | 混杂通道在总分里的音量 | **唯一建议搜索的**，例如 `{0.25, 0.5, 1, 2}` |
| `--w-acc` / `--w-pop` / `--w-area` / `--w-ctx` | `s_conf` 里四项谁更响 | 训练保持 1；设 0 做消融 |

为什么不用网格搜这 6 个：

1. **`g_acc` / `g_pop` / `g_area` 已经是可学习的尺度**（查表分、线性层）。训练时 `L_conf` 会把它们推向手工先验 `g̃`。再搜内部权重，多半和这些层互相抵消。
2. **`s_pref` 和 `s_conf` 的相对强弱** 才是一个自由度。只调 `--w-conf` 就够：写实榜太偏「近/热」就略降；混杂通道太弱、factual 几乎等于 deconf 就略升。
3. **更好的办法：训练用默认 1，预测时再扫 `--w-conf`，不必重训。**

```bash
# 同一份 checkpoint，只改混杂音量
python causal/predict.py --checkpoint ... --w-conf 0.5 --no-cuda
python causal/predict.py --checkpoint ... --w-conf 1.0 --no-cuda
python causal/predict.py --checkpoint ... --w-conf 2.0 --no-cuda
```

消融（预测时关掉某一项，看 Acc 掉多少）：

```bash
python causal/predict.py --checkpoint ... --w-acc 0 --no-cuda   # 不要距离
python causal/predict.py --checkpoint ... --w-pop 0 --no-cuda   # 不要热度
```

`lambda_*`（损失权重）和 `w_*`（分数音量）不是一回事：前者管训练时哪项 loss 更用力，后者管打分时哪路更响。

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
    → s_conf = w_acc g_acc + w_pop g_pop + w_area g_area + w_ctx <W_c h_c, ψ(p)>
    → s = w_pref s_pref + w_conf s_conf     # defaults all 1 (Appendix D)
    → L = L_main + λ_pref L_pref + λ_conf L_conf + λ_adv L_adv + λ_recon L_recon
```

`L_main` is CE on **total** `s` (fits `P(Y|H,C)`).
`L_pref` is same-distance-band CE on `s_pref` only.
`L_conf` aligns `s_conf` with a stop-grad hand-crafted `g̃` (near / popular / same-area).
`L_adv` / `L_recon` use a GRL so `h_z ≁ C` while `h_c` reconstructs discrete `C`.

---

## 安装环境（完整步骤）

因果代码和 GETNext **共用同一套环境**，没有单独的 `causal/requirements.txt`。始终在**仓库根目录**执行下面的命令。

### Python 版本

| 项 | 说明 |
|--|--|
| 支持 | **3.10 / 3.11 / 3.12**（本仓库在 3.12 上验证） |
| 不要用 | 3.9 及以下。原论文 `requirements.txt` 钉的是 `torch==1.7.1` + `numpy==1.19.2`，在现代 Python 上**没有轮子、装不上** |
| 不要用 | **3.13+**（`torch>=2.1,<2.5` 没有对应 wheel） |

只跑 `pip install -r requirements.txt` **不够**：`torch` 不在该文件里，必须按 CPU / GPU 先装对应 wheel。

原论文 freeze 里还有这些问题，已经从仓库里删掉：`data==0.4`（无关的 PyPI 包，代码从未 import）、`prettytable` / `matplotlib` / `torch_summary` / `torchsummary`（未使用）、`PyYAML==6.0`（6.0 在新 Python 上经常编不过，改用 `>=6.0.1`）。缺少的 `networkx`（`build_graph.py`）已补上。

### 1. 建虚拟环境

GPU 和 CPU **都是** `python3 -m venv`，命令一样。冲突的不是 Python，是 **同一个 venv 里只能有一份 `torch`**。

| 你的机器 | 用哪个目录 | 装哪种 torch |
|--|--|--|
| 只有 CPU | `.venv` | CPU wheel（`2.4.1+cpu`） |
| 只有 NVIDIA GPU | `.venv` | CUDA wheel（`2.4.1+cu121` 等） |
| 同一台机器两种都要 | `.venv-cpu` **和** `.venv-gpu` | 分开建、分开激活 |

不要把 CUDA torch 装进已经有 CPU torch 的 `.venv`：`2.4.1+cpu` 和 `2.4.1+cu121` 被 pip 当成同一版本，第二次 `pip install` 会提示 `Requirement already satisfied` 然后什么都不改。要用 GPU，要么新建 `.venv-gpu`，要么删掉旧环境再装。

```bash
cd /path/to/GETNext          # 仓库根目录，不是 causal/
python3 -m venv .venv        # 只有一种机器：就用这个名字
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -c "import sys; print(sys.version)"   # 确认是 3.10–3.12
```

同一台机器要并存 CPU / GPU 时：

```bash
python3 -m venv .venv-cpu
python3 -m venv .venv-gpu
# 用 CPU：source .venv-cpu/bin/activate
# 用 GPU：source .venv-gpu/bin/activate
```

### 2a. CPU（无 GPU / Cursor Cloud）

推荐一键脚本（建 venv、装 CPU torch、装依赖、解压 NYC）：

```bash
bash scripts/setup_cloud_env.sh
source .venv/bin/activate
```

手动等价步骤：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<2.5"
python -m pip install -r requirements-cpu.txt
unzip -o dataset/NYC.zip -d dataset/
```

### 2b. GPU（本机 CUDA）

先到 [pytorch.org](https://pytorch.org/get-started/locally/) 选和本机 CUDA 匹配的 index。例如 CUDA 12.1。

若这台机器**只有 GPU**，继续用上面的 `.venv`。若已经为 CPU 建过 `.venv`，不要往里面再装 CUDA torch，另建 `.venv-gpu`：

```bash
python3 -m venv .venv-gpu          # 或：本机只有 GPU 时用 .venv
source .venv-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.1,<2.5"
python -m pip install -r requirements.txt
unzip -o dataset/NYC.zip -d dataset/
```

其它常见 index：`cu118`（CUDA 11.8）、`cu124`（CUDA 12.4）。不要用默认 PyPI 的 `torch` 来代替上面的 index（CPU 机器会下到巨大的 CUDA 包）。

装好后版本字符串应带 `+cu121`（或你选的 CUDA），并且：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

有 GPU 时应打印 `True` 且 `device_count >= 1`。没有 NVIDIA 驱动的机器上，CUDA wheel 也能 import，但 `cuda.is_available()` 仍是 `False`，不能当训练验证。

### 3. 确认装好了

```bash
python - <<'PY'
import sys, torch, numpy, pandas, sklearn, yaml, tqdm, networkx, scipy
print("python", sys.version.split()[0])
print("torch ", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy ", numpy.__version__, "pandas", pandas.__version__)
PY
```

CPU 环境：`cuda` 为 `False`。venv/pip 的版本字符串带 `+cpu`；conda 的 CPU 包常常只显示 `2.4.1`。  
GPU 环境：有显卡时 `cuda` 为 `True`。venv/pip 带 `+cu121`（或 `cu118` / `cu124`）；conda 同样可能只显示 `2.4.1`，以 `cuda.is_available()` 为准。

新开 shell 时激活**当前要用的那个** venv（`.venv` / `.venv-cpu` / `.venv-gpu`）。

### Conda（和 venv 二选一）

也可以用 conda / mamba / micromamba，**不要和 `.venv` 同时 activate**。CPU / GPU 是两个环境名，和 venv 一样不能混装两份 torch。

需要先有 conda。没有的话装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Mambaforge，再开一个新终端。

**CPU：**

```bash
conda env create -f environment-cpu.yml
conda activate getnext-cpu
unzip -o dataset/NYC.zip -d dataset/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # False；conda CPU 常显示 2.4.1 而无 +cpu
```

一键（会解压 NYC）：

```bash
bash scripts/setup_conda_cpu.sh
conda activate getnext-cpu
```

**GPU（CUDA 12.1）：**

```bash
conda env create -f environment-gpu.yml
conda activate getnext-gpu
unzip -o dataset/NYC.zip -d dataset/
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

有显卡时应为 `True` 且 `device_count >= 1`。CUDA 11.8 把 `environment-gpu.yml` 里的 `pytorch-cuda=12.1` 改成 `11.8`。

同一台机器两种都要：

```bash
conda activate getnext-cpu
conda activate getnext-gpu
```

`mamba env create -f …` / `micromamba create -f …` 和上面的 yml 通用。装好后用第 3 节那段 `import torch, numpy, pandas…` 检查即可。

### 4. 数据文件

解压后至少要有：

- `dataset/NYC/NYC_train.csv`
- `dataset/NYC/NYC_val.csv`
- `dataset/NYC/NYC_test.csv`
- `dataset/NYC/graph_X.csv`（POI 表：坐标 / 类别；因果训练**不用** `graph_A.csv`）

### 5. 跑通一次（可选）

```bash
bash causal/run_cpu_smoke.sh
```

成功后会在 `runs/causal/<name>/predictions/metrics.json` 写出指标。

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
| `factual` | `w_pref s_pref + w_conf s_conf` with real `C(p)` | next hop under real constraints |
| `deconf_pref` | `s_pref` only | preferred `do(C)` interest ranking |
| `deconf_do` | `s_pref + s_conf(φ̄)` | access/pop replaced by training-mode buckets |
| `deconf_sum` | mix `g_acc` / `g_pop` over `P̂(c)` | cheap back-door marginalisation |

Outputs (when `--output-dir` is omitted, written next to the run):

- `predictions/metrics.json` — 顶层 `top1_acc` / `HR1` / `H5` / `H10` / `NDCG5` / `NDCG10` / `mAP20` / `mrr` 与 GETNext 相同（factual）；另含各 mode 的 overall + distance / pop / area slices
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
  ├── metrics-train.txt        # GETNext 同款字段 + HR/H/NDCG + pref/conf/adv/recon
  ├── metrics-val.txt          # GETNext 同款 factual 字段 + HR/H/NDCG + deconf
  ├── poi_table_meta.pkl
  ├── checkpoints/best_epoch.state.pt
  └── predictions/
        ├── metrics.json       # 顶层 top1_acc / HR1 / H5 / H10 / NDCG5 / NDCG10 / mAP20 / mrr
        └── predictions.jsonl
```

---

## What this spec does *not* include

Appendix D.8: IPS (§5.4) and Group DRO (§5.5) are orthogonal and not required.
Ring contrastive is already `L_pref`. Graph-edge reweighting is irrelevant because GCN is not used.

This is a deconfounded **training bias / dual inference** implementation, not a claim of identified causal effects (§3.3 / D.7).
