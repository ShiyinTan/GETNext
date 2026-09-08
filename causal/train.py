"""训练附录 D 的因果 next-POI 模型。

整体流程和 GETNext 几乎一样，方便对照：
  读轨迹 CSV → 按轨迹做成样本 → 补齐成 batch → 前向算分 → 算损失 → 反向更新
  → 每个 epoch 在验证集上看 Acc@k / mAP@20 / MRR，并立刻在测试集上再评一次（方便看能力；选 checkpoint 仍只用 val） → 存最好的 checkpoint

和 GETNext 不同、按因果规格改掉的部分：
  - 不用 GCN、不用 NodeAttnMap（e_p 仍是 nn.Embedding）
  - graph_A 拆成转移热度（进 s_conf）和残差相关（仅 factual 的 s_rel）
  - 混杂 C 不进 Transformer token
  - 损失是「总分 CE + 兴趣环带 + 混杂对齐 + 对抗 + 重建 + 时间 MSE」
  - 验证时同时报 factual（总分）和 deconf（只用兴趣分）两套指标
    选 checkpoint 只用 factual，避免用去混淆分数去刷写实 Acc（§7）

形状记号：B=batch，T=pad 后长度，N=POI 数，d / d_z / d_c 见 causal/README.md。
collate 之后一个 batch 的关键 tensor：
  poi/cat (B,T)  time (B,T)  user (B,)  pad (B,T) bool
  y_poi/y_cat (B,T)  pad 处为 -1 ； y_time (B,T) pad 处为 -1.0

读进来的表长什么样（NYC，和 GETNext 同一套 CSV）
----------------------------------------------
train_df / val_df / test_df = 签到明细，一行一次 check-in，不是一条轨迹一行。
  文件: dataset/NYC/NYC_train.csv （约 8.3 万行 / 1.1 万条轨迹 / 1047 用户）
        dataset/NYC/NYC_val.csv
        dataset/NYC/NYC_test.csv  （每个 epoch 的 val 之后评一次；不参与选模型）
  本文件真正用到的列：
    user_id            整数，如 470
    POI_id             Foursquare 字符串 id，如 '49bbd6c0f964a520f4531fe3'
    trajectory_id      '{user_id}_{第几段}'，如 '470_1'；同一 id 的多行按时间排成一条轨迹
    norm_in_day_time   [0, 1]，一天里的时刻（0.583 ≈ 下午 2 点）；列名由 --time-feature 指定
  CSV 里还有但这里不用：POI_catid / latitude / longitude / UTC_time / local_time …
    （类别和经纬度改从 nodes_df 取，保证和词表下标对齐）

  同一条轨迹的几行（示意）：
    user_id  POI_id                        norm_in_day_time  trajectory_id
    1000     4fbfe16ae4b0...               0.50              1000_16
    1000     4a513b17f964...               0.33              1000_16
    1000     42911d00f964...               0.35              1000_16
    1000     42911d00f964...               0.67              1000_16
  → Dataset 切成 输入 poi=[p0,p1,p2]  标签 y=[p1,p2,p3]

nodes_df = 地点表，一行一个 POI，行顺序就是模型里的下标 0..N-1。
  文件: dataset/NYC/graph_X.csv （NYC 约 4980 行，不是邻接矩阵）
    node_name/poi_id   和 train_df['POI_id'] 同一套字符串
    checkin_cnt        兜底热度；有训练集时会被 train 里的次数覆盖
    poi_catid          类别字符串，如 '4bf58dd8d48988d11d941735'（约 300 类）
    latitude / longitude
  邻接矩阵 graph_A.csv 行序与 graph_X 相同；这里不送进 GCN，只拆成
    log_tpop（目的地入度，进 s_conf）和 a_rel（残差相关，仅 factual）。
"""
import logging
import os
import pickle
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 保证从仓库根目录也能 import 原来的 utils.py 和 causal.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal.features import (
    build_poi_table,
    fill_graph_transition_tables,
    fill_transition_priors,
    load_nodes_df,
)
from causal.metrics import batch_last_step_metrics
from causal.model import CausalNextPOI
from causal.param_parser import parameter_parser
from utils import (
    RANKING_METRIC_KEYS,
    epoch_ckpt_metrics,
    format_epoch_summary,
    format_ranking_lines,
    increment_path,
    mean_or_nan,
    write_epoch_metrics_txt,
    zipdir,
)

SEP = '-' * 72


class TqdmLoggingHandler(logging.Handler):
    """日志走 tqdm.write，进度条才不会被 print 打乱（和 GETNext 相同）。"""

    def emit(self, record):
        """
        输入: logging.LogRecord（无 tensor）
        输出: 无返回，把格式化后的字符串写到屏幕
        """
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(save_dir, verbose=False):
    """文件里记全量日志；屏幕默认只显示关键 INFO。

    输入:
        save_dir: str
        verbose: bool
    输出:
        无返回。副作用：配置 root logger。
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'log_training.txt'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    root.addHandler(file_handler)
    console = TqdmLoggingHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    root.addHandler(console)


def resolve_device(args):
    """有 GPU 且没开 --no-cuda 就用 CUDA，否则 CPU。云端环境是 CPU。

    输入:
        args.no_cuda / args.device: 标量
    输出:
        torch.device
    """
    if args.no_cuda or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(args.device)


class TrajectoryDataset(Dataset):
    """把一条签到轨迹切成「用前缀预测下一步」，和 GETNext 相同。

    输入 df 是签到明细（train_df / val_df），按 trajectory_id 分组。
    每组多行 = 同一条轨迹上按时间排列的若干次 check-in。

    例：地点 [A, B, C, D]
      输入 (poi): A, B, C
      标签 (label_poi): B, C, D
    评估时只看最后一个时间步（用 A,B,C 预测 D），也和 GETNext 相同。
    """

    def __init__(self, df, poi_id2idx, user_id2idx, poi_idx2cat_idx, time_col,
                 min_len, skip_unknown_user=False):
        """
        输入:
            df: DataFrame，签到行。用到的列：
                trajectory_id, POI_id, time_col（默认 norm_in_day_time）
                用户从 trajectory_id 的下划线前半段解析，如 '470_1' → user '470'
            poi_id2idx: {原始 POI 字符串: 0..N-1}，来自 nodes_df 行序
            user_id2idx: {用户字符串: 0..n_user-1}，只含训练集出现过的用户
            poi_idx2cat_idx: {POI 下标: 类别下标}，来自 nodes_df['poi_catid']
            time_col: str
            min_len: int，输入前缀最短长度（默认 2，所以原始轨迹至少 3 个点）
            skip_unknown_user: bool，验证/测试时丢掉训练没见过的用户
        输出:
            无返回。self.samples: list[dict]，还不是 tensor。一条样本一例：
              traj_id='1000_16'  user_idx=3
              poi=[12, 45, 7]          # 长度 T_i，已映射成 0..N-1
              cat=[2, 8, 8]
              time=[0.50, 0.33, 0.35]  # 与 poi 对齐的日内时刻
              label_poi=[45, 7, 7]     # 下一步地点
              label_cat=[8, 8, 8]
              label_time=[0.33, 0.35, 0.67]
        """
        self.samples = []
        grouped = df.groupby('trajectory_id', sort=False)
        n_traj = df['trajectory_id'].nunique()
        for traj_id, traj_df in tqdm(grouped, total=n_traj,
                                      desc='Build trajectories', leave=False, dynamic_ncols=True):
            # trajectory_id 形如 "470_1"，下划线前面是 user_id
            user_id = str(str(traj_id).split('_')[0])
            if skip_unknown_user and user_id not in user_id2idx:
                continue
            if user_id not in user_id2idx:
                continue
            poi_idxs, times = [], []
            for poi_id, t in zip(traj_df['POI_id'].tolist(), traj_df[time_col].tolist()):
                if poi_id not in poi_id2idx:
                    continue  # 验证/测试里出现训练集没见过的地点则跳过该点
                poi_idxs.append(poi_id2idx[poi_id])
                times.append(float(t))
            if len(poi_idxs) < min_len + 1:
                continue
            inp = poi_idxs[:-1]
            lab = poi_idxs[1:]
            inp_t = times[:-1]
            lab_t = times[1:]
            if len(inp) < min_len:
                continue
            self.samples.append({
                'traj_id': traj_id,
                'user_idx': user_id2idx[user_id],
                'poi': inp,
                'time': inp_t,
                'label_poi': lab,
                'label_time': lab_t,
                'label_cat': [poi_idx2cat_idx[p] for p in lab],
                'cat': [poi_idx2cat_idx[p] for p in inp],
            })

    def __len__(self):
        """
        输入: 无
        输出: int，轨迹条数
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        输入:
            index: int
        输出:
            dict（Python list，不是 tensor），见 __init__
        """
        return self.samples[index]


def collate_pad(batch):
    """把一个 batch 里长短不一的轨迹补齐到相同 T。

    补齐位置：
      输入填 0（后面用 pad=True 让注意力忽略）
      标签填 -1（CrossEntropy 的 ignore_index，不算进损失）

    输入:
        batch: list[dict]，长度 B，每条是 Dataset 样本（Python list）
    输出: dict
        poi / cat: (B, T) long
        time:      (B, T) float
        user:      (B,)   long
        pad:       (B, T) bool，True=pad
        y_poi / y_cat: (B, T) long，pad=-1
        y_time:    (B, T) float，pad=-1.0
        lengths:   list[int] 长度 B，每条真实 T_i
        traj_ids:  list[str] 长度 B
    """
    lengths = [len(s['poi']) for s in batch]
    bsz, tmax = len(batch), max(lengths)
    poi = torch.zeros(bsz, tmax, dtype=torch.long)
    cat = torch.zeros(bsz, tmax, dtype=torch.long)
    time = torch.zeros(bsz, tmax, dtype=torch.float)
    y_poi = torch.full((bsz, tmax), -1, dtype=torch.long)
    y_time = torch.full((bsz, tmax), -1.0, dtype=torch.float)
    y_cat = torch.full((bsz, tmax), -1, dtype=torch.long)
    pad = torch.ones(bsz, tmax, dtype=torch.bool)
    user = torch.zeros(bsz, dtype=torch.long)
    traj_ids = []
    for i, s in enumerate(batch):
        n = lengths[i]
        poi[i, :n] = torch.tensor(s['poi'], dtype=torch.long)
        cat[i, :n] = torch.tensor(s['cat'], dtype=torch.long)
        time[i, :n] = torch.tensor(s['time'], dtype=torch.float)
        y_poi[i, :n] = torch.tensor(s['label_poi'], dtype=torch.long)
        y_time[i, :n] = torch.tensor(s['label_time'], dtype=torch.float)
        y_cat[i, :n] = torch.tensor(s['label_cat'], dtype=torch.long)
        pad[i, :n] = False
        user[i] = s['user_idx']
        traj_ids.append(s['traj_id'])
    return {
        'poi': poi, 'cat': cat, 'time': time, 'user': user, 'pad': pad,
        'y_poi': y_poi, 'y_time': y_time, 'y_cat': y_cat,
        'lengths': lengths, 'traj_ids': traj_ids,
    }


def masked_mse(pred, target, ignore=-1):
    """时间回归用：跳过标签为 -1 的补齐位置。

    输入:
        pred:   (B, T)  模型预测的时刻
        target: (B, T)  真值，pad 处为 ignore
        ignore: 标量，默认 -1
    输出:
        标量 tensor ()  MSE；若全是 pad 则返回 0
    """
    mask = target != ignore
    if mask.sum() == 0:
        return pred.new_zeros(())
    return ((pred[mask] - target[mask]) ** 2).mean()


def gather_c_of_y(origin, y_poi, time_feat, buffers, num_hour_bins):
    """取出「真值下一站 Y」上的离散混杂 C，给对抗 / 重建损失用（D.4.4）。

    补齐位置一律写成 -1，后面 CE 会忽略。

    输入:
        origin:    (B, T) long，当前步 POI
        y_poi:     (B, T) long，下一步真值，pad=-1
        time_feat: (B, T) float
        buffers:   dist_bin (N,N), pop_bin (N,), area_id (N,)
        num_hour_bins: int = H
    输出: 四个 (B, T) long
        c_acc:  起点→Y 的距离桶
        c_pop:  Y 的热度档
        c_area: Y 的区域
        c_hour: 当前时刻桶
        pad 位置全是 -1
    """
    valid = y_poi >= 0
    safe_o = origin.clamp(min=0)
    safe_y = y_poi.clamp(min=0)
    dist_row = buffers['dist_bin'][safe_o]                    # (B, T, N)
    c_acc = torch.gather(dist_row, 2, safe_y.unsqueeze(-1)).squeeze(-1)
    c_pop = buffers['pop_bin'][safe_y]
    c_area = buffers['area_id'][safe_y]
    c_hour = (time_feat.clamp(0, 0.999999) * num_hour_bins).long()
    ignore = torch.full_like(c_acc, -1)
    c_acc = torch.where(valid, c_acc, ignore)
    c_pop = torch.where(valid, c_pop, ignore)
    c_area = torch.where(valid, c_area, ignore)
    c_hour = torch.where(valid, c_hour, ignore)
    return c_acc, c_pop, c_area, c_hour


def compute_losses(model, batch, buffers, args, ce):
    """一次前向：编码 → 打分 → 五项损失（附录 D.4 / 算法 1）。

    L = L_main
      + λ_pref  * 同距离环带 CE（只在 s_pref 上）
      + λ_conf  * s_conf 对齐手工先验 g̃
      + λ_adv   * 用 h_z 猜 C（GRL 已在模型里反转梯度）
      + λ_recon * 用 h_c 重建 C
      + λ_cat   * 从 h_z 猜下一站类别（可选）
      + λ_time  * 时间 MSE（默认 10，与 GETNext --time-loss-weight 对齐）

    输入:
        model: CausalNextPOI
        batch: collate_pad 的 dict，见该函数输出
        buffers: table.to_torch() 的 dict
        args: 超参
        ce: CrossEntropyLoss(ignore_index=-1)
    输出: dict
        loss / main / pref / conf / adv / recon / cat / time: 标量 tensor
        s / s_pref: (B, T, N)
        h: (B, T, d)  h_z: (B, T, d_z)  h_c: (B, T, d_c)
        c_acc / c_pop / c_area: (B, T) long
    """
    poi = batch['poi']
    h, h_z, h_c = model.encode(poi, batch['time'], batch['cat'], batch['user'], batch['pad'])
    s, s_pref, s_conf, _ = model.score(h_z, h_c, poi, buffers, mode='factual', h=h)
    y = batch['y_poi']

    # D.4.1 主损失：总分 s 做全词表 CE，拟合 P(Y|H,C)
    loss_main = ce(s.transpose(1, 2), y)

    # D.4.2 兴趣通道：只在「和真值同一距离桶」的地点里做 softmax
    # 这样模型不能靠「更近」取巧，必须在一样远的集合里比偏好
    c_acc, c_pop, c_area, c_hour = gather_c_of_y(
        poi, y, batch['time'], buffers, args.time_units)
    dist_full = buffers['dist_bin'][poi.clamp(min=0)]
    ring = dist_full == c_acc.unsqueeze(-1)
    s_ring = s_pref.masked_fill(~ring, torch.tensor(-1e9, device=s_pref.device, dtype=s_pref.dtype))
    loss_pref = ce(s_ring.transpose(1, 2), y)

    # D.4.3 混杂通道：s_conf 去贴「近则高、热则高、同区则高」的手工分（不反传到 g̃）
    g_tilde = model.g_tilde(
        poi, buffers, args.align_alpha, args.align_beta,
        getattr(args, 'align_gamma', 0.3)).detach()
    valid = (y >= 0).unsqueeze(-1).float()
    denom = valid.sum() * s_conf.size(-1)
    # 对齐的是打分用的 s_conf（已含内部 w_*）。训练时内部权重请保持 1，以免和 g_* 对打。
    loss_conf = (((s_conf - g_tilde) ** 2) * valid).sum() / denom.clamp(min=1.0)
    if args.conf_aux_ce:
        # 可选：再给 s_conf 一个很弱的 CE；默认关掉，以免混杂通道抢走兴趣信号
        loss_conf = loss_conf + 0.1 * ce(s_conf.transpose(1, 2), y)

    # D.4.4 拆表征：h_z 不该轻易猜中 C；h_c 应该能重建 C
    adv = model.adv_logits(h_z, lambd=1.0)
    recon = model.recon_logits(h_c)
    targets = (c_acc, c_pop, c_area, c_hour)
    loss_adv = sum(ce(logit.transpose(1, 2), tgt) for logit, tgt in zip(adv, targets)) / 4.0
    loss_recon = sum(ce(logit.transpose(1, 2), tgt) for logit, tgt in zip(recon, targets)) / 4.0

    loss_cat = ce(model.cat_head(h_z).transpose(1, 2), batch['y_cat'])
    loss_time = masked_mse(model.time_head(h).squeeze(-1), batch['y_time'])

    loss = (loss_main
            + args.lambda_pref * loss_pref
            + args.lambda_conf * loss_conf
            + args.lambda_adv * loss_adv
            + args.lambda_recon * loss_recon
            + args.lambda_cat * loss_cat
            + args.lambda_time * loss_time)
    parts = {
        'loss': loss, 'main': loss_main, 'pref': loss_pref, 'conf': loss_conf,
        'adv': loss_adv, 'recon': loss_recon, 'cat': loss_cat, 'time': loss_time,
        's': s, 's_pref': s_pref, 'h': h, 'h_z': h_z, 'h_c': h_c,
        'c_acc': c_acc, 'c_pop': c_pop, 'c_area': c_area,
    }
    return parts


def _batch_rank_metrics(parts, batch, score_key):
    """和 GETNext 一样：一个 batch 里每条轨迹只评最后一步，再对 batch 取平均。"""
    y_np = batch['y_poi'].detach().cpu().numpy()
    s_np = parts[score_key].detach().cpu().numpy()
    return batch_last_step_metrics(y_np, s_np, batch['lengths'])


def _eval_split(model, loader, buffers, args, ce, max_batches, desc):
    """无梯度跑一个 split：factual（总分 s）+ deconf（兴趣分 s_pref）。

    聚合方式和 GETNext 相同：每个 batch 先对轨迹取 last-step 均值，再对 batch 取均值。
    """
    rank_keys = RANKING_METRIC_KEYS
    parts_acc = {k: [] for k in ('loss', 'poi', 'time', 'cat') + rank_keys}
    deconf_acc = {k: [] for k in rank_keys}
    with torch.no_grad():
        bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for b_idx, batch in enumerate(bar):
            if max_batches and b_idx >= max_batches:
                break
            batch = _to_device(batch, args.device)
            parts = compute_losses(model, batch, buffers, args, ce)
            fact_m = _batch_rank_metrics(parts, batch, 's')
            deconf_m = _batch_rank_metrics(parts, batch, 's_pref')
            parts_acc['loss'].append(float(parts['loss'].detach().cpu()))
            parts_acc['poi'].append(float(parts['main'].detach().cpu()))
            parts_acc['time'].append(float(parts['time'].detach().cpu()))
            parts_acc['cat'].append(float(parts['cat'].detach().cpu()))
            for k in rank_keys:
                parts_acc[k].append(fact_m[k])
                deconf_acc[k].append(deconf_m[k])
            bar.set_postfix(
                loss=f'{parts_acc["loss"][-1]:.2f}',
                avg=f'{float(np.mean(parts_acc["loss"])):.2f}',
                top1=f'{parts_acc["top1"][-1]:.3f}',
                refresh=False)
    return (
        {k: mean_or_nan(v) for k, v in parts_acc.items()},
        {k: mean_or_nan(v) for k, v in deconf_acc.items()},
    )


def train(args):
    """完整训练循环：读数据 → 建表 → epoch 更新 → 按 factual Acc 存 checkpoint。

    输入:
        args: argparse.Namespace，见 param_parser
    输出:
        无返回。写到 args.save_dir：
          checkpoints/best_epoch.state.pt
          poi_table_meta.pkl
          metrics-train.txt / metrics-val.txt / metrics-test.txt
    """
    # ---------- 0. 目录、日志、把本次参数存下来 ----------
    args.device = resolve_device(args)
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    os.makedirs(args.save_dir, exist_ok=True)
    setup_logger(args.save_dir, verbose=args.verbose)
    logging.info(SEP)
    logging.info(' Causal next-POI training (Appendix D)')
    logging.info(f' save_dir : {args.save_dir}')
    logging.info(f' device   : {args.device}')
    logging.info(f' epochs   : {args.epochs}  batch={args.batch}  lr={args.lr}')
    logging.info(f' lambdas  : pref={args.lambda_pref} conf={args.lambda_conf} '
                 f'adv={args.lambda_adv} recon={args.lambda_recon} '
                 f'cat={args.lambda_cat} time={args.lambda_time}')
    logging.info(f' score w  : pref={args.w_pref} conf={args.w_conf} '
                 f'acc={args.w_acc} pop={args.w_pop} tpop={args.w_tpop} '
                 f'area={args.w_area} ctx={args.w_ctx} rel={args.w_rel}')
    logging.info(f' decoder  : pref={getattr(args, "pref_decoder", "tied")} '
                 f'rel_source={getattr(args, "rel_source", "residual")}')
    logging.info(f' train    : {args.data_train}')
    logging.info(f' val      : {args.data_val}')
    logging.info(f' graph_A  : {args.data_adj_mtx}')
    if args.eval_test:
        logging.info(f' test     : {args.data_test}  (monitor only; ckpt uses val)')
    logging.info(SEP)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump({k: (str(v) if k == 'device' else v) for k, v in vars(args).items()},
                  f, sort_keys=False)
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(ROOT / 'causal', zipf, include_format=['.py'])
    zipf.close()

    # ---------- 1. 读 CSV，建立 id→下标，再算混杂表 C ----------
    logging.info('[1/4] Loading trajectories & POI confounder table...')
    # 签到明细：一行一次 check-in。NYC_train 约 8.3 万行，列见文件头注释。
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)
    test_df = None
    if args.eval_test:
        if os.path.isfile(args.data_test):
            test_df = pd.read_csv(args.data_test)
        else:
            logging.warning(f'Test CSV not found ({args.data_test}); skipping in-training test eval')
    # 地点表 graph_X.csv：一行一个 POI。NYC 约 4980 行 ×
    #   node_name/poi_id, checkin_cnt, poi_catid, poi_catid_code, poi_catname, latitude, longitude
    nodes_df = load_nodes_df(args.data_node_feats)

    # 词表下标必须和 Embedding 行对齐。POI 按下表行序 0..N-1，不要按字符串排序。
    poi_ids = list(nodes_df['node_name/poi_id'].tolist())
    poi_id2idx = dict(zip(poi_ids, range(len(poi_ids))))  # '49bbd6c0…' → 17
    cat_ids = list(dict.fromkeys(nodes_df[args.feature2].tolist()))  # 出现顺序，NYC ~313 类
    cat_id2idx = dict(zip(cat_ids, range(len(cat_ids))))  # '4bf58dd8…d11d941735' → 0
    poi_idx2cat_idx = {}
    for _, row in nodes_df.iterrows():
        poi_idx2cat_idx[poi_id2idx[row['node_name/poi_id']]] = cat_id2idx[row[args.feature2]]
    # 用户词表只用训练集，验证集里没见过的用户后面会丢掉
    user_ids = [str(u) for u in sorted(set(train_df['user_id'].astype(str).tolist()))]
    user_id2idx = dict(zip(user_ids, range(len(user_ids))))  # '470' → 12；NYC 训练集约 1047 人

    table = build_poi_table(nodes_df, train_df, args, poi_id2idx)
    graph_stats = fill_graph_transition_tables(
        table, args.data_adj_mtx, args.data_node_feats, poi_id2idx)
    if graph_stats.get('loaded'):
        logging.info(
            f'        graph_A loaded: mapped={graph_stats["n_mapped"]}/{graph_stats["n_pois"]} '
            f'nnz={graph_stats["nnz"]} tpop_max={graph_stats["tpop_max"]:.3f} '
            f'a_rel_std={graph_stats["a_rel_std"]:.3f}')
    else:
        logging.warning(
            f'graph_A not used ({graph_stats.get("reason", "unknown")}); '
            f'log_tpop/a_rel stay zero ({args.data_adj_mtx})')

    # ---------- 2. Dataset / DataLoader（验证集丢掉训练没见过的用户）----------
    logging.info('[2/4] Building dataloaders...')
    train_ds = TrajectoryDataset(
        train_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        args.time_feature, args.short_traj_thres, skip_unknown_user=False)
    val_ds = TrajectoryDataset(
        val_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        args.time_feature, args.short_traj_thres, skip_unknown_user=True)
    test_ds = None
    if test_df is not None:
        test_ds = TrajectoryDataset(
            test_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
            args.time_feature, args.short_traj_thres, skip_unknown_user=True)

    train_pairs = []
    for s in train_ds.samples:
        for o, d in zip(s['poi'], s['label_poi']):
            train_pairs.append((o, d))  # 都是 0..N-1 的 POI 下标，不是原始字符串 id
    fill_transition_priors(table, train_pairs)

    test_n = len(test_ds) if test_ds is not None else 0
    logging.info(f'        POIs={table.num_pois} cats={len(cat_id2idx)} users={len(user_id2idx)} '
                 f'train_trajs={len(train_ds)} val_trajs={len(val_ds)} test_trajs={test_n} '
                 f'areas={table.num_areas} acc_bins={table.num_acc_bins}')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=False,
        num_workers=args.workers, collate_fn=collate_pad)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.workers, collate_fn=collate_pad)
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False, drop_last=False,
            num_workers=args.workers, collate_fn=collate_pad)

    # ---------- 3. 搭模型（没有 GCN）----------
    logging.info('[3/4] Building causal model (no GCN / NodeAttnMap)...')
    model = CausalNextPOI(args, table.num_pois, len(user_id2idx), len(cat_id2idx), table)
    model = model.to(args.device)
    buffers = table.to_torch(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=args.lr_scheduler_factor)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=False, factor=args.lr_scheduler_factor)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    # 预测时要按同一套分桶重建距离表，把统计量存下来
    with open(os.path.join(args.save_dir, 'poi_table_meta.pkl'), 'wb') as f:
        pickle.dump({
            'poi_id2idx': poi_id2idx,
            'cat_id2idx': cat_id2idx,
            'user_id2idx': user_id2idx,
            'poi_idx2cat_idx': poi_idx2cat_idx,
            'pop': table.pop,
            'lat': table.lat,
            'lon': table.lon,
            'area_id': table.area_id,
            'pop_bin': table.pop_bin,
            'dist_edges': table.dist_edges,
            'pop_edges': table.pop_edges,
            'acc_prior': table.acc_prior,
            'pop_prior': table.pop_prior,
            'median_acc_bin': table.median_acc_bin,
            'median_pop_bin': table.median_pop_bin,
            'num_areas': table.num_areas,
            'num_acc_bins': table.num_acc_bins,
            'num_pop_bins': table.num_pop_bins,
            'log_tpop': table.log_tpop,
            'a_rel': table.a_rel,
            'a_raw': table.a_raw,
        }, f)

    # ---------- 4. epoch 循环 ----------
    logging.info('[4/4] Start training...')
    max_val_score = -np.inf
    train_hist, val_hist, test_hist = [], [], []
    rank_keys = RANKING_METRIC_KEYS
    train_loss_keys = ('loss', 'poi', 'time', 'cat', 'pref', 'conf', 'adv', 'recon')

    for epoch in range(args.epochs):
        model.train()
        tr_parts = {k: [] for k in train_loss_keys + rank_keys}
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs} train',
                    leave=False, dynamic_ncols=True)
        for b_idx, batch in enumerate(pbar):
            if args.max_batches and b_idx >= args.max_batches:
                break  # 冒烟测试：每个 epoch 只跑前几步
            batch = _to_device(batch, args.device)
            optimizer.zero_grad()
            parts = compute_losses(model, batch, buffers, args, ce)
            parts['loss'].backward()
            optimizer.step()

            batch_m = _batch_rank_metrics(parts, batch, 's')
            tr_parts['loss'].append(float(parts['loss'].detach().cpu()))
            tr_parts['poi'].append(float(parts['main'].detach().cpu()))
            tr_parts['time'].append(float(parts['time'].detach().cpu()))
            tr_parts['cat'].append(float(parts['cat'].detach().cpu()))
            tr_parts['pref'].append(float(parts['pref'].detach().cpu()))
            tr_parts['conf'].append(float(parts['conf'].detach().cpu()))
            tr_parts['adv'].append(float(parts['adv'].detach().cpu()))
            tr_parts['recon'].append(float(parts['recon'].detach().cpu()))
            for k in rank_keys:
                tr_parts[k].append(batch_m[k])
            pbar.set_postfix(loss=f'{tr_parts["loss"][-1]:.2f}',
                             avg=f'{float(np.mean(tr_parts["loss"])):.2f}',
                             top1=f'{tr_parts["top1"][-1]:.3f}', refresh=False)

        # 验证 + 测试：不算梯度；factual 用总分 s，deconf 用兴趣分 s_pref。
        # checkpoint 仍只看 val factual Acc@1/Acc@20，test 只用于快速看能力。
        model.eval()
        val_m, deconf = _eval_split(
            model, val_loader, buffers, args, ce, args.max_val_batches,
            desc=f'Epoch {epoch + 1}/{args.epochs} val  ')
        test_m, test_deconf = None, None
        if test_loader is not None:
            test_m, test_deconf = _eval_split(
                model, test_loader, buffers, args, ce, args.max_test_batches,
                desc=f'Epoch {epoch + 1}/{args.epochs} test ')

        train_m = {k: mean_or_nan(v) for k, v in tr_parts.items()}
        scheduler.step(val_m['loss'])
        # 和 GETNext 一样用 factual Acc@1/Acc@20 组合分挑最好的模型
        monitor_score = float(val_m['top1'] * 4 + val_m['top20']) if np.isfinite(val_m['top1']) else -np.inf
        saved = False
        if args.save_weights and np.isfinite(val_m['top1']) and monitor_score >= max_val_score:
            ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'user_id2idx_dict': user_id2idx,
                'poi_id2idx_dict': poi_id2idx,
                'cat_id2idx_dict': cat_id2idx,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx,
                'median_acc_bin': table.median_acc_bin,
                'median_pop_bin': table.median_pop_bin,
                'epoch_train_metrics': epoch_ckpt_metrics('train', train_m),
                'epoch_val_metrics': epoch_ckpt_metrics('val', val_m),
                'epoch_val_deconf': _deconf_ckpt_metrics('val', deconf),
            }
            if test_m is not None:
                state['epoch_test_metrics'] = epoch_ckpt_metrics('test', test_m)
                state['epoch_test_deconf'] = _deconf_ckpt_metrics('test', test_deconf)
            torch.save(state, os.path.join(ckpt_dir, 'best_epoch.state.pt'))
            with open(os.path.join(ckpt_dir, 'best_epoch.txt'), 'w') as f:
                dump = {**state['epoch_val_metrics'], **state['epoch_val_deconf']}
                if test_m is not None:
                    dump.update(state['epoch_test_metrics'])
                    dump.update(state['epoch_test_deconf'])
                print(dump, file=f)
            max_val_score = monitor_score
            saved = True

        extra_lines = format_ranking_lines(deconf, indent=' Deconf ')
        test_extra = None
        if test_deconf is not None:
            test_extra = format_ranking_lines(test_deconf, indent=' TDeconf ')
        logging.info(format_epoch_summary(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], train_m, val_m,
            saved_best=saved, best_score=max_val_score if saved else None,
            extra_lines=extra_lines, sep=SEP,
            test_m=test_m, test_extra_lines=test_extra))
        train_hist.append(train_m)
        val_row = dict(val_m)
        val_row.update({f'deconf_{k}': deconf[k] for k in rank_keys})
        val_hist.append(val_row)
        if test_m is not None:
            test_row = dict(test_m)
            test_row.update({f'deconf_{k}': test_deconf[k] for k in rank_keys})
            test_hist.append(test_row)
        _write_hist(args.save_dir, train_hist, val_hist, test_hist)

    logging.info(f'Training finished. Best val score={max_val_score:.4f}')
    logging.info(f'Checkpoints: {os.path.join(args.save_dir, "checkpoints")}')


def _to_device(batch, device):
    """把 batch 里的 Tensor 搬到 CPU 或 GPU；traj_id 这种字符串保持原样。

    输入:
        batch: dict，value 是 Tensor 或 list
        device: torch.device
    输出:
        新 dict，Tensor 形状不变，只是 device 变了
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _deconf_ckpt_metrics(split, m):
    """Checkpoint 里 deconf 字段，和 GETNext epoch_*_metrics 命名对齐。"""
    return {
        f'epoch_{split}_deconf_top1_acc': m['top1'],
        f'epoch_{split}_deconf_top5_acc': m['top5'],
        f'epoch_{split}_deconf_top10_acc': m['top10'],
        f'epoch_{split}_deconf_top20_acc': m['top20'],
        f'epoch_{split}_deconf_HR1': m['hr1'],
        f'epoch_{split}_deconf_H5': m['h5'],
        f'epoch_{split}_deconf_H10': m['h10'],
        f'epoch_{split}_deconf_NDCG5': m['ndcg5'],
        f'epoch_{split}_deconf_NDCG10': m['ndcg10'],
        f'epoch_{split}_deconf_mAP20': m['map20'],
        f'epoch_{split}_deconf_mrr': m['mrr'],
    }


def _deconf_hist_keys(split):
    """metrics-val.txt / metrics-test.txt 里 deconf 列表字段名。"""
    return [
        (f'{split}_epochs_deconf_top1_acc_list', 'deconf_top1'),
        (f'{split}_epochs_deconf_top5_acc_list', 'deconf_top5'),
        (f'{split}_epochs_deconf_top10_acc_list', 'deconf_top10'),
        (f'{split}_epochs_deconf_top20_acc_list', 'deconf_top20'),
        (f'{split}_epochs_deconf_hr1_list', 'deconf_hr1'),
        (f'{split}_epochs_deconf_h5_list', 'deconf_h5'),
        (f'{split}_epochs_deconf_h10_list', 'deconf_h10'),
        (f'{split}_epochs_deconf_ndcg5_list', 'deconf_ndcg5'),
        (f'{split}_epochs_deconf_ndcg10_list', 'deconf_ndcg10'),
        (f'{split}_epochs_deconf_mAP20_list', 'deconf_map20'),
        (f'{split}_epochs_deconf_mrr_list', 'deconf_mrr'),
    ]


def _write_hist(save_dir, train_hist, val_hist, test_hist=None):
    """每个 epoch 覆盖写入 metrics-train/val/test.txt，字段名和 GETNext 对齐。

    输入:
        save_dir: str
        train_hist / val_hist / test_hist: list[dict]
    输出:
        无返回。写文本文件。
    """
    write_epoch_metrics_txt(
        os.path.join(save_dir, 'metrics-train.txt'), 'train', train_hist,
        extra_keys=[
            ('train_epochs_pref_loss_list', 'pref'),
            ('train_epochs_conf_loss_list', 'conf'),
            ('train_epochs_adv_loss_list', 'adv'),
            ('train_epochs_recon_loss_list', 'recon'),
        ])
    write_epoch_metrics_txt(
        os.path.join(save_dir, 'metrics-val.txt'), 'val', val_hist,
        extra_keys=_deconf_hist_keys('val'))
    if test_hist:
        write_epoch_metrics_txt(
            os.path.join(save_dir, 'metrics-test.txt'), 'test', test_hist,
            extra_keys=_deconf_hist_keys('test'))


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
    warnings.filterwarnings('ignore', message='.*verbose parameter is deprecated.*')
    warnings.filterwarnings('ignore', message='.*mismatched src_key_padding_mask.*')
    args = parameter_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
