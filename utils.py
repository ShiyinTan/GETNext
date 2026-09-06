import glob
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


def calculate_laplacian_matrix(adj_mat, mat_type):
    """由邻接矩阵构造各类拉普拉斯 / 归一化矩阵。
    adj_mat: (N, N)
    返回同 shape (N, N)。训练使用 hat_rw_normd_lap_mat (GCN 常用形式)。
    """
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv: D̂^{-1} Â ，其中 Â=A+I, D̂=D+I
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    """带 mask 的 MSE：忽略 target==mask_value（padding）的位置。
    input/target: 任意同 shape，例如 (B, T_max)
    """
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def top_k_acc(y_true_seq, y_pred_seq, k):
    """整段序列逐步 Top-k 命中率（较少使用）。
    y_true_seq: (T,), y_pred_seq: (T, N_poi)
    """
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """next-POI 指标: 只看序列最后一步是否命中 Top-k。
    y_true_seq: (T,), y_pred_seq: (T, N_poi) → 取 [-1] 步，返回 0/1
    """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """next-POI 指标: 最后一步的 AP@k（正样本恒为 1 时退化为 1/rank）。"""
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """next-POI 指标: 最后一步真实 POI 的 Reciprocal Rank。"""
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


# Console / file / predict.json share these ranking keys.
RANKING_METRIC_KEYS = ('top1', 'top5', 'top10', 'top20', 'map20', 'mrr')
PREDICT_METRIC_KEY_MAP = {
    'top1': 'top1_acc',
    'top5': 'top5_acc',
    'top10': 'top10_acc',
    'top20': 'top20_acc',
    'map20': 'mAP20',
    'mrr': 'mrr',
}


def last_timestep_metric_dict(y_true_seq, y_pred_seq):
    """One trajectory, last-timestep ranking metrics (GETNext train/predict)."""
    return {
        'top1': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=1),
        'top5': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=5),
        'top10': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=10),
        'top20': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=20),
        'map20': mAP_metric_last_timestep(y_true_seq, y_pred_seq, k=20),
        'mrr': MRR_metric_last_timestep(y_true_seq, y_pred_seq),
    }


def batch_last_step_metrics(label_pois, pred_pois, seq_lens):
    """GETNext train/val batch eval: last-step metrics, then mean over the batch.

    label_pois: (B, T), pred_pois: (B, T, N), seq_lens: iterable of B lengths.
    """
    n = len(seq_lens)
    acc = {k: 0.0 for k in RANKING_METRIC_KEYS}
    if n == 0:
        return {k: float('nan') for k in RANKING_METRIC_KEYS}
    for y, s, L in zip(label_pois, pred_pois, seq_lens):
        m = last_timestep_metric_dict(y[:L], s[:L])
        for k in RANKING_METRIC_KEYS:
            acc[k] += m[k]
    return {k: acc[k] / n for k in RANKING_METRIC_KEYS}


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else float('nan')


def to_predict_metrics(m):
    """Map train-style keys (top1, map20) to predict.py metrics.json keys."""
    out = {}
    for src, dst in PREDICT_METRIC_KEY_MAP.items():
        v = None if m is None else m.get(src)
        if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
            out[dst] = None
        else:
            out[dst] = float(v)
    return out


def format_ranking_lines(m, indent='        '):
    """Two console lines: Acc@k then mAP@20 / MRR."""
    return [
        (f'{indent}Acc@1 {m["top1"]:.4f}  Acc@5 {m["top5"]:.4f}  '
         f'Acc@10 {m["top10"]:.4f}  Acc@20 {m["top20"]:.4f}'),
        f'{indent}mAP@20 {m["map20"]:.4f}  MRR {m["mrr"]:.4f}',
    ]


def format_epoch_summary(epoch, total_epochs, lr, train_m, val_m, saved_best=False,
                         best_score=None, extra_lines=None, sep='-' * 72):
    """Compact, aligned epoch metrics block for the console (GETNext layout)."""
    lines = [
        sep,
        f' Epoch {epoch + 1:>4d}/{total_epochs}  |  lr={lr:.2e}',
        sep,
        (f' Train  loss {train_m["loss"]:>8.4f}  '
         f'poi {train_m["poi"]:>7.4f}  time {train_m["time"]:>7.4f}  cat {train_m["cat"]:>7.4f}'),
        *format_ranking_lines(train_m),
        (f' Val    loss {val_m["loss"]:>8.4f}  '
         f'poi {val_m["poi"]:>7.4f}  time {val_m["time"]:>7.4f}  cat {val_m["cat"]:>7.4f}'),
        *format_ranking_lines(val_m),
    ]
    if extra_lines:
        lines.extend(extra_lines)
    if saved_best:
        lines.append(f' * Saved best checkpoint  (score={best_score:.4f})')
    lines.append(sep)
    return '\n'.join(lines)


def epoch_ckpt_metrics(split, m):
    """Checkpoint `epoch_{train,val}_metrics` dict used by GETNext train.py."""
    return {
        f'epoch_{split}_loss': m['loss'],
        f'epoch_{split}_poi_loss': m['poi'],
        f'epoch_{split}_time_loss': m['time'],
        f'epoch_{split}_cat_loss': m['cat'],
        f'epoch_{split}_top1_acc': m['top1'],
        f'epoch_{split}_top5_acc': m['top5'],
        f'epoch_{split}_top10_acc': m['top10'],
        f'epoch_{split}_top20_acc': m['top20'],
        f'epoch_{split}_mAP20': m['map20'],
        f'epoch_{split}_mrr': m['mrr'],
    }


def write_epoch_metrics_txt(path, split, rows, extra_keys=None):
    """Write GETNext `metrics-train.txt` / `metrics-val.txt` lists.

    rows: list of dicts with keys loss, poi, time, cat, top1..top20, map20, mrr.
    extra_keys: optional list of (file_name, dict_key), e.g. causal aux losses.
    """
    file_keys = [
        (f'{split}_epochs_loss_list', 'loss'),
        (f'{split}_epochs_poi_loss_list', 'poi'),
        (f'{split}_epochs_time_loss_list', 'time'),
        (f'{split}_epochs_cat_loss_list', 'cat'),
        (f'{split}_epochs_top1_acc_list', 'top1'),
        (f'{split}_epochs_top5_acc_list', 'top5'),
        (f'{split}_epochs_top10_acc_list', 'top10'),
        (f'{split}_epochs_top20_acc_list', 'top20'),
        (f'{split}_epochs_mAP20_list', 'map20'),
        (f'{split}_epochs_mrr_list', 'mrr'),
    ]
    if extra_keys:
        file_keys.extend(extra_keys)
    with open(path, 'w') as f:
        for file_key, dict_key in file_keys:
            vals = [float(f'{r[dict_key]:.4f}') for r in rows]
            print(f'{file_key}={vals}', file=f)


def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))
