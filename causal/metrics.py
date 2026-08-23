"""评价指标：沿用 GETNext 的「只看轨迹最后一步」，再按混杂 C 切片（附录 D / 正文 §7）。

为什么要切片？
  只报整体 Acc@k 时，模型就算只会推「附近的热门店」分数也会很高。
  所以还要分开看：远距离、冷门地点、跨区域跳转时，还能不能排对。

形状记号：L=去掉 padding 后的真实轨迹长度；N=POI 数。
GETNext 的 Acc@k / MRR 函数吃的是「一条轨迹」：
  y_true_seq: (L,)  每步的真值 POI
  y_pred_seq: (L, N) 每步对全部候选的分数
只评最后一步 L-1。
"""
from collections import defaultdict

import numpy as np

from utils import (
    MRR_metric_last_timestep,
    mAP_metric_last_timestep,
    top_k_acc_last_timestep,
)


def last_step_scores(label_row, pred_row, seq_len):
    """去掉 padding 后，只保留真实长度。GETNext 同样只评序列最后一步。

    输入:
        label_row: (T,)  pad 后的真值，有效部分在前 seq_len
        pred_row:  (T, N) pad 后的分数
        seq_len:   int = L，真实长度
    输出:
        y: (L,)
        s: (L, N)
    """
    y = label_row[:seq_len]
    s = pred_row[:seq_len]
    return y, s


def basic_metrics(y_true_seq, y_pred_seq):
    """一条轨迹的 last-step 指标。返回的是 0/1 命中或 0~1 的排名分数。

    输入:
        y_true_seq: (L,)  每步真值 POI
        y_pred_seq: (L, N) 每步对 N 个候选的分数
    输出:
        dict，六个标量：top1/top5/top10/top20/map20/mrr
        （内部只取最后一步 L-1）
    """
    return {
        'top1': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=1),
        'top5': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=5),
        'top10': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=10),
        'top20': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=20),
        'map20': mAP_metric_last_timestep(y_true_seq, y_pred_seq, k=20),
        'mrr': MRR_metric_last_timestep(y_true_seq, y_pred_seq),
    }


def mean_metric_dict(rows):
    """把多条轨迹的指标做平均。没有样本时填 None，避免除零。

    输入:
        rows: list[dict]，每个 dict 是 basic_metrics 的输出（六个标量）
    输出:
        dict，同样六个 key，值为 float 或 None
    """
    if not rows:
        return {k: None for k in ('top1', 'top5', 'top10', 'top20', 'map20', 'mrr')}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


class SliceMeter:
    """边推理边收集：整体 + 按距离桶 / 热度档 / 是否跨区。

    调用方每来一条轨迹就 add() 一次；最后 summary() 出一份可写入 metrics.json 的字典。
    """

    def __init__(self):
        """
        输入: 无
        输出: 无返回。内部列表稍后装 basic_metrics 的 dict。
        """
        self.overall = []
        self.by_acc = defaultdict(list)   # 距离桶 -> 指标列表
        self.by_pop = defaultdict(list)   # 热度四分位 -> 指标列表
        self.cross_area = []              # 起点区域 ≠ 终点区域
        self.same_area = []               # 同区域内转移

    def add(self, y_true_seq, y_pred_seq, acc_bin, pop_bin, same_area):
        """
        输入:
            y_true_seq: (L,)
            y_pred_seq: (L, N)
            acc_bin: 标量 int，真值下一站相对起点的距离桶
            pop_bin: 标量 int，真值下一站的热度档
            same_area: bool，起点区域 == 终点区域
        输出:
            无返回。把这条轨迹的指标追加进 overall / by_acc / by_pop / same|cross。
        """
        m = basic_metrics(y_true_seq, y_pred_seq)
        self.overall.append(m)
        self.by_acc[int(acc_bin)].append(m)
        self.by_pop[int(pop_bin)].append(m)
        if same_area:
            self.same_area.append(m)
        else:
            self.cross_area.append(m)

    def summary(self):
        """
        输入: 无（读 self 里已收集的指标）
        输出: dict
            overall: 六个标量的平均
            n: int
            by_distance_bucket / by_pop_quartile: {桶号: 平均指标}
            same_area / cross_area: 平均指标
            n_same_area / n_cross_area: int
        """
        out = {
            'overall': mean_metric_dict(self.overall),
            'n': len(self.overall),
            'by_distance_bucket': {str(k): mean_metric_dict(v) for k, v in sorted(self.by_acc.items())},
            'by_pop_quartile': {str(k): mean_metric_dict(v) for k, v in sorted(self.by_pop.items())},
            'same_area': mean_metric_dict(self.same_area),
            'cross_area': mean_metric_dict(self.cross_area),
            'n_same_area': len(self.same_area),
            'n_cross_area': len(self.cross_area),
        }
        return out
