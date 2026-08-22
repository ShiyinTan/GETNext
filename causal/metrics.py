"""Ranking metrics: GETNext last-timestep Acc@k plus Appendix D §7 slices."""
from collections import defaultdict

import numpy as np

from utils import (
    MRR_metric_last_timestep,
    mAP_metric_last_timestep,
    top_k_acc_last_timestep,
)


def last_step_scores(label_row, pred_row, seq_len):
    """Trim padding then keep the GETNext convention: evaluate the last step only."""
    y = label_row[:seq_len]
    s = pred_row[:seq_len]
    return y, s


def basic_metrics(y_true_seq, y_pred_seq):
    return {
        'top1': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=1),
        'top5': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=5),
        'top10': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=10),
        'top20': top_k_acc_last_timestep(y_true_seq, y_pred_seq, k=20),
        'map20': mAP_metric_last_timestep(y_true_seq, y_pred_seq, k=20),
        'mrr': MRR_metric_last_timestep(y_true_seq, y_pred_seq),
    }


def mean_metric_dict(rows):
    if not rows:
        return {k: None for k in ('top1', 'top5', 'top10', 'top20', 'map20', 'mrr')}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


class SliceMeter:
    """Collect last-step metrics overall and by C slices (distance / pop / area)."""

    def __init__(self):
        self.overall = []
        self.by_acc = defaultdict(list)
        self.by_pop = defaultdict(list)
        self.cross_area = []
        self.same_area = []

    def add(self, y_true_seq, y_pred_seq, acc_bin, pop_bin, same_area):
        m = basic_metrics(y_true_seq, y_pred_seq)
        self.overall.append(m)
        self.by_acc[int(acc_bin)].append(m)
        self.by_pop[int(pop_bin)].append(m)
        if same_area:
            self.same_area.append(m)
        else:
            self.cross_area.append(m)

    def summary(self):
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
