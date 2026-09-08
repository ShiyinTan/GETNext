"""诊断 causal 比 GETNext 差在哪：启发式天花板 + checkpoint 各通道 Acc。

不训练。两类用法：

  # 只看「近 / 热 / 原始 A / 残差 A_rel / g̃」各自能刷多高（约 15s CPU）
  python causal/diagnose.py --heuristics-only --no-cuda

  # 已有 causal checkpoint：再拆 s_pref / s_conf 各项 / s_rel / g̃ 的 Acc 和分数尺度
  python causal/diagnose.py --checkpoint runs/causal/<name>/checkpoints/best_epoch.state.pt

输出 JSON（默认 stdout；可 --output 存盘）。把文件贴回来即可继续分析。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal.features import (
    build_poi_table,
    fill_graph_transition_tables,
    fill_transition_priors,
    load_nodes_df,
)
from causal.model import CausalNextPOI
from causal.predict import rebuild_table
from causal.train import TrajectoryDataset, collate_pad, _to_device
from dataloader import load_graph_adj_mtx
from utils import last_timestep_metric_dict, mean_or_nan


RANK_KEYS = ('top1', 'top5', 'top10', 'top20', 'map20', 'mrr')


def parse_args():
    p = argparse.ArgumentParser(description='Causal vs GETNext 差距诊断')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--heuristics-only', action='store_true')
    p.add_argument('--data-test', type=str, default='dataset/NYC/NYC_test.csv')
    p.add_argument('--data-train', type=str, default='dataset/NYC/NYC_train.csv')
    p.add_argument('--data-node-feats', type=str, default='dataset/NYC/graph_X.csv')
    p.add_argument('--data-adj-mtx', type=str, default='dataset/NYC/graph_A.csv')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--short-traj-thres', type=int, default=2)
    p.add_argument('--time-feature', type=str, default='norm_in_day_time')
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--output', type=str, default=None, help='写入这份 JSON；默认打印')
    p.add_argument('--w-conf', type=float, default=None)
    p.add_argument('--w-rel', type=float, default=None)
    p.add_argument('--rel-source', type=str, default=None, choices=['residual', 'raw'])
    return p.parse_args()


class _FeatArgs:
    dist_bins = '0.5,1,2,5,10'
    pop_bins = 4
    area_grid_deg = 0.02
    time_units = 48
    feature1 = 'checkin_cnt'
    feature2 = 'poi_catid'
    feature3 = 'latitude'
    feature4 = 'longitude'
    time_feature = 'norm_in_day_time'
    align_alpha = 0.2
    align_beta = 0.3
    align_gamma = 0.3


def _load_maps(train_df, nodes_df, feat_args):
    poi_ids = list(nodes_df['node_name/poi_id'].tolist())
    poi_id2idx = dict(zip(poi_ids, range(len(poi_ids))))
    cat_ids = list(dict.fromkeys(nodes_df[feat_args.feature2].tolist()))
    cat_id2idx = dict(zip(cat_ids, range(len(cat_ids))))
    poi_idx2cat_idx = {}
    for _, row in nodes_df.iterrows():
        poi_idx2cat_idx[poi_id2idx[row['node_name/poi_id']]] = cat_id2idx[row[feat_args.feature2]]
    user_ids = [str(u) for u in sorted(set(train_df['user_id'].astype(str).tolist()))]
    user_id2idx = dict(zip(user_ids, range(len(user_ids))))
    return poi_id2idx, cat_id2idx, poi_idx2cat_idx, user_id2idx


def _metrics_from_last_scores(y_idx, scores):
    """y_idx: (n,) gold；scores: (n, N). 返回 GETNext 同款 last-step 均值 + rank 统计。"""
    rows, ranks = [], []
    for i in range(len(y_idx)):
        y = int(y_idx[i])
        s = np.asarray(scores[i], dtype=np.float64)
        rows.append(last_timestep_metric_dict(np.array([y]), s.reshape(1, -1)))
        order = np.argsort(-s)
        ranks.append(int(np.where(order == y)[0][0]) + 1)
    out = {k: float(np.mean([r[k] for r in rows])) for k in RANK_KEYS}
    out['mean_rank'] = float(np.mean(ranks))
    out['median_rank'] = float(np.median(ranks))
    return out


def _score_stats(mat):
    """mat: (n, N) last-step scores."""
    flat = np.asarray(mat, dtype=np.float64)
    return {
        'mean': float(flat.mean()),
        'std': float(flat.std()),
        'min': float(flat.min()),
        'max': float(flat.max()),
        'p05': float(np.percentile(flat, 5)),
        'p95': float(np.percentile(flat, 95)),
    }


def run_heuristics(cli):
    feat_args = _FeatArgs()
    train_df = pd.read_csv(cli.data_train)
    test_df = pd.read_csv(cli.data_test)
    nodes_df = load_nodes_df(cli.data_node_feats)
    poi_id2idx, _, poi_idx2cat_idx, user_id2idx = _load_maps(train_df, nodes_df, feat_args)
    table = build_poi_table(nodes_df, train_df, feat_args, poi_id2idx)
    gstats = fill_graph_transition_tables(
        table, cli.data_adj_mtx, cli.data_node_feats, poi_id2idx)
    test_ds = TrajectoryDataset(
        test_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        feat_args.time_feature, cli.short_traj_thres, skip_unknown_user=True)
    train_ds = TrajectoryDataset(
        train_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        feat_args.time_feature, cli.short_traj_thres)
    pairs = [(o, d) for s in train_ds.samples for o, d in zip(s['poi'], s['label_poi'])]
    fill_transition_priors(table, pairs)

    n = table.num_pois
    dist, log_pop, log_tpop = table.dist_km, table.log_pop, table.log_tpop
    a_rel, a_raw, area = table.a_rel, table.a_raw, table.area_id
    A_raw = np.asarray(load_graph_adj_mtx(cli.data_adj_mtx), dtype=np.float64)
    A_full = np.zeros((n, n), dtype=np.float64)
    gx = pd.read_csv(cli.data_node_feats)
    mapped_g, mapped_c = [], []
    for gi, poi in enumerate(gx.iloc[:, 0].tolist()):
        if gi >= A_raw.shape[0]:
            break
        idx = poi_id2idx.get(poi, poi_id2idx.get(str(poi)))
        if idx is not None:
            mapped_g.append(gi)
            mapped_c.append(int(idx))
    if mapped_g:
        g_idx = np.asarray(mapped_g)
        c_idx = np.asarray(mapped_c)
        A_full[np.ix_(c_idx, c_idx)] = A_raw[np.ix_(g_idx, g_idx)]

    origins, gold = [], []
    for s in test_ds.samples:
        origins.append(int(s['poi'][-1]))
        gold.append(int(s['label_poi'][-1]))
    origins = np.asarray(origins)
    gold = np.asarray(gold)

    def g_tilde_row(o):
        same = (area[o] == area).astype(np.float64)
        return (-feat_args.align_alpha * dist[o]
                + feat_args.align_beta * log_pop
                + feat_args.align_gamma * log_tpop
                + 0.15 * same)

    heuristics = {
        'nearest': lambda o: -dist[o],
        'log_pop': lambda o: np.broadcast_to(log_pop, (n,)).copy(),
        'log_tpop': lambda o: np.broadcast_to(log_tpop, (n,)).copy(),
        'graph_A_row': lambda o: A_full[o],
        'a_raw_log1p': lambda o: a_raw[o],
        'a_rel': lambda o: a_rel[o],
        'g_tilde': g_tilde_row,
        'g_tilde_plus_A': lambda o: g_tilde_row(o) + np.log1p(A_full[o]),
        'g_tilde_plus_rel': lambda o: g_tilde_row(o) + a_rel[o],
    }
    stacked = {name: np.stack([fn(int(o)) for o in origins]) for name, fn in heuristics.items()}
    metrics = {name: _metrics_from_last_scores(gold, mat) for name, mat in stacked.items()}

    gold_km = dist[origins, gold]
    edges = table.dist_edges
    bins = np.digitize(gold_km, edges, right=True)
    from collections import Counter
    bin_frac = {str(k): v / len(gold) for k, v in sorted(Counter(bins.tolist()).items())}
    return {
        'n_test_trajs': int(len(gold)),
        'n_pois': int(n),
        'graph_stats': gstats,
        'gold_next_hop': {
            'mean_km': float(gold_km.mean()),
            'median_km': float(np.median(gold_km)),
            'p90_km': float(np.percentile(gold_km, 90)),
            'same_area_frac': float((area[origins] == area[gold]).mean()),
            'graph_A_edge_frac': float((A_full[origins, gold] > 0).mean()),
            'dist_bin_frac': bin_frac,
            'dist_edges_km': edges.tolist(),
        },
        'heuristics': metrics,
        'note': (
            'graph_A_row ≈ GETNext NodeAttnMap 能直接用的 Markov 转移；'
            'a_rel 是拆掉热度+距离后的残差，单独几乎不能排序；'
            'g_tilde 是 L_conf 对齐目标，单独 Acc 很低。'
        ),
    }


def run_checkpoint(cli):
    device = torch.device('cpu' if cli.no_cuda or not torch.cuda.is_available() else 'cuda')
    try:
        ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(cli.checkpoint, map_location=device)
    args = ckpt['args']
    args.device = device
    args.no_cuda = cli.no_cuda
    if cli.w_conf is not None:
        args.w_conf = cli.w_conf
    if cli.w_rel is not None:
        args.w_rel = cli.w_rel
    if cli.rel_source is not None:
        args.rel_source = cli.rel_source
    for key in ('w_pref', 'w_conf', 'w_acc', 'w_pop', 'w_tpop', 'w_area', 'w_ctx', 'w_rel',
                'pref_decoder', 'rel_source'):
        if not hasattr(args, key):
            setattr(args, key, 'tied' if key == 'pref_decoder' else (
                'residual' if key == 'rel_source' else 1.0))

    user_id2idx = ckpt['user_id2idx_dict']
    poi_id2idx = ckpt['poi_id2idx_dict']
    cat_id2idx = ckpt['cat_id2idx_dict']
    poi_idx2cat_idx = ckpt['poi_idx2cat_idx_dict']
    meta_path = Path(cli.checkpoint).resolve().parents[1] / 'poi_table_meta.pkl'
    meta = None
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    table = rebuild_table(cli, args, poi_id2idx, meta)
    test_df = pd.read_csv(cli.data_test)
    dataset = TrajectoryDataset(
        test_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        cli.time_feature, cli.short_traj_thres, skip_unknown_user=True)
    loader = DataLoader(
        dataset, batch_size=cli.batch, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=collate_pad)

    model = CausalNextPOI(args, table.num_pois, len(user_id2idx), len(cat_id2idx), table)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    buffers = table.to_torch(device)

    channel_names = (
        's_pref', 's_conf', 's_rel', 's_acc', 's_pop', 's_tpop', 's_area', 's_ctx',
        'g_tilde', 'factual', 'pref_plus_rawA',
    )
    bags = {k: [] for k in channel_names}
    gold = []

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(loader, desc='Diagnose ckpt')):
            if cli.max_batches and b_idx >= cli.max_batches:
                break
            batch = _to_device(batch, device)
            poi = batch['poi']
            h, h_z, h_c = model.encode(poi, batch['time'], batch['cat'], batch['user'], batch['pad'])
            parts = model.score_parts(h_z, h_c, poi, buffers, h=h)
            g_tilde = model.g_tilde(
                poi, buffers,
                getattr(args, 'align_alpha', 0.2),
                getattr(args, 'align_beta', 0.3),
                getattr(args, 'align_gamma', 0.3))
            s_fact = model.w_pref * parts['s_pref'] + model.w_conf * parts['s_conf'] + model.w_rel * parts['s_rel']
            a_raw = buffers.get('a_raw')
            y_np = batch['y_poi'].detach().cpu().numpy()
            for i, L in enumerate(batch['lengths']):
                gold.append(int(y_np[i, L - 1]))
                for name in ('s_pref', 's_conf', 's_rel', 's_acc', 's_pop', 's_tpop', 's_area', 's_ctx'):
                    tensor = parts[name]
                    if tensor.dim() == 3 and tensor.size(0) == 1 and tensor.size(1) == 1:
                        vec = tensor[0, 0]
                    else:
                        vec = tensor[i, L - 1]
                    bags[name].append(vec.detach().cpu().numpy())
                bags['g_tilde'].append(g_tilde[i, L - 1].detach().cpu().numpy())
                bags['factual'].append(s_fact[i, L - 1].detach().cpu().numpy())
                pref = parts['s_pref'][i, L - 1]
                if a_raw is not None:
                    origin = int(poi[i, L - 1].item())
                    bags['pref_plus_rawA'].append(
                        (pref + a_raw[origin]).detach().cpu().numpy())
                else:
                    bags['pref_plus_rawA'].append(pref.detach().cpu().numpy())

    gold = np.asarray(gold)
    stacked = {k: np.stack(v) for k, v in bags.items() if v}
    out = {
        'checkpoint': cli.checkpoint,
        'n_eval': int(len(gold)),
        'pref_decoder': getattr(args, 'pref_decoder', 'tied'),
        'rel_source': getattr(model, 'rel_source', 'residual'),
        'score_weights': {
            'w_pref': float(model.w_pref), 'w_conf': float(model.w_conf),
            'w_acc': float(model.w_acc), 'w_pop': float(model.w_pop),
            'w_tpop': float(model.w_tpop), 'w_area': float(model.w_area),
            'w_ctx': float(model.w_ctx), 'w_rel': float(model.w_rel),
        },
        'load_state_dict': {
            'missing': list(missing),
            'unexpected': list(unexpected),
        },
        'channel_metrics': {k: _metrics_from_last_scores(gold, stacked[k]) for k in stacked},
        'channel_score_scale': {k: _score_stats(stacked[k]) for k in stacked},
    }
    # s_conf 是否已经被 L_conf 拉去贴 g̃
    sc = stacked['s_conf'].reshape(-1)
    gt = stacked['g_tilde'].reshape(-1)
    if sc.std() > 1e-8 and gt.std() > 1e-8:
        out['s_conf_vs_g_tilde_corr'] = float(np.corrcoef(sc, gt)[0, 1])
    else:
        out['s_conf_vs_g_tilde_corr'] = None
    sp = stacked['s_pref'].reshape(-1)
    out['std_ratio_conf_over_pref'] = float(sc.std() / (sp.std() + 1e-8))
    out['std_ratio_rel_over_pref'] = float(stacked['s_rel'].reshape(-1).std() / (sp.std() + 1e-8))
    return out


def main():
    cli = parse_args()
    if not cli.heuristics_only and not cli.checkpoint:
        raise SystemExit('pass --heuristics-only and/or --checkpoint')
    report = {'data_test': cli.data_test}
    if cli.heuristics_only or cli.checkpoint:
        report['heuristics'] = run_heuristics(cli)
    if cli.checkpoint:
        report['checkpoint'] = run_checkpoint(cli)
    text = json.dumps(report, indent=2, default=str)
    if cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(cli.output)), exist_ok=True)
        with open(cli.output, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Wrote {cli.output}')
    print(text)


if __name__ == '__main__':
    main()
