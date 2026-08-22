"""Load a causal checkpoint and rank next POI under factual / deconfounded modes.

Reports overall Acc@k / MRR plus Appendix D §7 slices (distance bucket, pop
quartile, same-area vs cross-area). Deconf scores are never mixed into the
factual summary.
"""
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
    fill_transition_priors,
    load_nodes_df,
    pairwise_haversine_km,
    bucketize,
)
from causal.metrics import SliceMeter
from causal.model import CausalNextPOI
from causal.train import TrajectoryDataset, collate_pad, _to_device


def parse_args():
    p = argparse.ArgumentParser(description='Causal next-POI prediction')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data-test', type=str, default='dataset/NYC/NYC_test.csv')
    p.add_argument('--data-train', type=str, default='dataset/NYC/NYC_train.csv',
                   help='Used to rebuild train-based pop / area tables')
    p.add_argument('--data-node-feats', type=str, default='dataset/NYC/graph_X.csv')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--short-traj-thres', type=int, default=2)
    p.add_argument('--time-feature', type=str, default='norm_in_day_time')
    p.add_argument('--feature1', type=str, default='checkin_cnt')
    p.add_argument('--feature2', type=str, default='poi_catid')
    p.add_argument('--feature3', type=str, default='latitude')
    p.add_argument('--feature4', type=str, default='longitude')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--no-cuda', action='store_true', default=False)
    p.add_argument('--modes', type=str, default='factual,deconf_pref,deconf_do,deconf_sum',
                   help='Comma-separated scoring modes (Appendix D.5)')
    p.add_argument('--max-batches', type=int, default=0)
    return p.parse_args()


def rebuild_table(cli, args, poi_id2idx, meta):
    train_df = pd.read_csv(cli.data_train)
    nodes_df = load_nodes_df(cli.data_node_feats)
    table = build_poi_table(nodes_df, train_df, args, poi_id2idx)
    if meta is not None:
        # Keep the training-time pop / area / priors so predict matches the ckpt.
        if 'pop' in meta:
            table.pop = meta['pop']
            table.log_pop = np.log1p(table.pop)
        if 'area_id' in meta:
            table.area_id = meta['area_id']
            table.num_areas = int(meta.get('num_areas', table.area_id.max() + 1))
        if 'pop_bin' in meta:
            table.pop_bin = meta['pop_bin']
        if 'acc_prior' in meta:
            table.acc_prior = meta['acc_prior']
            table.pop_prior = meta['pop_prior']
            table.median_acc_bin = int(meta.get('median_acc_bin', 0))
            table.median_pop_bin = int(meta.get('median_pop_bin', 0))
        else:
            fill_transition_priors(table, [(0, 0)])
        # Recompute distance tables (deterministic from lat/lon).
        table.lat = meta.get('lat', table.lat)
        table.lon = meta.get('lon', table.lon)
        table.dist_km = pairwise_haversine_km(table.lat, table.lon)
        edges = meta.get('dist_edges', table.dist_edges)
        table.dist_edges = edges
        table.dist_bin = bucketize(table.dist_km, edges)
        table.num_acc_bins = int(meta.get('num_acc_bins', table.dist_bin.max() + 1))
        table.num_pop_bins = int(meta.get('num_pop_bins', table.pop_bin.max() + 1))
    return table


def main():
    cli = parse_args()
    device = torch.device('cpu' if cli.no_cuda or not torch.cuda.is_available() else 'cuda')
    try:
        ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(cli.checkpoint, map_location=device)
    args = ckpt['args']
    args.device = device
    args.no_cuda = cli.no_cuda
    args.feature1 = getattr(args, 'feature1', cli.feature1)
    args.feature2 = getattr(args, 'feature2', cli.feature2)
    args.feature3 = getattr(args, 'feature3', cli.feature3)
    args.feature4 = getattr(args, 'feature4', cli.feature4)

    user_id2idx = ckpt['user_id2idx_dict']
    poi_id2idx = ckpt['poi_id2idx_dict']
    cat_id2idx = ckpt['cat_id2idx_dict']
    poi_idx2cat_idx = ckpt['poi_idx2cat_idx_dict']
    idx2poi = {v: k for k, v in poi_id2idx.items()}

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
        num_workers=cli.workers, collate_fn=collate_pad)

    model = CausalNextPOI(args, table.num_pois, len(user_id2idx), len(cat_id2idx), table)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    buffers = table.to_torch(device)
    bar_acc = int(ckpt.get('median_acc_bin', table.median_acc_bin))
    if getattr(args, 'bar_acc_bin', -1) >= 0:
        bar_acc = int(args.bar_acc_bin)

    modes = [m.strip() for m in cli.modes.split(',') if m.strip()]
    meters = {m: SliceMeter() for m in modes}
    predictions = []
    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(loader, desc='Predict')):
            if cli.max_batches and b_idx >= cli.max_batches:
                break
            batch = _to_device(batch, device)
            poi = batch['poi']
            h, h_z, h_c = model.encode(poi, batch['time'], batch['cat'], batch['user'], batch['pad'])
            y_np = batch['y_poi'].cpu().numpy()
            poi_np = poi.cpu().numpy()
            area = buffers['area_id'].cpu().numpy()
            scores_by_mode = {}
            for mode in modes:
                s, s_pref, s_conf, dist_km = model.score(
                    h_z, h_c, poi, buffers, mode=mode, bar_acc_bin=bar_acc)
                scores_by_mode[mode] = s.cpu().numpy()
                c_acc = torch.gather(
                    buffers['dist_bin'][poi.clamp(min=0)], 2,
                    batch['y_poi'].clamp(min=0).unsqueeze(-1)).squeeze(-1).cpu().numpy()
                c_pop = buffers['pop_bin'][batch['y_poi'].clamp(min=0)].cpu().numpy()
                for i, L in enumerate(batch['lengths']):
                    dest = int(y_np[i, L - 1])
                    origin = int(poi_np[i, L - 1])
                    meters[mode].add(
                        y_np[i, :L], scores_by_mode[mode][i, :L],
                        c_acc[i, L - 1], c_pop[i, L - 1],
                        bool(area[origin] == area[dest]))

            fact = scores_by_mode.get('factual', scores_by_mode[modes[0]])
            pref = scores_by_mode.get('deconf_pref', None)
            for i, L in enumerate(batch['lengths']):
                last_label = int(y_np[i, L - 1])
                last_s = fact[i, L - 1]
                topk = np.argsort(-last_s)[:cli.top_k]
                row = {
                    'traj_id': batch['traj_ids'][i],
                    'label_poi_idx': last_label,
                    'label_poi_id': idx2poi.get(last_label),
                    'factual_topk_poi_idx': topk.tolist(),
                    'factual_topk_poi_id': [idx2poi.get(int(j)) for j in topk],
                    'factual_topk_score': last_s[topk].tolist(),
                }
                if pref is not None:
                    topk_d = np.argsort(-pref[i, L - 1])[:cli.top_k]
                    row['deconf_pref_topk_poi_idx'] = topk_d.tolist()
                    row['deconf_pref_topk_poi_id'] = [idx2poi.get(int(j)) for j in topk_d]
                predictions.append(row)

    summary = {
        'num_trajectories': len(dataset),
        'checkpoint': cli.checkpoint,
        'data_test': cli.data_test,
        'device': str(device),
        'epoch': ckpt.get('epoch'),
        'modes': {m: meters[m].summary() for m in modes},
    }
    out_dir = cli.output_dir or str(Path(cli.checkpoint).resolve().parents[1] / 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'predictions.jsonl'), 'w', encoding='utf-8') as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({
        'num_trajectories': summary['num_trajectories'],
        'device': summary['device'],
        'overall': {m: summary['modes'][m]['overall'] for m in modes},
    }, indent=2))
    print(f'Wrote metrics to {out_dir}/metrics.json')
    print(f'Wrote predictions to {out_dir}/predictions.jsonl')


if __name__ == '__main__':
    main()
