"""Train the Appendix D causal next-POI model.

Running logic follows GETNext (traj CSV → padded batch → Transformer → last-step
Acc@k / MRR, checkpoint on val score) with causal replacements where the two
designs conflict: no GCN / NodeAttnMap, C stays out of tokens, dual scores.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal.features import build_poi_table, fill_transition_priors, load_nodes_df
from causal.metrics import SliceMeter, basic_metrics
from causal.model import CausalNextPOI
from causal.param_parser import parameter_parser
from utils import increment_path, zipdir

SEP = '-' * 72


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(save_dir, verbose=False):
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
    if args.no_cuda or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(args.device)


class TrajectoryDataset(Dataset):
    """GETNext-style next-step pairs, plus category ids needed by the causal aux head."""

    def __init__(self, df, poi_id2idx, user_id2idx, poi_idx2cat_idx, time_col,
                 min_len, skip_unknown_user=False):
        self.samples = []
        grouped = df.groupby('trajectory_id', sort=False)
        n_traj = df['trajectory_id'].nunique()
        for traj_id, traj_df in tqdm(grouped, total=n_traj,
                                      desc='Build trajectories', leave=False, dynamic_ncols=True):
            user_id = str(str(traj_id).split('_')[0])
            if skip_unknown_user and user_id not in user_id2idx:
                continue
            if user_id not in user_id2idx:
                continue
            poi_idxs, times = [], []
            for poi_id, t in zip(traj_df['POI_id'].tolist(), traj_df[time_col].tolist()):
                if poi_id not in poi_id2idx:
                    continue
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
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_pad(batch):
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
    mask = target != ignore
    if mask.sum() == 0:
        return pred.new_zeros(())
    return ((pred[mask] - target[mask]) ** 2).mean()


def gather_c_of_y(origin, y_poi, time_feat, buffers, num_hour_bins):
    """Discrete C(Y) used by L_adv / L_recon (D.4.4). Pads stay -1."""
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
    poi = batch['poi']
    h, h_z, h_c = model.encode(poi, batch['time'], batch['cat'], batch['user'], batch['pad'])
    s, s_pref, s_conf, _ = model.score(h_z, h_c, poi, buffers, mode='factual')
    y = batch['y_poi']

    loss_main = ce(s.transpose(1, 2), y)

    # D.4.2 same-distance-band CE on s_pref only.
    c_acc, c_pop, c_area, c_hour = gather_c_of_y(
        poi, y, batch['time'], buffers, args.time_units)
    dist_full = buffers['dist_bin'][poi.clamp(min=0)]
    ring = dist_full == c_acc.unsqueeze(-1)
    s_ring = s_pref.masked_fill(~ring, torch.tensor(-1e9, device=s_pref.device, dtype=s_pref.dtype))
    loss_pref = ce(s_ring.transpose(1, 2), y)

    g_tilde = model.g_tilde(poi, buffers, args.align_alpha, args.align_beta).detach()
    valid = (y >= 0).unsqueeze(-1).float()
    denom = valid.sum() * s_conf.size(-1)
    loss_conf = (((s_conf - g_tilde) ** 2) * valid).sum() / denom.clamp(min=1.0)
    if args.conf_aux_ce:
        loss_conf = loss_conf + 0.1 * ce(s_conf.transpose(1, 2), y)

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


@torch.no_grad()
def eval_batch_metrics(parts, batch, buffers, meter_fact, meter_deconf):
    y_np = batch['y_poi'].detach().cpu().numpy()
    s_np = parts['s'].detach().cpu().numpy()
    pref_np = parts['s_pref'].detach().cpu().numpy()
    area = buffers['area_id'].detach().cpu().numpy()
    c_acc = parts['c_acc'].detach().cpu().numpy()
    c_pop = parts['c_pop'].detach().cpu().numpy()
    poi = batch['poi'].detach().cpu().numpy()
    for i, L in enumerate(batch['lengths']):
        y_i, s_i = y_np[i, :L], s_np[i, :L]
        p_i = pref_np[i, :L]
        origin = int(poi[i, L - 1])
        dest = int(y_i[-1])
        same = bool(area[origin] == area[dest])
        meter_fact.add(y_i, s_i, c_acc[i, L - 1], c_pop[i, L - 1], same)
        meter_deconf.add(y_i, p_i, c_acc[i, L - 1], c_pop[i, L - 1], same)


def format_epoch_summary(epoch, total, lr, train_loss, fact, deconf, saved=False, score=None):
    lines = [
        SEP,
        f' Epoch {epoch + 1:>4d}/{total}  |  lr={lr:.2e}',
        SEP,
        (f' Train  loss {train_loss["loss"]:>8.4f}  main {train_loss["main"]:>7.4f}  '
         f'pref {train_loss["pref"]:>7.4f}  conf {train_loss["conf"]:>7.4f}'),
        (f'        Acc@1 {train_loss["top1"]:.4f}  Acc@5 {train_loss["top5"]:.4f}  '
         f'Acc@10 {train_loss["top10"]:.4f}  Acc@20 {train_loss["top20"]:.4f}  '
         f'MRR {train_loss["mrr"]:.4f}'),
        (f' Val    factual   Acc@1 {fact["top1"]:.4f}  Acc@5 {fact["top5"]:.4f}  '
         f'Acc@10 {fact["top10"]:.4f}  Acc@20 {fact["top20"]:.4f}  MRR {fact["mrr"]:.4f}'),
        (f'        deconf    Acc@1 {deconf["top1"]:.4f}  Acc@5 {deconf["top5"]:.4f}  '
         f'Acc@10 {deconf["top10"]:.4f}  Acc@20 {deconf["top20"]:.4f}  MRR {deconf["mrr"]:.4f}'),
    ]
    if saved:
        lines.append(f' * Saved best checkpoint  (score={score:.4f})')
    lines.append(SEP)
    return '\n'.join(lines)


def train(args):
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
                 f'adv={args.lambda_adv} recon={args.lambda_recon}')
    logging.info(SEP)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump({k: (str(v) if k == 'device' else v) for k, v in vars(args).items()},
                  f, sort_keys=False)
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(ROOT / 'causal', zipf, include_format=['.py'])
    zipf.close()

    logging.info('[1/4] Loading trajectories & POI confounder table...')
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)
    nodes_df = load_nodes_df(args.data_node_feats)

    poi_ids = list(nodes_df['node_name/poi_id'].tolist())
    poi_id2idx = dict(zip(poi_ids, range(len(poi_ids))))
    cat_ids = list(dict.fromkeys(nodes_df[args.feature2].tolist()))
    cat_id2idx = dict(zip(cat_ids, range(len(cat_ids))))
    poi_idx2cat_idx = {}
    for _, row in nodes_df.iterrows():
        poi_idx2cat_idx[poi_id2idx[row['node_name/poi_id']]] = cat_id2idx[row[args.feature2]]
    user_ids = [str(u) for u in sorted(set(train_df['user_id'].astype(str).tolist()))]
    user_id2idx = dict(zip(user_ids, range(len(user_ids))))

    table = build_poi_table(nodes_df, train_df, args, poi_id2idx)

    logging.info('[2/4] Building dataloaders...')
    train_ds = TrajectoryDataset(
        train_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        args.time_feature, args.short_traj_thres, skip_unknown_user=False)
    val_ds = TrajectoryDataset(
        val_df, poi_id2idx, user_id2idx, poi_idx2cat_idx,
        args.time_feature, args.short_traj_thres, skip_unknown_user=True)

    train_pairs = []
    for s in train_ds.samples:
        for o, d in zip(s['poi'], s['label_poi']):
            train_pairs.append((o, d))
    fill_transition_priors(table, train_pairs)

    logging.info(f'        POIs={table.num_pois} cats={len(cat_id2idx)} users={len(user_id2idx)} '
                 f'train_trajs={len(train_ds)} val_trajs={len(val_ds)} '
                 f'areas={table.num_areas} acc_bins={table.num_acc_bins}')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=False,
        num_workers=args.workers, collate_fn=collate_pad)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.workers, collate_fn=collate_pad)

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
        }, f)

    logging.info('[4/4] Start training...')
    max_val_score = -np.inf
    train_hist, val_hist = [], []

    for epoch in range(args.epochs):
        model.train()
        tr_parts = {k: [] for k in ('loss', 'main', 'pref', 'conf', 'adv', 'recon', 'top1', 'top5', 'top10', 'top20', 'mrr')}
        n_train = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs} train',
                    leave=False, dynamic_ncols=True)
        for b_idx, batch in enumerate(pbar):
            if args.max_batches and b_idx >= args.max_batches:
                break
            batch = _to_device(batch, args.device)
            optimizer.zero_grad()
            parts = compute_losses(model, batch, buffers, args, ce)
            parts['loss'].backward()
            optimizer.step()

            y_np = batch['y_poi'].detach().cpu().numpy()
            s_np = parts['s'].detach().cpu().numpy()
            batch_m = []
            for i, L in enumerate(batch['lengths']):
                batch_m.append(basic_metrics(y_np[i, :L], s_np[i, :L]))
            tr_parts['loss'].append(float(parts['loss'].detach().cpu()))
            tr_parts['main'].append(float(parts['main'].detach().cpu()))
            tr_parts['pref'].append(float(parts['pref'].detach().cpu()))
            tr_parts['conf'].append(float(parts['conf'].detach().cpu()))
            tr_parts['adv'].append(float(parts['adv'].detach().cpu()))
            tr_parts['recon'].append(float(parts['recon'].detach().cpu()))
            tr_parts['top1'].append(np.mean([m['top1'] for m in batch_m]))
            tr_parts['top5'].append(np.mean([m['top5'] for m in batch_m]))
            tr_parts['top10'].append(np.mean([m['top10'] for m in batch_m]))
            tr_parts['top20'].append(np.mean([m['top20'] for m in batch_m]))
            tr_parts['mrr'].append(np.mean([m['mrr'] for m in batch_m]))
            n_train += 1
            pbar.set_postfix(loss=f'{tr_parts["loss"][-1]:.2f}',
                             top1=f'{tr_parts["top1"][-1]:.3f}', refresh=False)

        model.eval()
        meter_f, meter_d = SliceMeter(), SliceMeter()
        val_losses = []
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{args.epochs} val  ',
                        leave=False, dynamic_ncols=True)
            for vb_idx, batch in enumerate(vbar):
                if args.max_val_batches and vb_idx >= args.max_val_batches:
                    break
                batch = _to_device(batch, args.device)
                parts = compute_losses(model, batch, buffers, args, ce)
                val_losses.append(float(parts['loss'].detach().cpu()))
                eval_batch_metrics(parts, batch, buffers, meter_f, meter_d)
                vbar.set_postfix(loss=f'{val_losses[-1]:.2f}', refresh=False)

        fact = meter_f.summary()['overall']
        deconf = meter_d.summary()['overall']
        train_m = {k: float(np.mean(v)) for k, v in tr_parts.items()}
        val_loss = float(np.mean(val_losses)) if val_losses else np.inf
        scheduler.step(val_loss)
        # Monitor factual Acc@1/Acc@20 like GETNext, but never select ckpt by deconf
        # scores (Appendix D / §7: do not "刷" factual Acc with deconf ranks).
        monitor_score = float(fact['top1'] * 4 + fact['top20']) if fact['top1'] is not None else -np.inf
        saved = False
        if args.save_weights and fact['top1'] is not None and monitor_score >= max_val_score:
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
                'epoch_val_factual': fact,
                'epoch_val_deconf': deconf,
            }
            torch.save(state, os.path.join(ckpt_dir, 'best_epoch.state.pt'))
            with open(os.path.join(ckpt_dir, 'best_epoch.txt'), 'w') as f:
                print({'factual': fact, 'deconf_pref': deconf, 'val_loss': val_loss}, file=f)
            max_val_score = monitor_score
            saved = True

        logging.info(format_epoch_summary(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], train_m, fact, deconf,
            saved=saved, score=max_val_score if saved else None))
        train_hist.append(train_m)
        val_hist.append({'loss': val_loss, 'factual': fact, 'deconf_pref': deconf})
        _write_hist(args.save_dir, train_hist, val_hist)

    logging.info(f'Training finished. Best val score={max_val_score:.4f}')
    logging.info(f'Checkpoints: {os.path.join(args.save_dir, "checkpoints")}')


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _write_hist(save_dir, train_hist, val_hist):
    def _dump(path, rows, prefix):
        with open(path, 'w') as f:
            if not rows:
                return
            for key in rows[0].keys():
                if key in ('factual', 'deconf_pref'):
                    continue
                vals = [float(f'{r[key]:.4f}') for r in rows]
                print(f'{prefix}_{key}_list={vals}', file=f)
    _dump(os.path.join(save_dir, 'metrics-train.txt'), train_hist, 'train')
    with open(os.path.join(save_dir, 'metrics-val.txt'), 'w') as f:
        print(f'val_loss_list={[float(f"{r["loss"]:.4f}") for r in val_hist]}', file=f)
        for split in ('factual', 'deconf_pref'):
            for key in ('top1', 'top5', 'top10', 'top20', 'mrr'):
                vals = [float(f'{r[split][key]:.4f}') for r in val_hist]
                print(f'val_{split}_{key}_list={vals}', file=f)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
    warnings.filterwarnings('ignore', message='.*verbose parameter is deprecated.*')
    args = parameter_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
