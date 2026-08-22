"""Command-line flags for causal next-POI (Appendix D).

Layout mirrors GETNext `param_parser.py` so the train/predict flow feels
the same. Causal-specific knobs (lambda_*, buckets, inference mode) are
added here; GCN / NodeAttnMap flags are intentionally omitted — Appendix D
uses `nn.Embedding` for e_p and keeps C out of the Transformer tokens.
"""
import argparse

import torch


def _default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(
        description='Causal next-POI (Appendix D: score split + h_z/h_c).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=str(_default_device()),
                        help='cpu / cuda / cuda:0')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Force CPU even if CUDA is available')

    # ---- Data (same CSV layout as GETNext; graph_A is NOT used) ----
    parser.add_argument('--data-train', type=str, default='dataset/NYC/NYC_train.csv')
    parser.add_argument('--data-val', type=str, default='dataset/NYC/NYC_val.csv')
    parser.add_argument('--data-node-feats', type=str, default='dataset/NYC/graph_X.csv',
                        help='POI table: id, checkin_cnt, category, lat, lon')
    parser.add_argument('--short-traj-thres', type=int, default=2)
    parser.add_argument('--time-units', type=int, default=48,
                        help='Hour-bucket count (0.5h × 48 = 24h)')
    parser.add_argument('--time-feature', type=str, default='norm_in_day_time')
    parser.add_argument('--feature1', type=str, default='checkin_cnt')
    parser.add_argument('--feature2', type=str, default='poi_catid')
    parser.add_argument('--feature3', type=str, default='latitude')
    parser.add_argument('--feature4', type=str, default='longitude')

    # ---- Encoder (Transformer + id embeddings; no GCN) ----
    parser.add_argument('--poi-embed-dim', type=int, default=128,
                        help='d_e for e_p; also default d_z so <h_z, e_p> is valid')
    parser.add_argument('--user-embed-dim', type=int, default=128)
    parser.add_argument('--time-embed-dim', type=int, default=32)
    parser.add_argument('--cat-embed-dim', type=int, default=32)
    parser.add_argument('--hz-dim', type=int, default=None,
                        help='Interest proxy dim; default = poi-embed-dim (tied dot-product)')
    parser.add_argument('--hc-dim', type=int, default=64,
                        help='Confounder summary h_c dim')
    parser.add_argument('--transformer-nhid', type=int, default=1024)
    parser.add_argument('--transformer-nlayers', type=int, default=2)
    parser.add_argument('--transformer-nhead', type=int, default=2)
    parser.add_argument('--transformer-dropout', type=float, default=0.3)

    # ---- Appendix D confounder buckets (D.2 / D.8) ----
    parser.add_argument('--dist-bins', type=str, default='0.5,1,2,5,10',
                        help='Distance bucket edges in km (last bin is +inf)')
    parser.add_argument('--pop-bins', type=int, default=4, help='Popularity quantiles')
    parser.add_argument('--area-grid-deg', type=float, default=0.02,
                        help='Lat/lon grid size in degrees (~2km at NYC latitude)')
    parser.add_argument('--bar-acc-bin', type=int, default=-1,
                        help='do(C) access bucket; -1 = median training bucket')
    parser.add_argument('--bar-pop-bin', type=int, default=-1,
                        help='do(C) pop bucket; -1 = median training bucket')

    # ---- Losses (D.4); main CE weight is always 1 ----
    parser.add_argument('--lambda-pref', type=float, default=0.05,
                        help='Weight of same-distance-band contrastive on s_pref')
    parser.add_argument('--lambda-conf', type=float, default=0.05,
                        help='Weight of s_conf <-> g_tilde MSE alignment')
    parser.add_argument('--lambda-adv', type=float, default=0.05,
                        help='Weight of GRL adversarial CE on h_z')
    parser.add_argument('--lambda-recon', type=float, default=0.05,
                        help='Weight of h_c reconstruction CE for C')
    parser.add_argument('--lambda-cat', type=float, default=0.05,
                        help='Optional category aux on h_z (D.4.2b); 0 to disable')
    parser.add_argument('--lambda-time', type=float, default=0.0,
                        help='Optional time-regression aux (GETNext-like); default off')
    parser.add_argument('--conf-aux-ce', action='store_true', default=False,
                        help='Add weak CE on s_conf (D.4.3b); off by default')
    parser.add_argument('--align-alpha', type=float, default=0.2,
                        help='Hand-crafted g_tilde: -alpha * dist_km')
    parser.add_argument('--align-beta', type=float, default=0.3,
                        help='Hand-crafted g_tilde: +beta * log(1+pop)  (pop bias into s_conf)')

    # ---- Train ----
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=0,
                        help='Debug/smoke: cap train batches per epoch (0 = all)')
    parser.add_argument('--max-val-batches', type=int, default=0,
                        help='Debug/smoke: cap val batches (0 = all)')
    parser.add_argument('--save-weights', action='store_true', default=True)
    parser.add_argument('--project', default='runs/causal',
                        help='Run root; causal outputs stay out of GETNext runs/train/')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)

    return parser.parse_args()
