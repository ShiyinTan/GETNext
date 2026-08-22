"""因果版训练的命令行参数（附录 D）。

排版故意贴近 GETNext 的 param_parser.py，所以 `--epochs / --batch / --lr / --no-cuda`
这些用法是一样的。多出来的是因果相关开关（lambda_*、距离/热度分桶）。

没有 GCN / NodeAttnMap 相关参数：附录 D 用 nn.Embedding 当 e_p，混杂 C 不进 token。
在仓库根目录运行: python causal/train.py --help
"""
import argparse

import torch


def _default_device():
    """有显卡就默认 cuda，没有就 cpu。云端这台机器没有 GPU。"""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(
        description='因果 next-POI（附录 D：分数分解 + h_z/h_c）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default=str(_default_device()),
                        help='cpu / cuda / cuda:0')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='即使有 GPU 也强制用 CPU')

    # ---- 数据：CSV 格式和 GETNext 相同；不用 graph_A.csv ----
    parser.add_argument('--data-train', type=str, default='dataset/NYC/NYC_train.csv',
                        help='训练轨迹 CSV')
    parser.add_argument('--data-val', type=str, default='dataset/NYC/NYC_val.csv',
                        help='验证轨迹 CSV')
    parser.add_argument('--data-node-feats', type=str, default='dataset/NYC/graph_X.csv',
                        help='地点表：id、签到次数、类别、经纬度（不用邻接矩阵）')
    parser.add_argument('--short-traj-thres', type=int, default=2,
                        help='短于该长度的轨迹丢掉（和 GETNext 一样）')
    parser.add_argument('--time-units', type=int, default=48,
                        help='一天划成多少个时刻桶（默认 0.5 小时 × 48）')
    parser.add_argument('--time-feature', type=str, default='norm_in_day_time',
                        help='CSV 里表示「一天中的时刻」的列名')
    parser.add_argument('--feature1', type=str, default='checkin_cnt')
    parser.add_argument('--feature2', type=str, default='poi_catid')
    parser.add_argument('--feature3', type=str, default='latitude')
    parser.add_argument('--feature4', type=str, default='longitude')

    # ---- 编码器：Transformer + id 嵌入，没有 GCN ----
    parser.add_argument('--poi-embed-dim', type=int, default=128,
                        help='地点向量 e_p 的维度；兴趣向量 h_z 默认与它相同，才能做点积')
    parser.add_argument('--user-embed-dim', type=int, default=128, help='用户嵌入维度')
    parser.add_argument('--time-embed-dim', type=int, default=32, help='时间嵌入维度')
    parser.add_argument('--cat-embed-dim', type=int, default=32, help='类别嵌入维度')
    parser.add_argument('--hz-dim', type=int, default=None,
                        help='兴趣代理 h_z 维度，默认等于 poi-embed-dim')
    parser.add_argument('--hc-dim', type=int, default=64, help='混杂摘要 h_c 维度')
    parser.add_argument('--transformer-nhid', type=int, default=1024, help='Transformer 隐层宽度')
    parser.add_argument('--transformer-nlayers', type=int, default=2, help='Transformer 层数')
    parser.add_argument('--transformer-nhead', type=int, default=2, help='注意力头数')
    parser.add_argument('--transformer-dropout', type=float, default=0.3)

    # ---- 附录 D.2 / D.8：把连续混杂切成桶，方便对抗、重建和边缘化 ----
    parser.add_argument('--dist-bins', type=str, default='0.5,1,2,5,10',
                        help='距离桶切分点，单位公里，最后一档是更远')
    parser.add_argument('--pop-bins', type=int, default=4, help='热度分成几档（分位数）')
    parser.add_argument('--area-grid-deg', type=float, default=0.02,
                        help='经纬度网格大小（度），纽约大约 2 公里')
    parser.add_argument('--bar-acc-bin', type=int, default=-1,
                        help='do(C) 时用的距离桶；-1 表示用训练集最常见的那一档')
    parser.add_argument('--bar-pop-bin', type=int, default=-1,
                        help='do(C) 时用的热度档；-1 表示用训练集最常见的那一档')

    # ---- 附录 D.4 损失权重；主 CE 的权重永远是 1 ----
    parser.add_argument('--lambda-pref', type=float, default=0.05,
                        help='同距离环带对比（只作用在兴趣分 s_pref 上）')
    parser.add_argument('--lambda-conf', type=float, default=0.05,
                        help='混杂分 s_conf 去对齐手工先验 g̃')
    parser.add_argument('--lambda-adv', type=float, default=0.05,
                        help='对抗：让 h_z 不容易猜出混杂 C')
    parser.add_argument('--lambda-recon', type=float, default=0.05,
                        help='重建：让 h_c 能够猜出混杂 C')
    parser.add_argument('--lambda-cat', type=float, default=0.05,
                        help='从 h_z 预测下一站类别（D.4.2b）；设 0 关闭')
    parser.add_argument('--lambda-time', type=float, default=0.0,
                        help='时间回归（GETNext 风格），默认关闭')
    parser.add_argument('--conf-aux-ce', action='store_true', default=False,
                        help='再给 s_conf 一个很弱的 CE；默认关，避免混杂通道太强')
    parser.add_argument('--align-alpha', type=float, default=0.2,
                        help='手工先验：-alpha × 距离(公里)')
    parser.add_argument('--align-beta', type=float, default=0.3,
                        help='手工先验：+beta × log(1+热度)，把流行度推进混杂通道')

    # ---- 训练超参（含义和 GETNext 相同）----
    parser.add_argument('--batch', type=int, default=20, help='一个 batch 几条轨迹')
    parser.add_argument('--epochs', type=int, default=200, help='训练多少轮')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.1,
                        help='验证损失不降时，学习率乘这个系数')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 正则')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader 进程数，云端建议 0')
    parser.add_argument('--max-batches', type=int, default=0,
                        help='每个 epoch 最多训练多少个 batch，0 表示全部（冒烟用）')
    parser.add_argument('--max-val-batches', type=int, default=0,
                        help='每个 epoch 最多验证多少个 batch，0 表示全部')
    parser.add_argument('--save-weights', action='store_true', default=True, help='是否存 checkpoint')
    parser.add_argument('--project', default='runs/causal',
                        help='输出根目录，和 GETNext 的 runs/train 分开')
    parser.add_argument('--name', default='exp', help='本次实验子目录名')
    parser.add_argument('--exist-ok', action='store_true', help='目录已存在也不自动改名')
    parser.add_argument('--verbose', action='store_true', default=False, help='打印更细的 batch 日志')

    return parser.parse_args()
