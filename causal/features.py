"""附录 D.2：事先算好每个 POI 的混杂特征 C，训练时只查表、不再现场发明。

人话
----
模型要处理三类「不是兴趣、但会影响下一站」的东西：
  c_acc  : 从当前点 p_T 走到候选点 p 有多远（分成几个距离桶）
  c_pop  : 这个地点在训练集里有多热门（分成几个热度档）
  c_area : 地点落在哪一块地理格子
  c_hour : 现在大概几点（用 GETNext 已有的日内时间特征分桶）

另外把 GETNext 的轨迹流图 graph_A **拆开用**，不灌进 h_z / e_p：
  log_tpop : 目的地入度 log1p(sum_i A_ij)，转移热度，进 s_conf
  a_rel    : 去掉热度 + 距离之后的残差相关，只进 factual 的 s_rel

为什么要事先算、而且按「候选点 p」来算？
  附录 D.3.4 要求：距离/热度不能偷偷写进共享向量 h（否则标签泄漏，
  模型等于提前看见了「答案离我多远」）。推理时要对词表里每一个 p
  复现同样的特征，所以这里做成 (起点, 终点) 的大表。
  附录 A：不要把原始 A 无约束地喂给 GCN / 拼进 token。

本文件不训练网络，只准备查找表，给 model.py / train.py 用。

形状记号：N=POI 数；K=距离桶；P=热度档；A=区域数；H=时刻桶。
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


def parse_dist_edges(dist_bins_str):
    """把命令行字符串 '0.5,1,2,5,10' 变成公里切分点。最后一档默认是「更远」。

    输入:
        dist_bins_str: str，例如 '0.5,1,2,5,10'
    输出:
        edges: (n_edges,) float64 numpy，n_edges 通常是 5
    """
    edges = [float(x.strip()) for x in dist_bins_str.split(',') if x.strip()]
    if not edges:
        raise ValueError('dist-bins must contain at least one edge')
    return np.asarray(edges, dtype=np.float64)


def pairwise_haversine_km(lat, lon):
    """地球表面两点距离（公里），一次性算完所有 POI 对。

    输入:
        lat: (N,) 纬度
        lon: (N,) 经度
    输出:
        (N, N) float64，第 (i,j) 格 = 从地点 i 到地点 j 的公里数
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat1 = np.radians(lat)[:, None]
    lon1 = np.radians(lon)[:, None]
    lat2 = np.radians(lat)[None, :]
    lon2 = np.radians(lon)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bucketize(values, edges):
    """连续值 → 整数桶号。例如距离 0.3km 落在第 0 档，8km 落在较后的档。

    输入:
        values: 任意形状，例如 dist_km (N, N) 或 pop (N,)
        edges:  (n_edges,) 切分点
    输出:
        与 values 同形状的 int64 桶号
    """
    return np.digitize(values, edges, right=True).astype(np.int64)


def quantile_edges(values, n_bins):
    """按分位数切热度档，让「很冷 / 较冷 / 较热 / 很热」样本数量差不多（附录 D.8）。

    输入:
        values: (M,) 例如非零热度
        n_bins: int = P，想切成几档
    输出:
        edges: (≤ P-1,) float64，去重后的分位切分点；n_bins<2 时为空数组
    """
    if n_bins < 2:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.unique(np.percentile(values, qs))
    return edges


def area_id_from_latlon(lat, lon, lat_min, lon_min, grid_deg, n_lon):
    """把经纬度投到网格上，得到区域整数 id（类似把城市切成棋盘格子）。

    输入:
        lat, lon: (N,)
        lat_min, lon_min, grid_deg: 标量
        n_lon: int，经度方向格子数
    输出:
        (N,) int64 区域 id（压缩编号之前）
    """
    lat_bin = np.floor((lat - lat_min) / grid_deg).astype(np.int64)
    lon_bin = np.floor((lon - lon_min) / grid_deg).astype(np.int64)
    lat_bin = np.clip(lat_bin, 0, None)
    lon_bin = np.clip(lon_bin, 0, n_lon - 1)
    return lat_bin * n_lon + lon_bin


@dataclass
class PoiConfounderTable:
    """全体 POI 共用的一张「混杂属性表」。下标就是模型里的 POI 编号 0..N-1。

    字段形状:
        lat / lon / pop / log_pop / area_id / pop_bin: (N,)
        log_tpop: (N,)  转移入度热度；a_rel: (N, N) 残差相关
        dist_km / dist_bin: (N, N)
        dist_edges: (n_edges,)  pop_edges: (≤ P-1,)
        acc_prior: (K,)  pop_prior: (P,)  填完 fill_transition_priors 之后才有
        其余 num_* / lat_min 等: 标量
    """
    num_pois: int
    lat: np.ndarray
    lon: np.ndarray
    pop: np.ndarray                 # 训练集签到次数（热度）
    log_pop: np.ndarray             # log(1+热度)，给混杂分用，避免极端值
    area_id: np.ndarray             # (N,) 每个点的区域
    pop_bin: np.ndarray             # (N,) 热度档
    dist_km: np.ndarray             # (N, N) 公里距离
    dist_bin: np.ndarray            # (N, N) 距离桶 = c_acc(起点, 终点)
    dist_edges: np.ndarray
    pop_edges: np.ndarray
    num_acc_bins: int
    num_pop_bins: int
    num_areas: int
    num_hour_bins: int
    lat_min: float
    lon_min: float
    grid_deg: float
    n_lon: int
    # 训练集里「下一跳」落在各距离桶 / 热度档的频率，给 deconf_sum 用
    acc_prior: np.ndarray = field(default=None)
    pop_prior: np.ndarray = field(default=None)
    median_acc_bin: int = 0
    median_pop_bin: int = 0
    # graph_A 拆开：转移热度进 s_conf；残差相关只进 factual s_rel
    log_tpop: np.ndarray = field(default=None)
    a_rel: np.ndarray = field(default=None)

    def hour_bin(self, norm_in_day_time):
        """GETNext 的时间特征在 [0,1]（一天里的比例）→ 时刻桶 0..47（默认半小时一档）。

        输入:
            norm_in_day_time: 任意形状 float，常见 (B, T) 或 (T,)
        输出:
            与输入同形状的 int64，取值 0 .. H-1
        """
        t = np.asarray(norm_in_day_time, dtype=np.float64)
        b = np.floor(np.clip(t, 0.0, 0.999999) * self.num_hour_bins).astype(np.int64)
        return b

    def to_torch(self, device):
        """把前向计算要用的大表搬到 CPU 或 GPU，避免训练时反复拷贝。

        输入:
            device: torch.device
        输出: dict
            dist_bin:  (N, N) long
            dist_km:   (N, N) float32
            log_pop:   (N,)   float32
            log_tpop:  (N,)   float32  转移入度热度
            a_rel:     (N, N) float32  残差相关（factual 专用）
            pop_bin:   (N,)   long
            area_id:   (N,)   long
            acc_prior: (K,)   float32
            pop_prior: (P,)   float32
        """
        n = int(self.num_pois)
        log_tpop = self.log_tpop if self.log_tpop is not None else np.zeros(n, dtype=np.float32)
        a_rel = self.a_rel if self.a_rel is not None else np.zeros((n, n), dtype=np.float32)
        return {
            'dist_bin': torch.from_numpy(self.dist_bin).to(device=device, dtype=torch.long),
            'dist_km': torch.from_numpy(self.dist_km.astype(np.float32)).to(device),
            'log_pop': torch.from_numpy(self.log_pop.astype(np.float32)).to(device),
            'log_tpop': torch.from_numpy(np.asarray(log_tpop, dtype=np.float32)).to(device),
            'a_rel': torch.from_numpy(np.asarray(a_rel, dtype=np.float32)).to(device),
            'pop_bin': torch.from_numpy(self.pop_bin).to(device=device, dtype=torch.long),
            'area_id': torch.from_numpy(self.area_id).to(device=device, dtype=torch.long),
            'acc_prior': torch.from_numpy(self.acc_prior.astype(np.float32)).to(device),
            'pop_prior': torch.from_numpy(self.pop_prior.astype(np.float32)).to(device),
        }


def build_poi_table(nodes_df, train_df, args, poi_id2idx):
    """从 graph_X.csv + 训练集签到，拼出附录 D.2 的静态表 T_poi。

    热度只用「训练集」次数，不用验证/测试里的未来签到，避免 C_pop 泄漏。
    graph_A.csv 不在这里读：训练时再调用 fill_graph_transition_tables，
    拆成转移热度 log_tpop 和残差相关 a_rel，不把原始 A 灌进 embedding。

    输入:
        nodes_df: DataFrame = graph_X.csv，一行一个 POI，见 train.py 文件头
        train_df: DataFrame = NYC_train.csv，一行一次签到，用来数每个 POI 的训练集热度
        args: 超参（dist_bins / pop_bins / area_grid_deg 等标量）
        poi_id2idx: dict，POI 字符串 id → 0..N-1（与 nodes_df 行序一致）
    输出:
        PoiConfounderTable，主要数组：
          lat/lon/pop/log_pop/area_id/pop_bin: (N,)
          dist_km / dist_bin: (N, N)
        此时 acc_prior / pop_prior 还是 None，要等 fill_transition_priors
    """
    num_pois = len(poi_id2idx)
    lat = np.zeros(num_pois, dtype=np.float64)
    lon = np.zeros(num_pois, dtype=np.float64)
    pop = np.zeros(num_pois, dtype=np.float64)

    # 节点表里的经纬度和兜底热度（行顺序不一定等于模型下标，所以按 id 映射）
    for _, row in nodes_df.iterrows():
        poi_id = row['node_name/poi_id']
        if poi_id not in poi_id2idx:
            continue
        idx = poi_id2idx[poi_id]
        lat[idx] = float(row[args.feature3])
        lon[idx] = float(row[args.feature4])
        pop[idx] = float(row[args.feature1])

    # 有训练集统计时，用训练集次数覆盖兜底热度
    train_counts = train_df['POI_id'].value_counts()
    for poi_id, cnt in train_counts.items():
        if poi_id in poi_id2idx:
            pop[poi_id2idx[poi_id]] = float(cnt)

    log_pop = np.log1p(pop)

    # 距离（公里）→ 距离桶
    dist_edges = parse_dist_edges(args.dist_bins)
    dist_km = pairwise_haversine_km(lat, lon)
    dist_bin = bucketize(dist_km, dist_edges)
    num_acc_bins = int(dist_bin.max()) + 1

    # 热度 → 分位档
    pop_edges = quantile_edges(pop[pop > 0] if np.any(pop > 0) else pop, args.pop_bins)
    pop_bin = bucketize(pop, pop_edges)
    num_pop_bins = int(max(pop_bin.max() + 1, args.pop_bins))

    # 经纬度网格。稍微外扩一点，避免刚好落在边界上的点被切出去
    grid = float(args.area_grid_deg)
    lat_min = float(lat.min()) - 1e-6
    lon_min = float(lon.min()) - 1e-6
    lat_max = float(lat.max()) + 1e-6
    lon_max = float(lon.max()) + 1e-6
    n_lat = int(np.ceil((lat_max - lat_min) / grid))
    n_lon = int(np.ceil((lon_max - lon_min) / grid))
    n_lat = max(n_lat, 1)
    n_lon = max(n_lon, 1)
    area_id = area_id_from_latlon(lat, lon, lat_min, lon_min, grid, n_lon)
    num_areas = int(n_lat * n_lon)
    # 很多格子是空的，把实际出现过的区域重新编号，embedding 表会小很多
    unique, compact = np.unique(area_id, return_inverse=True)
    area_id = compact.astype(np.int64)
    num_areas = int(unique.size)

    table = PoiConfounderTable(
        num_pois=num_pois,
        lat=lat,
        lon=lon,
        pop=pop,
        log_pop=log_pop,
        area_id=area_id,
        pop_bin=pop_bin,
        dist_km=dist_km,
        dist_bin=dist_bin,
        dist_edges=dist_edges,
        pop_edges=pop_edges,
        num_acc_bins=num_acc_bins,
        num_pop_bins=num_pop_bins,
        num_areas=num_areas,
        num_hour_bins=int(args.time_units),
        lat_min=lat_min,
        lon_min=lon_min,
        grid_deg=grid,
        n_lon=n_lon,
        log_tpop=np.zeros(num_pois, dtype=np.float32),
        a_rel=np.zeros((num_pois, num_pois), dtype=np.float32),
    )
    return table


def fill_transition_priors(table, train_pairs):
    """统计训练集「下一跳」落在各距离桶 / 热度档的比例 hat P(c)（附录 D.5 边缘化要用）。

    输入:
        table: PoiConfounderTable
        train_pairs: list[(origin, dest)]，长度 = 训练轨迹里有效的下一步数
    输出:
        同一个 table（原地写入）
          acc_prior: (K,)  pop_prior: (P,)
          median_acc_bin / median_pop_bin: 标量 int
    """
    acc_counts = np.zeros(table.num_acc_bins, dtype=np.float64)
    pop_counts = np.zeros(table.num_pop_bins, dtype=np.float64)
    for origin, dest in train_pairs:
        acc_counts[int(table.dist_bin[origin, dest])] += 1.0
        pop_counts[int(table.pop_bin[dest])] += 1.0
    acc_counts = np.maximum(acc_counts, 1.0)
    pop_counts = np.maximum(pop_counts, 1.0)
    table.acc_prior = acc_counts / acc_counts.sum()
    table.pop_prior = pop_counts / pop_counts.sum()
    # 最常见的那一档当作 do(C=c_bar) 的默认干预值
    table.median_acc_bin = int(np.argmax(table.acc_prior))
    table.median_pop_bin = int(np.argmax(table.pop_prior))
    return table


def decompose_graph_adj(A, dist_km, n_dist_bins=8, eps=1e-8):
    """把观测转移 A 拆成「目的地热度」和「去掉热度+距离后的残差相关」。

    log A_ij ≈ b_j + g(dist_ij) + r_ij
      b_j     : 列和（谁常被转到）→ log_tpop
      g(dist) : 按距离桶减去 PMI 均值（近邻转移不该冒充功能相关）
      r_ij    : 残差 a_rel，给 factual 的 s_rel 用

    输入:
        A: (N, N) 转移频次，A[i,j] = i→j
        dist_km: (N, N) 公里距离
        n_dist_bins: 拟合 g(dist) 时用的距离分位桶数
        eps: 数值稳定
    输出:
        log_tpop: (N,) float32 = log1p(列和)
        a_rel:    (N, N) float32，标准化并 clip 到 [-8, 8]，对角线为 0
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f'A must be square, got {A.shape}')
    n = A.shape[0]
    col_sum = A.sum(axis=0)
    row_sum = A.sum(axis=1)
    log_tpop = np.log1p(col_sum).astype(np.float32)

    total = float(A.sum()) + eps
    pmi = (np.log(A + eps)
           - np.log(row_sum[:, None] + eps)
           - np.log(col_sum[None, :] + eps)
           + np.log(total))

    dist = np.asarray(dist_km, dtype=np.float64)
    if dist.shape != A.shape:
        raise ValueError(f'dist_km shape {dist.shape} != A shape {A.shape}')
    finite = np.isfinite(dist)
    residual = pmi.copy()
    if np.any(finite):
        sample = dist[finite]
        bins = np.percentile(sample, np.linspace(0.0, 100.0, n_dist_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        # 分位重复时 digitize 仍给出 0..n_dist_bins-1
        bin_id = np.digitize(dist, bins[1:-1], right=True)
        for b in range(n_dist_bins):
            mask = (bin_id == b) & finite
            if np.any(mask):
                residual[mask] -= pmi[mask].mean()

    std = float(residual.std())
    if std > eps:
        residual = residual / std
    np.fill_diagonal(residual, 0.0)
    residual = np.clip(residual, -8.0, 8.0).astype(np.float32)
    if residual.shape != (n, n):
        raise ValueError('internal shape error in decompose_graph_adj')
    return log_tpop, residual


def fill_graph_transition_tables(table, adj_path, node_feats_path, poi_id2idx):
    """读 graph_A.csv，按 POI id 对齐到因果词表，写入 log_tpop / a_rel。

    graph_A 行序 = graph_X.csv 行序。因果词表也来自 graph_X，但 val/test
    多出来的点（若有）会对不上，对不上的行/列保持 0。

    找不到邻接矩阵时保持全 0，训练仍能跑，只是没有转移先验。

    输入:
        table: PoiConfounderTable（已有 dist_km）
        adj_path: graph_A.csv
        node_feats_path: graph_X.csv（用来把 A 的行号映射到 poi_id2idx）
        poi_id2idx: {POI 字符串: 0..N-1}
    输出:
        同一个 table（原地写入），以及 stats dict
    """
    n = int(table.num_pois)
    if table.log_tpop is None:
        table.log_tpop = np.zeros(n, dtype=np.float32)
    if table.a_rel is None:
        table.a_rel = np.zeros((n, n), dtype=np.float32)

    if not adj_path or not os.path.isfile(adj_path):
        return {'loaded': False, 'n_mapped': 0, 'reason': 'missing_adj'}

    from dataloader import load_graph_adj_mtx
    A_raw = np.asarray(load_graph_adj_mtx(adj_path), dtype=np.float64)
    if A_raw.ndim != 2 or A_raw.shape[0] != A_raw.shape[1]:
        return {'loaded': False, 'n_mapped': 0, 'reason': f'bad_shape:{A_raw.shape}'}

    n_g = int(A_raw.shape[0])
    graph_ids = None
    if node_feats_path and os.path.isfile(node_feats_path):
        gx = pd.read_csv(node_feats_path)
        graph_ids = list(gx.iloc[:, 0].tolist())

    A_full = np.zeros((n, n), dtype=np.float64)
    if graph_ids is None:
        n_use = min(n, n_g)
        A_full[:n_use, :n_use] = A_raw[:n_use, :n_use]
        n_mapped = n_use
    else:
        mapped_g, mapped_c = [], []
        for gi, poi in enumerate(graph_ids):
            if gi >= n_g:
                break
            idx = poi_id2idx.get(poi)
            if idx is None:
                idx = poi_id2idx.get(str(poi))
            if idx is not None:
                mapped_g.append(gi)
                mapped_c.append(int(idx))
        n_mapped = len(mapped_g)
        if n_mapped:
            g_idx = np.asarray(mapped_g, dtype=np.int64)
            c_idx = np.asarray(mapped_c, dtype=np.int64)
            A_full[np.ix_(c_idx, c_idx)] = A_raw[np.ix_(g_idx, g_idx)]

    log_tpop, a_rel = decompose_graph_adj(A_full, table.dist_km)
    table.log_tpop = log_tpop
    table.a_rel = a_rel
    return {
        'loaded': True,
        'n_mapped': int(n_mapped),
        'n_graph': n_g,
        'n_pois': n,
        'nnz': int((A_full > 0).sum()),
        'tpop_max': float(log_tpop.max()),
        'a_rel_std': float(a_rel.std()),
    }


def load_nodes_df(path):
    """读取 GETNext 的 graph_X.csv（地点 id、类别、经纬度等），不是邻接矩阵。

    输入:
        path: str，默认 dataset/NYC/graph_X.csv
    输出:
        DataFrame，NYC 约 4980 行，列：
          node_name/poi_id, checkin_cnt, poi_catid, poi_catid_code,
          poi_catname, latitude, longitude
        一行一个 POI；行顺序 = 模型 embedding 下标 0..N-1。
    """
    return pd.read_csv(path)
