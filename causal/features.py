"""附录 D.2：事先算好每个 POI 的混杂特征 C，训练时只查表、不再现场发明。

人话
----
模型要处理三类「不是兴趣、但会影响下一站」的东西：
  c_acc  : 从当前点 p_T 走到候选点 p 有多远（分成几个距离桶）
  c_pop  : 这个地点在训练集里有多热门（分成几个热度档）
  c_area : 地点落在哪一块地理格子
  c_hour : 现在大概几点（用 GETNext 已有的日内时间特征分桶）

为什么要事先算、而且按「候选点 p」来算？
  附录 D.3.4 要求：距离/热度不能偷偷写进共享向量 h（否则标签泄漏，
  模型等于提前看见了「答案离我多远」）。推理时要对词表里每一个 p
  复现同样的特征，所以这里做成 (起点, 终点) 的大表。

本文件不训练网络，只准备查找表，给 model.py / train.py 用。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


def parse_dist_edges(dist_bins_str):
    """把命令行字符串 '0.5,1,2,5,10' 变成公里切分点。最后一档默认是「更远」。"""
    edges = [float(x.strip()) for x in dist_bins_str.split(',') if x.strip()]
    if not edges:
        raise ValueError('dist-bins must contain at least one edge')
    return np.asarray(edges, dtype=np.float64)


def pairwise_haversine_km(lat, lon):
    """地球表面两点距离（公里），一次性算完所有 POI 对。

    输入 lat/lon: 每个地点一个经纬度，长度 N
    输出: (N, N) 矩阵，第 (i,j) 格 = 从地点 i 到地点 j 的公里数
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
    """连续值 → 整数桶号。例如距离 0.3km 落在第 0 档，8km 落在较后的档。"""
    return np.digitize(values, edges, right=True).astype(np.int64)


def quantile_edges(values, n_bins):
    """按分位数切热度档，让「很冷 / 较冷 / 较热 / 很热」样本数量差不多（附录 D.8）。"""
    if n_bins < 2:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.unique(np.percentile(values, qs))
    return edges


def area_id_from_latlon(lat, lon, lat_min, lon_min, grid_deg, n_lon):
    """把经纬度投到网格上，得到区域整数 id（类似把城市切成棋盘格子）。"""
    lat_bin = np.floor((lat - lat_min) / grid_deg).astype(np.int64)
    lon_bin = np.floor((lon - lon_min) / grid_deg).astype(np.int64)
    lat_bin = np.clip(lat_bin, 0, None)
    lon_bin = np.clip(lon_bin, 0, n_lon - 1)
    return lat_bin * n_lon + lon_bin


@dataclass
class PoiConfounderTable:
    """全体 POI 共用的一张「混杂属性表」。下标就是模型里的 POI 编号 0..N-1。"""
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

    def hour_bin(self, norm_in_day_time):
        """GETNext 的时间特征在 [0,1]（一天里的比例）→ 时刻桶 0..47（默认半小时一档）。"""
        t = np.asarray(norm_in_day_time, dtype=np.float64)
        b = np.floor(np.clip(t, 0.0, 0.999999) * self.num_hour_bins).astype(np.int64)
        return b

    def to_torch(self, device):
        """把前向计算要用的大表搬到 CPU 或 GPU，避免训练时反复拷贝。"""
        return {
            'dist_bin': torch.from_numpy(self.dist_bin).to(device=device, dtype=torch.long),
            'dist_km': torch.from_numpy(self.dist_km.astype(np.float32)).to(device),
            'log_pop': torch.from_numpy(self.log_pop.astype(np.float32)).to(device),
            'pop_bin': torch.from_numpy(self.pop_bin).to(device=device, dtype=torch.long),
            'area_id': torch.from_numpy(self.area_id).to(device=device, dtype=torch.long),
            'acc_prior': torch.from_numpy(self.acc_prior.astype(np.float32)).to(device),
            'pop_prior': torch.from_numpy(self.pop_prior.astype(np.float32)).to(device),
        }


def build_poi_table(nodes_df, train_df, args, poi_id2idx):
    """从 graph_X.csv + 训练集签到，拼出附录 D.2 的静态表 T_poi。

    热度只用「训练集」次数，不用验证/测试里的未来签到，避免 C_pop 泄漏。
    也不用轨迹流图 graph_A.csv（附录 A：GCN 会把热度/近邻再灌一遍）。
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
    )
    return table


def fill_transition_priors(table, train_pairs):
    """统计训练集「下一跳」落在各距离桶 / 热度档的比例 hat P(c)（附录 D.5 边缘化要用）。"""
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


def load_nodes_df(path):
    """读取 GETNext 的 graph_X.csv（地点 id、类别、经纬度等），不是邻接矩阵。"""
    return pd.read_csv(path)
