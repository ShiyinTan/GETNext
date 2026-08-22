"""Appendix D.2: static POI table and discrete confounders C.

C = (c_acc, c_pop, c_area, c_hour). All candidate-level geographic / popularity
features are functions of (origin p_T, candidate p) so they can be replayed
at inference for every p in the vocabulary (D.3.4, no label leak into h).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


def parse_dist_edges(dist_bins_str):
    """'0.5,1,2,5,10' -> array of finite edges; last implicit bin is +inf."""
    edges = [float(x.strip()) for x in dist_bins_str.split(',') if x.strip()]
    if not edges:
        raise ValueError('dist-bins must contain at least one edge')
    return np.asarray(edges, dtype=np.float64)


def pairwise_haversine_km(lat, lon):
    """Vectorized pairwise Haversine. lat/lon: (N,) degrees -> (N, N) km."""
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
    """Right-open bins: (-inf, e0], (e0, e1], ..., (e_{k-1}, +inf). Returns int64."""
    return np.digitize(values, edges, right=True).astype(np.int64)


def quantile_edges(values, n_bins):
    """Interior quantile cuts so each bin is ~equally populated (D.8: bucket C)."""
    if n_bins < 2:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.unique(np.percentile(values, qs))
    return edges


def area_id_from_latlon(lat, lon, lat_min, lon_min, grid_deg, n_lon):
    lat_bin = np.floor((lat - lat_min) / grid_deg).astype(np.int64)
    lon_bin = np.floor((lon - lon_min) / grid_deg).astype(np.int64)
    lat_bin = np.clip(lat_bin, 0, None)
    lon_bin = np.clip(lon_bin, 0, n_lon - 1)
    return lat_bin * n_lon + lon_bin


@dataclass
class PoiConfounderTable:
    """Shared lookup tables. Index is the model POI index in [0, num_pois)."""
    num_pois: int
    lat: np.ndarray
    lon: np.ndarray
    pop: np.ndarray                 # raw train check-in count
    log_pop: np.ndarray             # log(1+pop)
    area_id: np.ndarray             # (N,)
    pop_bin: np.ndarray             # (N,)
    dist_km: np.ndarray             # (N, N)
    dist_bin: np.ndarray            # (N, N) c_acc(origin, dest)
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
    # Empirical P(c_acc) / P(c_pop) on training next-hop pairs (for deconf-sum).
    acc_prior: np.ndarray = field(default=None)
    pop_prior: np.ndarray = field(default=None)
    median_acc_bin: int = 0
    median_pop_bin: int = 0

    def hour_bin(self, norm_in_day_time):
        """Map GETNext's [0,1] time feature to an hour bucket in [0, time_units)."""
        t = np.asarray(norm_in_day_time, dtype=np.float64)
        b = np.floor(np.clip(t, 0.0, 0.999999) * self.num_hour_bins).astype(np.int64)
        return b

    def to_torch(self, device):
        """Move dense tables used in the forward pass onto the train device."""
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
    """Build T_poi from graph_X.csv + train check-ins (Appendix D.2).

    Popularity uses *training* frequency, not the raw graph checkin_cnt mixed
    with val/test, so C_pop does not leak future counts into training.
    """
    num_pois = len(poi_id2idx)
    lat = np.zeros(num_pois, dtype=np.float64)
    lon = np.zeros(num_pois, dtype=np.float64)
    pop = np.zeros(num_pois, dtype=np.float64)

    # Coordinates / fallback pop from the node table (row order is not index order).
    for _, row in nodes_df.iterrows():
        poi_id = row['node_name/poi_id']
        if poi_id not in poi_id2idx:
            continue
        idx = poi_id2idx[poi_id]
        lat[idx] = float(row[args.feature3])
        lon[idx] = float(row[args.feature4])
        pop[idx] = float(row[args.feature1])

    # Override pop with train-split counts when available.
    train_counts = train_df['POI_id'].value_counts()
    for poi_id, cnt in train_counts.items():
        if poi_id in poi_id2idx:
            pop[poi_id2idx[poi_id]] = float(cnt)

    log_pop = np.log1p(pop)

    dist_edges = parse_dist_edges(args.dist_bins)
    dist_km = pairwise_haversine_km(lat, lon)
    dist_bin = bucketize(dist_km, dist_edges)
    num_acc_bins = int(dist_bin.max()) + 1

    pop_edges = quantile_edges(pop[pop > 0] if np.any(pop > 0) else pop, args.pop_bins)
    pop_bin = bucketize(pop, pop_edges)
    num_pop_bins = int(max(pop_bin.max() + 1, args.pop_bins))

    # Geographic grid. Pad slightly so min/max POIs do not sit on the outer edge.
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
    # Compact unused area ids so embedding tables stay small.
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
    """Estimate hat P(c_acc), hat P(c_pop) from (origin, dest) training hops (D.5.3)."""
    acc_counts = np.zeros(table.num_acc_bins, dtype=np.float64)
    pop_counts = np.zeros(table.num_pop_bins, dtype=np.float64)
    for origin, dest in train_pairs:
        acc_counts[int(table.dist_bin[origin, dest])] += 1.0
        pop_counts[int(table.pop_bin[dest])] += 1.0
    acc_counts = np.maximum(acc_counts, 1.0)
    pop_counts = np.maximum(pop_counts, 1.0)
    table.acc_prior = acc_counts / acc_counts.sum()
    table.pop_prior = pop_counts / pop_counts.sum()
    table.median_acc_bin = int(np.argmax(table.acc_prior))
    table.median_pop_bin = int(np.argmax(table.pop_prior))
    return table


def load_nodes_df(path):
    return pd.read_csv(path)
