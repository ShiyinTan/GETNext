"""从签到序列构建用户无关的全局轨迹流图 (Trajectory Flow Map)"""
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_global_POI_checkin_graph(df, exclude_user=None):
    """从全体用户签到轨迹构建有向加权 POI 转移图。
    节点=POI，边=同一 trajectory 内相邻签到转移，边权=转移频次。
    """
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # ---------- 步骤1: 添加/更新 POI 节点及其属性 ----------
        for i, row in user_df.iterrows():
            node = row['POI_id']
            if node not in G.nodes():
                G.add_node(row['POI_id'],
                           checkin_cnt=1,
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # ---------- 步骤2: 按轨迹顺序添加转移边 prev_POI -> cur_POI ----------
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            # 轨迹起点或切换到新轨迹时不连边
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # 已有边则 weight+=1，否则新建边 weight=1
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    # 返回: NetworkX DiGraph，|V|=N_poi, |E|=转移边数
    return G


def save_graph_to_csv(G, dst_dir):
    """将图保存为邻接矩阵与节点特征表，行列顺序与节点列表一致。
    输出:
      graph_A.csv: A, shape (N_poi, N_poi)，A[i,j]=从节点i到j的转移频次
      graph_X.csv: 每行一个节点，含 checkin_cnt/类别/经纬度等
    """

    # ---------- 步骤3: 导出邻接矩阵 A ----------
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)  # sparse, shape (N_poi, N_poi)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # ---------- 步骤4: 导出节点特征表 ----------
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]，长度 N_poi
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},{poi_catname},'
                  f'{latitude},{longitude}', file=f)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """加载邻接矩阵。
    返回 A: shape (N_poi, N_poi)，A[i,j] 为从 i 到 j 的边权(转移频次)
    """
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """加载节点特征。
    返回 X: shape (N_poi, 4) = [签到次数, 类别, 纬度, 经度]
    """
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    dst_dir = r'dataset/NYC'

    # 主流程: 读训练签到 → 建全局 POI 转移图 → 落盘
    train_df = pd.read_csv(os.path.join(dst_dir, 'NYC_train.csv'))
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)

    # 保存 graph.pkl / graph_A.csv+graph_X.csv / edgelist
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)
