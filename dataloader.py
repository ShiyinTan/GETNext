import numpy as np
import pandas as pd


def load_graph_adj_mtx(path):
    """加载邻接矩阵。
    返回 A: shape (N_poi, N_poi)，A[i,j] 表示从 i 到 j 的转移频次
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
