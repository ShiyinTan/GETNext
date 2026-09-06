import logging
import os
import pathlib
import pickle
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, maksed_mse_loss, \
    batch_last_step_metrics, format_epoch_summary, write_epoch_metrics_txt, epoch_ckpt_metrics, \
    mean_or_nan

SEP = '-' * 72


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so bars stay intact."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(save_dir, verbose=False):
    """File keeps full detail; console shows only key INFO lines."""
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
    logging.getLogger('matplotlib.font_manager').disabled = True


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    setup_logger(args.save_dir, verbose=args.verbose)

    # Save run settings (full args only in file / yaml)
    logging.debug('Full args: %s', args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    logging.info(SEP)
    logging.info(' GETNext training')
    logging.info(f' save_dir : {args.save_dir}')
    logging.info(f' device   : {args.device}')
    logging.info(f' epochs   : {args.epochs}  batch={args.batch}  lr={args.lr}')
    logging.info(f' train    : {args.data_train}')
    logging.info(f' val      : {args.data_val}')
    if args.eval_test:
        logging.info(f' test     : {args.data_test}  (monitor only; ckpt uses val)')
    logging.info(SEP)

    # %% ====================== Load data ======================
    # 步骤1: 读取签到轨迹 CSV
    logging.info('[1/4] Loading trajectories & POI graph...')
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)
    test_df = None
    if args.eval_test:
        if os.path.isfile(args.data_test):
            test_df = pd.read_csv(args.data_test)
        else:
            logging.warning(f'Test CSV not found ({args.data_test}); skipping in-training test eval')

    # 步骤2: 加载全局轨迹流图 (由 build_graph.py 从 train 构建)
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)  # (N_poi, N_poi)，边权=转移频次
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)  # (N_poi, 4)
    logging.debug(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.debug(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]  # N_poi

    # 步骤3: 对 POI 类别做 One-Hot，拼回节点特征矩阵 X
    logging.debug('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()  # (N_poi, num_cats)
    num_cats = one_hot_rlt.shape[-1]
    # X: [checkin_cnt | cat_onehot | lat | lon]，shape (N_poi, 1+num_cats+2)
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.debug(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.debug(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # 步骤4: 邻接矩阵做 GCN 用的随机游走归一化拉普拉斯
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')  # (N_poi, N_poi)
    logging.info(f'        POIs={num_pois}  cats={num_cats}  '
                 f'train_rows={len(train_df)}  val_rows={len(val_df)}'
                 f'{"" if test_df is None else f"  test_rows={len(test_df)}"}')

    # 步骤5: 构建 id ↔ index 映射字典
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))  # POI_id → [0, N_poi)

    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))  # cat_id → [0, num_cats)

    # poi 图索引 → 类别索引
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))  # user_id → [0, N_user)

    traj_list = list(set(train_df['trajectory_id'].tolist()))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        """将每条轨迹切成 next-step 样本。
        每条样本:
          traj_id: str
          input_seq:  list[(poi_idx, time)], 长度 T=轨迹长度-1
          label_seq:  list[(next_poi_idx, next_time)]，与 input 等长
        """
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist()),
                                desc='Build train set', leave=False, dynamic_ncols=True):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]  # 长度 L
                time_feature = traj_df[args.time_feature].to_list()    # 长度 L

                # 滑动构造: 用第 i 步预测第 i+1 步 → 得到 T=L-1 对
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        """验证/测试集构造同训练集；过滤训练未见过的 user / POI。"""
        def __init__(self, df, split_name='val'):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist()),
                                desc=f'Build {split_name} set', leave=False, dynamic_ncols=True):
                user_id = traj_id.split('_')[0]

                # 跳过训练集中未出现的用户
                if user_id not in user_id2idx_dict.keys():
                    continue

                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # 跳过训练集中未出现的 POI
                        continue

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    logging.info('[2/4] Building dataloaders...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df, split_name='val')
    test_dataset = None
    if test_df is not None:
        test_dataset = TrajectoryDatasetVal(test_df, split_name='test')
    test_n = len(test_dataset) if test_dataset is not None else 0
    logging.info(f'        train_trajs={len(train_dataset)}  val_trajs={len(val_dataset)}  '
                 f'test_trajs={test_n}  users={len(user_id2idx_dict)}')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)
    test_loader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch,
                                 shuffle=False, drop_last=False,
                                 pin_memory=True, num_workers=args.workers,
                                 collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # 步骤6: 图特征/邻接转 Tensor，并构建各子模块
    logging.info('[3/4] Building models...')
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)  # (N_poi, F)
        A = torch.from_numpy(A)  # (N_poi, N_poi)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    # 6.1 GCN: 节点特征+图结构 → POI 嵌入, 输出 (N_poi, poi_embed_dim)
    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)

    # 6.2 NodeAttnMap: 生成 POI→POI 转移打分矩阵 (N_poi, N_poi)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    # 6.3 User Embedding: user_idx → (user_embed_dim,)
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # 6.4 Time2Vec: 时间标量 → (time_embed_dim,)
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # 6.5 Category Embedding: cat_idx → (cat_embed_dim,)
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # 6.6 两路融合: Fuse(user,poi) 与 Fuse(time,cat)
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

    # 6.7 Transformer 序列模型；输入维 = 四路嵌入之和
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)

    # 联合优化全部子模块参数
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # padding 位置 label=-1 忽略
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_time = maksed_mse_loss  # 忽略 target==-1 的时间 MSE

    # Prefer no-verbose scheduler API (PyTorch 2.x); fall back for older installs.
    try:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=args.lr_scheduler_factor)
    except TypeError:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=False, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings):
        """将一条轨迹的每个时间步编码为融合嵌入序列。
        Args:
          sample: (traj_id, input_seq, label_seq)
            input_seq: list[(poi_idx, time)], 长度 T
          poi_embeddings: (N_poi, poi_embed_dim)，来自 GCN(X, A)
        Returns:
          input_seq_embed: list，长度 T；每个元素 shape (D,)
            D = user_embed_dim + poi_embed_dim + time_embed_dim + cat_embed_dim
        """
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]       # (T,) poi_idx
        input_seq_time = [each[1] for each in sample[1]]  # (T,) time
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]  # (T,) cat_idx

        # 用户嵌入 (整条轨迹共享同一 user)
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)  # (1,)
        user_embedding = user_embed_model(input)  # (1, user_embed_dim)
        user_embedding = torch.squeeze(user_embedding)  # (user_embed_dim,)

        # 逐步: POI/Time/Cat 嵌入 → 两路 Fuse → concat
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]  # (poi_embed_dim,)
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)  # (time_embed_dim,)

            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)  # (1,)
            cat_embedding = cat_embed_model(cat_idx)  # (1, cat_embed_dim)
            cat_embedding = torch.squeeze(cat_embedding)  # (cat_embed_dim,)

            # Fuse1: user||poi → (user_dim+poi_dim,)
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            # Fuse2: time||cat → (time_dim+cat_dim,)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

            # 最终逐步嵌入: (D,)
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi, batch_input_seqs):
        """用轨迹流图注意力校正 Transformer 的下一 POI logits。
        Args:
          y_pred_poi: (B, T_max, N_poi)，Transformer 原始 POI 打分
          batch_input_seqs: list[list[poi_idx]]，长度 B
        Returns:
          y_pred_poi_adjusted: (B, T_max, N_poi)
            对每个时间步 j: adjusted[i,j,:] = attn_map[当前POI_j, :] + y_pred_poi[i,j,:]
        """
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)  # (B, T_max, N_poi)
        attn_map = node_attn_model(X, A)  # (N_poi, N_poi)

        for i in range(len(batch_input_seqs)):
            traj_i_input = batch_input_seqs[i]  # list，长度 T_i，元素为当前步 poi_idx
            for j in range(len(traj_i_input)):
                # 取「当前 POI」对应的一行转移先验，加到该步 logits 上
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    def eval_split(loader, split_name, max_batches, epoch):
        """Val/test eval: no backward. Returns the same metric dict as an epoch row."""
        parts = {k: [] for k in (
            'top1', 'top5', 'top10', 'top20', 'ndcg5', 'ndcg10', 'map20', 'mrr',
            'loss', 'poi', 'time', 'cat')}
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        pbar = tqdm(loader,
                    desc=f'Epoch {epoch + 1}/{args.epochs} {split_name:<4}',
                    leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for b_idx, batch in enumerate(pbar):
                if max_batches and b_idx >= max_batches:
                    break
                if len(batch) != args.batch:
                    src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
    
                batch_input_seqs = []
                batch_seq_lens = []
                batch_seq_embeds = []
                batch_seq_labels_poi = []
                batch_seq_labels_time = []
                batch_seq_labels_cat = []
    
                poi_embeddings = poi_embed_model(X, A)
    
                for sample in batch:
                    input_seq = [each[0] for each in sample[1]]
                    label_seq = [each[0] for each in sample[2]]
                    label_seq_time = [each[1] for each in sample[2]]
                    label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                    input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                    batch_seq_embeds.append(input_seq_embed)
                    batch_seq_lens.append(len(input_seq))
                    batch_input_seqs.append(input_seq)
                    batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                    batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                    batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
    
                batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
                label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
                label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
                label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
    
                x = batch_padded.to(device=args.device, dtype=torch.float)
                y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
                y_time = label_padded_time.to(device=args.device, dtype=torch.float)
                y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
                y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)
                y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi, batch_input_seqs)
    
                loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
                loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
                loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
                loss = loss_poi + loss_time * args.time_loss_weight + loss_cat
    
                batch_m = batch_last_step_metrics(
                    y_poi.detach().cpu().numpy(),
                    y_pred_poi_adjusted.detach().cpu().numpy(),
                    batch_seq_lens)
                parts['top1'].append(batch_m['top1'])
                parts['top5'].append(batch_m['top5'])
                parts['top10'].append(batch_m['top10'])
                parts['top20'].append(batch_m['top20'])
                parts['ndcg5'].append(batch_m['ndcg5'])
                parts['ndcg10'].append(batch_m['ndcg10'])
                parts['map20'].append(batch_m['map20'])
                parts['mrr'].append(batch_m['mrr'])
                parts['loss'].append(loss.detach().cpu().numpy())
                parts['poi'].append(loss_poi.detach().cpu().numpy())
                parts['time'].append(loss_time.detach().cpu().numpy())
                parts['cat'].append(loss_cat.detach().cpu().numpy())
                pbar.set_postfix(
                    loss=f'{loss.item():.2f}',
                    avg=f'{float(np.mean(parts["loss"])):.2f}',
                    top1=f'{batch_m["top1"]:.3f}',
                    refresh=False)
    
                if args.verbose and (b_idx % max(args.batch * 2, 1)) == 0:
                    sample_idx = 0
                    batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                    batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
                    batch_pred_times = y_pred_time.detach().cpu().numpy()
                    batch_pred_cats = y_pred_cat.detach().cpu().numpy()
                    logging.debug(
                        f'Epoch:{epoch}, batch:{b_idx}, '
                        f'{split_name}_batch_loss:{loss.item():.2f}, '
                        f'{split_name}_batch_top1_acc:{batch_m["top1"]:.2f}, '
                        f'{split_name}_move_loss:{np.mean(parts["loss"]):.2f} \n'
                        f'{split_name}_move_poi_loss:{np.mean(parts["poi"]):.2f} \n'
                        f'{split_name}_move_time_loss:{np.mean(parts["time"]):.2f} \n'
                        f'{split_name}_move_top1_acc:{np.mean(parts["top1"]):.4f} \n'
                        f'{split_name}_move_top5_acc:{np.mean(parts["top5"]):.4f} \n'
                        f'{split_name}_move_top10_acc:{np.mean(parts["top10"]):.4f} \n'
                        f'{split_name}_move_top20_acc:{np.mean(parts["top20"]):.4f} \n'
                        f'{split_name}_move_mAP20:{np.mean(parts["map20"]):.4f} \n'
                        f'{split_name}_move_MRR:{np.mean(parts["mrr"]):.4f} \n'
                        f'traj_id:{batch[sample_idx][0]}\n'
                        f'input_seq:{batch[sample_idx][1]}\n'
                        f'label_seq:{batch[sample_idx][2]}\n'
                        f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                        f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                        f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                        f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                        f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                        f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])}'
                    )
        return {
            'loss': mean_or_nan(parts['loss']),
            'poi': mean_or_nan(parts['poi']),
            'time': mean_or_nan(parts['time']),
            'cat': mean_or_nan(parts['cat']),
            'top1': mean_or_nan(parts['top1']),
            'top5': mean_or_nan(parts['top5']),
            'top10': mean_or_nan(parts['top10']),
            'top20': mean_or_nan(parts['top20']),
            'ndcg5': mean_or_nan(parts['ndcg5']),
            'ndcg10': mean_or_nan(parts['ndcg10']),
            'map20': mean_or_nan(parts['map20']),
            'mrr': mean_or_nan(parts['mrr']),
        }

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_ndcg5_list = []
    train_epochs_ndcg10_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_ndcg5_list = []
    val_epochs_ndcg10_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    test_hist = []
    # For saving ckpt
    max_val_score = -np.inf

    logging.info('[4/4] Start training...')
    for epoch in range(args.epochs):
        logging.debug(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}")
        poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_ndcg5_list = []
        train_batches_ndcg10_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)  # (B, B)
        # ---------- 训练 batch 循环 ----------
        train_pbar = tqdm(train_loader,
                          desc=f'Epoch {epoch + 1}/{args.epochs} train',
                          leave=False,
                          dynamic_ncols=True)
        for b_idx, batch in enumerate(train_pbar):
            if args.max_batches and b_idx >= args.max_batches:
                break
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # 收集本 batch 各轨迹信息，稍后 pad 成统一长度
            batch_input_seqs = []      # list[list[poi_idx]]，长度 B
            batch_seq_lens = []        # list[int]，每条真实长度 T_i
            batch_seq_embeds = []      # list[Tensor(T_i, D)]
            batch_seq_labels_poi = []  # list[LongTensor(T_i,)]
            batch_seq_labels_time = [] # list[FloatTensor(T_i,)]
            batch_seq_labels_cat = []  # list[LongTensor(T_i,)]

            # 步骤7: GCN 得到全体 POI 嵌入 (N_poi, poi_embed_dim)
            poi_embeddings = poi_embed_model(X, A)

            # 步骤8: 每条轨迹 → 逐步融合嵌入 + 标签
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]          # (T_i,)
                label_seq = [each[0] for each in sample[2]]          # (T_i,) next poi
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]     # (T_i,) next time
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                # stack 后: (T_i, D)
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # 步骤9: pad 到 batch 内最大长度 T_max；padding_value=-1
            # batch_padded: (B, T_max, D)
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)    # (B, T_max)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)  # (B, T_max)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)    # (B, T_max)

            # 步骤10: Transformer 前向 → 三任务预测
            x = batch_padded.to(device=args.device, dtype=torch.float)          # (B, T_max, D)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)   # (B, T_max)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)
            # y_pred_poi: (B, T_max, N_poi), y_pred_time: (B, T_max, 1), y_pred_cat: (B, T_max, num_cats)

            # 步骤11: 轨迹流图注意力校正 POI logits → (B, T_max, N_poi)
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi, batch_input_seqs)

            # 步骤12: 多任务损失
            # CE 需要 (B, C, T)，故对 poi/cat 做 transpose(1,2)
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # total = CE(poi) + λ * MSE(time) + CE(cat)
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # 步骤13: 按「最后时间步」计算 Top-k / mAP / MRR
            batch_label_pois = y_poi.detach().cpu().numpy()              # (B, T_max)
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy() # (B, T_max, N_poi)
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            batch_m = batch_last_step_metrics(batch_label_pois, batch_pred_pois, batch_seq_lens)
            batch_top1 = batch_m['top1']
            train_batches_top1_acc_list.append(batch_m['top1'])
            train_batches_top5_acc_list.append(batch_m['top5'])
            train_batches_top10_acc_list.append(batch_m['top10'])
            train_batches_top20_acc_list.append(batch_m['top20'])
            train_batches_ndcg5_list.append(batch_m['ndcg5'])
            train_batches_ndcg10_list.append(batch_m['ndcg10'])
            train_batches_mAP20_list.append(batch_m['map20'])
            train_batches_mrr_list.append(batch_m['mrr'])
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            train_pbar.set_postfix(
                loss=f'{loss.item():.2f}',
                avg=f'{float(np.mean(train_batches_loss_list)):.2f}',
                top1=f'{batch_top1:.3f}',
                refresh=False)

            # Optional detailed sample dump (file + console when --verbose)
            if args.verbose and (b_idx % max(args.batch * 5, 1)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.debug(
                    f'Epoch:{epoch}, batch:{b_idx}, '
                    f'train_batch_loss:{loss.item():.2f}, '
                    f'train_batch_top1_acc:{batch_top1:.2f}, '
                    f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                    f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                    f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                    f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                    f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                    f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                    f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                    f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                    f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                    f'traj_id:{batch[sample_idx][0]}\n'
                    f'input_seq: {batch[sample_idx][1]}\n'
                    f'label_seq:{batch[sample_idx][2]}\n'
                    f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                    f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                    f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                    f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                    f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                    f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])}'
                )

        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        # Val + optional test (test is monitor-only; checkpoint still uses val)
        val_m = eval_split(val_loader, 'val', args.max_val_batches, epoch)
        test_m = None
        if test_loader is not None:
            test_m = eval_split(test_loader, 'test', args.max_test_batches, epoch)

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_ndcg5 = np.mean(train_batches_ndcg5_list)
        epoch_train_ndcg10 = np.mean(train_batches_ndcg10_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_ndcg5_list.append(epoch_train_ndcg5)
        train_epochs_ndcg10_list.append(epoch_train_ndcg10)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(val_m['loss'])
        val_epochs_poi_loss_list.append(val_m['poi'])
        val_epochs_time_loss_list.append(val_m['time'])
        val_epochs_cat_loss_list.append(val_m['cat'])
        val_epochs_top1_acc_list.append(val_m['top1'])
        val_epochs_top5_acc_list.append(val_m['top5'])
        val_epochs_top10_acc_list.append(val_m['top10'])
        val_epochs_top20_acc_list.append(val_m['top20'])
        val_epochs_ndcg5_list.append(val_m['ndcg5'])
        val_epochs_ndcg10_list.append(val_m['ndcg10'])
        val_epochs_mAP20_list.append(val_m['map20'])
        val_epochs_mrr_list.append(val_m['mrr'])
        if test_m is not None:
            test_hist.append(test_m)

        # 用验证损失调度学习率；用 top1/top20 组合分选 best ckpt（不用 test）
        monitor_loss = val_m['loss']
        monitor_score = np.mean(val_m['top1'] * 4 + val_m['top20'])

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)
        current_lr = optimizer.param_groups[0]['lr']

        train_m = {
            'loss': epoch_train_loss, 'poi': epoch_train_poi_loss,
            'time': epoch_train_time_loss, 'cat': epoch_train_cat_loss,
            'top1': epoch_train_top1_acc, 'top5': epoch_train_top5_acc,
            'top10': epoch_train_top10_acc, 'top20': epoch_train_top20_acc,
            'ndcg5': epoch_train_ndcg5, 'ndcg10': epoch_train_ndcg10,
            'map20': epoch_train_mAP20, 'mrr': epoch_train_mrr,
        }

        # Save poi and user embeddings
        saved_best = False
        if args.save_embeds:
            embeddings_save_dir = os.path.join(args.save_dir, 'embeddings')
            if not os.path.exists(embeddings_save_dir): os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score:
                # Save poi embeddings
                poi_embeddings = poi_embed_model(X, A).detach().cpu().numpy()
                poi_embedding_list = []
                for poi_idx in range(len(poi_id2idx_dict)):
                    poi_embedding = poi_embeddings[poi_idx]
                    poi_embedding_list.append(poi_embedding)
                save_poi_embeddings = np.array(poi_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_poi_embeddings'), save_poi_embeddings)
                # Save user embeddings
                user_embedding_list = []
                for user_idx in range(len(user_id2idx_dict)):
                    input = torch.LongTensor([user_idx]).to(device=args.device)
                    user_embedding = user_embed_model(input).detach().cpu().numpy().flatten()
                    user_embedding_list.append(user_embedding)
                user_embeddings = np.array(user_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_user_embeddings'), user_embeddings)
                # Save cat embeddings
                cat_embedding_list = []
                for cat_idx in range(len(cat_id2idx_dict)):
                    input = torch.LongTensor([cat_idx]).to(device=args.device)
                    cat_embedding = cat_embed_model(input).detach().cpu().numpy().flatten()
                    cat_embedding_list.append(cat_embedding)
                cat_embeddings = np.array(cat_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_cat_embeddings'), cat_embeddings)
                # Save time embeddings
                time_embedding_list = []
                for time_idx in range(args.time_units):
                    input = torch.FloatTensor([time_idx]).to(device=args.device)
                    time_embedding = time_embed_model(input).detach().cpu().numpy().flatten()
                    time_embedding_list.append(time_embedding)
                time_embeddings = np.array(time_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_time_embeddings'), time_embeddings)

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
                'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                'node_attn_map': node_attn_model(X, A),
                'args': args,
                'epoch_train_metrics': epoch_ckpt_metrics('train', train_m),
                'epoch_val_metrics': epoch_ckpt_metrics('val', val_m),
            }
            if test_m is not None:
                state_dict['epoch_test_metrics'] = epoch_ckpt_metrics('test', test_m)
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    dump = dict(state_dict['epoch_val_metrics'])
                    if test_m is not None:
                        dump.update(state_dict['epoch_test_metrics'])
                    print(dump, file=f)
                max_val_score = monitor_score
                saved_best = True

        logging.info(format_epoch_summary(
            epoch, args.epochs, current_lr, train_m, val_m,
            saved_best=saved_best, best_score=max_val_score if saved_best else None,
            test_m=test_m))

        # Save train/val/test metrics for plotting purpose
        write_epoch_metrics_txt(
            os.path.join(args.save_dir, 'metrics-train.txt'), 'train',
            [{'loss': l, 'poi': p, 'time': t, 'cat': c,
              'top1': a1, 'top5': a5, 'top10': a10, 'top20': a20,
              'ndcg5': n5, 'ndcg10': n10, 'map20': mp, 'mrr': mr}
             for l, p, t, c, a1, a5, a10, a20, n5, n10, mp, mr in zip(
                 train_epochs_loss_list, train_epochs_poi_loss_list,
                 train_epochs_time_loss_list, train_epochs_cat_loss_list,
                 train_epochs_top1_acc_list, train_epochs_top5_acc_list,
                 train_epochs_top10_acc_list, train_epochs_top20_acc_list,
                 train_epochs_ndcg5_list, train_epochs_ndcg10_list,
                 train_epochs_mAP20_list, train_epochs_mrr_list)])
        write_epoch_metrics_txt(
            os.path.join(args.save_dir, 'metrics-val.txt'), 'val',
            [{'loss': l, 'poi': p, 'time': t, 'cat': c,
              'top1': a1, 'top5': a5, 'top10': a10, 'top20': a20,
              'ndcg5': n5, 'ndcg10': n10, 'map20': mp, 'mrr': mr}
             for l, p, t, c, a1, a5, a10, a20, n5, n10, mp, mr in zip(
                 val_epochs_loss_list, val_epochs_poi_loss_list,
                 val_epochs_time_loss_list, val_epochs_cat_loss_list,
                 val_epochs_top1_acc_list, val_epochs_top5_acc_list,
                 val_epochs_top10_acc_list, val_epochs_top20_acc_list,
                 val_epochs_ndcg5_list, val_epochs_ndcg10_list,
                 val_epochs_mAP20_list, val_epochs_mrr_list)])
        if test_hist:
            write_epoch_metrics_txt(
                os.path.join(args.save_dir, 'metrics-test.txt'), 'test', test_hist)

    logging.info(f'Training finished. Best val score={max_val_score:.4f}')
    logging.info(f'Checkpoints: {os.path.join(args.save_dir, "checkpoints")}')


if __name__ == '__main__':
    # Keep console focused on training progress (model warnings go to log file only).
    warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
    warnings.filterwarnings('ignore', message='.*verbose parameter is deprecated.*')
    args = parameter_parser()
    if args.no_cuda or not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)
