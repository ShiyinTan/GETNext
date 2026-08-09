"""Load a trained GETNext checkpoint and run next-POI prediction on a split."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import (
    CategoryEmbeddings,
    FuseEmbeddings,
    GCN,
    NodeAttnMap,
    Time2Vec,
    TransformerModel,
    UserEmbeddings,
)
from utils import (
    MRR_metric_last_timestep,
    calculate_laplacian_matrix,
    mAP_metric_last_timestep,
    top_k_acc_last_timestep,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GETNext prediction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_epoch.state.pt")
    parser.add_argument("--data-test", type=str, default="dataset/NYC/NYC_test.csv")
    parser.add_argument("--data-adj-mtx", type=str, default="dataset/NYC/graph_A.csv")
    parser.add_argument("--data-node-feats", type=str, default="dataset/NYC/graph_X.csv")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--short-traj-thres", type=int, default=2)
    parser.add_argument("--time-feature", type=str, default="norm_in_day_time")
    parser.add_argument("--feature1", type=str, default="checkin_cnt")
    parser.add_argument("--feature2", type=str, default="poi_catid")
    parser.add_argument("--feature3", type=str, default="latitude")
    parser.add_argument("--feature4", type=str, default="longitude")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device("cpu" if cli.no_cuda or not torch.cuda.is_available() else "cuda")

    ckpt = torch.load(cli.checkpoint, map_location=device)
    args = ckpt["args"]
    args.device = device
    args.no_cuda = cli.no_cuda

    user_id2idx_dict = ckpt["user_id2idx_dict"]
    poi_id2idx_dict = ckpt["poi_id2idx_dict"]
    cat_id2idx_dict = ckpt["cat_id2idx_dict"]
    poi_idx2cat_idx_dict = ckpt["poi_idx2cat_idx_dict"]

    raw_A = load_graph_adj_mtx(cli.data_adj_mtx)
    raw_X = load_graph_node_features(
        cli.data_node_feats, cli.feature1, cli.feature2, cli.feature3, cli.feature4
    )
    num_pois = raw_X.shape[0]

    one_hot_path = Path(cli.checkpoint).resolve().parents[1] / "one-hot-encoder.pkl"
    if one_hot_path.exists():
        with open(one_hot_path, "rb") as f:
            one_hot_encoder = pickle.load(f)
        one_hot_rlt = one_hot_encoder.transform([[x] for x in raw_X[:, 1]]).toarray()
    else:
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit([[x] for x in raw_X[:, 1]])
        one_hot_rlt = one_hot_encoder.transform([[x] for x in raw_X[:, 1]]).toarray()

    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1 : num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1 :] = raw_X[:, 2:]
    A = calculate_laplacian_matrix(raw_A, mat_type="hat_rw_normd_lap_mat")

    test_df = pd.read_csv(cli.data_test)

    class TrajectoryDataset(Dataset):
        def __init__(self, df):
            self.traj_seqs, self.input_seqs, self.label_seqs = [], [], []
            for traj_id in tqdm(set(df["trajectory_id"].tolist()), desc="Build dataset"):
                user_id = traj_id.split("_")[0]
                if user_id not in user_id2idx_dict:
                    continue
                traj_df = df[df["trajectory_id"] == traj_id]
                poi_idxs = []
                for poi_id in traj_df["POI_id"].tolist():
                    if poi_id in poi_id2idx_dict:
                        poi_idxs.append(poi_id2idx_dict[poi_id])
                time_feature = traj_df[cli.time_feature].tolist()
                input_seq, label_seq = [], []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))
                if len(input_seq) < cli.short_traj_thres:
                    continue
                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index]

    dataset = TrajectoryDataset(test_df)
    loader = DataLoader(
        dataset,
        batch_size=cli.batch,
        shuffle=False,
        drop_last=False,
        num_workers=cli.workers,
        collate_fn=lambda x: x,
    )

    X = torch.from_numpy(X).to(device=device, dtype=torch.float)
    A = torch.from_numpy(A).to(device=device, dtype=torch.float)

    poi_embed_model = GCN(
        ninput=X.shape[1],
        nhid=args.gcn_nhid,
        noutput=args.poi_embed_dim,
        dropout=args.gcn_dropout,
    )
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    user_embed_model = UserEmbeddings(len(user_id2idx_dict), args.user_embed_dim)
    time_embed_model = Time2Vec("sin", out_dim=args.time_embed_dim)
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)
    seq_input_embed = (
        args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    )
    seq_model = TransformerModel(
        num_pois,
        num_cats,
        seq_input_embed,
        args.transformer_nhead,
        args.transformer_nhid,
        args.transformer_nlayers,
        dropout=args.transformer_dropout,
    )

    poi_embed_model.load_state_dict(ckpt["poi_embed_state_dict"])
    node_attn_model.load_state_dict(ckpt["node_attn_state_dict"])
    user_embed_model.load_state_dict(ckpt["user_embed_state_dict"])
    time_embed_model.load_state_dict(ckpt["time_embed_state_dict"])
    cat_embed_model.load_state_dict(ckpt["cat_embed_state_dict"])
    embed_fuse_model1.load_state_dict(ckpt["embed_fuse1_state_dict"])
    embed_fuse_model2.load_state_dict(ckpt["embed_fuse2_state_dict"])
    seq_model.load_state_dict(ckpt["seq_model_state_dict"])

    models = [
        poi_embed_model,
        node_attn_model,
        user_embed_model,
        time_embed_model,
        cat_embed_model,
        embed_fuse_model1,
        embed_fuse_model2,
        seq_model,
    ]
    for m in models:
        m.to(device)
        m.eval()

    idx2poi_id = {v: k for k, v in poi_id2idx_dict.items()}

    def input_traj_to_embeddings(sample, poi_embeddings):
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]
        user_id = traj_id.split("_")[0]
        user_idx = user_id2idx_dict[user_id]
        user_embedding = torch.squeeze(
            user_embed_model(torch.LongTensor([user_idx]).to(device))
        )
        embeds = []
        for idx in range(len(input_seq)):
            poi_embedding = torch.squeeze(poi_embeddings[input_seq[idx]]).to(device)
            time_embedding = torch.squeeze(
                time_embed_model(
                    torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device)
                )
            )
            cat_embedding = torch.squeeze(
                cat_embed_model(torch.LongTensor([input_seq_cat[idx]]).to(device))
            )
            fused1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused2 = embed_fuse_model2(time_embedding, cat_embedding)
            embeds.append(torch.cat((fused1, fused2), dim=-1))
        return embeds

    def adjust_pred_prob_by_graph(y_pred_poi, batch_input_seqs, batch_seq_lens):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)
        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]
        return y_pred_poi_adjusted

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
    metrics = {
        "top1": [],
        "top5": [],
        "top10": [],
        "top20": [],
        "mAP20": [],
        "mrr": [],
        "poi_loss": [],
    }
    predictions = []

    with torch.no_grad():
        src_mask = seq_model.generate_square_subsequent_mask(cli.batch).to(device)
        for batch in tqdm(loader, desc="Predict"):
            if len(batch) != cli.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(device)

            batch_input_seqs, batch_seq_lens = [], []
            batch_seq_embeds, batch_seq_labels_poi = [], []
            poi_embeddings = poi_embed_model(X, A)

            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                batch_seq_embeds.append(torch.stack(input_traj_to_embeddings(sample, poi_embeddings)))
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))

            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            x = batch_padded.to(device=device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=device, dtype=torch.long)
            y_pred_poi, _, _ = seq_model(x, src_mask)
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(
                y_pred_poi, batch_input_seqs, batch_seq_lens
            )
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            metrics["poi_loss"].append(float(loss_poi.detach().cpu()))

            batch_label = y_poi.detach().cpu().numpy()
            batch_pred = y_pred_poi_adjusted.detach().cpu().numpy()
            for sample, label_pois, pred_pois, seq_len in zip(
                batch, batch_label, batch_pred, batch_seq_lens
            ):
                label_pois = label_pois[:seq_len]
                pred_pois = pred_pois[:seq_len, :]
                metrics["top1"].append(top_k_acc_last_timestep(label_pois, pred_pois, k=1))
                metrics["top5"].append(top_k_acc_last_timestep(label_pois, pred_pois, k=5))
                metrics["top10"].append(top_k_acc_last_timestep(label_pois, pred_pois, k=10))
                metrics["top20"].append(top_k_acc_last_timestep(label_pois, pred_pois, k=20))
                metrics["mAP20"].append(mAP_metric_last_timestep(label_pois, pred_pois, k=20))
                metrics["mrr"].append(MRR_metric_last_timestep(label_pois, pred_pois))

                last_label = int(label_pois[-1])
                last_pred = pred_pois[-1]
                topk_idx = np.argsort(-last_pred)[: cli.top_k]
                predictions.append(
                    {
                        "traj_id": sample[0],
                        "label_poi_idx": last_label,
                        "label_poi_id": idx2poi_id.get(last_label),
                        "pred_topk_poi_idx": topk_idx.tolist(),
                        "pred_topk_poi_id": [idx2poi_id.get(int(i)) for i in topk_idx],
                        "pred_topk_score": last_pred[topk_idx].tolist(),
                    }
                )

    summary = {
        "num_trajectories": len(dataset),
        "checkpoint": cli.checkpoint,
        "data_test": cli.data_test,
        "device": str(device),
        "epoch": ckpt.get("epoch"),
        "poi_loss": float(np.mean(metrics["poi_loss"])) if metrics["poi_loss"] else None,
        "top1_acc": float(np.mean(metrics["top1"])) if metrics["top1"] else None,
        "top5_acc": float(np.mean(metrics["top5"])) if metrics["top5"] else None,
        "top10_acc": float(np.mean(metrics["top10"])) if metrics["top10"] else None,
        "top20_acc": float(np.mean(metrics["top20"])) if metrics["top20"] else None,
        "mAP20": float(np.mean(metrics["mAP20"])) if metrics["mAP20"] else None,
        "mrr": float(np.mean(metrics["mrr"])) if metrics["mrr"] else None,
    }

    out_dir = cli.output_dir or str(Path(cli.checkpoint).resolve().parents[1] / "predictions")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics to {out_dir}/metrics.json")
    print(f"Wrote predictions to {out_dir}/predictions.jsonl")


if __name__ == "__main__":
    main()
