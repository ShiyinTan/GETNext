import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NodeAttnMap(nn.Module):
    """基于节点特征与邻接结构，生成 POI→POI 转移注意力打分矩阵 (类似 GAT)。
    用于在解码阶段校正 Transformer 的下一 POI logits。
    """
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))  # (F, H)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))  # (2H, 1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        # X: (N_poi, F), A: (N_poi, N_poi)
        Wh = torch.mm(X, self.W)  # (N_poi, H)

        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) → (N_poi, N_poi)
        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # 无边位置置 0

        A = A + 1  # 邻接权从约 0~1 平移到 1~2，避免乘零抹掉分数
        e = e * A  # 用图结构调制注意力，仍为 (N_poi, N_poi)

        return e  # attn_map[i, j]: 从 POI_i 到 POI_j 的转移打分

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh: (N_poi, H)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])   # (N_poi, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])   # (N_poi, 1)
        e = Wh1 + Wh2.T  # 广播相加 → (N_poi, N_poi)
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    """单层图卷积: output = A · (X W) [+ bias]
    input: (N, in_features), adj: (N, N) → output: (N, out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # (Fin, Fout)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # (Fout,)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: (N, Fin), adj: (N, N)
        support = torch.mm(input, self.weight)  # (N, Fout)
        output = torch.spmm(adj, support)       # (N, Fout)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """多层 GCN，将节点特征+邻接结构编码为 POI 嵌入。
    默认通道: ninput → gcn_nhid[0] → ... → noutput(=poi_embed_dim)
    输入 x:(N_poi, F), adj:(N_poi, N_poi) → 输出:(N_poi, poi_embed_dim)
    """
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        # 前 L-1 层: GCN + LeakyReLU
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        # 最后一层: Dropout + GCN
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)  # (N_poi, noutput)

        return x


class UserEmbeddings(nn.Module):
    """用户 ID → 嵌入查表。
    输入 user_idx: (B,) 或 (1,) → 输出: (B, user_embed_dim)
    """
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    """POI 类别 ID → 嵌入查表。
    输入 cat_idx: (B,) → 输出: (B, cat_embed_dim)
    """
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    """拼接两路嵌入后经 Linear+LeakyReLU 融合。
    输入各为 1D: (d1,), (d2,) → concat:(d1+d2,) → 输出:(d1+d2,)
    """
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        # 注意: 此处按 dim=0 拼接，约定输入为 1D 向量
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """Time2Vec 核心: [周期性激活映射 | 线性项] 拼接。
    tau: (B, 1) → 返回 (B, out_features)
    """
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)  # (B, out_features-1)
    v2 = torch.matmul(tau, w0) + b0       # (B, 1)
    return torch.cat([v1, v2], 1)         # (B, out_features)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    """将标量时间特征映射为向量。
    输入 x: (B, 1) 或可广播的时间标量 → 输出: (B, out_dim)
    """
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。
    输入 x: (S, B, d_model) → 输出同 shape (S, B, d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (S, B, d_model)；将前 S 个位置编码加到序列维上
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """轨迹序列编码器 + 三任务解码头 (下一 POI / 时间 / 类别)。
    输入 src 约定与 nn.TransformerEncoder 一致: (S, B, embed_size)
    输出:
      out_poi:  (S, B, num_poi)
      out_time: (S, B, 1)
      out_cat:  (S, B, num_cat)
    注: train.py 中 pad 后为 (B, T, D)，按代码原样送入；索引时按 (B, T, *) 使用。
    """
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)   # → num_poi logits
        self.decoder_time = nn.Linear(embed_size, 1)       # → 时间回归
        self.decoder_cat = nn.Linear(embed_size, num_cat)   # → 类别 logits
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        # 下三角因果 mask，shape (sz, sz)；可见位置为 0，不可见为 -inf
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        # src: (S, B, E) 或训练侧传入的 (B, T, E)；src_mask: (S, S)
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)  # 同 src 的前两维
        out_poi = self.decoder_poi(x)    # (..., num_poi)
        out_time = self.decoder_time(x)  # (..., 1)
        out_cat = self.decoder_cat(x)    # (..., num_cat)
        return out_poi, out_time, out_cat
