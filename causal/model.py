"""附录 D.3–D.4 的网络：编码器 → 拆成 (h_z, h_c) → 兴趣分 + 混杂分。

对照 GETNext，这里按因果规格改掉的地方（冲突以附录 D 为准）：
  1. 地点向量 e_p 用普通查表 nn.Embedding，不用 GCN（附录 A / D.8：
     GCN 会把热度、转移次数再灌进表征，和混杂通道重复计算）
  2. 不再把 NodeAttnMap 加到 POI 分数上（那也是近邻先验）
  3. 混杂 C 不拼进 Transformer 的输入 token（D.3.4，避免泄漏）
  4. 排序用 s = s_pref + s_conf，而不是一个混在一起的 CE 头

Time2Vec / 用户嵌入 / 类别嵌入仍从原来的 model.py 借用，只读不改。

数据形状约定（和 train.py 一致）：
  B = batch 里有几条轨迹
  T = 补齐后的时间步长度
  N = 地点词表大小（NYC 大约 5000）
"""
import math

import torch
import torch.nn as nn

from model import CategoryEmbeddings, Time2Vec, UserEmbeddings


class GradientReversalFn(torch.autograd.Function):
    """梯度反转层 GRL（附录 D.4.4）。

    前向：原样输出 h_z。
    反向：把流回来的梯度乘 -λ。

    效果：上面的「用 h_z 去猜混杂 C」的分类器照常学；
    但编码器会被推着让 h_z 变得「猜不准 C」——也就是兴趣向量少带近/热/区。
    一次反向就能同时更新两边，不必写两套 optimizer。
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    """给 h_z 套上 GRL 后再送给对抗分类器。"""
    return GradientReversalFn.apply(x, lambd)


class PositionalEncoding(nn.Module):
    """给序列每个位置加上「第几步」的正弦编码。输入输出都是 (B, T, d)。"""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)，随模型保存但不训练

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FuseEmbeddings(nn.Module):
    """GETNext 同款融合：两段向量拼起来，过一层 Linear + LeakyReLU。

    和原版的差别：这里按最后一维拼接，所以一次能处理整个 (B, T, *) batch，
    不用像原 train.py 那样逐步用 Python 循环。
    """

    def __init__(self, dim_a, dim_b):
        super().__init__()
        embed_dim = dim_a + dim_b
        self.fuse = nn.Linear(embed_dim, embed_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, a, b):
        return self.act(self.fuse(torch.cat((a, b), dim=-1)))


class CausalNextPOI(nn.Module):
    """整网四块（附录 D.3 的示意图）：

      轨迹 H ──► 编码器 E ──► h ──► 拆分 S ──► (h_z, h_c)
                                              │
                     每个候选地点 p ──────────┼──► 解码 D ──► 分数 s(p)
                                              │
                                              └──► 混杂头（训练时用，帮 h_c 学会 C）
    """

    def __init__(self, args, num_pois, num_users, num_cats, table):
        super().__init__()
        self.num_pois = num_pois
        self.poi_embed_dim = args.poi_embed_dim
        # 兴趣向量维度必须等于 e_p，才能做点积 s_pref = <h_z, e_p>
        self.hz_dim = args.hz_dim or args.poi_embed_dim
        self.hc_dim = args.hc_dim
        if self.hz_dim != args.poi_embed_dim:
            raise ValueError('hz_dim must equal poi-embed-dim so s_pref=<h_z, e_p> is well-defined')

        # Transformer 输入维 = 四路嵌入拼起来（和 GETNext 一样，只是后面拆分不同）
        d_model = (args.poi_embed_dim + args.user_embed_dim
                   + args.time_embed_dim + args.cat_embed_dim)
        self.d_model = d_model

        # ---- 输入嵌入：地点查表同时当作解码用的 e_p（tied embedding）----
        self.poi_embedding = nn.Embedding(num_pois, args.poi_embed_dim)
        self.user_embedding = UserEmbeddings(num_users, args.user_embed_dim)
        self.time_encoder = Time2Vec('sin', out_dim=args.time_embed_dim)
        self.cat_embedding = CategoryEmbeddings(num_cats, args.cat_embed_dim)
        self.fuse_up = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)   # 用户+地点
        self.fuse_tc = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)   # 时间+类别

        self.pos_encoder = PositionalEncoding(d_model, dropout=args.transformer_dropout)
        # 不用 batch_first，这样 GETNext 的 GPU 环境（torch 1.7）和云端 CPU（2.x）都能跑
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=args.transformer_nhead,
            dim_feedforward=args.transformer_nhid,
            dropout=args.transformer_dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=args.transformer_nlayers)

        # ---- D.3.2 表征拆分：一份变兴趣，一份变混杂（软拆分，不是强制垂直）----
        self.split_z = nn.Sequential(nn.Linear(d_model, self.hz_dim), nn.LeakyReLU(0.2))
        self.split_c = nn.Sequential(nn.Linear(d_model, self.hc_dim), nn.LeakyReLU(0.2))

        # ---- D.3.3 混杂通道：距离 / 热度 / 区域 三个查表分，再加 h_c 与地点的匹配 ----
        self.g_acc = nn.Embedding(table.num_acc_bins, 1)   # 每个距离桶一个标量分
        self.g_pop = nn.Linear(1, 1)                       # log 热度 → 标量分
        self.area_emb = nn.Embedding(table.num_areas, 16)  # 区域向量
        self.g_area = nn.Linear(1, 1)                      # 起点·终点区域相似度 → 标量
        # <W_c h_c, ψ(p)>：查表覆盖不到的情境混杂（例如时段），可选但规格里有
        self.psi = nn.Embedding(num_pois, self.hc_dim)
        self.W_c = nn.Linear(self.hc_dim, self.hc_dim, bias=False)

        # ---- D.4.4 辅助头：对抗看 h_z，重建看 h_c；四个离散 C 各一个分类器 ----
        self.adv_acc = nn.Linear(self.hz_dim, table.num_acc_bins)
        self.adv_pop = nn.Linear(self.hz_dim, table.num_pop_bins)
        self.adv_area = nn.Linear(self.hz_dim, table.num_areas)
        self.adv_hour = nn.Linear(self.hz_dim, table.num_hour_bins)
        self.recon_acc = nn.Linear(self.hc_dim, table.num_acc_bins)
        self.recon_pop = nn.Linear(self.hc_dim, table.num_pop_bins)
        self.recon_area = nn.Linear(self.hc_dim, table.num_areas)
        self.recon_hour = nn.Linear(self.hc_dim, table.num_hour_bins)

        # 可选：从 h_z 预测下一站类别（更贴近兴趣 Z）；时间回归默认关掉
        self.cat_head = nn.Linear(self.hz_dim, num_cats)
        self.time_head = nn.Linear(d_model, 1)

        # 混杂头从 0 起步，先让兴趣通道有机会学，再由 L_conf 把近/热推进 s_conf
        nn.init.zeros_(self.g_acc.weight)
        nn.init.zeros_(self.g_pop.weight)
        nn.init.zeros_(self.g_pop.bias)
        nn.init.zeros_(self.g_area.weight)
        nn.init.zeros_(self.g_area.bias)

    def _token_embed(self, poi_idx, time_feat, cat_idx, user_idx):
        """把一步的 (地点, 时间, 类别, 用户) 融合成 Transformer 的一个 token。

        poi/time/cat: (B, T)   user: (B,) 整条轨迹共用一个用户
        返回 fused: (B, T, d_model)
        注意：这里没有把距离、热度拼进去（D.3.4）。
        """
        bsz, seqlen = poi_idx.size()
        poi_e = self.poi_embedding(poi_idx.clamp(min=0))  # padding 下标先夹成 0，后面用 mask 忽略
        user_e = self.user_embedding(user_idx).unsqueeze(1).expand(bsz, seqlen, -1)
        time_flat = time_feat.reshape(-1, 1)
        time_e = self.time_encoder(time_flat).view(bsz, seqlen, -1)
        cat_e = self.cat_embedding(cat_idx.clamp(min=0))
        fused = torch.cat((self.fuse_up(user_e, poi_e), self.fuse_tc(time_e, cat_e)), dim=-1)
        return fused

    def encode(self, poi_idx, time_feat, cat_idx, user_idx, pad_mask):
        """编码器 E + 拆分 S。pad_mask 里 True 表示补齐位置，不参加注意力。

        返回:
          h   : (B, T, d)  还没拆开的上下文
          h_z : (B, T, d_z) 兴趣代理
          h_c : (B, T, d_c) 混杂摘要
        """
        src = self._token_embed(poi_idx, time_feat, cat_idx, user_idx)  # (B, T, d)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        seqlen = src.size(1)
        # 上三角 -inf：第 t 步只能看见 1..t，和 GETNext「每一步预测下一步」一致
        causal = torch.triu(
            torch.ones(seqlen, seqlen, device=src.device, dtype=src.dtype) * float('-inf'),
            diagonal=1,
        )
        # PyTorch Transformer 默认序列维在最前：(T, B, d)
        h = self.encoder(src.transpose(0, 1), mask=causal, src_key_padding_mask=pad_mask)
        h = h.transpose(0, 1)
        h_z = self.split_z(h)
        h_c = self.split_c(h)
        return h, h_z, h_c

    def _s_pref(self, h_z):
        """兴趣通道：s_pref(p) = h_z 和地点向量 e_p 的点积。输出 (B, T, N)。"""
        e_p = self.poi_embedding.weight  # (N, d_z)，和输入嵌入绑在一起
        return torch.matmul(h_z, e_p.transpose(0, 1))

    def _gather_origin_tables(self, origin_idx, buffers):
        """按当前起点 p_T 取出「到每一个候选点」的距离桶 / 公里数 / 起点区域。"""
        safe = origin_idx.clamp(min=0)
        dist_bin = buffers['dist_bin'][safe]          # (B, T, N)
        dist_km = buffers['dist_km'][safe]
        origin_area = buffers['area_id'][safe]        # (B, T)
        return dist_bin, dist_km, origin_area

    def _s_conf_from_phi(self, h_c, dist_bin, origin_area, buffers):
        """混杂通道：s_conf = 距离分 + 热度分 + 区域分 + h_c 匹配分。"""
        s_acc = self.g_acc(dist_bin.clamp(min=0)).squeeze(-1)             # (B, T, N)
        s_pop = self.g_pop(buffers['log_pop'].unsqueeze(-1)).squeeze(-1)  # (N,)，与轨迹无关
        dest_area_e = self.area_emb(buffers['area_id'])                   # (N, 16)
        origin_area_e = self.area_emb(origin_area.clamp(min=0))           # (B, T, 16)
        # 起点区域向量 · 每个终点区域向量 → 同区更高
        area_match = torch.matmul(origin_area_e, dest_area_e.transpose(0, 1))
        s_area = self.g_area(area_match.unsqueeze(-1)).squeeze(-1)
        ctx = torch.matmul(self.W_c(h_c), self.psi.weight.transpose(0, 1))
        s_pop_b = s_pop.view(1, 1, -1)
        return s_acc + s_pop_b + s_area + ctx, {
            's_acc': s_acc,
            's_pop': s_pop_b,
            's_area': s_area,
            's_ctx': ctx,
        }

    def score(self, h_z, h_c, origin_idx, buffers, mode='factual',
              bar_acc_bin=None, bar_pop_log=None):
        """按推理模式给出总分 / 兴趣分 / 混杂分（附录 D.5）。

        返回 (s, s_pref, s_conf, dist_km)，前三个形状都是 (B, T, N)。

        mode 含义（人话）：
          factual     用真实距离和热度，回答「现实约束下下一站会去哪」
          deconf_pref 只用兴趣分，回答「若远近热度都不管，兴趣指向谁」（规格首选）
          deconf_do   把所有候选的距离桶、热度换成同一个干预值 c_bar
          deconf_sum  对近/热做边缘化：排序时去掉会随候选变化的距离/热度头
        """
        s_pref = self._s_pref(h_z)
        dist_bin, dist_km, origin_area = self._gather_origin_tables(origin_idx, buffers)

        if mode == 'deconf_pref':
            s_conf = torch.zeros_like(s_pref)
            return s_pref, s_pref, s_conf, dist_km

        if mode == 'deconf_do':
            # do(C=c_bar)：每个候选都用同一个距离桶，热度换成常数
            # 这样「更近 / 更热」不再能拉开名次
            if bar_acc_bin is None:
                bar_acc_bin = 0
            dist_bin = torch.full_like(dist_bin, int(bar_acc_bin))
            s_conf, parts = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
            if bar_pop_log is None:
                pop_const = self.g_pop(buffers['log_pop'].mean().view(1, 1)).view(1, 1, 1)
            else:
                pop_const = self.g_pop(
                    torch.tensor([[float(bar_pop_log)]], device=h_z.device, dtype=h_z.dtype)
                ).view(1, 1, 1)
            s_conf = s_conf - parts['s_pop'] + pop_const
            return s_pref + s_conf, s_pref, s_conf, dist_km

        if mode == 'deconf_sum':
            # 若给所有候选同一个 g_acc / 平均 g_pop，名次不变，等价于丢掉这两项
            _, parts = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
            s_conf = parts['s_area'] + parts['s_ctx']
            return s_pref + s_conf, s_pref, s_conf, dist_km

        # factual：真实 C(p)
        s_conf, _ = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
        return s_pref + s_conf, s_pref, s_conf, dist_km

    def g_tilde(self, origin_idx, buffers, alpha, beta):
        """手工先验混杂分 g̃（附录 D.4.3，训练时 stop-grad）。

        越近越高（-α × 公里），越热越高（+β × log热度），同区域再加一点。
        用来把「近/热/同区」从兴趣通道里挤到 s_conf。
        """
        dist_km = buffers['dist_km'][origin_idx.clamp(min=0)]
        log_pop = buffers['log_pop'].view(1, 1, -1)
        origin_area = buffers['area_id'][origin_idx.clamp(min=0)]
        dest_area = buffers['area_id'].view(1, 1, -1)
        same_area = (origin_area.unsqueeze(-1) == dest_area).float()
        return (-alpha * dist_km) + beta * log_pop + 0.15 * same_area

    def adv_logits(self, h_z, lambd):
        """用 GRL(h_z) 去猜 C 的四个桶。分类器想猜对，编码器被反转梯度逼着猜不对。"""
        z = grad_reverse(h_z, lambd)
        return self.adv_acc(z), self.adv_pop(z), self.adv_area(z), self.adv_hour(z)

    def recon_logits(self, h_c):
        """用 h_c 去重建 C：逼混杂摘要真的装着可观测的近/热/区/时。"""
        return self.recon_acc(h_c), self.recon_pop(h_c), self.recon_area(h_c), self.recon_hour(h_c)
