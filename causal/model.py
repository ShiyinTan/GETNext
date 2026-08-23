"""附录 D.3–D.4 的网络：编码器 → 拆成 (h_z, h_c) → 兴趣分 + 混杂分。

对照 GETNext，这里按因果规格改掉的地方（冲突以附录 D 为准）：
  1. 地点向量 e_p 用普通查表 nn.Embedding，不用 GCN（附录 A / D.8：
     GCN 会把热度、转移次数再灌进表征，和混杂通道重复计算）
  2. 不再把 NodeAttnMap 加到 POI 分数上（那也是近邻先验）
  3. 混杂 C 不拼进 Transformer 的输入 token（D.3.4，避免泄漏）
  4. 排序用 s = s_pref + s_conf，而不是一个混在一起的 CE 头

Time2Vec / 用户嵌入 / 类别嵌入仍从原来的 model.py 借用，只读不改。

形状记号（每个函数的「输入 / 输出」都用这套）：
  B = batch 里有几条轨迹
  T = 补齐后的时间步长度
  N = 地点词表大小（NYC 大约 5000）
  d = d_model = poi+user+time+cat 嵌入维之和
  d_z = poi_embed_dim（兴趣向量，必须和 e_p 同宽才能点积）
  d_c = hc_dim（混杂向量）
  K / P / A / H = 距离桶 / 热度档 / 区域数 / 时刻桶
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
        """
        输入:
            x: 任意形状，训练时是 h_z，(B, T, d_z)
            lambd: 标量 float，反转强度
        输出:
            与 x 同形状，数值不变
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        输入:
            grad_output: 与 forward 的 x 同形状
        输出:
            (对 x 的梯度, 对 lambd 的梯度)
            对 x: 同形状，等于 -λ * grad_output
            对 lambd: None（不更新这个标量）
        """
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    """给 h_z 套上 GRL 后再送给对抗分类器。

    输入:
        x: (B, T, d_z)
        lambd: 标量
    输出:
        (B, T, d_z) 前向原样；反向梯度乘 -λ
    """
    return GradientReversalFn.apply(x, lambd)


class PositionalEncoding(nn.Module):
    """给序列每个位置加上「第几步」的正弦编码。输入输出都是 (B, T, d)。"""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        """
        输入:
            d_model: int = d
            dropout: float
            max_len: int，最长位置
        输出:
            无返回。buffer pe: (1, max_len, d)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)，随模型保存但不训练

    def forward(self, x):
        """
        输入:
            x: (B, T, d)
        输出:
            (B, T, d)  加上位置编码后 dropout
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FuseEmbeddings(nn.Module):
    """GETNext 同款融合：两段向量拼起来，过一层 Linear + LeakyReLU。

    和原版的差别：这里按最后一维拼接，所以一次能处理整个 (B, T, *) batch，
    不用像原 train.py 那样逐步用 Python 循环。
    """

    def __init__(self, dim_a, dim_b):
        """
        输入:
            dim_a, dim_b: int，两路向量最后一维
        输出:
            无返回。Linear 权重: (dim_a+dim_b, dim_a+dim_b)
        """
        super().__init__()
        embed_dim = dim_a + dim_b
        self.fuse = nn.Linear(embed_dim, embed_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, a, b):
        """
        输入:
            a: (..., dim_a)  常见 (B, T, user_dim) 或 (B, T, time_dim)
            b: (..., dim_b)  常见 (B, T, poi_dim)  或 (B, T, cat_dim)
        输出:
            (..., dim_a+dim_b)
        """
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
        """
        输入:
            args: 超参对象（标量，无 tensor）
            num_pois: int = N
            num_users: int = n_user
            num_cats: int = n_cat
            table: PoiConfounderTable，只用 K/P/A/H 这些桶数
        输出:
            无返回。主要权重形状：
              poi_embedding.weight: (N, d_z)
              user_embedding: (n_user, user_embed_dim)
              cat_embedding: (n_cat, cat_embed_dim)
              split_z: d → d_z ； split_c: d → d_c
              g_acc.weight: (K, 1)
              psi.weight: (N, d_c)
              adv_*: d_z → K/P/A/H ； recon_*: d_c → K/P/A/H
              cat_head: d_z → n_cat ； time_head: d → 1
              w_pref / w_conf / w_acc / w_pop / w_area / w_ctx: 标量，默认 1
        """
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

        # 分数怎么加：附录 D 是直接相加。这里加可调权重，默认全是 1，行为不变。
        # g_acc / g_pop / g_area 自己已有可学的尺度，所以内部四项默认不要网格搜索。
        self.apply_score_weights(args)

    def apply_score_weights(self, args):
        """从 args 读分数权重。缺省（旧 checkpoint）一律当 1。

        输入:
            args: 有 w_pref / w_conf / w_acc / w_pop / w_area / w_ctx 的对象
        输出:
            无返回。写入 self 上的 6 个标量 float。
        """
        self.w_pref = float(getattr(args, 'w_pref', 1.0))
        self.w_conf = float(getattr(args, 'w_conf', 1.0))
        self.w_acc = float(getattr(args, 'w_acc', 1.0))
        self.w_pop = float(getattr(args, 'w_pop', 1.0))
        self.w_area = float(getattr(args, 'w_area', 1.0))
        self.w_ctx = float(getattr(args, 'w_ctx', 1.0))

    def _token_embed(self, poi_idx, time_feat, cat_idx, user_idx):
        """把一步的 (地点, 时间, 类别, 用户) 融合成 Transformer 的一个 token。

        注意：这里没有把距离、热度拼进去（D.3.4）。

        输入:
            poi_idx:  (B, T) long
            time_feat:(B, T) float
            cat_idx:  (B, T) long
            user_idx: (B,)   long，整条轨迹共用一个用户
        输出:
            fused: (B, T, d)
        中间:
            poi_e:  (B, T, d_z)
            user_e: (B, T, user_dim)  把 (B, user_dim) 扩到每个时间步
            time_e: (B, T, time_dim)  Time2Vec 吃 (B*T, 1) 再 reshape
            cat_e:  (B, T, cat_dim)
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

        输入:
            poi_idx:   (B, T) long
            time_feat: (B, T) float
            cat_idx:   (B, T) long
            user_idx:  (B,)   long
            pad_mask:  (B, T) bool，True=pad
        输出:
            h:   (B, T, d)    还没拆开的上下文
            h_z: (B, T, d_z)  兴趣代理
            h_c: (B, T, d_c)  混杂摘要
        中间:
            src:    (B, T, d)  token
            causal: (T, T)     上三角 -inf，第 t 步只能看见 1..t
            encoder 内部: (T, B, d)  seq-first
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
        """兴趣通道：s_pref(p) = h_z 和地点向量 e_p 的点积。

        输入:
            h_z: (B, T, d_z)
        输出:
            s_pref: (B, T, N)
        中间:
            e_p: (N, d_z)  和输入嵌入绑在一起
        """
        e_p = self.poi_embedding.weight  # (N, d_z)，和输入嵌入绑在一起
        return torch.matmul(h_z, e_p.transpose(0, 1))

    def _gather_origin_tables(self, origin_idx, buffers):
        """按当前起点 p_T 取出「到每一个候选点」的距离桶 / 公里数 / 起点区域。

        输入:
            origin_idx: (B, T) long，当前步所在 POI（通常就是输入轨迹 poi）
            buffers: dict
                dist_bin: (N, N) long
                dist_km:  (N, N) float
                area_id:  (N,)   long
        输出:
            dist_bin:    (B, T, N)  从每个起点到全部候选的距离桶
            dist_km:     (B, T, N)  同上，公里
            origin_area: (B, T)     每个起点自己的区域 id
        """
        safe = origin_idx.clamp(min=0)
        dist_bin = buffers['dist_bin'][safe]          # (B, T, N)
        dist_km = buffers['dist_km'][safe]
        origin_area = buffers['area_id'][safe]        # (B, T)
        return dist_bin, dist_km, origin_area

    def _s_conf_from_phi(self, h_c, dist_bin, origin_area, buffers):
        """混杂通道：四项先各自算出，再按 w_acc / w_pop / w_area / w_ctx 加权求和。

        默认权重全是 1，等于附录 D 的直接相加。
        返回的 s_conf 已经乘过内部权重；parts 里仍是未加权的原始项，
        方便 deconf_do 替换热度后再 mix 一次。

        输入:
            h_c:         (B, T, d_c)
            dist_bin:    (B, T, N) long
            origin_area: (B, T)    long
            buffers:
                log_pop: (N,)
                area_id: (N,)
        输出:
            s_conf: (B, T, N)
            parts: dict
                s_acc:  (B, T, N)  距离桶查表分
                s_pop:  (1, 1, N)  热度分，与轨迹无关，广播到 B,T
                s_area: (B, T, N)  起点区域 · 终点区域
                s_ctx:  (B, T, N)  <W_c h_c, ψ(p)>
        """
        s_acc = self.g_acc(dist_bin.clamp(min=0)).squeeze(-1)             # (B, T, N)
        s_pop = self.g_pop(buffers['log_pop'].unsqueeze(-1)).squeeze(-1)  # (N,)，与轨迹无关
        dest_area_e = self.area_emb(buffers['area_id'])                   # (N, 16)
        origin_area_e = self.area_emb(origin_area.clamp(min=0))           # (B, T, 16)
        # 起点区域向量 · 每个终点区域向量 → 同区更高
        area_match = torch.matmul(origin_area_e, dest_area_e.transpose(0, 1))
        s_area = self.g_area(area_match.unsqueeze(-1)).squeeze(-1)
        ctx = torch.matmul(self.W_c(h_c), self.psi.weight.transpose(0, 1))
        s_pop_b = s_pop.view(1, 1, -1)
        parts = {
            's_acc': s_acc,
            's_pop': s_pop_b,
            's_area': s_area,
            's_ctx': ctx,
        }
        return self._mix_s_conf(parts), parts

    def _mix_s_conf(self, parts):
        """s_conf = w_acc*距离 + w_pop*热度 + w_area*区域 + w_ctx*情境。默认权重全是 1。

        输入:
            parts: dict，未乘权重的四项，形状见 _s_conf_from_phi
        输出:
            s_conf: (B, T, N)
        """
        return (self.w_acc * parts['s_acc']
                + self.w_pop * parts['s_pop']
                + self.w_area * parts['s_area']
                + self.w_ctx * parts['s_ctx'])

    def _combine_scores(self, s_pref, s_conf):
        """总分 s = w_pref * s_pref + w_conf * s_conf。默认都是 1。

        输入 / 输出: 均为 (B, T, N)
        """
        return self.w_pref * s_pref + self.w_conf * s_conf

    def score(self, h_z, h_c, origin_idx, buffers, mode='factual',
              bar_acc_bin=None, bar_pop_log=None):
        """按推理模式给出总分 / 兴趣分 / 混杂分（附录 D.5）。

        输入:
            h_z:        (B, T, d_z)
            h_c:        (B, T, d_c)
            origin_idx: (B, T) long
            buffers:    见 _gather_origin_tables / _s_conf_from_phi
            mode:       str
            bar_acc_bin / bar_pop_log: do(C) 用的标量干预值
        输出:
            s:      (B, T, N)  w_pref * s_pref + w_conf * s_conf，用来排序
            s_pref: (B, T, N)  未乘 w_pref（L_pref 仍用原始兴趣分）
            s_conf: (B, T, N)  已乘内部四项权重
            dist_km:(B, T, N)  真实公里数（评估切片用，不受 mode 改写）

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
            return self._combine_scores(s_pref, s_conf), s_pref, s_conf, dist_km

        if mode == 'deconf_do':
            # do(C=c_bar)：每个候选都用同一个距离桶，热度换成常数
            # 这样「更近 / 更热」不再能拉开名次
            if bar_acc_bin is None:
                bar_acc_bin = 0
            dist_bin = torch.full_like(dist_bin, int(bar_acc_bin))
            _, parts = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
            if bar_pop_log is None:
                pop_const = self.g_pop(buffers['log_pop'].mean().view(1, 1)).view(1, 1, 1)
            else:
                pop_const = self.g_pop(
                    torch.tensor([[float(bar_pop_log)]], device=h_z.device, dtype=h_z.dtype)
                ).view(1, 1, 1)
            parts = dict(parts)
            parts['s_pop'] = pop_const.expand_as(parts['s_pop'])
            s_conf = self._mix_s_conf(parts)
            return self._combine_scores(s_pref, s_conf), s_pref, s_conf, dist_km

        if mode == 'deconf_sum':
            # 若给所有候选同一个 g_acc / 平均 g_pop，名次不变，等价于丢掉这两项
            _, parts = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
            s_conf = self.w_area * parts['s_area'] + self.w_ctx * parts['s_ctx']
            return self._combine_scores(s_pref, s_conf), s_pref, s_conf, dist_km

        # factual：真实 C(p)
        s_conf, _ = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
        return self._combine_scores(s_pref, s_conf), s_pref, s_conf, dist_km

    def g_tilde(self, origin_idx, buffers, alpha, beta):
        """手工先验混杂分 g̃（附录 D.4.3，训练时 stop-grad）。

        越近越高（-α × 公里），越热越高（+β × log热度），同区域再加一点。
        用来把「近/热/同区」从兴趣通道里挤到 s_conf。

        输入:
            origin_idx: (B, T) long
            buffers: dist_km (N,N), log_pop (N,), area_id (N,)
            alpha, beta: 标量
        输出:
            g_tilde: (B, T, N)
        """
        dist_km = buffers['dist_km'][origin_idx.clamp(min=0)]
        log_pop = buffers['log_pop'].view(1, 1, -1)
        origin_area = buffers['area_id'][origin_idx.clamp(min=0)]
        dest_area = buffers['area_id'].view(1, 1, -1)
        same_area = (origin_area.unsqueeze(-1) == dest_area).float()
        return (-alpha * dist_km) + beta * log_pop + 0.15 * same_area

    def adv_logits(self, h_z, lambd):
        """用 GRL(h_z) 去猜 C 的四个桶。分类器想猜对，编码器被反转梯度逼着猜不对。

        输入:
            h_z: (B, T, d_z)
            lambd: 标量
        输出: 四个分类 logit
            acc:  (B, T, K)
            pop:  (B, T, P)
            area: (B, T, A)
            hour: (B, T, H)
        """
        z = grad_reverse(h_z, lambd)
        return self.adv_acc(z), self.adv_pop(z), self.adv_area(z), self.adv_hour(z)

    def recon_logits(self, h_c):
        """用 h_c 去重建 C：逼混杂摘要真的装着可观测的近/热/区/时。

        输入:
            h_c: (B, T, d_c)
        输出: 四个分类 logit
            acc:  (B, T, K)
            pop:  (B, T, P)
            area: (B, T, A)
            hour: (B, T, H)
        """
        return self.recon_acc(h_c), self.recon_pop(h_c), self.recon_area(h_c), self.recon_hour(h_c)
