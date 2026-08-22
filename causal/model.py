"""Appendix D.3–D.4 model: encoder, (h_z, h_c) split, additive score, aux heads.

Conflicts with GETNext that follow the causal spec:
  * e_p is nn.Embedding, not GCN(checkin_cnt, A)          — Appendix A / D.8.1
  * no NodeAttnMap added into logits                      — would double-count C
  * C is NOT concatenated into Transformer tokens         — D.3.4
  * ranking uses s = s_pref + s_conf, not a single CE head

Reusable GETNext modules (Time2Vec / user / category embeddings) are imported
read-only from the original `model.py`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CategoryEmbeddings, Time2Vec, UserEmbeddings


class GradientReversalFn(torch.autograd.Function):
    """GRL: identity in forward, multiply incoming grad by -lambda in backward.

    Lets us train the adversarial C-predictors and the encoder in one backward
    (Appendix D.4.4 / Algorithm 1 step 9: encoder minimises -L_adv).
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReversalFn.apply(x, lambd)


class PositionalEncoding(nn.Module):
    """Sinusoidal PE with batch-first layout (B, T, d)."""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FuseEmbeddings(nn.Module):
    """Batched GETNext-style fuse: concat last dim, Linear + LeakyReLU."""

    def __init__(self, dim_a, dim_b):
        super().__init__()
        embed_dim = dim_a + dim_b
        self.fuse = nn.Linear(embed_dim, embed_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, a, b):
        return self.act(self.fuse(torch.cat((a, b), dim=-1)))


class CausalNextPOI(nn.Module):
    """Encoder E, split S, decoder D, confounder heads G (Appendix D.3)."""

    def __init__(self, args, num_pois, num_users, num_cats, table):
        super().__init__()
        self.num_pois = num_pois
        self.poi_embed_dim = args.poi_embed_dim
        self.hz_dim = args.hz_dim or args.poi_embed_dim
        self.hc_dim = args.hc_dim
        if self.hz_dim != args.poi_embed_dim:
            raise ValueError('hz_dim must equal poi_embed_dim so s_pref=<h_z, e_p> is well-defined')

        d_model = (args.poi_embed_dim + args.user_embed_dim
                   + args.time_embed_dim + args.cat_embed_dim)
        self.d_model = d_model

        # Token embeddings. e_p is tied: input lookup AND preference-dot vocabulary.
        self.poi_embedding = nn.Embedding(num_pois, args.poi_embed_dim)
        self.user_embedding = UserEmbeddings(num_users, args.user_embed_dim)
        self.time_encoder = Time2Vec('sin', out_dim=args.time_embed_dim)
        self.cat_embedding = CategoryEmbeddings(num_cats, args.cat_embed_dim)
        self.fuse_up = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
        self.fuse_tc = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

        self.pos_encoder = PositionalEncoding(d_model, dropout=args.transformer_dropout)
        # seq-first encoder: works on GETNext's GPU pin (torch 1.7) and Cloud CPU (2.x).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=args.transformer_nhead,
            dim_feedforward=args.transformer_nhid,
            dropout=args.transformer_dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=args.transformer_nlayers)

        # D.3.2 split: h -> (h_z, h_c). Partial disentanglement, not a hard orthogonality.
        self.split_z = nn.Sequential(nn.Linear(d_model, self.hz_dim), nn.LeakyReLU(0.2))
        self.split_c = nn.Sequential(nn.Linear(d_model, self.hc_dim), nn.LeakyReLU(0.2))

        # D.3.3 confounder channel g(c, p). Scalar heads over discrete / numeric phi.
        self.g_acc = nn.Embedding(table.num_acc_bins, 1)
        self.g_pop = nn.Linear(1, 1)
        self.area_emb = nn.Embedding(table.num_areas, 16)
        self.g_area = nn.Linear(1, 1)  # scalar head on origin·dest area match
        # Context term <W_c h_c, psi(p)> for confounders not fully captured by lookup.
        self.psi = nn.Embedding(num_pois, self.hc_dim)
        self.W_c = nn.Linear(self.hc_dim, self.hc_dim, bias=False)

        # D.4.4 heads. Adv sees GRL(h_z); recon sees h_c.
        self.adv_acc = nn.Linear(self.hz_dim, table.num_acc_bins)
        self.adv_pop = nn.Linear(self.hz_dim, table.num_pop_bins)
        self.adv_area = nn.Linear(self.hz_dim, table.num_areas)
        self.adv_hour = nn.Linear(self.hz_dim, table.num_hour_bins)
        self.recon_acc = nn.Linear(self.hc_dim, table.num_acc_bins)
        self.recon_pop = nn.Linear(self.hc_dim, table.num_pop_bins)
        self.recon_area = nn.Linear(self.hc_dim, table.num_areas)
        self.recon_hour = nn.Linear(self.hc_dim, table.num_hour_bins)

        # Optional aux: category from h_z (D.4.2b); time from h (GETNext leftover, default off).
        self.cat_head = nn.Linear(self.hz_dim, num_cats)
        self.time_head = nn.Linear(d_model, 1)

        nn.init.zeros_(self.g_acc.weight)
        nn.init.zeros_(self.g_pop.weight)
        nn.init.zeros_(self.g_pop.bias)
        nn.init.zeros_(self.g_area.weight)
        nn.init.zeros_(self.g_area.bias)

    def _token_embed(self, poi_idx, time_feat, cat_idx, user_idx):
        """Build fused sequence tokens. Shapes: poi/time/cat (B,T), user (B,)."""
        bsz, seqlen = poi_idx.size()
        poi_e = self.poi_embedding(poi_idx.clamp(min=0))  # (B, T, d_e)
        user_e = self.user_embedding(user_idx).unsqueeze(1).expand(bsz, seqlen, -1)
        time_flat = time_feat.reshape(-1, 1)
        time_e = self.time_encoder(time_flat).view(bsz, seqlen, -1)
        cat_e = self.cat_embedding(cat_idx.clamp(min=0))
        fused = torch.cat((self.fuse_up(user_e, poi_e), self.fuse_tc(time_e, cat_e)), dim=-1)
        return fused

    def encode(self, poi_idx, time_feat, cat_idx, user_idx, pad_mask):
        """E(H,u) then S(h). pad_mask True = padding (ignored by the encoder)."""
        src = self._token_embed(poi_idx, time_feat, cat_idx, user_idx)  # (B, T, d)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        seqlen = src.size(1)
        # Causal mask so step t only sees 1..t (GETNext next-step training convention).
        causal = torch.triu(
            torch.ones(seqlen, seqlen, device=src.device, dtype=src.dtype) * float('-inf'),
            diagonal=1,
        )
        h = self.encoder(src.transpose(0, 1), mask=causal, src_key_padding_mask=pad_mask)
        h = h.transpose(0, 1)  # back to (B, T, d)
        h_z = self.split_z(h)
        h_c = self.split_c(h)
        return h, h_z, h_c

    def _s_pref(self, h_z):
        # s_pref(p) = <h_z, e_p>  — D.3.3 preference channel (tied with input e_p).
        e_p = self.poi_embedding.weight  # (N, d_z)
        return torch.matmul(h_z, e_p.transpose(0, 1))  # (B, T, N)

    def _gather_origin_tables(self, origin_idx, buffers):
        """Lookup (origin, *) candidate features. origin_idx: (B, T)."""
        safe = origin_idx.clamp(min=0)
        dist_bin = buffers['dist_bin'][safe]          # (B, T, N)
        dist_km = buffers['dist_km'][safe]            # (B, T, N)
        origin_area = buffers['area_id'][safe]        # (B, T)
        return dist_bin, dist_km, origin_area

    def _s_conf_from_phi(self, h_c, dist_bin, origin_area, buffers):
        """s_conf(p) = g_acc + g_pop + g_area + <W_c h_c, psi(p)>."""
        s_acc = self.g_acc(dist_bin.clamp(min=0)).squeeze(-1)          # (B, T, N)
        s_pop = self.g_pop(buffers['log_pop'].unsqueeze(-1)).squeeze(-1)  # (N,)
        dest_area_e = self.area_emb(buffers['area_id'])  # (N, 16)
        origin_area_e = self.area_emb(origin_area.clamp(min=0))  # (B, T, 16)
        # Area interaction: origin embedding dotted with every dest embedding.
        area_match = torch.matmul(origin_area_e, dest_area_e.transpose(0, 1))  # (B, T, N)
        s_area = self.g_area(area_match.unsqueeze(-1)).squeeze(-1)
        ctx = torch.matmul(self.W_c(h_c), self.psi.weight.transpose(0, 1))  # (B, T, N)
        s_pop_b = s_pop.view(1, 1, -1)
        return s_acc + s_pop_b + s_area + ctx, {
            's_acc': s_acc,
            's_pop': s_pop_b,
            's_area': s_area,
            's_ctx': ctx,
        }

    def score(self, h_z, h_c, origin_idx, buffers, mode='factual',
              bar_acc_bin=None, bar_pop_log=None):
        """Return s, s_pref, s_conf under a scoring mode (D.5).

        mode:
          factual      — real C(p)
          deconf_pref  — s_pref only  (preferred do(C) / interest ranking)
          deconf_do    — s_pref + s_conf with access/pop replaced by bar values
          deconf_sum   — mix s_conf over empirical P(c_acc) (and mean pop)
        """
        s_pref = self._s_pref(h_z)
        dist_bin, dist_km, origin_area = self._gather_origin_tables(origin_idx, buffers)

        if mode == 'deconf_pref':
            s_conf = torch.zeros_like(s_pref)
            return s_pref, s_pref, s_conf, dist_km

        if mode == 'deconf_do':
            # do(C=c_bar): same access bucket for every candidate; pop replaced by a
            # constant (mean log-pop); area/context kept so the graph still conditions
            # on h_c. Ranking then cannot exploit "closer / more popular" shortcuts.
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
            # Mix over access/pop worlds (D.5.3). Assigning every candidate the same
            # g_acc(j) / mean g_pop is rank-invariant, so the mixture keeps area+context
            # and drops the candidate-varying access/pop heads.
            _, parts = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
            s_conf = parts['s_area'] + parts['s_ctx']
            return s_pref + s_conf, s_pref, s_conf, dist_km

        # factual
        s_conf, _ = self._s_conf_from_phi(h_c, dist_bin, origin_area, buffers)
        return s_pref + s_conf, s_pref, s_conf, dist_km

    def g_tilde(self, origin_idx, buffers, alpha, beta):
        """Stop-grad hand-crafted confounder score (D.4.3a)."""
        dist_km = buffers['dist_km'][origin_idx.clamp(min=0)]  # (B, T, N)
        log_pop = buffers['log_pop'].view(1, 1, -1)
        origin_area = buffers['area_id'][origin_idx.clamp(min=0)]
        dest_area = buffers['area_id'].view(1, 1, -1)
        same_area = (origin_area.unsqueeze(-1) == dest_area).float()
        return (-alpha * dist_km) + beta * log_pop + 0.15 * same_area

    def adv_logits(self, h_z, lambd):
        z = grad_reverse(h_z, lambd)
        return self.adv_acc(z), self.adv_pop(z), self.adv_area(z), self.adv_hour(z)

    def recon_logits(self, h_c):
        return self.recon_acc(h_c), self.recon_pop(h_c), self.recon_area(h_c), self.recon_hour(h_c)
