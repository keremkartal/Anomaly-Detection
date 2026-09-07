"""
Değiştirilebilir zamansal ve uzamsal kodlayıcılar (F4 benchmark).

Hakem 1 M1 / Hakem 2 M1: STGCN, DCRNN ve Graph WaveNet ile karşılaştırma.
Bu mimariler düğüm-düzeyi çok değişkenli tahmin için tasarlanmıştır; buradaki
görev (tek aracın 50 adımlık penceresi -> tek segment, ikili sınıflandırma)
yapısal olarak farklıdır. Bu yüzden her mimarinin ÇEKİRDEK OPERATÖRÜ
(gated temporal conv, difüzyon konvolüsyonu, uyarlanabilir komşuluk, dilated
causal conv) izole edilip aynı görev ve aynı bütçe altında karşılaştırılır.
Bu adaptasyon kararı makalede açıkça belgelenmelidir.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj

from . import config as C


# ===================================================================== ZAMANSAL
class LSTMEncoder(nn.Module):
    """Makale Tablo 3. Çıktı: (B, hidden)"""
    name = "lstm"

    def __init__(self, in_dim=4, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.drop(o[:, -1, :])


class GRUEncoder(nn.Module):
    """DCRNN'in tekrarlayan omurgası (grafsız hâli)."""
    name = "gru"

    def __init__(self, in_dim=4, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x):
        o, _ = self.gru(x)
        return self.drop(o[:, -1, :])


class GatedTCN(nn.Module):
    """
    STGCN'in gated temporal convolution bloğu (Yu et al. 2018):
    P, Q = conv(x).chunk(2);  out = P * sigmoid(Q)
    """

    def __init__(self, c_in, c_out, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, 2 * c_out, kernel, dilation=dilation)

    def forward(self, x):                      # (B, C, T)
        x = F.pad(x, (self.pad, 0))            # causal
        p, q = self.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)


class STGCNTemporalEncoder(nn.Module):
    """STGCN tarzı yığılmış gated temporal conv."""
    name = "stgcn_temporal"

    def __init__(self, in_dim=4, hidden=128, layers=2, dropout=0.2, kernel=3):
        super().__init__()
        chans = [in_dim] + [hidden] * layers
        self.blocks = nn.ModuleList([GatedTCN(chans[i], chans[i + 1], kernel)
                                     for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x):                      # (B, T, F)
        h = x.transpose(1, 2)                  # (B, F, T)
        for b in self.blocks:
            h = b(h)
        return self.drop(self.norm(h[:, :, -1]))


class WaveNetTemporalEncoder(nn.Module):
    """
    Graph WaveNet'in dilated causal conv yığını (Wu et al. 2019),
    residual + skip bağlantılarıyla.
    """
    name = "wavenet_temporal"

    def __init__(self, in_dim=4, hidden=128, layers=4, dropout=0.2, kernel=2):
        super().__init__()
        self.start = nn.Conv1d(in_dim, hidden, 1)
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.pads = []
        for i in range(layers):
            d = 2 ** i
            self.pads.append((kernel - 1) * d)
            self.filters.append(nn.Conv1d(hidden, hidden, kernel, dilation=d))
            self.gates.append(nn.Conv1d(hidden, hidden, kernel, dilation=d))
            self.residuals.append(nn.Conv1d(hidden, hidden, 1))
            self.skips.append(nn.Conv1d(hidden, hidden, 1))
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x):                      # (B, T, F)
        h = self.start(x.transpose(1, 2))
        skip = 0
        for f, g, r, s, p in zip(self.filters, self.gates, self.residuals,
                                 self.skips, self.pads):
            hp = F.pad(h, (p, 0))
            z = torch.tanh(f(hp)) * torch.sigmoid(g(hp))
            skip = skip + s(z)
            h = h + r(z)
        return self.drop(self.norm(F.relu(skip)[:, :, -1]))


class TransformerTemporalEncoder(nn.Module):
    """Standart Transformer encoder + son adım havuzlaması."""
    name = "transformer_temporal"

    def __init__(self, in_dim=4, hidden=128, layers=2, dropout=0.2, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, C.SEQUENCE_LENGTH, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        return self.drop(self.enc(h)[:, -1, :])


TEMPORAL_ENCODERS = {
    "lstm": LSTMEncoder,
    "gru": GRUEncoder,
    "stgcn_temporal": STGCNTemporalEncoder,
    "wavenet_temporal": WaveNetTemporalEncoder,
    "transformer_temporal": TransformerTemporalEncoder,
}


# ===================================================================== UZAMSAL
class GATSpatial(nn.Module):
    """Makale Tablo 4 (önerilen)."""
    name = "gat"

    def __init__(self, in_dim, hidden=32, out_dim=64, heads=2, dropout=0.2, **kw):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_dim, heads=1, concat=False,
                             dropout=dropout)
        self.p = dropout
        self.out_dim = out_dim

    def forward(self, x, edge_index, cache=None):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.p, training=self.training)
        return self.conv2(h, edge_index)


class ChebSpatial(nn.Module):
    """
    STGCN'in uzamsal bileşeni: Chebyshev polinom graf konvolüsyonu (K hop).
    Yu et al. 2018.
    """
    name = "cheb"

    def __init__(self, in_dim, hidden=32, out_dim=64, K=3, dropout=0.2, **kw):
        super().__init__()
        self.K = K
        self.lin1 = nn.Linear(in_dim * K, hidden)
        self.lin2 = nn.Linear(hidden * K, out_dim)
        self.p = dropout
        self.out_dim = out_dim

    @staticmethod
    def _norm_adj(edge_index, n, device):
        A = to_dense_adj(edge_index, max_num_nodes=n)[0]
        A = A + A.t()
        A = (A > 0).float()
        d = A.sum(1).clamp(min=1)
        Dm = torch.diag(d.pow(-0.5))
        L = torch.eye(n, device=device) - Dm @ A @ Dm
        return L - torch.eye(n, device=device)          # ölçekli Laplasyen

    def _cheb(self, x, L):
        outs, t0 = [x], x
        if self.K > 1:
            t1 = L @ x
            outs.append(t1)
            for _ in range(2, self.K):
                t2 = 2 * (L @ t1) - t0
                outs.append(t2)
                t0, t1 = t1, t2
        return torch.cat(outs, dim=1)

    def forward(self, x, edge_index, cache=None):
        n = x.size(0)
        L = cache.get("cheb_L") if cache is not None else None
        if L is None:
            L = self._norm_adj(edge_index, n, x.device)
            if cache is not None:
                cache["cheb_L"] = L
        h = F.relu(self.lin1(self._cheb(x, L)))
        h = F.dropout(h, p=self.p, training=self.training)
        return self.lin2(self._cheb(h, L))


class DiffusionSpatial(nn.Module):
    """
    DCRNN'in çift yönlü difüzyon konvolüsyonu (Li et al. 2018), K hop.
    Makaledeki §V-H varyantının tam ve düzgün hâli.
    """
    name = "diffusion"

    def __init__(self, in_dim, hidden=32, out_dim=64, K=2, dropout=0.2, **kw):
        super().__init__()
        self.K = K
        span = 2 * K + 1                    # ileri + geri + kendi
        self.lin1 = nn.Linear(in_dim * span, hidden)
        self.lin2 = nn.Linear(hidden * span, out_dim)
        self.p = dropout
        self.out_dim = out_dim

    @staticmethod
    def _transitions(edge_index, n, device):
        A = to_dense_adj(edge_index, max_num_nodes=n)[0]
        Pf = A / A.sum(1, keepdim=True).clamp(min=1)
        Pb = A.t() / A.t().sum(1, keepdim=True).clamp(min=1)
        return Pf, Pb

    def _diffuse(self, x, Pf, Pb):
        outs, hf, hb = [x], x, x
        for _ in range(self.K):
            hf = Pf @ hf
            hb = Pb @ hb
            outs += [hf, hb]
        return torch.cat(outs, dim=1)

    def forward(self, x, edge_index, cache=None):
        n = x.size(0)
        P = cache.get("diff_P") if cache is not None else None
        if P is None:
            P = self._transitions(edge_index, n, x.device)
            if cache is not None:
                cache["diff_P"] = P
        Pf, Pb = P
        h = F.relu(self.lin1(self._diffuse(x, Pf, Pb)))
        h = F.dropout(h, p=self.p, training=self.training)
        return self.lin2(self._diffuse(h, Pf, Pb))


class AdaptiveSpatial(nn.Module):
    """
    Graph WaveNet'in öğrenilebilir uyarlanabilir komşuluk matrisi (Wu et al. 2019):
    A_adp = softmax(relu(E1 @ E2^T)) — önceden tanımlı grafla toplanır.
    """
    name = "adaptive"

    def __init__(self, in_dim, hidden=32, out_dim=64, num_nodes=None,
                 emb_dim=10, K=2, dropout=0.2, **kw):
        super().__init__()
        if num_nodes is None:
            raise ValueError("AdaptiveSpatial için num_nodes gerekli")
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(emb_dim, num_nodes) * 0.01)
        self.K = K
        span = 2 * K + 1
        self.lin1 = nn.Linear(in_dim * span, hidden)
        self.lin2 = nn.Linear(hidden * span, out_dim)
        self.p = dropout
        self.out_dim = out_dim

    def _props(self, edge_index, n, device, cache):
        A0 = cache.get("adp_A0") if cache is not None else None
        if A0 is None:
            A = to_dense_adj(edge_index, max_num_nodes=n)[0]
            A0 = A / A.sum(1, keepdim=True).clamp(min=1)
            if cache is not None:
                cache["adp_A0"] = A0
        Aadp = F.softmax(F.relu(self.E1 @ self.E2), dim=1)
        return A0, Aadp

    def _prop(self, x, A0, Aadp):
        outs, h0, ha = [x], x, x
        for _ in range(self.K):
            h0 = A0 @ h0
            ha = Aadp @ ha
            outs += [h0, ha]
        return torch.cat(outs, dim=1)

    def forward(self, x, edge_index, cache=None):
        A0, Aadp = self._props(edge_index, x.size(0), x.device, cache)
        h = F.relu(self.lin1(self._prop(x, A0, Aadp)))
        h = F.dropout(h, p=self.p, training=self.training)
        return self.lin2(self._prop(h, A0, Aadp))


class NoSpatial(nn.Module):
    """Uzamsal dal yok — sıfır vektör döner (temporal-only kontrol)."""
    name = "none"

    def __init__(self, in_dim, out_dim=64, **kw):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x, edge_index, cache=None):
        return torch.zeros(x.size(0), self.out_dim, device=x.device)


SPATIAL_ENCODERS = {
    "gat": GATSpatial,
    "cheb": ChebSpatial,
    "diffusion": DiffusionSpatial,
    "adaptive": AdaptiveSpatial,
    "none": NoSpatial,
}
