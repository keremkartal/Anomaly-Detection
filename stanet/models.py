"""
Model tanımları — TEK yer.

Orijinal notebook'ta bu sınıflar 6 ayrı hücrede kopyalanmıştı; aralarındaki
tutarsızlık (GATModel'de `fc` başlığının bazı kopyalarda olup bazılarında
olmaması) hücre 16'da hibrit modelin yüklenememesine ve karşılaştırma
figürlerinde önerilen modelin hiç yer almamasına yol açmıştı.

`legacy_head=True` -> orijinal checkpoint'lerdeki kullanılmayan dal başlıklarını
(`fc`) da oluşturur, böylece eski ağırlıklar strict=True ile yüklenebilir.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from . import config as C


# --------------------------------------------------------------------- dallar
class LSTMBranch(nn.Module):
    """Zamansal dal (makale Tablo 3). Gömme döner."""

    def __init__(self, input_dim=C.LSTM_INPUT_DIM, hidden_dim=C.LSTM_HIDDEN_DIM,
                 num_layers=C.LSTM_LAYERS, dropout=C.DROPOUT, legacy_head=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1) if legacy_head else None

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1, :])


class GATBranch(nn.Module):
    """Uzamsal dal (makale Tablo 4). Tüm düğümlerin gömmesini döner."""

    def __init__(self, num_node_features=17, hidden_dim=C.GAT_HIDDEN_DIM,
                 output_dim=C.GAT_OUTPUT_DIM, heads=C.GAT_HEADS,
                 dropout=C.DROPOUT, legacy_head=False):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1,
                             concat=False, dropout=dropout)
        self.dropout_ratio = dropout
        self.fc = nn.Linear(output_dim, 1) if legacy_head else None

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return self.conv2(x, edge_index)


# --------------------------------------------------------------------- hibrit
FUSION_TYPES = ("gated", "concat", "weighted_sum", "cross_attention")


class FusionHybrid(nn.Module):
    """
    Paylaşılan LSTM + GAT dalları; yalnızca füzyon modülü değişir.

    gated           : makalenin önerdiği örnek-başına skaler kapı (Eş. 16-17)
    concat          : sabit birleştirme, yeniden ağırlıklandırma yok
    weighted_sum    : tüm örnekler için ortak tek öğrenilebilir skaler
    cross_attention : 4-head standart çapraz-dikkat (temporal query, {t,s} key/value)
    """

    def __init__(self, fusion_type="gated", num_node_features=17,
                 lstm_hidden=C.LSTM_HIDDEN_DIM, gat_hidden=C.GAT_HIDDEN_DIM,
                 gat_out=C.GAT_OUTPUT_DIM, fusion_dim=C.FUSION_DIM,
                 dropout=C.DROPOUT, legacy_head=False):
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"bilinmeyen fusion_type: {fusion_type}")
        self.fusion_type = fusion_type

        self.lstm_model = LSTMBranch(hidden_dim=lstm_hidden, dropout=dropout,
                                     legacy_head=legacy_head)
        self.gnn_model = GATBranch(num_node_features=num_node_features,
                                   hidden_dim=gat_hidden, output_dim=gat_out,
                                   dropout=dropout, legacy_head=legacy_head)
        self.project_lstm = nn.Linear(lstm_hidden, fusion_dim)
        self.project_gnn = nn.Linear(gat_out, fusion_dim)

        if fusion_type == "gated":
            self.attention_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, 32), nn.Tanh(),
                nn.Linear(32, 1), nn.Sigmoid())
            clf_in = fusion_dim
        elif fusion_type == "concat":
            clf_in = fusion_dim * 2
        elif fusion_type == "weighted_sum":
            self.static_weight = nn.Parameter(torch.tensor(0.0))
            clf_in = fusion_dim
        else:  # cross_attention
            self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads=4,
                                                    batch_first=True)
            clf_in = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x_lstm, node_x, edge_index, seg_idx):
        emb_t = self.lstm_model(x_lstm)
        emb_s = self.gnn_model(node_x, edge_index)[seg_idx]

        p_t = F.relu(self.project_lstm(emb_t))
        p_s = F.relu(self.project_gnn(emb_s))

        if self.fusion_type == "gated":
            w = self.attention_gate(torch.cat([p_t, p_s], dim=1))
            fused = w * p_t + (1 - w) * p_s
        elif self.fusion_type == "concat":
            fused = torch.cat([p_t, p_s], dim=1)
            w = torch.full((x_lstm.size(0), 1), 0.5, device=x_lstm.device)
        elif self.fusion_type == "weighted_sum":
            s = torch.sigmoid(self.static_weight)
            fused = s * p_t + (1 - s) * p_s
            w = s.detach().expand(x_lstm.size(0), 1)
        else:  # cross_attention
            q = p_t.unsqueeze(1)
            kv = torch.stack([p_t, p_s], dim=1)
            out, attn = self.cross_attn(q, kv, kv)
            fused = out.squeeze(1)
            w = attn[:, 0, 0].unsqueeze(1)

        return self.classifier(fused), w


# --------------------------------------------------------------------- tek dal
class LSTMOnly(nn.Module):
    """Zamansal-only baseline (makale Deney A)."""

    def __init__(self, input_dim=C.LSTM_INPUT_DIM, hidden_dim=C.LSTM_HIDDEN_DIM,
                 num_layers=C.LSTM_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, *args):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(self.dropout(out[:, -1, :]))), None


class GATOnly(nn.Module):
    """Uzamsal-only baseline (makale Deney B). Dikkat: orijinal hidden_dim=64."""

    def __init__(self, num_node_features=17, hidden_dim=64,
                 output_dim=C.GAT_OUTPUT_DIM, heads=C.GAT_HEADS, dropout=C.DROPOUT):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1,
                             concat=False, dropout=dropout)
        self.dropout_ratio = dropout
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x_lstm, node_x, edge_index, seg_idx):
        x = F.elu(self.conv1(node_x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x[seg_idx]), None


# --------------------------------------------------------------------- fabrika
def build_model(name: str, num_node_features: int = 17, **kw) -> nn.Module:
    if name in FUSION_TYPES:
        return FusionHybrid(fusion_type=name, num_node_features=num_node_features, **kw)
    if name == "lstm_only":
        return LSTMOnly(**kw)
    if name == "gat_only":
        return GATOnly(num_node_features=num_node_features, **kw)
    raise ValueError(f"bilinmeyen model: {name}")
