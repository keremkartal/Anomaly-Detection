"""
Genel hibrit model — zamansal kodlayıcı × uzamsal kodlayıcı × füzyon
üçlüsünün her kombinasyonunu tek sınıfla kurar.

F3 (füzyon ekseni) ve F4 (kodlayıcı ekseni) bu sınıfı kullanır; böylece
karşılaştırmalar arasında sınıflandırıcı, projeksiyon boyutu ve eğitim
protokolü birebir sabit kalır.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .encoders import SPATIAL_ENCODERS, TEMPORAL_ENCODERS

FUSION_TYPES = ("gated", "concat", "weighted_sum", "cross_attention")


class GenericHybrid(nn.Module):
    def __init__(self, temporal="lstm", spatial="gat", fusion="gated",
                 num_node_features=17, num_nodes=None, fusion_dim=C.FUSION_DIM,
                 dropout=C.DROPOUT, temporal_hidden=C.LSTM_HIDDEN_DIM,
                 spatial_hidden=C.GAT_HIDDEN_DIM, spatial_out=C.GAT_OUTPUT_DIM):
        super().__init__()
        if fusion not in FUSION_TYPES:
            raise ValueError(f"bilinmeyen fusion: {fusion}")
        self.temporal_name, self.spatial_name, self.fusion_type = temporal, spatial, fusion

        self.temporal = TEMPORAL_ENCODERS[temporal](
            in_dim=C.LSTM_INPUT_DIM, hidden=temporal_hidden, dropout=dropout)
        self.spatial = SPATIAL_ENCODERS[spatial](
            in_dim=num_node_features, hidden=spatial_hidden, out_dim=spatial_out,
            num_nodes=num_nodes, dropout=dropout)

        self.project_t = nn.Linear(self.temporal.out_dim, fusion_dim)
        self.project_s = nn.Linear(self.spatial.out_dim, fusion_dim)

        if fusion == "gated":
            self.attention_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, 32), nn.Tanh(),
                nn.Linear(32, 1), nn.Sigmoid())
            clf_in = fusion_dim
        elif fusion == "concat":
            clf_in = fusion_dim * 2
        elif fusion == "weighted_sum":
            self.static_weight = nn.Parameter(torch.tensor(0.0))
            clf_in = fusion_dim
        else:
            self.cross_attn = nn.MultiheadAttention(fusion_dim, 4, batch_first=True)
            clf_in = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid())

        self._cache = {}     # graf operatörleri (Laplasyen vb.) tekrar hesaplanmasın

    def fusion_parameters(self) -> int:
        keys = ("attention_gate", "static_weight", "cross_attn")
        return sum(p.numel() for n, p in self.named_parameters()
                   if any(k in n for k in keys))

    def branch_parameters(self) -> dict:
        out = {"temporal": 0, "spatial": 0, "fusion": 0, "classifier": 0, "projection": 0}
        for n, p in self.named_parameters():
            if n.startswith("temporal."):
                out["temporal"] += p.numel()
            elif n.startswith("spatial."):
                out["spatial"] += p.numel()
            elif n.startswith("project_"):
                out["projection"] += p.numel()
            elif n.startswith("classifier."):
                out["classifier"] += p.numel()
            else:
                out["fusion"] += p.numel()
        return out

    def forward(self, x_seq, node_x, edge_index, seg_idx):
        emb_t = self.temporal(x_seq)
        emb_s = self.spatial(node_x, edge_index, self._cache)[seg_idx]

        p_t = F.relu(self.project_t(emb_t))
        p_s = F.relu(self.project_s(emb_s))

        if self.fusion_type == "gated":
            w = self.attention_gate(torch.cat([p_t, p_s], dim=1))
            fused = w * p_t + (1 - w) * p_s
        elif self.fusion_type == "concat":
            fused = torch.cat([p_t, p_s], dim=1)
            w = torch.full((x_seq.size(0), 1), 0.5, device=x_seq.device)
        elif self.fusion_type == "weighted_sum":
            s = torch.sigmoid(self.static_weight)
            fused = s * p_t + (1 - s) * p_s
            w = s.detach().expand(x_seq.size(0), 1)
        else:
            q = p_t.unsqueeze(1)
            kv = torch.stack([p_t, p_s], dim=1)
            out, attn = self.cross_attn(q, kv, kv)
            fused = out.squeeze(1)
            w = attn[:, 0, 0].unsqueeze(1)

        return self.classifier(fused), w
