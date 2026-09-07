"""
Yol ağı grafı.

merge_by_osm=False -> orijinal: her trip kendi segment kopyasını üretir (4.678 düğüm,
                      bunların yalnızca 628'i tekil fiziksel yol; 538 kopya grubu
                      birebir aynı özniteliğe sahip -> GAT ayırt edemez)
merge_by_osm=True  -> F2 onarımı: aynı osm_segment_id tek düğüme indirgenir
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from . import config as C


@dataclass
class RoadGraph:
    node_features: np.ndarray          # (N, F)
    edge_index: torch.Tensor           # (2, E)
    segment_to_idx: dict               # segment_id -> node index
    feature_names: list
    info: dict

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_features(self) -> int:
        return self.node_features.shape[1]

    def to_tensors(self, device):
        return (torch.tensor(self.node_features, dtype=torch.float, device=device),
                self.edge_index.to(device))


def _build_edges(df_nodes: pd.DataFrame) -> torch.Tensor:
    """
    osm_segment_id = "<start_osm_node>-<end_osm_node>".
    A.end == B.start ise A -> B yönlü kenarı kurulur.
    """
    ids = df_nodes["osm_segment_id"].astype(str)
    # rsplit: OSM düğüm kimlikleri negatif olabilir, son '-' ayırıcıdır
    parts = ids.str.rsplit("-", n=1, expand=True)
    start, end = parts[0].values, parts[1].values

    start_map = {}
    for i, s in enumerate(start):
        start_map.setdefault(s, []).append(i)

    src, dst = [], []
    for i, e in enumerate(end):
        for j in start_map.get(e, ()):
            src.append(i)
            dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def build_graph(merge_by_osm: bool = False) -> RoadGraph:
    df = pd.read_csv(C.GNN_DATA)
    n_rows = len(df)

    df_nodes = df.drop_duplicates(subset=["segment_id"]).copy()
    df_nodes = df_nodes.sort_values("segment_id").reset_index(drop=True)
    df_nodes["length_log"] = np.log1p(df_nodes["length_meters"])

    if merge_by_osm:
        # aynı fiziksel yolu tek düğüme indir; öznitelikler kopyalarda özdeş
        # olduğu için ilk kaydı almak yeterli (538/546 grup zaten birebir aynı)
        rep = df_nodes.drop_duplicates(subset=["osm_segment_id"]).copy()
        rep = rep.sort_values("osm_segment_id").reset_index(drop=True)
        osm_to_idx = {o: i for i, o in enumerate(rep["osm_segment_id"])}
        # her segment_id, kendi fiziksel yolunun düğümüne işaret eder
        segment_to_idx = {int(s): osm_to_idx[o]
                          for s, o in zip(df_nodes["segment_id"], df_nodes["osm_segment_id"])}
        node_df = rep
    else:
        segment_to_idx = {int(s): i for i, s in enumerate(df_nodes["segment_id"])}
        node_df = df_nodes

    num = MinMaxScaler().fit_transform(node_df[C.GNN_NUM_COLS].values)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    road_type = ohe.fit_transform(node_df[["road_type"]])
    binary = node_df[C.GNN_BIN_COLS].values.astype(float)

    feats = np.hstack([num, binary, road_type]).astype(np.float32)
    names = C.GNN_NUM_COLS + C.GNN_BIN_COLS + list(ohe.get_feature_names_out(["road_type"]))

    edge_index = _build_edges(node_df)
    connected = int(torch.unique(edge_index).numel()) if edge_index.numel() else 0

    info = {
        "merge_by_osm": merge_by_osm,
        "source_rows": n_rows,
        "num_nodes": len(node_df),
        "num_unique_osm": int(df_nodes["osm_segment_id"].nunique()),
        "num_edges": int(edge_index.shape[1]),
        "avg_degree": float(edge_index.shape[1] / len(node_df)),
        "connected_nodes": connected,
        "isolated_nodes": int(len(node_df) - connected),
        "self_loops": int((edge_index[0] == edge_index[1]).sum()) if edge_index.numel() else 0,
        "num_features": feats.shape[1],
    }
    return RoadGraph(feats, edge_index, segment_to_idx, names, info)


def map_segments(meta: np.ndarray, graph: RoadGraph):
    """
    meta[:,1] (segment_id) -> düğüm indeksi.
    Eşleşmeyenleri SESSİZCE 0'a atmaz; sayısını raporlar.
    """
    idx, missing = [], 0
    for s in meta[:, 1]:
        s = int(s)
        if s in graph.segment_to_idx:
            idx.append(graph.segment_to_idx[s])
        else:
            idx.append(0)
            missing += 1
    return np.asarray(idx, dtype=np.int64), missing
