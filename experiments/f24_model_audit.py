"""
F24 — Mimari tablolarini KODDAN uret.

Denklem 15 ile Tablo 5 arasindaki uyusmazlik degerlendirmede isaretlendi:
metin tek katmanli bir kapi gosteriyordu, kod iki katmanliydi. O hata elle
yazilmis bir mimari ayrintiydi ve hicbir kontrol onu goremezdi; sayi
dogrulayici yalnizca SONUC dosyalarina bakiyordu, mimariye degil.

Bu betik modelleri gercekten kurar, katman boyutlarini ve parametre
sayilarini olcer ve `F24_model_audit.json` icine yazar. Boylece Tablo 3, 4
ve 5'teki her sayi da denetlenebilir hale gelir: makale ile kod arasinda
sessiz bir ayrisma bir daha mumkun olmaz.

KULLANIM
  python experiments/f24_model_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn as nn

from stanet import config as C
from stanet.graph import build_graph
from stanet.hybrid import GenericHybrid
from stanet.utils import init_console, save_json

init_console()
OUT = C.RESULTS_DIR / "F24_model_audit.json"
FUSIONS = ["gated", "concat", "weighted_sum", "cross_attention"]


def nparam(mod):
    return int(sum(p.numel() for p in mod.parameters()))


def layer_map(model):
    """Her adlandirilmis katmanin turu ve boyutlari."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out[name] = {"type": "Linear", "in": mod.in_features,
                         "out": mod.out_features,
                         "params": nparam(mod)}
        elif isinstance(mod, nn.LSTM):
            out[name] = {"type": "LSTM", "in": mod.input_size,
                         "hidden": mod.hidden_size, "layers": mod.num_layers,
                         "dropout": float(mod.dropout), "params": nparam(mod)}
        elif mod.__class__.__name__ == "GATConv":
            out[name] = {"type": "GATConv", "in": mod.in_channels,
                         "out": mod.out_channels, "heads": int(mod.heads),
                         "params": nparam(mod)}
        elif isinstance(mod, (nn.Tanh, nn.Sigmoid, nn.ReLU, nn.ELU)):
            out[name] = {"type": mod.__class__.__name__}
    return out


def main():
    g = build_graph(merge_by_osm=False)
    out = {
        "graph": {"num_nodes": g.num_nodes,
                  "num_edges": int(g.edge_index.shape[1]),
                  "num_features": g.num_features,
                  "num_unique_osm": g.info["num_unique_osm"],
                  "feature_names": g.feature_names},
        "models": {},
    }

    for fusion in FUSIONS:
        m = GenericHybrid("lstm", "gat", fusion,
                          num_node_features=g.num_features,
                          num_nodes=g.num_nodes)
        total = nparam(m)
        rec = {
            "total_params": total,
            "temporal_params": nparam(m.temporal),
            "spatial_params": nparam(m.spatial),
            "fusion_params": (m.fusion_parameters()
                              if hasattr(m, "fusion_parameters") else None),
            "temporal_share": nparam(m.temporal) / total,
            "spatial_share": nparam(m.spatial) / total,
            "layers": layer_map(m),
        }
        out["models"][fusion] = rec

    ref = out["models"]["gated"]
    print("GRAF     dugum %d | kenar %d | oznitelik %d | tekil yol %d"
          % (out["graph"]["num_nodes"], out["graph"]["num_edges"],
             out["graph"]["num_features"], out["graph"]["num_unique_osm"]))
    print("\nPARAMETRE (gated referans)")
    print("  toplam    %8d" % ref["total_params"])
    print("  zamansal  %8d  (%%%.1f)" % (ref["temporal_params"],
                                         100 * ref["temporal_share"]))
    print("  uzamsal   %8d  (%%%.1f)" % (ref["spatial_params"],
                                         100 * ref["spatial_share"]))
    print("\nFUZYON PARAMETRELERI")
    for f in FUSIONS:
        print("  %-16s %8s" % (f, out["models"][f]["fusion_params"]))
    print("\nKATMANLAR (gated)")
    for name, d in ref["layers"].items():
        if d["type"] == "Linear":
            print("  %-26s Linear(%d, %d)" % (name, d["in"], d["out"]))
        elif d["type"] == "LSTM":
            print("  %-26s LSTM(in=%d, hidden=%d, layers=%d, dropout=%.2f)"
                  % (name, d["in"], d["hidden"], d["layers"], d["dropout"]))
        elif d["type"] == "GATConv":
            print("  %-26s GATConv(in=%d, out=%d, heads=%d)"
                  % (name, d["in"], d["out"], d["heads"]))
        else:
            print("  %-26s %s" % (name, d["type"]))

    # kapi gercekten iki katmanli mi — Denklem 15'in dogrulugu
    gate = [n for n in ref["layers"] if n.startswith("attention_gate")]
    out["gate_is_two_layer"] = bool(
        sum(1 for n in gate if ref["layers"][n]["type"] == "Linear") == 2)
    out["gate_has_tanh"] = any(ref["layers"][n]["type"] == "Tanh" for n in gate)
    print("\nKAPI  iki katmanli: %s | tanh var: %s  (Denklem 15 boyle olmali)"
          % (out["gate_is_two_layer"], out["gate_has_tanh"]))

    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
