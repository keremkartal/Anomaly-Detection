"""
F2 — Graf onarimi.

Ayni fiziksel yolu (osm_segment_id) tek dugume indirir ve etkiyi olcer.
Ayrica epoch suresini olcer (F3/F4/F5 butce planlamasi icin).

REVIZYON (v2), makalenin Tablo 15'ine yonelik iki elestiriyi kapatir:

1. FUZYON ETIKETI. Ilk surumde bu tablo yalnizca "Hybrid" diyordu; oysa
   calistirilan model `gated` fuzyonuydu. Tablo 12'nin `weighted_sum`
   satiriyla karsilastirilinca "ayni model, farkli sonuc" gibi gorunuyordu.
   Artik HER IKI fuzyon da her iki grafta kosuluyor ve tabloya fuzyon
   sutunu giriyor.

2. MODEL SINIFI. Ilk surumde F2 `models.FusionHybrid`, F3/F4 ise
   `hybrid.GenericHybrid` kullaniyordu. Ikisi mimari olarak ayni ama
   parametre ISIMLERI farkli oldugu icin ayni seed'de farkli baslangic
   agirliklari cekiliyordu. Artik tek sinif: GenericHybrid, `runstore`
   uzerinden. trip_instance/weighted_sum kosusu F3, F4 ve F1b ile
   birebir paylasilir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stanet import config as C
from stanet.data import build_dataset
from stanet.graph import build_graph, map_segments
from stanet.runstore import make_spec, run_seeds
from stanet.stats import aggregate_seeds
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
SEEDS = C.SEEDS          # v2: F3/F4/F1b ile ayni 5 seed

# (etiket, spec kwargs) — gat_only uzamsal-only kontrol, digerleri hibrit
VARIANTS = [
    ("gat_only",     dict(arch="gat_only")),
    ("weighted_sum", dict(temporal="lstm", spatial="gat", fusion="weighted_sum")),
    ("gated",        dict(temporal="lstm", spatial="gat", fusion="gated")),
]


def node_idx_for(ds, graph):
    tr, m1 = map_segments(ds.meta_train, graph)
    va, m2 = map_segments(ds.meta_val, graph)
    te, m3 = map_segments(ds.meta_test, graph)
    return {"train": tr, "val": va, "test": te}, m1 + m2 + m3


def label_ambiguity(ds, graph):
    """Ayni dugume dusen sekanslarin etiket tutarliligi — GAT'in teorik tavani."""
    rows = []
    for sp in ("train", "val", "test"):
        idx, _ = map_segments(getattr(ds, f"meta_{sp}"), graph)
        rows.append(pd.DataFrame({"node": idx, "y": getattr(ds, f"y_{sp}")}))
    df = pd.concat(rows)
    agg = df.groupby("node")["y"].agg(["mean", "count"])
    mixed = agg[(agg["mean"] > 0) & (agg["mean"] < 1)]
    ceiling = float((agg["mean"].apply(lambda p: max(p, 1 - p)) * agg["count"]).sum()
                    / agg["count"].sum())
    return {"nodes_used": int(len(agg)), "mixed_label_nodes": int(len(mixed)),
            "seq_share_in_mixed": float(mixed["count"].sum() / agg["count"].sum()),
            "theoretical_max_accuracy": ceiling}


def main():
    device = get_device()
    ds = build_dataset(legacy=False)   # sizintisi duzeltilmis hat
    stamp = protocol_stamp()
    print(f"cihaz: {device} | veri: legacy=False (scaler yalnizca train)")
    print(f"protokol {stamp['hash']} | {len(SEEDS)} seed x {len(VARIANTS)} varyant "
          f"x 2 graf\n", flush=True)

    results = {"protocol": stamp, "config": {"seeds": SEEDS},
               "dataset_info": ds.info, "graphs": {}, "runs": {}}

    for merge in (False, True):
        tag = "merged" if merge else "original"
        graph = build_graph(merge_by_osm=merge)
        nidx, missing = node_idx_for(ds, graph)
        amb = label_ambiguity(ds, graph)
        results["graphs"][tag] = {**graph.info, "unmapped": missing,
                                  "ambiguity": amb,
                                  "graph_tag": "osm_merged" if merge else "trip_instance"}

        print(f"=== graf: {tag} ===")
        print(f"  dugum {graph.info['num_nodes']} | kenar {graph.info['num_edges']} "
              f"| ort.derece {graph.info['avg_degree']:.2f}")
        print(f"  kullanilan dugum {amb['nodes_used']} | karisik etiketli "
              f"{amb['mixed_label_nodes']} | bu dugumlerdeki sekans payi "
              f"%{amb['seq_share_in_mixed']*100:.1f}")
        print(f"  yol kimliginden teorik en iyi dogruluk: "
              f"%{amb['theoretical_max_accuracy']*100:.2f}", flush=True)

        for label, kw in VARIANTS:
            spec = make_spec(merge_by_osm=merge, **kw)
            runs, _ = run_seeds(spec, SEEDS, ds, nidx, graph, device)
            agg = aggregate_seeds(runs)
            results["runs"][f"{tag}/{label}"] = {
                "spec": spec, "runs": runs, "aggregate": agg,
                "fusion": kw.get("fusion", "-"),
                "model_class": "GenericHybrid" if "fusion" in kw else "GATOnly",
                # geriye donuk uyumluluk (makale/_verify_numbers.py bu adlari okur)
                "f1_mean": agg["f1"]["mean"], "f1_std": agg["f1"]["std"],
                "auc_mean": agg["roc_auc"]["mean"], "auc_std": agg["roc_auc"]["std"],
                "sec_per_epoch": float(np.mean([r["sec_per_epoch"] for r in runs])),
            }
            print(f"    -> {label:13s} F1 {agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f} | "
                  f"AUC {agg['roc_auc']['mean']:.4f}+-{agg['roc_auc']['std']:.4f}",
                  flush=True)
        print()

    # ---- ozet
    print("=" * 74)
    print("F2 OZET")
    print("=" * 74)
    print(f"  {'graf/varyant':26s} {'fuzyon':14s} {'F1':>16s} {'ROC-AUC':>16s}")
    for k, v in results["runs"].items():
        print(f"  {k:26s} {v['fusion']:14s} {v['f1_mean']:.4f}+-{v['f1_std']:.4f}  "
              f"{v['auc_mean']:.4f}+-{v['auc_std']:.4f}")

    g_o = results["runs"]["original/gat_only"]["auc_mean"]
    g_m = results["runs"]["merged/gat_only"]["auc_mean"]
    results["karar_B_gat_only"] = {"auc_original": g_o, "auc_merged": g_m,
                                   "delta": g_m - g_o}
    print(f"\n  GAT-only AUC: {g_o:.4f} (trip-instance) -> {g_m:.4f} (osm-merged) "
          f"| D={g_m-g_o:+.4f}")

    # hibrit tarafinda graf etkisi — fuzyon sabit tutularak
    results["graph_effect_by_fusion"] = {}
    for fus in ("weighted_sum", "gated"):
        o = results["runs"][f"original/{fus}"]["f1_mean"]
        m = results["runs"][f"merged/{fus}"]["f1_mean"]
        results["graph_effect_by_fusion"][fus] = {"trip_instance": o,
                                                  "osm_merged": m, "delta": m - o}
        print(f"  hibrit ({fus:12s}) F1: {o:.4f} -> {m:.4f} | D={m-o:+.4f}")
    print("  [KARAR B'nin asil testi F5'te tip-1 stratifikasyonu ile yapilir]")

    spe = np.mean([v["sec_per_epoch"] for v in results["runs"].values()])
    print(f"\n  ortalama epoch suresi ~ {spe:.1f} s")

    save_json(results, C.RESULTS_DIR / "F2_graph.json")
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F2_graph.json'} (protokol {stamp['hash']})")


if __name__ == "__main__":
    main()
