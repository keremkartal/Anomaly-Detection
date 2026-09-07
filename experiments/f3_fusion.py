"""
F3 — Fuzyon ablasyonu (Hakem 1, Madde 3).

Dort fuzyon mekanizmasi; kodlayicilar, projeksiyon boyutu, siniflandirici,
optimizer, butce, checkpoint kriteri ve esik secimi BIREBIR ayni.
Tek degisen: fuzyon operatoru.

REVIZYON (v2): egitim artik `stanet.runstore` uzerinden yapiliyor.
`lstm+gat+weighted_sum` konfigurasyonu F4 (T:lstm), F1b (hibrit) ve F2
(trip-instance) ile PAYLASILIR — dort tablo ayni diskteki kosuyu okur,
dolayisiyla bit-ayni sayi verir. Protokol ozeti JSON'a yazilir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.data import build_dataset
from stanet.graph import build_graph, map_segments
from stanet.hybrid import FUSION_TYPES
from stanet.runstore import make_spec, run_seeds
from stanet.stats import aggregate_seeds, bootstrap_diff, mcnemar
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
SEEDS = C.SEEDS          # 5 seed
MERGE = False            # F2 karari: trip-instance graf (bkz. BULGULAR F2-2)


def main():
    device = get_device()
    ds = build_dataset(legacy=False)
    graph = build_graph(merge_by_osm=MERGE)
    tr, _ = map_segments(ds.meta_train, graph)
    va, _ = map_segments(ds.meta_val, graph)
    te, _ = map_segments(ds.meta_test, graph)
    nidx = {"train": tr, "val": va, "test": te}

    stamp = protocol_stamp()
    print(f"cihaz {device} | graf {graph.info['num_nodes']} dugum "
          f"(merge={MERGE}) | {len(SEEDS)} seed x {len(FUSION_TYPES)} varyant")
    print(f"protokol {stamp['hash']}: {C.MAX_EPOCHS} epoch, patience {C.PATIENCE}, "
          f"select_by=val_f1, esik val'de seciliyor, "
          f"cudnn.deterministic={stamp['cudnn_deterministic']}\n", flush=True)

    results = {"protocol": stamp,
               "config": {"seeds": SEEDS, "merge_by_osm": MERGE,
                          "graph_tag": "trip_instance", "select_by": "val_f1"},
               "graph_info": graph.info, "variants": {}}
    preds = {}

    for fusion in FUSION_TYPES:
        spec = make_spec("lstm", "gat", fusion, merge_by_osm=MERGE)
        runs, pmat = run_seeds(spec, SEEDS, ds, nidx, graph, device)
        agg = aggregate_seeds(runs)
        results["variants"][fusion] = {
            "spec": spec, "runs": runs, "aggregate": agg,
            "fusion_params": runs[0]["fusion_params"],
            "total_params": runs[0]["total_params"],
            "gate_range_mean": float(np.mean([r["gate"]["range"] for r in runs])),
            "gate_class_delta_mean": float(np.mean(
                [r["gate"]["mean_anomaly"] - r["gate"]["mean_normal"] for r in runs])),
        }
        preds[fusion] = pmat
        print(f"  -> {fusion}: F1 {agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f} | "
              f"AUC {agg['roc_auc']['mean']:.4f}+-{agg['roc_auc']['std']:.4f}\n",
              flush=True)

    # -------------------------------------------------- gated vs digerleri
    print("=" * 70)
    print("GATED vs DIGER FUZYONLAR (medyan seed, eslestirilmis)")
    print("=" * 70)
    y_test = ds.y_test
    med = len(SEEDS) // 2
    g_prob = preds["gated"][med]
    g_th = results["variants"]["gated"]["runs"][med]["threshold_val"]
    comp = {}
    for f in FUSION_TYPES:
        if f == "gated":
            continue
        o_prob = preds[f][med]
        mc = mcnemar(y_test, (g_prob > g_th).astype(int),
                     (o_prob > results["variants"][f]["runs"][med]["threshold_val"]
                      ).astype(int))
        bd = bootstrap_diff(
            y_test, g_prob, o_prob, threshold=g_th,
            threshold_b=results["variants"][f]["runs"][med]["threshold_val"], seed=0)
        gm = results["variants"]["gated"]["aggregate"]["f1"]
        om = results["variants"][f]["aggregate"]["f1"]
        # NOT: mean_delta cok-seed ortalamalarindan, bootstrap_diff medyan
        # seed'den gelir. Ikisi FARKLI niceliktir; makalede ayri sutunlarda
        # ve acik etiketle raporlanir (bkz. F1b Tablo 10 duzeltmesi).
        comp[f] = {"mcnemar": mc, "bootstrap_diff": bd,
                   "mean_f1_gated": gm["mean"], "mean_f1_other": om["mean"],
                   "mean_delta": gm["mean"] - om["mean"],
                   "median_seed_delta": float(
                       results["variants"]["gated"]["runs"][med]["f1"]
                       - results["variants"][f]["runs"][med]["f1"]),
                   "median_seed_index": med}
        print(f"  gated vs {f:16s} dF1(ort)={comp[f]['mean_delta']:+.4f}  "
              f"dF1(medyan seed)={comp[f]['median_seed_delta']:+.4f}  "
              f"boot GA[{bd['lo']:+.4f},{bd['hi']:+.4f}]  McNemar p={mc['p_value']:.4f} "
              f"{'ANLAMLI' if mc['significant'] else 'anlamsiz'}", flush=True)
    results["gated_vs_others"] = comp

    # -------------------------------------------------- siralama
    order = sorted(FUSION_TYPES,
                   key=lambda f: -results["variants"][f]["aggregate"]["f1"]["mean"])
    results["ranking"] = order
    print("\n" + "=" * 70)
    print(f"SIRALAMA ({len(SEEDS)} seed ortalamasi)")
    print("=" * 70)
    print(f"  {'fuzyon':18s} {'F1':>16s} {'ROC-AUC':>16s} {'fuz.par':>9s} {'gate araligi':>13s}")
    for f in order:
        v = results["variants"][f]
        a = v["aggregate"]
        print(f"  {f:18s} {a['f1']['mean']:.4f}+-{a['f1']['std']:.4f}  "
              f"{a['roc_auc']['mean']:.4f}+-{a['roc_auc']['std']:.4f}  "
              f"{v['fusion_params']:>9,} {v['gate_range_mean']:>13.4f}")
    results["winner"] = order[0]
    results["gated_rank"] = order.index("gated") + 1
    worst = {f: min(r["f1"] for r in results["variants"][f]["runs"])
             for f in FUSION_TYPES}
    results["worst_seed_f1"] = worst
    print(f"\n  kazanan: {order[0]} | gated sirasi: {order.index('gated')+1}/4")
    print(f"  seed bazinda en dusuk F1: " +
          "  ".join(f"{f}={v:.4f}" for f, v in worst.items()))

    save_json(results, C.RESULTS_DIR / "F3_fusion.json")
    np.savez_compressed(C.CACHE_DIR / "F3_predictions.npz", y_test=y_test,
                        atype=ds.type_test,
                        **{f: preds[f] for f in FUSION_TYPES})
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F3_fusion.json'} "
          f"(protokol {stamp['hash']})")


if __name__ == "__main__":
    main()
