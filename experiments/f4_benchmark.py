"""
F4 — Benchmark (Hakem 1 Madde 1 + Hakem 2 Madde 1).

Hakemler STGCN, DCRNN ve Graph WaveNet ile karşılaştırma istedi. Bu mimariler
"her düğümde sürekli zaman serisi -> tüm düğümlerde ortak tahmin" formülasyonu
için tasarlanmıştır; buradaki görev (tek aracın 50 adımlık penceresi -> tek
segment, ikili sınıflandırma) yapısal olarak farklıdır ve doğrudan uygulanamaz.

Adaptasyon stratejisi (makalede belgelenecek): her mimarinin ÇEKİRDEK OPERATÖRÜ
izole edilip aynı görev, aynı bütçe, aynı split ve aynı sınıflandırma başlığı
altında karşılaştırılır. İki eksen:

  A) Zamansal kodlayıcı ekseni  (uzamsal = GAT)
     lstm | gru | stgcn_temporal | wavenet_temporal | transformer_temporal
  B) Uzamsal kodlayıcı ekseni   (zamansal = LSTM)
     gat | cheb (STGCN) | diffusion (DCRNN) | adaptive (Graph WaveNet) | none

F3 bulgusu: gated füzyon 5 seed'in 1'inde çöküyor. Kodlayıcı farklarını izole
edebilmek için en KARARLI füzyon operatörü kullanılır (weighted_sum: std 0,0138,
çökme 0/5). Bu seçim makalede belgelenecek.

KULLANIM
  python experiments/f4_benchmark.py            # eksik konfigleri sirayla
  python experiments/f4_benchmark.py T:gru      # tek konfig
  python experiments/f4_benchmark.py --report   # sadece rapor
Her konfig bittiginde sonuc diske yazilir; surec duserse kaldigi yerden devam eder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.data import build_dataset
from stanet.evaluate import compute_metrics, metrics_by_type, pick_threshold, predict
from stanet.graph import build_graph, map_segments
from stanet.runstore import make_spec, run_seeds
from stanet.stats import aggregate_seeds, bootstrap_diff, mcnemar
from stanet.utils import (get_device, init_console, load_json, protocol_stamp,
                          save_json)

init_console()
SEEDS = C.SEEDS          # v2: F3/F1b/F2 ile ayni 5 seed
MERGE = False
FUSION = "weighted_sum"

OUT = C.RESULTS_DIR / "F4_benchmark.json"
PRED = C.CACHE_DIR / "F4_predictions.npz"

# konfig adi -> (temporal, spatial, eksen, koken)
CONFIGS = {
    "T:lstm":        ("lstm", "gat", "temporal", "STANet (onerilen)"),
    "T:gru":         ("gru", "gat", "temporal", "DCRNN tekrarlayan omurga"),
    "T:stgcn":       ("stgcn_temporal", "gat", "temporal", "STGCN gated TCN"),
    "T:wavenet":     ("wavenet_temporal", "gat", "temporal", "Graph WaveNet dilated conv"),
    "T:transformer": ("transformer_temporal", "gat", "temporal", "Transformer encoder"),
    "S:cheb":        ("lstm", "cheb", "spatial", "STGCN Chebyshev graf konv."),
    "S:diffusion":   ("lstm", "diffusion", "spatial", "DCRNN cift yonlu difuzyon"),
    "S:adaptive":    ("lstm", "adaptive", "spatial", "Graph WaveNet uyarlanabilir A"),
    "S:none":        ("lstm", "none", "spatial", "uzamsal dal yok (kontrol)"),
}


def load_state():
    stamp = protocol_stamp()
    res = load_json(OUT) if OUT.exists() else None
    # Protokol degistiyse eski sonuclari TASIMA — sessiz karisimin kaynagi budur.
    if res is not None and res.get("protocol", {}).get("hash") != stamp["hash"]:
        print(f"  [protokol degisti: {res.get('protocol', {}).get('hash')} -> "
              f"{stamp['hash']}] eski F4 sonuclari yok sayiliyor", flush=True)
        res = None
    if res is None:
        res = {"protocol": stamp,
               "config": {"seeds": SEEDS, "merge_by_osm": MERGE, "fusion": FUSION,
                          "graph_tag": "trip_instance"},
               "configs": {}}
        preds = {}
    else:
        preds = {k: v for k, v in np.load(PRED).items()} if PRED.exists() else {}
    res["protocol"] = stamp
    res["config"]["seeds"] = SEEDS
    return res, preds


def run_one(name, ds, nidx, graph, device):
    temporal, spatial, axis, origin = CONFIGS[name]
    # T:lstm -> lstm+gat+weighted_sum: F3 'weighted_sum', F1b 'hibrit' ve
    # F2 'trip_instance' ile AYNI kosu. Depo tek egitimi dort yerde paylasir.
    spec = make_spec(temporal, spatial, FUSION, merge_by_osm=MERGE)
    runs, pmat = run_seeds(spec, SEEDS, ds, nidx, graph, device)
    agg = aggregate_seeds(runs)
    print(f"  -> {name}: F1 {agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f} | "
          f"AUC {agg['roc_auc']['mean']:.4f}+-{agg['roc_auc']['std']:.4f}\n", flush=True)
    entry = {"spec": spec, "runs": runs, "aggregate": agg,
             "temporal": temporal, "spatial": spatial,
             "axis": axis, "origin": origin, "total_params": runs[0]["total_params"],
             "branch_params": runs[0].get("branch_params")}
    return entry, pmat


def report(res, ds):
    rows = []
    for name, v in res["configs"].items():
        a = v["aggregate"]
        rows.append((name, a["f1"]["mean"], a["f1"]["std"], a["roc_auc"]["mean"],
                     a["roc_auc"]["std"], v["total_params"], v["origin"], v["axis"]))
    rows.sort(key=lambda r: -r[1])
    print("=" * 90)
    print(f"F4 BENCHMARK — {len(rows)} konfigurasyon, {len(SEEDS)} seed, "
          f"fuzyon={FUSION}, graf={'birlestirilmis' if MERGE else 'orijinal'}")
    print("=" * 90)
    print(f"  {'konfig':16s} {'eksen':9s} {'F1':>17s} {'ROC-AUC':>17s} {'param':>10s}  koken")
    for n, f1m, f1s, am, as_, p, org, ax in rows:
        print(f"  {n:16s} {ax:9s} {f1m:.4f}+-{f1s:.4f}  {am:.4f}+-{as_:.4f} "
              f"{p:>10,}  {org}")
    res["ranking"] = [r[0] for r in rows]
    res["winner"] = rows[0][0] if rows else None

    if PRED.exists() and "T:lstm" in res["configs"]:
        preds = {k: v for k, v in np.load(PRED).items()}
        med = len(SEEDS) // 2
        ref = preds["T:lstm"][med]
        ref_th = res["configs"]["T:lstm"]["runs"][med]["threshold_val"]
        comp = {}
        print(f"\n  STANet (T:lstm) referansina gore (medyan seed, eslestirilmis):")
        for n in res["ranking"]:
            if n == "T:lstm" or n not in preds:
                continue
            o = preds[n][med]
            o_th = res["configs"][n]["runs"][med]["threshold_val"]
            mc = mcnemar(ds.y_test, (ref > ref_th).astype(int), (o > o_th).astype(int))
            bd = bootstrap_diff(ds.y_test, ref, o, threshold=ref_th,
                                threshold_b=o_th, seed=0)
            # mean_delta = cok-seed ortalama farki (nokta tahmini, N seed)
            # median_seed_delta = bootstrap GA'nin uzerinde hesaplandigi FARKLI
            # nicelik (tek seed). Ikisi ayni satirda karistirilmamalidir —
            # ilk surumde RF icin GA [+0,038,+0,116] ortalamayi (+0,0137)
            # icermiyordu, sebebi tam olarak buydu.
            comp[n] = {"mcnemar": mc, "bootstrap_diff": bd,
                       "mean_delta": (res["configs"]["T:lstm"]["aggregate"]["f1"]["mean"]
                                      - res["configs"][n]["aggregate"]["f1"]["mean"]),
                       "median_seed_delta": float(
                           res["configs"]["T:lstm"]["runs"][med]["f1"]
                           - res["configs"][n]["runs"][med]["f1"]),
                       "median_seed_index": med}
            print(f"    vs {n:16s} dF1(ort)={comp[n]['mean_delta']:+.4f} "
                  f"dF1(medyan)={comp[n]['median_seed_delta']:+.4f} "
                  f"GA[{bd['lo']:+.4f},{bd['hi']:+.4f}] p={mc['p_value']:.4f} "
                  f"{'ANLAMLI' if mc['significant'] else 'anlamsiz'}")
        res["stanet_vs_others"] = comp
    return res


def main():
    args = list(sys.argv[1:])
    only_report = "--report" in args
    targets = [a for a in args if a in CONFIGS]

    device = get_device()
    ds = build_dataset(legacy=False)
    graph = build_graph(merge_by_osm=MERGE)
    tr, _ = map_segments(ds.meta_train, graph)
    va, _ = map_segments(ds.meta_val, graph)
    te, _ = map_segments(ds.meta_test, graph)
    nidx = {"train": tr, "val": va, "test": te}

    res, preds = load_state()
    if not only_report:
        todo = targets or [k for k in CONFIGS if k not in res["configs"]]
        print(f"cihaz {device} | graf {graph.info['num_nodes']} dugum | "
              f"fuzyon={FUSION} | yapilacak: {todo}\n", flush=True)
        for name in todo:
            entry, pmat = run_one(name, ds, nidx, graph, device)
            res["configs"][name] = entry
            preds[name] = pmat
            save_json(res, OUT)
            np.savez_compressed(PRED, y_test=ds.y_test, atype=ds.type_test, **preds)
            print(f"  [kaydedildi: {name}]\n", flush=True)

    res = report(res, ds)
    save_json(res, OUT)
    print(f"\nkaydedildi: {OUT}")


if __name__ == "__main__":
    main()
