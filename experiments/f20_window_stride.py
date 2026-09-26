"""
F20 — Pencere uzunlugu ve adim araliginin etkisi (degerlendirme: %80 ortusme).

Makale butun sonuclari tau = 50, adim = 10 ile uretiyor; ardisik iki pencere
50 adimin 40'ini paylasiyor, yani %80 ortusme var. Bu secim gerekcelendirilmis
degildi ve iki ayri soruyu birden etkiliyor:

  1. MODELE NE KADAR GECMIS VERILIYOR (tau).
  2. GOZLEMLER NE KADAR BAGIMLI (adim / tau orani). Ortusme buyudukce
     pencere sayisi artar ama ETKIN ornek sayisi artmaz; pencere duzeyindeki
     her test bundan etkilenir.

Burada tau in {25, 50, 100} x adim in {10, 25, 50} taranir. Her hucre icin
ortusme orani 1 - adim/tau olarak raporlanir; adim > tau olan hucre yoktur
(ortusme negatif olmaz, pencereler arasinda bosluk olur — bu da bilgi kaybi
oldugu icin ayrica isaretlenir).

PROTOKOL
--------
tau ve adim protokol damgasinin icindedir; her hucre KENDI protokol ozetine
ve kendi kosu deposuna yazilir. Bu sayede farkli pencere ayarlarinin sonuclari
kazara ayni tabloda karisamaz.

Karsilastirma birimi TRIP'tir ve trip sayisi (90) pencere ayarindan bagimsiz
oldugu icin hucreler arasi karsilastirma gecerlidir; pencere sayisi ise
degisir ve raporlanir.

KULLANIM
  python experiments/f20_window_stride.py --pilot
  python experiments/f20_window_stride.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import nadeau_bengio
from stanet.data import build_cv_folds
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F20_window_stride.json"

TAUS = [25, 50, 100]
STRIDES = [10, 25, 50]
SEEDS = [42]
MERGE = False
CONFIGS = {
    "hybrid_gat": dict(temporal="lstm", spatial="gat",  fusion="weighted_sum"),
    "lstm_only":  dict(temporal="lstm", spatial="none", fusion="weighted_sum"),
}


def cell(tau, stride, configs, seeds, n_folds, graph, device):
    """Bir (tau, adim) hucresini kosar. C sabitleri gecici olarak degisir."""
    old = (C.SEQUENCE_LENGTH, C.STEP_SIZE)
    C.SEQUENCE_LENGTH, C.STEP_SIZE = tau, stride
    try:
        stamp = protocol_stamp()                  # yeni tau/adim ile yeni ozet
        folds = build_cv_folds(n_folds=n_folds, legacy=False)
        idx = {f.info["fold"]: {sp: map_segments(getattr(f, "meta_" + sp), graph)[0]
                                for sp in ("train", "val", "test")} for f in folds}
        n_win = {sp: int(sum(len(getattr(f, "y_" + sp)) for f in folds))
                 for sp in ("train", "val", "test")}
        out = {"tau": tau, "stride": stride,
               "overlap": 1.0 - stride / tau,
               "gap": stride > tau,
               "protocol_hash": stamp["hash"],
               "n_windows": n_win,
               "n_trips": folds[0].info["trips"],
               "models": {}}
        for name, cfg in configs.items():
            runs = []
            for f in folds:
                k = f.info["fold"]
                for seed in seeds:
                    spec = make_spec(merge_by_osm=MERGE, fold=k, cv=n_folds, **cfg)
                    r = get_or_train(spec, seed, f, idx[k], graph, device,
                                     verbose=False)
                    m = dict(r["metrics"])
                    m.update({"fold": k, "seed": seed})
                    runs.append(m)
                    print("    %-12s fold%d s%d: F1=%.4f%s"
                          % (name, k, seed, m["f1"],
                             "  [onbellek]" if r["cached"] else ""), flush=True)
            f1 = np.array([m["f1"] for m in runs])
            out["models"][name] = {
                "runs": runs, "f1_mean": float(f1.mean()),
                "f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "f1_min": float(f1.min()), "n": int(len(f1))}
        # uzamsal dalin katkisi bu hucrede
        if set(("hybrid_gat", "lstm_only")) <= set(out["models"]):
            a = {(m["fold"], m["seed"]): m["f1"] for m in out["models"]["hybrid_gat"]["runs"]}
            b = {(m["fold"], m["seed"]): m["f1"] for m in out["models"]["lstm_only"]["runs"]}
            keys = sorted(set(a) & set(b))
            if len(keys) > 1:
                fi = [f.info for f in folds]
                out["spatial_contribution"] = nadeau_bengio(
                    [a[k] - b[k] for k in keys],
                    n_train=int(np.mean([x["trips"]["train"] for x in fi])),
                    n_test=int(np.mean([x["trips"]["test"] for x in fi])))
        return out
    finally:
        C.SEQUENCE_LENGTH, C.STEP_SIZE = old


def main():
    pilot = "--pilot" in sys.argv
    taus, strides = (([50], [10, 50]) if pilot else (TAUS, STRIDES))
    configs = ({"hybrid_gat": CONFIGS["hybrid_gat"]} if pilot else CONFIGS)
    device = get_device()
    graph = build_graph(merge_by_osm=MERGE)
    grid = [(t, s) for t in taus for s in strides if s <= t]
    print("cihaz %s | %d hucre x %d konfig x %d fold x %d tohum = %d kosu\n"
          % (device, len(grid), len(configs), C.N_FOLDS, len(SEEDS),
             len(grid) * len(configs) * C.N_FOLDS * len(SEEDS)), flush=True)

    results = {"baseline_protocol": protocol_stamp(), "pilot": pilot,
               "grid": {"tau": taus, "stride": strides},
               "note": ("her hucre kendi protokol ozetine yazilir; tau ve adim "
                        "damganin icindedir, dolayisiyla hucreler karisamaz"),
               "cells": {}}
    for tau, stride in grid:
        print("HUCRE tau=%d adim=%d (ortusme %.0f%%)"
              % (tau, stride, 100 * (1 - stride / tau)), flush=True)
        c = cell(tau, stride, configs, SEEDS, C.N_FOLDS, graph, device)
        results["cells"]["tau%d_stride%d" % (tau, stride)] = c
        print("  -> pencere: egitim %d / test %d | %s\n"
              % (c["n_windows"]["train"], c["n_windows"]["test"],
                 " | ".join("%s %.4f±%.4f" % (n, m["f1_mean"], m["f1_sd"])
                            for n, m in c["models"].items())), flush=True)

    print("=" * 96)
    print("PENCERE / ADIM DUYARLILIGI")
    print("=" * 96)
    print("  %-14s %8s %9s %9s %s"
          % ("hucre", "ortusme", "test pen.", "protokol",
             " | ".join("%-16s" % n for n in configs)))
    for key, c in results["cells"].items():
        print("  %-14s %7.0f%% %9d %9s %s"
              % (key, 100 * c["overlap"], c["n_windows"]["test"],
                 c["protocol_hash"][:8],
                 " | ".join("%.4f±%.4f  " % (m["f1_mean"], m["f1_sd"])
                            for m in c["models"].values())))
    if not pilot:
        print("\n  UZAMSAL DALIN KATKISI (hybrid_gat - lstm_only, NB duzeltmeli)")
        print("  %-14s %9s %24s %8s" % ("hucre", "ort fark", "%95 GA", "p"))
        for key, c in results["cells"].items():
            sc = c.get("spatial_contribution")
            if sc:
                print("  %-14s %+9.4f [%+.4f, %+.4f] %8.4f"
                      % (key, sc["mean_diff"], sc["ci_lo"], sc["ci_hi"],
                         sc["p_corrected"]))

    path = (C.RESULTS_DIR / "F20_pilot.json") if pilot else OUT
    save_json(results, path)
    print("\nkaydedildi: %s" % path)


if __name__ == "__main__":
    main()
