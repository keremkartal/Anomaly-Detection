"""
F28 — Revizyon icin uretilen veri kumelerinde ana karsilastirma.

AMAC
----
Makalenin merkezi bulgusu "hicbir mimari fark cozulemiyor" ve bu bulgu tek
bir veri kumesinden geliyordu. Burada ayni karsilastirma, bagimsiz bir yol
agi (Hamburg) ve iki farkli eksik-veri doldurma stratejisi uzerinde
tekrarlanir.

SIZINTI UYARISI — bu betigin ikinci amaci
-----------------------------------------
Yeni kumelerde `tunnel == 1` olan HER satirin etiketi 9'dur (olculen oran
0,9926-1,0000; zorunlu on isleme filtresi kalan istisnalari silerek bunu
tam 1,0000 yapar). `tunnel` bir GRAF DUGUM OZNITELIGIDIR, yani uzamsal
dala dogrudan girer. Kocaeli'de anomali 9 butun anomalilerin ~%60'i,
Hamburg'da ~%1.

Dolayisiyla Kocaeli'de grafa erisen bir model, anomalilerin yarisindan
fazlasini tek bir ikili oznitelikten OKUYABILIR. Bu ogrenilmis uzamsal
akil yurutme degil, deterministik bir sizintidir.

Betik her kumeyi IKI kurulumda kosar:

  tunnel_in    oznitelik grafta (verinin geldigi hali)
  tunnel_out   oznitelik graftan cikarilmis (sizintisiz kontrol)

Ikisi arasindaki fark sizintinin BUYUKLUGUNU olcer. Bu, makalenin kendi
konusuna dogrudan katki: sizmis tek bir oznitelik bir mimari sonucu ne
kadar sisirir, sayiyla gosterilmis olur.

KULLANIM
  python experiments/f28_new_datasets.py --list
  python experiments/f28_new_datasets.py hamburg_knn_n0_5_a5
  python experiments/f28_new_datasets.py hamburg_knn_n0_5_a5 --no-tunnel-control
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stanet import config as C
from stanet import datasets as D
from stanet.cluster_stats import holm_correction, nadeau_bengio, tost_equivalence
from stanet.data import build_cv_folds
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
SEEDS = [42, 43]
MERGE = False
DELTA = 0.02
COLLAPSE = 0.30
CONFIGS = {
    "hybrid_gat":      dict(temporal="lstm", spatial="gat",  fusion="weighted_sum"),
    "stanet_gated":    dict(temporal="lstm", spatial="gat",  fusion="gated"),
    "hybrid_concat":   dict(temporal="lstm", spatial="gat",  fusion="concat"),
    "hybrid_crossatt": dict(temporal="lstm", spatial="gat",  fusion="cross_attention"),
    "lstm_only":       dict(temporal="lstm", spatial="none", fusion="weighted_sum"),
}


def leak_report(path):
    """Veri kumesindeki tunnel-anomali bagintisini olcer."""
    d = pd.read_csv(path, usecols=["anomaly", "tunnel"])
    t1, a9 = d.tunnel == 1, d.anomaly == 9
    n_anom = int((d.anomaly != 0).sum())
    return {
        "n_rows": int(len(d)), "n_anomalous": n_anom,
        "n_tunnel_rows": int(t1.sum()),
        "p_anomaly9_given_tunnel": float((t1 & a9).sum() / max(t1.sum(), 1)),
        "p_tunnel_given_anomaly9": float((t1 & a9).sum() / max(a9.sum(), 1)),
        "share_of_anomalies_that_are_type9": float((a9).sum() / max(n_anom, 1)),
        "share_of_anomalies_on_tunnels": float((t1 & (d.anomaly != 0)).sum()
                                               / max(n_anom, 1)),
    }


def run_setting(name, drop_tunnel, device):
    """Bir veri kumesini bir kurulumda kosar; cell sozlugu doner."""
    info = D.use(name, drop_tunnel_feature=drop_tunnel)
    stamp = protocol_stamp()
    graph = build_graph(merge_by_osm=MERGE)
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    tag = "tunnel_out" if drop_tunnel else "tunnel_in"
    print("  kurulum %-11s protokol %s | graf %d dugum / %d oznitelik | "
          "%d fold" % (tag, stamp["hash"], graph.num_nodes, graph.num_features,
                       len(folds)), flush=True)

    fold_idx = {}
    for f in folds:
        k = f.info["fold"]
        fold_idx[k] = {sp: map_segments(getattr(f, "meta_" + sp), graph)[0]
                       for sp in ("train", "val", "test")}

    cells, runmap = {}, {}
    for cfg_name, cfg in CONFIGS.items():
        runs = []
        for f in folds:
            k = f.info["fold"]
            for seed in SEEDS:
                spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS, **cfg)
                r = get_or_train(spec, seed, f, fold_idx[k], graph, device,
                                 verbose=False)
                m = dict(r["metrics"])
                m.update({"fold": k, "seed": seed})
                runs.append(m)
                runmap.setdefault(cfg_name, {})[(k, seed)] = float(m["f1"])
                print("    %-11s %-16s f%d s%d: F1=%.4f%s"
                      % (tag, cfg_name, k, seed, m["f1"],
                         "  [onbellek]" if r["cached"] else ""), flush=True)
        v = np.array([m["f1"] for m in runs])
        cells[cfg_name] = {
            "runs": runs, "f1_mean": float(v.mean()),
            "f1_sd": float(v.std(ddof=1)), "f1_min": float(v.min()),
            "n": int(len(v)), "n_collapsed": int((v < COLLAPSE).sum())}
        print("    -> %-16s F1 %.4f+-%.4f  en kotu %.4f  cokme %d/%d\n"
              % (cfg_name, v.mean(), cells[cfg_name]["f1_sd"],
                 v.min(), cells[cfg_name]["n_collapsed"], len(v)), flush=True)

    fi = [f.info for f in folds]
    n_tr = int(np.mean([x["trips"]["train"] for x in fi]))
    n_te = int(np.mean([x["trips"]["test"] for x in fi]))
    comps, pv = {}, {}
    for a, b in itertools.combinations(CONFIGS, 2):
        keys = sorted(set(runmap[a]) & set(runmap[b]))
        diffs = [runmap[a][k] - runmap[b][k] for k in keys]
        nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
        ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
        comps[a + "_vs_" + b] = {"nadeau_bengio": nb, "tost": ts,
                                 "n_paired": len(keys)}
        pv[a + "_vs_" + b] = nb["p_corrected"]
    for k, v in holm_correction(pv).items():
        comps[k]["holm"] = v
    for k, v in comps.items():
        sig = v.get("holm", {}).get("significant")
        v["verdict"] = ("ESDEGER" if v["tost"]["equivalent"]
                        else ("FARKLI" if sig else "KARARSIZ"))

    return {"protocol": stamp, "dataset": name, "drop_tunnel_feature": drop_tunnel,
            "dataset_info": {k: str(v) for k, v in info.items()},
            "n_train_trips": n_tr, "n_test_trips": n_te,
            "fold_info": fi, "models": cells, "comparisons": comps}


def main():
    av = D.available()
    if "--list" in sys.argv:
        print("kullanilabilir veri kumeleri (%d):" % len(av))
        for k in sorted(av):
            print("  " + k)
        return 0
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not names:
        print("kullanim: f28_new_datasets.py <veri_kumesi> [...] "
              "[--no-tunnel-control]")
        return 1
    settings = [False] if "--no-tunnel-control" in sys.argv else [False, True]

    device = get_device()
    out = {"device": str(device), "seeds": SEEDS, "delta": DELTA,
           "datasets": {}}
    for name in names:
        if name not in av:
            print("ATLANDI, diskte yok: %s" % name)
            continue
        print("\n" + "=" * 96)
        print("VERI KUMESI: %s" % name)
        print("=" * 96, flush=True)
        lk = leak_report(av[name]["trajectory"])
        print("  SIZINTI OLCUMU")
        print("    P(anomali=9 | tunnel=1) = %.4f" % lk["p_anomaly9_given_tunnel"])
        print("    anomalilerin %%%.1f'i tip 9, %%%.1f'i tunel satirinda"
              % (100 * lk["share_of_anomalies_that_are_type9"],
                 100 * lk["share_of_anomalies_on_tunnels"]), flush=True)
        rec = {"leak": lk, "settings": {}}
        for drop in settings:
            tag = "tunnel_out" if drop else "tunnel_in"
            rec["settings"][tag] = run_setting(name, drop, device)
        out["datasets"][name] = rec

        # iki kurulum arasindaki fark = sizintinin bedeli
        if len(settings) == 2:
            print("  " + "-" * 92)
            print("  SIZINTININ BEDELI (tunnel_in eksi tunnel_out)")
            print("    %-18s %10s %10s %9s" % ("konfig", "tunnel_in", "tunnel_out", "fark"))
            diff = {}
            for cfg in CONFIGS:
                a = rec["settings"]["tunnel_in"]["models"][cfg]["f1_mean"]
                b = rec["settings"]["tunnel_out"]["models"][cfg]["f1_mean"]
                diff[cfg] = a - b
                print("    %-18s %10.4f %10.4f %+9.4f" % (cfg, a, b, a - b))
            spatial = [v for k, v in diff.items() if k != "lstm_only"]
            print("    uzamsal dalli modellerde ortalama sisme: %+.4f"
                  % float(np.mean(spatial)))
            print("    uzamsal dalsiz kontrolde (beklenen ~0)   : %+.4f"
                  % diff["lstm_only"])
            rec["leak_cost"] = {"per_config": diff,
                                "mean_spatial": float(np.mean(spatial)),
                                "control": diff["lstm_only"]}

    path = C.RESULTS_DIR / ("F28_%s.json" % "_".join(names)[:60])
    save_json(out, path)
    print("\nkaydedildi: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
