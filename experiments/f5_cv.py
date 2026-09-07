"""
F5 — Trip-bazli Group 5-fold CV: fuzyon ablasyonu + KARAR B + kalibrasyon.

Amaclar:
  1. Tek sabit bolmenin dengesizligini (test %3,27 vs train %7,90) ortadan kaldirmak
  2. Tum sekanslarin test tahmini almasini saglamak -> havuzlanmis alt-tip analizi
  3. KARAR B: tip-1 (hiz asimi) sekanslarinda uzamsal dal gercekten katki sagliyor mu?
  4. Kalibrasyon (Brier, ECE)

REVIZYON (v2) — hakemin iki elestirisini kapatir:

  (a) FUZYON ABLASYONU CV'DE. Ilk surumde dort fuzyonun karsilastirmasi
      YALNIZCA tek sabit bolmede yapilmisti (Tablo 12); CV'de yalnizca iki
      varyant vardi. Hakem hakli olarak "bulgu bolmeye mi ozgu?" diye sordu.
      Artik dort fuzyonun tamami 5 fold x 2 seed = 10 kosuda degerlendiriliyor.

  (b) ISTATISTIKSEL BIRIM. Pencereler %80 ortusuyor ve ayni tripin pencereleri
      bagimsiz degil. KARAR B artik pencere duzeyinde McNemar/bootstrap YERINE
      trip duzeyinde kume bootstrap + isaret-degistirme permutasyonu ve Holm
      duzeltmesi kullaniyor (bkz. stanet/cluster_stats.py). Pencere duzeyindeki
      degerler karsilastirma icin ayrica raporlanir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import (cluster_bootstrap_diff, cluster_permutation_test,
                                  cluster_summary, holm_correction)
from stanet.data import build_cv_folds
from stanet.evaluate import compute_metrics
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.stats import aggregate_seeds, bootstrap_diff, mcnemar
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
SEEDS = [42, 43]          # fold x seed = 10 gozlem / konfigurasyon
MERGE = False             # F2 karari: trip-instance graf

# Dort fuzyonun tamami + uzamsal-dalsiz kontrol.
CONFIGS = {
    "hybrid_gat":      dict(temporal="lstm", spatial="gat",  fusion="weighted_sum"),
    "stanet_gated":    dict(temporal="lstm", spatial="gat",  fusion="gated"),
    "hybrid_concat":   dict(temporal="lstm", spatial="gat",  fusion="concat"),
    "hybrid_crossatt": dict(temporal="lstm", spatial="gat",  fusion="cross_attention"),
    "lstm_only":       dict(temporal="lstm", spatial="none", fusion="weighted_sum"),
}
FUSION_MODELS = ["hybrid_gat", "stanet_gated", "hybrid_concat", "hybrid_crossatt"]
ISOLATION_PAIR = ("hybrid_gat", "lstm_only")   # uzamsal dal var / yok
COLLAPSE_THRESHOLD = 0.30                      # F1 < bu -> cokme sayilir


def calibration(y_true, y_prob, bins=10):
    """Brier skoru ve Expected Calibration Error."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    brier = float(np.mean((y_prob - y_true) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece, curve = 0.0, []
    for i in range(bins):
        m = (y_prob > edges[i]) & (y_prob <= edges[i + 1]) if i else (y_prob <= edges[1])
        if m.sum() == 0:
            continue
        conf, acc = float(y_prob[m].mean()), float(y_true[m].mean())
        ece += m.sum() / len(y_prob) * abs(conf - acc)
        curve.append({"bin": i, "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    return {"brier": brier, "ece": float(ece), "reliability": curve}


def main():
    device = get_device()
    graph = build_graph(merge_by_osm=MERGE)
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    stamp = protocol_stamp()
    print(f"cihaz {device} | {C.N_FOLDS}-fold trip-grouped CV | "
          f"graf {graph.info['num_nodes']} dugum | protokol {stamp['hash']}")
    print(f"{len(CONFIGS)} konfig x {C.N_FOLDS} fold x {len(SEEDS)} seed = "
          f"{len(CONFIGS)*C.N_FOLDS*len(SEEDS)} kosu\n", flush=True)
    for f in folds:
        i = f.info
        print(f"  fold {i['fold']}: trip {i['trips']} | sekans {i['sequences']} | "
              f"anomali oranlari "
              f"{ {k: round(v * 100, 2) for k, v in i['anomaly_ratio'].items()} }",
              flush=True)
    print()

    results = {"protocol": stamp,
               "config": {"n_folds": C.N_FOLDS, "seeds": SEEDS,
                          "merge_by_osm": MERGE, "graph_tag": "trip_instance",
                          "collapse_threshold": COLLAPSE_THRESHOLD},
               "fold_info": [f.info for f in folds], "models": {}}
    pooled = {}

    # fold basina dugum indeksleri ve trip kimlikleri bir kez
    fold_idx, fold_trips = {}, {}
    for f in folds:
        k = f.info["fold"]
        tr, _ = map_segments(f.meta_train, graph)
        va, _ = map_segments(f.meta_val, graph)
        te, _ = map_segments(f.meta_test, graph)
        fold_idx[k] = {"train": tr, "val": va, "test": te}
        fold_trips[k] = np.asarray(f.meta_test)[:, 0]      # META_COLS[0] = trip_id

    for name, cfg in CONFIGS.items():
        run_metrics, y_all, p_all, t_all, g_all = [], [], [], [], []
        for f in folds:
            k = f.info["fold"]
            for seed in SEEDS:
                spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS, **cfg)
                r = get_or_train(spec, seed, f, fold_idx[k], graph, device,
                                 verbose=False)
                m = dict(r["metrics"])
                m.update({"fold": k, "seed": seed})
                run_metrics.append(m)
                print(f"  {name:16s} fold{k} s{seed}: F1={m['f1']:.4f} "
                      f"AUC={m.get('roc_auc', float('nan')):.4f} "
                      f"n_pos={m['n_pos']}{'  [onbellek]' if r['cached'] else ''}",
                      flush=True)
                if seed == SEEDS[0]:      # havuzlama TEK seed'le (her sekans bir kez)
                    y_all.append(f.y_test)
                    p_all.append(r["prob_test"])
                    t_all.append(f.type_test)
                    g_all.append(fold_trips[k])

        y = np.concatenate(y_all); p = np.concatenate(p_all)
        t = np.concatenate(t_all); g = np.concatenate(g_all)
        agg = aggregate_seeds(run_metrics)
        th_pool = float(np.mean([m["threshold_val"] for m in run_metrics]))
        pooled_m = compute_metrics(y, p, th_pool)

        # alt-tip stratifikasyonu (havuzlanmis - tum veri test'e girdi)
        by_type, neg = {}, y == 0
        for tt in np.unique(t[y == 1]):
            mask = neg | ((y == 1) & (t == tt))
            by_type[C.ANOMALY_NAMES.get(int(tt), str(tt))] = \
                compute_metrics(y[mask], p[mask], th_pool)

        f1s = np.array([m["f1"] for m in run_metrics])
        results["models"][name] = {
            "config": cfg, "spec_base": {k2: v for k2, v in cfg.items()},
            "runs": run_metrics, "aggregate": agg,
            "pooled": pooled_m, "pooled_threshold": th_pool,
            "by_type_pooled": by_type, "calibration": calibration(y, p),
            "n_collapsed": int((f1s < COLLAPSE_THRESHOLD).sum()),
            "n_runs": int(len(f1s)), "worst_run_f1": float(f1s.min()),
            "pooled_from_seed": SEEDS[0],
        }
        pooled[name] = (y, p, t, g, th_pool)
        print(f"  -> {name}: kosu-ort F1 {agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f} | "
              f"havuz F1 {pooled_m['f1']:.4f} | cokme "
              f"{results['models'][name]['n_collapsed']}/{len(f1s)} | "
              f"Brier {results['models'][name]['calibration']['brier']:.4f} "
              f"ECE {results['models'][name]['calibration']['ece']:.4f}\n", flush=True)

    # ---------------------------------------------------- FUZYON ABLASYONU (CV)
    print("=" * 84)
    print(f"FUZYON ABLASYONU — CAPRAZ DOGRULAMA ({C.N_FOLDS} fold x {len(SEEDS)} seed "
          f"= {C.N_FOLDS*len(SEEDS)} kosu / varyant)")
    print("=" * 84)
    order = sorted(FUSION_MODELS,
                   key=lambda n: -results["models"][n]["aggregate"]["f1"]["mean"])
    print(f"  {'varyant':18s} {'F1 (ort+-std)':>18s} {'en kotu':>9s} {'cokme':>7s} "
          f"{'havuz F1':>9s} {'ECE':>8s}")
    for n in order:
        v = results["models"][n]
        a = v["aggregate"]["f1"]
        print(f"  {n:18s} {a['mean']:.4f}+-{a['std']:.4f}  {v['worst_run_f1']:>9.4f} "
              f"{v['n_collapsed']:>3d}/{v['n_runs']:<3d} {v['pooled']['f1']:>9.4f} "
              f"{v['calibration']['ece']:>8.4f}")
    results["fusion_ablation_cv"] = {
        "ranking": order,
        "gated_rank": order.index("stanet_gated") + 1,
        "winner": order[0],
        "n_runs_per_variant": C.N_FOLDS * len(SEEDS),
        "table": {n: {"f1_mean": results["models"][n]["aggregate"]["f1"]["mean"],
                      "f1_std": results["models"][n]["aggregate"]["f1"]["std"],
                      "worst": results["models"][n]["worst_run_f1"],
                      "n_collapsed": results["models"][n]["n_collapsed"],
                      "pooled_f1": results["models"][n]["pooled"]["f1"],
                      "ece": results["models"][n]["calibration"]["ece"]}
                  for n in FUSION_MODELS},
    }
    print(f"\n  CV'de kazanan: {order[0]} | gated sirasi: "
          f"{order.index('stanet_gated')+1}/{len(FUSION_MODELS)}")

    # gated vs digerleri — trip duzeyinde, Holm duzeltmeli
    yg, pg, _, gg, thg = pooled["stanet_gated"]
    perm_p = {}
    fus_stats = {}
    for n in FUSION_MODELS:
        if n == "stanet_gated":
            continue
        yo, po, _, go, tho = pooled[n]
        assert np.array_equal(gg, go), "havuzlama sirasi uyusmuyor"
        a_bin = (pg > thg).astype(int)
        b_bin = (po > tho).astype(int)
        cb = cluster_bootstrap_diff(yg, a_bin, b_bin, gg, seed=0)
        cp = cluster_permutation_test(yg, a_bin, b_bin, gg, seed=0)
        fus_stats[n] = {"cluster_bootstrap": cb, "cluster_permutation": cp,
                        "mean_delta": (results["models"]["stanet_gated"]["aggregate"]["f1"]["mean"]
                                       - results["models"][n]["aggregate"]["f1"]["mean"])}
        perm_p[n] = cp["p_value"]
    for n, h in holm_correction(perm_p).items():
        fus_stats[n]["holm"] = h
    results["fusion_ablation_cv"]["gated_vs_others_trip_level"] = fus_stats
    print(f"\n  gated vs digerleri (TRIP duzeyi, Holm duzeltmeli):")
    for n in FUSION_MODELS:
        if n == "stanet_gated":
            continue
        s = fus_stats[n]
        print(f"    vs {n:18s} dF1(havuz)={s['cluster_bootstrap']['observed_diff']:+.4f} "
              f"GA[{s['cluster_bootstrap']['lo']:+.4f},{s['cluster_bootstrap']['hi']:+.4f}] "
              f"p={s['cluster_permutation']['p_value']:.4f} "
              f"Holm={s['holm']['p_holm']:.4f} "
              f"{'ANLAMLI' if s['holm']['significant'] else 'anlamsiz'}")

    # ---------------------------------------------------- KARAR B (trip duzeyi)
    print("\n" + "=" * 84)
    print("KARAR B — Uzamsal dal, hiz asimi (tip 1) anomalilerinde katki sagliyor mu?")
    print("=" * 84)
    a_name, b_name = ISOLATION_PAIR
    ya, pa, ta, ga, tha = pooled[a_name]
    yb, pb, tb, gb, thb = pooled[b_name]
    assert np.array_equal(ya, yb) and np.array_equal(ta, tb), "fold siralari uyusmuyor"
    ci = cluster_summary(ga)
    print(f"  izolasyon: {a_name} (uzamsal var) vs {b_name} (uzamsal yok), "
          f"fuzyon her ikisinde de weighted_sum")
    print(f"  istatistiksel birim: TRIP ({ci['n_trips']} trip, {ci['n_windows']} pencere, "
          f"trip basina ort. {ci['windows_per_trip_mean']:.1f} pencere, "
          f"pencere ortusmesi %80)\n")

    karar, karar_p = {}, {}
    for tt, tname in [(1, "hiz_asimi"), (2, "uzun_duraklama"), (9, "sinyal_donmasi")]:
        mask = (ya == 0) | ((ya == 1) & (ta == tt))
        n_pos = int(((ya == 1) & (ta == tt)).sum())
        if n_pos == 0:
            continue
        ma = compute_metrics(ya[mask], pa[mask], tha)
        mb = compute_metrics(yb[mask], pb[mask], thb)
        abin = (pa[mask] > tha).astype(int)
        bbin = (pb[mask] > thb).astype(int)
        gm = ga[mask]
        cb = cluster_bootstrap_diff(ya[mask], abin, bbin, gm, seed=0)
        cp = cluster_permutation_test(ya[mask], abin, bbin, gm, seed=0)
        # pencere duzeyi degerler — ilk surumle karsilastirma icin, gecersiz
        # oldugu acikca etiketlenerek saklanir
        wl_mc = mcnemar(ya[mask], abin, bbin)
        wl_bd = bootstrap_diff(ya[mask], pa[mask], pb[mask], threshold=tha,
                               threshold_b=thb, seed=0)
        karar[tname] = {
            "n_pos": n_pos, "n_trips": cb["n_trips"],
            "hybrid_f1": ma["f1"], "lstm_only_f1": mb["f1"],
            "delta": ma["f1"] - mb["f1"],
            "hybrid_recall": ma["recall"], "lstm_recall": mb["recall"],
            "cluster_bootstrap": cb, "cluster_permutation": cp,
            "window_level_INVALID": {"mcnemar": wl_mc, "bootstrap_diff": wl_bd,
                                     "note": "pencereler %80 ortusur; yalnizca "
                                             "ilk surumle karsilastirma icin"},
        }
        karar_p[tname] = cp["p_value"]
        print(f"  {tname:18s} n={n_pos:4d} ({cb['n_trips']} trip)  "
              f"hybrid F1={ma['f1']:.4f}  LSTM-only F1={mb['f1']:.4f}  "
              f"d={ma['f1']-mb['f1']:+.4f}")
        print(f"  {'':18s}   TRIP  GA[{cb['lo']:+.4f},{cb['hi']:+.4f}] "
              f"p={cp['p_value']:.4f}")
        print(f"  {'':18s}   pencere (GECERSIZ) GA[{wl_bd['lo']:+.4f},"
              f"{wl_bd['hi']:+.4f}] p={wl_mc['p_value']:.4f}")

    holm = holm_correction(karar_p)
    for tname, h in holm.items():
        karar[tname]["holm"] = h
    print(f"\n  Holm duzeltmesi ({len(karar_p)} karsilastirma):")
    for tname in karar_p:
        h = karar[tname]["holm"]
        print(f"    {tname:18s} p={h['p_raw']:.4f} -> Holm {h['p_holm']:.4f} "
              f"{'ANLAMLI' if h['significant'] else 'anlamsiz'}")

    hz = karar.get("hiz_asimi", {})
    verdict = ("UZAMSAL_KATKI_VAR"
               if hz.get("delta", 0) > 0 and hz.get("holm", {}).get("significant", False)
               else "UZAMSAL_KATKI_YOK")
    karar["verdict"] = verdict
    karar["cluster_info"] = ci
    results["karar_B"] = karar
    print(f"\n  KARAR B: {verdict}  (trip duzeyi + Holm)")

    # ---------------------------------------------------- genel karsilastirma
    print("\n" + "=" * 84)
    print("HAVUZLANMIS SONUCLAR (tum sekanslar test tahmini aldi)")
    print("=" * 84)
    print(f"  {'model':18s} {'kosu-ort F1':>18s} {'havuz F1':>10s} {'AUC':>8s} "
          f"{'Brier':>8s} {'ECE':>7s}")
    for n, v in results["models"].items():
        a, pm, cal = v["aggregate"], v["pooled"], v["calibration"]
        print(f"  {n:18s} {a['f1']['mean']:.4f}+-{a['f1']['std']:.4f}  "
              f"{pm['f1']:>10.4f} {pm.get('roc_auc', float('nan')):>8.4f} "
              f"{cal['brier']:>8.4f} {cal['ece']:>7.4f}")

    save_json(results, C.RESULTS_DIR / "F5_cv.json")
    np.savez_compressed(C.CACHE_DIR / "F5_pooled.npz",
                        y=pooled[a_name][0], atype=pooled[a_name][2],
                        trip=pooled[a_name][3].astype(str),
                        **{f"p_{n}": v[1] for n, v in pooled.items()})
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F5_cv.json'} (protokol {stamp['hash']})")


if __name__ == "__main__":
    main()
