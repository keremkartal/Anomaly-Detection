"""
F7 — Sabit bolmedeki TUM karsilastirmalarin trip duzeyinde yeniden hesabi.

SORUN
-----
F1b, F3 ve F4'teki eslestirilmis testler (McNemar, bootstrap) PENCERE duzeyinde
yapiliyordu. Pencereler `tau=50`, `adim=10` ile uretildigi icin ardisik iki
pencere %80 ortusur; ustelik ayni tripin tum pencereleri ayni surucu, arac ve
rotaya aittir. Pencereler bagimsiz olmadigindan bu testler etkin ornek sayisini
oldugundan buyuk sayar: guven araliklari dar, p degerleri kucuk cikar.

COZUM
-----
Ayni karsilastirmalar trip duzeyinde tekrarlanir:
  - kume (cluster) bootstrap: trip'ler yerine konarak orneklenir
  - isaret-degistirme permutasyonu: takas birimi trip
  - Holm-Bonferroni: her karsilastirma ailesi icinde FWER kontrolu

Her aile icin TUM seed'lerde test kosulur; medyan seed birincil raporlanir,
seed'ler arasi p araligi da kaydedilir (tek kosuya bagimlilik gorunur olsun).

Pencere duzeyindeki eski degerler `window_level_INVALID` altinda saklanir —
silinmez, boylece makalede "eski deger su idi, dogru birimde su" denebilir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import (cluster_bootstrap_ci, cluster_bootstrap_diff,
                                  cluster_permutation_test, cluster_summary,
                                  holm_correction)
from stanet.data import build_dataset
from stanet.stats import bootstrap_diff, mcnemar
from stanet.utils import init_console, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F7_cluster_stats.json"


def binarize(prob, th):
    return (np.asarray(prob).ravel() > th).astype(int)


def family(y, trips, preds, thresholds, reference, label, n_boot=5000, n_perm=20000):
    """
    Bir karsilastirma ailesini trip duzeyinde degerlendirir.

    preds : {model: (n_seed, N) veya (N,)} olasiliklar
    thresholds: {model: esik veya esik listesi}
    """
    def seed_count(k):
        a = np.asarray(preds[k])
        return a.shape[0] if a.ndim == 2 else 1

    def get(k, s):
        a = np.asarray(preds[k])
        return a[s] if a.ndim == 2 else a

    def get_th(k, s):
        t = thresholds[k]
        return t[s] if isinstance(t, (list, tuple, np.ndarray)) else t

    n_seed = max(seed_count(k) for k in preds)
    med = n_seed // 2
    others = [k for k in preds if k != reference]

    out = {"label": label, "reference": reference, "n_seeds": n_seed,
           "median_seed_index": med, "cluster_info": cluster_summary(trips),
           "comparisons": {}}

    # referansin kendi trip-duzeyi GA'si
    out["reference_ci"] = cluster_bootstrap_ci(
        y, binarize(get(reference, med), get_th(reference, med)), trips,
        n_boot=n_boot, seed=0)

    perm_p = {}
    for k in others:
        per_seed = []
        ns = min(seed_count(reference), seed_count(k))
        for s in range(ns):
            a = binarize(get(reference, s), get_th(reference, s))
            b = binarize(get(k, s), get_th(k, s))
            per_seed.append(cluster_permutation_test(y, a, b, trips,
                                                     n_perm=n_perm, seed=s))
        a_m = binarize(get(reference, med), get_th(reference, med))
        b_m = binarize(get(k, med), get_th(k, med))
        cb = cluster_bootstrap_diff(y, a_m, b_m, trips, n_boot=n_boot, seed=0)
        cp = per_seed[med] if med < len(per_seed) else per_seed[0]

        # pencere duzeyi (gecersiz, karsilastirma icin)
        wl = {"mcnemar": mcnemar(y, a_m, b_m),
              "bootstrap_diff": bootstrap_diff(y, get(reference, med), get(k, med),
                                               threshold=get_th(reference, med), seed=0),
              "note": "pencereler %80 ortusur - bagimsizlik varsayimi ihlal"}

        ps = [r["p_value"] for r in per_seed]
        out["comparisons"][k] = {
            "cluster_bootstrap": cb, "cluster_permutation": cp,
            "per_seed_p": ps, "p_min": float(min(ps)), "p_max": float(max(ps)),
            "n_seeds_significant": int(sum(p < 0.05 for p in ps)),
            "window_level_INVALID": wl,
            "ci_width_trip": float(cb["hi"] - cb["lo"]),
            "ci_width_window": float(wl["bootstrap_diff"]["hi"]
                                     - wl["bootstrap_diff"]["lo"]),
        }
        out["comparisons"][k]["ci_widening"] = (
            out["comparisons"][k]["ci_width_trip"]
            / max(out["comparisons"][k]["ci_width_window"], 1e-9))
        perm_p[k] = cp["p_value"]

    for k, h in holm_correction(perm_p).items():
        out["comparisons"][k]["holm"] = h
    return out


def show(fam):
    ci = fam["cluster_info"]
    print("=" * 96)
    print(f"{fam['label']}  —  referans: {fam['reference']}")
    print(f"  birim: TRIP ({ci['n_trips']} trip / {ci['n_windows']} pencere, "
          f"trip basina ort. {ci['windows_per_trip_mean']:.1f}, ortusme %80) | "
          f"{fam['n_seeds']} seed, medyan seed #{fam['median_seed_index']}")
    r = fam["reference_ci"]
    print(f"  referans F1 = {r['point']:.4f}  trip-GA [{r['lo']:.4f}, {r['hi']:.4f}]")
    print("=" * 96)
    print(f"  {'rakip':20s} {'dF1':>8s} {'trip GA':>20s} {'p':>8s} {'Holm':>8s} "
          f"{'karar':>10s} {'GA gen.':>8s} {'seed p araligi':>16s}")
    for k, v in sorted(fam["comparisons"].items(),
                       key=lambda kv: kv[1]["holm"]["p_holm"]):
        cb, h = v["cluster_bootstrap"], v["holm"]
        print(f"  {k:20s} {cb['observed_diff']:+8.4f} "
              f"[{cb['lo']:+.4f},{cb['hi']:+.4f}] {v['cluster_permutation']['p_value']:>8.4f} "
              f"{h['p_holm']:>8.4f} {'ANLAMLI' if h['significant'] else 'anlamsiz':>10s} "
              f"{v['ci_widening']:>7.2f}x "
              f"[{v['p_min']:.3f},{v['p_max']:.3f}]".rjust(0))
    print()


def main():
    ds = build_dataset(legacy=False)
    trips = np.asarray(ds.meta_test)[:, 0]         # META_COLS[0] = trip_id
    y = ds.y_test.astype(int)
    stamp = protocol_stamp()

    ci = cluster_summary(trips)
    print(f"protokol {stamp['hash']}")
    print(f"test seti: {ci['n_windows']} pencere, {ci['n_trips']} trip "
          f"(trip basina {ci['windows_per_trip_min']}-{ci['windows_per_trip_max']}, "
          f"ort. {ci['windows_per_trip_mean']:.1f})")
    print(f"ETKIN ORNEK SAYISI pencere sayisinin "
          f"{ci['n_windows']/ci['n_trips']:.1f} kati KUCUK\n")

    results = {"protocol": stamp, "test_cluster_info": ci, "families": {}}

    # -------------------------------------------------- 1) F1b: hibrit vs temeller
    f1b = load_json(C.RESULTS_DIR / "F1b_fair.json")
    z1 = np.load(C.CACHE_DIR / "F1b_predictions.npz")
    assert np.array_equal(z1["y_test"].astype(int), y), "F1b test seti uyusmuyor"
    preds, ths = {}, {}
    for key, mkey in [("hybrid_lstm_gat_median", "hybrid_lstm_gat"),
                      ("lstm_only_median", "lstm_only"),
                      ("lightgbm_median", "lightgbm"),
                      ("xgboost_median", "xgboost"),
                      ("random_forest_median", "random_forest")]:
        if key in z1.files:
            preds[mkey] = z1[key]
            ths[mkey] = f1b["models"][mkey]["median_threshold"]
    rule = [k for k in z1.files if k.startswith("rule_")]
    if rule:
        preds[rule[0]] = z1[rule[0]]
        ths[rule[0]] = 0.5
    fam = family(y, trips, preds, ths, "hybrid_lstm_gat",
                 "F1b — STANet (LSTM+GAT) vs temel modeller")
    results["families"]["baselines"] = fam
    show(fam)

    # -------------------------------------------------- 2) F3: fuzyon ablasyonu
    f3 = load_json(C.RESULTS_DIR / "F3_fusion.json")
    z3 = np.load(C.CACHE_DIR / "F3_predictions.npz")
    preds = {f: z3[f] for f in f3["variants"] if f in z3.files}
    ths = {f: [r["threshold_val"] for r in f3["variants"][f]["runs"]] for f in preds}
    fam = family(y, trips, preds, ths, "gated", "F3 — gated fuzyon vs alternatifler")
    results["families"]["fusion"] = fam
    show(fam)

    # -------------------------------------------------- 3) F4: kodlayici benchmark
    f4 = load_json(C.RESULTS_DIR / "F4_benchmark.json")
    z4 = np.load(C.CACHE_DIR / "F4_predictions.npz")
    preds = {k: z4[k] for k in f4["configs"] if k in z4.files}
    ths = {k: [r["threshold_val"] for r in f4["configs"][k]["runs"]] for k in preds}
    fam = family(y, trips, preds, ths, "T:lstm",
                 "F4 — STANet (LSTM+GAT) vs STGCN / DCRNN / Graph WaveNet / Transformer")
    results["families"]["benchmark"] = fam
    show(fam)

    # -------------------------------------------------- ozet
    print("=" * 96)
    print("OZET — pencere duzeyinden trip duzeyine gecisin etkisi")
    print("=" * 96)
    print(f"  {'aile':14s} {'karsilastirma':22s} {'p (pencere)':>12s} {'p (trip)':>10s} "
          f"{'Holm':>8s} {'GA genisleme':>13s}")
    flips = []
    for fname, fam in results["families"].items():
        for k, v in fam["comparisons"].items():
            pw = v["window_level_INVALID"]["mcnemar"]["p_value"]
            pt = v["cluster_permutation"]["p_value"]
            print(f"  {fname:14s} {k:22s} {pw:>12.4f} {pt:>10.4f} "
                  f"{v['holm']['p_holm']:>8.4f} {v['ci_widening']:>12.2f}x")
            if (pw < 0.05) != v["holm"]["significant"]:
                flips.append((fname, k, pw, v["holm"]["p_holm"]))
    results["conclusion_flips"] = [
        {"family": a, "comparison": b, "p_window": c, "p_holm_trip": d}
        for a, b, c, d in flips]
    print(f"\n  Karar degistiren karsilastirma sayisi: {len(flips)}")
    for a, b, c, d in flips:
        print(f"    {a}/{b}: pencere p={c:.4f} (anlamli) -> trip Holm p={d:.4f} "
              f"(anlamsiz)" if c < 0.05 else
              f"    {a}/{b}: pencere p={c:.4f} -> trip Holm p={d:.4f} (anlamli)")

    save_json(results, OUT)
    print(f"\nkaydedildi: {OUT}")


if __name__ == "__main__":
    main()
