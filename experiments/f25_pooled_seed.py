"""
F25 — Havuzlanmis trip duzeyi analizi HANGI TOHUMLA yapilirsa yapilsin ayni
sonucu veriyor mu.

DURUM
-----
Capraz dogrulamada her konfigurasyon 5 fold x 2 tohum kosuyor, ama
HAVUZLANMIS analiz (alt tip kirilimi, trip duzeyi kume bootstrap'i ve
permutasyon testi) yalnizca ILK tohumun tahminleriyle yapiliyor. Sebebi
mesru: iki tohumu birlestirmek her pencereyi iki kez saymak olurdu ve kume
bootstrap'i bagimsizligi zaten trip duzeyinde varsayiyor.

Ama bu, okunmasi kolay bir itirap birakiyor: "merkezi olumsuz bulgunuz tek
bir tohumun tahminlerine mi dayaniyor?" Cevap ancak digeriyle tekrarlayip
bakarak verilebilir.

BU BETIK
--------
Ayni analizi HER tohum icin ayri ayri kosar ve KARARLARIN degisip
degismedigine bakar. Onemli olan p degerlerinin birebir ayni cikmasi degil
-- cikmayacaktir -- ayni SONUCA varilip varilmadigidir.

Egitim yapmaz; onbellekteki tahminleri okur.

KULLANIM
  python experiments/f25_pooled_seed.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import (cluster_bootstrap_diff, cluster_permutation_test,
                                  holm_correction)
from stanet.data import build_cv_folds
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f5_cv import pooled_by_type, pooled_metrics          # noqa: E402

init_console()
OUT = C.RESULTS_DIR / "F25_pooled_seed.json"
SEEDS = [42, 43]
MERGE = False
REFERENCE = "lstm_only"            # uzamsal dal YOK — karsilastirma tabani
CONFIGS = {
    "hybrid_gat":      dict(temporal="lstm", spatial="gat",  fusion="weighted_sum"),
    "stanet_gated":    dict(temporal="lstm", spatial="gat",  fusion="gated"),
    "hybrid_concat":   dict(temporal="lstm", spatial="gat",  fusion="concat"),
    "hybrid_crossatt": dict(temporal="lstm", spatial="gat",  fusion="cross_attention"),
    "lstm_only":       dict(temporal="lstm", spatial="none", fusion="weighted_sum"),
}


def pooled_for_seed(seed, folds, fold_idx, fold_trips, graph, device):
    """Tek tohumun havuzlanmis dizileri: y, olasilik, tip, trip, ikili."""
    out = {}
    for name, cfg in CONFIGS.items():
        y, p, b, t, g = [], [], [], [], []
        for f in folds:
            k = f.info["fold"]
            spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS, **cfg)
            r = get_or_train(spec, seed, f, fold_idx[k], graph, device, verbose=False)
            th = float(r["metrics"]["threshold_val"])
            y.append(f.y_test)
            p.append(r["prob_test"])
            b.append((r["prob_test"] > th).astype(int))
            t.append(f.type_test)
            g.append(fold_trips[k])
        out[name] = tuple(np.concatenate(x) for x in (y, p, b, t, g))
    return out


def analyse(pooled):
    """Referansa gore trip duzeyi karsilastirma + alt tip kirilimi."""
    ry, _, rb, _, rg = pooled[REFERENCE]
    comps, pv = {}, {}
    for name in CONFIGS:
        if name == REFERENCE:
            continue
        y, p, b, t, g = pooled[name]
        bo = cluster_bootstrap_diff(y, b, rb, g)
        pe = cluster_permutation_test(y, b, rb, g)
        comps[name] = {"bootstrap": bo, "permutation": pe}
        pv[name] = pe["p_value"]
    for k, v in holm_correction(pv).items():
        comps[k]["holm"] = v
    summary = {name: pooled_metrics(*[pooled[name][i] for i in (0, 1, 2)],
                                    "per_fold_validation")
               for name in CONFIGS}
    return comps, summary


def main():
    device = get_device()
    stamp = protocol_stamp()
    graph = build_graph(merge_by_osm=MERGE)
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    fold_idx, fold_trips = {}, {}
    for f in folds:
        k = f.info["fold"]
        fold_idx[k] = {sp: map_segments(getattr(f, "meta_" + sp), graph)[0]
                       for sp in ("train", "val", "test")}
        fold_trips[k] = np.asarray(f.meta_test)[:, 0]
    print("protokol %s | referans %s | tohumlar %s\n"
          % (stamp["hash"], REFERENCE, SEEDS), flush=True)

    results = {"protocol": stamp, "reference": REFERENCE, "seeds": SEEDS,
               "note": ("havuzlanmis analiz tek tohumla yapilir; bu betik onu "
                        "her tohum icin ayri kosup KARARLARIN degisip "
                        "degismedigine bakar"),
               "by_seed": {}}

    for seed in SEEDS:
        pooled = pooled_for_seed(seed, folds, fold_idx, fold_trips, graph, device)
        comps, summ = analyse(pooled)
        results["by_seed"][str(seed)] = {"comparisons": comps, "pooled": summ}
        print("TOHUM %d" % seed)
        print("  %-18s %9s %22s %8s %8s %8s"
              % ("konfig", "havuz F1", "%95 GA (kume)", "perm p", "Holm p", "karar"))
        for name in CONFIGS:
            f1 = summ[name]["f1"]
            if name == REFERENCE:
                print("  %-18s %9.4f  %-22s %8s %8s %8s"
                      % (name, f1, "(referans)", "", "", ""))
                continue
            c = comps[name]
            sig = c.get("holm", {}).get("significant")
            print("  %-18s %9.4f [%+.4f, %+.4f] %8.4f %8.4f %8s"
                  % (name, f1, c["bootstrap"]["lo"], c["bootstrap"]["hi"],
                     c["permutation"]["p_value"],
                     c.get("holm", {}).get("p_holm", float("nan")),
                     "FARK" if sig else "-"))
        print(flush=True)

    # ---------------------------------------------------- kararlar degisti mi
    print("=" * 84)
    print("KARARLAR TOHUMA GORE DEGISIYOR MU")
    print("=" * 84)
    a, b = (results["by_seed"][str(s)]["comparisons"] for s in SEEDS)
    flips, rows = [], []
    for name in CONFIGS:
        if name == REFERENCE:
            continue
        sa = bool(a[name].get("holm", {}).get("significant"))
        sb = bool(b[name].get("holm", {}).get("significant"))
        da = a[name]["bootstrap"]["observed_diff"]
        db = b[name]["bootstrap"]["observed_diff"]
        rows.append((name, da, sa, db, sb))
        if sa != sb:
            flips.append(name)
        print("  %-18s tohum %d: %+.4f %-5s | tohum %d: %+.4f %-5s  %s"
              % (name, SEEDS[0], da, "FARK" if sa else "-",
                 SEEDS[1], db, "FARK" if sb else "-",
                 "<<< KARAR DEGISTI" if sa != sb else ""))
    same_sign = all(np.sign(r[1]) == np.sign(r[3]) for r in rows)
    print("\n  karar degistiren karsilastirma: %d / %d"
          % (len(flips), len(rows)))
    print("  fark yonu her ikisinde ayni mi: %s" % ("EVET" if same_sign else "HAYIR"))
    sign_flips = [r[0] for r in rows if np.sign(r[1]) != np.sign(r[3])]
    # Iki ayri sorunun cevabi ayni degil: KARAR degismiyor olabilir ama
    # farkin ISARETI degisiyor olabilir. Ikincisi birincisinden daha carpici
    # bir kararsizliktir ve yalnizca "dayanikli" demek onu gizler.
    results["stability"] = {
        "decisions_flipped": flips,
        "sign_flipped": sign_flips,
        "n_comparisons": len(rows),
        "same_direction": bool(same_sign),
        "conclusion": (
            ("kararlar dayanikli (%d/%d karsilastirmada ayni sonuc)"
             % (len(rows) - len(flips), len(rows)))
            + ("; ancak farkin ISARETI %d/%d karsilastirmada degisiyor — "
               "nokta tahminleri yonunu bile korumuyor"
               % (len(sign_flips), len(rows)) if sign_flips else
               "; farkin isareti de korunuyor")),
    }
    print("  isaret degistiren      : %d / %d  %s"
          % (len(sign_flips), len(rows), ", ".join(sign_flips)))
    print("  -> %s" % results["stability"]["conclusion"])

    save_json(results, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
