"""
F8 — Trip bazli dagilim, etki boyutlari ve tam alt-tip raporu.

Iki hakem maddesini kapatir:

  Hakem 1, madde 2: "consider reporting additional effect-size measures and
  per-trip performance distributions"

  Hakem 3, madde 8: "report window and positive-trip counts for all nine
  subtypes, and define subtype-specific F1 and false-positive handling. Add
  subtype precision, recall, and false-positive counts or rates, treating
  sparse subtypes descriptively."

Yeniden egitim yok: havuzlanmis capraz dogrulama tahminlerinden hesaplanir.
Ikili tahminler F5'te her fold'un KENDI dogrulama esigiyle uretildi; burada
yeniden esikleme YAPILMAZ.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import cluster_bootstrap_diff, cluster_permutation_test
from stanet.utils import init_console, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F8_effects_subtypes.json"

# alt-tip pozitif sayisi bunun altindaysa metrik raporlanmaz, betimsel kalir
SPARSE_TRIPS = 5          # pozitif trip sayisi
SPARSE_WINDOWS = 30       # pozitif pencere sayisi

EN = {"hiz_asimi": "Speed-limit violation", "uzun_duraklama": "Abnormal dwell",
      "anormal_ivme": "Abnormal acceleration", "kaza": "Crash",
      "anormal_yon_degisimi": "Abnormal heading change",
      "gps_hiz_artefakti": "GPS speed artefact",
      "gps_ivme_artefakti": "GPS acceleration artefact",
      "gps_yon_artefakti": "GPS heading artefact",
      "sinyal_donmasi": "Signal freeze"}


def counts(y, b):
    tp = int(((y == 1) & (b == 1)).sum())
    fp = int(((y == 0) & (b == 1)).sum())
    fn = int(((y == 1) & (b == 0)).sum())
    tn = int(((y == 0) & (b == 0)).sum())
    return tp, fp, fn, tn


def prf(y, b):
    tp, fp, fn, tn = counts(y, b)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fp_rate": fp / max(fp + tn, 1)}


def per_trip_delta(y, ba, bb, trips):
    """
    Her trip icin A ve B'nin F1'i ve farki.

    F1 trip basina hesaplanabilmesi icin o tripte en az bir pozitif olmali;
    pozitifi olmayan tripler icin yalnizca yanlis pozitif sayisi anlamlidir,
    onlar ayri raporlanir.
    """
    out, no_pos = [], []
    for t in np.unique(trips):
        m = trips == t
        yt = y[m]
        if yt.sum() == 0:
            fa = int(((yt == 0) & (ba[m] == 1)).sum())
            fb = int(((yt == 0) & (bb[m] == 1)).sum())
            no_pos.append({"trip": str(t), "n": int(m.sum()),
                           "fp_a": fa, "fp_b": fb})
            continue
        a, b = prf(yt, ba[m]), prf(yt, bb[m])
        out.append({"trip": str(t), "n": int(m.sum()), "n_pos": int(yt.sum()),
                    "f1_a": a["f1"], "f1_b": b["f1"], "delta": a["f1"] - b["f1"]})
    return out, no_pos


def effect_size(deltas):
    """Eslestirilmis Cohen's d — trip basina farklarin standartlastirilmisi."""
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2:
        return None
    sd = float(np.std(d, ddof=1))
    return {
        "n_trips": int(len(d)),
        "mean_delta": float(d.mean()),
        "sd_delta": sd,
        "cohens_d": float(d.mean() / sd) if sd > 0 else None,
        "median_delta": float(np.median(d)),
        "q25": float(np.percentile(d, 25)), "q75": float(np.percentile(d, 75)),
        "min": float(d.min()), "max": float(d.max()),
        "trips_favouring_a": int((d > 0).sum()),
        "trips_favouring_b": int((d < 0).sum()),
        "trips_tied": int((d == 0).sum()),
    }


def main():
    z = np.load(C.CACHE_DIR / "F5_pooled.npz", allow_pickle=True)
    y = z["y"].astype(int)
    atype = z["atype"].astype(int)
    trips = z["trip"].astype(str)
    stamp = protocol_stamp()

    f5 = load_json(C.RESULTS_DIR / "F5_cv.json")
    assert f5["protocol"]["hash"] == stamp["hash"], "F5 protokolu uyusmuyor"

    models = [k[2:] for k in z.files if k.startswith("b_")]
    B = {m: z[f"b_{m}"].astype(int) for m in models}

    results = {"protocol": stamp,
               "source": "F5_pooled.npz (capraz dogrulama, fold bazli esik)",
               "threshold_policy": "per_fold_validation",
               "sparse_rule": {"min_positive_trips": SPARSE_TRIPS,
                               "min_positive_windows": SPARSE_WINDOWS},
               "n_windows": int(len(y)), "n_trips": int(len(np.unique(trips)))}

    # ---------------------------------------------------------- A3 etki boyutu
    print("=" * 84)
    print("A3 — TRIP BAZLI DAGILIM VE ETKI BOYUTU (hibrit vs LSTM-only)")
    print("=" * 84)
    pairs = [("hybrid_gat", "lstm_only"), ("stanet_gated", "lstm_only"),
             ("stanet_gated", "hybrid_gat")]
    eff = {}
    for a, b in pairs:
        rows, no_pos = per_trip_delta(y, B[a], B[b], trips)
        e = effect_size([r["delta"] for r in rows])
        cb = cluster_bootstrap_diff(y, B[a], B[b], trips, seed=0)
        cp = cluster_permutation_test(y, B[a], B[b], trips, seed=0)
        eff[f"{a}_vs_{b}"] = {"per_trip": rows, "trips_without_positives": no_pos,
                              "effect_size": e, "cluster_bootstrap": cb,
                              "cluster_permutation": cp}
        print(f"\n  {a} vs {b}")
        print(f"    pozitifli trip        : {e['n_trips']}"
              f"  (pozitifsiz {len(no_pos)})")
        print(f"    ortalama fark         : {e['mean_delta']:+.4f}"
              f"  (sd {e['sd_delta']:.4f})")
        print(f"    Cohen's d             : {e['cohens_d']:+.3f}")
        print(f"    medyan / IQR          : {e['median_delta']:+.4f}"
              f"  [{e['q25']:+.4f}, {e['q75']:+.4f}]")
        print(f"    aralik                : [{e['min']:+.4f}, {e['max']:+.4f}]")
        print(f"    A lehine / B lehine   : {e['trips_favouring_a']}"
              f" / {e['trips_favouring_b']}  (esit {e['trips_tied']})")
        print(f"    havuz farki           : {cb['observed_diff']:+.4f}"
              f"  GA [{cb['lo']:+.4f}, {cb['hi']:+.4f}]  p={cp['p_value']:.4f}")
    results["effects"] = eff

    # ---------------------------------------------------------- A5 alt tipler
    print("\n" + "=" * 84)
    print("A5 — TAM ALT-TIP RAPORU")
    print("=" * 84)
    neg = y == 0
    sub = {}
    present = sorted(set(atype[y == 1].tolist()))
    print(f"\n  taksonomide 9 tip tanimli, veride {len(present)} tanesi var: "
          f"{present}")
    missing = [t for t in sorted(C.ANOMALY_NAMES) if t > 0 and t not in present]
    print(f"  hic uretilmemis: {missing}\n")
    results["subtypes_present"] = present
    results["subtypes_absent"] = missing

    hdr = (f"  {'alt tip':26s} {'pencere':>8s} {'trip':>5s} "
           f"{'hib P':>7s} {'hib R':>7s} {'hib F1':>7s} "
           f"{'lstm F1':>8s} {'hib FP':>7s} {'durum':>10s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in present:
        name = C.ANOMALY_NAMES.get(int(t), str(t))
        pos = (y == 1) & (atype == t)
        n_w = int(pos.sum())
        n_t = int(len(np.unique(trips[pos])))
        m = neg | pos
        h = prf(y[m], B["hybrid_gat"][m])
        l = prf(y[m], B["lstm_only"][m])
        sparse = (n_t < SPARSE_TRIPS) or (n_w < SPARSE_WINDOWS)
        entry = {"name_tr": name, "name_en": EN.get(name, name),
                 "n_windows": n_w, "n_positive_trips": n_t,
                 "sparse": bool(sparse),
                 "hybrid": h, "lstm_only": l,
                 "delta_f1": h["f1"] - l["f1"]}
        if not sparse:
            cb = cluster_bootstrap_diff(y[m], B["hybrid_gat"][m],
                                        B["lstm_only"][m], trips[m], seed=0)
            cp = cluster_permutation_test(y[m], B["hybrid_gat"][m],
                                          B["lstm_only"][m], trips[m], seed=0)
            entry["cluster_bootstrap"] = cb
            entry["cluster_permutation"] = cp
        sub[name] = entry
        flag = "BETIMSEL" if sparse else ""
        print(f"  {EN.get(name, name):26s} {n_w:>8d} {n_t:>5d} "
              f"{h['precision']:>7.4f} {h['recall']:>7.4f} {h['f1']:>7.4f} "
              f"{l['f1']:>8.4f} {h['fp']:>7d} {flag:>10s}")
    results["subtypes"] = sub

    n_sparse = sum(1 for v in sub.values() if v["sparse"])
    print(f"\n  {n_sparse}/{len(sub)} alt tip seyrek (pozitif trip < {SPARSE_TRIPS}"
          f" veya pozitif pencere < {SPARSE_WINDOWS}) -> yalnizca betimsel")
    print(f"\n  YANLIS POZITIF TANIMI: alt-tip kirilimlarinda negatifler ORTAK "
          f"tutulur;\n  yalnizca pozitif alt kumesi degisir. Bu yuzden FP sayisi "
          f"alt tipler arasinda\n  buyuk olculde ayni, F1 farki recall'dan gelir.")

    save_json(results, OUT)
    print(f"\nkaydedildi: {OUT} (protokol {stamp['hash']})")


if __name__ == "__main__":
    main()
