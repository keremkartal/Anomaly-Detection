"""Bootstrap güven aralıkları ve McNemar testi."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score


def bootstrap_ci(y_true, y_prob, metric="f1", threshold=0.5,
                 n_boot=2000, alpha=0.05, seed=0):
    """Tek modelin metriği için bootstrap %95 GA."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt = y_true[i]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "f1":
            vals.append(f1_score(yt, (y_prob[i] > threshold).astype(int), zero_division=0))
        elif metric == "roc_auc":
            vals.append(roc_auc_score(yt, y_prob[i]))
    vals = np.asarray(vals)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_boot": int(len(vals))}


def bootstrap_diff(y_true, prob_a, prob_b, metric="f1", threshold=0.5,
                   threshold_b=None, n_boot=2000, alpha=0.05, seed=0):
    """
    A - B farkinin bootstrap dagilimi (eslestirilmis, PENCERE duzeyi).

    `threshold`   A modelinin validation'da secilen esigi
    `threshold_b` B modelinin KENDI esigi. Verilmezse `threshold` kullanilir.

    KRITIK: v1'de tek esik her iki modele uygulaniyordu. Modellerin olasilik
    kalibrasyonlari farkli oldugu icin bu, B modelini kendi calisma noktasi
    disinda degerlendiriyor ve farki sistematik olarak A lehine sisiriyordu.
    Tablo 10'da Random Forest icin GA [+0,038, +0,116] cikarken ortalama fark
    +0,0137 yaziyordu; tutarsizligin kaynagi buydu.

    UYARI: Bu test pencereleri bagimsiz sayar. Pencereler %80 ortustugu icin
    varsayim ihlal edilir; sonuclar yalnizca v1 ile karsilastirma amaciyla
    raporlanir. Gecerli olan trip duzeyi testtir
    (bkz. stanet/cluster_stats.py).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    pa, pb = np.asarray(prob_a).ravel(), np.asarray(prob_b).ravel()
    tb = threshold if threshold_b is None else threshold_b
    ba = (pa > threshold).astype(int)
    bb = (pb > tb).astype(int)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt = y_true[i]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "f1":
            d = (f1_score(yt, ba[i], zero_division=0)
                 - f1_score(yt, bb[i], zero_division=0))
        else:
            d = roc_auc_score(yt, pa[i]) - roc_auc_score(yt, pb[i])
        diffs.append(d)
    diffs = np.asarray(diffs)
    obs = (f1_score(y_true, ba, zero_division=0)
           - f1_score(y_true, bb, zero_division=0)) if metric == "f1" else None
    return {"unit": "window_INVALID", "observed_diff": obs,
            "threshold_a": float(threshold), "threshold_b": float(tb),
            "mean_diff": float(diffs.mean()),
            "lo": float(np.percentile(diffs, 100 * alpha / 2)),
            "hi": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            "p_a_better": float((diffs > 0).mean()),
            "significant": bool(np.percentile(diffs, 100 * alpha / 2) > 0
                                or np.percentile(diffs, 100 * (1 - alpha / 2)) < 0)}


def mcnemar(y_true, pred_a, pred_b):
    """Eşleştirilmiş sınıflandırıcı karşılaştırması (exact binom)."""
    y_true = np.asarray(y_true).ravel()
    a = np.asarray(pred_a).ravel() == y_true
    b = np.asarray(pred_b).ravel() == y_true
    n01 = int((a & ~b).sum())   # yalnız A doğru
    n10 = int((~a & b).sum())   # yalnız B doğru
    if n01 + n10 == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0,
                "significant": False}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"only_a_correct": n01, "only_b_correct": n10,
            "p_value": float(p), "significant": bool(p < 0.05)}


def aggregate_seeds(runs, keys=("f1", "roc_auc", "precision", "recall", "accuracy")):
    """Çok-seed koşularından mean ± std."""
    out = {}
    for k in keys:
        v = np.asarray([r[k] for r in runs if k in r], dtype=float)
        if len(v):
            out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
                      "min": float(v.min()), "max": float(v.max()), "n": int(len(v)),
                      "values": v.tolist()}
    return out
