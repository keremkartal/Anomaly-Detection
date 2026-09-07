"""
Trip duzeyinde kume (cluster) istatistikleri.

NEDEN GEREKLI
-------------
Pencereler `STEP_SIZE=10` adimla, `SEQUENCE_LENGTH=50` uzunlukta kaydirilarak
uretiliyor. Ardisik iki pencere 40 adimi paylasir — **%80 ortusme**. Ustelik
ayni trip'in tum pencereleri ayni surucuye, ayni araca ve ayni rotaya aittir.

Dolayisiyla pencereler BAGIMSIZ DEGILDIR. Pencere duzeyinde yapilan bootstrap
ve McNemar testleri etkin ornek sayisini oldugundan buyuk sayar; guven
araliklari gercekte olduklarindan dar, p degerleri oldugundan kucuk cikar.
Hakemin isaretledigi kusur budur.

Dogru istatistiksel birim **trip**'tir. Bu modul uc test saglar:

1. `cluster_bootstrap_diff`  — trip'ler yerine konarak yeniden orneklenir
2. `cluster_permutation_test` — trip duzeyinde isaret degistirme (sign-flip)
3. `holm_correction`         — coklu karsilastirma duzeltmesi

HIZLI UYGULAMA
--------------
F1 yalnizca (TP, FP, FN) toplamlarina baglidir ve bu toplamlar trip'ler
uzerinde TOPLANABILIR. Bu yuzden her trip icin uclu sayimlar bir kez
hesaplanir; bootstrap bir multinom agirlikli toplam, permutasyon ise bir
matris carpimi haline gelir. 100.000 permutasyon saniyeler surer ve sonuc
naif donguyle birebir aynidir.
"""
import numpy as np

EPS = 1e-12


# --------------------------------------------------------------- temel sayimlar
def _counts_per_group(y, pred, groups):
    """Her grup icin (TP, FP, FN) — F1 bu uclunun toplamindan hesaplanir."""
    y = np.asarray(y).ravel().astype(np.int64)
    pred = np.asarray(pred).ravel().astype(np.int64)
    uniq, inv = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    g = len(uniq)
    tp = np.bincount(inv, weights=(y * pred), minlength=g)
    fp = np.bincount(inv, weights=((1 - y) * pred), minlength=g)
    fn = np.bincount(inv, weights=(y * (1 - pred)), minlength=g)
    return uniq, np.stack([tp, fp, fn], axis=1)      # (G, 3)


def _f1_from_counts(c):
    """c: (..., 3) -> F1. Son eksen (TP, FP, FN)."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return 2.0 * tp / np.maximum(2.0 * tp + fp + fn, EPS)


def cluster_summary(groups) -> dict:
    """Etkin ornek sayisi raporu — makalede pencere/trip ayrimi icin."""
    g = np.asarray(groups).ravel()
    uniq, cnt = np.unique(g, return_counts=True)
    return {"n_windows": int(len(g)), "n_trips": int(len(uniq)),
            "windows_per_trip_mean": float(cnt.mean()),
            "windows_per_trip_median": float(np.median(cnt)),
            "windows_per_trip_min": int(cnt.min()),
            "windows_per_trip_max": int(cnt.max()),
            "window_overlap_ratio": 0.8}      # tau=50, adim=10


# --------------------------------------------------------------- 1) bootstrap
def cluster_bootstrap_diff(y, pred_a, pred_b, groups, n_boot=5000,
                           alpha=0.05, seed=0) -> dict:
    """
    A - B F1 farki icin trip duzeyinde kume bootstrap %95 GA.

    Trip'ler (pencereler degil) yerine konarak yeniden orneklenir; secilen
    trip'in TUM pencereleri birlikte gelir. Boylece trip ici bagimlilik
    korunur ve GA gercekci genislige ulasir.

    `pred_a` / `pred_b` IKILI tahminlerdir (esik zaten validation'da secilmis
    olmalidir). Esik bootstrap icinde yeniden secilmez — bu, degerlendirme
    kuralinin bir parcasi olarak sabittir.
    """
    uniq, ca = _counts_per_group(y, pred_a, groups)
    _, cb = _counts_per_group(y, pred_b, groups)
    g = len(uniq)
    rng = np.random.default_rng(seed)

    obs = float(_f1_from_counts(ca.sum(0)) - _f1_from_counts(cb.sum(0)))

    # her tekrarda trip'lerin kac kez secildigi -> multinom agirlik
    w = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(np.float64)
    sa, sb = w @ ca, w @ cb                    # (n_boot, 3)
    diffs = _f1_from_counts(sa) - _f1_from_counts(sb)

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"unit": "trip", "n_trips": int(g), "n_boot": int(n_boot),
            "observed_diff": obs, "mean_diff": float(diffs.mean()),
            "std_diff": float(diffs.std(ddof=1)),
            "lo": float(lo), "hi": float(hi),
            "p_a_better": float((diffs > 0).mean()),
            "significant": bool(lo > 0 or hi < 0)}


def cluster_bootstrap_ci(y, pred, groups, n_boot=5000, alpha=0.05, seed=0) -> dict:
    """Tek modelin F1'i icin trip duzeyinde kume bootstrap GA."""
    uniq, c = _counts_per_group(y, pred, groups)
    g = len(uniq)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(np.float64)
    vals = _f1_from_counts(w @ c)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"unit": "trip", "n_trips": int(g), "n_boot": int(n_boot),
            "point": float(_f1_from_counts(c.sum(0))),
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "lo": float(lo), "hi": float(hi)}


# --------------------------------------------------------------- 2) permutasyon
def cluster_permutation_test(y, pred_a, pred_b, groups, n_perm=20000,
                             seed=0, alternative="two-sided") -> dict:
    """
    Trip duzeyinde eslestirilmis isaret-degistirme permutasyon testi.

    Sifir hipotezi: A ve B ayni tripte degistirilebilir (exchangeable).
    Her tekrarda her trip icin bagimsiz bir yazi-tura atilir; tura gelirse
    O TRIP'IN tum pencerelerinde A ve B tahminleri takas edilir. Boylece
    permutasyon birimi pencere degil trip olur.

    Bu, ortusen pencerelerde gecersiz olan McNemar testinin dogru muadilidir:
    McNemar her pencereyi bagimsiz bir deneme sayar, bu test saymaz.
    """
    uniq, ca = _counts_per_group(y, pred_a, groups)
    _, cb = _counts_per_group(y, pred_b, groups)
    g = len(uniq)
    rng = np.random.default_rng(seed)

    tot_a, tot_b = ca.sum(0), cb.sum(0)
    obs = float(_f1_from_counts(tot_a) - _f1_from_counts(tot_b))

    d = ca - cb                                   # trip basina fark katkisi
    # s=+1 takas yok, s=-1 takas var. Takas edilmis toplamlar:
    #   A' = (ca+cb)/2 + s*(ca-cb)/2 ; B' = (ca+cb)/2 - s*(ca-cb)/2
    half_sum = (ca + cb) / 2.0
    base = half_sum.sum(0)
    s = rng.choice([1.0, -1.0], size=(n_perm, g))
    delta = (s @ d) / 2.0                          # (n_perm, 3)
    null = _f1_from_counts(base + delta) - _f1_from_counts(base - delta)

    if alternative == "two-sided":
        p = (1.0 + np.sum(np.abs(null) >= abs(obs) - EPS)) / (n_perm + 1.0)
    elif alternative == "greater":
        p = (1.0 + np.sum(null >= obs - EPS)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(null <= obs + EPS)) / (n_perm + 1.0)

    return {"unit": "trip", "n_trips": int(g), "n_perm": int(n_perm),
            "observed_diff": obs, "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "p_value": float(p), "alternative": alternative,
            "significant": bool(p < 0.05)}


# --------------------------------------------------------------- 3) Holm
def holm_correction(pvalues: dict, alpha=0.05) -> dict:
    """
    Holm-Bonferroni adimsal indirgeme.

    Aile-bazli hata oraninI (FWER) alpha'da tutar ve Bonferroni'den daha
    gucludur. Girdi {karsilastirma_adi: p}; cikti her ad icin duzeltilmis p
    ve karar.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj                    # monotonluk (step-down)
        out[name] = {"p_raw": float(p), "p_holm": float(adj),
                     "rank": i + 1, "n_family": m,
                     "significant": bool(adj < alpha)}
    return out


# --------------------------------------------------------------- kolay arayuz
def compare_models(y, preds: dict, groups, reference: str,
                   n_boot=5000, n_perm=20000, seed=0, alpha=0.05) -> dict:
    """
    Referans modeli tum rakiplerle trip duzeyinde karsilastirir ve
    Holm duzeltmesi uygular.

    preds: {model_adi: ikili tahmin dizisi}
    """
    others = [k for k in preds if k != reference]
    raw = {}
    for k in others:
        raw[k] = {
            "bootstrap": cluster_bootstrap_diff(y, preds[reference], preds[k],
                                                groups, n_boot=n_boot, seed=seed),
            "permutation": cluster_permutation_test(y, preds[reference], preds[k],
                                                    groups, n_perm=n_perm, seed=seed),
        }
    holm = holm_correction({k: v["permutation"]["p_value"] for k, v in raw.items()},
                           alpha=alpha)
    for k in others:
        raw[k]["holm"] = holm[k]
    return {"reference": reference, "alpha": alpha,
            "cluster_info": cluster_summary(groups), "comparisons": raw}
