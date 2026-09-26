"""
F13 — Anomali yayginligina duyarlilik (degerlendirme: tek yayginlikta olculmus performans).

Isaretlenen sorun: bildirilen performans TEK bir anomali yayginliginda
olculdu; anomalilerin daha seyrek ya da daha sik oldugu bir ortamda
dedektorun nasil davranacagi belirsiz kaliyor.

YAKLASIM — alt orneklemek YERINE kapali form
--------------------------------------------
Pencereleri atarak yayginlik degistirmek iki sorun yaratir: (i) hangi
pencerenin atildigi rastgeledir, (ii) atma islemi trip yapisini bozar.
Gerek yok: bir esikteki duyarlilik (TPR) ve yanlis alarm orani (FPR) sinif
ICINDE hesaplanir ve yayginliktan BAGIMSIZDIR. Dolayisiyla herhangi bir
yayginlik pi icin kesinlik ve F1 kapali formda yazilir:

    recall    = TPR
    precision = pi*TPR / (pi*TPR + (1-pi)*FPR)
    F1        = 2*P*R / (P+R)

Bu bir tahmin degil, ayni modelin ayni esikteki davranisinin farkli sinif
dengesindeki tam degeridir. Alt orneklemenin getirdigi ek gurultu yoktur.

IKI ESIK POLITIKASI
-------------------
  (a) SABIT esik — esik gozlenen yayginlikta dogrulama uzerinde secilmis,
      yayginlik degisince degistirilmiyor. Modeli oldugu gibi yeni bir
      dagilima tasimak budur.
  (b) YENIDEN AYARLI esik — her hedef yayginlik icin esik, dogrulama
      kumesinin ROC egrisi uzerinde o yayginliktaki analitik F1'i en
      buyuten noktada secilir. Yeni dagilimda kalibrasyon yapma imkani
      varsa elde edilecek olan budur.

(a) ile (b) arasindaki mesafe, "bu model baska bir yayginlikta yeniden
esiklenmeden kullanilabilir mi" sorusunun cevabidir.

ROC-AUC yayginliktan bagimsizdir ve degismez; PR-AUC degisir. Bu ayrim da
raporlanir cunku makalede iki deger yan yana duruyor.

Bu betik ONBELLEKTEKI dogrulama/test olasiliklarini okur; egitim yapmaz,
GPU kullanmaz.

KULLANIM
  python experiments/f13_prevalence.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.utils import init_console, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F13_prevalence.json"

# Hedef yayginliklar. Gozlenen deger de listeye eklenir (kontrol noktasi).
TARGETS = [0.01, 0.03, 0.08, 0.15, 0.25]

# CV kosularinin dosya adi kalibi: <taban>-cv5-fold<k>__<graf>__s<seed>
MODELS = {
    "hybrid_gat":      "lstm-gat-weighted_sum",
    "stanet_gated":    "lstm-gat-gated",
    "hybrid_concat":   "lstm-gat-concat",
    "hybrid_crossatt": "lstm-gat-cross_attention",
    "lstm_only":       "lstm-none-weighted_sum",
}
GRAPH = "trip_instance"
SEEDS = [42, 43]


def roc_points(y, prob):
    """Tum esiklerde (esik, TPR, FPR). Esikler olasilik degerleridir."""
    y = np.asarray(y).ravel().astype(int)
    p = np.asarray(prob).ravel().astype(float)
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    P, N = int(ys.sum()), int((1 - ys).sum())
    if P == 0 or N == 0:
        return None
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # ayni olasilik degerinde birden fazla ornek varsa son indisi al
    keep = np.r_[np.diff(ps) != 0, True]
    return ps[keep], tp[keep] / P, fp[keep] / N


def f1_at(pi, tpr, fpr):
    """Verilen yayginlikta analitik F1 (dizi girdi kabul eder)."""
    tpr = np.asarray(tpr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    num = pi * tpr
    den = pi * tpr + (1 - pi) * fpr
    prec = np.where(den > 0, num / np.maximum(den, 1e-15), 0.0)
    rec = tpr
    s = prec + rec
    return np.where(s > 0, 2 * prec * rec / np.maximum(s, 1e-15), 0.0), prec, rec


def rate_at_threshold(y, prob, th):
    y = np.asarray(y).ravel().astype(int)
    b = (np.asarray(prob).ravel() > th).astype(int)
    P, N = int(y.sum()), int((1 - y).sum())
    if P == 0 or N == 0:
        return None
    tpr = float(((y == 1) & (b == 1)).sum() / P)
    fpr = float(((y == 0) & (b == 1)).sum() / N)
    return tpr, fpr


def main():
    stamp = protocol_stamp()
    run_dir = C.RUNS_DIR / stamp["hash"]
    print("protokol %s | onbellek %s\n" % (stamp["hash"], run_dir.name), flush=True)

    results = {"protocol": stamp,
               "method": ("kapali form: TPR/FPR sinif icinde hesaplandigi icin "
                          "yayginliktan bagimsizdir; precision ve F1 verilen pi "
                          "icin tam olarak yeniden yazilir"),
               "targets": TARGETS, "models": {}}

    for name, base in MODELS.items():
        per_run = []
        for k in range(C.N_FOLDS):
            for seed in SEEDS:
                key = "%s-cv%d-fold%d__%s__s%d" % (base, C.N_FOLDS, k, GRAPH, seed)
                jf, nf = run_dir / (key + ".json"), run_dir / (key + ".npz")
                if not (jf.exists() and nf.exists()):
                    print("  ATLANDI (onbellekte yok): %s" % key)
                    continue
                meta = json.loads(jf.read_text(encoding="utf-8"))
                arr = np.load(nf)
                yv, pv = arr["y_val"], arr["prob_val"]
                yt, pt = arr["y_test"], arr["prob_test"]
                th_obs = float(meta["metrics"]["threshold_val"])
                pi_obs = float(np.asarray(yt).ravel().mean())

                rv, rt = roc_points(yv, pv), roc_points(yt, pt)
                if rv is None or rt is None:
                    continue
                thv, tprv, fprv = rv
                row = {"fold": k, "seed": seed, "prevalence_observed": pi_obs,
                       "threshold_observed": th_obs, "by_prevalence": {}}
                for pi in TARGETS:
                    # (a) sabit esik
                    r = rate_at_threshold(yt, pt, th_obs)
                    f_fix, p_fix, rec_fix = f1_at(pi, r[0], r[1])
                    # (b) yeniden ayarli esik — dogrulama ROC'u uzerinde en iyi
                    fv, _, _ = f1_at(pi, tprv, fprv)
                    th_new = float(thv[int(np.argmax(fv))])
                    r2 = rate_at_threshold(yt, pt, th_new)
                    f_new, p_new, rec_new = f1_at(pi, r2[0], r2[1])
                    row["by_prevalence"]["%.4f" % pi] = {
                        "fixed_threshold": {"threshold": th_obs,
                                            "tpr": r[0], "fpr": r[1],
                                            "precision": float(p_fix),
                                            "recall": float(rec_fix),
                                            "f1": float(f_fix)},
                        "retuned_threshold": {"threshold": th_new,
                                              "tpr": r2[0], "fpr": r2[1],
                                              "precision": float(p_new),
                                              "recall": float(rec_new),
                                              "f1": float(f_new)},
                    }
                row["roc_auc"] = float(meta["metrics"].get("roc_auc", float("nan")))
                row["pr_auc_observed"] = float(meta["metrics"].get("pr_auc", float("nan")))
                per_run.append(row)
        if not per_run:
            continue
        results["models"][name] = {"runs": per_run, "summary": {}}
        # sutunlar yalnizca ONCEDEN belirlenmis hedeflerdir. Gozlenen
        # yaygınlik fold'lar arasinda %3,4 ile %11 arasinda degisiyor; onu
        # sutun yapmak fold'lari farkli sayilarla karsilastirmak olurdu.
        keys = ["%.4f" % pi for pi in TARGETS]
        for pk in keys:
            for pol in ("fixed_threshold", "retuned_threshold"):
                v = np.array([r["by_prevalence"][pk][pol]["f1"] for r in per_run
                              if pk in r["by_prevalence"]])
                if len(v) == 0:
                    continue
                results["models"][name]["summary"].setdefault(pk, {})[pol] = {
                    "f1_mean": float(v.mean()), "f1_sd": float(v.std(ddof=1)),
                    "n": int(len(v))}
        auc = np.array([r["roc_auc"] for r in per_run])
        results["models"][name]["roc_auc_mean"] = float(np.nanmean(auc))
        pio = np.array([r["prevalence_observed"] for r in per_run])
        results["models"][name]["prevalence_observed"] = {
            "mean": float(pio.mean()), "min": float(pio.min()),
            "max": float(pio.max()),
            "note": "fold test kumelerinin gercek yayginligi"}

    # ------------------------------------------------------------------ cikti
    print("=" * 94)
    print("YAYGINLIK DUYARLILIGI — %d fold x %d tohum, kapali form" % (C.N_FOLDS, len(SEEDS)))
    print("her hucre: F1 ort+-sd  (ust satir SABIT esik, alt satir YENIDEN AYARLI)")
    print("=" * 94)
    any_model = next(iter(results["models"].values()))
    cols = sorted(any_model["summary"].keys(), key=float)
    print("  %-17s" % "model" + "".join("%13s" % ("pi=%.1f%%" % (float(c) * 100))
                                        for c in cols))
    for name, d in results["models"].items():
        line = "  %-17s" % name
        for c in cols:
            s = d["summary"][c]["fixed_threshold"]
            line += "%13s" % ("%.3f±%.3f" % (s["f1_mean"], s["f1_sd"]))
        print(line)
        line = "  %-17s" % ""
        for c in cols:
            s = d["summary"][c]["retuned_threshold"]
            line += "%13s" % ("%.3f±%.3f" % (s["f1_mean"], s["f1_sd"]))
        print(line)
        po = d["prevalence_observed"]
        print("  %-17s ROC-AUC %.4f (yayginliktan bagimsiz) | gozlenen pi "
              "ort %.4f, aralik %.4f-%.4f"
              % ("", d["roc_auc_mean"], po["mean"], po["min"], po["max"]))
    save_json(results, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
