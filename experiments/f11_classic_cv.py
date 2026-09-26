"""
F11 — Klasik modeller ve kural tabanli siniflandirici GRUPLU CV'de (degerlendirme notu).

Degerlendirmede isaretlenen kusur: klasik modeller ve kural tabanli
siniflandirici yalnizca sabit bolmede degerlendirilmisti, derin modeller ise
ayrica capraz dogrulanmisti; iki taraf esit tasarimda karsilastirilmiyordu.

Dogru itiraz. F1b bu modelleri yalnizca sabit bolmede degerlendiriyordu, F5
ise derin modelleri bes fold'da. Tablo 13'te yan yana duran sayilar farkli
degerlendirme tasarimlarindan geliyordu.

Burada AYNI fold'lar, AYNI trip gruplamasi, AYNI esik politikasi kullanilir:
her fold kendi dogrulama esigini kendi test tahminlerine uygular (A1 duzeltmesi).
Klasik modeller CPU'da kosar, bu yuzden derin kosumlarla ayni anda kosabilir.

Kural tabanli siniflandirici parametresizdir; tohum dongusu yoktur, ama yine
de fold bazinda degerlendirilir cunku test kumeleri farklidir.

KULLANIM
  python experiments/f11_classic_cv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from stanet import config as C
from stanet.baselines import rule_predictions, window_features
from stanet.data import build_cv_folds, inverse_kinematics
from stanet.evaluate import compute_metrics, pick_threshold
from stanet.stats import aggregate_seeds
from stanet.utils import init_console, protocol_stamp, save_json, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f5_cv import pooled_by_type, pooled_metrics          # noqa: E402  ayni esik politikasi

init_console()
OUT = C.RESULTS_DIR / "F11_classic_cv.json"
SEEDS = [42, 43]                 # F5 ile ayni
TREES = ("xgboost", "lightgbm", "random_forest")
RULES = ("dwell_only", "dwell_accel", "best", "all_rules")


def raw_windows(ds, split):
    X = getattr(ds, "X_" + split)
    s, a = inverse_kinematics(X, ds.scalers)
    return np.stack([s, a, X[:, :, 2], X[:, :, 3]], axis=2)


def make(name, seed, spw):
    if name == "xgboost":
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=seed, n_jobs=4, tree_method="hist")
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=spw, random_state=seed,
                              n_jobs=4, verbose=-1)
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=seed, n_jobs=4)


def main():
    stamp = protocol_stamp()
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    print("protokol %s | %d-fold trip-gruplu CV | %d agac x %d tohum + %d kural\n"
          % (stamp["hash"], C.N_FOLDS, len(TREES), len(SEEDS), len(RULES)), flush=True)

    # fold basina oznitelikler bir kez
    prep = {}
    for f in folds:
        k = f.info["fold"]
        Xtr, Xva, Xte = (raw_windows(f, s) for s in ("train", "val", "test"))
        prep[k] = {
            "F": (window_features(Xtr), window_features(Xva), window_features(Xte)),
            "y": (f.y_train, f.y_val, f.y_test),
            "Xte_raw": Xte, "type_test": f.type_test,
            "trips": np.asarray(f.meta_test)[:, 0],
            "spw": float((len(f.y_train) - f.y_train.sum()) / max(f.y_train.sum(), 1)),
        }
        print("  fold %d: oznitelik %d | egitim %d pencere | pos_weight %.2f"
              % (k, prep[k]["F"][0].shape[1], len(f.y_train), prep[k]["spw"]),
              flush=True)
    print()

    results = {"protocol": stamp,
               "config": {"n_folds": C.N_FOLDS, "seeds": SEEDS,
                          "threshold_policy": "per_fold_validation",
                          "note": ("F5 ile ayni fold'lar ve ayni esik politikasi; "
                                   "Tablo 13 artik esit tasarimda")},
               "fold_info": [f.info for f in folds], "models": {}}
    pooled_store = {}

    # ------------------------------------------------------------ agac modelleri
    for name in TREES:
        runs, y_all, p_all, b_all, t_all, g_all, th_all = [], [], [], [], [], [], []
        for f in folds:
            k = f.info["fold"]
            P = prep[k]
            Ftr, Fva, Fte = P["F"]
            ytr, yva, yte = P["y"]
            for seed in SEEDS:
                set_seed(seed)
                clf = make(name, seed, P["spw"]).fit(Ftr, ytr)
                pv = clf.predict_proba(Fva)[:, 1]
                pt = clf.predict_proba(Fte)[:, 1]
                th = pick_threshold(yva, pv)
                m = compute_metrics(yte, pt, th)
                m.update({"fold": k, "seed": seed, "threshold_val": float(th)})
                runs.append(m)
                print("  %-14s fold%d s%d: F1=%.4f AUC=%.4f esik=%.3f"
                      % (name, k, seed, m["f1"], m.get("roc_auc", float("nan")), th),
                      flush=True)
                if seed == SEEDS[0]:          # havuzlama tek tohumla
                    y_all.append(yte)
                    p_all.append(pt)
                    b_all.append((pt > th).astype(int))
                    t_all.append(P["type_test"])
                    g_all.append(P["trips"])
                    th_all.append(float(th))
        y = np.concatenate(y_all)
        p = np.concatenate(p_all)
        b = np.concatenate(b_all)
        t = np.concatenate(t_all)
        g = np.concatenate(g_all)
        note = "per_fold_validation"
        results["models"][name] = {
            "type": "classic_ml", "runs": runs, "aggregate": aggregate_seeds(runs),
            "pooled": pooled_metrics(y, p, b, note),
            "pooled_threshold_policy": note, "pooled_fold_thresholds": th_all,
            "by_type_pooled": pooled_by_type(y, p, b, t, note),
            "pooled_from_seed": SEEDS[0],
        }
        a = results["models"][name]["aggregate"]["f1"]
        print("  -> %s: kosu-ort F1 %.4f+-%.4f | havuz F1 %.4f\n"
              % (name, a["mean"], a["std"], results["models"][name]["pooled"]["f1"]),
              flush=True)
        pooled_store[name] = (y, p, t, g, b)

    # ------------------------------------------------------------ kural (parametresiz)
    for variant in RULES:
        runs, y_all, b_all, t_all, g_all = [], [], [], [], []
        for f in folds:
            k = f.info["fold"]
            P = prep[k]
            X = P["Xte_raw"]
            pred, _ = rule_predictions(X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3],
                                       variant=variant)
            m = compute_metrics(P["y"][2], pred.astype(float), 0.5)
            m.update({"fold": k, "seed": None})
            runs.append(m)
            y_all.append(P["y"][2])
            b_all.append(pred.astype(int))
            t_all.append(P["type_test"])
            g_all.append(P["trips"])
        y = np.concatenate(y_all)
        b = np.concatenate(b_all)
        t = np.concatenate(t_all)
        g = np.concatenate(g_all)
        f1s = np.array([m["f1"] for m in runs])
        note = "rule_is_parameter_free_no_threshold"
        key = "rule_" + variant
        results["models"][key] = {
            "type": "rule", "runs": runs,
            "aggregate": {"f1": {"mean": float(f1s.mean()),
                                 "std": float(f1s.std(ddof=1)),
                                 "n": int(len(f1s))}},
            "pooled": pooled_metrics(y, b.astype(float), b, note),
            "by_type_pooled": pooled_by_type(y, b.astype(float), b, t, note),
            "pooled_threshold_policy": note,
        }
        print("  %-18s: fold-ort F1 %.4f+-%.4f | havuz F1 %.4f"
              % (key, f1s.mean(), f1s.std(ddof=1),
                 results["models"][key]["pooled"]["f1"]), flush=True)
        pooled_store[key] = (y, b.astype(float), t, g, b)

    # ------------------------------------------------------------ ozet
    print("\n" + "=" * 84)
    print("GRUPLU CV — KLASIK MODELLER VE KURAL (esik: her fold kendi dogrulamasi)")
    print("=" * 84)
    print("  %-20s %18s %10s %13s" % ("model", "kosu-ort F1", "havuz F1", "havuz PR-AUC"))
    for name, d in results["models"].items():
        a = d["aggregate"]["f1"]
        pr = d["pooled"].get("pr_auc")
        prs = ("%.4f" % pr) if pr is not None else "-"
        print("  %-20s %.4f+-%.4f  %10.4f %13s"
              % (name, a["mean"], a.get("std", 0.0), d["pooled"]["f1"], prs))

    arrays = {}
    fields = ("y", "prob", "type", "trip", "bin")
    for k2, v in pooled_store.items():
        for i, fl in enumerate(fields):
            arrays[k2 + "_" + fl] = v[i]
    np.savez_compressed(C.RESULTS_DIR / "F11_pooled.npz", **arrays)
    save_json(results, OUT)
    print("\nkaydedildi: %s" % OUT)
    print("kaydedildi: %s" % (C.RESULTS_DIR / "F11_pooled.npz"))


if __name__ == "__main__":
    main()
